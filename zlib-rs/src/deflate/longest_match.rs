#![forbid(unsafe_code)]

use crate::deflate::{
    hash_calc::{HashCalcVariant, RollHashCalc, StandardHashCalc},
    Pos, State, STD_MAX_MATCH, STD_MIN_MATCH,
};

const EARLY_EXIT_TRIGGER_LEVEL: i8 = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SearchCursor {
    /// Position stored in the hash chain. For slow matching this may point at
    /// an interior substring of the candidate rather than its first byte.
    hash_pos: Pos,
    /// Offset of `hash_pos` from the beginning of the candidate.
    offset: usize,
}

impl SearchCursor {
    fn candidate_start(self) -> Option<usize> {
        usize::from(self.hash_pos).checked_sub(self.offset)
    }
}

/// Which chain-walking heuristic to use for one immutable match search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MatchSearchKind {
    Standard,
    Slow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MatchSearchResult {
    pub(crate) length: usize,
    pub(crate) start: Pos,
}

/// Owned scheduler token for one longest-match computation.
///
/// It carries no stream borrow. The scheduler may queue this token while the producer is paused,
/// then materialize an immutable request from that stream's State when the search is dispatched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MatchSearchWork {
    cur_match: Pos,
    kind: MatchSearchKind,
}

impl MatchSearchWork {
    pub(crate) fn new(cur_match: Pos, kind: MatchSearchKind) -> Self {
        Self { cur_match, kind }
    }

    pub(crate) fn request<'a>(self, state: &'a State<'_>) -> MatchSearchRequest<'a> {
        MatchSearchRequest::from_state(state, self.cur_match, self.kind)
    }
}

/// Immutable description of one longest-match search.
///
/// Hash insertion and all other stream mutation happen before this value is created. Execution only
/// reads initialized window bytes and the hash-chain snapshot exposed by `State`, which makes the
/// expensive search a detachable unit for future scoped CPU batches or device submission.
/// Borrowed immutable storage shared by one or more match-search descriptions.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MatchSearchResources<'a> {
    initialized: &'a [u8],
    prev: &'a [Pos],
    head: &'a [Pos],
}

/// Small per-search control/state values, separate from the borrowed window/hash storage.
///
/// Keep these in their native host types for now. A future device backend can define a checked
/// packed representation at its own boundary instead of silently narrowing compressor indexes here.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MatchSearchParams {
    cur_match: Pos,
    strstart: usize,
    lookahead: usize,
    wmask: usize,
    max_dist: usize,
    prev_length: u16,
    match_start: Pos,
    max_chain_length: u16,
    good_match: u16,
    nice_match: u16,
    level: i8,
    hash_calc_variant: HashCalcVariant,
    kind: MatchSearchKind,
}

#[derive(Debug)]
pub(crate) struct MatchSearchRequest<'a> {
    resources: MatchSearchResources<'a>,
    params: MatchSearchParams,
}

impl MatchSearchParams {
    #[inline(always)]
    fn update_hash(self, hash: u32, value: u32) -> u32 {
        match self.hash_calc_variant {
            HashCalcVariant::Standard => StandardHashCalc::update_hash(hash, value),
            HashCalcVariant::Roll => RollHashCalc::update_hash(hash, value),
        }
    }

    fn is_slow(self) -> bool {
        self.kind == MatchSearchKind::Slow
    }
}

impl<'a> MatchSearchRequest<'a> {
    pub(crate) fn from_state(state: &'a State<'_>, cur_match: Pos, kind: MatchSearchKind) -> Self {
        let initialized_end = state
            .strstart
            .checked_add(state.lookahead)
            .expect("deflate window position overflow");
        let initialized = state
            .window
            .filled()
            .get(..initialized_end)
            .expect("lookahead extends beyond the deflate window");
        let wmask = state.w_mask();

        Self {
            resources: MatchSearchResources {
                initialized,
                prev: &state.prev.as_slice()[..=wmask],
                head: state.head.as_slice(),
            },
            params: MatchSearchParams {
                cur_match,
                strstart: state.strstart,
                lookahead: state.lookahead,
                wmask,
                max_dist: state.max_dist(),
                prev_length: state.prev_length,
                match_start: state.match_start,
                max_chain_length: state.max_chain_length,
                good_match: state.good_match,
                nice_match: state.nice_match,
                level: state.level,
                hash_calc_variant: state.hash_calc_variant,
                kind,
            },
        }
    }

    pub(crate) fn execute(&self) -> MatchSearchResult {
        longest_match_help(self.resources, self.params)
    }
}

/// Backend contract for executing an ordered batch of independent immutable match searches.
///
/// Backends may parallelize or offload execution, but must preserve request/result indexing and may
/// not mutate compressor state. Request lifetimes bound any borrowed window/hash snapshot to the
/// dispatch scope.
pub(crate) trait MatchSearchBackend {
    fn execute(&self, requests: &[MatchSearchRequest<'_>], results: &mut [MatchSearchResult]);
}

struct ScalarMatchSearchBackend;

impl MatchSearchBackend for ScalarMatchSearchBackend {
    fn execute(&self, requests: &[MatchSearchRequest<'_>], results: &mut [MatchSearchResult]) {
        for (request, result) in requests.iter().zip(results) {
            *result = request.execute();
        }
    }
}

pub(crate) fn execute_match_search_batch_with_backend<B: MatchSearchBackend + ?Sized>(
    backend: &B,
    requests: &[MatchSearchRequest<'_>],
    results: &mut [MatchSearchResult],
) {
    assert_eq!(
        requests.len(),
        results.len(),
        "match-search batch result count must equal request count"
    );
    backend.execute(requests, results);
}

/// Execute one match-search batch with the default scalar backend.
///
/// The default remains allocation-free and synchronous. Backend choice is deliberately outside the
/// token producers so later CPU/device executors do not contaminate compression semantics.
pub(crate) fn execute_match_search_batch(
    requests: &[MatchSearchRequest<'_>],
    results: &mut [MatchSearchResult],
) {
    execute_match_search_batch_with_backend(&ScalarMatchSearchBackend, requests, results);
}

fn longest_match_help(
    resources: MatchSearchResources<'_>,
    params: MatchSearchParams,
) -> MatchSearchResult {
    let scan = &resources.initialized[params.strstart..];
    let limit_base = params.strstart.saturating_sub(params.max_dist);

    let mut best_len = if params.prev_length > 0 {
        usize::from(params.prev_length)
    } else {
        STD_MIN_MATCH - 1
    };
    let mut match_start = params.match_start;

    // A previous lazy match can be longer than the input remaining at end of
    // stream. There is then no candidate that can improve it.
    if best_len >= params.lookahead {
        return MatchSearchResult {
            length: params.lookahead,
            start: match_start,
        };
    }

    let mut chain_remaining = params.max_chain_length;
    if best_len >= usize::from(params.good_match) {
        chain_remaining >>= 2;
    }
    if chain_remaining == 0 {
        return break_matching(params, best_len, match_start);
    }

    let mut cursor = SearchCursor {
        hash_pos: params.cur_match,
        offset: 0,
    };

    if params.is_slow() && best_len >= STD_MIN_MATCH {
        let Some(next) = initial_slow_cursor(resources, params, scan, best_len, cursor, limit_base)
        else {
            return break_matching(params, best_len, match_start);
        };
        cursor = next;
    }

    let early_exit = !params.is_slow() && params.level < EARLY_EXIT_TRIGGER_LEVEL;

    loop {
        let Some(candidate_start) = cursor.candidate_start() else {
            break;
        };
        if candidate_start >= params.strstart {
            break;
        }

        let max_len = params.lookahead.min(STD_MAX_MATCH);
        if best_len >= max_len {
            break;
        }

        let candidate = &resources.initialized[candidate_start..candidate_start + max_len];
        let scan = &scan[..max_len];

        if !could_beat_best(scan, candidate, best_len) {
            let Some(next) = next_in_chain(
                cursor,
                resources.prev,
                params.wmask,
                limit_base,
                &mut chain_remaining,
            ) else {
                break;
            };
            cursor = next;
            continue;
        }

        let len = common_match_length(scan, candidate);

        if len > best_len {
            match_start = candidate_start as Pos;

            if len >= params.lookahead {
                return MatchSearchResult {
                    length: params.lookahead,
                    start: match_start,
                };
            }

            best_len = len;
            if best_len >= usize::from(params.nice_match) {
                return MatchSearchResult {
                    length: best_len,
                    start: match_start,
                };
            }

            if params.is_slow() && len > STD_MIN_MATCH && candidate_start + len < params.strstart {
                let Some(next) = next_slow_cursor(
                    resources,
                    params,
                    scan,
                    candidate_start,
                    len,
                    limit_base,
                    params.wmask,
                ) else {
                    return break_matching(params, best_len, match_start);
                };
                cursor = next;
                continue;
            }
        } else if early_exit {
            break;
        }

        let Some(next) = next_in_chain(
            cursor,
            resources.prev,
            params.wmask,
            limit_base,
            &mut chain_remaining,
        ) else {
            break;
        };
        cursor = next;
    }

    MatchSearchResult {
        length: best_len,
        start: match_start,
    }
}

fn initial_slow_cursor(
    resources: MatchSearchResources<'_>,
    params: MatchSearchParams,
    scan: &[u8],
    best_len: usize,
    mut cursor: SearchCursor,
    limit_base: usize,
) -> Option<SearchCursor> {
    let probe = scan.get(..best_len + 1)?;
    let mut hash = 0;
    hash = params.update_hash(hash, u32::from(probe[1]));
    hash = params.update_hash(hash, u32::from(probe[2]));

    // Each update below hashes the next three-byte substring beginning at
    // offsets 1..=best_len-2. Prefer a more distant chain exactly as the old
    // slow matcher did.
    for (index, byte) in probe[3..].iter().enumerate() {
        hash = params.update_hash(hash, u32::from(*byte));
        let hash_pos = resources.head[hash as usize];
        if hash_pos < cursor.hash_pos {
            cursor = SearchCursor {
                hash_pos,
                offset: index + 1,
            };
        }
    }

    (usize::from(cursor.hash_pos) > limit_base + cursor.offset).then_some(cursor)
}

fn next_slow_cursor(
    resources: MatchSearchResources<'_>,
    params: MatchSearchParams,
    scan: &[u8],
    candidate_start: usize,
    match_len: usize,
    limit_base: usize,
    wmask: usize,
) -> Option<SearchCursor> {
    let prev = resources.prev;
    let mut next = SearchCursor {
        hash_pos: candidate_start as Pos,
        offset: 0,
    };

    for offset in 0..=match_len - STD_MIN_MATCH {
        let hash_pos = prev[(candidate_start + offset) & wmask];
        if hash_pos < next.hash_pos {
            if usize::from(hash_pos) <= limit_base + offset {
                return None;
            }
            next = SearchCursor { hash_pos, offset };
        }
    }

    // Also inspect the current hash head near the end of the match. Using
    // MIN_MATCH + 1 leaves one byte for the main comparison to extend by.
    let offset = match_len - (STD_MIN_MATCH + 1);
    let bytes = scan.get(offset..offset + STD_MIN_MATCH)?;
    let mut hash = 0;
    for byte in bytes {
        hash = params.update_hash(hash, u32::from(*byte));
    }
    let hash_pos = resources.head[hash as usize];
    if hash_pos < next.hash_pos {
        if usize::from(hash_pos) <= limit_base + offset {
            return None;
        }
        next = SearchCursor { hash_pos, offset };
    }

    Some(next)
}

fn next_in_chain(
    cursor: SearchCursor,
    prev: &[Pos],
    wmask: usize,
    limit_base: usize,
    chain_remaining: &mut u16,
) -> Option<SearchCursor> {
    *chain_remaining -= 1;
    if *chain_remaining == 0 {
        return None;
    }

    let hash_pos = prev[usize::from(cursor.hash_pos) & wmask];
    (usize::from(hash_pos) > limit_base + cursor.offset)
        .then_some(SearchCursor { hash_pos, ..cursor })
}

#[inline]
fn could_beat_best(scan: &[u8], candidate: &[u8], best_len: usize) -> bool {
    debug_assert_eq!(scan.len(), candidate.len());
    debug_assert!(best_len < scan.len());

    let prefix_len = (best_len + 1).min(8);
    if scan[..prefix_len] != candidate[..prefix_len] {
        return false;
    }

    if best_len + 1 > prefix_len {
        let tail_start = (best_len + 1).saturating_sub(8);
        if scan[tail_start..=best_len] != candidate[tail_start..=best_len] {
            return false;
        }
    }

    true
}

#[inline]
fn common_match_length(scan: &[u8], candidate: &[u8]) -> usize {
    debug_assert_eq!(scan.len(), candidate.len());
    debug_assert!(scan.len() <= STD_MAX_MATCH);

    if scan.len() == STD_MAX_MATCH {
        if scan[0] != candidate[0] {
            return 0;
        }
        if scan[1] != candidate[1] {
            return 1;
        }
        return crate::deflate::compare256::compare256_slice(&scan[2..], &candidate[2..]) + 2;
    }

    scan.iter()
        .zip(candidate)
        .position(|(left, right)| left != right)
        .unwrap_or(scan.len())
}

fn break_matching(
    params: MatchSearchParams,
    best_len: usize,
    match_start: Pos,
) -> MatchSearchResult {
    MatchSearchResult {
        length: best_len.min(params.lookahead),
        start: match_start,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ScopedThreadMatchSearchBackend {
        workers: usize,
    }

    impl MatchSearchBackend for ScopedThreadMatchSearchBackend {
        fn execute(&self, requests: &[MatchSearchRequest<'_>], results: &mut [MatchSearchResult]) {
            if requests.is_empty() {
                return;
            }

            let workers = self.workers.max(1).min(requests.len());
            let chunk_len = requests.len().div_ceil(workers);
            std::thread::scope(|scope| {
                for (request_chunk, result_chunk) in requests
                    .chunks(chunk_len)
                    .zip(results.chunks_mut(chunk_len))
                {
                    scope.spawn(move || {
                        ScalarMatchSearchBackend.execute(request_chunk, result_chunk);
                    });
                }
            });
        }
    }

    #[test]
    fn requests_are_worker_safe_and_batchable() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MatchSearchRequest<'static>>();

        let initialized_a = [b'a'; 32];
        let mut initialized_b = [b'b'; 32];
        initialized_b[13] = b'c';
        let prev = [0u16; 32];
        let head = [0u16; 1];

        fn request<'a>(
            initialized: &'a [u8],
            prev: &'a [Pos],
            head: &'a [Pos],
        ) -> MatchSearchRequest<'a> {
            MatchSearchRequest {
                resources: MatchSearchResources {
                    initialized,
                    prev,
                    head,
                },
                params: MatchSearchParams {
                    cur_match: 0,
                    strstart: 16,
                    lookahead: 16,
                    wmask: 31,
                    max_dist: 16,
                    prev_length: 0,
                    match_start: 0,
                    max_chain_length: 1,
                    good_match: 4,
                    nice_match: 16,
                    level: 6,
                    hash_calc_variant: HashCalcVariant::Standard,
                    kind: MatchSearchKind::Standard,
                },
            }
        }

        let requests = [
            request(&initialized_a, &prev, &head),
            request(&initialized_b, &prev, &head),
        ];
        // Independent requests can coexist and execute as one batch without borrowing mutable
        // compressor state.
        let mut results = [MatchSearchResult {
            length: 0,
            start: 0,
        }; 2];
        execute_match_search_batch(&requests, &mut results);

        let mut parallel_results = [MatchSearchResult {
            length: 0,
            start: 0,
        }; 2];
        execute_match_search_batch_with_backend(
            &ScopedThreadMatchSearchBackend { workers: 2 },
            &requests,
            &mut parallel_results,
        );
        assert_eq!(parallel_results, results);

        assert_eq!(
            results[0],
            MatchSearchResult {
                length: 16,
                start: 0
            }
        );
        assert_eq!(
            results[1],
            MatchSearchResult {
                length: 13,
                start: 0
            }
        );
    }

    #[test]
    fn common_match_length_finds_every_mismatch_position() {
        let scan = [b'a'; STD_MAX_MATCH];
        let mut candidate = scan;

        for mismatch in 0..STD_MAX_MATCH {
            candidate[mismatch] = b'b';
            assert_eq!(common_match_length(&scan, &candidate), mismatch);
            candidate[mismatch] = b'a';
        }
        assert_eq!(common_match_length(&scan, &candidate), STD_MAX_MATCH);
    }

    #[test]
    fn common_match_length_respects_short_initialized_prefixes() {
        let scan = [b'x'; 37];
        let mut candidate = scan;
        candidate[36] = b'y';

        assert_eq!(common_match_length(&scan[..7], &candidate[..7]), 7);
        assert_eq!(common_match_length(&scan, &candidate), 36);
    }

    #[test]
    fn search_cursor_maps_interior_hash_positions_to_candidate_start() {
        assert_eq!(
            SearchCursor {
                hash_pos: 123,
                offset: 17,
            }
            .candidate_start(),
            Some(106)
        );
        assert_eq!(
            SearchCursor {
                hash_pos: 3,
                offset: 4,
            }
            .candidate_start(),
            None
        );
    }

    #[test]
    fn quick_rejection_never_discards_an_improving_match() {
        let scan = [b'a'; 32];
        let mut candidate = scan;

        for best_len in 0..scan.len() {
            assert!(could_beat_best(&scan, &candidate, best_len));

            for mismatch in 0..scan.len() {
                candidate[mismatch] = b'b';
                let actual_len = common_match_length(&scan, &candidate);
                if actual_len > best_len {
                    assert!(
                        could_beat_best(&scan, &candidate, best_len),
                        "rejected improving match: best_len={best_len}, mismatch={mismatch}"
                    );
                }
                candidate[mismatch] = b'a';
            }
        }

        candidate[0] = b'b';
        assert!(!could_beat_best(&scan, &candidate, 15));
        candidate[0] = b'a';

        candidate[15] = b'b';
        assert!(!could_beat_best(&scan, &candidate, 15));
        candidate[15] = b'a';

        // For a longer known match the first and last eight bytes leave an
        // unprobed middle. False positives there are intentional: the full
        // prefix comparison decides the actual length.
        candidate[12] = b'b';
        assert!(could_beat_best(&scan, &candidate, 23));
    }
}
