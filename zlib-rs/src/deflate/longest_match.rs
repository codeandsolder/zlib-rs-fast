#![forbid(unsafe_code)]

use crate::deflate::{
    hash_calc::{HashCalcVariant, RollHashCalc, ROLL_HASH_SIZE},
    Pos, State, STD_MAX_MATCH, STD_MIN_MATCH,
};

const EARLY_EXIT_TRIGGER_LEVEL: i8 = 5;
const MAX_WINDOW_SIZE: usize = 1 << crate::MAX_WBITS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SearchCursor {
    /// Position stored in the hash chain. For slow matching this may point at
    /// an interior substring of the candidate rather than its first byte.
    hash_pos: Pos,
    /// Offset of `hash_pos` from the beginning of the candidate.
    offset: usize,
}

impl SearchCursor {
    // Hash-chain positions have an exact 16-bit representation. Retaining
    // that representation makes it impossible for a search to return a start
    // that cannot be stored in the DEFLATE state.
    fn candidate_start(self) -> Option<Pos> {
        self.hash_pos.checked_sub(Pos::try_from(self.offset).ok()?)
    }
}

/// Which chain-walking heuristic to use for one immutable match search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatchSearchKind {
    Standard,
    /// Shifted-chain search over the three-byte rolling hash layout.
    RollingOffset,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatchSearchResult {
    /// Length in the DEFLATE domain (0..=258), not the width of the host pointer.
    pub(crate) length: u16,
    pub(crate) start: Pos,
}

impl MatchSearchResult {
    /// Encode a backend result into the format's bounded match-length domain.
    ///
    /// A search backend may use host-width indexes or wider device counters,
    /// but an individual DEFLATE match can cover at most 258 bytes. Truncating
    /// the *search* to that prefix is valid; truncating its integer bits is not.
    #[inline]
    pub(crate) fn new(length: usize, start: Pos) -> Self {
        const _: () = assert!(STD_MAX_MATCH <= u16::MAX as usize);
        let bounded = length.min(STD_MAX_MATCH);
        // Conversion cannot fail because STD_MAX_MATCH is at most u16::MAX.
        let length = u16::try_from(bounded).unwrap_or(u16::MAX);
        Self { length, start }
    }
}

/// Owned scheduler token for one longest-match computation.
///
/// It carries no stream borrow. The scheduler may queue this token while the producer is paused,
/// then materialize an immutable request from that stream's State when the search is dispatched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatchSearchWork {
    HashChain {
        cur_match: Pos,
        kind: MatchSearchKind,
    },
    External,
}

/// An invalid detached match-search request, producer phase, or completion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatchSearchError {
    /// An external/device match request cannot be run by the hash-chain backend.
    ExternalWork,
    /// Hash-chain work was routed to an external/device-only match backend.
    #[cfg(feature = "gpu")]
    WrongBackend,
    /// The initialized window end cannot be represented in host index space.
    WindowPositionOverflow,
    /// The advertised lookahead includes bytes not initialized in the window.
    UninitializedWindow,
    /// The advertised history index cannot be represented by the allocated table.
    InvalidHashHistory,
    /// Offset search requires the three-byte rolling hash history.
    IncompatibleHashLayout,
    /// A token producer was asked to advance before its prior search completed.
    OutstandingSearch,
    /// A match result was delivered without a pending request.
    UnexpectedMatchResult,
}

impl MatchSearchWork {
    pub(crate) const fn new(cur_match: Pos, kind: MatchSearchKind) -> Self {
        Self::HashChain { cur_match, kind }
    }

    pub(crate) const fn external() -> Self {
        Self::External
    }

    #[cfg(any(feature = "gpu", test))]
    pub(crate) const fn is_external(self) -> bool {
        matches!(self, Self::External)
    }

    #[inline]
    pub(crate) fn request<'a>(
        self,
        state: &'a State<'_>,
    ) -> Result<MatchSearchRequest<'a>, MatchSearchError> {
        let Self::HashChain { cur_match, kind } = self else {
            return Err(MatchSearchError::ExternalWork);
        };
        MatchSearchRequest::from_state(state, cur_match, kind)
    }
}

/// Immutable description of one longest-match search.
///
/// Hash insertion and all other stream mutation happen before this value is created. Execution only
/// reads initialized window bytes and the hash-chain snapshot exposed by `State`, which makes the
/// expensive search a detachable unit for future scoped CPU batches or device submission.
/// Borrowed immutable storage shared by one or more match-search descriptions.
#[derive(Clone, Copy, Debug)]
pub struct MatchSearchResources<'a> {
    initialized: &'a [u8],
    prev: &'a [Pos],
    head: &'a [Pos],
}

/// Small per-search control/state values, separate from the borrowed window/hash storage.
///
/// Keep these in their native host types for now. A future device backend can define a checked
/// packed representation at its own boundary instead of silently narrowing compressor indexes here.
#[derive(Clone, Copy, Debug)]
pub struct MatchSearchParams {
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
    kind: MatchSearchKind,
}

#[derive(Debug)]
pub struct MatchSearchRequest<'a> {
    resources: MatchSearchResources<'a>,
    params: MatchSearchParams,
}

impl MatchSearchParams {
    /// Start the bounded hash-chain walk from the lazy-match length and
    /// reduce its search budget when that previous match was already good.
    #[inline]
    fn initial_search_budget(self) -> (usize, u16) {
        let best_len = if self.prev_length > 0 {
            usize::from(self.prev_length)
        } else {
            STD_MIN_MATCH - 1
        };
        let chain_remaining = if best_len >= usize::from(self.good_match) {
            self.max_chain_length >> 2
        } else {
            self.max_chain_length
        };
        (best_len, chain_remaining)
    }
}

impl<'a> MatchSearchRequest<'a> {
    #[inline]
    pub(crate) fn from_state(
        state: &'a State<'_>,
        cur_match: Pos,
        kind: MatchSearchKind,
    ) -> Result<Self, MatchSearchError> {
        if kind == MatchSearchKind::RollingOffset
            && state.hash_calc_variant != HashCalcVariant::Roll
        {
            return Err(MatchSearchError::IncompatibleHashLayout);
        }
        let initialized_end = state
            .strstart
            .checked_add(state.lookahead)
            .ok_or(MatchSearchError::WindowPositionOverflow)?;
        let initialized = state
            .window
            .filled()
            .get(..initialized_end)
            .ok_or(MatchSearchError::UninitializedWindow)?;
        let wmask = state.w_mask();
        let prev = state
            .prev
            .as_slice()
            .get(..=wmask)
            .ok_or(MatchSearchError::InvalidHashHistory)?;

        Ok(Self {
            resources: MatchSearchResources {
                initialized,
                prev,
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
                kind,
            },
        })
    }

    #[inline]
    pub(crate) fn execute(&self) -> MatchSearchResult {
        match self.params.kind {
            MatchSearchKind::Standard => execute_standard(self.resources, self.params),
            MatchSearchKind::RollingOffset => execute_slow(self.resources, self.params),
        }
    }
}

#[inline(never)]
fn execute_standard(
    resources: MatchSearchResources<'_>,
    params: MatchSearchParams,
) -> MatchSearchResult {
    if params.lookahead >= STD_MAX_MATCH {
        longest_match_help::<false, true, 0>(resources, params)
    } else {
        longest_match_help::<false, false, 0>(resources, params)
    }
}

#[inline(never)]
fn execute_slow(
    resources: MatchSearchResources<'_>,
    params: MatchSearchParams,
) -> MatchSearchResult {
    if params.wmask == MAX_WINDOW_SIZE - 1 && params.lookahead >= STD_MAX_MATCH {
        return longest_match_help::<true, true, MAX_WINDOW_SIZE>(resources, params);
    }
    if params.lookahead >= STD_MAX_MATCH {
        longest_match_help::<true, true, 0>(resources, params)
    } else {
        longest_match_help::<true, false, 0>(resources, params)
    }
}

/// Test helper for independent detached requests; production CPU compression runs whole streams.
#[cfg(test)]
pub fn execute_match_search_batch(
    requests: &[MatchSearchRequest<'_>],
    results: &mut [MatchSearchResult],
) {
    assert_eq!(requests.len(), results.len());
    for (request, result) in requests.iter().zip(results) {
        *result = request.execute();
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "keep the shared chain walk and its full/short lookahead specializations together"
)]
fn longest_match_help<const SLOW: bool, const FULL: bool, const RING: usize>(
    resources: MatchSearchResources<'_>,
    params: MatchSearchParams,
) -> MatchSearchResult {
    // For the full-size rolling window, make the exact table extents
    // visible to LLVM. The fixed mask then proves every chain index valid.
    // All smaller windows and short lookahead keep the dynamic path.
    let resources = if RING != 0 {
        MatchSearchResources {
            prev: &resources.prev[..RING],
            head: &resources.head[..ROLL_HASH_SIZE],
            ..resources
        }
    } else {
        resources
    };
    let wmask = if RING != 0 { RING - 1 } else { params.wmask };
    let scan = &resources.initialized[params.strstart..];
    let limit_base = params.strstart.saturating_sub(params.max_dist);

    let (mut best_len, mut chain_remaining) = params.initial_search_budget();
    let mut match_start = params.match_start;

    // A previous lazy match can be longer than the input remaining at end of
    // stream. There is then no candidate that can improve it.
    if best_len >= params.lookahead {
        return MatchSearchResult::new(params.lookahead, match_start);
    }
    if chain_remaining == 0 {
        return break_matching(params, best_len, match_start);
    }

    let mut cursor = SearchCursor {
        hash_pos: params.cur_match,
        offset: 0,
    };

    if SLOW && best_len >= STD_MIN_MATCH {
        let Some(next) = initial_slow_cursor(resources, scan, best_len, cursor, limit_base) else {
            return break_matching(params, best_len, match_start);
        };
        cursor = next;
    }

    let early_exit = !SLOW && params.level < EARLY_EXIT_TRIGGER_LEVEL;

    // Full lookahead is overwhelmingly common in the producer. Select its
    // fixed-size comparison domain once, not for each hash-chain candidate.
    let max_len = if FULL {
        STD_MAX_MATCH
    } else {
        params.lookahead.min(STD_MAX_MATCH)
    };
    let scan = &scan[..max_len];
    let mut scan_words = ScanProbeWords::new(scan, max_len, best_len);

    while let Some(candidate_pos) = cursor.candidate_start() {
        let candidate_start = usize::from(candidate_pos);
        if candidate_start >= params.strstart {
            break;
        }

        if best_len >= max_len {
            break;
        }

        // For shifted chains most long-match candidates are rejected. Probe
        // the initialized window directly; form the full candidate only on
        // the path that actually needs an exact prefix comparison.
        let probe = if SLOW && best_len >= 8 {
            let tail_start = best_len + 1 - 8;
            if scan_words.tail != load_u64_le(resources.initialized, candidate_start + tail_start)
                || scan_words.first != load_u64_le(resources.initialized, candidate_start)
            {
                CandidateProbe::Reject
            } else {
                CandidateProbe::CompareFrom(8)
            }
        } else {
            let candidate = &resources.initialized[candidate_start..candidate_start + max_len];
            probe_candidate_cached(scan, candidate, best_len, scan_words.first, scan_words.tail)
        };
        let len = match probe {
            CandidateProbe::Reject => {
                let Some(next) = next_in_chain::<RING>(
                    cursor,
                    resources.prev,
                    wmask,
                    limit_base,
                    &mut chain_remaining,
                ) else {
                    break;
                };
                cursor = next;
                continue;
            }
            CandidateProbe::Exact(len) => len,
            CandidateProbe::CompareFrom(prefix) => {
                let candidate =
                    &resources.initialized[candidate_start + prefix..candidate_start + max_len];
                prefix + common_match_length(&scan[prefix..], candidate)
            }
        };

        if len > best_len {
            match_start = candidate_pos;

            if len >= params.lookahead {
                return MatchSearchResult::new(params.lookahead, match_start);
            }

            best_len = len;
            scan_words.update_tail(scan, best_len);
            if best_len >= usize::from(params.nice_match) {
                return MatchSearchResult::new(best_len, match_start);
            }

            if SLOW && len > STD_MIN_MATCH && candidate_start + len < params.strstart {
                let Some(next) =
                    next_slow_cursor(resources, scan, candidate_pos, len, limit_base, wmask)
                else {
                    return break_matching(params, best_len, match_start);
                };
                cursor = next;
                continue;
            }
        } else if early_exit {
            break;
        }

        let Some(next) = next_in_chain::<RING>(
            cursor,
            resources.prev,
            wmask,
            limit_base,
            &mut chain_remaining,
        ) else {
            break;
        };
        cursor = next;
    }

    MatchSearchResult::new(best_len, match_start)
}

#[inline]
fn initial_slow_cursor(
    resources: MatchSearchResources<'_>,
    scan: &[u8],
    best_len: usize,
    mut cursor: SearchCursor,
    limit_base: usize,
) -> Option<SearchCursor> {
    let probe = scan.get(..best_len + 1)?;
    let mut hash = 0;
    hash = RollHashCalc::update_hash(hash, u32::from(probe[1]));
    hash = RollHashCalc::update_hash(hash, u32::from(probe[2]));

    // Each update below hashes the next three-byte substring beginning at
    // offsets 1..=best_len-2. Prefer a more distant chain exactly as the old
    // slow matcher did.
    for (index, byte) in probe[3..].iter().enumerate() {
        hash = RollHashCalc::update_hash(hash, u32::from(*byte));
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

#[inline]
fn next_slow_cursor(
    resources: MatchSearchResources<'_>,
    scan: &[u8],
    candidate_start: Pos,
    match_len: usize,
    limit_base: usize,
    wmask: usize,
) -> Option<SearchCursor> {
    let prev = resources.prev;
    let mut next = SearchCursor {
        hash_pos: candidate_start,
        offset: 0,
    };
    let candidate_start = usize::from(candidate_start);

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
        hash = RollHashCalc::update_hash(hash, u32::from(*byte));
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

#[inline]
fn next_in_chain<const RING: usize>(
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

    let hash_pos = if RING != 0 {
        prev[..RING][usize::from(cursor.hash_pos) & (RING - 1)]
    } else {
        prev[usize::from(cursor.hash_pos) & wmask]
    };
    (usize::from(hash_pos) > limit_base + cursor.offset)
        .then_some(SearchCursor { hash_pos, ..cursor })
}

/// Scan-side words shared by all hash-chain candidates. The second word
/// depends on the best match length and is refreshed only on improvement.
struct ScanProbeWords {
    first: u64,
    tail: u64,
    max_len: usize,
}

impl ScanProbeWords {
    #[inline]
    fn new(scan: &[u8], max_len: usize, best_len: usize) -> Self {
        debug_assert!(max_len <= scan.len());
        let first = if max_len >= 8 {
            load_u64_le(scan, 0)
        } else {
            0
        };
        let tail = if best_len >= 8 && best_len < max_len {
            load_u64_le(scan, best_len + 1 - 8)
        } else {
            0
        };
        Self {
            first,
            tail,
            max_len,
        }
    }

    #[inline]
    fn update_tail(&mut self, scan: &[u8], best_len: usize) {
        if best_len >= 8 && best_len < self.max_len {
            self.tail = load_u64_le(scan, best_len + 1 - 8);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CandidateProbe {
    Reject,
    /// The first-word mismatch gives the exact match length, so no wider compare is needed.
    Exact(usize),
    /// The prefix up to this offset is already known equal; continue comparing after it.
    CompareFrom(usize),
}

/// Test-facing uncached entrypoint. Production reuses the invariant scan
/// words across all candidates and updates the tail only as `best_len` grows.
#[cfg(test)]
fn probe_candidate(scan: &[u8], candidate: &[u8], best_len: usize) -> CandidateProbe {
    let first = if scan.len() >= 8 {
        load_u64_le(scan, 0)
    } else {
        0
    };
    let tail = if best_len >= 8 && best_len < scan.len() {
        load_u64_le(scan, best_len + 1 - 8)
    } else {
        0
    };
    probe_candidate_cached(scan, candidate, best_len, first, tail)
}

#[inline]
fn probe_candidate_cached(
    scan: &[u8],
    candidate: &[u8],
    best_len: usize,
    scan_first: u64,
    scan_tail: u64,
) -> CandidateProbe {
    debug_assert_eq!(scan.len(), candidate.len());
    debug_assert!(best_len < scan.len());

    // The old matcher used fixed-width word probes to reject almost every hash-chain candidate
    // before entering the full prefix comparison. Keep that shape, but only after proving eight
    // initialized bytes are available: no raw pointers, over-read, or uninitialized-byte access.
    if scan.len() < 8 {
        return CandidateProbe::CompareFrom(0);
    }

    let first_diff = scan_first ^ load_u64_le(candidate, 0);

    if best_len < 8 {
        if first_diff == 0 {
            return CandidateProbe::CompareFrom(8);
        }

        let mismatch = first_diff.trailing_zeros() as usize / 8;
        return if mismatch > best_len {
            CandidateProbe::Exact(mismatch)
        } else {
            CandidateProbe::Reject
        };
    }

    if first_diff != 0 {
        return CandidateProbe::Reject;
    }

    // To improve on best_len, bytes through best_len inclusive must match. Probe the last eight of
    // those bytes as one word, just as the historical hot loop did, then let the exact SIMD matcher
    // cover the gap between the known-equal first word and this tail probe.
    let tail_start = best_len + 1 - 8;
    if scan_tail != load_u64_le(candidate, tail_start) {
        return CandidateProbe::Reject;
    }

    CandidateProbe::CompareFrom(8)
}

#[inline]
fn load_u64_le(bytes: &[u8], offset: usize) -> u64 {
    // Callers have already established eight initialized bytes from offset.
    // An array chunk retains that precondition without a fallible conversion.
    u64::from_le_bytes(bytes[offset..].as_chunks::<8>().0[0])
}

#[inline]
fn common_match_length(scan: &[u8], candidate: &[u8]) -> usize {
    debug_assert_eq!(scan.len(), candidate.len());
    debug_assert!(scan.len() <= STD_MAX_MATCH);
    crate::deflate::match_compare::common_prefix_len(scan, candidate)
}

fn break_matching(
    params: MatchSearchParams,
    best_len: usize,
    match_start: Pos,
) -> MatchSearchResult {
    MatchSearchResult::new(best_len.min(params.lookahead), match_start)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_request<'a>(
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
                kind: MatchSearchKind::Standard,
            },
        }
    }

    #[test]
    fn external_work_has_an_explicit_request_error() -> Result<(), crate::ReturnCode> {
        use crate::deflate::{DeflateConfig, DeflateStream};
        let stream = DeflateStream::new(DeflateConfig::new(6))?;
        assert!(matches!(
            MatchSearchWork::external().request(&stream.state),
            Err(MatchSearchError::ExternalWork)
        ));
        Ok(())
    }

    #[test]
    fn invalid_window_snapshots_are_rejected_before_search() -> Result<(), crate::ReturnCode> {
        use crate::deflate::{DeflateConfig, DeflateStream};
        let mut stream = DeflateStream::new(DeflateConfig::new(6))?;
        let state = &mut *stream.state;
        let work = MatchSearchWork::new(1, MatchSearchKind::Standard);
        state.strstart = 1;
        state.lookahead = usize::MAX;
        assert!(matches!(
            work.request(state),
            Err(MatchSearchError::WindowPositionOverflow)
        ));
        state.lookahead = 2;
        assert!(matches!(
            work.request(state),
            Err(MatchSearchError::UninitializedWindow)
        ));
        state.strstart = 0;
        state.lookahead = 0;
        let window_size = state.w_size;
        state.w_size = state.prev.as_slice().len() + 1;
        assert!(matches!(
            work.request(state),
            Err(MatchSearchError::InvalidHashHistory)
        ));
        state.w_size = window_size;
        Ok(())
    }

    #[test]
    fn offset_search_requires_rolling_hash_history() -> Result<(), crate::ReturnCode> {
        use crate::deflate::{DeflateConfig, DeflateStream};
        for level in 0..=9 {
            let stream = DeflateStream::new(DeflateConfig::new(level))?;
            let offset = MatchSearchWork::new(0, MatchSearchKind::RollingOffset);
            let standard = MatchSearchWork::new(0, MatchSearchKind::Standard);
            assert!(standard.request(&stream.state).is_ok());
            if stream.state.hash_calc_variant == HashCalcVariant::Roll {
                assert!(offset.request(&stream.state).is_ok());
            } else {
                assert!(matches!(
                    offset.request(&stream.state),
                    Err(MatchSearchError::IncompatibleHashLayout)
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn search_result_length_uses_deflate_domain_even_for_wide_backend_counters() {
        for length in [0, 1, 3, STD_MAX_MATCH] {
            let result = MatchSearchResult::new(length, 19);
            assert_eq!(usize::from(result.length), length);
            assert_eq!(result.start, 19);
        }
        for length in [STD_MAX_MATCH + 1, usize::from(u16::MAX), usize::MAX] {
            let result = MatchSearchResult::new(length, 19);
            assert_eq!(usize::from(result.length), STD_MAX_MATCH);
            assert_eq!(result.start, 19);
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

        let requests = [
            test_request(&initialized_a, &prev, &head),
            test_request(&initialized_b, &prev, &head),
        ];
        // Independent requests can coexist and execute as one batch without borrowing mutable
        // compressor state.
        let mut results = [MatchSearchResult {
            length: 0,
            start: 0,
        }; 2];
        execute_match_search_batch(&requests, &mut results);

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
    fn search_cursor_rejects_offsets_beyond_the_16_bit_position_domain() {
        let maximum = usize::from(Pos::MAX);
        assert_eq!(
            SearchCursor {
                hash_pos: Pos::MAX,
                offset: 0,
            }
            .candidate_start(),
            Some(Pos::MAX)
        );
        assert_eq!(
            SearchCursor {
                hash_pos: Pos::MAX,
                offset: maximum,
            }
            .candidate_start(),
            Some(0)
        );
        assert_eq!(
            SearchCursor {
                hash_pos: Pos::MAX,
                offset: maximum + 1,
            }
            .candidate_start(),
            None
        );
    }

    #[test]
    fn word_probe_reads_exactly_the_initialized_eight_byte_array() {
        let bytes = *b"0123456789abcdef";
        assert_eq!(
            load_u64_le(&bytes[..8], 0),
            u64::from_le_bytes(*b"01234567")
        );
        assert_eq!(load_u64_le(&bytes, 8), u64::from_le_bytes(*b"89abcdef"));
    }

    #[test]
    fn candidate_probe_rejects_non_improving_first_word_mismatch() {
        let scan = *b"abcdefghijklmnop";
        let mut candidate = scan;
        candidate[3] = b'X';

        assert_eq!(
            probe_candidate(&scan, &candidate, 4),
            CandidateProbe::Reject
        );
    }

    #[test]
    fn candidate_probe_returns_exact_short_improvement() {
        let scan = *b"abcdefghijklmnop";
        let mut candidate = scan;
        candidate[5] = b'X';

        assert_eq!(
            probe_candidate(&scan, &candidate, 2),
            CandidateProbe::Exact(5)
        );
    }

    #[test]
    fn candidate_probe_skips_known_equal_first_word() {
        let scan = *b"abcdefghijklmnop";
        let mut candidate = scan;
        candidate[12] = b'X';

        assert_eq!(
            probe_candidate(&scan, &candidate, 4),
            CandidateProbe::CompareFrom(8)
        );
    }

    #[test]
    fn candidate_probe_checks_best_match_tail_word() {
        let scan = [b'a'; 32];
        let mut candidate = scan;
        candidate[12] = b'b';
        assert_eq!(
            probe_candidate(&scan, &candidate, 12),
            CandidateProbe::Reject
        );

        candidate[12] = b'a';
        candidate[20] = b'b';
        assert_eq!(
            probe_candidate(&scan, &candidate, 12),
            CandidateProbe::CompareFrom(8)
        );
    }

    #[test]
    fn candidate_probe_keeps_short_logical_inputs_exact() {
        let scan = *b"abcdefg";
        let mut candidate = scan;
        candidate[5] = b'X';

        assert_eq!(
            probe_candidate(&scan, &candidate, 2),
            CandidateProbe::CompareFrom(0)
        );
        assert_eq!(common_match_length(&scan, &candidate), 5);
    }

    #[test]
    fn cached_probe_matches_original_across_tail_refreshes() {
        // Exercise widths that cross the 8-byte fast-probe boundary and
        // every match length near the DEFLATE maximum. A changing best_len
        // must never leave an obsolete scan-tail word in the next probe.
        for width in [7, 8, 9, 16, 32, STD_MAX_MATCH] {
            let scan = vec![b'a'; width];
            let mut candidate = scan.clone();
            let scan_first = if width >= 8 { load_u64_le(&scan, 0) } else { 0 };
            for best_len in (0..width).filter(|n| *n < 10 || *n % 13 == 0 || *n + 1 == width) {
                let scan_tail = if best_len >= 8 {
                    load_u64_le(&scan, best_len + 1 - 8)
                } else {
                    0
                };
                for mismatch in 0..width {
                    candidate[mismatch] = b'b';
                    assert_eq!(
                        probe_candidate_cached(&scan, &candidate, best_len, scan_first, scan_tail),
                        probe_candidate(&scan, &candidate, best_len),
                        "width={width} best_len={best_len} mismatch={mismatch}"
                    );
                    candidate[mismatch] = b'a';
                }
            }
        }
    }

    #[test]
    fn fixed_rolling_tables_match_dynamic_chain_search() {
        let mut random = 0x3812_5b7d_8fed_1a03u64;
        let mut next_random = || {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            random
        };
        let mut window = vec![0u8; 2 * MAX_WINDOW_SIZE];
        let mut prev = vec![0; MAX_WINDOW_SIZE];
        let mut head = vec![0; super::super::HASH_SIZE];
        // Include arbitrary bytes, repeated patterns, ring wrap, varying
        // chain budget, ties, shifted offsets, and the full valid lookahead.
        for pattern in [0, 3, 17, 255] {
            for byte in &mut window {
                *byte = next_random().to_le_bytes()[0] & pattern;
            }
            for value in prev.iter_mut().chain(&mut head) {
                let bytes = next_random().to_le_bytes();
                *value = u16::from_le_bytes([bytes[0], bytes[1]]);
            }
            let resources = MatchSearchResources {
                initialized: &window,
                prev: &prev,
                head: &head,
            };
            for strstart in [
                512,
                MAX_WINDOW_SIZE - 1,
                MAX_WINDOW_SIZE,
                MAX_WINDOW_SIZE + 1,
                2 * MAX_WINDOW_SIZE - STD_MAX_MATCH,
            ] {
                for prev_length in [2, 3, 7, 8, 17, 64, 257] {
                    for max_chain_length in [1, 4, 32, 4096] {
                        let params = MatchSearchParams {
                            cur_match: u16::try_from(strstart - 1).unwrap_or_default(),
                            strstart,
                            lookahead: STD_MAX_MATCH,
                            wmask: MAX_WINDOW_SIZE - 1,
                            max_dist: MAX_WINDOW_SIZE - crate::deflate::MIN_LOOKAHEAD,
                            prev_length,
                            match_start: 0,
                            max_chain_length,
                            good_match: 32,
                            nice_match: 258,
                            level: 9,
                            kind: MatchSearchKind::RollingOffset,
                        };
                        assert_eq!(
                            execute_slow(resources, params),
                            longest_match_help::<true, true, 0>(resources, params),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn bounded_chain_search_matches_scalar_candidate_oracle() {
        for width in [7, 8, 9, 16, 17, 32, STD_MAX_MATCH] {
            for start in [1, 7, 31, 127] {
                let strstart = start + width;
                let candidate_pos = u16::try_from(start).unwrap_or_default();
                let mut window = vec![b'a'; strstart + width];
                // All shifted heads point at the same candidate, so the
                // initial rolling search keeps its unshifted chain cursor.
                let prev = [0u16; 512];
                let head = vec![candidate_pos; 1 << 15];
                for best_len in 2..width {
                    let params = MatchSearchParams {
                        cur_match: candidate_pos,
                        strstart,
                        lookahead: width,
                        wmask: prev.len() - 1,
                        max_dist: strstart,
                        prev_length: u16::try_from(best_len).unwrap_or_default(),
                        match_start: 0,
                        max_chain_length: 1,
                        good_match: 258,
                        nice_match: u16::try_from(best_len + 1).unwrap_or_default(),
                        level: 9,
                        kind: MatchSearchKind::RollingOffset,
                    };
                    for mismatch in 0..=width {
                        if mismatch < width {
                            window[start + mismatch] = b'b';
                        }
                        let resources = MatchSearchResources {
                            initialized: &window,
                            prev: &prev,
                            head: &head,
                        };
                        let expected = if mismatch > best_len {
                            MatchSearchResult::new(mismatch, candidate_pos)
                        } else {
                            MatchSearchResult::new(best_len, 0)
                        };
                        assert_eq!(
                            execute_slow(resources, params),
                            expected,
                            "width={width} start={start} best_len={best_len} mismatch={mismatch}",
                        );
                        assert_eq!(execute_standard(resources, params), expected);
                        if mismatch < width {
                            window[start + mismatch] = b'a';
                        }
                    }
                }
            }
        }
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
    fn candidate_probe_never_discards_an_improving_match() {
        let scan = [b'a'; 32];
        let mut candidate = scan;

        for best_len in 0..scan.len() {
            assert_ne!(
                probe_candidate(&scan, &candidate, best_len),
                CandidateProbe::Reject
            );

            for mismatch in 0..scan.len() {
                candidate[mismatch] = b'b';
                let actual_len = common_match_length(&scan, &candidate);
                let probe = probe_candidate(&scan, &candidate, best_len);
                if actual_len > best_len {
                    assert_ne!(
                        probe,
                        CandidateProbe::Reject,
                        "rejected improving match: best_len={best_len}, mismatch={mismatch}"
                    );
                    if let CandidateProbe::Exact(length) = probe {
                        assert_eq!(length, actual_len);
                    }
                }
                candidate[mismatch] = b'a';
            }
        }

        candidate[0] = b'b';
        assert_eq!(
            probe_candidate(&scan, &candidate, 15),
            CandidateProbe::Reject
        );
        candidate[0] = b'a';

        candidate[15] = b'b';
        assert_eq!(
            probe_candidate(&scan, &candidate, 15),
            CandidateProbe::Reject
        );
        candidate[15] = b'a';

        // For a longer known match the first and last eight bytes leave an
        // unprobed middle. False positives there are intentional: the full
        // prefix comparison decides the actual length.
        candidate[12] = b'b';
        assert_eq!(
            probe_candidate(&scan, &candidate, 23),
            CandidateProbe::CompareFrom(8)
        );
    }
}
