#![forbid(unsafe_code)]

use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;

use super::{
    match_compare, trees_tbl, State, StaticTreeDesc, D_CODES, LENGTH_CODES, STD_MAX_MATCH,
    STD_MIN_MATCH,
};

/// One useful LZ77 alternative at an input position.
///
/// Candidates are reduced losslessly by DEFLATE distance symbol: within one distance-code bucket,
/// all distances have the same encoded bit length (the extra-bit *value* differs, but its width
/// does not), so only the longest available match in that bucket can ever be useful to a parser.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MatchCandidate {
    pub(crate) length: u16,
    pub(crate) distance: u16,
}

impl MatchCandidate {
    const NONE: Self = Self {
        length: 0,
        distance: 0,
    };
}

/// Sparse whole-input candidate table in CSR form.
///
/// `offsets[position]..offsets[position + 1]` indexes the alternatives for one input byte. Entries
/// are ordered by distance code, and there is at most one entry for each of DEFLATE's 30 distance
/// symbols. This boundary is rich enough for arbitrary Huffman cost models without forcing ordinary
/// Fast/Medium/Slow parsing to consume more than its current single-best projection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MatchCandidates {
    offsets: Vec<u32>,
    entries: Vec<MatchCandidate>,
}

impl MatchCandidates {
    pub(crate) fn for_position(&self, position: usize) -> &[MatchCandidate] {
        let start = self.offsets[position] as usize;
        let end = self.offsets[position + 1] as usize;
        &self.entries[start..end]
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TokenCosts {
    literal: [u8; 256],
    length: [u8; LENGTH_CODES],
    distance: [u8; D_CODES],
}

impl TokenCosts {
    /// Fixed-Huffman token costs. Block headers and end-of-block are intentionally excluded because
    /// they are constant for a parse of one already-chosen block.
    pub(crate) fn fixed() -> Self {
        let mut literal = [0; 256];
        for (byte, cost) in literal.iter_mut().enumerate() {
            *cost = trees_tbl::STATIC_LTREE[byte].len() as u8;
        }

        let mut length = [0; LENGTH_CODES];
        for (code, cost) in length.iter_mut().enumerate() {
            *cost = trees_tbl::STATIC_LTREE[257 + code].len() as u8;
        }

        let mut distance = [0; D_CODES];
        for (code, cost) in distance.iter_mut().enumerate() {
            *cost = trees_tbl::STATIC_DTREE[code].len() as u8;
        }

        Self {
            literal,
            length,
            distance,
        }
    }

    fn literal_bits(self, byte: u8) -> u64 {
        u64::from(self.literal[byte as usize])
    }

    #[cfg(test)]
    fn match_bits(self, length: usize, distance: usize) -> u64 {
        debug_assert!((STD_MIN_MATCH..=STD_MAX_MATCH).contains(&length));
        debug_assert!(distance > 0);

        let length_code = trees_tbl::LENGTH_CODE[length - STD_MIN_MATCH] as usize;
        let distance_code = State::d_code(distance - 1) as usize;
        u64::from(self.length[length_code])
            + u64::from(StaticTreeDesc::EXTRA_LBITS[length_code])
            + u64::from(self.distance[distance_code])
            + u64::from(StaticTreeDesc::EXTRA_DBITS[distance_code])
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ParseToken {
    Literal(u8),
    Match { length: u16, distance: u16 },
}

impl ParseToken {
    fn input_len(self) -> usize {
        match self {
            Self::Literal(_) => 1,
            Self::Match { length, .. } => usize::from(length),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ParseResult {
    pub(crate) cost_bits: u64,
    pub(crate) tokens: Vec<ParseToken>,
}

#[derive(Clone, Copy, Debug)]
struct ParseStep {
    previous: usize,
    token: ParseToken,
}

/// Build the exact useful-match frontier for a complete input.
///
/// Suffix-array ranks make this exact without walking every equal-prefix predecessor. For each DEFLATE
/// distance symbol, a sliding ordered set contains exactly the suffix ranks currently inside that
/// symbol's distance interval. The lexicographic predecessor or successor of the current suffix
/// maximizes LCP over that set, so two probes recover the longest match for the whole distance class.
/// Retaining that one maximum per class is lossless for arbitrary DEFLATE Huffman code lengths.
pub(crate) fn build_match_candidates(input: &[u8], max_distance: usize) -> MatchCandidates {
    assert!(max_distance <= u16::MAX as usize);
    assert!(input.len() <= u32::MAX as usize);

    let suffixes = SuffixRanks::new(input);
    let ranges = distance_code_ranges(max_distance);
    let mut active: [BTreeSet<usize>; D_CODES] = core::array::from_fn(|_| BTreeSet::new());
    let mut offsets = Vec::with_capacity(input.len() + 1);
    let mut entries = Vec::new();
    offsets.push(0);

    for position in 0..input.len() {
        let rank = suffixes.rank[position];
        let available = STD_MAX_MATCH.min(input.len() - position);

        for (code, &(min_distance, max_distance)) in ranges.iter().enumerate() {
            if min_distance == 0 {
                continue;
            }

            if let Some(add) = position.checked_sub(min_distance) {
                active[code].insert(suffixes.rank[add]);
            }
            if let Some(remove) = position.checked_sub(max_distance + 1) {
                active[code].remove(&suffixes.rank[remove]);
            }

            if available < STD_MIN_MATCH || active[code].is_empty() {
                continue;
            }

            let predecessor = active[code].range(..rank).next_back().copied();
            let successor = active[code].range(rank.saturating_add(1)..).next().copied();
            let mut best = MatchCandidate::NONE;

            for candidate_rank in [predecessor, successor].into_iter().flatten() {
                let candidate_position = suffixes.sa[candidate_rank];
                let distance = position - candidate_position;
                debug_assert!((min_distance..=max_distance).contains(&distance));
                let length = match_compare::common_prefix_len(
                    &input[position..position + available],
                    &input[candidate_position..candidate_position + available],
                );
                if length >= STD_MIN_MATCH
                    && (length > usize::from(best.length)
                        || (length == usize::from(best.length)
                            && distance < usize::from(best.distance)))
                {
                    best = MatchCandidate {
                        length: length as u16,
                        distance: distance as u16,
                    };
                }
            }

            if best.length != 0 {
                entries.push(best);
            }
        }
        offsets.push(entries.len() as u32);
    }

    MatchCandidates { offsets, entries }
}

fn distance_code_ranges(max_distance: usize) -> [(usize, usize); D_CODES] {
    core::array::from_fn(|code| {
        let min_distance = usize::from(trees_tbl::BASE_DIST[code]) + 1;
        if min_distance > max_distance {
            return (0, 0);
        }
        let format_max = if code + 1 < D_CODES {
            usize::from(trees_tbl::BASE_DIST[code + 1])
        } else {
            32_768
        };
        (min_distance, format_max.min(max_distance))
    })
}

struct SuffixRanks {
    sa: Vec<usize>,
    rank: Vec<usize>,
}

impl SuffixRanks {
    fn new(input: &[u8]) -> Self {
        if input.is_empty() {
            return Self {
                sa: Vec::new(),
                rank: Vec::new(),
            };
        }

        let mut sa = (0..input.len()).collect::<Vec<_>>();
        let mut rank = input
            .iter()
            .map(|&byte| usize::from(byte) + 1)
            .collect::<Vec<_>>();
        let mut next_rank = vec![0usize; input.len()];
        let mut span = 1usize;

        loop {
            sa.sort_unstable_by_key(|&position| {
                (
                    rank[position],
                    position
                        .checked_add(span)
                        .filter(|&next| next < input.len())
                        .map_or(0, |next| rank[next]),
                )
            });

            let mut class = 1usize;
            next_rank[sa[0]] = class;
            for pair in sa.windows(2) {
                let left = pair[0];
                let right = pair[1];
                let left_key = (
                    rank[left],
                    left.checked_add(span)
                        .filter(|&next| next < input.len())
                        .map_or(0, |next| rank[next]),
                );
                let right_key = (
                    rank[right],
                    right
                        .checked_add(span)
                        .filter(|&next| next < input.len())
                        .map_or(0, |next| rank[next]),
                );
                if left_key != right_key {
                    class += 1;
                }
                next_rank[right] = class;
            }

            core::mem::swap(&mut rank, &mut next_rank);
            if class == input.len() {
                break;
            }
            span = span
                .checked_mul(2)
                .expect("suffix-array doubling span overflow");
        }

        // The doubling ranks are 1-based equivalence classes. The parser wants direct indices into
        // `sa`, so materialize the exact inverse suffix-array permutation once.
        for (suffix_rank, &position) in sa.iter().enumerate() {
            rank[position] = suffix_rank;
        }

        Self { sa, rank }
    }
}

const MIN_BLOCK_LENGTH: usize = 5_000;
const SOFT_MAX_BLOCK_LENGTH: usize = 60_000;
const OBSERVATIONS_PER_CHECK: u32 = 512;
const SPLIT_OBSERVATION_TYPES: usize = 10;
const MIN_SPLIT_MATCH: usize = 4;

#[derive(Clone, Debug, Default)]
struct BlockSplitStats {
    old: [u32; SPLIT_OBSERVATION_TYPES],
    recent: [u32; SPLIT_OBSERVATION_TYPES],
    old_count: u32,
    recent_count: u32,
}

impl BlockSplitStats {
    fn observe_literal(&mut self, byte: u8) {
        // Eight coarse value bands are enough to notice common useful boundaries (text/binary,
        // low/high-valued binary regions, etc.) without turning the splitter into a compressor.
        let class = usize::from(byte >> 5);
        self.recent[class] += 1;
        self.recent_count += 1;
    }

    fn observe_match(&mut self, length: usize) {
        self.recent[8 + usize::from(length >= 16)] += 1;
        self.recent_count += 1;
    }

    fn merge_recent(&mut self) {
        for (old, recent) in self.old.iter_mut().zip(&mut self.recent) {
            *old += *recent;
            *recent = 0;
        }
        self.old_count += self.recent_count;
        self.recent_count = 0;
    }

    /// Start a new block at the previous check while preserving the already-scanned recent chunk as
    /// the first statistics for that block.
    fn keep_recent_as_new_block(&mut self) {
        self.old = self.recent;
        self.old_count = self.recent_count;
        self.recent = [0; SPLIT_OBSERVATION_TYPES];
        self.recent_count = 0;
    }

    fn distribution_changed(&mut self, block_length: usize) -> bool {
        if self.old_count != 0 {
            // Compare the two normalized categorical distributions without division. `total_delta /
            // (old_count * recent_count)` is their L1 distance, in [0, 2]. Long blocks get a
            // gradually lower threshold because paying for a fresh Huffman tree becomes easier to
            // justify as a block grows. The floor prevents low-amplitude noise from fragmenting a
            // long homogeneous region indefinitely.
            let mut total_delta = 0u64;
            for index in 0..SPLIT_OBSERVATION_TYPES {
                let expected = u64::from(self.old[index]) * u64::from(self.recent_count);
                let actual = u64::from(self.recent[index]) * u64::from(self.old_count);
                total_delta += expected.abs_diff(actual);
            }
            let denominator = u64::from(self.old_count) * u64::from(self.recent_count);
            let length_discount = (block_length.min(80_000) / 512) as u64;
            let threshold_per_mille = 480u64.saturating_sub(length_discount).max(320);
            if total_delta * 1_000 >= denominator * threshold_per_mille {
                return true;
            }
        }
        self.merge_recent();
        false
    }
}

/// Pick inexpensive candidate block boundaries before running the expensive exact parser.
///
/// The splitter deliberately uses only a one-candidate rolling hash probe. Its job is not to choose
/// the final token stream; it merely classifies the input into coarse literal / short-match /
/// long-match observations and looks for distribution changes. Exact match discovery and dynamic
/// programming happen only after a boundary has been selected.
pub(crate) fn choose_block_ranges(input: &[u8], max_distance: usize) -> Vec<Range<usize>> {
    if input.is_empty() {
        return core::iter::once(0..0).collect();
    }

    const HASH_SIZE: usize = 1 << 16;
    let mut head = vec![usize::MAX; HASH_SIZE];
    let mut blocks = Vec::new();
    let mut stats = BlockSplitStats::default();
    let mut block_start = 0usize;
    let mut previous_check = None;
    let mut position = 0usize;

    while position < input.len() {
        let soft_end = block_start
            .saturating_add(SOFT_MAX_BLOCK_LENGTH)
            .min(input.len());
        // Avoid manufacturing a tiny tail block solely because of the soft maximum. 60K + <5K is
        // still below DEFLATE's 65,535-byte stored-block limit and our 65,535-token symbol limit.
        let block_limit = if input.len() - soft_end < MIN_BLOCK_LENGTH {
            input.len()
        } else {
            soft_end
        };

        if position >= block_limit {
            blocks.push(block_start..position);
            block_start = position;
            stats = BlockSplitStats::default();
            previous_check = None;
            continue;
        }

        let available = STD_MAX_MATCH.min(block_limit - position);
        let mut match_length = 0usize;
        if available >= STD_MIN_MATCH && position + 2 < input.len() {
            let hash = split_hash3(input, position);
            let candidate = head[hash];
            if candidate != usize::MAX && position - candidate <= max_distance {
                match_length = match_compare::common_prefix_len(
                    &input[position..position + available],
                    &input[candidate..candidate + available],
                );
            }
        }

        let step = if match_length >= MIN_SPLIT_MATCH {
            stats.observe_match(match_length);
            match_length
        } else {
            stats.observe_literal(input[position]);
            1
        };

        let next = position + step;
        // Even bytes skipped by the splitter's cheap greedy observation become valid history for
        // later probes. This keeps boundary detection stable without doing a full match search.
        for inserted in position..next {
            if inserted + 2 < input.len() {
                head[split_hash3(input, inserted)] = inserted;
            }
        }
        position = next;

        if position >= block_limit {
            continue;
        }
        if stats.recent_count < OBSERVATIONS_PER_CHECK
            || position - block_start < MIN_BLOCK_LENGTH
            || input.len() - position < MIN_BLOCK_LENGTH
        {
            continue;
        }

        if stats.distribution_changed(position - block_start) {
            if let Some(boundary) = previous_check {
                debug_assert!(boundary > block_start);
                blocks.push(block_start..boundary);
                block_start = boundary;
                stats.keep_recent_as_new_block();
            } else {
                blocks.push(block_start..position);
                block_start = position;
                stats = BlockSplitStats::default();
            }
            previous_check = None;
        } else {
            previous_check = Some(position);
        }
    }

    if block_start < input.len() {
        blocks.push(block_start..input.len());
    }
    debug_assert!(!blocks.is_empty());
    debug_assert_eq!(blocks[0].start, 0);
    debug_assert_eq!(blocks.last().unwrap().end, input.len());
    debug_assert!(blocks.windows(2).all(|pair| pair[0].end == pair[1].start));
    debug_assert!(blocks.iter().all(|block| block.len() <= u16::MAX as usize));
    blocks
}

#[inline]
fn split_hash3(input: &[u8], position: usize) -> usize {
    debug_assert!(position + 2 < input.len());
    let value = u32::from(input[position])
        | (u32::from(input[position + 1]) << 8)
        | (u32::from(input[position + 2]) << 16);
    (value.wrapping_mul(2_654_435_761) >> 16) as usize
}

/// Minimum-token-cost parse for one already-selected block.
///
/// `block` may start after a history prefix in `input`; candidates in the block may therefore refer
/// to matches before `block.start`, which is exactly what multi-block DEFLATE needs. The dynamic
/// program itself is local to the selected block and never crosses its end.
#[cfg(test)]
pub(crate) fn parse_block(
    input: &[u8],
    candidates: &MatchCandidates,
    costs: TokenCosts,
) -> ParseResult {
    parse_range(input, candidates, 0..input.len(), costs)
}

pub(crate) fn parse_range(
    input: &[u8],
    candidates: &MatchCandidates,
    block: Range<usize>,
    costs: TokenCosts,
) -> ParseResult {
    assert_eq!(candidates.len_for_parse(), input.len());
    assert!(block.start <= block.end && block.end <= input.len());

    const INF: u64 = u64::MAX / 4;
    let block_len = block.len();
    let mut best_cost = vec![INF; block_len + 1];
    let mut steps = vec![None; block_len + 1];
    best_cost[0] = 0;

    for local_position in 0..block_len {
        let position = block.start + local_position;
        let prefix_cost = best_cost[local_position];
        debug_assert_ne!(prefix_cost, INF);

        relax(
            &mut best_cost,
            &mut steps,
            local_position + 1,
            prefix_cost + costs.literal_bits(input[position]),
            ParseStep {
                previous: local_position,
                token: ParseToken::Literal(input[position]),
            },
        );

        let available = STD_MAX_MATCH.min(block.end - position);
        if available < STD_MIN_MATCH {
            continue;
        }

        let mut distance_cost_at_max = [None::<(u64, u16)>; STD_MAX_MATCH + 1];
        for candidate in candidates.for_position(position) {
            let max_len = usize::from(candidate.length).min(available);
            let distance = usize::from(candidate.distance);
            let distance_code = State::d_code(distance - 1) as usize;
            let distance_bits = u64::from(costs.distance[distance_code])
                + u64::from(StaticTreeDesc::EXTRA_DBITS[distance_code]);
            let slot = &mut distance_cost_at_max[max_len];
            if slot.is_none_or(|(old, _)| distance_bits < old) {
                *slot = Some((distance_bits, candidate.distance));
            }
        }

        let mut cheapest_capable = None::<(u64, u16)>;
        for length in (STD_MIN_MATCH..=available).rev() {
            if let Some(candidate) = distance_cost_at_max[length] {
                if cheapest_capable.is_none_or(|(old, _)| candidate.0 < old) {
                    cheapest_capable = Some(candidate);
                }
            }
            let Some((distance_bits, distance)) = cheapest_capable else {
                continue;
            };
            let length_code = trees_tbl::LENGTH_CODE[length - STD_MIN_MATCH] as usize;
            let token_cost = u64::from(costs.length[length_code])
                + u64::from(StaticTreeDesc::EXTRA_LBITS[length_code])
                + distance_bits;
            relax(
                &mut best_cost,
                &mut steps,
                local_position + length,
                prefix_cost + token_cost,
                ParseStep {
                    previous: local_position,
                    token: ParseToken::Match {
                        length: length as u16,
                        distance,
                    },
                },
            );
        }
    }

    let mut tokens = Vec::new();
    let mut cursor = block_len;
    while cursor != 0 {
        let step = steps[cursor].expect("every reachable endpoint has a predecessor");
        tokens.push(step.token);
        cursor = step.previous;
    }
    tokens.reverse();

    debug_assert_eq!(
        tokens.iter().map(|token| token.input_len()).sum::<usize>(),
        block_len
    );
    ParseResult {
        cost_bits: best_cost[block_len],
        tokens,
    }
}

impl MatchCandidates {
    fn len_for_parse(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }
}

fn relax(
    best_cost: &mut [u64],
    steps: &mut [Option<ParseStep>],
    endpoint: usize,
    cost: u64,
    step: ParseStep,
) {
    if cost < best_cost[endpoint] {
        best_cost[endpoint] = cost;
        steps[endpoint] = Some(step);
    }
}

#[cfg(test)]
fn common_prefix(left: &[u8], right: &[u8]) -> usize {
    match_compare::common_prefix_len(left, right)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute_frontier(input: &[u8], position: usize, max_distance: usize) -> Vec<MatchCandidate> {
        if position + STD_MIN_MATCH > input.len() {
            return Vec::new();
        }
        let mut best = [MatchCandidate::NONE; D_CODES];
        let max_len = STD_MAX_MATCH.min(input.len() - position);
        for candidate in (position.saturating_sub(max_distance)..position).rev() {
            let length = common_prefix(
                &input[position..position + max_len],
                &input[candidate..candidate + max_len],
            );
            if length < STD_MIN_MATCH {
                continue;
            }
            let distance = position - candidate;
            let code = State::d_code(distance - 1) as usize;
            if length > usize::from(best[code].length) {
                best[code] = MatchCandidate {
                    length: length as u16,
                    distance: distance as u16,
                };
            }
        }
        best.into_iter()
            .filter(|candidate| candidate.length != 0)
            .collect()
    }

    fn frontier_signature(candidates: &[MatchCandidate]) -> Vec<(u8, u16)> {
        candidates
            .iter()
            .map(|candidate| {
                (
                    State::d_code(usize::from(candidate.distance) - 1),
                    candidate.length,
                )
            })
            .collect()
    }

    fn expand(tokens: &[ParseToken]) -> Vec<u8> {
        let mut output = Vec::new();
        for &token in tokens {
            match token {
                ParseToken::Literal(byte) => output.push(byte),
                ParseToken::Match { length, distance } => {
                    for _ in 0..length {
                        let source = output.len() - usize::from(distance);
                        let byte = output[source];
                        output.push(byte);
                    }
                }
            }
        }
        output
    }

    fn brute_cost(input: &[u8], costs: TokenCosts, max_distance: usize) -> u64 {
        const INF: u64 = u64::MAX / 4;
        let mut best = vec![INF; input.len() + 1];
        best[0] = 0;
        for position in 0..input.len() {
            best[position + 1] =
                best[position + 1].min(best[position] + costs.literal_bits(input[position]));
            let max_len = STD_MAX_MATCH.min(input.len() - position);
            for candidate in position.saturating_sub(max_distance)..position {
                let matched = common_prefix(
                    &input[position..position + max_len],
                    &input[candidate..candidate + max_len],
                );
                for length in STD_MIN_MATCH..=matched {
                    let distance = position - candidate;
                    best[position + length] = best[position + length]
                        .min(best[position] + costs.match_bits(length, distance));
                }
            }
        }
        best[input.len()]
    }

    #[test]
    fn suffix_rank_frontier_matches_exhaustive_distance_code_frontier() {
        let inputs: &[&[u8]] = &[
            b"abcabcabc/abcabcabc/xyzxyzxyz",
            b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            b"012345670123456701234567--01234567",
            b"no useful repetition here",
        ];
        for input in inputs {
            let table = build_match_candidates(input, 32);
            assert_eq!(table.len(), input.len());
            for position in 0..input.len() {
                assert_eq!(
                    frontier_signature(table.for_position(position)),
                    frontier_signature(&brute_frontier(input, position, 32)),
                    "position {position} for {input:?}"
                );
            }
        }
    }

    #[test]
    fn suffix_rank_frontier_matches_bruteforce_on_generated_inputs() {
        let mut state = 0x1234_5678_9abc_def0u64;
        for len in 3..96 {
            for alphabet in [2u8, 3, 7, 31, 251] {
                let mut input = vec![0u8; len];
                for byte in &mut input {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    *byte = (state >> 32) as u8 % alphabet;
                }
                for max_distance in [1usize, 2, 3, 7, 16, 31, 64] {
                    let max_distance = max_distance.min(32_506);
                    let table = build_match_candidates(&input, max_distance);
                    for position in 0..input.len() {
                        assert_eq!(
                            frontier_signature(table.for_position(position)),
                            frontier_signature(&brute_frontier(&input, position, max_distance)),
                            "len={len} alphabet={alphabet} distance={max_distance} position={position}"
                        );
                    }
                }
            }
        }
    }

    pub(super) fn two_distribution_input() -> Vec<u8> {
        let mut state = 0x8f6e_37a1_4c92_d5b7u64;
        let mut input = Vec::with_capacity(24_000);
        for _ in 0..12_000 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            input.push(b'a' + ((state >> 32) as u8 % 26));
        }
        for _ in 0..12_000 {
            state = state
                .wrapping_mul(2_862_933_555_777_941_757)
                .wrapping_add(3_037_000_493);
            input.push(0x80 | ((state >> 32) as u8 & 0x7f));
        }
        input
    }

    #[test]
    fn splitter_detects_distribution_change_and_preserves_block_invariants() {
        let input = two_distribution_input();
        let blocks = choose_block_ranges(&input, 32_506);
        assert!(
            blocks.len() >= 2,
            "expected a split near the data transition: {blocks:?}"
        );
        let first_boundary = blocks[0].end;
        assert!(
            (9_000..=15_000).contains(&first_boundary),
            "unexpected first boundary {first_boundary}: {blocks:?}"
        );
        assert_eq!(blocks[0].start, 0);
        assert_eq!(blocks.last().unwrap().end, input.len());
        assert!(blocks.windows(2).all(|pair| pair[0].end == pair[1].start));
        assert!(blocks.iter().all(|block| block.len() <= MAX_BLOCK_INPUT));
    }

    #[test]
    fn splitter_soft_limit_never_creates_an_illegal_or_tiny_tail() {
        let input = vec![0x5a; 150_000];
        let blocks = choose_block_ranges(&input, 32_506);
        assert_eq!(blocks[0].start, 0);
        assert_eq!(blocks.last().unwrap().end, input.len());
        assert!(blocks.iter().all(|block| block.len() <= MAX_BLOCK_INPUT));
        assert!(blocks
            .iter()
            .take(blocks.len() - 1)
            .all(|block| block.len() >= MIN_BLOCK_LENGTH));
        assert!(blocks.last().unwrap().len() >= MIN_BLOCK_LENGTH);
    }

    #[test]
    fn range_parse_can_reference_history_before_the_block() {
        let mut state = 0x517c_c1b7_2722_0a95u64;
        let mut history = vec![0u8; 6_000];
        for byte in &mut history {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *byte = (state >> 24) as u8;
        }
        let mut input = history.clone();
        input.extend_from_slice(&history);
        let table = build_match_candidates(&input, 32_506);
        let parsed = parse_range(&input, &table, 6_000..12_000, TokenCosts::fixed());
        assert!(matches!(
            parsed.tokens.first(),
            Some(ParseToken::Match { length, distance })
                if *length >= 32 && *distance == 6_000
        ));
    }

    #[test]
    fn minimum_cost_parse_matches_exhaustive_oracle() {
        let input = b"abcabcabcxabcaabcabcabc---abcabc";
        let table = build_match_candidates(input, 32);
        let fixed = TokenCosts::fixed();
        let parsed = parse_block(input, &table, fixed);
        assert_eq!(parsed.cost_bits, brute_cost(input, fixed, 32));
        assert_eq!(expand(&parsed.tokens), input);
    }

    #[test]
    fn distance_code_reduction_is_valid_for_non_monotone_huffman_costs() {
        let input = b"aaaaaaaaabaaaaaaaaacaaaaaaaaadaaaaaaaaa";
        let table = build_match_candidates(input, 32);
        let mut costs = TokenCosts::fixed();
        // Deliberately make farther distance symbols cheaper than nearby ones. Keeping the longest
        // match per distance *symbol* must remain exact even when distance costs are non-monotone.
        for (code, cost) in costs.distance.iter_mut().enumerate() {
            *cost = 15 - (code % 15) as u8;
        }
        for (byte, cost) in costs.literal.iter_mut().enumerate() {
            *cost = 3 + (byte % 11) as u8;
        }

        let parsed = parse_block(input, &table, costs);
        assert_eq!(parsed.cost_bits, brute_cost(input, costs, 32));
        assert_eq!(expand(&parsed.tokens), input);
    }
}

/// Private near-optimal states use all 65,536 symbol slots supported by the existing tree
/// frequency representation. One slot remains reserved by `SymBuf`, so a block may contain at most
/// 65,535 tokens; keeping the byte length at or below the same value also preserves stored-block
/// fallback through the production encoder.
pub(crate) const NEAR_OPTIMAL_LIT_BUFSIZE: usize = 1 << 16;
pub(crate) const MAX_BLOCK_INPUT: usize = NEAR_OPTIMAL_LIT_BUFSIZE - 1;

/// Reparse one selected block using code lengths learned from the previous parse.
pub(crate) fn refined_parse_range(
    state: &mut super::State<'_>,
    input: &[u8],
    candidates: &MatchCandidates,
    block: Range<usize>,
    refinement_passes: usize,
) -> ParseResult {
    let mut costs = TokenCosts::fixed();
    let mut parsed = parse_range(input, candidates, block.clone(), costs);

    for _ in 0..refinement_passes {
        tally_tokens(state, &parsed.tokens);
        costs = dynamic_costs_from_tallied_block(state);
        state.init_block();
        parsed = parse_range(input, candidates, block.clone(), costs);
    }

    parsed
}

pub(crate) fn tally_tokens(state: &mut super::State<'_>, tokens: &[ParseToken]) {
    for &token in tokens {
        match token {
            ParseToken::Literal(byte) => {
                let _ = state.tally_lit(byte);
            }
            ParseToken::Match { length, distance } => {
                let _ =
                    state.tally_dist(usize::from(distance), usize::from(length) - STD_MIN_MATCH);
            }
        }
    }
}

fn dynamic_costs_from_tallied_block(state: &mut super::State<'_>) -> TokenCosts {
    let literal_used =
        core::array::from_fn::<_, 256, _>(|index| state.l_desc.dyn_tree[index].freq() != 0);
    let length_used = core::array::from_fn::<_, LENGTH_CODES, _>(|index| {
        state.l_desc.dyn_tree[257 + index].freq() != 0
    });
    let distance_used =
        core::array::from_fn::<_, D_CODES, _>(|index| state.d_desc.dyn_tree[index].freq() != 0);

    {
        let mut tree = super::TreeDesc::EMPTY;
        core::mem::swap(&mut tree, &mut state.l_desc);
        super::build_tree(state, &mut tree);
        core::mem::swap(&mut tree, &mut state.l_desc);
    }
    {
        let mut tree = super::TreeDesc::EMPTY;
        core::mem::swap(&mut tree, &mut state.d_desc);
        super::build_tree(state, &mut tree);
        core::mem::swap(&mut tree, &mut state.d_desc);
    }

    let fixed = TokenCosts::fixed();
    let mut costs = fixed;
    for (index, &used) in literal_used.iter().enumerate() {
        if used {
            costs.literal[index] = state.l_desc.dyn_tree[index].len() as u8;
        }
    }
    for (index, &used) in length_used.iter().enumerate() {
        if used {
            costs.length[index] = state.l_desc.dyn_tree[257 + index].len() as u8;
        }
    }
    for (index, &used) in distance_used.iter().enumerate() {
        if used {
            costs.distance[index] = state.d_desc.dyn_tree[index].len() as u8;
        }
    }
    costs
}

#[cfg(test)]
mod encoded_tests {
    use super::*;
    use crate::{
        compress_bound,
        inflate::{decompress_slice, InflateConfig},
        ReturnCode,
    };

    #[test]
    fn near_optimal_compressor_round_trips_with_refined_costs() {
        let input = b"abcabcabc---abcabcabc---the quick brown fox---abcabcabcabcabc";
        let mut compressed = vec![0u8; compress_bound(input.len())];
        let (compressed, code) =
            super::super::compress_slice_near_optimal(&mut compressed, input, 1);
        assert_eq!(code, ReturnCode::Ok);

        let mut decoded = vec![0u8; input.len()];
        let (decoded, code) = decompress_slice(&mut decoded, compressed, InflateConfig::default());
        assert_eq!(code, ReturnCode::Ok);
        assert_eq!(decoded, input);
    }

    #[test]
    fn adaptive_multi_block_stream_round_trips() {
        let input = super::tests::two_distribution_input();
        let blocks = choose_block_ranges(&input, 32_506);
        assert!(
            blocks.len() >= 2,
            "test input must exercise multiple blocks"
        );

        let mut compressed = vec![0u8; compress_bound(input.len())];
        let (compressed, code) =
            super::super::compress_slice_near_optimal(&mut compressed, &input, 1);
        assert_eq!(code, ReturnCode::Ok);

        let mut decoded = vec![0u8; input.len()];
        let (decoded, code) = decompress_slice(&mut decoded, compressed, InflateConfig::default());
        assert_eq!(code, ReturnCode::Ok);
        assert_eq!(decoded, input);
    }
}
