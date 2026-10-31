#![forbid(unsafe_code)]

use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

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

/// Minimum-token-cost parse for one already-selected block.
///
/// The implementation is deliberately straightforward for the first correctness checkpoint: for
/// each position it derives the cheapest distance choice capable of every length 3..=258 and relaxes
/// those endpoints. The candidate representation is the durable API; this O(258*n) dynamic program
/// can be replaced independently with a range-minimum implementation once its semantics are locked.
pub(crate) fn parse_block(
    input: &[u8],
    candidates: &MatchCandidates,
    costs: TokenCosts,
) -> ParseResult {
    assert_eq!(candidates.len_for_parse(), input.len());

    const INF: u64 = u64::MAX / 4;
    let mut best_cost = vec![INF; input.len() + 1];
    let mut steps = vec![None; input.len() + 1];
    best_cost[0] = 0;

    for position in 0..input.len() {
        let prefix_cost = best_cost[position];
        debug_assert_ne!(prefix_cost, INF);

        relax(
            &mut best_cost,
            &mut steps,
            position + 1,
            prefix_cost + costs.literal_bits(input[position]),
            ParseStep {
                previous: position,
                token: ParseToken::Literal(input[position]),
            },
        );

        let available = STD_MAX_MATCH.min(input.len() - position);
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
                position + length,
                prefix_cost + token_cost,
                ParseStep {
                    previous: position,
                    token: ParseToken::Match {
                        length: length as u16,
                        distance,
                    },
                },
            );
        }
    }

    let mut tokens = Vec::new();
    let mut cursor = input.len();
    while cursor != 0 {
        let step = steps[cursor].expect("every reachable endpoint has a predecessor");
        tokens.push(step.token);
        cursor = step.previous;
    }
    tokens.reverse();

    debug_assert_eq!(
        tokens.iter().map(|token| token.input_len()).sum::<usize>(),
        input.len()
    );
    ParseResult {
        cost_bits: best_cost[input.len()],
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

/// The default memLevel=8 symbol buffer has 16,384 entries, with one entry reserved by the
/// ordinary block producer before it requests a flush. Keeping this first complete-buffer path to
/// one such block lets us measure the parser without inventing a second block-splitting policy yet.
pub(crate) const ONE_BLOCK_MAX_INPUT: usize = (1usize << (super::DEF_MEM_LEVEL as usize + 6)) - 1;

/// Reparse a block using code lengths learned from the previous parse.
///
/// A symbol absent from the previous parse falls back to its fixed-Huffman length rather than zero;
/// this keeps every legal transition available and prevents an iteration from assigning imaginary
/// zero-cost codes to symbols that simply were not used last time.
pub(crate) fn refined_parse(
    state: &mut super::State<'_>,
    input: &[u8],
    candidates: &MatchCandidates,
    refinement_passes: usize,
) -> ParseResult {
    let mut costs = TokenCosts::fixed();
    let mut parsed = parse_block(input, candidates, costs);

    for _ in 0..refinement_passes {
        tally_tokens(state, &parsed.tokens);
        costs = dynamic_costs_from_tallied_block(state);
        state.init_block();
        parsed = parse_block(input, candidates, costs);
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
    for index in 0..256 {
        if literal_used[index] {
            costs.literal[index] = state.l_desc.dyn_tree[index].len() as u8;
        }
    }
    for index in 0..LENGTH_CODES {
        if length_used[index] {
            costs.length[index] = state.l_desc.dyn_tree[257 + index].len() as u8;
        }
    }
    for index in 0..D_CODES {
        if distance_used[index] {
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
    fn one_block_compressor_round_trips_with_refined_costs() {
        let input = b"abcabcabc---abcabcabc---the quick brown fox---abcabcabcabcabc";
        let mut compressed = vec![0u8; compress_bound(input.len())];
        let (compressed, code) =
            super::super::compress_slice_near_optimal_block(&mut compressed, input, 1);
        assert_eq!(code, ReturnCode::Ok);

        let mut decoded = vec![0u8; input.len()];
        let (decoded, code) = decompress_slice(&mut decoded, compressed, InflateConfig::default());
        assert_eq!(code, ReturnCode::Ok);
        assert_eq!(decoded, input);
    }
}
