#![forbid(unsafe_code)]

use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;

use crate::ReturnCode;

mod prefix_groups;
use prefix_groups::{group_local_frontier, PrefixGroups};

use super::{
    match_compare, trees_tbl, State, StaticTreeDesc, D_CODES, LENGTH_CODES, L_CODES, STD_MAX_MATCH,
    STD_MIN_MATCH,
};

/// Convert a validated DEFLATE match length into its compact token width.
/// Match candidates and DP-selected tokens must obey the same 3..=258 limit.
#[inline]
fn compact_match_length(length: usize) -> u16 {
    assert!(
        (STD_MIN_MATCH..=STD_MAX_MATCH).contains(&length),
        "match exceeds DEFLATE length limit"
    );
    let Ok(length) = u16::try_from(length) else {
        unreachable!("DEFLATE match lengths fit in u16");
    };
    length
}

/// Distance bounds are independent of the compact field's u16 capacity:
/// DEFLATE uses 1..=32768 bytes of history.
#[inline]
fn compact_match_distance(distance: usize) -> u16 {
    assert!(
        (1..=1usize << crate::MAX_WBITS).contains(&distance),
        "match exceeds DEFLATE distance limit"
    );
    let Ok(distance) = u16::try_from(distance) else {
        unreachable!("DEFLATE distances fit in u16");
    };
    distance
}

/// One useful LZ77 alternative at an input position.
///
/// Candidates are reduced losslessly by DEFLATE distance symbol: within one distance-code bucket,
/// all distances have the same encoded bit length (the extra-bit *value* differs, but its width
/// does not), so only the longest available match in that bucket can ever be useful to a parser.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatchCandidate {
    pub(crate) length: u16,
    pub(crate) distance: u16,
}

impl MatchCandidate {
    const NONE: Self = Self {
        length: 0,
        distance: 0,
    };

    /// Update the best match for one distance-code bucket. Both discovery
    /// strategies deliver byte positions as usize; they share the same compact
    /// match-length encoder as DP tokens, and store the bounded distance in u16.
    #[inline]
    fn consider(&mut self, length: usize, distance: usize) {
        if length < STD_MIN_MATCH {
            return;
        }
        assert!(
            length <= STD_MAX_MATCH,
            "match exceeds DEFLATE length limit"
        );
        assert!(
            (1..=1usize << crate::MAX_WBITS).contains(&distance),
            "match exceeds DEFLATE distance limit"
        );
        if length > usize::from(self.length)
            || (length == usize::from(self.length) && distance < usize::from(self.distance))
        {
            *self = Self {
                length: compact_match_length(length),
                distance: compact_match_distance(distance),
            };
        }
    }
}

/// Exact candidate records for a contiguous range of query positions.
///
/// The suffix-order input includes the history before this range, but only
/// queried positions own CSR records. Offsets are relative to the first query,
/// and each record list is ordered by DEFLATE distance code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatchCandidates {
    // CSR offsets count candidate *entries*, not input bytes. A position may
    // contribute up to D_CODES entries, so an input fitting in a u32 index can
    // still produce more than u32::MAX entries. Keep the offsets in the same
    // index domain as the backing Vec instead of silently wrapping them.
    first_query: usize,
    offsets: Vec<usize>,
    entries: Vec<MatchCandidate>,
}

impl MatchCandidates {
    pub(crate) fn for_position(&self, position: usize) -> &[MatchCandidate] {
        let index = position - self.first_query;
        let start = self.offsets[index];
        let end = self.offsets[index + 1];
        &self.entries[start..end]
    }

    #[cfg(test)]
    const fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    const fn end(&self) -> usize {
        self.first_query + self.offsets.len() - 1
    }

    /// Extend the query coverage to the left without discovering the
    /// already-computed suffix records again. Both tables use exactly the
    /// same input context and suffix order.
    fn prepend(&mut self, mut prefix: Self) {
        assert_eq!(prefix.end(), self.first_query);
        let added = prefix.entries.len();
        self.entries.append(&mut prefix.entries);
        self.entries.rotate_right(added);
        for index in &mut self.offsets {
            *index += added;
        }
        let positions = prefix.offsets.len() - 1;
        prefix.offsets.truncate(positions);
        self.offsets.append(&mut prefix.offsets);
        self.offsets.rotate_right(positions);
        self.first_query = prefix.first_query;
    }
}

/// Approximate parser costs use 1/16-bit fixed-point units. Final block selection always uses the
/// exact whole-bit cost returned by the production Huffman tree builder.
const COST_SCALE: u16 = 16;

#[inline]
fn literal_observation_class(byte: u8) -> usize {
    usize::from(((byte >> 5) & 0x06) | (byte & 1))
}

#[inline]
fn match_observation_class(length: usize) -> usize {
    8 + usize::from(length >= 9)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BlockObservations {
    counts: [u32; SPLIT_OBSERVATION_TYPES],
    total: u32,
}

impl BlockObservations {
    const fn empty() -> Self {
        Self {
            counts: [0; SPLIT_OBSERVATION_TYPES],
            total: 0,
        }
    }

    fn observe_literal(&mut self, byte: u8) {
        // Match libdeflate's splitter observation geometry: high two value bits plus parity. This
        // distinguishes text-like and binary distributions better than eight contiguous bands.
        let class = literal_observation_class(byte);
        self.counts[class] += 1;
        self.total += 1;
    }

    fn observe_match(&mut self, length: usize) {
        self.counts[match_observation_class(length)] += 1;
        self.total += 1;
    }

    fn from_tokens(tokens: &[ParseToken]) -> Self {
        let mut observations = Self::empty();
        for &token in tokens {
            match token {
                ParseToken::Literal(byte) => observations.observe_literal(byte),
                ParseToken::Match { length, .. } => {
                    observations.observe_match(usize::from(length));
                }
            }
        }
        observations
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenCosts {
    literal: [u16; 256],
    length: [u16; LENGTH_CODES],
    distance: [u16; D_CODES],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseModel {
    costs: TokenCosts,
    observations: BlockObservations,
}

impl TokenCosts {
    /// Fixed-Huffman token costs. Block headers and end-of-block are intentionally excluded because
    /// they are constant for a parse of one already-chosen block.
    pub(crate) fn fixed() -> Self {
        let mut literal = [0; 256];
        for (byte, cost) in literal.iter_mut().enumerate() {
            *cost = trees_tbl::STATIC_LTREE[byte].len() * COST_SCALE;
        }

        let mut length = [0; LENGTH_CODES];
        for (code, cost) in length.iter_mut().enumerate() {
            *cost = trees_tbl::STATIC_LTREE[257 + code].len() * COST_SCALE;
        }

        let mut distance = [0; D_CODES];
        for (code, cost) in distance.iter_mut().enumerate() {
            *cost = trees_tbl::STATIC_DTREE[code].len() * COST_SCALE;
        }

        Self {
            literal,
            length,
            distance,
        }
    }

    /// Finite priors for symbols unseen by a trained Huffman tree.
    const fn unseen() -> Self {
        Self {
            literal: [13 * COST_SCALE; 256],
            length: [13 * COST_SCALE; LENGTH_CODES],
            distance: [10 * COST_SCALE; D_CODES],
        }
    }

    fn literal_cost(&self, byte: u8) -> u32 {
        u32::from(self.literal[usize::from(byte)])
    }

    #[inline]
    fn length_symbol_cost(&self, code: usize) -> u32 {
        u32::from(self.length[code])
            + u32::from(StaticTreeDesc::EXTRA_LBITS[code]) * u32::from(COST_SCALE)
    }

    #[inline]
    fn distance_symbol_cost(&self, code: usize) -> u32 {
        u32::from(self.distance[code])
            + u32::from(StaticTreeDesc::EXTRA_DBITS[code]) * u32::from(COST_SCALE)
    }

    #[cfg(test)]
    fn match_cost(&self, length: usize, distance: usize) -> u32 {
        debug_assert!((STD_MIN_MATCH..=STD_MAX_MATCH).contains(&length));
        debug_assert!((1..=1usize << crate::MAX_WBITS).contains(&distance));

        let length_code = usize::from(trees_tbl::LENGTH_CODE[length - STD_MIN_MATCH]);
        let distance_code = usize::from(State::d_code(distance - 1));
        self.length_symbol_cost(length_code) + self.distance_symbol_cost(distance_code)
    }

    /// One token fits u32; whole-parse dynamic-programming totals use u64.
    #[cfg(test)]
    fn token_cost(&self, token: ParseToken) -> u32 {
        match token {
            ParseToken::Literal(byte) => self.literal_cost(byte),
            ParseToken::Match { length, distance } => {
                self.match_cost(usize::from(length), usize::from(distance))
            }
        }
    }
}

/// Frozen token reassignment cost, computed once from both Huffman models.
/// DEFLATE extra-bit counts are token-dependent and cancel in their difference.
struct TokenCostDelta {
    literal: [i32; 256],
    length: [i32; LENGTH_CODES],
    distance: [i32; D_CODES],
}

impl TokenCostDelta {
    fn between(left: &TokenCosts, right: &TokenCosts) -> Self {
        Self {
            literal: core::array::from_fn(|i| {
                i32::from(left.literal[i]) - i32::from(right.literal[i])
            }),
            length: core::array::from_fn(|i| {
                i32::from(left.length[i]) - i32::from(right.length[i])
            }),
            distance: core::array::from_fn(|i| {
                i32::from(left.distance[i]) - i32::from(right.distance[i])
            }),
        }
    }

    fn token_delta(&self, token: ParseToken) -> i64 {
        match token {
            ParseToken::Literal(byte) => i64::from(self.literal[usize::from(byte)]),
            ParseToken::Match { length, distance } => {
                let length_code =
                    usize::from(trees_tbl::LENGTH_CODE[usize::from(length) - STD_MIN_MATCH]);
                let distance_code = usize::from(State::d_code(usize::from(distance) - 1));
                i64::from(self.length[length_code]) + i64::from(self.distance[distance_code])
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParseToken {
    Literal(u8),
    Match { length: u16, distance: u16 },
}

impl ParseToken {
    #[inline]
    fn matching(length: usize, distance: u16) -> Self {
        debug_assert!((1..=1usize << crate::MAX_WBITS).contains(&usize::from(distance)));
        Self::Match {
            length: compact_match_length(length),
            distance,
        }
    }

    fn input_len(self) -> usize {
        match self {
            Self::Literal(_) => 1,
            Self::Match { length, .. } => usize::from(length),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseResult {
    pub(crate) cost_x16: u64,
    pub(crate) tokens: Vec<ParseToken>,
}

// A bounded tile lets us process each large suffix-rank frontier consecutively.
// Its staging buffer preserves the position-major CSR order visible to parsers.
const DIRECT_SCAN_MAX: usize = 8;
const FRONTIER_TILE_POSITIONS: usize = 512;
const _: () = assert!(FRONTIER_TILE_POSITIONS * D_CODES <= u16::MAX as usize);

#[derive(Clone, Copy)]
struct StagedCandidate {
    local_position: u16,
    candidate: MatchCandidate,
}

struct FrontierTileScratch {
    counts: [u16; FRONTIER_TILE_POSITIONS],
    starts: [u16; FRONTIER_TILE_POSITIONS],
    ranks: [usize; FRONTIER_TILE_POSITIONS],
    available: [u16; FRONTIER_TILE_POSITIONS],
    staged: Vec<StagedCandidate>,
}

impl FrontierTileScratch {
    fn new() -> Self {
        Self {
            counts: [0; FRONTIER_TILE_POSITIONS],
            starts: [0; FRONTIER_TILE_POSITIONS],
            ranks: [0; FRONTIER_TILE_POSITIONS],
            available: [0; FRONTIER_TILE_POSITIONS],
            staged: Vec::with_capacity(FRONTIER_TILE_POSITIONS * D_CODES),
        }
    }

    fn prepare(&mut self, suffixes: &SuffixRanks, input_len: usize, tile_start: usize, len: usize) {
        self.counts[..len].fill(0);
        self.staged.clear();
        for (local, (rank, available)) in self.ranks[..len]
            .iter_mut()
            .zip(self.available[..len].iter_mut())
            .enumerate()
        {
            let position = tile_start + local;
            *rank = suffixes.rank_of(position);
            *available = u16::try_from(STD_MAX_MATCH.min(input_len - position))
                .unwrap_or_else(|_| unreachable!("DEFLATE maximum match length fits in u16"));
        }
    }

    fn scatter(&mut self, len: usize, entries: &mut Vec<MatchCandidate>, offsets: &mut Vec<usize>) {
        let mut next = 0u16;
        for (start, &count) in self.starts[..len].iter_mut().zip(&self.counts[..len]) {
            *start = next;
            next += count;
        }
        debug_assert_eq!(usize::from(next), self.staged.len());

        let output_start = entries.len();
        entries.resize(output_start + self.staged.len(), MatchCandidate::NONE);
        let mut cursors = self.starts;
        for staged in &self.staged {
            let local = usize::from(staged.local_position);
            let index = output_start + usize::from(cursors[local]);
            entries[index] = staged.candidate;
            cursors[local] += 1;
        }
        let mut next_offset = output_start;
        for &count in &self.counts[..len] {
            next_offset += usize::from(count);
            offsets.push(next_offset);
        }
        debug_assert_eq!(next_offset, entries.len());
    }
}

#[inline]
fn stage_candidate(
    local_position: usize,
    candidate: MatchCandidate,
    counts: &mut [u16; FRONTIER_TILE_POSITIONS],
    staged: &mut Vec<StagedCandidate>,
) {
    counts[local_position] += 1;
    staged.push(StagedCandidate {
        local_position: u16::try_from(local_position)
            .unwrap_or_else(|_| unreachable!("tile-local index fits in u16")),
        candidate,
    });
}

#[inline]
fn direct_bucket_match(
    input: &[u8],
    position: usize,
    available: usize,
    min_distance: usize,
    max_distance: usize,
) -> MatchCandidate {
    let mut best = MatchCandidate::NONE;
    for distance in min_distance..=max_distance.min(position) {
        let candidate_position = position - distance;
        let length = match_compare::common_prefix_len(
            &input[position..position + available],
            &input[candidate_position..candidate_position + available],
        );
        best.consider(length, distance);
    }
    best
}

// The leading distance-code classes of width <= DIRECT_SCAN_MAX
// end at distance 32. Later truncated classes can also be narrow;
// only the leading contiguous group uses the single SIMD mask.
const NEARBY_DISTANCE_MASK_BITS: usize = u32::BITS as usize;

/// One lane for each of the preceding 32 positions. Compare the first
/// three exact bytes together, so nonmatches need no suffix comparison.
#[inline]
fn direct_prefix_mask(input: &[u8], position: usize) -> u32 {
    use core::simd::{cmp::SimdPartialEq, Simd};
    let start = position - NEARBY_DISTANCE_MASK_BITS;
    let query = &input[position..position + STD_MIN_MATCH];
    let mut equal = Simd::<u8, NEARBY_DISTANCE_MASK_BITS>::from_slice(
        &input[start..start + NEARBY_DISTANCE_MASK_BITS],
    )
    .simd_eq(Simd::splat(query[0]));
    for (offset, &byte) in query.iter().enumerate().skip(1) {
        equal &=
            Simd::from_slice(&input[start + offset..start + offset + NEARBY_DISTANCE_MASK_BITS])
                .simd_eq(Simd::splat(byte));
    }
    // Bit zero represents the nearest (distance-one) candidate.
    let [a, b, c, d, _, _, _, _] = equal.to_bitmask().to_le_bytes();
    u32::from_le_bytes([a, b, c, d]).reverse_bits()
}

fn scan_direct_buckets(
    input: &[u8],
    ranges: &[(usize, usize)],
    tile_start: usize,
    tile_len: usize,
    scratch: &mut FrontierTileScratch,
) {
    let max_direct = ranges.iter().map(|&(_, end)| end).max().unwrap_or(0);
    if max_direct == 0 {
        return;
    }
    let leading =
        max_direct <= NEARBY_DISTANCE_MASK_BITS && ranges.first().is_some_and(|r| r.0 == 1);
    for (local, &available) in scratch.available[..tile_len].iter().enumerate() {
        let available = usize::from(available);
        if available < STD_MIN_MATCH {
            continue;
        }
        let position = tile_start + local;
        if leading && position >= NEARBY_DISTANCE_MASK_BITS {
            let mut mask = direct_prefix_mask(input, position)
                & (u32::MAX >> (NEARBY_DISTANCE_MASK_BITS - max_direct));
            let mut code = None;
            let mut best = MatchCandidate::NONE;
            while mask != 0 {
                let distance = mask.trailing_zeros() as usize + 1;
                mask &= mask - 1;
                let next_code = State::d_code(distance - 1);
                if code != Some(next_code) {
                    if best.length != 0 {
                        stage_candidate(local, best, &mut scratch.counts, &mut scratch.staged);
                    }
                    best = MatchCandidate::NONE;
                    code = Some(next_code);
                }
                let candidate_position = position - distance;
                let length = STD_MIN_MATCH
                    + match_compare::common_prefix_len(
                        &input[position + STD_MIN_MATCH..position + available],
                        &input[candidate_position + STD_MIN_MATCH..candidate_position + available],
                    );
                best.consider(length, distance);
                // The distance mask is visited nearest first, so once
                // the full available prefix matches, no remaining member
                // of this class can improve length or distance ties.
                if length == available {
                    let (_, maximum) = ranges[usize::from(next_code)];
                    mask &= u32::MAX
                        .checked_shl(u32::try_from(maximum).unwrap_or(32))
                        .unwrap_or(0);
                }
            }
            if best.length != 0 {
                stage_candidate(local, best, &mut scratch.counts, &mut scratch.staged);
            }
            continue;
        }
        for &(min_distance, max_distance) in ranges {
            if min_distance == 0 {
                continue;
            }
            let best = direct_bucket_match(input, position, available, min_distance, max_distance);
            if best.length != 0 {
                stage_candidate(local, best, &mut scratch.counts, &mut scratch.staged);
            }
        }
    }
}

fn scan_rank_buckets(
    input: &[u8],
    suffixes: &SuffixRanks,
    ranges: &[(usize, usize); D_CODES],
    active: &mut [Option<RankSet>; D_CODES],
    codes: core::ops::Range<usize>,
    tile: core::ops::Range<usize>,
    scratch: &mut FrontierTileScratch,
) {
    for code in codes {
        let (min_distance, max_distance) = ranges[code];
        if min_distance == 0 {
            continue;
        }
        let Some(active) = &mut active[code] else {
            unreachable!("only RankSet-backed classes enter the tiled traversal");
        };
        for (local, &available) in scratch.available[..tile.len()].iter().enumerate() {
            let available = usize::from(available);
            if available < STD_MIN_MATCH {
                continue;
            }
            let position = tile.start + local;
            let Some(add) = position.checked_sub(min_distance) else {
                continue;
            };
            active.insert(suffixes.rank_of(add));
            if let Some(remove) = position.checked_sub(max_distance + 1) {
                active.remove(suffixes.rank_of(remove));
            }

            let mut best = MatchCandidate::NONE;
            let (predecessor, successor) = active.neighbors(scratch.ranks[local]);
            let mut consider_rank = |rank: usize| {
                let candidate_position = suffixes.position_at(rank);
                let distance = position - candidate_position;
                debug_assert!((min_distance..=max_distance).contains(&distance));
                let length = match_compare::common_prefix_len(
                    &input[position..position + available],
                    &input[candidate_position..candidate_position + available],
                );
                best.consider(length, distance);
            };
            if let Some(rank) = predecessor {
                consider_rank(rank);
            }
            if let Some(rank) = successor {
                consider_rank(rank);
            }
            if best.length != 0 {
                stage_candidate(local, best, &mut scratch.counts, &mut scratch.staged);
            }
        }
    }
}

/// Whole-input helper for the exact frontier oracle and parser fixtures.
#[cfg(test)]
fn build_match_candidates(
    input: &[u8],
    max_distance: usize,
) -> Result<MatchCandidates, ReturnCode> {
    FrontierSource::new(input, max_distance)?.build(0..input.len())
}

/// Immutable suffix order and distance geometry for one input context.
/// Build only the query ranges a parser needs. A preceding history byte
/// remains an eligible match source even when no record is built for it.
/// Holding the input borrow here prevents a suffix order from being reused
/// with another byte slice.
struct FrontierSource<'a> {
    input: &'a [u8],
    max_distance: usize,
    ranges: [(usize, usize); D_CODES],
    first_rank_code: usize,
    suffixes: SuffixRanks,
}

impl<'a> FrontierSource<'a> {
    fn new(input: &'a [u8], max_distance: usize) -> Result<Self, ReturnCode> {
        if max_distance > 1usize << crate::MAX_WBITS {
            return Err(ReturnCode::StreamError);
        }
        let ranges = distance_code_ranges(max_distance);
        let first_rank_code = ranges
            .iter()
            .position(|&(min, max)| min != 0 && max - min + 1 > DIRECT_SCAN_MAX)
            .unwrap_or(D_CODES);
        let suffixes = SuffixRanks::new(input, first_rank_code < D_CODES)?;
        Ok(Self {
            input,
            max_distance,
            ranges,
            first_rank_code,
            suffixes,
        })
    }

    /// In each wide distance interval, the predecessor/successor in
    /// suffix order recover the longest match for that entire code.
    /// Preserve the complete ordered record sequence for any query range.
    fn build(&self, query: Range<usize>) -> Result<MatchCandidates, ReturnCode> {
        if query.start > query.end || query.end > self.input.len() {
            return Err(ReturnCode::StreamError);
        }
        let input = self.input;
        let suffixes = &self.suffixes;
        let ranges = &self.ranges;
        let first_rank_code = self.first_rank_code;
        let first_query = query.start;
        if let Some(groups) = &suffixes.prefix_groups {
            return Ok(group_local_frontier(
                input,
                suffixes,
                groups,
                ranges,
                self.max_distance,
                query,
            ));
        }
        let mut active: [Option<RankSet>; D_CODES] = core::array::from_fn(|code| {
            let (min_distance, max_distance) = ranges[code];
            (min_distance != 0 && max_distance - min_distance + 1 > DIRECT_SCAN_MAX)
                .then(|| RankSet::new(input.len()))
        });
        if first_query != 0 {
            for (set, &(min, max)) in active.iter_mut().zip(ranges) {
                if let Some(set) = set {
                    if let Some(end) = first_query.checked_sub(min) {
                        for position in first_query.saturating_sub(max + 1)..end {
                            set.insert(suffixes.rank_of(position));
                        }
                    }
                }
            }
        }
        let last_rank_code = active
            .iter()
            .rposition(Option::is_some)
            .map_or(first_rank_code, |code| code + 1);

        let mut offsets = Vec::with_capacity(query.len() + 1);
        let mut entries = Vec::new();
        let mut scratch = FrontierTileScratch::new();
        offsets.push(0);

        for tile_start in query.clone().step_by(FRONTIER_TILE_POSITIONS) {
            let tile_len = tile_start
                .saturating_add(FRONTIER_TILE_POSITIONS)
                .min(query.end)
                - tile_start;
            scratch.prepare(suffixes, input.len(), tile_start, tile_len);

            // Short classes are position-major. Wide classes are class-major
            // for locality. A truncated final short class must be emitted last.
            scan_direct_buckets(
                input,
                &ranges[..first_rank_code],
                tile_start,
                tile_len,
                &mut scratch,
            );
            scan_rank_buckets(
                input,
                suffixes,
                ranges,
                &mut active,
                first_rank_code..last_rank_code,
                tile_start..tile_start + tile_len,
                &mut scratch,
            );
            scan_direct_buckets(
                input,
                &ranges[last_rank_code..],
                tile_start,
                tile_len,
                &mut scratch,
            );
            scratch.scatter(tile_len, &mut entries, &mut offsets);
        }
        debug_assert_eq!(offsets.last().copied(), Some(entries.len()));
        Ok(MatchCandidates {
            first_query,
            offsets,
            entries,
        })
    }
}

/// Dense ordered set over suffix ranks. Each level is a bitset whose set bits identify non-empty
/// words in the level below it, so predecessor/successor queries skip arbitrarily large empty rank
/// intervals without pointer chasing or per-entry allocation.
/// Common bounded geometry has at most 4096 base words and 64 middle
/// words. Preserve the generic hierarchy for arbitrarily large suffix sets.
enum RankSet {
    Three(ThreeLevelRankSet),
    Generic(GenericRankSet),
}

struct ThreeLevelRankSet {
    base: Vec<u64>,
    middle: Vec<u64>,
    top: u64,
}

struct GenericRankSet {
    levels: Vec<Vec<u64>>,
}

impl RankSet {
    fn new(universe: usize) -> Self {
        let base_words = universe.div_ceil(64).max(1);
        if base_words <= 64 * 64 {
            let middle_words = base_words.div_ceil(64).max(1);
            Self::Three(ThreeLevelRankSet {
                base: vec![0; base_words],
                middle: vec![0; middle_words],
                top: 0,
            })
        } else {
            Self::Generic(GenericRankSet::new(universe))
        }
    }

    #[inline]
    fn insert(&mut self, rank: usize) {
        match self {
            Self::Three(set) => set.insert(rank),
            Self::Generic(set) => set.insert(rank),
        }
    }

    #[inline]
    fn remove(&mut self, rank: usize) {
        match self {
            Self::Three(set) => set.remove(rank),
            Self::Generic(set) => set.remove(rank),
        }
    }

    #[inline]
    fn neighbors(&self, rank: usize) -> (Option<usize>, Option<usize>) {
        match self {
            Self::Three(set) => set.neighbors(rank),
            Self::Generic(set) => set.neighbors(rank),
        }
    }
}

impl ThreeLevelRankSet {
    #[inline]
    fn insert(&mut self, rank: usize) {
        let base_index = rank / 64;
        let base_bit = 1u64 << (rank % 64);
        let base_word = &mut self.base[base_index];
        debug_assert_eq!(
            *base_word & base_bit,
            0,
            "rank is inserted only once while active"
        );
        let base_was_empty = *base_word == 0;
        *base_word |= base_bit;
        if !base_was_empty {
            return;
        }

        let middle_index = base_index / 64;
        let middle_bit = 1u64 << (base_index % 64);
        let middle_word = &mut self.middle[middle_index];
        let middle_was_empty = *middle_word == 0;
        *middle_word |= middle_bit;
        if middle_was_empty {
            self.top |= 1u64 << middle_index;
        }
    }

    #[inline]
    fn remove(&mut self, rank: usize) {
        let base_index = rank / 64;
        let base_bit = 1u64 << (rank % 64);
        let base_word = &mut self.base[base_index];
        debug_assert_ne!(*base_word & base_bit, 0);
        *base_word &= !base_bit;
        if *base_word != 0 {
            return;
        }

        let middle_index = base_index / 64;
        let middle_bit = 1u64 << (base_index % 64);
        let middle_word = &mut self.middle[middle_index];
        debug_assert_ne!(*middle_word & middle_bit, 0);
        *middle_word &= !middle_bit;
        if *middle_word == 0 {
            self.top &= !(1u64 << middle_index);
        }
    }

    #[inline]
    fn neighbors(&self, rank: usize) -> (Option<usize>, Option<usize>) {
        let word_index = rank / 64;
        let raw_word = self.base[word_index];
        let bit_index = rank % 64;
        let bit = 1u64 << bit_index;
        let lower_mask = bit - 1;
        let lower = raw_word & lower_mask;
        let upper = raw_word & !(lower_mask | bit);

        if lower != 0 && upper != 0 {
            return (
                Some(word_index * 64 + highest_bit(lower)),
                Some(word_index * 64 + upper.trailing_zeros() as usize),
            );
        }

        let middle_index = word_index / 64;
        let middle_bit_index = word_index % 64;
        let middle_word = self.middle[middle_index];
        let middle_bit = 1u64 << middle_bit_index;
        let middle_lower_mask = middle_bit - 1;

        let predecessor = if lower != 0 {
            Some(word_index * 64 + highest_bit(lower))
        } else {
            let previous_in_middle = middle_word & middle_lower_mask;
            let previous_word = if previous_in_middle != 0 {
                Some(middle_index * 64 + highest_bit(previous_in_middle))
            } else if middle_index != 0 {
                let top_bit = 1u64 << middle_index;
                let previous_middle = self.top & (top_bit - 1);
                (previous_middle != 0).then(|| {
                    let middle_index = highest_bit(previous_middle);
                    let word = self.middle[middle_index];
                    debug_assert_ne!(word, 0);
                    middle_index * 64 + highest_bit(word)
                })
            } else {
                None
            };
            previous_word.map(|previous_word| {
                let word = self.base[previous_word];
                debug_assert_ne!(word, 0);
                previous_word * 64 + highest_bit(word)
            })
        };

        let successor = if upper != 0 {
            Some(word_index * 64 + upper.trailing_zeros() as usize)
        } else {
            let next_in_middle = middle_word & !(middle_lower_mask | middle_bit);
            let next_word = if next_in_middle != 0 {
                Some(middle_index * 64 + next_in_middle.trailing_zeros() as usize)
            } else {
                let top_bit = 1u64 << middle_index;
                let next_middle = self.top & !((top_bit - 1) | top_bit);
                (next_middle != 0).then(|| {
                    let middle_index = next_middle.trailing_zeros() as usize;
                    let word = self.middle[middle_index];
                    debug_assert_ne!(word, 0);
                    middle_index * 64 + word.trailing_zeros() as usize
                })
            };
            next_word.map(|next_word| {
                let word = self.base[next_word];
                debug_assert_ne!(word, 0);
                next_word * 64 + word.trailing_zeros() as usize
            })
        };

        (predecessor, successor)
    }
}

impl GenericRankSet {
    fn new(universe: usize) -> Self {
        let mut levels = Vec::new();
        let mut words = universe.div_ceil(64).max(1);
        levels.push(vec![0; words]);
        while words > 1 {
            words = words.div_ceil(64);
            levels.push(vec![0; words]);
        }
        Self { levels }
    }

    #[inline]
    fn insert(&mut self, rank: usize) {
        let mut index = rank;
        for level in 0..self.levels.len() {
            let word_index = index / 64;
            let bit = 1u64 << (index % 64);
            let word = &mut self.levels[level][word_index];
            if level == 0 {
                debug_assert_eq!(*word & bit, 0, "rank is inserted only once while active");
            }
            let was_empty = *word == 0;
            *word |= bit;
            if !was_empty {
                return;
            }
            index = word_index;
        }
    }

    #[inline]
    fn remove(&mut self, rank: usize) {
        let mut index = rank;
        for level in 0..self.levels.len() {
            let word_index = index / 64;
            let bit = 1u64 << (index % 64);
            let word = &mut self.levels[level][word_index];
            if level == 0 {
                debug_assert_ne!(*word & bit, 0);
            }
            *word &= !bit;
            if *word != 0 {
                return;
            }
            index = word_index;
        }
    }

    /// Return both neighbors from one local bitset lookup. Strictly exclude
    /// the query rank, then consult the hierarchy only for a missing side.
    #[inline]
    fn neighbors(&self, rank: usize) -> (Option<usize>, Option<usize>) {
        let word_index = rank / 64;
        let bit_index = rank % 64;
        let word = self.levels[0][word_index];

        let previous = word & low_bits(bit_index);
        let predecessor = if previous != 0 {
            Some(word_index * 64 + highest_bit(previous))
        } else {
            let previous_word = self.previous_set_bit(1, word_index);
            previous_word.map(|index| {
                let bits = self.levels[0][index];
                debug_assert_ne!(bits, 0);
                index * 64 + highest_bit(bits)
            })
        };

        let following = word & !low_bits(bit_index + 1);
        let successor = if following != 0 {
            Some(word_index * 64 + following.trailing_zeros() as usize)
        } else {
            let next_word = self.next_set_bit(1, word_index + 1);
            next_word.map(|index| {
                let bits = self.levels[0][index];
                debug_assert_ne!(bits, 0);
                index * 64 + bits.trailing_zeros() as usize
            })
        };

        (predecessor, successor)
    }

    #[inline]
    fn previous_set_bit(&self, target_level: usize, before: usize) -> Option<usize> {
        let mut level = target_level;
        let mut before = before;
        while level < self.levels.len() && before != 0 {
            let word_index = (before - 1) / 64;
            let bits = (before - 1) % 64 + 1;
            let word = self.levels[level][word_index] & low_bits(bits);
            if word != 0 {
                let mut index = word_index * 64 + highest_bit(word);
                while level > target_level {
                    level -= 1;
                    let word = self.levels[level][index];
                    debug_assert_ne!(word, 0);
                    index = index * 64 + highest_bit(word);
                }
                return Some(index);
            }
            before = word_index;
            level += 1;
        }
        None
    }

    #[inline]
    fn next_set_bit(&self, target_level: usize, start: usize) -> Option<usize> {
        let mut level = target_level;
        let mut start = start;
        while level < self.levels.len() {
            let word_index = start / 64;
            let &raw_word = self.levels[level].get(word_index)?;
            let word = raw_word & !low_bits(start % 64);
            if word != 0 {
                let mut index = word_index * 64 + word.trailing_zeros() as usize;
                while level > target_level {
                    level -= 1;
                    let word = self.levels[level][index];
                    debug_assert_ne!(word, 0);
                    index = index * 64 + word.trailing_zeros() as usize;
                }
                return Some(index);
            }
            start = word_index + 1;
            level += 1;
        }
        None
    }
}

#[inline]
const fn low_bits(bits: usize) -> u64 {
    if bits == 0 {
        0
    } else if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

#[inline]
const fn highest_bit(word: u64) -> usize {
    63 - word.leading_zeros() as usize
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

/// Convert an already-bounded suffix-array index to the platform slice index.
///
/// All values passed here index the input, its suffix permutation, or the
/// associated counting-sort tables. Their backing arrays have already been
/// allocated at a representable usize length, so even on a narrow-pointer
/// target the index is bounded by a real allocation rather than arbitrary u32.
#[inline]
const fn suffix_index(value: u32) -> usize {
    value as usize
}

/// A suffix permutation and its inverse. Both arrays remain packed u32; all
/// consumers operate on ordinary slice indices through `rank_of/position_at`.
/// This is the only boundary converting the compact rank/position storage.
struct SuffixRanks {
    sa: Vec<u32>,
    rank: Vec<u32>,
    prefix_groups: Option<PrefixGroups>,
}

/// The packed suffix permutation and counting-sort offsets require indices
/// representable as u32, independently of the host pointer width.
pub(super) fn checked_suffix_input_len(len: usize) -> Result<u32, ReturnCode> {
    u32::try_from(len).map_err(|_| ReturnCode::StreamError)
}

impl SuffixRanks {
    #[inline]
    fn rank_of(&self, position: usize) -> usize {
        suffix_index(self.rank[position])
    }

    #[inline]
    fn position_at(&self, suffix_rank: usize) -> usize {
        suffix_index(self.sa[suffix_rank])
    }

    fn new(input: &[u8], collect_groups: bool) -> Result<Self, ReturnCode> {
        let length = checked_suffix_input_len(input.len())?;
        if input.is_empty() {
            return Ok(Self {
                sa: Vec::new(),
                rank: Vec::new(),
                prefix_groups: None,
            });
        }

        let mut sa = (0..length).collect::<Vec<_>>();
        let mut scratch = vec![0u32; input.len()];
        let mut rank = input
            .iter()
            .map(|&byte| u32::from(byte) + 1)
            .collect::<Vec<_>>();
        let mut next_rank = vec![0u32; input.len()];
        let max_rank = rank.iter().copied().max().unwrap_or(0);
        let mut counts = vec![0u32; input.len().max(256) + 1];
        let mut span = 1usize;
        let mut prefix_groups = None;
        counting_sort_suffixes(&sa, &mut scratch, max_rank, &mut counts, |position| {
            suffix_second_rank(&rank, position, span)
        });
        counting_sort_suffixes(&scratch, &mut sa, max_rank, &mut counts, |position| {
            rank[suffix_index(position)]
        });

        loop {
            let mut class = 1u32;
            counts[1] = 0;
            next_rank[suffix_index(sa[0])] = class;
            let mut previous_pair = (
                rank[suffix_index(sa[0])],
                suffix_second_rank(&rank, sa[0], span),
            );
            for (suffix_rank, &position) in (1..length).zip(&sa[1..]) {
                let pair = (
                    rank[suffix_index(position)],
                    suffix_second_rank(&rank, position, span),
                );
                if pair != previous_pair {
                    class += 1;
                    counts[suffix_index(class)] = suffix_rank;
                }
                next_rank[suffix_index(position)] = class;
                previous_pair = pair;
            }

            if collect_groups {
                if span == 1 && class == length {
                    // Unique two-byte prefixes also imply unique triples.
                    prefix_groups = Some(PrefixGroups::unique());
                } else if span == 2 {
                    prefix_groups =
                        PrefixGroups::from_refinement(input, &sa, &counts[1..=suffix_index(class)]);
                }
            }
            core::mem::swap(&mut rank, &mut next_rank);
            if class == length {
                break;
            }
            span = span.checked_mul(2).ok_or(ReturnCode::StreamError)?;

            // Each equivalence class occupies a contiguous range of the
            // previous SA. Reclassification already recorded its first slot,
            // so no histogram or prefix sum is needed in this round.
            let shift =
                u32::try_from(span.min(input.len())).map_err(|_| ReturnCode::StreamError)?;
            for position in (length - shift..length).chain(
                sa.iter()
                    .filter_map(|&position| position.checked_sub(shift)),
            ) {
                let key = suffix_index(rank[suffix_index(position)]);
                scratch[suffix_index(counts[key])] = position;
                counts[key] += 1;
            }
            core::mem::swap(&mut sa, &mut scratch);
        }

        // The doubling ranks are 1-based equivalence classes. The parser wants direct indices into
        // `sa`, so materialize the exact inverse suffix-array permutation once.
        for (suffix_rank, &position) in (0..length).zip(&sa) {
            rank[suffix_index(position)] = suffix_rank;
        }

        Ok(Self {
            sa,
            rank,
            prefix_groups,
        })
    }
}

/// Stable counting sort over compact suffix positions using a caller-selected
/// rank key. Both radix passes must use identical offset/prefix accounting;
/// the only difference is which half of the suffix-rank pair provides the key.
fn counting_sort_suffixes(
    source: &[u32],
    target: &mut [u32],
    max_rank: u32,
    counts: &mut [u32],
    key_of: impl Fn(u32) -> u32,
) {
    debug_assert_eq!(source.len(), target.len());
    let counts = &mut counts[..=suffix_index(max_rank)];
    counts.fill(0);
    for &position in source {
        let key = suffix_index(key_of(position));
        counts[key] += 1;
    }
    counts_to_offsets(counts);
    for &position in source {
        let key = suffix_index(key_of(position));
        target[suffix_index(counts[key])] = position;
        counts[key] += 1;
    }
}

fn counts_to_offsets(counts: &mut [u32]) {
    let mut offset = 0u32;
    for count in counts {
        let frequency = *count;
        *count = offset;
        offset += frequency;
    }
}

#[inline]
const fn suffix_second_rank(rank: &[u32], position: u32, span: usize) -> u32 {
    let position = suffix_index(position);
    if position < rank.len().saturating_sub(span) {
        rank[position + span]
    } else {
        0
    }
}

const MIN_BLOCK_LENGTH: usize = 5_000;
// Keep ordinary coarse ranges individually stored-capable. The tiny-tail rule may extend one range
// beyond this; exact parsing then either proves the long Huffman block worthwhile and token-safe or
// returns a split point before emission.
const SOFT_MAX_BLOCK_LENGTH: usize = 60_000;
pub const MAX_COARSE_BLOCK_INPUT: usize = SOFT_MAX_BLOCK_LENGTH + MIN_BLOCK_LENGTH - 1;
const OBSERVATIONS_PER_CHECK: u32 = 512;
const SPLIT_OBSERVATION_TYPES: usize = 10;
// Match libdeflate level 12's near-optimal search budget for block-statistics gathering.
const SPLIT_MATCH_SEARCH_DEPTH: usize = 300;

#[inline]
const fn choose_min_match_len(used_literals: usize) -> usize {
    match used_literals {
        0..=5 => 9,
        6..=7 => 8,
        8..=9 => 7,
        10..=15 => 6,
        16..=44 => 5,
        45..=79 => 4,
        _ => 3,
    }
}

fn splitter_min_match_len(input: &[u8], block_start: usize, block_limit: usize) -> usize {
    let sample_end = block_start
        .saturating_add(4_096)
        .min(block_limit)
        .min(input.len());
    if sample_end.saturating_sub(block_start) < 512 {
        return STD_MIN_MATCH;
    }

    let mut used = [false; 256];
    for &byte in &input[block_start..sample_end] {
        used[usize::from(byte)] = true;
    }
    choose_min_match_len(used.into_iter().filter(|used| *used).count())
}

fn splitter_block_limit(input_len: usize, block_start: usize) -> usize {
    let soft_end = block_start
        .saturating_add(SOFT_MAX_BLOCK_LENGTH)
        .min(input_len);
    if input_len - soft_end < MIN_BLOCK_LENGTH {
        input_len
    } else {
        soft_end
    }
}

#[derive(Clone, Debug, Default)]
struct BlockSplitStats {
    old: [u32; SPLIT_OBSERVATION_TYPES],
    recent: [u32; SPLIT_OBSERVATION_TYPES],
    old_count: u32,
    recent_count: u32,
}

impl BlockSplitStats {
    fn observe_literal(&mut self, byte: u8) {
        // High two value bits plus parity gives eight cheap categories while still distinguishing
        // text-like and binary distributions; this is the same geometry used by libdeflate.
        let class = literal_observation_class(byte);
        self.recent[class] += 1;
        self.recent_count += 1;
    }

    fn observe_match(&mut self, length: usize) {
        self.recent[match_observation_class(length)] += 1;
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
    const fn keep_recent_as_new_block(&mut self) {
        self.old = self.recent;
        self.old_count = self.recent_count;
        self.recent = [0; SPLIT_OBSERVATION_TYPES];
        self.recent_count = 0;
    }

    fn distribution_changed(&mut self, block_length: usize) -> bool {
        if self.old_count != 0 {
            // Compare normalized categorical distributions without division. This is libdeflate's
            // block-split criterion expressed in wider arithmetic: an L1 distance threshold of
            // 200/512, an extra penalty for very short blocks, and a slowly increasing bias toward
            // splitting as a block gets longer.
            let mut total_delta = 0u64;
            for index in 0..SPLIT_OBSERVATION_TYPES {
                let expected = u64::from(self.old[index]) * u64::from(self.recent_count);
                let actual = u64::from(self.recent[index]) * u64::from(self.old_count);
                total_delta += expected.abs_diff(actual);
            }

            let num_items = u64::from(self.old_count) + u64::from(self.recent_count);
            let mut cutoff = u64::from(self.recent_count) * 200 / 512 * u64::from(self.old_count);
            if block_length < 10_000 && num_items < 8_192 {
                cutoff += cutoff * (8_192 - num_items) / 8_192;
            }
            let length_bias = (block_length / 4_096) as u64 * u64::from(self.old_count);
            if total_delta + length_bias >= cutoff {
                return true;
            }
        }
        self.merge_recent();
        false
    }
}

/// Pick inexpensive candidate block boundaries before running the expensive exact parser.
///
/// The splitter uses a bounded rolling hash chain to classify the input into literal / short-match /
/// long-match observations and detect distribution changes. This is deliberately cheaper than the
/// exact suffix-rank frontier; exact match discovery and dynamic programming happen only after a
/// boundary has been selected.
pub fn choose_block_ranges(
    input: &[u8],
    max_distance: usize,
) -> Result<Vec<Range<usize>>, ReturnCode> {
    const HASH_SIZE: usize = 1 << 16;

    // A caller-supplied distance can otherwise overflow the ring size
    // before the exact candidate builder gets a chance to reject it.
    if max_distance > 1usize << crate::MAX_WBITS {
        return Err(ReturnCode::StreamError);
    }
    if input.is_empty() {
        return Ok(core::iter::once(0..0).collect());
    }
    let mut head = vec![usize::MAX; HASH_SIZE];
    // Keep one previous-link slot for every position in the legal distance window. Unlike storing a
    // link for the entire input, this stays bounded; a slot cannot be overwritten while its position
    // is still reachable because the ring is one entry wider than `max_distance`.
    let mut previous = vec![usize::MAX; max_distance + 1];
    let mut blocks = Vec::new();
    let mut stats = BlockSplitStats::default();
    let mut block_start = 0usize;
    let mut previous_check = None;
    let mut position = 0usize;
    // Avoid manufacturing a tiny tail solely because of the soft maximum: the final coarse range
    // may extend by less than MIN_BLOCK_LENGTH and is capacity-checked after exact parsing.
    let mut block_limit = splitter_block_limit(input.len(), block_start);
    let mut min_split_match = splitter_min_match_len(input, block_start, block_limit);

    while position < input.len() {
        if position >= block_limit {
            blocks.push(block_start..position);
            block_start = position;
            stats = BlockSplitStats::default();
            previous_check = None;
            block_limit = splitter_block_limit(input.len(), block_start);
            min_split_match = splitter_min_match_len(input, block_start, block_limit);
            continue;
        }

        let available = STD_MAX_MATCH.min(block_limit - position);
        let mut match_length = 0usize;
        if available >= STD_MIN_MATCH && position + 2 < input.len() {
            let hash = split_hash3(input, position);
            let mut candidate = head[hash];
            for _ in 0..SPLIT_MATCH_SEARCH_DEPTH {
                if candidate == usize::MAX || position - candidate > max_distance {
                    break;
                }
                let length = match_compare::common_prefix_len(
                    &input[position..position + available],
                    &input[candidate..candidate + available],
                );
                match_length = match_length.max(length);
                if match_length == available {
                    break;
                }
                candidate = previous[candidate % previous.len()];
            }
        }

        let step = if match_length >= min_split_match {
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
                let hash = split_hash3(input, inserted);
                let slot = inserted % previous.len();
                previous[slot] = head[hash];
                head[hash] = inserted;
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
            block_limit = splitter_block_limit(input.len(), block_start);
            min_split_match = splitter_min_match_len(input, block_start, block_limit);
        } else {
            previous_check = Some(position);
        }
    }

    if block_start < input.len() {
        blocks.push(block_start..input.len());
    }
    debug_assert_eq!(blocks.first().map(|range| range.start), Some(0));
    debug_assert_eq!(blocks.last().map(|range| range.end), Some(input.len()));
    debug_assert!(blocks.windows(2).all(|pair| pair[0].end == pair[1].start));
    debug_assert!(blocks
        .iter()
        .all(|block| block.len() <= MAX_COARSE_BLOCK_INPUT));
    Ok(blocks)
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
pub fn parse_block(input: &[u8], candidates: &MatchCandidates, costs: &TokenCosts) -> ParseResult {
    parse_range(input, candidates, 0..input.len(), costs)
}

/// Reuse a bounded cost horizon and block-sized traceback across cost-model
/// passes. Every forward edge is at most one DEFLATE match long, so finalized
/// costs can be discarded without losing any subsequent relaxation.
#[derive(Default)]
struct ParseScratch {
    best_cost: Vec<u64>,
    steps: Vec<u16>,
}

impl ParseScratch {
    const INF: u64 = u64::MAX / 4;
    // A power-of-two tile at least as large as the longest edge gives
    // contiguous relaxation spans even at the final position of a tile.
    const COST_TILE: usize = (STD_MAX_MATCH + 1).next_power_of_two();
    const COST_CAPACITY: usize = Self::COST_TILE + STD_MAX_MATCH;

    fn advance_cost_horizon(costs: &mut [u64]) {
        let retained = costs.len() - Self::COST_TILE;
        costs.copy_within(Self::COST_TILE.., 0);
        costs[retained..].fill(Self::INF);
    }

    fn reset(&mut self, block_len: usize) {
        assert!(block_len < usize::MAX, "DP block has too many positions");
        let slots = block_len + 1;
        // clear() retains each Vec's allocation; resize() then initializes
        // every slot exactly once, including after a shorter previous parse.
        self.best_cost.clear();
        self.best_cost
            .resize(slots.min(Self::COST_CAPACITY), Self::INF);
        self.best_cost[0] = 0;
        self.steps.clear();
        self.steps.resize(slots, 0);
    }
}

pub fn parse_range(
    input: &[u8],
    candidates: &MatchCandidates,
    block: Range<usize>,
    costs: &TokenCosts,
) -> ParseResult {
    parse_range_with_scratch(
        input,
        candidates,
        block,
        costs,
        &mut ParseScratch::default(),
    )
}

fn parse_range_with_scratch(
    input: &[u8],
    candidates: &MatchCandidates,
    block: Range<usize>,
    costs: &TokenCosts,
    scratch: &mut ParseScratch,
) -> ParseResult {
    const INF: u64 = ParseScratch::INF;

    assert_eq!(candidates.end(), input.len());
    assert!(candidates.first_query <= block.start);
    assert!(block.start <= block.end && block.end <= input.len());

    let block_len = block.len();
    scratch.reset(block_len);
    let (best_cost, steps) = (&mut scratch.best_cost, &mut scratch.steps);
    let prices = LengthPrices::new(costs);
    // DEFLATE distance cost depends on the Huffman distance symbol, not the
    // numeric distance. Build 30 entries rather than allocating and filling
    // one entry per possible byte offset in the 32 KiB window.
    let distance_costs: [u32; D_CODES] =
        core::array::from_fn(|code| costs.distance_symbol_cost(code));
    // The per-position bitmap identifies live distance slots and already
    // records every length where the cheapest capable distance can change.
    // Reset only these five words, not the values, at each new position.
    let mut distance_cost_at_max = [0u32; STD_MAX_MATCH + 1];
    let mut cost_base = 0;
    for local_position in 0..block_len {
        if local_position - cost_base == ParseScratch::COST_TILE {
            // All earlier positions are finalized. Keep the outstanding
            // targets from their matches, never resetting a live cost.
            ParseScratch::advance_cost_horizon(best_cost);
            cost_base = local_position;
        }
        let cost_position = local_position - cost_base;
        let position = block.start + local_position;
        let prefix_cost = best_cost[cost_position];
        debug_assert_ne!(prefix_cost, INF);

        relax(
            &mut best_cost[cost_position + 1],
            &mut steps[local_position + 1],
            prefix_cost + u64::from(costs.literal_cost(input[position])),
            1,
        );

        let available = STD_MAX_MATCH.min(block.end - position);
        if available < STD_MIN_MATCH {
            continue;
        }

        let alternatives = candidates.for_position(position);
        if alternatives.is_empty() {
            continue;
        }
        let mut markers = [0u64; STD_MAX_MATCH / 64 + 1];
        let mut longest_match = 0usize;
        for candidate in alternatives {
            let max_len = usize::from(candidate.length).min(available);
            longest_match = longest_match.max(max_len);
            let marker = 1u64 << (max_len % 64);
            let word = &mut markers[max_len / 64];
            let code = usize::from(State::d_code(usize::from(candidate.distance) - 1));
            let cost = distance_costs[code];
            let slot = &mut distance_cost_at_max[max_len];
            if *word & marker == 0 || cost < *slot {
                *slot = cost;
                *word |= marker;
            }
        }

        if longest_match < STD_MIN_MATCH {
            continue;
        }
        if longest_match >= GROUPED_RELAX_MIN_LENGTH {
            MatchRelaxation {
                costs: &mut best_cost[cost_position..],
                steps: &mut steps[local_position..],
                prices: &prices,
                prefix_cost,
            }
            .long_matches(markers, longest_match, &distance_cost_at_max);
            continue;
        }
        let mut distance_cost = distance_cost_at_max[longest_match];
        for length in (STD_MIN_MATCH..=longest_match).rev() {
            if markers[0] & (1u64 << length) != 0 {
                distance_cost = distance_cost.min(distance_cost_at_max[length]);
            }
            let token_cost = u64::from(prices.costs[length]) + u64::from(distance_cost);
            relax(
                &mut best_cost[cost_position + length],
                &mut steps[local_position + length],
                prefix_cost + token_cost,
                compact_match_length(length),
            );
        }
    }

    ParseResult {
        cost_x16: best_cost[block_len - cost_base],
        tokens: reconstruct_tokens(steps, input, candidates, block, &distance_costs),
    }
}

/// Only the winning predecessor length is needed for the traceback. The
/// immutable frontier and distance-cost model determine the same distance
/// as the forward relaxation: cheapest capable symbol, then longest
/// (clipped) maximum, then first occurrence in frontier order.
fn reconstruct_tokens(
    steps: &[u16],
    input: &[u8],
    candidates: &MatchCandidates,
    block: Range<usize>,
    distance_costs: &[u32; D_CODES],
) -> Vec<ParseToken> {
    // The inverse maximum length fits in the low field. A distance
    // symbol cost is a u16 model price plus at most 13 DEFLATE extra
    // bits at COST_SCALE, so the complete unsigned ordering key fits u32.
    const LENGTH_BITS: u32 = STD_MAX_MATCH.ilog2() + 1;
    let max_match = compact_match_length(STD_MAX_MATCH);
    let mut tokens = Vec::new();
    let mut cursor = block.len();
    while cursor != 0 {
        let length = usize::from(steps[cursor]);
        debug_assert!(length != 0 && length <= cursor);
        cursor -= length;
        let position = block.start + cursor;
        let token = if length == 1 {
            ParseToken::Literal(input[position])
        } else {
            let available = compact_match_length(STD_MAX_MATCH.min(block.end - position));
            let best = candidates
                .for_position(position)
                .iter()
                .filter(|candidate| usize::from(candidate.length) >= length)
                .min_by_key(|candidate| {
                    let code = usize::from(State::d_code(usize::from(candidate.distance) - 1));
                    let cost = distance_costs[code];
                    debug_assert!(cost <= u32::MAX >> LENGTH_BITS);
                    (cost << LENGTH_BITS) | u32::from(max_match - candidate.length.min(available))
                })
                .unwrap_or_else(|| unreachable!("winning match must have a capable distance"));
            ParseToken::matching(length, best.distance)
        };
        tokens.push(token);
    }
    tokens.reverse();
    debug_assert_eq!(
        tokens.iter().map(|token| token.input_len()).sum::<usize>(),
        block.len()
    );
    tokens
}

struct LengthPrices {
    costs: [u32; STD_MAX_MATCH + 1],
    run_start: [usize; STD_MAX_MATCH + 1],
}

impl LengthPrices {
    fn new(model: &TokenCosts) -> Self {
        let mut costs = [0; STD_MAX_MATCH + 1];
        let mut run_start = [STD_MIN_MATCH; STD_MAX_MATCH + 1];
        let mut first = STD_MIN_MATCH;
        let mut previous = 0;
        for length in STD_MIN_MATCH..=STD_MAX_MATCH {
            let code = usize::from(trees_tbl::LENGTH_CODE[length - STD_MIN_MATCH]);
            let cost = model.length_symbol_cost(code);
            if cost != previous {
                first = length;
            }
            costs[length] = cost;
            run_start[length] = first;
            previous = cost;
        }
        Self { costs, run_start }
    }
}

// Short matches are cheaper to enumerate directly. Longer matches contain
// many equal-price endpoints, which can be relaxed in contiguous spans.
const GROUPED_RELAX_MIN_LENGTH: usize = 32;
// The direct short-match traversal tests occupancy in the first word.
const _: () = assert!(GROUPED_RELAX_MIN_LENGTH <= u64::BITS as usize);

struct MatchRelaxation<'a> {
    costs: &'a mut [u64],
    steps: &'a mut [u16],
    prices: &'a LengthPrices,
    prefix_cost: u64,
}

impl MatchRelaxation<'_> {
    fn long_matches(
        &mut self,
        markers: [u64; STD_MAX_MATCH / 64 + 1],
        longest_match: usize,
        distance_cost_at_max: &[u32; STD_MAX_MATCH + 1],
    ) {
        // Only lengths at which another distance becomes capable can
        // change the chosen distance. A descending walk preserves the
        // longer-maximum-first and strict improvement tie rules.
        let mut best = distance_cost_at_max[longest_match];
        let mut high = longest_match;
        for (word, mut mask) in markers.into_iter().enumerate().rev() {
            while mask != 0 {
                let len = word * 64 + highest_bit(mask);
                mask ^= 1 << (len % 64);
                let candidate = distance_cost_at_max[len];
                if candidate < best {
                    self.span(len + 1..=high, best);
                    best = candidate;
                    high = len;
                }
            }
        }
        self.span(STD_MIN_MATCH..=high, best);
    }

    /// Each endpoint is unique for this starting position; traversal
    /// order inside a span cannot affect the DP tie-break.
    #[inline]
    fn span(&mut self, lengths: core::ops::RangeInclusive<usize>, distance_cost: u32) {
        let low = *lengths.start();
        let mut high = *lengths.end();
        let prefix_cost = self.prefix_cost + u64::from(distance_cost);
        loop {
            let start = self.prices.run_start[high].max(low);
            let cost = prefix_cost + u64::from(self.prices.costs[high]);
            let first = compact_match_length(start);
            for ((old, step), length) in self.costs[start..=high]
                .iter_mut()
                .zip(&mut self.steps[start..=high])
                .zip(first..)
            {
                if cost < *old {
                    *old = cost;
                    *step = length;
                }
            }
            if start == low {
                break;
            }
            high = start - 1;
        }
    }
}

#[inline]
const fn relax(best_cost: &mut u64, step: &mut u16, cost: u64, length: u16) {
    if cost < *best_cost {
        *best_cost = cost;
        *step = length;
    }
}

#[cfg(test)]
fn common_prefix(left: &[u8], right: &[u8]) -> usize {
    match_compare::common_prefix_len(left, right)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_match_candidates_position_major(
        input: &[u8],
        max_distance: usize,
    ) -> Result<MatchCandidates, ReturnCode> {
        const DIRECT_SCAN_MAX: usize = 8;
        // Reject invalid history as an API error, not a process-wide panic.
        if max_distance > 1usize << crate::MAX_WBITS {
            return Err(ReturnCode::StreamError);
        }
        // Candidate offsets remain usize because there may be D_CODES entries per byte.
        let suffixes = SuffixRanks::new(input, false)?;
        let ranges = distance_code_ranges(max_distance);
        let mut active: [Option<RankSet>; D_CODES] = core::array::from_fn(|code| {
            let (min_distance, max_distance) = ranges[code];
            (min_distance != 0 && max_distance - min_distance + 1 > DIRECT_SCAN_MAX)
                .then(|| RankSet::new(input.len()))
        });
        let mut offsets = Vec::with_capacity(input.len() + 1);
        let mut entries = Vec::new();
        offsets.push(0);

        for position in 0..input.len() {
            let rank = suffixes.rank_of(position);
            let available = STD_MAX_MATCH.min(input.len() - position);

            for (code, &(min_distance, max_distance)) in ranges.iter().enumerate() {
                if min_distance == 0 {
                    continue;
                }

                let mut best = MatchCandidate::NONE;
                // Both suffix-rank probes and direct scans use exactly the same
                // comparison, distance-bucket tie-breaking and token conversion.
                let mut consider_position = |candidate_position: usize| {
                    debug_assert!(candidate_position < position);
                    let distance = position - candidate_position;
                    debug_assert!((min_distance..=max_distance).contains(&distance));
                    let length = match_compare::common_prefix_len(
                        &input[position..position + available],
                        &input[candidate_position..candidate_position + available],
                    );
                    best.consider(length, distance);
                };

                if let Some(active) = &mut active[code] {
                    if available < STD_MIN_MATCH {
                        continue;
                    }
                    let Some(add) = position.checked_sub(min_distance) else {
                        continue;
                    };
                    active.insert(suffixes.rank_of(add));
                    if let Some(remove) = position.checked_sub(max_distance + 1) {
                        active.remove(suffixes.rank_of(remove));
                    }

                    let (predecessor, successor) = active.neighbors(rank);
                    if let Some(candidate_rank) = predecessor {
                        consider_position(suffixes.position_at(candidate_rank));
                    }
                    if let Some(candidate_rank) = successor {
                        consider_position(suffixes.position_at(candidate_rank));
                    }
                } else if available >= STD_MIN_MATCH {
                    for distance in min_distance..=max_distance.min(position) {
                        consider_position(position - distance);
                    }
                }

                if best.length != 0 {
                    entries.push(best);
                }
            }
            offsets.push(entries.len());
        }

        Ok(MatchCandidates {
            first_query: 0,
            offsets,
            entries,
        })
    }

    #[test]
    fn nearby_simd_mask_matches_every_scalar_candidate() {
        let mut random = 0x92ad_7c51u32;
        let mut low_entropy = [0u8; 289];
        for value in &mut low_entropy {
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            *value = random.to_le_bytes()[0] & 3;
        }
        let periodic = b"abcabcabdabc0123".repeat(33);
        let constant = [b'a'; 288];
        for input in [low_entropy.as_slice(), periodic.as_slice(), &constant] {
            for position in NEARBY_DISTANCE_MASK_BITS..=input.len() - STD_MIN_MATCH {
                let expected = (1..=NEARBY_DISTANCE_MASK_BITS).fold(0u32, |mask, distance| {
                    let candidate = position - distance;
                    if input[position..position + STD_MIN_MATCH]
                        == input[candidate..candidate + STD_MIN_MATCH]
                    {
                        mask | (1 << (distance - 1))
                    } else {
                        mask
                    }
                });
                assert_eq!(direct_prefix_mask(input, position), expected);
            }
        }
    }

    #[test]
    fn vector_frontier_retains_exact_records_at_truncated_distance_boundaries(
    ) -> Result<(), ReturnCode> {
        let mut input = b"abcabcabdabc0123".repeat(69);
        for position in (0..input.len()).step_by(53) {
            input[position] = b'X';
        }
        for distance in (0..=40).chain([
            127, 128, 129, 130, 131, 255, 256, 257, 258, 1023, 1024, 1025, 1028, 1032, 1033, 32768,
        ]) {
            let actual = build_match_candidates(&input, distance)?;
            let expected = build_match_candidates_position_major(&input, distance)?;
            assert_eq!(actual, expected, "distance={distance}");
        }
        Ok(())
    }

    #[test]
    fn rank_set_queries_match_reference_across_hierarchy_shapes() {
        use std::collections::BTreeSet;
        use std::ops::Bound::{Excluded, Unbounded};

        for universe in [65usize, 4_097, 262_144, 262_145, 1_000_000] {
            let mut ranks = vec![
                0,
                1,
                63,
                64,
                65,
                4_095,
                4_096,
                4_097,
                262_143,
                262_144,
                universe - 1,
            ];
            ranks.extend((0..512).map(|i| (i * 7_919 + i * i * 17) % universe));
            ranks.retain(|&rank| rank < universe);
            ranks.sort_unstable();
            ranks.dedup();

            let mut set = RankSet::new(universe);
            let mut reference = BTreeSet::new();
            for &rank in &ranks {
                set.insert(rank);
                reference.insert(rank);
            }

            let mut queries = ranks.clone();
            queries.extend((0..512).map(|i| (i * 12_289 + i * i * 29) % universe));
            queries.sort_unstable();
            queries.dedup();

            let check = |set: &RankSet, reference: &BTreeSet<usize>| {
                for &query in &queries {
                    assert_eq!(
                        set.neighbors(query),
                        (
                            reference.range(..query).next_back().copied(),
                            reference
                                .range((Excluded(query), Unbounded))
                                .next()
                                .copied(),
                        )
                    );
                }
            };
            check(&set, &reference);

            for &rank in ranks.iter().step_by(3) {
                set.remove(rank);
                reference.remove(&rank);
            }
            check(&set, &reference);
        }
    }

    #[test]
    fn iteration_history_checks_full_tokens_across_best_and_worse_paths() {
        let best = vec![ParseToken::Literal(b'a'), ParseToken::Literal(b'b')];
        let worse = vec![ParseToken::Literal(b'a'), ParseToken::Literal(b'c')];
        let improvement = vec![ParseToken::Literal(b'z')];

        let mut history = PreviousParse::None;
        assert!(!history.converged(Some(&best), &best));

        history = PreviousParse::Best;
        assert!(history.converged(Some(&best), &best));
        assert!(!history.converged(Some(&best), &worse));

        history = PreviousParse::Other(worse);
        assert!(history.converged(
            Some(&best),
            &[ParseToken::Literal(b'a'), ParseToken::Literal(b'c')]
        ));
        assert!(!history.converged(Some(&best), &best));
        assert!(!history.converged(Some(&best), &improvement));

        // A later better path becomes the previous path by reference to the
        // selected result, without maintaining a separate copy of its tokens.
        history = PreviousParse::Best;
        assert!(history.converged(Some(&improvement), &improvement));
        assert!(!history.converged(Some(&best), &improvement));
    }

    #[test]
    fn suffix_ranks_preserve_lexicographic_order() -> Result<(), ReturnCode> {
        let mut state = 0x243f_6a88_85a3_08d3u64;
        let mut input = vec![0u8; 32 * 1024];
        for byte in &mut input {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *byte = state.to_le_bytes()[4] & 0x1f;
        }

        let suffixes = SuffixRanks::new(&input, false)?;
        for rank in 0..input.len() {
            let position = suffixes.position_at(rank);
            assert_eq!(suffixes.rank_of(position), rank);
        }
        for pair in suffixes.sa.windows(2) {
            assert!(input[suffix_index(pair[0])..] < input[suffix_index(pair[1])..]);
        }
        Ok(())
    }

    #[test]
    fn shifted_suffix_order_matches_exhaustive_small_alphabets() -> Result<(), ReturnCode> {
        for len in 0..=8usize {
            let combinations = 3usize.pow(u32::try_from(len).map_err(|_| ReturnCode::StreamError)?);
            for mut value in 0..combinations {
                let input = (0..len)
                    .map(|_| {
                        let byte = b'a' + u8::try_from(value % 3).unwrap_or_default();
                        value /= 3;
                        byte
                    })
                    .collect::<Vec<_>>();
                let suffixes = SuffixRanks::new(&input, false)?;
                let mut expected = (0..u32::try_from(len).map_err(|_| ReturnCode::StreamError)?)
                    .collect::<Vec<_>>();
                expected.sort_unstable_by(|&left, &right| {
                    input[suffix_index(left)..].cmp(&input[suffix_index(right)..])
                });
                assert_eq!(suffixes.sa, expected, "input={input:?}");
                for (rank, &position) in expected.iter().enumerate() {
                    assert_eq!(suffixes.rank_of(suffix_index(position)), rank);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn suffix_counting_sort_preserves_order_with_equal_keys() {
        let source = [5u32, 2, 4, 1, 3, 0];
        let keys = [1u32, 2, 0, 2, 1, 0];
        let mut target = [0u32; 6];
        let mut counts = [0u32; 3];
        counting_sort_suffixes(&source, &mut target, 2, &mut counts, |position| {
            keys[suffix_index(position)]
        });
        // The suffix index is the payload. Equal-key order must be stable for
        // the two-pass LSD radix sort to preserve the lexicographic tie break.
        assert_eq!(target, [5, 2, 4, 0, 1, 3]);
        assert_eq!(counts, [2, 4, 6]);
    }

    #[test]
    fn suffix_ranks_invert_repetitive_and_boundary_length_inputs() -> Result<(), ReturnCode> {
        for length in [
            0usize, 1, 2, 3, 63, 64, 65, 255, 256, 257, 511, 512, 513, 1_023, 1_024, 1_025, 4_095,
            4_096, 4_097,
        ] {
            for pattern in 0..3 {
                let input: Vec<u8> = (0..length)
                    .map(|position| match pattern {
                        0 => b'a',
                        1 => b'a' + u8::try_from(position % 3).unwrap_or_default(),
                        _ => u8::try_from((position * 73 + position / 7) % 251).unwrap_or_default(),
                    })
                    .collect();

                let suffixes = SuffixRanks::new(&input, false)?;
                assert_eq!(suffixes.sa.len(), length);
                assert_eq!(suffixes.rank.len(), length);
                for rank in 0..length {
                    let position = suffixes.position_at(rank);
                    assert!(position < length);
                    assert_eq!(
                        suffixes.rank_of(position),
                        rank,
                        "length={length} pattern={pattern} rank={rank}"
                    );
                }
                for rank in 1..length {
                    assert!(
                        input[suffixes.position_at(rank - 1)..]
                            < input[suffixes.position_at(rank)..],
                        "suffix order: length={length} pattern={pattern} rank={rank}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn traceback_recovers_distance_with_clipped_ties() {
        let input = [b'a'; 40];
        let entries = vec![
            MatchCandidate {
                length: 4,
                distance: 1,
            },
            MatchCandidate {
                length: 8,
                distance: 2,
            },
            MatchCandidate {
                length: 12,
                distance: 3,
            },
        ];
        let mut offsets = vec![0; input.len() + 1];
        offsets[21..].fill(entries.len());
        let frontier = MatchCandidates {
            first_query: 0,
            offsets,
            entries,
        };
        let steps = [0, 0, 0, 0, 4];
        let mut prices = [100; D_CODES];
        // All equal: clipping to four bytes leaves the first candidate.
        assert_eq!(
            reconstruct_tokens(&steps, &input, &frontier, 20..24, &prices),
            vec![ParseToken::matching(4, 1)]
        );
        prices[usize::from(State::d_code(1))] = 99;
        assert_eq!(
            reconstruct_tokens(&steps, &input, &frontier, 20..24, &prices),
            vec![ParseToken::matching(4, 2)]
        );
    }

    #[test]
    fn rational_seed_log2_matches_reference_costs() {
        assert_eq!(log2_ratio_x16(4, 3), 6);
        assert_eq!(log2_ratio_x16(8, 3), 22);
        assert_eq!(log2_ratio_x16(12, 3), 32);
        assert_eq!(log2_x16(LENGTH_CODES * 4), 109);
        assert_eq!(log2_x16(LENGTH_CODES * 2), 93);
        assert_eq!(log2_ratio_x16(LENGTH_CODES * 4, 3), 84);
    }

    #[test]
    fn previous_model_blend_uses_expected_weights() {
        assert_eq!(blend_cost(80, 160, 0), 100);
        assert_eq!(blend_cost(80, 160, 1), 120);
        assert_eq!(blend_cost(80, 160, 2), 130);
        assert_eq!(blend_cost(80, 160, 3), 140);
    }

    #[test]
    fn observation_classes_match_split_geometry() {
        let mut observations = BlockObservations::empty();
        observations.observe_literal(0b1010_0101);
        observations.observe_literal(0b1010_0100);
        observations.observe_match(8);
        observations.observe_match(9);
        assert_eq!(observations.counts[5], 1);
        assert_eq!(observations.counts[4], 1);
        assert_eq!(observations.counts[8], 1);
        assert_eq!(observations.counts[9], 1);
        assert_eq!(observations.total, 4);
    }

    #[test]
    fn minimum_match_heuristic_matches_reference_boundaries() {
        for (used_literals, expected) in [
            (0, 9),
            (5, 9),
            (6, 8),
            (8, 7),
            (10, 6),
            (16, 5),
            (45, 4),
            (80, 3),
            (256, 3),
        ] {
            assert_eq!(choose_min_match_len(used_literals), expected);
        }
    }

    #[test]
    fn exact_frontier_rejects_distances_outside_deflate_window() {
        // A u16 can encode more than the 32 KiB permitted by DEFLATE.
        // Reject that input at the candidate boundary, not when the token
        // eventually reaches the Huffman writer.
        assert!(matches!(
            build_match_candidates(b"aaaaaaa", (1usize << crate::MAX_WBITS) + 1),
            Err(ReturnCode::StreamError)
        ));
    }

    #[test]
    fn compact_match_candidate_selects_longest_then_nearest() {
        let mut best = MatchCandidate::NONE;
        best.consider(STD_MIN_MATCH - 1, 1);
        assert_eq!(best, MatchCandidate::NONE);

        best.consider(STD_MIN_MATCH, 2);
        assert_eq!(
            best,
            MatchCandidate {
                length: 3,
                distance: 2
            }
        );
        best.consider(STD_MIN_MATCH, 32_768);
        assert_eq!(best.distance, 2, "equal-length matches prefer the nearest");
        best.consider(STD_MIN_MATCH, 1);
        assert_eq!(best.distance, 1);
        best.consider(STD_MAX_MATCH, 32_768);
        assert_eq!(
            best,
            MatchCandidate {
                length: 258,
                distance: 32_768,
            }
        );
        best.consider(STD_MAX_MATCH - 1, 1);
        assert_eq!(
            best.distance, 32_768,
            "a shorter match must not replace a longer one"
        );
        assert_eq!(core::mem::size_of::<MatchCandidate>(), 4);
    }

    #[test]
    #[should_panic(expected = "match exceeds DEFLATE length limit")]
    fn compact_match_candidate_rejects_overlength() {
        let mut candidate = MatchCandidate::NONE;
        candidate.consider(STD_MAX_MATCH + 1, 1);
    }

    #[test]
    #[should_panic(expected = "match exceeds DEFLATE distance limit")]
    fn compact_match_candidate_rejects_out_of_window_distance() {
        let mut candidate = MatchCandidate::NONE;
        candidate.consider(STD_MIN_MATCH, (1usize << crate::MAX_WBITS) + 1);
    }

    #[test]
    fn exact_frontier_reaches_full_deflate_window() -> Result<(), ReturnCode> {
        let mut state = 0x94d0_49bb_1331_11ebu64;
        let mut input = vec![0u8; 32_768];
        for byte in &mut input {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *byte = state.to_le_bytes()[4];
        }
        let repeated = input[..STD_MAX_MATCH].to_vec();
        input.extend_from_slice(&repeated);

        let table = build_match_candidates(&input, 32_768)?;
        assert!(table.for_position(32_768).iter().any(|candidate| {
            usize::from(candidate.distance) == 32_768
                && usize::from(candidate.length) == STD_MAX_MATCH
        }));
        Ok(())
    }

    #[test]
    fn exact_frontier_does_not_cross_deflate_window() -> Result<(), ReturnCode> {
        let mut state = 0x94d0_49bb_1331_11ebu64;
        let mut input = vec![0u8; 32_769];
        for byte in &mut input {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *byte = state.to_le_bytes()[4];
        }
        let repeated = input[..STD_MAX_MATCH].to_vec();
        assert_ne!(&input[1..=STD_MAX_MATCH], repeated.as_slice());
        input.extend_from_slice(&repeated);

        let table = build_match_candidates(&input, 32_768)?;
        assert!(table
            .for_position(32_769)
            .iter()
            .all(|candidate| usize::from(candidate.length) < STD_MAX_MATCH));
        Ok(())
    }

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
                    length: compact_match_length(length),
                    distance: compact_match_distance(distance),
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

    fn brute_cost(input: &[u8], costs: &TokenCosts, max_distance: usize) -> u64 {
        const INF: u64 = u64::MAX / 4;
        let mut best = vec![INF; input.len() + 1];
        best[0] = 0;
        for position in 0..input.len() {
            best[position + 1] = best[position + 1]
                .min(best[position] + u64::from(costs.literal_cost(input[position])));
            let max_len = STD_MAX_MATCH.min(input.len() - position);
            for candidate in position.saturating_sub(max_distance)..position {
                let matched = common_prefix(
                    &input[position..position + max_len],
                    &input[candidate..candidate + max_len],
                );
                for length in STD_MIN_MATCH..=matched {
                    let distance = position - candidate;
                    best[position + length] = best[position + length]
                        .min(best[position] + u64::from(costs.match_cost(length, distance)));
                }
            }
        }
        best[input.len()]
    }

    fn exhaustive_parse_from_frontier(
        input: &[u8],
        candidates: &MatchCandidates,
        block: Range<usize>,
        costs: &TokenCosts,
    ) -> ParseResult {
        let mut paths = vec![(u64::MAX, ParseToken::Literal(0)); block.len() + 1];
        paths[0].0 = 0;
        for position in block.clone() {
            let local = position - block.start;
            let prefix = paths[local].0;
            let mut update = |len: usize, token: ParseToken| {
                let cost = prefix + u64::from(costs.token_cost(token));
                if cost < paths[local + len].0 {
                    paths[local + len] = (cost, token);
                }
            };
            update(1, ParseToken::Literal(input[position]));
            let available = (block.end - position).min(STD_MAX_MATCH);
            for len in STD_MIN_MATCH..=available {
                let selected = candidates
                    .for_position(position)
                    .iter()
                    .filter(|candidate| usize::from(candidate.length) >= len)
                    .min_by_key(|candidate| {
                        let code = usize::from(State::d_code(usize::from(candidate.distance) - 1));
                        (
                            costs.distance_symbol_cost(code),
                            core::cmp::Reverse(usize::from(candidate.length).min(available)),
                        )
                    });
                if let Some(candidate) = selected {
                    update(len, ParseToken::matching(len, candidate.distance));
                }
            }
        }
        let mut cursor = block.len();
        let mut tokens = core::iter::from_fn(|| {
            if cursor == 0 {
                return None;
            }
            let token = paths[cursor].1;
            cursor -= token.input_len();
            Some(token)
        })
        .collect::<Vec<_>>();
        tokens.reverse();
        ParseResult {
            cost_x16: paths[block.len()].0,
            tokens,
        }
    }

    #[test]
    fn grouped_relaxation_preserves_exact_costs_and_ties() -> Result<(), ReturnCode> {
        let inputs = [b"ab".repeat(180), b"xyz0123456789".repeat(28), {
            let mut bytes = b"abaca".repeat(80);
            for index in (0..bytes.len()).step_by(37) {
                bytes[index] = b'X';
            }
            bytes
        }];
        let mut models = vec![TokenCosts::fixed(), TokenCosts::unseen()];
        for seed in 0..5u16 {
            models.push(TokenCosts {
                literal: core::array::from_fn(|i| {
                    (u16::try_from(i).unwrap_or_default() + seed) % 17 + 1
                }),
                length: core::array::from_fn(|i| {
                    (u16::try_from(i).unwrap_or_default() * (seed + 1)) % 29 + 1
                }),
                distance: core::array::from_fn(|i| {
                    (u16::try_from(i).unwrap_or_default() * seed) % 23 + 1
                }),
            });
        }
        for input in &inputs {
            let candidates = build_match_candidates(input, 32768)?;
            let mut scratch = ParseScratch::default();
            for block in [0..input.len(), 17..input.len() - 19, 31..65] {
                for costs in &models {
                    assert_eq!(
                        parse_range_with_scratch(
                            input,
                            &candidates,
                            block.clone(),
                            costs,
                            &mut scratch
                        ),
                        exhaustive_parse_from_frontier(input, &candidates, block.clone(), costs),
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn sparse_length_occupancy_preserves_parse_across_word_boundaries() {
        let input = vec![b'a'; 3 * ParseScratch::COST_TILE + STD_MAX_MATCH];
        let lengths = [3, 31, 32, 63, 64, 65, 127, 128, 129, 255, 256, 257, 258];
        let mut offsets = vec![0];
        let mut entries = Vec::new();
        for position in 0..input.len() {
            // Interleave empty positions with matches, including duplicate
            // maximum lengths in different distance classes.
            if position >= 16 && position % 5 != 0 {
                let length = lengths[position % lengths.len()];
                for (distance, adjustment) in [(1, 0), (2, 0), (8, 1), (16, 2)] {
                    let length = (length + adjustment)
                        .min(STD_MAX_MATCH)
                        .min(input.len() - position);
                    if length >= STD_MIN_MATCH {
                        entries.push(MatchCandidate {
                            length: compact_match_length(length),
                            distance: compact_match_distance(distance),
                        });
                    }
                }
            }
            offsets.push(entries.len());
        }
        let candidates = MatchCandidates {
            first_query: 0,
            offsets,
            entries,
        };
        let mut tied = TokenCosts::fixed();
        // Equal total distance cost, including extra bits. Preserve the
        // existing first-candidate and longest-maximum tie ordering.
        for (code, cost) in tied.distance.iter_mut().enumerate() {
            *cost = (16 - u16::from(StaticTreeDesc::EXTRA_DBITS[code])) * COST_SCALE;
        }
        let mut nonmonotone = tied.clone();
        nonmonotone.distance[0] = 200;
        nonmonotone.distance[1] = 400;
        let expensive = TokenCosts {
            literal: [u16::MAX; 256],
            length: [u16::MAX; LENGTH_CODES],
            distance: [u16::MAX; D_CODES],
        };
        let mut scratch = ParseScratch::default();
        for costs in [&tied, &nonmonotone, &expensive] {
            let tile = ParseScratch::COST_TILE;
            for block in [
                0..input.len(),
                0..tile,
                0..tile + 1,
                17..17 + tile + STD_MAX_MATCH - 1,
                17..17 + tile + STD_MAX_MATCH,
                31..64,
                96..353,
                18..330,
            ] {
                let expected =
                    exhaustive_parse_from_frontier(&input, &candidates, block.clone(), costs);
                let actual = parse_range_with_scratch(
                    &input,
                    &candidates,
                    block.clone(),
                    costs,
                    &mut scratch,
                );
                assert_eq!(actual, expected, "range={block:?}");
                assert_eq!(
                    scratch.best_cost.len(),
                    (block.len() + 1).min(ParseScratch::COST_CAPACITY)
                );
                assert_eq!(scratch.steps.len(), block.len() + 1);
                if block.start == 0 {
                    assert_eq!(expand(&actual.tokens), input[block]);
                }
            }
        }
    }

    #[test]
    fn suffix_rank_frontier_matches_exhaustive_distance_code_frontier() -> Result<(), ReturnCode> {
        let inputs: &[&[u8]] = &[
            b"abcabcabc/abcabcabc/xyzxyzxyz",
            b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            b"012345670123456701234567--01234567",
            b"no useful repetition here",
        ];
        for input in inputs {
            let table = build_match_candidates(input, 32)?;
            assert_eq!(table.len(), input.len());
            for position in 0..input.len() {
                assert_eq!(
                    frontier_signature(table.for_position(position)),
                    frontier_signature(&brute_frontier(input, position, 32)),
                    "position {position} for {input:?}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn suffix_rank_frontier_matches_bruteforce_on_generated_inputs() -> Result<(), ReturnCode> {
        let mut state = 0x1234_5678_9abc_def0u64;
        for len in 3..96 {
            for alphabet in [2u8, 3, 7, 31, 251] {
                let mut input = vec![0u8; len];
                for byte in &mut input {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    *byte = state.to_le_bytes()[4] % alphabet;
                }
                for max_distance in [1usize, 2, 3, 7, 16, 31, 50, 64] {
                    let max_distance = max_distance.min(32_506);
                    let table = build_match_candidates(&input, max_distance)?;
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
        Ok(())
    }

    #[test]
    fn tiled_frontier_matches_exact_records_across_tile_boundaries() -> Result<(), ReturnCode> {
        // Exercise truncated direct-scan buckets and the tile edges that
        // shorter generated-input tests cannot reach. Compare full candidates
        // (including the selected distance), not just code/length.
        let mut state = 0x243f_6a88_85a3_08d3u64;
        for len in [
            127usize, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025,
        ] {
            for alphabet in [2u8, 31, 251] {
                let mut input = vec![0u8; len];
                for byte in &mut input {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    *byte = state.to_le_bytes()[4] % alphabet;
                }
                for max_distance in [1usize, 7, 31, 50, 64, 257] {
                    let table = build_match_candidates(&input, max_distance)?;
                    let reference = build_match_candidates_position_major(&input, max_distance)?;
                    assert_eq!(
                        table, reference,
                        "len={len} alphabet={alphabet} max_distance={max_distance}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn query_ranges_and_lazy_prefix_preserve_every_candidate() -> Result<(), ReturnCode> {
        // Dense repeated text plus a broad-alphabet context containing
        // small, word-sized and multi-tile exact prefix groups.
        let dense = b"abracadabra0123456789".repeat(513);
        let mut seed = 0x93c5_112a_cab8_4621u64;
        let mut sparse = Vec::new();
        for _ in 0..3_500 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            sparse.push(seed.to_le_bytes()[0] & 0x7f);
        }
        for (marker, count) in [(0xf0, 16usize), (0xf1, 64), (0xf2, 65), (0xf3, 513)] {
            for i in 0..count {
                sparse.extend_from_slice(&[0xff, 0xfe, marker, i.to_le_bytes()[0] & 0x7f]);
            }
        }
        let copy = sparse[..3500].to_vec();
        sparse.extend_from_slice(&copy);
        for input in [&dense[..], &sparse[..], &[], b"a", b"ab", b"abc"] {
            for distance in [0, 1, 31, 32, 33, 40, 41, 65, 129, 1025, 32768] {
                let source = FrontierSource::new(input, distance)?;
                let full = build_match_candidates_position_major(input, distance)?;
                let n = input.len();
                let points = [
                    0,
                    1.min(n),
                    31.min(n),
                    511.min(n),
                    512.min(n),
                    513.min(n),
                    n / 2,
                    n.saturating_sub(2),
                    n,
                ];
                for (i, &start) in points.iter().enumerate() {
                    let end = points[i..]
                        .iter()
                        .copied()
                        .find(|&p| p > start)
                        .unwrap_or(n);
                    let partial = source.build(start..end)?;
                    assert_eq!(partial.first_query, start);
                    assert_eq!(partial.end(), end);
                    for position in start..end {
                        assert_eq!(
                            partial.for_position(position),
                            full.for_position(position),
                            "distance={distance} query={start}..{end} position={position}"
                        );
                    }
                }
                // Reconstruct a complete CSR from arbitrarily aligned
                // adjacent query ranges, including empty prefixes.
                let first = n / 3;
                let second = n * 2 / 3;
                let mut suffix = source.build(second..n)?;
                suffix.prepend(source.build(first..second)?);
                suffix.prepend(source.build(0..first)?);
                assert_eq!(suffix, full, "distance={distance}, bytes={n}");
                assert_eq!(source.build(n..n)?.offsets, [0]);
                assert!(matches!(
                    source.build(n..n + 1),
                    Err(ReturnCode::StreamError)
                ));
                assert!(matches!(
                    source.build(core::ops::Range {
                        start: n + 1,
                        end: n
                    }),
                    Err(ReturnCode::StreamError)
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn group_local_frontier_matches_exact_records_across_group_sizes() -> Result<(), ReturnCode> {
        // Exercise all three local scanners, their transition boundaries,
        // and the final truncated direct class. Use a broad-alphabet
        // region to make the local strategy profitable regardless of the
        // denser groups, with separated runs crossing multiple tiles.
        let groups = [2usize, 15, 16, 17, 63, 64, 65, 511, 512, 513];
        let mut input = Vec::new();
        let mut seed = 0x93c5_112a_cab8_4621u64;
        for _ in 0..8_192 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            input.push(seed.to_le_bytes()[0] & 0x7f);
        }
        let atom = input.clone();
        for (marker, &count) in (0u8..).zip(&groups) {
            for index in 0..count {
                input.extend_from_slice(&[0xf5, 0xe4, marker, index.to_le_bytes()[0] & 0x7f]);
                if index == count / 2 {
                    input.extend_from_slice(&atom[..257]);
                }
            }
        }
        input.extend_from_slice(&atom);
        let suffixes = SuffixRanks::new(&input, true)?;
        let Some(prefix_groups) = &suffixes.prefix_groups else {
            return Err(ReturnCode::StreamError);
        };
        for (marker, &count) in (0u8..).zip(&groups) {
            let prefix = [0xf5, 0xe4, marker];
            let actual = suffixes
                .sa
                .iter()
                .filter(|&&pos| {
                    input[suffix_index(pos)..].get(..STD_MIN_MATCH) == Some(prefix.as_slice())
                })
                .count();
            assert_eq!(actual, count, "group marker={marker}");
        }
        for distance in [
            0usize, 31, 32, 33, 39, 40, 41, 63, 64, 65, 127, 129, 257, 1025, 32768,
        ] {
            let ranges = distance_code_ranges(distance);
            let actual = group_local_frontier(
                &input,
                &suffixes,
                prefix_groups,
                &ranges,
                distance,
                0..input.len(),
            );
            assert_eq!(
                actual,
                build_match_candidates_position_major(&input, distance)?,
                "distance={distance}"
            );
        }
        Ok(())
    }

    pub(super) fn two_distribution_input() -> Vec<u8> {
        let mut state = 0x8f6e_37a1_4c92_d5b7u64;
        let mut input = Vec::with_capacity(24_000);
        for _ in 0..12_000 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            input.push(b'a' + (state.to_le_bytes()[4] % 26));
        }
        for _ in 0..12_000 {
            state = state
                .wrapping_mul(2_862_933_555_777_941_757)
                .wrapping_add(3_037_000_493);
            input.push(0x80 | (state.to_le_bytes()[4] & 0x7f));
        }
        input
    }

    #[test]
    fn splitter_rejects_out_of_window_history_before_allocating() {
        for input in [b"".as_slice(), b"abcabc".as_slice()] {
            assert!(matches!(
                choose_block_ranges(input, usize::MAX),
                Err(ReturnCode::StreamError)
            ));
            assert!(matches!(
                choose_block_ranges(input, (1usize << crate::MAX_WBITS) + 1),
                Err(ReturnCode::StreamError)
            ));
        }
    }

    #[test]
    fn splitter_detects_distribution_change_and_preserves_block_invariants(
    ) -> Result<(), ReturnCode> {
        let input = two_distribution_input();
        let blocks = choose_block_ranges(&input, 32_506)?;
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
        assert_eq!(blocks.last().map(|block| block.end), Some(input.len()));
        assert!(blocks.windows(2).all(|pair| pair[0].end == pair[1].start));
        assert!(blocks
            .iter()
            .all(|block| block.len() <= MAX_COARSE_BLOCK_INPUT));
        Ok(())
    }

    #[test]
    fn splitter_soft_limit_never_creates_an_illegal_or_tiny_tail() -> Result<(), ReturnCode> {
        let input = vec![0x5a; 150_000];
        let blocks = choose_block_ranges(&input, 32_506)?;
        assert_eq!(blocks[0].start, 0);
        assert_eq!(blocks.last().map(|block| block.end), Some(input.len()));
        assert!(blocks
            .iter()
            .all(|block| block.len() <= MAX_COARSE_BLOCK_INPUT));
        let (tail, prefix) = blocks.split_last().ok_or(ReturnCode::StreamError)?;
        assert!(prefix.iter().all(|block| block.len() >= MIN_BLOCK_LENGTH));
        assert!(tail.len() >= MIN_BLOCK_LENGTH);
        Ok(())
    }

    #[test]
    fn splitter_soft_limit_boundary_matrix_is_exact() -> Result<(), ReturnCode> {
        // Keep both endpoints explicit: the fixture checks contiguous ranges,
        // including the zero-length stream, not just the number of splits.
        for (len, expected) in [
            (0, &[(0, 0)][..]),
            (59_999, &[(0, 59_999)][..]),
            (60_000, &[(0, 60_000)][..]),
            (60_001, &[(0, 60_001)][..]),
            (64_999, &[(0, 64_999)][..]),
            (65_000, &[(0, 60_000), (60_000, 65_000)][..]),
            (65_001, &[(0, 60_000), (60_000, 65_001)][..]),
            (119_999, &[(0, 60_000), (60_000, 119_999)][..]),
            (120_000, &[(0, 60_000), (60_000, 120_000)][..]),
        ] {
            let input = vec![0x5a; len];
            let ranges = choose_block_ranges(&input, 32_768)?;
            let boundaries = ranges
                .iter()
                .map(|range| (range.start, range.end))
                .collect::<Vec<_>>();
            assert_eq!(boundaries, expected, "len={len}");
        }
        Ok(())
    }

    #[test]
    fn range_parse_can_reference_history_before_the_block() -> Result<(), ReturnCode> {
        let mut state = 0x517c_c1b7_2722_0a95u64;
        let mut history = vec![0u8; 6_000];
        for byte in &mut history {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *byte = state.to_le_bytes()[3];
        }
        let mut input = history.clone();
        input.extend_from_slice(&history);
        let table = build_match_candidates(&input, 32_506)?;
        let parsed = parse_range(&input, &table, 6_000..12_000, &TokenCosts::fixed());
        assert!(matches!(
            parsed.tokens.first(),
            Some(ParseToken::Match { length, distance })
                if *length >= 32 && *distance == 6_000
        ));
        Ok(())
    }

    #[test]
    fn fixed_point_log2_matches_seed_cost_landmarks() {
        assert_eq!(log2_x16(1), 0);
        assert_eq!(log2_x16(2), 16);
        assert_eq!(log2_x16(3), 25);
        assert_eq!(log2_x16(30), 78);
        assert_eq!(log2_x16(58), 93);
        assert_eq!(log2_x16(116), 109);
        assert_eq!(log2_x16(256), 128);
    }

    #[test]
    fn minimum_cost_parse_matches_exhaustive_oracle() -> Result<(), ReturnCode> {
        let input = b"abcabcabcxabcaabcabcabc---abcabc";
        let table = build_match_candidates(input, 32)?;
        let fixed = TokenCosts::fixed();
        let parsed = parse_block(input, &table, &fixed);
        assert_eq!(parsed.cost_x16, brute_cost(input, &fixed, 32));
        assert_eq!(expand(&parsed.tokens), input);
        Ok(())
    }

    #[test]
    fn maximal_symbol_weights_keep_dp_exact_without_u16_overflow() -> Result<(), ReturnCode> {
        // A cost model may assign u16::MAX to a Huffman symbol. Extra-bit
        // costs are then added before DP comparison and must not wrap u16,
        // including for distance symbols spanning the full DEFLATE window.
        let input = vec![b'a'; 96];
        let candidates = build_match_candidates(&input, 64)?;
        let mut costs = TokenCosts::fixed();
        costs.literal.fill(u16::MAX);
        costs.length.fill(u16::MAX);
        costs.distance.fill(u16::MAX);

        let parsed = parse_block(&input, &candidates, &costs);
        assert_eq!(parsed.cost_x16, brute_cost(&input, &costs, 64));
        assert_eq!(expand(&parsed.tokens), input);
        assert!(parsed
            .tokens
            .iter()
            .any(|token| matches!(token, ParseToken::Match { .. })));

        let length_code = usize::from(trees_tbl::LENGTH_CODE[STD_MAX_MATCH - STD_MIN_MATCH]);
        let expected_length = u32::from(costs.length[length_code])
            + u32::from(StaticTreeDesc::EXTRA_LBITS[length_code]) * u32::from(COST_SCALE);
        for distance in 1..=1usize << crate::MAX_WBITS {
            let distance_code = usize::from(State::d_code(distance - 1));
            let expected_distance = u32::from(costs.distance[distance_code])
                + u32::from(StaticTreeDesc::EXTRA_DBITS[distance_code]) * u32::from(COST_SCALE);
            assert_eq!(
                costs.match_cost(STD_MAX_MATCH, distance),
                expected_length + expected_distance,
                "distance={distance}",
            );
        }
        Ok(())
    }

    #[test]
    fn reused_parse_scratch_resets_across_ranges_and_cost_models() -> Result<(), ReturnCode> {
        let input = b"abcabcabc---abcabcabc---the quick brown fox---abcabcabcabcabc";
        let candidates = build_match_candidates(input, 32)?;
        let fixed = TokenCosts::fixed();
        let mut varied = fixed.clone();
        varied.literal[b'a' as usize] = 2 * COST_SCALE;
        varied.distance.fill(5 * COST_SCALE);

        let mut scratch = ParseScratch::default();
        let mut capacity = None;
        for (range, costs) in [
            (0..input.len(), &fixed),
            (12..36, &varied),
            (24..24, &fixed),
            (4..input.len(), &varied),
            (0..input.len(), &fixed),
        ] {
            let expected = parse_range(input, &candidates, range.clone(), costs);
            let actual =
                parse_range_with_scratch(input, &candidates, range.clone(), costs, &mut scratch);
            assert_eq!(actual, expected, "range={range:?}");
            assert_eq!(
                scratch.best_cost.len(),
                (range.len() + 1).min(ParseScratch::COST_CAPACITY)
            );
            assert_eq!(scratch.steps.len(), range.len() + 1);
            if let Some((cost_capacity, step_capacity)) = capacity {
                assert_eq!(scratch.best_cost.capacity(), cost_capacity);
                assert_eq!(scratch.steps.capacity(), step_capacity);
            } else {
                capacity = Some((scratch.best_cost.capacity(), scratch.steps.capacity()));
            }
        }
        Ok(())
    }

    #[test]
    fn distance_code_reduction_is_valid_for_non_monotone_huffman_costs() -> Result<(), ReturnCode> {
        let input = b"aaaaaaaaabaaaaaaaaacaaaaaaaaadaaaaaaaaa";
        let table = build_match_candidates(input, 32)?;
        let mut costs = TokenCosts::fixed();
        // Deliberately make farther distance symbols cheaper than nearby ones. Keeping the longest
        // match per distance *symbol* must remain exact even when distance costs are non-monotone.
        for (code, cost) in costs.distance.iter_mut().enumerate() {
            let remainder = u16::try_from(code % 15).unwrap_or_else(|_| unreachable!());
            *cost = (15 - remainder) * COST_SCALE;
        }
        for (byte, cost) in costs.literal.iter_mut().enumerate() {
            let remainder = u16::try_from(byte % 11).unwrap_or_else(|_| unreachable!());
            *cost = (3 + remainder) * COST_SCALE;
        }

        let parsed = parse_block(input, &table, &costs);
        assert_eq!(parsed.cost_x16, brute_cost(input, &costs, 32));
        assert_eq!(expand(&parsed.tokens), input);
        Ok(())
    }
}

/// Private near-optimal states use all 65,536 symbol slots supported by the existing tree
/// frequency representation. `SymBuf` itself reserves one slot, and the literal/length Huffman tree
/// needs one additional frequency count for EOB. Therefore at most 65,534 LZ77 tokens may enter one
/// tree without overflowing a 16-bit internal frequency sum. Input byte length is independent: a
/// match token can represent up to 258 bytes, so compressible blocks may legally span well beyond
/// the 65,535-byte stored-block cap.
pub const NEAR_OPTIMAL_LIT_BUFSIZE: usize = 1 << 16;
pub const MAX_BLOCK_TOKENS: usize = NEAR_OPTIMAL_LIT_BUFSIZE - 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BlockScore {
    bits: usize,
    encoded_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TokenHistogram {
    literal_length: [u16; L_CODES],
    distance: [u16; D_CODES],
    // A frequency bound for an individual symbol is not enough: the sum of
    // all LZ77 tokens must also fit SymBuf, with EOB kept as a separate symbol.
    total_tokens: usize,
}

impl TokenHistogram {
    const fn empty() -> Self {
        let mut literal_length = [0; L_CODES];
        literal_length[256] = 1;
        Self {
            literal_length,
            distance: [0; D_CODES],
            total_tokens: 0,
        }
    }

    fn from_tokens(tokens: &[ParseToken]) -> Result<Self, ReturnCode> {
        if tokens.len() > MAX_BLOCK_TOKENS {
            return Err(ReturnCode::StreamError);
        }
        let mut histogram = Self::empty();
        for &token in tokens {
            histogram.add_token(token)?;
        }
        Ok(histogram)
    }

    // Reject malformed compact match tokens before computing table indices.
    // A literal can use any u8, but a match must satisfy both DEFLATE limits.
    fn token_bins(token: ParseToken) -> Result<(usize, Option<usize>), ReturnCode> {
        match token {
            ParseToken::Literal(byte) => Ok((usize::from(byte), None)),
            ParseToken::Match { length, distance } => {
                if !(STD_MIN_MATCH..=STD_MAX_MATCH).contains(&usize::from(length))
                    || !(1..=1usize << crate::MAX_WBITS).contains(&usize::from(distance))
                {
                    return Err(ReturnCode::StreamError);
                }
                let length_code =
                    usize::from(trees_tbl::LENGTH_CODE[usize::from(length) - STD_MIN_MATCH]);
                let distance_code = usize::from(State::d_code(usize::from(distance) - 1));
                Ok((257 + length_code, Some(distance_code)))
            }
        }
    }

    fn add_token(&mut self, token: ParseToken) -> Result<(), ReturnCode> {
        if self.total_tokens >= MAX_BLOCK_TOKENS {
            return Err(ReturnCode::StreamError);
        }
        let (literal_length, distance) = Self::token_bins(token)?;
        // Compute both updates before committing either. A failure cannot
        // leave a partially updated histogram that may be scored afterward.
        let new_literal = self.literal_length[literal_length]
            .checked_add(1)
            .ok_or(ReturnCode::StreamError)?;
        let new_distance = distance
            .map(|index| {
                self.distance[index]
                    .checked_add(1)
                    .ok_or(ReturnCode::StreamError)
            })
            .transpose()?;
        self.literal_length[literal_length] = new_literal;
        if let (Some(index), Some(frequency)) = (distance, new_distance) {
            self.distance[index] = frequency;
        }
        self.total_tokens += 1;
        Ok(())
    }

    fn remove_token(&mut self, token: ParseToken) -> Result<(), ReturnCode> {
        let (literal_length, distance) = Self::token_bins(token)?;
        let new_total = self
            .total_tokens
            .checked_sub(1)
            .ok_or(ReturnCode::StreamError)?;
        let new_literal = self.literal_length[literal_length]
            .checked_sub(1)
            .ok_or(ReturnCode::StreamError)?;
        let new_distance = distance
            .map(|index| {
                self.distance[index]
                    .checked_sub(1)
                    .ok_or(ReturnCode::StreamError)
            })
            .transpose()?;
        self.literal_length[literal_length] = new_literal;
        if let (Some(index), Some(frequency)) = (distance, new_distance) {
            self.distance[index] = frequency;
        }
        self.total_tokens = new_total;
        Ok(())
    }

    fn merged(left: &Self, right: &Self) -> Result<Self, ReturnCode> {
        let total_tokens = left
            .total_tokens
            .checked_add(right.total_tokens)
            .filter(|&total| total <= MAX_BLOCK_TOKENS)
            .ok_or(ReturnCode::StreamError)?;
        let mut merged = left.clone();
        for index in 0..L_CODES {
            if index == 256 {
                continue;
            }
            merged.literal_length[index] = merged.literal_length[index]
                .checked_add(right.literal_length[index])
                .ok_or(ReturnCode::StreamError)?;
        }
        for index in 0..D_CODES {
            merged.distance[index] = merged.distance[index]
                .checked_add(right.distance[index])
                .ok_or(ReturnCode::StreamError)?;
        }
        merged.total_tokens = total_tokens;
        Ok(merged)
    }

    fn apply_to_state(&self, state: &mut super::State<'_>) {
        state.init_block();
        for (value, &frequency) in state.l_desc.dyn_tree[..L_CODES]
            .iter_mut()
            .zip(&self.literal_length)
        {
            *value.freq_mut() = frequency;
        }
        for (value, &frequency) in state.d_desc.dyn_tree[..D_CODES]
            .iter_mut()
            .zip(&self.distance)
        {
            *value.freq_mut() = frequency;
        }
    }
}

#[derive(Clone, Debug)]
pub struct PlannedBlock {
    pub(crate) range: Range<usize>,
    pub(crate) tokens: Vec<ParseToken>,
}

impl PlannedBlock {
    fn input_len(&self) -> usize {
        self.range.len()
    }

    fn token_input_len(&self) -> usize {
        self.tokens.iter().map(|token| token.input_len()).sum()
    }
}

pub struct PlannedBlockSource<'a> {
    pub(crate) block: PlannedBlock,
    model: ParseModel,
    context_start: usize,
    context: &'a [u8],
    source: Option<FrontierSource<'a>>,
    candidates: MatchCandidates,
}

/// The last optimization iteration is either the selected best path or an
/// independently owned worse path. Refer to the best path when the two coincide
/// instead of duplicating its potentially large token vector just to detect
/// convergence on the next pass.
#[derive(Default)]
enum PreviousParse {
    #[default]
    None,
    Best,
    Other(Vec<ParseToken>),
}

impl PreviousParse {
    fn converged(&self, best: Option<&[ParseToken]>, next: &[ParseToken]) -> bool {
        match self {
            Self::None => false,
            Self::Best => best.is_some_and(|best| best == next),
            Self::Other(previous) => previous == next,
        }
    }
}

impl PlannedBlockSource<'_> {
    pub(crate) fn into_parts(self) -> (PlannedBlock, ParseModel) {
        (self.block, self.model)
    }

    pub(crate) fn refresh_model(&mut self, state: &mut super::State<'_>) -> Result<(), ReturnCode> {
        self.model = model_from_tokens(state, &self.block.tokens)?;
        Ok(())
    }

    /// Re-run the parser after a boundary change. A rightward shift needs
    /// no match discovery. For a leftward shift inside the original
    /// context, extend only the missing query prefix using its retained
    /// suffix order. Do not reinterpret a different history context.
    pub(crate) fn reparse_with_existing_candidates(
        &mut self,
        state: &mut super::State<'_>,
        optimization_passes: usize,
        previous_model: Option<&ParseModel>,
    ) -> Result<bool, ReturnCode> {
        if self.block.range.start < self.context_start
            || self.block.range.end != self.context_start + self.context.len()
        {
            self.refresh_model(state)?;
            return Ok(false);
        }

        if optimization_passes == 0 {
            self.refresh_model(state)?;
            return Ok(false);
        }

        let context = self.context;
        let local_range = (self.block.range.start - self.context_start)..context.len();
        if local_range.start < self.candidates.first_query {
            let Some(source) = &self.source else {
                return Err(ReturnCode::StreamError);
            };
            let prefix = source.build(local_range.start..self.candidates.first_query)?;
            self.candidates.prepend(prefix);
        }
        // The original block is one of the possible winners. Learn its costs
        // now so the final model can be retained without rebuilding the same
        // Huffman trees after the optimization loop.
        let (mut best_costs, mut best_score) = costs_and_block_score(state, &self.block.tokens)?;
        let (default_costs, observations) =
            initial_statistical_costs(context, &self.candidates, local_range.clone());
        let mut costs = previous_model
            .map(|previous| blend_previous_model(&default_costs, observations, previous))
            .unwrap_or(default_costs);
        let mut previous = PreviousParse::None;
        let mut best_tokens: Option<Vec<ParseToken>> = None;
        let mut scratch = ParseScratch::default();

        for _ in 0..optimization_passes {
            let reparsed = parse_range_with_scratch(
                context,
                &self.candidates,
                local_range.clone(),
                &costs,
                &mut scratch,
            );
            if reparsed.tokens.len() > MAX_BLOCK_TOKENS {
                break;
            }
            // A repeated token sequence has the same histogram, exact
            // Huffman score and learned model as the preceding iteration.
            // That preceding result was already evaluated, even if worse
            // than the best parse, so there is no reason to build its trees.
            if previous.converged(best_tokens.as_deref(), &reparsed.tokens) {
                break;
            }
            let (next_costs, reparsed_score) = costs_and_block_score(state, &reparsed.tokens)?;
            if reparsed_score.bits < best_score.bits {
                best_score = reparsed_score;
                best_costs = next_costs.clone();
                best_tokens = Some(reparsed.tokens);
                previous = PreviousParse::Best;
            } else {
                previous = PreviousParse::Other(reparsed.tokens);
            }
            costs = next_costs;
        }

        let changed = if let Some(tokens) = best_tokens {
            self.block.tokens = tokens;
            true
        } else {
            false
        };
        // The exact-score winner's cost model was captured with its tree;
        // only its observations depend on which token vector was selected.
        self.model = ParseModel {
            costs: best_costs,
            observations: BlockObservations::from_tokens(&self.block.tokens),
        };
        debug_assert_eq!(self.block.token_input_len(), self.block.range.len());
        Ok(changed)
    }
}

#[inline]
fn log2_x16(value: usize) -> u16 {
    debug_assert!(value > 0);
    let integer = usize::BITS - 1 - value.leading_zeros();
    debug_assert!(
        integer <= 31,
        "fixed-point seed log2 only handles small alphabets"
    );
    let mut normalized = (value as u64) << (31 - integer);
    let mut fraction = 0u16;
    for bit in (0..4).rev() {
        normalized = (normalized * normalized) >> 31;
        if normalized >= (2u64 << 31) {
            normalized >>= 1;
            fraction |= 1 << bit;
        }
    }
    u16::try_from(integer).unwrap_or_else(|_| unreachable!("log2 exponent is bounded to 31"))
        * COST_SCALE
        + fraction
}

#[inline]
fn log2_ratio_x16(numerator: usize, denominator: usize) -> u16 {
    const Q_BITS: u32 = 48;

    debug_assert!(numerator >= denominator && denominator != 0);
    // Represent the rational with 48 fractional bits before extracting four binary log2 fraction
    // bits. The inputs here are tiny DEFLATE alphabet sizes; Q48 makes the approximation error far
    // below a 1/16-bit boundary while preserving exact powers of two.
    let q = ((numerator as u128) << Q_BITS) / denominator as u128;
    let integer = u128::BITS - 1 - q.leading_zeros();
    debug_assert!((Q_BITS..=63).contains(&integer));
    let mut normalized = q << (63 - integer);
    let mut fraction = 0u16;
    for bit in (0..4).rev() {
        normalized = (normalized * normalized) >> 63;
        if normalized >= (2u128 << 63) {
            normalized >>= 1;
            fraction |= 1 << bit;
        }
    }
    u16::try_from(integer - Q_BITS).unwrap_or_else(|_| unreachable!("bounded Q48 log2 exponent"))
        * COST_SCALE
        + fraction
}

fn initial_statistical_costs(
    input: &[u8],
    candidates: &MatchCandidates,
    block: Range<usize>,
) -> (TokenCosts, BlockObservations) {
    let mut literal_frequencies = [0u32; 256];
    for &byte in &input[block.clone()] {
        literal_frequencies[usize::from(byte)] += 1;
    }
    let cutoff = block.len() >> 11;
    let used_literals = literal_frequencies
        .iter()
        .filter(|&&frequency| frequency as usize > cutoff)
        .count()
        .max(1);

    // Mirror the near-optimal seed heuristic used by libdeflate: estimate the eventual literal/match
    // mixture from a cheap greedy observation pass, but do not constrain the actual DP to this
    // minimum match length. Our exact frontier is already resident, so this adds no match search.
    let min_match = choose_min_match_len(used_literals);
    let mut literal_count = block.len();
    let mut match_count = 0usize;
    let mut observations = BlockObservations::empty();
    let mut position = block.start;
    while position < block.end {
        let available = block.end - position;
        let best_length = candidates
            .for_position(position)
            .iter()
            .map(|candidate| usize::from(candidate.length).min(available))
            .max()
            .unwrap_or(0);
        if best_length >= min_match {
            match_count += 1;
            literal_count = literal_count.saturating_sub(best_length);
            observations.observe_match(best_length);
            position += best_length;
        } else {
            observations.observe_literal(input[position]);
            position += 1;
        }
    }

    let (literal_cost, length_cost) = if match_count > literal_count {
        // Approximate match probability 3/4: literals share the remaining 1/4, while length symbols
        // share the 3/4 match probability. Subtracting log2(3) gives the rational denominator in
        // the same 1/16-bit fixed-point domain.
        (
            log2_x16(used_literals * 4),
            log2_ratio_x16(LENGTH_CODES * 4, 3),
        )
    } else if match_count.saturating_mul(4) > literal_count {
        // Approximate match probability 1/2.
        (log2_x16(used_literals * 2), log2_x16(LENGTH_CODES * 2))
    } else {
        // Approximate match probability 1/4: literals share 3/4 of the mass.
        (
            log2_ratio_x16(used_literals * 4, 3),
            log2_x16(LENGTH_CODES * 4),
        )
    };

    (
        TokenCosts {
            literal: [literal_cost; 256],
            length: [length_cost; LENGTH_CODES],
            // Assume distance symbols are initially equiprobable; exact tree lengths replace this
            // after the first optimization pass. Extra distance bits are added by `match_cost`.
            distance: [log2_x16(D_CODES); D_CODES],
        },
        observations,
    )
}

#[inline]
fn blend_cost(previous: u16, default: u16, change_amount: u8) -> u16 {
    let previous = u32::from(previous);
    let default = u32::from(default);
    let blended = match change_amount {
        0 => (default + 3 * previous) / 4,
        1 => u32::midpoint(default, previous),
        2 => (5 * default + 3 * previous) / 8,
        _ => (3 * default + previous) / 4,
    };
    u16::try_from(blended)
        .unwrap_or_else(|_| unreachable!("a convex combination of u16 costs fits u16"))
}

fn blend_previous_model(
    default: &TokenCosts,
    current: BlockObservations,
    previous: &ParseModel,
) -> TokenCosts {
    if current.total == 0 || previous.observations.total == 0 {
        return default.clone();
    }

    let mut total_delta = 0u64;
    for index in 0..SPLIT_OBSERVATION_TYPES {
        let previous_scaled =
            u64::from(previous.observations.counts[index]) * u64::from(current.total);
        let current_scaled =
            u64::from(current.counts[index]) * u64::from(previous.observations.total);
        total_delta += previous_scaled.abs_diff(current_scaled);
    }
    let cutoff = u64::from(previous.observations.total) * u64::from(current.total) * 200 / 512;
    if total_delta > 3 * cutoff {
        return default.clone();
    }
    let change_amount = if 4 * total_delta > 9 * cutoff {
        3
    } else if 2 * total_delta > 3 * cutoff {
        2
    } else {
        u8::from(2 * total_delta > cutoff)
    };

    let mut blended = default.clone();
    for index in 0..blended.literal.len() {
        blended.literal[index] = blend_cost(
            previous.costs.literal[index],
            default.literal[index],
            change_amount,
        );
    }
    for index in 0..blended.length.len() {
        blended.length[index] = blend_cost(
            previous.costs.length[index],
            default.length[index],
            change_amount,
        );
    }
    for index in 0..blended.distance.len() {
        blended.distance[index] = blend_cost(
            previous.costs.distance[index],
            default.distance[index],
            change_amount,
        );
    }
    blended
}

/// Reparse one selected block using code lengths learned from the previous parse.
///
/// The first search uses a fractional statistical cost model derived from the block's literal
/// diversity and a cheap greedy match-frequency estimate. Later searches use actual code lengths
/// learned from the previous path, while symbols absent from that path retain a high finite prior so
/// they can still be introduced when worthwhile. Every path is scored with the production Huffman
/// tree builder, including dynamic tree-description overhead, and the best exact score is retained.
/// Stop early only when the token sequence itself converges: a temporarily worse iteration can still
/// lead to a better next model, while an identical token sequence is a true fixed point.
pub fn optimized_parse_range(
    state: &mut super::State<'_>,
    input: &[u8],
    candidates: &MatchCandidates,
    block: Range<usize>,
    optimization_passes: usize,
    previous_model: Option<&ParseModel>,
) -> Result<(ParseResult, ParseModel), ReturnCode> {
    if optimization_passes == 0 {
        let parsed = parse_range(input, candidates, block, &TokenCosts::fixed());
        let (costs, _) = costs_and_block_score(state, &parsed.tokens)?;
        let model = ParseModel {
            costs,
            observations: BlockObservations::from_tokens(&parsed.tokens),
        };
        return Ok((parsed, model));
    }

    let (default_costs, initial_observations) =
        initial_statistical_costs(input, candidates, block.clone());
    let mut costs = previous_model
        .map(|previous| blend_previous_model(&default_costs, initial_observations, previous))
        .unwrap_or(default_costs);
    let mut previous = PreviousParse::None;
    let mut best: Option<ParseResult> = None;
    let mut best_costs = None;
    let mut best_score = usize::MAX;
    let mut scratch = ParseScratch::default();

    for _ in 0..optimization_passes {
        let next = parse_range_with_scratch(input, candidates, block.clone(), &costs, &mut scratch);
        // Identical paths have identical learned trees and exact scores.
        // Their prior evaluation has already recorded a possible winner.
        if previous.converged(best.as_ref().map(|p| p.tokens.as_slice()), &next.tokens) {
            break;
        }
        let (next_costs, next_score) = costs_and_block_score(state, &next.tokens)?;
        if next_score.bits < best_score {
            best_score = next_score.bits;
            best_costs = Some(next_costs.clone());
            best = Some(next);
            previous = PreviousParse::Best;
        } else {
            previous = PreviousParse::Other(next.tokens);
        }
        costs = next_costs;
    }

    let best = best.ok_or(ReturnCode::StreamError)?;
    let model = ParseModel {
        costs: best_costs.ok_or(ReturnCode::StreamError)?,
        observations: BlockObservations::from_tokens(&best.tokens),
    };
    Ok((best, model))
}

pub fn model_from_tokens(
    state: &mut super::State<'_>,
    tokens: &[ParseToken],
) -> Result<ParseModel, ReturnCode> {
    let (costs, _) = costs_and_block_score(state, tokens)?;
    Ok(ParseModel {
        costs,
        observations: BlockObservations::from_tokens(tokens),
    })
}

pub fn tally_tokens(state: &mut super::State<'_>, tokens: &[ParseToken]) -> Result<(), ReturnCode> {
    if tokens.len() > MAX_BLOCK_TOKENS
        || tokens.len() > state.sym_buf.block_capacity()
        || !state.sym_buf.is_empty()
    {
        return Err(ReturnCode::StreamError);
    }
    // Preflight the *entire* input before changing the compressor state. In
    // particular, invalid compact match lengths/distances must not reach
    // tally_dist_full_window's assertions after preceding tokens were tallied.
    let configured_window = state.w_mask() + 1;
    for &token in tokens {
        let _ = TokenHistogram::token_bins(token)?;
        if let ParseToken::Match { distance, .. } = token {
            if usize::from(distance) > configured_window {
                return Err(ReturnCode::StreamError);
            }
        }
    }
    for &token in tokens {
        match token {
            ParseToken::Literal(byte) => {
                let _ = state.tally_lit(byte);
            }
            ParseToken::Match { length, distance } => {
                let _ = state.tally_dist_full_window(
                    usize::from(distance),
                    usize::from(length) - STD_MIN_MATCH,
                );
            }
        }
    }
    Ok(())
}

fn costs_and_block_score(
    state: &mut super::State<'_>,
    tokens: &[ParseToken],
) -> Result<(TokenCosts, BlockScore), ReturnCode> {
    let histogram = TokenHistogram::from_tokens(tokens)?;
    let mut costs = TokenCosts::unseen();
    let score = score_histogram(state, &histogram, Some(&mut costs));
    Ok((costs, score))
}

/// Build the same production Huffman trees for scoring and cost-model learning.
/// Most boundary alternatives need only the bit count, not all symbol costs.
fn score_histogram(
    state: &mut super::State<'_>,
    histogram: &TokenHistogram,
    learned_costs: Option<&mut TokenCosts>,
) -> BlockScore {
    histogram.apply_to_state(state);
    super::build_tree(&mut state.l_desc, &mut state.opt_len, &mut state.static_len);
    super::build_tree(&mut state.d_desc, &mut state.opt_len, &mut state.static_len);

    if let Some(costs) = learned_costs {
        // The original histogram is required: build_tree replaces frequency
        // storage with Huffman codes. Unseen symbols retain finite priors.
        for (index, cost) in costs.literal.iter_mut().enumerate() {
            if histogram.literal_length[index] != 0 {
                *cost = state.l_desc.dyn_tree[index].len() * COST_SCALE;
            }
        }
        for (index, cost) in costs.length.iter_mut().enumerate() {
            if histogram.literal_length[257 + index] != 0 {
                *cost = state.l_desc.dyn_tree[257 + index].len() * COST_SCALE;
            }
        }
        for (index, cost) in costs.distance.iter_mut().enumerate() {
            if histogram.distance[index] != 0 {
                *cost = state.d_desc.dyn_tree[index].len() * COST_SCALE;
            }
        }
    }

    // Exact production tree chooser: prefer static on rounded encoded-byte ties.
    let _ = super::build_bl_tree(state);
    let dynamic_bytes = (state.opt_len + 3 + 7) >> 3;
    let static_bytes = (state.static_len + 3 + 7) >> 3;
    let score = if static_bytes <= dynamic_bytes {
        BlockScore {
            bits: state.static_len + 3,
            encoded_bytes: static_bytes,
        }
    } else {
        BlockScore {
            bits: state.opt_len + 3,
            encoded_bytes: dynamic_bytes,
        }
    };
    state.init_block();
    score
}

fn compressed_histogram_score(
    state: &mut super::State<'_>,
    histogram: &TokenHistogram,
    input_len: usize,
    learned_costs: Option<&mut TokenCosts>,
) -> Option<BlockScore> {
    let score = score_histogram(state, histogram, learned_costs);
    // Stored blocks byte-align before their payload, so their exact cost depends on the preceding
    // block's residual bit count. Boundary optimization is intentionally limited to pairs for which
    // production would choose a Huffman-coded representation; stored-block regions keep the cheap
    // splitter's boundary unchanged.
    (input_len > super::MAX_STORED || input_len + 4 > score.encoded_bytes).then_some(score)
}

const EXACT_BOUNDARY_CANDIDATES: usize = 4;
const BOUNDARY_CANDIDATE_SEPARATION: usize = 256;

/// Shift a block boundary by an unsigned byte count. The direction is part
/// of the type, so neither the input length nor the absolute input position
/// ever needs conversion through isize.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BoundaryShift {
    Right(usize),
    Left(usize),
}

impl BoundaryShift {
    const fn is_near(self, other: Self) -> bool {
        match (self, other) {
            (Self::Right(a), Self::Right(b)) | (Self::Left(a), Self::Left(b)) => {
                a.abs_diff(b) < BOUNDARY_CANDIDATE_SEPARATION
            }
            _ => false,
        }
    }

    fn lengths(self, left: usize, right: usize) -> Option<(usize, usize)> {
        match self {
            Self::Right(bytes) => Some((left.checked_add(bytes)?, right.checked_sub(bytes)?)),
            Self::Left(bytes) => Some((left.checked_sub(bytes)?, right.checked_add(bytes)?)),
        }
    }

    const fn boundary(self, offset: usize) -> Option<usize> {
        match self {
            Self::Right(bytes) => offset.checked_add(bytes),
            Self::Left(bytes) => offset.checked_sub(bytes),
        }
    }
}

/// Only the frozen-cost change and the token boundary are needed to rank
/// nominees. Build full histograms for at most four winners after the scan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BoundaryCandidate {
    delta: i64,
    shift: BoundaryShift,
}

/// Small, sorted shortlist with one overflow slot to reproduce the existing
/// sort-then-truncate tie behavior exactly. No allocation is needed while
/// scanning boundary shifts, including when the fifth candidate is considered.
#[derive(Clone, Debug)]
struct BoundaryShortlist {
    entries: [BoundaryCandidate; EXACT_BOUNDARY_CANDIDATES + 1],
    len: usize,
}

impl Default for BoundaryShortlist {
    fn default() -> Self {
        Self {
            entries: [BoundaryCandidate {
                delta: 0,
                shift: BoundaryShift::Right(0),
            }; EXACT_BOUNDARY_CANDIDATES + 1],
            len: 0,
        }
    }
}

impl BoundaryShortlist {
    fn as_slice(&self) -> &[BoundaryCandidate] {
        &self.entries[..self.len]
    }

    fn consider(&mut self, delta: i64, shift: BoundaryShift) {
        if delta >= 0 {
            return;
        }

        if let Some(index) = self
            .as_slice()
            .iter()
            .position(|candidate| candidate.shift.is_near(shift))
        {
            if delta >= self.entries[index].delta {
                // The shortlist is already sorted, so no sorting is needed
                // when the existing nearby representative is still better.
                return;
            }
            self.entries[index] = BoundaryCandidate { delta, shift };
        } else {
            if self.len == EXACT_BOUNDARY_CANDIDATES && delta > self.entries[self.len - 1].delta {
                // Strictly worse than the weakest finalist. Equality must
                // still go through the old five-item unstable sort because
                // truncation could retain either tied representative.
                return;
            }
            self.entries[self.len] = BoundaryCandidate { delta, shift };
            self.len += 1;
        }

        self.entries[..self.len].sort_unstable_by_key(|candidate| candidate.delta);
        self.len = self.len.min(EXACT_BOUNDARY_CANDIDATES);
    }
}

fn fixed_cost_shift_candidates(
    left: &PlannedBlock,
    right: &PlannedBlock,
    left_costs: &TokenCosts,
    right_costs: &TokenCosts,
) -> BoundaryShortlist {
    debug_assert_eq!(left.range.end, right.range.start);
    let mut candidates = BoundaryShortlist::default();
    let token_deltas = TokenCostDelta::between(left_costs, right_costs);

    // Positive shift: move a prefix of the right block into the left block.
    let mut moved = 0usize;
    let mut moved_tokens = 0usize;
    let mut delta = 0i64;
    for &token in &right.tokens {
        moved += token.input_len();
        moved_tokens += 1;
        if left.tokens.len() + moved_tokens > MAX_BLOCK_TOKENS
            || right.input_len().saturating_sub(moved) < MIN_BLOCK_LENGTH
        {
            break;
        }
        delta += token_deltas.token_delta(token);
        candidates.consider(delta, BoundaryShift::Right(moved));
    }

    // Negative shift: move a suffix of the left block into the right block.
    moved = 0;
    moved_tokens = 0;
    delta = 0;
    for &token in left.tokens.iter().rev() {
        moved += token.input_len();
        moved_tokens += 1;
        if right.tokens.len() + moved_tokens > MAX_BLOCK_TOKENS
            || left.input_len().saturating_sub(moved) < MIN_BLOCK_LENGTH
        {
            break;
        }
        delta -= token_deltas.token_delta(token);
        candidates.consider(delta, BoundaryShift::Left(moved));
    }

    candidates
}

/// Reconstruct exact Huffman frequencies only for shortlisted shifts. Each
/// direction is replayed once up to its most distant nominee; snapshots retain
/// the original delta-ranked order so exact-score tie handling does not change.
fn shortlisted_shift_histograms(
    left: &PlannedBlock,
    right: &PlannedBlock,
    left_histogram: &TokenHistogram,
    right_histogram: &TokenHistogram,
    candidates: &[BoundaryCandidate],
) -> Result<[Option<(TokenHistogram, TokenHistogram)>; EXACT_BOUNDARY_CANDIDATES], ReturnCode> {
    if candidates.len() > EXACT_BOUNDARY_CANDIDATES {
        return Err(ReturnCode::StreamError);
    }
    let mut snapshots = core::array::from_fn(|_| None);

    for moving_right in [true, false] {
        let furthest = candidates
            .iter()
            .filter_map(|candidate| match (moving_right, candidate.shift) {
                (true, BoundaryShift::Right(bytes)) | (false, BoundaryShift::Left(bytes)) => {
                    Some(bytes)
                }
                _ => None,
            })
            .max()
            .unwrap_or(0);
        if furthest == 0 {
            continue;
        }

        let mut left_counts = left_histogram.clone();
        let mut right_counts = right_histogram.clone();
        let mut moved = 0usize;
        let token_count = if moving_right {
            right.tokens.len()
        } else {
            left.tokens.len()
        };
        for index in 0..token_count {
            let token = if moving_right {
                right.tokens[index]
            } else {
                left.tokens[left.tokens.len() - 1 - index]
            };
            moved += token.input_len();
            if moving_right {
                // Remove before adding: never transiently exceed block capacity
                // when the candidate is itself a valid full-size partition.
                right_counts.remove_token(token)?;
                left_counts.add_token(token)?;
            } else {
                left_counts.remove_token(token)?;
                right_counts.add_token(token)?;
            }
            let shift = if moving_right {
                BoundaryShift::Right(moved)
            } else {
                BoundaryShift::Left(moved)
            };
            for (candidate, snapshot) in candidates.iter().zip(&mut snapshots) {
                if candidate.shift == shift {
                    *snapshot = Some((left_counts.clone(), right_counts.clone()));
                }
            }
            if moved == furthest {
                break;
            }
        }
        if moved != furthest {
            return Err(ReturnCode::StreamError);
        }
    }

    if !snapshots.iter().take(candidates.len()).all(Option::is_some) {
        return Err(ReturnCode::StreamError);
    }
    Ok(snapshots)
}

/// Exact-score the frozen-cost shortlist against the best unchanged/merged
/// partition. Returning only strict improvements preserves the original tie
/// preference and keeps partition application separate from candidate scoring.
fn best_exact_boundary_shift(
    state: &mut super::State<'_>,
    left: &PlannedBlock,
    right: &PlannedBlock,
    left_histogram: &TokenHistogram,
    right_histogram: &TokenHistogram,
    candidates: &[BoundaryCandidate],
    mut best_total_bits: usize,
) -> Result<Option<(BoundaryShift, usize)>, ReturnCode> {
    let snapshots =
        shortlisted_shift_histograms(left, right, left_histogram, right_histogram, candidates)?;
    let mut best_shift = None;
    for (candidate, snapshot) in candidates.iter().copied().zip(snapshots) {
        // All nominees end at token boundaries, verified before scoring.
        let Some((shifted_left, shifted_right)) = snapshot else {
            continue;
        };
        let Some((new_left_len, new_right_len)) =
            candidate.shift.lengths(left.input_len(), right.input_len())
        else {
            continue;
        };
        let Some(boundary) = candidate.shift.boundary(left.range.end) else {
            continue;
        };
        let Some(new_left_score) =
            compressed_histogram_score(state, &shifted_left, new_left_len, None)
        else {
            continue;
        };
        let Some(new_right_score) =
            compressed_histogram_score(state, &shifted_right, new_right_len, None)
        else {
            continue;
        };
        let total_bits = new_left_score.bits + new_right_score.bits;
        if total_bits < best_total_bits {
            best_total_bits = total_bits;
            best_shift = Some((candidate.shift, boundary));
        }
    }
    Ok(best_shift)
}

/// Apply a chosen shift directly to the owned token vectors. Partition scoring
/// is complete before this is called, so copying both blocks to construct the
/// new partition would add work without preserving anything useful.
fn shift_tokens(
    left: &mut PlannedBlock,
    right: &mut PlannedBlock,
    shift: BoundaryShift,
) -> Result<(), ReturnCode> {
    match shift {
        BoundaryShift::Right(wanted) => {
            let mut bytes = 0usize;
            let mut split = 0usize;
            while split < right.tokens.len() && bytes < wanted {
                bytes = bytes
                    .checked_add(right.tokens[split].input_len())
                    .ok_or(ReturnCode::StreamError)?;
                split += 1;
            }
            if bytes != wanted
                || left
                    .tokens
                    .len()
                    .checked_add(split)
                    .is_none_or(|len| len > MAX_BLOCK_TOKENS)
            {
                return Err(ReturnCode::StreamError);
            }
            left.tokens.extend(right.tokens.drain(..split));
        }
        BoundaryShift::Left(wanted) => {
            let mut bytes = 0usize;
            let mut split = left.tokens.len();
            while split > 0 && bytes < wanted {
                split -= 1;
                bytes = bytes
                    .checked_add(left.tokens[split].input_len())
                    .ok_or(ReturnCode::StreamError)?;
            }
            if bytes != wanted
                || right
                    .tokens
                    .len()
                    .checked_add(left.tokens.len() - split)
                    .is_none_or(|len| len > MAX_BLOCK_TOKENS)
            {
                return Err(ReturnCode::StreamError);
            }
            let mut moved = left.tokens.split_off(split);
            moved.append(&mut right.tokens);
            right.tokens = moved;
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub enum BoundaryOptimization {
    Unchanged,
    Shifted,
    Merged(PlannedBlock),
}

/// Choose the cheapest local Huffman partition for two adjacent parsed blocks.
///
/// Frozen code lengths cheaply nominate up to four distinct boundary shifts. Those shifts and the
/// optional one-block merge are all rescored by rebuilding the production Huffman trees, then the
/// single cheapest of unchanged / shifted / merged is applied. Match discovery and token parsing are
/// deliberately not repeated here; a caller may use the retained right-hand candidate table for a
/// follow-up DP-only reparse after the partition changes.
///
/// A merged result takes ownership of both blocks' token vectors, leaving them empty.
/// A shifted result edits the original vectors in place without cloning either one.
pub fn optimize_token_partition(
    state: &mut super::State<'_>,
    left: &mut PlannedBlock,
    right: &mut PlannedBlock,
) -> Result<BoundaryOptimization, ReturnCode> {
    // A pair of individually well-formed ranges can still be nonadjacent.
    // Check those independent contracts before interpreting token spans.
    if left.range.start > left.range.end || right.range.start > right.range.end {
        return Err(ReturnCode::StreamError);
    }
    if left.range.end != right.range.start {
        return Err(ReturnCode::StreamError);
    }
    if left.token_input_len() != left.range.len() || right.token_input_len() != right.range.len() {
        return Err(ReturnCode::StreamError);
    }

    let left_histogram = TokenHistogram::from_tokens(&left.tokens)?;
    let right_histogram = TokenHistogram::from_tokens(&right.tokens)?;
    let mut left_costs = TokenCosts::unseen();
    let Some(left_score) = compressed_histogram_score(
        state,
        &left_histogram,
        left.input_len(),
        Some(&mut left_costs),
    ) else {
        return Ok(BoundaryOptimization::Unchanged);
    };
    let mut right_costs = TokenCosts::unseen();
    let Some(right_score) = compressed_histogram_score(
        state,
        &right_histogram,
        right.input_len(),
        Some(&mut right_costs),
    ) else {
        return Ok(BoundaryOptimization::Unchanged);
    };

    let mut best_total_bits = left_score.bits + right_score.bits;
    let mut merge_is_best = false;

    let merged_len = left
        .input_len()
        .checked_add(right.input_len())
        .ok_or(ReturnCode::StreamError)?;
    if left.tokens.len() + right.tokens.len() <= MAX_BLOCK_TOKENS {
        let merged_histogram = TokenHistogram::merged(&left_histogram, &right_histogram)?;
        if let Some(merged_score) =
            compressed_histogram_score(state, &merged_histogram, merged_len, None)
        {
            if merged_score.bits < best_total_bits {
                best_total_bits = merged_score.bits;
                merge_is_best = true;
            }
        }
    }

    let candidates = fixed_cost_shift_candidates(left, right, &left_costs, &right_costs);
    let best_shift = best_exact_boundary_shift(
        state,
        left,
        right,
        &left_histogram,
        &right_histogram,
        candidates.as_slice(),
        best_total_bits,
    )?;

    let outcome = if let Some((shift, boundary)) = best_shift {
        shift_tokens(left, right, shift)?;
        left.range.end = boundary;
        right.range.start = boundary;
        BoundaryOptimization::Shifted
    } else if merge_is_best {
        let mut tokens = core::mem::take(&mut left.tokens);
        tokens.append(&mut right.tokens);
        BoundaryOptimization::Merged(PlannedBlock {
            range: left.range.start..right.range.end,
            tokens,
        })
    } else {
        BoundaryOptimization::Unchanged
    };

    match &outcome {
        BoundaryOptimization::Merged(merged) => {
            debug_assert_eq!(merged.token_input_len(), merged.range.len());
            debug_assert!(merged.tokens.len() <= MAX_BLOCK_TOKENS);
        }
        BoundaryOptimization::Unchanged | BoundaryOptimization::Shifted => {
            debug_assert_eq!(left.range.end, right.range.start);
            debug_assert!(left.tokens.len() <= MAX_BLOCK_TOKENS);
            debug_assert!(right.tokens.len() <= MAX_BLOCK_TOKENS);
            debug_assert_eq!(left.token_input_len(), left.range.len());
            debug_assert_eq!(right.token_input_len(), right.range.len());
        }
    }
    Ok(outcome)
}

/// Parse one splitter range into a production-ready token block. A caller only needs to retain two
/// adjacent blocks at a time to optimize their shared Huffman boundary.
pub fn parse_planned_block_source<'a>(
    state: &mut super::State<'_>,
    input: &'a [u8],
    range: Range<usize>,
    max_distance: usize,
    optimization_passes: usize,
    previous_model: Option<&ParseModel>,
) -> Result<PlannedBlockSource<'a>, ReturnCode> {
    // This API accepts a caller-provided range, not merely the trusted
    // output of the built-in splitter. Reject invalid bounds before slicing
    // or updating any compression model.
    if range.start > range.end || range.end > input.len() {
        return Err(ReturnCode::StreamError);
    }
    let history_start = range.start.saturating_sub(max_distance);
    let context = &input[history_start..range.end];
    let local_range = (range.start - history_start)..context.len();
    let source = FrontierSource::new(context, max_distance)?;
    let candidates = source.build(local_range.clone())?;
    // Only a later boundary shift toward the history could need the
    // suffix order again. Release it for the first block or fixed parse.
    let source = (optimization_passes > 0 && local_range.start != 0).then_some(source);
    let (parsed, model) = optimized_parse_range(
        state,
        context,
        &candidates,
        local_range,
        optimization_passes,
        previous_model,
    )?;
    let block = PlannedBlock {
        range,
        tokens: parsed.tokens,
    };
    debug_assert_eq!(block.token_input_len(), block.range.len());
    Ok(PlannedBlockSource {
        block,
        model,
        context_start: history_start,
        context,
        source,
        candidates,
    })
}

#[cfg(test)]
mod boundary_tests {
    use super::*;

    #[test]
    fn histogram_rejects_invalid_matches_and_capacity_without_mutation() -> Result<(), ReturnCode> {
        let full_tokens = vec![ParseToken::Literal(b'a'); MAX_BLOCK_TOKENS];
        let mut histogram = TokenHistogram::from_tokens(&full_tokens)?;
        assert_eq!(histogram.total_tokens, MAX_BLOCK_TOKENS);
        assert_eq!(
            usize::from(histogram.literal_length[usize::from(b'a')]),
            MAX_BLOCK_TOKENS
        );
        assert_eq!(histogram.literal_length[256], 1);

        let original = histogram.clone();
        assert!(matches!(
            histogram.add_token(ParseToken::Literal(b'a')),
            Err(ReturnCode::StreamError)
        ));
        assert_eq!(histogram, original);
        assert!(matches!(
            TokenHistogram::from_tokens(&vec![ParseToken::Literal(b'a'); MAX_BLOCK_TOKENS + 1]),
            Err(ReturnCode::StreamError)
        ));

        let other = TokenHistogram::from_tokens(&[ParseToken::Literal(b'b')])?;
        assert!(matches!(
            TokenHistogram::merged(&histogram, &other),
            Err(ReturnCode::StreamError)
        ));
        assert_eq!(histogram, original);
        assert_eq!(other.total_tokens, 1);

        for invalid in [
            ParseToken::Match {
                length: 0,
                distance: 1,
            },
            ParseToken::Match {
                length: 2,
                distance: 1,
            },
            ParseToken::Match {
                length: 259,
                distance: 1,
            },
            ParseToken::Match {
                length: 3,
                distance: 0,
            },
            ParseToken::Match {
                length: 3,
                distance: 32_769,
            },
            ParseToken::Match {
                length: u16::MAX,
                distance: u16::MAX,
            },
        ] {
            assert!(matches!(
                TokenHistogram::from_tokens(&[ParseToken::Literal(b'x'), invalid]),
                Err(ReturnCode::StreamError)
            ));
            let mut partial = TokenHistogram::from_tokens(&[ParseToken::Literal(b'x')])?;
            let saved = partial.clone();
            assert_eq!(partial.add_token(invalid), Err(ReturnCode::StreamError));
            assert_eq!(partial, saved);
            assert_eq!(partial.remove_token(invalid), Err(ReturnCode::StreamError));
            assert_eq!(partial, saved);
        }

        let mut singleton = TokenHistogram::from_tokens(&[ParseToken::Literal(b'x')])?;
        let original = singleton.clone();
        assert_eq!(
            singleton.remove_token(ParseToken::Literal(b'y')),
            Err(ReturnCode::StreamError)
        );
        assert_eq!(singleton, original);
        singleton.remove_token(ParseToken::Literal(b'x'))?;
        assert_eq!(singleton, TokenHistogram::empty());
        assert_eq!(
            singleton.remove_token(ParseToken::Literal(b'x')),
            Err(ReturnCode::StreamError)
        );
        assert_eq!(singleton, TokenHistogram::empty());
        Ok(())
    }

    #[test]
    fn invalid_boundary_shifts_leave_both_blocks_unchanged() -> Result<(), ReturnCode> {
        let mut left = PlannedBlock {
            range: 0..2,
            tokens: vec![ParseToken::Literal(b'a'); 2],
        };
        let mut right = PlannedBlock {
            range: 2..4,
            tokens: vec![ParseToken::Literal(b'b'); 2],
        };
        let original = (left.clone(), right.clone());

        assert_eq!(
            shift_tokens(&mut left, &mut right, BoundaryShift::Right(3)),
            Err(ReturnCode::StreamError)
        );
        assert_eq!(
            (&left.range, &left.tokens),
            (&original.0.range, &original.0.tokens)
        );
        assert_eq!(
            (&right.range, &right.tokens),
            (&original.1.range, &original.1.tokens)
        );

        let candidates = [BoundaryCandidate {
            delta: -1,
            shift: BoundaryShift::Right(3),
        }];
        assert!(matches!(
            shortlisted_shift_histograms(
                &left,
                &right,
                &TokenHistogram::from_tokens(&left.tokens)?,
                &TokenHistogram::from_tokens(&right.tokens)?,
                &candidates
            ),
            Err(ReturnCode::StreamError)
        ));
        let candidates = [candidates[0]; EXACT_BOUNDARY_CANDIDATES + 1];
        assert!(matches!(
            shortlisted_shift_histograms(
                &left,
                &right,
                &TokenHistogram::from_tokens(&left.tokens)?,
                &TokenHistogram::from_tokens(&right.tokens)?,
                &candidates
            ),
            Err(ReturnCode::StreamError)
        ));

        let mut stream = super::super::DeflateStream::new(super::super::DeflateConfig::new(9))?;
        left.range = 0..3;
        let snapshot = left.clone();
        assert!(matches!(
            optimize_token_partition(&mut stream.state, &mut left, &mut right),
            Err(ReturnCode::StreamError)
        ));
        assert_eq!(left.range, snapshot.range);
        assert_eq!(left.tokens, snapshot.tokens);
        assert_eq!(right.range, original.1.range);
        assert_eq!(right.tokens, original.1.tokens);
        Ok(())
    }

    #[test]
    fn tally_preflight_rejects_invalid_or_oversized_window_before_mutation(
    ) -> Result<(), ReturnCode> {
        let config = super::super::DeflateConfig {
            window_bits: 9,
            ..super::super::DeflateConfig::new(9)
        };
        let mut stream = super::super::DeflateStream::new(config)?;
        let state = &mut *stream.state;
        let original_frequency = state.l_desc.dyn_tree[usize::from(b'a')].freq();
        // A normal streaming deflater owns a smaller symbol allocation than
        // the special 65,536-slot whole-input compressor. Reject that capacity
        // mismatch before SymBuf::push_lit can run off the end.
        let oversized_for_state =
            vec![ParseToken::Literal(b'a'); state.sym_buf.block_capacity() + 1];
        assert_eq!(
            tally_tokens(state, &oversized_for_state),
            Err(ReturnCode::StreamError)
        );
        assert!(state.sym_buf.is_empty());
        assert_eq!(
            state.l_desc.dyn_tree[usize::from(b'a')].freq(),
            original_frequency
        );

        for invalid in [
            ParseToken::Match {
                length: 3,
                distance: 512 + 1,
            },
            ParseToken::Match {
                length: 2,
                distance: 1,
            },
            ParseToken::Match {
                length: 3,
                distance: 0,
            },
        ] {
            assert_eq!(
                tally_tokens(state, &[ParseToken::Literal(b'a'), invalid]),
                Err(ReturnCode::StreamError)
            );
            assert!(state.sym_buf.is_empty());
            assert_eq!(
                state.l_desc.dyn_tree[usize::from(b'a')].freq(),
                original_frequency
            );
        }
        tally_tokens(state, &[ParseToken::Literal(b'a')])?;
        assert!(!state.sym_buf.is_empty());
        assert_eq!(
            tally_tokens(state, &[ParseToken::Literal(b'a')]),
            Err(ReturnCode::StreamError)
        );
        assert_eq!(
            state.l_desc.dyn_tree[usize::from(b'a')].freq(),
            original_frequency + 1
        );
        Ok(())
    }

    #[test]
    fn planned_range_rejects_invalid_offsets_and_history_without_panicking(
    ) -> Result<(), ReturnCode> {
        let mut stream = super::super::DeflateStream::new(super::super::DeflateConfig::new(9))?;
        let state = &mut *stream.state;
        let input = b"test input";

        for range in [
            Range { start: 5, end: 3 },
            0..input.len() + 1,
            input.len() + 1..input.len() + 2,
        ] {
            assert!(matches!(
                parse_planned_block_source(state, input, range, 32_768, 2, None),
                Err(ReturnCode::StreamError)
            ));
        }
        assert!(matches!(
            parse_planned_block_source(state, input, 0..input.len(), 32_769, 2, None),
            Err(ReturnCode::StreamError)
        ));
        Ok(())
    }

    #[test]
    fn boundary_reparse_keeps_exact_model_of_selected_tokens() -> Result<(), crate::ReturnCode> {
        // Test both a warm previous model and no previous model. A reparse
        // may retain the original path, select an improved path, or converge
        // after a worse intermediate iteration; in all cases the retained
        // model must describe the exact final token histogram.
        let input = (0..15_000)
            .map(|index| {
                if index % 257 < 220 {
                    b"abcde"[index % 5]
                } else {
                    b"0123456789"[index % 10]
                }
            })
            .collect::<Vec<_>>();
        let mut stream = super::super::DeflateStream::new(super::super::DeflateConfig::new(9))?;
        let state = &mut *stream.state;
        for passes in [1, 2, 4] {
            let mut block =
                parse_planned_block_source(state, &input, 4_000..13_000, 32_768, passes, None)?;
            let previous = model_from_tokens(state, &block.block.tokens)?;
            for prior in [None, Some(&previous)] {
                let _changed = block.reparse_with_existing_candidates(state, passes, prior)?;
                let independently_scored = model_from_tokens(state, &block.block.tokens)?;
                assert_eq!(
                    block.model,
                    independently_scored,
                    "passes={passes}, previous_model={}",
                    prior.is_some()
                );
                assert_eq!(block.block.token_input_len(), block.block.range.len());
            }
        }
        Ok(())
    }

    #[test]
    fn boundary_reparse_extends_only_the_missing_history_queries() -> Result<(), ReturnCode> {
        let input = (0..18_000usize)
            .map(|i| b"xyzabc123"[i % 9])
            .collect::<Vec<_>>();
        let mut stream = super::super::DeflateStream::new(super::super::DeflateConfig::new(9))?;
        let state = &mut *stream.state;
        for passes in [1, 3] {
            let mut block =
                parse_planned_block_source(state, &input, 8_000..16_000, 4096, passes, None)?;
            let original_start = block.candidates.first_query;
            let context = block.context;
            let full = build_match_candidates(context, 4096)?;
            assert_eq!(original_start, 4096);
            let original = block.candidates.clone();
            // Move the boundary left by a sequence of valid literals.
            let shift = 513;
            let mut tokens = input[8000 - shift..8000]
                .iter()
                .copied()
                .map(ParseToken::Literal)
                .collect::<Vec<_>>();
            tokens.append(&mut block.block.tokens);
            block.block.tokens = tokens;
            block.block.range.start -= shift;
            let mut reference = PlannedBlockSource {
                block: block.block.clone(),
                model: block.model.clone(),
                context_start: block.context_start,
                context,
                source: None,
                candidates: full.clone(),
            };
            let actual = block.reparse_with_existing_candidates(state, passes, None)?;
            let expected = reference.reparse_with_existing_candidates(state, passes, None)?;
            assert_eq!(actual, expected);
            assert_eq!(block.block.tokens, reference.block.tokens);
            assert_eq!(block.model, reference.model);
            assert_eq!(block.candidates.first_query, original_start - shift);
            for position in original_start - shift..context.len() {
                assert_eq!(
                    block.candidates.for_position(position),
                    full.for_position(position)
                );
                if position >= original_start {
                    assert_eq!(
                        block.candidates.for_position(position),
                        original.for_position(position)
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn exact_score_with_and_without_learned_costs_agree() -> Result<(), crate::ReturnCode> {
        let mut stream = super::super::DeflateStream::new(super::super::DeflateConfig::new(9))?;
        let state = &mut *stream.state;
        let samples = [
            vec![ParseToken::Literal(b'a'); 96],
            (0u8..=250)
                .cycle()
                .take(4_096)
                .map(ParseToken::Literal)
                .collect(),
            vec![
                ParseToken::Literal(b'a'),
                ParseToken::Literal(b'b'),
                ParseToken::Match {
                    length: 3,
                    distance: 1,
                },
                ParseToken::Match {
                    length: 258,
                    distance: 32_768,
                },
                ParseToken::Match {
                    length: 25,
                    distance: 2_048,
                },
            ],
        ];
        for tokens in samples {
            let histogram = TokenHistogram::from_tokens(&tokens)?;
            let (expected_costs, expected_score) = costs_and_block_score(state, &tokens)?;
            assert_eq!(score_histogram(state, &histogram, None), expected_score);

            let mut costs = TokenCosts::unseen();
            assert_eq!(
                score_histogram(state, &histogram, Some(&mut costs)),
                expected_score
            );
            assert_eq!(costs, expected_costs);

            let input_len = tokens.iter().map(|token| token.input_len()).sum();
            let without_costs = compressed_histogram_score(state, &histogram, input_len, None);
            let mut filtered_costs = TokenCosts::unseen();
            let with_costs =
                compressed_histogram_score(state, &histogram, input_len, Some(&mut filtered_costs));
            assert_eq!(without_costs, with_costs);
            assert_eq!(filtered_costs, expected_costs);
        }
        Ok(())
    }

    #[test]
    fn relative_frozen_token_cost_matches_full_cost_difference() {
        // Exercise all 256 length values, boundary distance symbols, and
        // literal codes, with asymmetric and extreme valid u16 cost models.
        let mut left = TokenCosts::unseen();
        let mut right = TokenCosts::unseen();
        for byte in u8::MIN..=u8::MAX {
            let index = usize::from(byte);
            left.literal[index] = u16::from(byte) * 199 + 13;
            right.literal[index] = u16::from(byte) * 127 + 123;
        }
        for index in 0..LENGTH_CODES {
            left.length[index] = if index % 2 == 0 { 0 } else { u16::MAX };
            right.length[index] = if index % 2 == 0 { u16::MAX } else { 0 };
        }
        for index in 0..D_CODES {
            left.distance[index] = if index % 3 == 0 { u16::MAX } else { 7 };
            right.distance[index] = if index % 3 == 0 { 0 } else { u16::MAX };
        }

        let deltas = TokenCostDelta::between(&left, &right);
        let reverse = TokenCostDelta::between(&right, &left);
        for byte in u8::MIN..=u8::MAX {
            let token = ParseToken::Literal(byte);
            let expected = i64::from(left.token_cost(token)) - i64::from(right.token_cost(token));
            assert_eq!(deltas.token_delta(token), expected);
            assert_eq!(reverse.token_delta(token), -expected);
        }

        for length in 3_u16..=258 {
            for distance in 1_u16..=32_768 {
                // Every valid length/distance combination is representable
                // by the DEFLATE coding tables; verify the exact cancellation.
                let token = ParseToken::Match { length, distance };
                let expected =
                    i64::from(left.token_cost(token)) - i64::from(right.token_cost(token));
                assert_eq!(deltas.token_delta(token), expected);
                assert_eq!(reverse.token_delta(token), -expected);
            }
        }
    }

    #[test]
    fn fixed_capacity_shortlist_preserves_sort_and_tie_semantics() {
        // Exact oracle: the previous allocation-backed candidate admission,
        // unstable sort and truncation, including ties at the fourth slot.
        fn reference_consider(
            candidates: &mut Vec<BoundaryCandidate>,
            delta: i64,
            shift: BoundaryShift,
        ) {
            if delta >= 0 {
                return;
            }
            if let Some(existing) = candidates
                .iter_mut()
                .find(|candidate| candidate.shift.is_near(shift))
            {
                if delta < existing.delta {
                    *existing = BoundaryCandidate { delta, shift };
                }
            } else {
                candidates.push(BoundaryCandidate { delta, shift });
            }
            candidates.sort_unstable_by_key(|candidate| candidate.delta);
            candidates.truncate(EXACT_BOUNDARY_CANDIDATES);
        }

        let mut previous = Vec::new();
        let mut shortlist = BoundaryShortlist::default();
        let mut rng = 0x5b25_a65d_f197_20b3u64;
        for iteration in 0..20_000 {
            rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let delta = i64::try_from((rng >> 9) % 117).unwrap_or_default() - 90;
            let distance = usize::try_from((rng >> 18) % 4_096).unwrap_or_default() + 1;
            let shift = if (rng >> 49) & 1 == 0 {
                BoundaryShift::Left(distance)
            } else {
                BoundaryShift::Right(distance)
            };
            reference_consider(&mut previous, delta, shift);
            shortlist.consider(delta, shift);
            assert_eq!(
                shortlist.as_slice(),
                previous.as_slice(),
                "iteration={iteration}, delta={delta}, shift={shift:?}"
            );
        }
    }

    #[test]
    fn boundary_shift_supports_full_usize_positions_and_checks_lengths() {
        let boundary = (isize::MAX as usize) + 16;
        assert_eq!(
            BoundaryShift::Right(7).boundary(boundary),
            Some(boundary + 7)
        );
        assert_eq!(
            BoundaryShift::Left(7).boundary(boundary),
            Some(boundary - 7)
        );
        assert_eq!(BoundaryShift::Left(7).lengths(30, 40), Some((23, 47)));
        assert_eq!(BoundaryShift::Right(7).lengths(30, 40), Some((37, 33)));
        assert_eq!(BoundaryShift::Left(31).lengths(30, 40), None);
        assert_eq!(BoundaryShift::Right(41).lengths(30, 40), None);
        assert_eq!(BoundaryShift::Left(1).boundary(0), None);
        assert_eq!(BoundaryShift::Right(1).boundary(usize::MAX), None);
        assert!(BoundaryShift::Left(256).is_near(BoundaryShift::Left(257)));
        assert!(!BoundaryShift::Left(256).is_near(BoundaryShift::Right(256)));
        assert!(!BoundaryShift::Right(256).is_near(BoundaryShift::Right(512)));
    }

    #[test]
    fn shortlisted_histograms_match_exact_mixed_token_repartitions() -> Result<(), ReturnCode> {
        // Ranking is by frozen cost, not movement size or direction. Replaying
        // both directions must preserve each nominee's original index while
        // walking whole literal/match tokens with different DEFLATE codes.
        let mut left_tokens = vec![ParseToken::Literal(b'a'); 512];
        left_tokens.extend([
            ParseToken::Match {
                length: 7,
                distance: 1,
            },
            ParseToken::Literal(b'c'),
            ParseToken::Match {
                length: 258,
                distance: 32_768,
            },
            ParseToken::Literal(b'd'),
        ]);
        let mut right_tokens = vec![ParseToken::Literal(b'b'); 300];
        right_tokens.extend([
            ParseToken::Match {
                length: 3,
                distance: 12,
            },
            ParseToken::Literal(b'e'),
            ParseToken::Match {
                length: 258,
                distance: 30_000,
            },
            ParseToken::Literal(b'f'),
        ]);
        let left = PlannedBlock {
            range: 0..779,
            tokens: left_tokens,
        };
        let right = PlannedBlock {
            range: 779..1342,
            tokens: right_tokens,
        };
        assert_eq!(left.token_input_len(), left.range.len());
        assert_eq!(right.token_input_len(), right.range.len());
        let nominees = [
            BoundaryCandidate {
                delta: -50,
                shift: BoundaryShift::Left(259),
            },
            BoundaryCandidate {
                delta: -40,
                shift: BoundaryShift::Right(303),
            },
            BoundaryCandidate {
                delta: -30,
                shift: BoundaryShift::Left(1),
            },
            BoundaryCandidate {
                delta: -20,
                shift: BoundaryShift::Right(300),
            },
        ];
        let snapshots = shortlisted_shift_histograms(
            &left,
            &right,
            &TokenHistogram::from_tokens(&left.tokens)?,
            &TokenHistogram::from_tokens(&right.tokens)?,
            &nominees,
        )?;
        assert!(snapshots.iter().all(Option::is_some));
        for (nominee, (left_hist, right_hist)) in
            nominees.into_iter().zip(snapshots.into_iter().flatten())
        {
            let mut moved_left = left.clone();
            let mut moved_right = right.clone();
            shift_tokens(&mut moved_left, &mut moved_right, nominee.shift)?;
            assert_eq!(
                left_hist,
                TokenHistogram::from_tokens(&moved_left.tokens)?,
                "{:?}",
                nominee.shift
            );
            assert_eq!(
                right_hist,
                TokenHistogram::from_tokens(&moved_right.tokens)?,
                "{:?}",
                nominee.shift
            );
            assert_eq!(
                moved_left.token_input_len() + moved_right.token_input_len(),
                left.input_len() + right.input_len()
            );
        }
        Ok(())
    }

    #[test]
    fn fixed_cost_boundary_shift_left_preserves_whole_tokens() -> Result<(), ReturnCode> {
        let mut left_tokens = vec![ParseToken::Literal(b'a'); 5_500];
        left_tokens.extend(core::iter::repeat_n(ParseToken::Literal(b'b'), 500));
        let mut left = PlannedBlock {
            range: 0..6_000,
            tokens: left_tokens,
        };
        let mut right = PlannedBlock {
            range: 6_000..12_000,
            tokens: vec![ParseToken::Literal(b'b'); 6_000],
        };
        let mut left_costs = TokenCosts::fixed();
        let mut right_costs = TokenCosts::fixed();
        left_costs.literal[usize::from(b'a')] = COST_SCALE;
        left_costs.literal[usize::from(b'b')] = 8 * COST_SCALE;
        right_costs.literal[usize::from(b'a')] = 8 * COST_SCALE;
        right_costs.literal[usize::from(b'b')] = COST_SCALE;
        let left_histogram = TokenHistogram::from_tokens(&left.tokens)?;
        let right_histogram = TokenHistogram::from_tokens(&right.tokens)?;
        let shifts = fixed_cost_shift_candidates(&left, &right, &left_costs, &right_costs);
        let snapshots = shortlisted_shift_histograms(
            &left,
            &right,
            &left_histogram,
            &right_histogram,
            shifts.as_slice(),
        )?;
        let snapshot = &snapshots[0];
        assert_eq!(shifts.as_slice()[0].shift, BoundaryShift::Left(500));
        shift_tokens(&mut left, &mut right, shifts.as_slice()[0].shift)?;
        let new_left = &left.tokens;
        let new_right = &right.tokens;
        assert_eq!(new_left.len(), 5_500);
        assert_eq!(new_right.len(), 6_500);
        assert_eq!(
            snapshot.as_ref().map(|(left, _)| left),
            Some(&TokenHistogram::from_tokens(new_left)?)
        );
        assert_eq!(
            snapshot.as_ref().map(|(_, right)| right),
            Some(&TokenHistogram::from_tokens(new_right)?)
        );
        assert!(new_right[..500]
            .iter()
            .all(|token| *token == ParseToken::Literal(b'b')));
        Ok(())
    }

    #[test]
    fn fixed_cost_boundary_shift_moves_whole_tokens_and_conserves_input() -> Result<(), ReturnCode>
    {
        let mut left = PlannedBlock {
            range: 0..6_000,
            tokens: vec![ParseToken::Literal(b'a'); 6_000],
        };
        let mut right_tokens = vec![ParseToken::Literal(b'a'); 500];
        right_tokens.extend(core::iter::repeat_n(ParseToken::Literal(b'b'), 5_500));
        let mut right = PlannedBlock {
            range: 6_000..12_000,
            tokens: right_tokens,
        };

        let mut left_costs = TokenCosts::fixed();
        let mut right_costs = TokenCosts::fixed();
        left_costs.literal[b'a' as usize] = COST_SCALE;
        left_costs.literal[b'b' as usize] = 8 * COST_SCALE;
        right_costs.literal[b'a' as usize] = 8 * COST_SCALE;
        right_costs.literal[b'b' as usize] = COST_SCALE;

        let left_histogram = TokenHistogram::from_tokens(&left.tokens)?;
        let right_histogram = TokenHistogram::from_tokens(&right.tokens)?;
        let shifts = fixed_cost_shift_candidates(&left, &right, &left_costs, &right_costs);
        let snapshots = shortlisted_shift_histograms(
            &left,
            &right,
            &left_histogram,
            &right_histogram,
            shifts.as_slice(),
        )?;
        let snapshot = &snapshots[0];
        assert_eq!(shifts.as_slice()[0].shift, BoundaryShift::Right(500));
        let shift = shifts.as_slice()[0].shift;
        shift_tokens(&mut left, &mut right, shift)?;
        let shifted_left = &left.tokens;
        let shifted_right = &right.tokens;
        assert_eq!(
            snapshot.as_ref().map(|(left, _)| left),
            Some(&TokenHistogram::from_tokens(shifted_left)?)
        );
        assert_eq!(
            snapshot.as_ref().map(|(_, right)| right),
            Some(&TokenHistogram::from_tokens(shifted_right)?)
        );
        assert_eq!(shifted_left.len(), 6_500);
        assert_eq!(shifted_right.len(), 5_500);
        assert_eq!(
            shifted_left
                .iter()
                .map(|token| token.input_len())
                .sum::<usize>(),
            6_500
        );
        assert_eq!(
            shifted_right
                .iter()
                .map(|token| token.input_len())
                .sum::<usize>(),
            5_500
        );
        assert!(shifted_left[6_000..]
            .iter()
            .all(|token| *token == ParseToken::Literal(b'a')));
        Ok(())
    }
}

#[cfg(test)]
mod encoded_tests {
    use super::*;
    use crate::{
        compress_bound,
        inflate::{decompress_slice, InflateConfig},
        ReturnCode,
    };

    fn assert_round_trip(input: &[u8], optimization_passes: usize) {
        let mut compressed = vec![0u8; compress_bound(input.len())];
        let (compressed, code) =
            super::super::compress_slice_near_optimal(&mut compressed, input, optimization_passes);
        assert_eq!(
            code,
            ReturnCode::Ok,
            "near-optimal compression failed for {} input bytes",
            input.len()
        );

        let mut decoded = vec![0u8; input.len()];
        let (decoded, code) = decompress_slice(&mut decoded, compressed, InflateConfig::default());
        assert_eq!(
            code,
            ReturnCode::Ok,
            "near-optimal decompression failed for {} input bytes",
            input.len()
        );
        assert_eq!(
            decoded,
            input,
            "round-trip mismatch for {} bytes",
            input.len()
        );
    }

    #[test]
    fn near_optimal_compressor_round_trips_with_optimized_costs() {
        let input = b"abcabcabc---abcabcabc---the quick brown fox---abcabcabcabcabc";
        assert_round_trip(input, 1);
    }

    #[test]
    fn additional_optimization_pass_never_enlarges_single_block_stream() -> Result<(), ReturnCode> {
        // Keep this below twice MIN_BLOCK_LENGTH so adaptive partitioning cannot change the block
        // layout between runs. For a fixed range, the optimization loop retains the cheapest exact
        // production-scored path seen so far, so an additional search must not enlarge the stream.
        let input = super::tests::two_distribution_input()[..8_000].to_vec();
        let ranges = choose_block_ranges(&input, 32_506)?;
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0], 0..input.len());

        let mut previous_len = usize::MAX;
        for passes in 1..=4 {
            let mut buffer = vec![0u8; compress_bound(input.len())];
            let (encoded, code) =
                super::super::compress_slice_near_optimal(&mut buffer, &input, passes);
            assert_eq!(code, ReturnCode::Ok);
            assert!(
                encoded.len() <= previous_len,
                "pass count {passes}: {} > {previous_len}",
                encoded.len(),
            );
            previous_len = encoded.len();
        }
        Ok(())
    }

    #[test]
    fn long_huffman_block_round_trips_beyond_stored_limit() -> Result<(), ReturnCode> {
        let pattern = b"abcdefghijklmnopqrstuvwxyz012345";
        let input = (0..120_000)
            .map(|index| pattern[index % pattern.len()])
            .collect::<Vec<_>>();
        assert_eq!(
            choose_block_ranges(&input, 32_768)?,
            [0..60_000, 60_000..120_000]
        );

        assert_round_trip(&input, 1);
        Ok(())
    }

    #[test]
    fn adaptive_multi_block_stream_round_trips() -> Result<(), ReturnCode> {
        let input = super::tests::two_distribution_input();
        let blocks = choose_block_ranges(&input, 32_506)?;
        assert!(
            blocks.len() >= 2,
            "test input must exercise multiple blocks"
        );

        assert_round_trip(&input, 1);
        assert_round_trip(&input, 3);
        Ok(())
    }

    #[test]
    fn full_window_distance_round_trips() {
        let mut state = 0x94d0_49bb_1331_11ebu64;
        let mut input = vec![0u8; 32_768];
        for byte in &mut input {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *byte = state.to_le_bytes()[4];
        }
        let repeated = input[..STD_MAX_MATCH].to_vec();
        input.extend_from_slice(&repeated);

        assert_round_trip(&input, 1);
    }

    #[test]
    fn every_tiny_input_length_round_trips() {
        for len in 0..=STD_MAX_MATCH + 1 {
            let input = (0..len)
                .map(|index| (index.wrapping_mul(73) ^ (index >> 3)).to_le_bytes()[0])
                .collect::<Vec<_>>();
            assert_round_trip(&input, 0);
        }
    }

    #[test]
    fn block_and_stored_length_boundaries_round_trip() {
        for len in [
            MAX_COARSE_BLOCK_INPUT,
            MAX_COARSE_BLOCK_INPUT + 1,
            super::super::MAX_STORED,
            super::super::MAX_STORED + 1,
        ] {
            let input = vec![0x5a; len];
            assert_round_trip(&input, 0);
        }
    }

    #[test]
    fn too_small_for_any_zlib_stream_does_not_touch_output() {
        let input = vec![0x5a; 240_000];
        for passes in [0, 1, 3] {
            for size in 0..8 {
                let mut output = vec![0xa5; size];
                let (encoded, code) =
                    super::super::compress_slice_near_optimal(&mut output, &input, passes);
                assert_eq!(code, ReturnCode::BufError, "size={size} passes={passes}");
                assert_eq!(encoded, &[]);
                assert!(output.iter().all(|&byte| byte == 0xa5));
            }
        }

        // The eight-byte lower bound is tight: an empty zlib stream fits.
        let mut minimum = [0u8; 8];
        let (encoded, code) = super::super::compress_slice_near_optimal(&mut minimum, b"", 0);
        assert_eq!(code, ReturnCode::Ok);
        assert_eq!(encoded.len(), minimum.len());
    }

    #[test]
    fn insufficient_output_only_writes_a_valid_stream_prefix() {
        let input = b"abcabcabc---abcabcabc---the quick brown fox---abcabcabcabcabc";
        let mut full_buffer = vec![0u8; compress_bound(input.len())];
        let (full, code) = super::super::compress_slice_near_optimal(&mut full_buffer, input, 0);
        assert_eq!(code, ReturnCode::Ok);
        let full = full.to_vec();

        for cutoff in 0..full.len() {
            // Complementing the expected byte makes untouched storage distinguishable from a byte
            // that was actually emitted, even when the compressed stream contains zeroes.
            let untouched = full[..cutoff].iter().map(|byte| !byte).collect::<Vec<_>>();
            let mut output = untouched.clone();
            let (compressed, code) =
                super::super::compress_slice_near_optimal(&mut output, input, 0);
            assert_eq!(code, ReturnCode::BufError, "cutoff={cutoff}");
            assert!(compressed.is_empty(), "cutoff={cutoff}");

            let written = output
                .iter()
                .zip(&full)
                .take_while(|(actual, expected)| actual == expected)
                .count();
            assert_eq!(
                &output[written..],
                &untouched[written..],
                "non-prefix write at cutoff={cutoff}"
            );
        }
    }
}
