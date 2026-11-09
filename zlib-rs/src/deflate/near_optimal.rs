#![forbid(unsafe_code)]

use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;

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

/// Sparse whole-input candidate table in CSR form.
///
/// `offsets[position]..offsets[position + 1]` indexes the alternatives for one input byte. Entries
/// are ordered by distance code, and there is at most one entry for each of DEFLATE's 30 distance
/// symbols. This boundary is rich enough for arbitrary Huffman cost models without forcing ordinary
/// Fast/Medium/Slow parsing to consume more than its current single-best projection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatchCandidates {
    // CSR offsets count candidate *entries*, not input bytes. A position may
    // contribute up to D_CODES entries, so an input fitting in a u32 index can
    // still produce more than u32::MAX entries. Keep the offsets in the same
    // index domain as the backing Vec instead of silently wrapping them.
    offsets: Vec<usize>,
    entries: Vec<MatchCandidate>,
}

impl MatchCandidates {
    pub(crate) fn for_position(&self, position: usize) -> &[MatchCandidate] {
        let start = self.offsets[position];
        let end = self.offsets[position + 1];
        &self.entries[start..end]
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
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
    fn empty() -> Self {
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

    fn match_cost(&self, length: usize, distance: usize) -> u32 {
        debug_assert!((STD_MIN_MATCH..=STD_MAX_MATCH).contains(&length));
        debug_assert!((1..=1usize << crate::MAX_WBITS).contains(&distance));

        let length_code = usize::from(trees_tbl::LENGTH_CODE[length - STD_MIN_MATCH]);
        let distance_code = usize::from(State::d_code(distance - 1));
        self.length_symbol_cost(length_code) + self.distance_symbol_cost(distance_code)
    }

    /// One token fits u32; whole-parse dynamic-programming totals use u64.
    fn token_cost(&self, token: ParseToken) -> u32 {
        match token {
            ParseToken::Literal(byte) => self.literal_cost(byte),
            ParseToken::Match { length, distance } => {
                self.match_cost(usize::from(length), usize::from(distance))
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

#[derive(Clone, Copy, Debug)]
struct ParseStep(u32);

impl ParseStep {
    const MATCH_FLAG: u32 = 1 << 31;

    #[inline]
    fn new(token: ParseToken) -> Self {
        Self(match token {
            ParseToken::Literal(byte) => u32::from(byte),
            ParseToken::Match { length, distance } => {
                debug_assert!((STD_MIN_MATCH..=STD_MAX_MATCH).contains(&usize::from(length)));
                debug_assert_ne!(distance, 0);
                Self::MATCH_FLAG | (u32::from(length) << 16) | u32::from(distance)
            }
        })
    }

    #[inline]
    fn token(self) -> ParseToken {
        if self.0 & Self::MATCH_FLAG == 0 {
            ParseToken::Literal(self.0 as u8)
        } else {
            ParseToken::Match {
                length: ((self.0 >> 16) & 0x1ff) as u16,
                distance: self.0 as u16,
            }
        }
    }
}

/// Build the exact useful-match frontier for a complete input.
///
/// Suffix-array ranks make this exact without walking every equal-prefix predecessor. For each DEFLATE
/// distance symbol, a sliding ordered set contains exactly the suffix ranks currently inside that
/// symbol's distance interval. The lexicographic predecessor or successor of the current suffix
/// maximizes LCP over that set, so two probes recover the longest match for the whole distance class.
/// Retaining that one maximum per class is lossless for arbitrary DEFLATE Huffman code lengths.
pub fn build_match_candidates(input: &[u8], max_distance: usize) -> MatchCandidates {
    const DIRECT_SCAN_MAX: usize = 8;
    // The parser emits DEFLATE tokens, not arbitrary 16-bit offsets.
    assert!(
        max_distance <= 1usize << crate::MAX_WBITS,
        "candidate history exceeds the DEFLATE window"
    );
    // SuffixRanks checks that its compact u32 backing can address the input.
    // Candidate entry offsets remain usize because there may be D_CODES entries per byte.
    let suffixes = SuffixRanks::new(input);
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

    MatchCandidates { offsets, entries }
}

/// Dense ordered set over suffix ranks. Each level is a bitset whose set bits identify non-empty
/// words in the level below it, so predecessor/successor queries skip arbitrarily large empty rank
/// intervals without pointer chasing or per-entry allocation.
struct RankSet {
    levels: Vec<Vec<u64>>,
}

impl RankSet {
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

    #[inline]
    fn neighbors(&self, rank: usize) -> (Option<usize>, Option<usize>) {
        (self.predecessor(rank), self.successor(rank))
    }

    #[inline]
    fn predecessor(&self, rank: usize) -> Option<usize> {
        if rank == 0 {
            return None;
        }
        let word_index = (rank - 1) / 64;
        let bits = (rank - 1) % 64 + 1;
        let word = self.levels[0][word_index] & low_bits(bits);
        if word != 0 {
            return Some(word_index * 64 + highest_bit(word));
        }
        let previous_word = self.previous_set_bit(1, word_index)?;
        let word = self.levels[0][previous_word];
        debug_assert_ne!(word, 0);
        Some(previous_word * 64 + highest_bit(word))
    }

    #[inline]
    fn successor(&self, rank: usize) -> Option<usize> {
        let start = rank.checked_add(1)?;
        let word_index = start / 64;
        let &raw_word = self.levels[0].get(word_index)?;
        let word = raw_word & !low_bits(start % 64);
        if word != 0 {
            return Some(word_index * 64 + word.trailing_zeros() as usize);
        }
        let next_word = self.next_set_bit(1, word_index + 1)?;
        let word = self.levels[0][next_word];
        debug_assert_ne!(word, 0);
        Some(next_word * 64 + word.trailing_zeros() as usize)
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
fn low_bits(bits: usize) -> u64 {
    if bits == 0 {
        0
    } else if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

#[inline]
fn highest_bit(word: u64) -> usize {
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
/// consumers operate on ordinary slice indices through rank_of/position_at.
/// This is the only boundary converting the compact rank/position storage.
struct SuffixRanks {
    sa: Vec<u32>,
    rank: Vec<u32>,
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

    fn new(input: &[u8]) -> Self {
        if input.is_empty() {
            return Self {
                sa: Vec::new(),
                rank: Vec::new(),
            };
        }
        let length = u32::try_from(input.len())
            .unwrap_or_else(|_| panic!("input exceeds the suffix-rank index range"));

        let mut sa = (0..length).collect::<Vec<_>>();
        let mut scratch = vec![0u32; input.len()];
        let mut rank = input
            .iter()
            .map(|&byte| u32::from(byte) + 1)
            .collect::<Vec<_>>();
        let mut next_rank = vec![0u32; input.len()];
        let mut max_rank = rank.iter().copied().max().unwrap_or(0);
        let mut counts = vec![0u32; input.len().max(256) + 1];
        let mut span = 1usize;

        loop {
            // Prefix-doubling ranks are dense integers after the first round. Sort the pair
            // (rank[pos], rank[pos + span]) with two stable counting passes instead of doing a
            // comparison sort of all suffixes at every doubling step.
            counting_sort_suffixes(&sa, &mut scratch, max_rank, &mut counts, |position| {
                suffix_second_rank(&rank, position, span)
            });
            counting_sort_suffixes(&scratch, &mut sa, max_rank, &mut counts, |position| {
                rank[suffix_index(position)]
            });

            let mut class = 1u32;
            next_rank[suffix_index(sa[0])] = class;
            for pair in sa.windows(2) {
                let left = pair[0];
                let right = pair[1];
                if rank[suffix_index(left)] != rank[suffix_index(right)]
                    || suffix_second_rank(&rank, left, span)
                        != suffix_second_rank(&rank, right, span)
                {
                    class += 1;
                }
                next_rank[suffix_index(right)] = class;
            }

            core::mem::swap(&mut rank, &mut next_rank);
            if class == length {
                break;
            }
            max_rank = class;
            span = span
                .checked_mul(2)
                .expect("suffix-array doubling span overflow");
        }

        // The doubling ranks are 1-based equivalence classes. The parser wants direct indices into
        // `sa`, so materialize the exact inverse suffix-array permutation once.
        for (suffix_rank, &position) in (0..length).zip(&sa) {
            rank[suffix_index(position)] = suffix_rank;
        }

        Self { sa, rank }
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
fn suffix_second_rank(rank: &[u32], position: u32, span: usize) -> u32 {
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
fn choose_min_match_len(used_literals: usize) -> usize {
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
    fn keep_recent_as_new_block(&mut self) {
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
pub fn choose_block_ranges(input: &[u8], max_distance: usize) -> Vec<Range<usize>> {
    if input.is_empty() {
        return core::iter::once(0..0).collect();
    }

    const HASH_SIZE: usize = 1 << 16;
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
    debug_assert!(!blocks.is_empty());
    debug_assert_eq!(blocks[0].start, 0);
    debug_assert_eq!(blocks.last().unwrap().end, input.len());
    debug_assert!(blocks.windows(2).all(|pair| pair[0].end == pair[1].start));
    debug_assert!(blocks
        .iter()
        .all(|block| block.len() <= MAX_COARSE_BLOCK_INPUT));
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
pub fn parse_block(input: &[u8], candidates: &MatchCandidates, costs: &TokenCosts) -> ParseResult {
    parse_range(input, candidates, 0..input.len(), costs)
}

/// Reuse the block-sized dynamic-programming arrays across cost-model passes;
/// each completed parse still owns its token output independently.
#[derive(Default)]
struct ParseScratch {
    best_cost: Vec<u64>,
    steps: Vec<ParseStep>,
}

impl ParseScratch {
    const INF: u64 = u64::MAX / 4;

    fn reset(&mut self, block_len: usize) {
        assert!(block_len < usize::MAX, "DP block has too many positions");
        let slots = block_len + 1;
        // clear() retains each Vec's allocation; resize() then initializes
        // every slot exactly once, including after a shorter previous parse.
        self.best_cost.clear();
        self.best_cost.resize(slots, Self::INF);
        self.best_cost[0] = 0;
        self.steps.clear();
        self.steps.resize(slots, ParseStep(0));
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
    assert_eq!(candidates.len_for_parse(), input.len());
    assert!(block.start <= block.end && block.end <= input.len());

    let block_len = block.len();
    scratch.reset(block_len);
    let (best_cost, steps) = (&mut scratch.best_cost, &mut scratch.steps);
    const INF: u64 = ParseScratch::INF;
    let mut length_costs = [0u32; STD_MAX_MATCH + 1];
    for (length, cost) in length_costs.iter_mut().enumerate().skip(STD_MIN_MATCH) {
        let code = usize::from(trees_tbl::LENGTH_CODE[length - STD_MIN_MATCH]);
        *cost = costs.length_symbol_cost(code);
    }
    // DEFLATE distance cost depends on the Huffman distance symbol, not the
    // numeric distance. Build 30 entries rather than allocating and filling
    // one entry per possible byte offset in the 32 KiB window.
    let distance_costs: [u32; D_CODES] =
        core::array::from_fn(|code| costs.distance_symbol_cost(code));
    // One candidate can become newly capable at each maximum match length. Keep the sparse values
    // in a generation-tagged table instead of clearing a 259-entry Option array at every input byte.
    // Generation zero is never live, so the all-zero initial table needs no per-position clearing.
    #[derive(Clone, Copy)]
    struct DistanceAtMax {
        generation: u32,
        cost: u32,
        distance: u16,
    }
    assert!(block_len <= u32::MAX as usize);
    let mut distance_cost_at_max = [DistanceAtMax {
        generation: 0,
        cost: 0,
        distance: 0,
    }; STD_MAX_MATCH + 1];
    for local_position in 0..block_len {
        let position = block.start + local_position;
        let prefix_cost = best_cost[local_position];
        debug_assert_ne!(prefix_cost, INF);

        relax(
            best_cost.as_mut_slice(),
            steps.as_mut_slice(),
            local_position + 1,
            prefix_cost + u64::from(costs.literal_cost(input[position])),
            ParseToken::Literal(input[position]),
        );

        let available = STD_MAX_MATCH.min(block.end - position);
        if available < STD_MIN_MATCH {
            continue;
        }

        let generation = (local_position + 1) as u32;
        let mut longest_match = 0usize;
        for candidate in candidates.for_position(position) {
            let max_len = usize::from(candidate.length).min(available);
            longest_match = longest_match.max(max_len);
            let code = usize::from(State::d_code(usize::from(candidate.distance) - 1));
            let cost = distance_costs[code];
            let slot = &mut distance_cost_at_max[max_len];
            if slot.generation != generation || cost < slot.cost {
                *slot = DistanceAtMax {
                    generation,
                    cost,
                    distance: candidate.distance,
                };
            }
        }

        if longest_match < STD_MIN_MATCH {
            continue;
        }
        let mut cheapest_capable = None::<(u32, u16)>;
        for length in (STD_MIN_MATCH..=longest_match).rev() {
            let candidate = distance_cost_at_max[length];
            if candidate.generation == generation
                && cheapest_capable.is_none_or(|(old, _)| candidate.cost < old)
            {
                cheapest_capable = Some((candidate.cost, candidate.distance));
            }
            let Some((distance_cost, distance)) = cheapest_capable else {
                continue;
            };
            let token_cost = u64::from(length_costs[length]) + u64::from(distance_cost);
            relax(
                best_cost.as_mut_slice(),
                steps.as_mut_slice(),
                local_position + length,
                prefix_cost + token_cost,
                ParseToken::matching(length, distance),
            );
        }
    }

    let mut tokens = Vec::new();
    let mut cursor = block_len;
    while cursor != 0 {
        let token = steps[cursor].token();
        tokens.push(token);
        cursor -= token.input_len();
    }
    tokens.reverse();

    debug_assert_eq!(
        tokens.iter().map(|token| token.input_len()).sum::<usize>(),
        block_len
    );
    ParseResult {
        cost_x16: best_cost[block_len],
        tokens,
    }
}

impl MatchCandidates {
    fn len_for_parse(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }
}

#[inline]
fn relax(
    best_cost: &mut [u64],
    steps: &mut [ParseStep],
    endpoint: usize,
    cost: u64,
    token: ParseToken,
) {
    if cost < best_cost[endpoint] {
        best_cost[endpoint] = cost;
        steps[endpoint] = ParseStep::new(token);
    }
}

#[cfg(test)]
fn common_prefix(left: &[u8], right: &[u8]) -> usize {
    match_compare::common_prefix_len(left, right)
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn suffix_ranks_preserve_lexicographic_order() {
        let mut state = 0x243f_6a88_85a3_08d3u64;
        let mut input = vec![0u8; 32 * 1024];
        for byte in &mut input {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *byte = ((state >> 32) as u8) & 0x1f;
        }

        let suffixes = SuffixRanks::new(&input);
        for rank in 0..input.len() {
            let position = suffixes.position_at(rank);
            assert_eq!(suffixes.rank_of(position), rank);
        }
        for pair in suffixes.sa.windows(2) {
            assert!(input[suffix_index(pair[0])..] < input[suffix_index(pair[1])..]);
        }
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
    fn suffix_ranks_invert_repetitive_and_boundary_length_inputs() {
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

                let suffixes = SuffixRanks::new(&input);
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
    }

    #[test]
    fn packed_parse_step_round_trips_tokens() {
        for &token in &[
            ParseToken::Literal(0),
            ParseToken::Literal(u8::MAX),
            ParseToken::Match {
                length: STD_MIN_MATCH as u16,
                distance: 1,
            },
            ParseToken::Match {
                length: STD_MAX_MATCH as u16,
                distance: 32_768,
            },
        ] {
            let step = ParseStep::new(token);
            assert_eq!(step.token(), token);
        }
        assert_eq!(core::mem::size_of::<ParseStep>(), 4);
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
    #[should_panic(expected = "candidate history exceeds the DEFLATE window")]
    fn exact_frontier_rejects_distances_outside_deflate_window() {
        // A u16 can encode more than the 32 KiB permitted by DEFLATE.
        // Reject that input at the candidate boundary, not when the token
        // eventually reaches the Huffman writer.
        let _ = build_match_candidates(b"aaaaaaa", (1usize << crate::MAX_WBITS) + 1);
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
    fn exact_frontier_reaches_full_deflate_window() {
        let mut state = 0x94d0_49bb_1331_11ebu64;
        let mut input = vec![0u8; 32_768];
        for byte in &mut input {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *byte = (state >> 32) as u8;
        }
        let repeated = input[..STD_MAX_MATCH].to_vec();
        input.extend_from_slice(&repeated);

        let table = build_match_candidates(&input, 32_768);
        assert!(table.for_position(32_768).iter().any(|candidate| {
            usize::from(candidate.distance) == 32_768
                && usize::from(candidate.length) == STD_MAX_MATCH
        }));
    }

    #[test]
    fn exact_frontier_does_not_cross_deflate_window() {
        let mut state = 0x94d0_49bb_1331_11ebu64;
        let mut input = vec![0u8; 32_769];
        for byte in &mut input {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *byte = (state >> 32) as u8;
        }
        let repeated = input[..STD_MAX_MATCH].to_vec();
        assert_ne!(&input[1..=STD_MAX_MATCH], repeated.as_slice());
        input.extend_from_slice(&repeated);

        let table = build_match_candidates(&input, 32_768);
        assert!(table
            .for_position(32_769)
            .iter()
            .all(|candidate| usize::from(candidate.length) < STD_MAX_MATCH));
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
                for max_distance in [1usize, 2, 3, 7, 16, 31, 50, 64] {
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
        assert!(blocks
            .iter()
            .all(|block| block.len() <= MAX_COARSE_BLOCK_INPUT));
    }

    #[test]
    fn splitter_soft_limit_never_creates_an_illegal_or_tiny_tail() {
        let input = vec![0x5a; 150_000];
        let blocks = choose_block_ranges(&input, 32_506);
        assert_eq!(blocks[0].start, 0);
        assert_eq!(blocks.last().unwrap().end, input.len());
        assert!(blocks
            .iter()
            .all(|block| block.len() <= MAX_COARSE_BLOCK_INPUT));
        assert!(blocks
            .iter()
            .take(blocks.len() - 1)
            .all(|block| block.len() >= MIN_BLOCK_LENGTH));
        assert!(blocks.last().unwrap().len() >= MIN_BLOCK_LENGTH);
    }

    #[test]
    fn splitter_soft_limit_boundary_matrix_is_exact() {
        for (len, expected) in [
            (0, vec![0..0]),
            (59_999, vec![0..59_999]),
            (60_000, vec![0..60_000]),
            (60_001, vec![0..60_001]),
            (64_999, vec![0..64_999]),
            (65_000, vec![0..60_000, 60_000..65_000]),
            (65_001, vec![0..60_000, 60_000..65_001]),
            (119_999, vec![0..60_000, 60_000..119_999]),
            (120_000, vec![0..60_000, 60_000..120_000]),
        ] {
            let input = vec![0x5a; len];
            assert_eq!(choose_block_ranges(&input, 32_768), expected, "len={len}");
        }
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
        let parsed = parse_range(&input, &table, 6_000..12_000, &TokenCosts::fixed());
        assert!(matches!(
            parsed.tokens.first(),
            Some(ParseToken::Match { length, distance })
                if *length >= 32 && *distance == 6_000
        ));
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
    fn minimum_cost_parse_matches_exhaustive_oracle() {
        let input = b"abcabcabcxabcaabcabcabc---abcabc";
        let table = build_match_candidates(input, 32);
        let fixed = TokenCosts::fixed();
        let parsed = parse_block(input, &table, &fixed);
        assert_eq!(parsed.cost_x16, brute_cost(input, &fixed, 32));
        assert_eq!(expand(&parsed.tokens), input);
    }

    #[test]
    fn maximal_symbol_weights_keep_dp_exact_without_u16_overflow() {
        // A cost model may assign u16::MAX to a Huffman symbol. Extra-bit
        // costs are then added before DP comparison and must not wrap u16,
        // including for distance symbols spanning the full DEFLATE window.
        let input = vec![b'a'; 96];
        let candidates = build_match_candidates(&input, 64);
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
    }

    #[test]
    fn reused_parse_scratch_resets_across_ranges_and_cost_models() {
        let input = b"abcabcabc---abcabcabc---the quick brown fox---abcabcabcabcabc";
        let candidates = build_match_candidates(input, 32);
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
            assert_eq!(scratch.best_cost.len(), range.len() + 1);
            assert_eq!(scratch.steps.len(), range.len() + 1);
            if let Some((cost_capacity, step_capacity)) = capacity {
                assert_eq!(scratch.best_cost.capacity(), cost_capacity);
                assert_eq!(scratch.steps.capacity(), step_capacity);
            } else {
                capacity = Some((scratch.best_cost.capacity(), scratch.steps.capacity()));
            }
        }
    }

    #[test]
    fn distance_code_reduction_is_valid_for_non_monotone_huffman_costs() {
        let input = b"aaaaaaaaabaaaaaaaaacaaaaaaaaadaaaaaaaaa";
        let table = build_match_candidates(input, 32);
        let mut costs = TokenCosts::fixed();
        // Deliberately make farther distance symbols cheaper than nearby ones. Keeping the longest
        // match per distance *symbol* must remain exact even when distance costs are non-monotone.
        for (code, cost) in costs.distance.iter_mut().enumerate() {
            *cost = (15 - (code % 15) as u16) * COST_SCALE;
        }
        for (byte, cost) in costs.literal.iter_mut().enumerate() {
            *cost = (3 + (byte % 11) as u16) * COST_SCALE;
        }

        let parsed = parse_block(input, &table, &costs);
        assert_eq!(parsed.cost_x16, brute_cost(input, &costs, 32));
        assert_eq!(expand(&parsed.tokens), input);
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

#[derive(Clone, Copy, Debug)]
struct BlockScore {
    bits: usize,
    encoded_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TokenHistogram {
    literal_length: [u16; L_CODES],
    distance: [u16; D_CODES],
}

impl TokenHistogram {
    fn empty() -> Self {
        let mut literal_length = [0; L_CODES];
        literal_length[256] = 1;
        Self {
            literal_length,
            distance: [0; D_CODES],
        }
    }

    fn from_tokens(tokens: &[ParseToken]) -> Self {
        assert!(
            tokens.len() <= MAX_BLOCK_TOKENS,
            "near-optimal token block exceeds symbol-buffer capacity"
        );
        let mut histogram = Self::empty();
        for &token in tokens {
            histogram.add_token(token);
        }
        histogram
    }

    fn add_token(&mut self, token: ParseToken) {
        let (literal_length, distance) = Self::token_bins(token);
        self.literal_length[literal_length] = self.literal_length[literal_length]
            .checked_add(1)
            .expect("near-optimal block symbol frequency fits u16");
        if let Some(distance) = distance {
            self.distance[distance] = self.distance[distance]
                .checked_add(1)
                .expect("near-optimal block distance frequency fits u16");
        }
    }

    fn remove_token(&mut self, token: ParseToken) {
        let (literal_length, distance) = Self::token_bins(token);
        debug_assert_ne!(self.literal_length[literal_length], 0);
        self.literal_length[literal_length] -= 1;
        if let Some(distance) = distance {
            debug_assert_ne!(self.distance[distance], 0);
            self.distance[distance] -= 1;
        }
    }

    fn token_bins(token: ParseToken) -> (usize, Option<usize>) {
        match token {
            ParseToken::Literal(byte) => (usize::from(byte), None),
            ParseToken::Match { length, distance } => {
                let length_code =
                    trees_tbl::LENGTH_CODE[usize::from(length) - STD_MIN_MATCH] as usize;
                let distance_code = State::d_code(usize::from(distance) - 1) as usize;
                (257 + length_code, Some(distance_code))
            }
        }
    }

    fn merged(left: &Self, right: &Self) -> Self {
        let mut merged = left.clone();
        for index in 0..L_CODES {
            if index == 256 {
                continue;
            }
            merged.literal_length[index] = merged.literal_length[index]
                .checked_add(right.literal_length[index])
                .expect("merged near-optimal block symbol frequency fits u16");
        }
        for index in 0..D_CODES {
            merged.distance[index] = merged.distance[index]
                .checked_add(right.distance[index])
                .expect("merged near-optimal block distance frequency fits u16");
        }
        merged
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

pub struct PlannedBlockSource {
    pub(crate) block: PlannedBlock,
    model: ParseModel,
    context_start: usize,
    context_end: usize,
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

impl PlannedBlockSource {
    pub(crate) fn into_parts(self) -> (PlannedBlock, ParseModel) {
        (self.block, self.model)
    }

    pub(crate) fn refresh_model(&mut self, state: &mut super::State<'_>) {
        self.model = model_from_tokens(state, &self.block.tokens);
    }

    /// Re-run only the minimum-cost parser after a boundary change. The exact match frontier built
    /// for the original right-hand block already covers up to `max_distance` bytes of preceding
    /// history, so most leftward shifts and every rightward shift need no new match discovery.
    pub(crate) fn reparse_with_existing_candidates(
        &mut self,
        state: &mut super::State<'_>,
        input: &[u8],
        optimization_passes: usize,
        previous_model: Option<&ParseModel>,
    ) -> bool {
        if self.block.range.start < self.context_start || self.block.range.end != self.context_end {
            self.refresh_model(state);
            return false;
        }

        if optimization_passes == 0 {
            self.refresh_model(state);
            return false;
        }

        let context = &input[self.context_start..self.context_end];
        let local_range = (self.block.range.start - self.context_start)..context.len();
        let (_, mut best_score) = costs_and_block_score(state, &self.block.tokens);
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
            let converged = previous.converged(best_tokens.as_deref(), &reparsed.tokens);
            let (next_costs, reparsed_score) = costs_and_block_score(state, &reparsed.tokens);
            if reparsed_score.bits < best_score.bits {
                best_score = reparsed_score;
                best_tokens = Some(reparsed.tokens);
                previous = PreviousParse::Best;
            } else {
                previous = PreviousParse::Other(reparsed.tokens);
            }
            if converged {
                break;
            }
            costs = next_costs;
        }

        let changed = if let Some(tokens) = best_tokens {
            self.block.tokens = tokens;
            true
        } else {
            false
        };
        self.refresh_model(state);
        debug_assert_eq!(self.block.token_input_len(), self.block.range.len());
        changed
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
    integer as u16 * COST_SCALE + fraction
}

#[inline]
fn log2_ratio_x16(numerator: usize, denominator: usize) -> u16 {
    debug_assert!(numerator >= denominator && denominator != 0);
    // Represent the rational with 48 fractional bits before extracting four binary log2 fraction
    // bits. The inputs here are tiny DEFLATE alphabet sizes; Q48 makes the approximation error far
    // below a 1/16-bit boundary while preserving exact powers of two.
    const Q_BITS: u32 = 48;
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
    ((integer - Q_BITS) as u16) * COST_SCALE + fraction
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
        1 => (default + previous) / 2,
        2 => (5 * default + 3 * previous) / 8,
        _ => (3 * default + previous) / 4,
    };
    blended as u16
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
    } else if 2 * total_delta > cutoff {
        1
    } else {
        0
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
) -> (ParseResult, ParseModel) {
    if optimization_passes == 0 {
        let parsed = parse_range(input, candidates, block, &TokenCosts::fixed());
        let (costs, _) = costs_and_block_score(state, &parsed.tokens);
        let model = ParseModel {
            costs,
            observations: BlockObservations::from_tokens(&parsed.tokens),
        };
        return (parsed, model);
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
        let converged =
            previous.converged(best.as_ref().map(|p| p.tokens.as_slice()), &next.tokens);
        let (next_costs, next_score) = costs_and_block_score(state, &next.tokens);
        if next_score.bits < best_score {
            best_score = next_score.bits;
            best_costs = Some(next_costs.clone());
            best = Some(next);
            previous = PreviousParse::Best;
        } else {
            previous = PreviousParse::Other(next.tokens);
        }
        if converged {
            break;
        }
        costs = next_costs;
    }

    let best = best.expect("positive optimization pass count always evaluates one parse");
    let model = ParseModel {
        costs: best_costs.expect("best parse always records its learned costs"),
        observations: BlockObservations::from_tokens(&best.tokens),
    };
    (best, model)
}

pub fn model_from_tokens(state: &mut super::State<'_>, tokens: &[ParseToken]) -> ParseModel {
    let (costs, _) = costs_and_block_score(state, tokens);
    ParseModel {
        costs,
        observations: BlockObservations::from_tokens(tokens),
    }
}

pub fn tally_tokens(state: &mut super::State<'_>, tokens: &[ParseToken]) {
    assert!(
        tokens.len() <= MAX_BLOCK_TOKENS,
        "near-optimal token block exceeds symbol-buffer capacity"
    );
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
}

fn costs_and_block_score(
    state: &mut super::State<'_>,
    tokens: &[ParseToken],
) -> (TokenCosts, BlockScore) {
    let histogram = TokenHistogram::from_tokens(tokens);
    costs_and_block_score_from_histogram(state, &histogram)
}

fn costs_and_block_score_from_histogram(
    state: &mut super::State<'_>,
    histogram: &TokenHistogram,
) -> (TokenCosts, BlockScore) {
    histogram.apply_to_state(state);

    let literal_used =
        core::array::from_fn::<_, 256, _>(|index| histogram.literal_length[index] != 0);
    let length_used = core::array::from_fn::<_, LENGTH_CODES, _>(|index| {
        histogram.literal_length[257 + index] != 0
    });
    let distance_used =
        core::array::from_fn::<_, D_CODES, _>(|index| histogram.distance[index] != 0);

    // Build directly in the state-owned descriptors. Only the bit-cost
    // counters are shared across trees; no temporary descriptor swap is needed.
    super::build_tree(&mut state.l_desc, &mut state.opt_len, &mut state.static_len);
    super::build_tree(&mut state.d_desc, &mut state.opt_len, &mut state.static_len);

    // Keep unseen symbols available to later passes, but price them high enough that a pass does not
    // churn the tree merely because fixed-Huffman fallback made an absent symbol look cheap.
    let mut costs = TokenCosts {
        literal: [13 * COST_SCALE; 256],
        length: [13 * COST_SCALE; LENGTH_CODES],
        distance: [10 * COST_SCALE; D_CODES],
    };
    for (index, &used) in literal_used.iter().enumerate() {
        if used {
            costs.literal[index] = state.l_desc.dyn_tree[index].len() * COST_SCALE;
        }
    }
    for (index, &used) in length_used.iter().enumerate() {
        if used {
            costs.length[index] = state.l_desc.dyn_tree[257 + index].len() * COST_SCALE;
        }
    }
    for (index, &used) in distance_used.iter().enumerate() {
        if used {
            costs.distance[index] = state.d_desc.dyn_tree[index].len() * COST_SCALE;
        }
    }

    // Add the exact dynamic-tree representation cost. Mirror the production block chooser: it
    // rounds static/dynamic candidates to bytes first and prefers static on a rounded tie, then emits
    // the selected representation without byte-aligning it. Stored-block choice is parse-independent
    // and therefore cannot make a worse token parse preferable to a better one.
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
    (costs, score)
}

fn compressed_histogram_score(
    state: &mut super::State<'_>,
    histogram: &TokenHistogram,
    input_len: usize,
) -> Option<(TokenCosts, BlockScore)> {
    let (costs, score) = costs_and_block_score_from_histogram(state, histogram);
    // Stored blocks byte-align before their payload, so their exact cost depends on the preceding
    // block's residual bit count. Boundary optimization is intentionally limited to pairs for which
    // production would choose a Huffman-coded representation; stored-block regions keep the cheap
    // splitter's boundary unchanged.
    (input_len > super::MAX_STORED || input_len + 4 > score.encoded_bytes).then_some((costs, score))
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
    fn is_near(self, other: Self) -> bool {
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

    fn boundary(self, offset: usize) -> Option<usize> {
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

fn consider_shift_candidate(
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

fn fixed_cost_shift_candidates(
    left: &PlannedBlock,
    right: &PlannedBlock,
    left_costs: &TokenCosts,
    right_costs: &TokenCosts,
) -> Vec<BoundaryCandidate> {
    debug_assert_eq!(left.range.end, right.range.start);
    let mut candidates = Vec::with_capacity(EXACT_BOUNDARY_CANDIDATES);

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
        delta += i64::from(left_costs.token_cost(token)) - i64::from(right_costs.token_cost(token));
        consider_shift_candidate(&mut candidates, delta, BoundaryShift::Right(moved));
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
        delta += i64::from(right_costs.token_cost(token)) - i64::from(left_costs.token_cost(token));
        consider_shift_candidate(&mut candidates, delta, BoundaryShift::Left(moved));
    }

    candidates.sort_unstable_by_key(|candidate| candidate.delta);
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
) -> [Option<(TokenHistogram, TokenHistogram)>; EXACT_BOUNDARY_CANDIDATES] {
    assert!(candidates.len() <= EXACT_BOUNDARY_CANDIDATES);
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
                left_counts.add_token(token);
                right_counts.remove_token(token);
            } else {
                left_counts.remove_token(token);
                right_counts.add_token(token);
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
    }

    assert!(
        snapshots.iter().take(candidates.len()).all(Option::is_some),
        "every shortlisted shift must end at a token boundary"
    );
    snapshots
}

/// Apply a chosen shift directly to the owned token vectors. Partition scoring
/// is complete before this is called, so copying both blocks to construct the
/// new partition would add work without preserving anything useful.
fn shift_tokens(left: &mut PlannedBlock, right: &mut PlannedBlock, shift: BoundaryShift) {
    match shift {
        BoundaryShift::Right(wanted) => {
            let mut bytes = 0usize;
            let mut split = 0usize;
            while split < right.tokens.len() && bytes < wanted {
                bytes += right.tokens[split].input_len();
                split += 1;
            }
            assert_eq!(bytes, wanted, "boundary shift must end between tokens");
            left.tokens.extend(right.tokens.drain(..split));
        }
        BoundaryShift::Left(wanted) => {
            let mut bytes = 0usize;
            let mut split = left.tokens.len();
            while split > 0 && bytes < wanted {
                split -= 1;
                bytes += left.tokens[split].input_len();
            }
            assert_eq!(bytes, wanted, "boundary shift must end between tokens");
            let mut moved = left.tokens.split_off(split);
            moved.append(&mut right.tokens);
            right.tokens = moved;
        }
    }
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
) -> BoundaryOptimization {
    debug_assert_eq!(left.range.end, right.range.start);

    let left_histogram = TokenHistogram::from_tokens(&left.tokens);
    let right_histogram = TokenHistogram::from_tokens(&right.tokens);
    let Some((left_costs, left_score)) =
        compressed_histogram_score(state, &left_histogram, left.input_len())
    else {
        return BoundaryOptimization::Unchanged;
    };
    let Some((right_costs, right_score)) =
        compressed_histogram_score(state, &right_histogram, right.input_len())
    else {
        return BoundaryOptimization::Unchanged;
    };

    let mut best_total_bits = left_score.bits + right_score.bits;
    let mut best_shift = None;
    let mut merge_is_best = false;

    let merged_len = left.input_len() + right.input_len();
    if left.tokens.len() + right.tokens.len() <= MAX_BLOCK_TOKENS {
        let merged_histogram = TokenHistogram::merged(&left_histogram, &right_histogram);
        if let Some((_, merged_score)) =
            compressed_histogram_score(state, &merged_histogram, merged_len)
        {
            if merged_score.bits < best_total_bits {
                best_total_bits = merged_score.bits;
                merge_is_best = true;
            }
        }
    }

    let candidates = fixed_cost_shift_candidates(left, right, &left_costs, &right_costs);
    let snapshots =
        shortlisted_shift_histograms(left, right, &left_histogram, &right_histogram, &candidates);
    for (candidate, snapshot) in candidates.into_iter().zip(snapshots) {
        // All nominees are produced at exact token boundaries, verified by
        // shortlisted_shift_histograms before any exact rescoring starts.
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
        let Some((_, new_left_score)) =
            compressed_histogram_score(state, &shifted_left, new_left_len)
        else {
            continue;
        };
        let Some((_, new_right_score)) =
            compressed_histogram_score(state, &shifted_right, new_right_len)
        else {
            continue;
        };
        let total_bits = new_left_score.bits + new_right_score.bits;
        if total_bits < best_total_bits {
            best_total_bits = total_bits;
            best_shift = Some((candidate.shift, boundary));
            merge_is_best = false;
        }
    }

    let outcome = if merge_is_best {
        let mut tokens = core::mem::take(&mut left.tokens);
        tokens.append(&mut right.tokens);
        BoundaryOptimization::Merged(PlannedBlock {
            range: left.range.start..right.range.end,
            tokens,
        })
    } else if let Some((shift, boundary)) = best_shift {
        shift_tokens(left, right, shift);
        left.range.end = boundary;
        right.range.start = boundary;
        BoundaryOptimization::Shifted
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
    outcome
}

/// Parse one splitter range into a production-ready token block. A caller only needs to retain two
/// adjacent blocks at a time to optimize their shared Huffman boundary.
pub fn parse_planned_block_source(
    state: &mut super::State<'_>,
    input: &[u8],
    range: Range<usize>,
    max_distance: usize,
    optimization_passes: usize,
    previous_model: Option<&ParseModel>,
) -> PlannedBlockSource {
    let history_start = range.start.saturating_sub(max_distance);
    let context_end = range.end;
    let context = &input[history_start..context_end];
    let local_range = (range.start - history_start)..context.len();
    let candidates = build_match_candidates(context, max_distance);
    let (parsed, model) = optimized_parse_range(
        state,
        context,
        &candidates,
        local_range,
        optimization_passes,
        previous_model,
    );
    let block = PlannedBlock {
        range,
        tokens: parsed.tokens,
    };
    debug_assert_eq!(block.token_input_len(), block.range.len());
    PlannedBlockSource {
        block,
        model,
        context_start: history_start,
        context_end,
        candidates,
    }
}

#[cfg(test)]
mod boundary_tests {
    use super::*;

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
    fn shortlisted_histograms_match_exact_mixed_token_repartitions() {
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
            &TokenHistogram::from_tokens(&left.tokens),
            &TokenHistogram::from_tokens(&right.tokens),
            &nominees,
        );
        assert!(snapshots.iter().all(Option::is_some));
        for (nominee, (left_hist, right_hist)) in
            nominees.into_iter().zip(snapshots.into_iter().flatten())
        {
            let mut moved_left = left.clone();
            let mut moved_right = right.clone();
            shift_tokens(&mut moved_left, &mut moved_right, nominee.shift);
            assert_eq!(
                left_hist,
                TokenHistogram::from_tokens(&moved_left.tokens),
                "{:?}",
                nominee.shift
            );
            assert_eq!(
                right_hist,
                TokenHistogram::from_tokens(&moved_right.tokens),
                "{:?}",
                nominee.shift
            );
            assert_eq!(
                moved_left.token_input_len() + moved_right.token_input_len(),
                left.input_len() + right.input_len()
            );
        }
    }

    #[test]
    fn fixed_cost_boundary_shift_left_preserves_whole_tokens() {
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
        let left_histogram = TokenHistogram::from_tokens(&left.tokens);
        let right_histogram = TokenHistogram::from_tokens(&right.tokens);
        let shifts = fixed_cost_shift_candidates(&left, &right, &left_costs, &right_costs);
        let snapshots =
            shortlisted_shift_histograms(&left, &right, &left_histogram, &right_histogram, &shifts);
        let snapshot = &snapshots[0];
        assert_eq!(shifts[0].shift, BoundaryShift::Left(500));
        shift_tokens(&mut left, &mut right, shifts[0].shift);
        let new_left = &left.tokens;
        let new_right = &right.tokens;
        assert_eq!(new_left.len(), 5_500);
        assert_eq!(new_right.len(), 6_500);
        assert_eq!(
            snapshot.as_ref().map(|(left, _)| left),
            Some(&TokenHistogram::from_tokens(new_left))
        );
        assert_eq!(
            snapshot.as_ref().map(|(_, right)| right),
            Some(&TokenHistogram::from_tokens(new_right))
        );
        assert!(new_right[..500]
            .iter()
            .all(|token| *token == ParseToken::Literal(b'b')));
    }

    #[test]
    fn fixed_cost_boundary_shift_moves_whole_tokens_and_conserves_input() {
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

        let left_histogram = TokenHistogram::from_tokens(&left.tokens);
        let right_histogram = TokenHistogram::from_tokens(&right.tokens);
        let shifts = fixed_cost_shift_candidates(&left, &right, &left_costs, &right_costs);
        let snapshots =
            shortlisted_shift_histograms(&left, &right, &left_histogram, &right_histogram, &shifts);
        let snapshot = &snapshots[0];
        assert_eq!(shifts[0].shift, BoundaryShift::Right(500));
        let shift = shifts[0].shift;
        shift_tokens(&mut left, &mut right, shift);
        let shifted_left = &left.tokens;
        let shifted_right = &right.tokens;
        assert_eq!(
            snapshot.as_ref().map(|(left, _)| left),
            Some(&TokenHistogram::from_tokens(shifted_left))
        );
        assert_eq!(
            snapshot.as_ref().map(|(_, right)| right),
            Some(&TokenHistogram::from_tokens(shifted_right))
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
    fn additional_optimization_pass_never_enlarges_single_block_stream() {
        // Keep this below twice MIN_BLOCK_LENGTH so adaptive partitioning cannot change the block
        // layout between runs. For a fixed range, the optimization loop retains the cheapest exact
        // production-scored path seen so far, so an additional search must not enlarge the stream.
        let input = super::tests::two_distribution_input()[..8_000].to_vec();
        assert_eq!(choose_block_ranges(&input, 32_506), [0..input.len()]);

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
    }

    #[test]
    fn long_huffman_block_round_trips_beyond_stored_limit() {
        let pattern = b"abcdefghijklmnopqrstuvwxyz012345";
        let input = (0..120_000)
            .map(|index| pattern[index % pattern.len()])
            .collect::<Vec<_>>();
        assert_eq!(
            choose_block_ranges(&input, 32_768),
            [0..60_000, 60_000..120_000]
        );

        assert_round_trip(&input, 1);
    }

    #[test]
    fn adaptive_multi_block_stream_round_trips() {
        let input = super::tests::two_distribution_input();
        let blocks = choose_block_ranges(&input, 32_506);
        assert!(
            blocks.len() >= 2,
            "test input must exercise multiple blocks"
        );

        assert_round_trip(&input, 1);
        assert_round_trip(&input, 3);
    }

    #[test]
    fn full_window_distance_round_trips() {
        let mut state = 0x94d0_49bb_1331_11ebu64;
        let mut input = vec![0u8; 32_768];
        for byte in &mut input {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *byte = (state >> 32) as u8;
        }
        let repeated = input[..STD_MAX_MATCH].to_vec();
        input.extend_from_slice(&repeated);

        assert_round_trip(&input, 1);
    }

    #[test]
    fn every_tiny_input_length_round_trips() {
        for len in 0..=STD_MAX_MATCH + 1 {
            let input = (0..len)
                .map(|index| ((index.wrapping_mul(73) ^ (index >> 3)) & 0xff) as u8)
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
