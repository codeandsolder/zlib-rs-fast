//! Exact prefix-group-local match frontier construction.
//!
//! All candidates for a DEFLATE match share the first three bytes.
//! A group-local rank interval therefore contains every useful candidate
//! and excludes only suffixes that could never emit a match. Preserve the
//! dense scanner's ordered records and its distinct narrow/wide tie rules.

use alloc::{vec, vec::Vec};
use core::ops::Range;

use super::{
    compact_match_length, highest_bit, match_compare, stage_candidate, suffix_index,
    FrontierTileScratch, MatchCandidate, MatchCandidates, RankSet, State, SuffixRanks,
    DIRECT_SCAN_MAX, D_CODES, FRONTIER_TILE_POSITIONS, STD_MAX_MATCH, STD_MIN_MATCH,
};

// A short rank interval is cheaper to enumerate than to maintain a
// position-ordered index. Groups fitting a word use exact prefix masks;
// larger groups use the existing hierarchical RankSet.
const SMALL_GROUP_DIRECT: usize = 16;
const WORD_GROUP_SIZE: usize = u64::BITS as usize;
const _: () = assert!(D_CODES <= u32::BITS as usize);

#[inline]
fn word_neighbors(mask: u64, rank: usize) -> (Option<usize>, Option<usize>) {
    let left = mask & ((1u64 << rank) - 1);
    let right = mask & (u64::MAX << rank) & !(1u64 << rank);
    (
        (left != 0).then(|| highest_bit(left)),
        (right != 0).then(|| right.trailing_zeros() as usize),
    )
}

/// Position is the high word so numeric sorting gives input order;
/// the low word retains its rank within this exact prefix group.
#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
struct Occurrence(u64);

impl Occurrence {
    #[inline]
    const fn new(position: u32, rank: u32) -> Self {
        Self(((position as u64) << 32) | rank as u64)
    }

    #[inline]
    const fn position(self) -> usize {
        let [_, _, _, _, a, b, c, d] = self.0.to_le_bytes();
        suffix_index(u32::from_le_bytes([a, b, c, d]))
    }

    #[inline]
    const fn rank(self) -> usize {
        let [a, b, c, d, ..] = self.0.to_le_bytes();
        suffix_index(u32::from_le_bytes([a, b, c, d]))
    }
}

#[inline]
fn group_occurrences(group: &[u32]) -> impl Iterator<Item = Occurrence> + '_ {
    (0u32..)
        .zip(group)
        .map(|(rank, &position)| Occurrence::new(position, rank))
}

#[inline]
fn candidate_count(count: usize) -> u8 {
    debug_assert!(count <= D_CODES);
    u8::try_from(count).unwrap_or_else(|_| unreachable!("distance alphabet fits in u8"))
}

#[inline]
fn grouped_compare(
    input: &[u8],
    position: usize,
    other: usize,
    available: usize,
    best: &mut MatchCandidate,
) {
    debug_assert!(position > other && available >= STD_MIN_MATCH);
    debug_assert_eq!(
        input[position..position + STD_MIN_MATCH],
        input[other..other + STD_MIN_MATCH],
    );
    let length = STD_MIN_MATCH
        + match_compare::common_prefix_len(
            &input[position + STD_MIN_MATCH..position + available],
            &input[other + STD_MIN_MATCH..other + available],
        );
    best.consider(length, position - other);
}

struct GroupOutput {
    entries: Vec<MatchCandidate>,
    first_query: usize,
    starts: Vec<usize>,
    counts: Vec<u8>,
}
impl GroupOutput {
    fn new(query: &Range<usize>) -> Self {
        let length = query.len();
        Self {
            entries: Vec::new(),
            first_query: query.start,
            starts: vec![0; length + 1],
            counts: vec![0; length],
        }
    }
    fn finish(self) -> MatchCandidates {
        let mut offsets = self.starts;
        let mut entries = Vec::with_capacity(self.entries.len());
        for (start, count) in offsets.iter_mut().zip(self.counts) {
            let old = *start;
            *start = entries.len();
            entries.extend_from_slice(&self.entries[old..old + usize::from(count)]);
        }
        let last = offsets.len() - 1;
        offsets[last] = entries.len();
        MatchCandidates {
            first_query: self.first_query,
            offsets,
            entries,
        }
    }
}

/// In a small group, visit nearest ranks first on each side. Wide classes
/// need only the first eligible rank of each side; direct classes retain
/// the exhaustive nearest-distance tie rule of the original scanner.
fn scan_small_group(
    input: &[u8],
    group: &[u32],
    wide: u32,
    max_distance: usize,
    query_range: &Range<usize>,
    out: &mut GroupOutput,
) {
    for (rank, &query) in group.iter().enumerate() {
        let position = suffix_index(query);
        if !query_range.contains(&position) {
            continue;
        }
        let available = STD_MAX_MATCH.min(input.len() - position);
        let mut best = [MatchCandidate::NONE; D_CODES];
        let mut occupied = 0u32;
        let mut side = |start: usize, count: usize, reverse: bool| {
            let mut seen = 0u32;
            for i in 0..count {
                let other = suffix_index(group[if reverse { start - i } else { start + i }]);
                if other >= position {
                    continue;
                }
                let distance = position - other;
                if distance > max_distance {
                    continue;
                }
                let code = usize::from(State::d_code(distance - 1));
                let marker = 1u32 << code;
                if wide & marker != 0 {
                    if seen & marker != 0 {
                        continue;
                    }
                    seen |= marker;
                }
                grouped_compare(input, position, other, available, &mut best[code]);
                occupied |= marker;
            }
        };
        if rank != 0 {
            side(rank - 1, rank, true);
        }
        side(rank + 1, group.len() - rank - 1, false);
        out.starts[position - out.first_query] = out.entries.len();
        out.counts[position - out.first_query] = candidate_count(occupied.count_ones() as usize);
        while occupied != 0 {
            let code = occupied.trailing_zeros() as usize;
            occupied &= occupied - 1;
            out.entries.push(best[code]);
        }
    }
}

/// Exact membership in a time interval of a word-sized group is the
/// xor of its two prefix masks: each suffix rank occurs exactly once.
/// Enumerate only distance classes with at least one prior occurrence.
fn scan_word_group(
    input: &[u8],
    group: &[u32],
    ranges: &[(usize, usize); D_CODES],
    max_distance: usize,
    query_range: &Range<usize>,
    out: &mut GroupOutput,
) {
    let mut packed = [Occurrence::default(); WORD_GROUP_SIZE];
    let mut prefix = [0u64; WORD_GROUP_SIZE + 1];
    let positions = &mut packed[..group.len()];
    for (slot, record) in positions.iter_mut().zip(group_occurrences(group)) {
        *slot = record;
    }
    positions.sort_unstable();
    for (i, &word) in positions.iter().enumerate() {
        prefix[i + 1] = prefix[i] | (1u64 << (word.rank()));
    }
    for (i, &query) in positions.iter().enumerate() {
        let position = query.position();
        if !query_range.contains(&position) {
            continue;
        }
        let rank = query.rank();
        let available = STD_MAX_MATCH.min(input.len() - position);
        out.starts[position - out.first_query] = out.entries.len();
        let mut end = i;
        while end != 0 {
            let other = positions[end - 1].position();
            let distance = position - other;
            if distance > max_distance {
                break;
            }
            let code = usize::from(State::d_code(distance - 1));
            let (minimum, maximum) = ranges[code];
            let lower = position.saturating_sub(maximum);
            let begin = positions[..end].partition_point(|&word| word.position() < lower);
            let mut best = MatchCandidate::NONE;
            if maximum - minimum + 1 > DIRECT_SCAN_MAX {
                let mask = prefix[end] ^ prefix[begin];
                let neighbors: [_; 2] = word_neighbors(mask, rank).into();
                for neighbor in neighbors.into_iter().flatten() {
                    grouped_compare(
                        input,
                        position,
                        suffix_index(group[neighbor]),
                        available,
                        &mut best,
                    );
                }
            } else {
                for &word in positions[begin..end].iter().rev() {
                    grouped_compare(input, position, word.position(), available, &mut best);
                    if usize::from(best.length) == available {
                        break;
                    }
                }
            }
            debug_assert_ne!(best.length, 0);
            out.entries.push(best);
            end = begin;
        }
        out.counts[position - out.first_query] =
            candidate_count(out.entries.len() - out.starts[position - out.first_query]);
    }
}

struct ClassWindow {
    add: usize,
    remove: usize,
    ranks: Option<RankSet>,
}

impl ClassWindow {
    fn new(group_len: usize, (min, max): (usize, usize)) -> Self {
        Self {
            add: 0,
            remove: 0,
            ranks: (min != 0 && max - min + 1 > DIRECT_SCAN_MAX).then(|| RankSet::new(group_len)),
        }
    }

    #[inline]
    fn best(
        &mut self,
        source: &GroupInput<'_>,
        query: Occurrence,
        available: usize,
        (minimum, maximum): (usize, usize),
    ) -> MatchCandidate {
        let position = query.position();
        let Some(upper) = position.checked_sub(minimum) else {
            return MatchCandidate::NONE;
        };
        let lower = position.saturating_sub(maximum);
        let positions = source.positions;
        let mut best = MatchCandidate::NONE;
        if let Some(set) = &mut self.ranks {
            while self.remove < self.add && positions[self.remove].position() < lower {
                set.remove(positions[self.remove].rank());
                self.remove += 1;
            }
            if self.remove == self.add {
                // A gap may make an entire run expire before it
                // could participate in any query in this class.
                while self.add < positions.len() && positions[self.add].position() < lower {
                    self.add += 1;
                }
                self.remove = self.add;
            }
            while self.add < positions.len() && positions[self.add].position() <= upper {
                set.insert(positions[self.add].rank());
                self.add += 1;
            }
            let neighbors: [_; 2] = set.neighbors(query.rank()).into();
            for rank in neighbors.into_iter().flatten() {
                grouped_compare(
                    source.input,
                    position,
                    suffix_index(source.group[rank]),
                    available,
                    &mut best,
                );
            }
        } else {
            // Truncated final narrow class. Preserve nearest-distance ties
            // even if a lexicographically closer suffix matches equally far.
            while self.add < positions.len() && positions[self.add].position() <= upper {
                self.add += 1;
            }
            while self.remove < self.add && positions[self.remove].position() < lower {
                self.remove += 1;
            }
            for other in positions[self.remove..self.add].iter().rev() {
                grouped_compare(
                    source.input,
                    position,
                    other.position(),
                    available,
                    &mut best,
                );
                if usize::from(best.length) == available {
                    break;
                }
            }
        }
        best
    }
}

struct GroupInput<'a> {
    input: &'a [u8],
    group: &'a [u32],
    positions: &'a [Occurrence],
}

/// All leading narrow classes share a nearest-first scan through
/// equal-prefix positions. Stop comparing once a class has reached
/// the entire available match length, but keep emitting later classes.
fn scan_group_nearby(
    source: &GroupInput<'_>,
    tile: Range<usize>,
    near_max: usize,
    near_start: &mut usize,
    scratch: &mut FrontierTileScratch,
) {
    if near_max == 0 {
        return;
    }
    for (local, query_index) in tile.enumerate() {
        let position = source.positions[query_index].position();
        while *near_start < query_index
            && position - source.positions[*near_start].position() > near_max
        {
            *near_start += 1;
        }
        let available = usize::from(scratch.available[local]);
        let mut best = MatchCandidate::NONE;
        let mut current = None;
        for record in source.positions[*near_start..query_index].iter().rev() {
            let other = record.position();
            let code = State::d_code(position - other - 1);
            if current != Some(code) {
                if best.length != 0 {
                    stage_candidate(local, best, &mut scratch.counts, &mut scratch.staged);
                }
                best = MatchCandidate::NONE;
                current = Some(code);
            }
            if usize::from(best.length) != available {
                grouped_compare(source.input, position, other, available, &mut best);
            }
        }
        if best.length != 0 {
            stage_candidate(local, best, &mut scratch.counts, &mut scratch.staged);
        }
    }
}

struct GroupScratch {
    tile: FrontierTileScratch,
    positions: Vec<Occurrence>,
    offsets: Vec<usize>,
}
impl GroupScratch {
    fn new() -> Self {
        Self {
            tile: FrontierTileScratch::new(),
            positions: Vec::new(),
            offsets: Vec::with_capacity(FRONTIER_TILE_POSITIONS + 1),
        }
    }

    fn scan_large(
        &mut self,
        input: &[u8],
        group: &[u32],
        ranges: &[(usize, usize); D_CODES],
        query_range: &Range<usize>,
        out: &mut GroupOutput,
    ) {
        self.positions.clear();
        self.positions.extend(group_occurrences(group));
        self.positions.sort_unstable();
        let source = GroupInput {
            input,
            group,
            positions: &self.positions,
        };
        let mut windows: [ClassWindow; D_CODES] =
            core::array::from_fn(|code| ClassWindow::new(group.len(), ranges[code]));
        let first_wide = ranges
            .iter()
            .position(|&(min, max)| min != 0 && max - min + 1 > DIRECT_SCAN_MAX)
            .unwrap_or(D_CODES);
        let near_max = ranges[..first_wide]
            .iter()
            .map(|&(_, max)| max)
            .max()
            .unwrap_or(0);
        let mut near_start = 0;
        let scratch = &mut self.tile;
        let query_start = source
            .positions
            .partition_point(|record| record.position() < query_range.start);
        let query_end = source
            .positions
            .partition_point(|record| record.position() < query_range.end);
        for (tile_index, tile) in source.positions[query_start..query_end]
            .chunks(FRONTIER_TILE_POSITIONS)
            .enumerate()
        {
            let len = tile.len();
            scratch.counts[..len].fill(0);
            scratch.staged.clear();
            for (available, record) in scratch.available[..len].iter_mut().zip(tile) {
                *available =
                    compact_match_length(STD_MAX_MATCH.min(input.len() - record.position()));
            }
            let start = query_start + tile_index * FRONTIER_TILE_POSITIONS;
            scan_group_nearby(
                &source,
                start..start + len,
                near_max,
                &mut near_start,
                scratch,
            );
            for (window, &range) in windows.iter_mut().zip(ranges).skip(first_wide) {
                if range.0 == 0 {
                    continue;
                }
                for (local, &record) in tile.iter().enumerate() {
                    let best = window.best(
                        &source,
                        record,
                        usize::from(scratch.available[local]),
                        range,
                    );
                    if best.length != 0 {
                        stage_candidate(local, best, &mut scratch.counts, &mut scratch.staged);
                    }
                }
            }
            self.offsets.clear();
            self.offsets.push(out.entries.len());
            scratch.scatter(len, &mut out.entries, &mut self.offsets);
            for (local, record) in tile.iter().enumerate() {
                let position = record.position();
                out.starts[position - out.first_query] = self.offsets[local];
                out.counts[position - out.first_query] =
                    candidate_count(usize::from(scratch.counts[local]));
            }
        }
    }
}

/// Exact three-byte prefix intervals recovered from the existing four-byte
/// equivalence classes. Later suffix refinement only changes order within
/// these classes, so the intervals remain valid in the final permutation.
pub(super) struct PrefixGroups {
    intervals: Vec<Range<usize>>,
}

impl PrefixGroups {
    pub(super) const fn unique() -> Self {
        Self {
            intervals: Vec::new(),
        }
    }

    pub(super) fn from_refinement(input: &[u8], sa: &[u32], starts: &[u32]) -> Option<Self> {
        // A small three-byte group can contain only small four-byte
        // classes. Their total size is a cheap upper bound on the share
        // of positions eligible for the word-sized strategy.
        let length = input.len();
        let ends = starts
            .iter()
            .copied()
            .skip(1)
            .map(suffix_index)
            .chain(core::iter::once(length));
        let upper = starts
            .iter()
            .copied()
            .map(suffix_index)
            .zip(ends)
            .filter(|&(start, end)| end - start <= WORD_GROUP_SIZE)
            .map(|(start, end)| end - start)
            .sum::<usize>();
        if upper <= length / 2 {
            return None;
        }
        let mut groups = Self::unique();
        let mut word_positions = 0;
        let mut group_start = 0;
        let mut prefix = input[suffix_index(sa[0])..].get(..STD_MIN_MATCH);
        for &start in &starts[1..] {
            let start = suffix_index(start);
            let current = input[suffix_index(sa[start])..].get(..STD_MIN_MATCH);
            if current != prefix {
                word_positions += groups.record(group_start..start, prefix.is_some());
                group_start = start;
                prefix = current;
            }
        }
        word_positions += groups.record(group_start..length, prefix.is_some());
        (word_positions > length / 2).then_some(groups)
    }

    fn record(&mut self, interval: Range<usize>, complete_prefix: bool) -> usize {
        let len = interval.len();
        if complete_prefix && len > 1 {
            self.intervals.push(interval);
        }
        if len <= WORD_GROUP_SIZE {
            len
        } else {
            0
        }
    }
}

#[inline(never)]
pub(super) fn group_local_frontier(
    input: &[u8],
    suffixes: &SuffixRanks,
    groups: &PrefixGroups,
    ranges: &[(usize, usize); D_CODES],
    max_distance: usize,
    query_range: Range<usize>,
) -> MatchCandidates {
    if groups.intervals.is_empty() {
        return MatchCandidates {
            first_query: query_range.start,
            offsets: vec![0; query_range.len() + 1],
            entries: Vec::new(),
        };
    }
    let mut wide = 0u32;
    for (code, &(min, max)) in ranges.iter().enumerate() {
        if min != 0 && max - min + 1 > DIRECT_SCAN_MAX {
            wide |= 1 << code;
        }
    }
    let mut output = GroupOutput::new(&query_range);
    let mut scratch = GroupScratch::new();
    for interval in &groups.intervals {
        let group = &suffixes.sa[interval.clone()];
        match group.len() {
            n if n <= SMALL_GROUP_DIRECT => {
                scan_small_group(input, group, wide, max_distance, &query_range, &mut output);
            }
            n if n <= WORD_GROUP_SIZE => {
                scan_word_group(
                    input,
                    group,
                    ranges,
                    max_distance,
                    &query_range,
                    &mut output,
                );
            }
            _ => scratch.scan_large(input, group, ranges, &query_range, &mut output),
        }
    }
    output.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_group_rank_neighbors_preserve_exact_ties_at_boundaries() {
        for rank in 0..WORD_GROUP_SIZE {
            for mask in [
                0,
                1,
                1u64 << 63,
                u64::MAX,
                0x8000_0001_0000_0001,
                0x5555_aaaa_5555_aaaa,
            ] {
                let left = (0..rank).rev().find(|&n| mask & (1u64 << n) != 0);
                let right = (rank + 1..WORD_GROUP_SIZE).find(|&n| mask & (1u64 << n) != 0);
                assert_eq!(word_neighbors(mask, rank), (left, right));
            }
        }
    }
}
