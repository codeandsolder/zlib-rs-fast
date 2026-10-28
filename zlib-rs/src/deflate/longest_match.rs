#![forbid(unsafe_code)]

use crate::deflate::{Pos, State, STD_MAX_MATCH, STD_MIN_MATCH};

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

/// Find the (length, offset) in the window of the longest match for the string
/// at offset `cur_match`.
pub fn longest_match(state: &State, cur_match: u16) -> (usize, u16) {
    longest_match_help::<false>(state, cur_match)
}

pub fn longest_match_slow(state: &State, cur_match: u16) -> (usize, u16) {
    longest_match_help::<true>(state, cur_match)
}

fn longest_match_help<const SLOW: bool>(state: &State, cur_match: Pos) -> (usize, Pos) {
    let strstart = state.strstart;
    let lookahead = state.lookahead;
    let initialized_end = strstart
        .checked_add(lookahead)
        .expect("deflate window position overflow");
    let window = state.window.filled();
    let initialized = window
        .get(..initialized_end)
        .expect("lookahead extends beyond the deflate window");
    let scan = &initialized[strstart..];
    let prev = &state.prev.as_slice()[..=state.w_mask()];
    let wmask = state.w_mask();
    let limit_base = strstart.saturating_sub(state.max_dist());

    let mut best_len = if state.prev_length > 0 {
        usize::from(state.prev_length)
    } else {
        STD_MIN_MATCH - 1
    };
    let mut match_start = state.match_start;

    // A previous lazy match can be longer than the input remaining at end of
    // stream. There is then no candidate that can improve it.
    if best_len >= lookahead {
        return (lookahead, match_start);
    }

    let mut chain_remaining = state.max_chain_length;
    if best_len >= usize::from(state.good_match) {
        chain_remaining >>= 2;
    }
    if chain_remaining == 0 {
        return break_matching(state, best_len, match_start);
    }

    let mut cursor = SearchCursor {
        hash_pos: cur_match,
        offset: 0,
    };

    if SLOW && best_len >= STD_MIN_MATCH {
        let Some(next) = initial_slow_cursor(state, scan, best_len, cursor, limit_base) else {
            return break_matching(state, best_len, match_start);
        };
        cursor = next;
    }

    let early_exit = !SLOW && state.level < EARLY_EXIT_TRIGGER_LEVEL;

    loop {
        let Some(candidate_start) = cursor.candidate_start() else {
            break;
        };
        if candidate_start >= strstart {
            break;
        }

        let max_len = lookahead.min(STD_MAX_MATCH);
        if best_len >= max_len {
            break;
        }

        let candidate = &initialized[candidate_start..candidate_start + max_len];
        let scan = &scan[..max_len];

        // Reject candidates that provably cannot improve the current best
        // before doing the full prefix comparison. The probes are entirely
        // inside initialized input; unlike the old pointer path they never
        // depend on bytes beyond lookahead.
        if !could_beat_best(scan, candidate, best_len) {
            let Some(next) = next_in_chain(cursor, prev, wmask, limit_base, &mut chain_remaining)
            else {
                break;
            };
            cursor = next;
            continue;
        }

        let len = common_match_length(scan, candidate);

        if len > best_len {
            match_start = candidate_start as Pos;

            if len >= lookahead {
                return (lookahead, match_start);
            }

            best_len = len;
            if best_len >= usize::from(state.nice_match) {
                return (best_len, match_start);
            }

            // The high-compression search can jump into a more distant chain
            // rooted at an interior substring of the match. Keep that heuristic,
            // but represent the rebasing explicitly instead of manufacturing
            // pointers before the window allocation.
            if SLOW && len > STD_MIN_MATCH && candidate_start + len < strstart {
                let Some(next) =
                    next_slow_cursor(state, scan, candidate_start, len, limit_base, wmask)
                else {
                    return break_matching(state, best_len, match_start);
                };
                cursor = next;
                continue;
            }
        } else if early_exit {
            break;
        }

        let Some(next) = next_in_chain(cursor, prev, wmask, limit_base, &mut chain_remaining)
        else {
            break;
        };
        cursor = next;
    }

    (best_len, match_start)
}

fn initial_slow_cursor(
    state: &State,
    scan: &[u8],
    best_len: usize,
    mut cursor: SearchCursor,
    limit_base: usize,
) -> Option<SearchCursor> {
    let probe = scan.get(..best_len + 1)?;
    let mut hash = 0;
    hash = state.update_hash(hash, u32::from(probe[1]));
    hash = state.update_hash(hash, u32::from(probe[2]));

    // Each update below hashes the next three-byte substring beginning at
    // offsets 1..=best_len-2. Prefer a more distant chain exactly as the old
    // slow matcher did.
    for (index, byte) in probe[3..].iter().enumerate() {
        hash = state.update_hash(hash, u32::from(*byte));
        let hash_pos = state.head.as_slice()[hash as usize];
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
    state: &State,
    scan: &[u8],
    candidate_start: usize,
    match_len: usize,
    limit_base: usize,
    wmask: usize,
) -> Option<SearchCursor> {
    let prev = state.prev.as_slice();
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
        hash = state.update_hash(hash, u32::from(*byte));
    }
    let hash_pos = state.head.as_slice()[hash as usize];
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

fn break_matching(state: &State, best_len: usize, match_start: Pos) -> (usize, Pos) {
    (best_len.min(state.lookahead), match_start)
}

#[cfg(test)]
mod tests {
    use super::*;

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
