#![forbid(unsafe_code)]

use super::{drive_standard, BlockContinuation, ProduceStatus, TokenProducer};
use crate::deflate::hash_calc::StandardHashCalc;
use crate::{
    deflate::{
        BlockState, Compressor, MatchCandidate, State, MIN_LOOKAHEAD, STD_MIN_MATCH, WANT_MIN_MATCH,
    },
    DeflateFlush,
};

type Match = MatchCandidate;

pub fn deflate_medium(stream: &mut Compressor, flush: DeflateFlush) -> BlockState {
    let mut producer = MediumProducer::new(stream.state);
    drive_standard(stream, flush, &mut producer)
}

struct MediumProducer {
    early_exit: bool,
    current_match: Match,
    next_match: Match,
}

impl MediumProducer {
    fn new(state: &State<'_>) -> Self {
        Self {
            early_exit: state.level < 5,
            current_match: Match::EMPTY,
            next_match: state.medium_next_match,
        }
    }

    fn clear_saved_next(&mut self, state: &mut State<'_>) {
        self.next_match = Match::EMPTY;
        state.medium_next_match = Match::EMPTY;
    }

    fn save_next(&self, state: &mut State<'_>) {
        state.medium_next_match = self.next_match;
    }
}

impl TokenProducer for MediumProducer {
    fn produce(&mut self, state: &mut State<'_>, allow_short_tail: bool) -> ProduceStatus {
        loop {
            if state.lookahead == 0 || (!allow_short_tail && state.lookahead < MIN_LOOKAHEAD) {
                return ProduceStatus::NeedInput;
            }

            let mut hash_head;

            // If we already have a future match from a previous round, just use that.
            if !self.early_exit && self.next_match.match_length > 0 {
                self.current_match = self.next_match;
                self.next_match.match_length = 0;
            } else {
                hash_head = 0;
                if state.lookahead >= WANT_MIN_MATCH {
                    hash_head = StandardHashCalc::quick_insert_string(state, state.strstart);
                }

                self.current_match.strstart = state.strstart as u16;
                self.current_match.orgstart = self.current_match.strstart;

                let dist = state.strstart as i64 - hash_head as i64;
                if (1..=state.max_dist() as i64).contains(&dist) && hash_head != 0 {
                    let (match_length, match_start) =
                        crate::deflate::longest_match::longest_match(state, hash_head);
                    state.match_start = match_start;
                    self.current_match.match_length = match_length as u16;
                    self.current_match.match_start = match_start;
                    if usize::from(self.current_match.match_length) < WANT_MIN_MATCH
                        || self.current_match.match_start >= self.current_match.strstart
                    {
                        self.current_match.match_length = 1;
                    }
                } else {
                    self.current_match.match_start = 0;
                    self.current_match.match_length = 1;
                }
            }

            insert_match(state, self.current_match);

            // Probe one match ahead. The candidate is saved into persistent state when a full
            // token block is yielded, so an output-blocking return cannot change later choices.
            let current_len = usize::from(self.current_match.match_length);
            let future_start = usize::from(self.current_match.strstart) + current_len;
            let future_lookahead = state.lookahead.saturating_sub(current_len);
            if !self.early_exit
                && state.lookahead > MIN_LOOKAHEAD
                && future_lookahead >= WANT_MIN_MATCH
                && future_start < state.window_size - MIN_LOOKAHEAD
            {
                let current_lookahead = state.lookahead;
                state.strstart = future_start;
                state.lookahead = future_lookahead;
                hash_head = StandardHashCalc::quick_insert_string(state, state.strstart);

                self.next_match.strstart = state.strstart as u16;
                self.next_match.orgstart = self.next_match.strstart;

                let dist = state.strstart as i64 - hash_head as i64;
                if (1..=state.max_dist() as i64).contains(&dist) && hash_head != 0 {
                    let (match_length, match_start) =
                        crate::deflate::longest_match::longest_match(state, hash_head);
                    state.match_start = match_start;
                    self.next_match.match_length = match_length as u16;
                    self.next_match.match_start = match_start;

                    if self.next_match.match_start >= self.next_match.strstart {
                        self.next_match.match_length = 1;
                    }
                    if usize::from(self.next_match.match_length) < WANT_MIN_MATCH {
                        self.next_match.match_length = 1;
                    } else {
                        fizzle_matches(
                            state.window.filled(),
                            state.max_dist(),
                            &mut self.current_match,
                            &mut self.next_match,
                        );
                    }
                } else {
                    self.next_match.match_start = 0;
                    self.next_match.match_length = 1;
                }

                state.lookahead = current_lookahead;
                state.strstart = usize::from(self.current_match.strstart);
            } else {
                self.next_match.match_length = 0;
            }

            let block_ready = emit_match(state, self.current_match);
            state.strstart += usize::from(self.current_match.match_length);

            if block_ready {
                self.save_next(state);
                return ProduceStatus::BlockReady(BlockContinuation::None);
            }
        }
    }

    fn after_refill(&mut self, state: &mut State<'_>) {
        // A refill may slide the window, invalidating a speculative absolute position.
        self.clear_saved_next(state);
    }

    fn finish(&mut self, state: &mut State<'_>) -> usize {
        self.clear_saved_next(state);
        state.strstart.min(STD_MIN_MATCH - 1)
    }
}

fn emit_match(state: &mut State, m: Match) -> bool {
    let mut bflush = false;

    /* matches that are not long enough we need to emit as literals */
    if (m.match_length as usize) < WANT_MIN_MATCH {
        for lc in &state.window.filled()[state.strstart..][..m.match_length as usize] {
            bflush |= State::tally_lit_help(&mut state.sym_buf, &mut state.l_desc, *lc);
        }
    } else {
        // check_match(s, m.strstart, m.match_start, m.match_length);

        bflush |= state.tally_dist(
            (m.strstart - m.match_start) as usize,
            m.match_length as usize - STD_MIN_MATCH,
        );
    }
    state.lookahead -= m.match_length as usize;

    bflush
}

#[inline(always)]
fn insert_match(state: &mut State, mut m: Match) {
    if state.lookahead <= (m.match_length as usize + WANT_MIN_MATCH) {
        return;
    }

    /* matches that are not long enough we need to emit as literals */
    if (m.match_length as usize) < WANT_MIN_MATCH {
        m.strstart += 1;
        m.match_length -= 1;
        if m.match_length > 0 && m.strstart >= m.orgstart {
            if m.strstart + m.match_length > m.orgstart {
                state.insert_string(m.strstart as usize, m.match_length as usize);
            } else {
                state.insert_string(m.strstart as usize, (m.orgstart - m.strstart + 1) as usize);
            }
        }
        return;
    }

    // Insert new strings in the hash table only if the match length
    // is not too large. This saves time but degrades compression.
    if usize::from(m.match_length) <= 16 * usize::from(state.max_insert_length)
        && state.lookahead >= WANT_MIN_MATCH
    {
        m.match_length -= 1; /* string at strstart already in table */
        m.strstart += 1;

        if m.strstart >= m.orgstart {
            if m.strstart + m.match_length > m.orgstart {
                state.insert_string(m.strstart as usize, m.match_length as usize);
            } else {
                state.insert_string(m.strstart as usize, (m.orgstart - m.strstart + 1) as usize);
            }
        } else if m.orgstart < m.strstart + m.match_length {
            state.insert_string(
                m.orgstart as usize,
                (m.strstart + m.match_length - m.orgstart) as usize,
            );
        }
    } else {
        m.strstart += m.match_length;
        m.match_length = 0;

        if (m.strstart as usize) >= (STD_MIN_MATCH - 2) {
            StandardHashCalc::quick_insert_string(state, m.strstart as usize + 2 - STD_MIN_MATCH);
        }

        /* If lookahead < WANT_MIN_MATCH, ins_h is garbage, but it does not
         * matter since it will be recomputed at next deflate call.
         */
    }
}

fn fizzle_matches(window: &[u8], max_dist: usize, current: &mut Match, next: &mut Match) {
    /* step zero: sanity checks */

    if current.match_length <= 1 {
        return;
    }

    if current.match_length > 1 + next.match_start {
        return;
    }

    if current.match_length > 1 + next.strstart {
        return;
    }

    let m = &window[(-(current.match_length as isize) + 1 + next.match_start as isize) as usize..];
    let orig = &window[(-(current.match_length as isize) + 1 + next.strstart as isize) as usize..];

    /* quick exit check.. if this fails then don't bother with anything else */
    if m[0] != orig[0] {
        return;
    }

    /* step one: try to move the "next" match to the left as much as possible */
    let limit = next.strstart.saturating_sub(max_dist as u16);

    let mut c = *current;
    let mut n = *next;

    let m = &window[..n.match_start as usize];
    let orig = &window[..n.strstart as usize];

    let mut m = m.iter().rev();
    let mut orig = orig.iter().rev();

    let mut changed = 0;

    while m.next() == orig.next() {
        if c.match_length < 1 {
            break;
        }
        if n.strstart <= limit {
            break;
        }
        if n.match_length >= 256 {
            break;
        }
        if n.match_start <= 1 {
            break;
        }

        n.strstart -= 1;
        n.match_start -= 1;
        n.match_length += 1;
        c.match_length -= 1;
        changed += 1;
    }

    if changed == 0 {
        return;
    }

    if c.match_length <= 1 && n.match_length != 2 {
        n.orgstart += 1;
        *current = c;
        *next = n;
    }
}
