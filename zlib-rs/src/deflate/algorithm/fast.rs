#![forbid(unsafe_code)]

use super::{BlockContinuation, ProduceStatus, TokenProducer};
use crate::deflate::hash_calc::StandardHashCalc;
use crate::deflate::{State, MIN_LOOKAHEAD, STD_MIN_MATCH, WANT_MIN_MATCH};

pub(super) struct FastProducer;

impl TokenProducer for FastProducer {
    fn produce(&mut self, state: &mut State<'_>, allow_short_tail: bool) -> ProduceStatus {
        loop {
            if state.lookahead == 0 || (!allow_short_tail && state.lookahead < MIN_LOOKAHEAD) {
                return ProduceStatus::NeedInput;
            }

            let literal = if state.lookahead >= WANT_MIN_MATCH {
                let value = u32::from_le_bytes(
                    state.window.filled()[state.strstart..state.strstart + 4]
                        .try_into()
                        .expect("fast deflate probes four initialized bytes"),
                );
                let hash_head = StandardHashCalc::quick_insert_value(state, state.strstart, value);
                let distance = state.strstart as isize - hash_head as isize;

                if (1..=state.max_dist() as isize).contains(&distance) && hash_head != 0 {
                    let (mut match_len, match_start) =
                        crate::deflate::longest_match::longest_match(state, hash_head);
                    state.match_start = match_start;
                    if match_len >= WANT_MIN_MATCH {
                        let block_ready = state.tally_dist(
                            state.strstart - usize::from(match_start),
                            match_len - STD_MIN_MATCH,
                        );

                        state.lookahead -= match_len;
                        if match_len <= usize::from(state.max_insert_length)
                            && state.lookahead >= WANT_MIN_MATCH
                        {
                            match_len -= 1;
                            state.strstart += 1;
                            state.insert_string(state.strstart, match_len);
                            state.strstart += match_len;
                        } else {
                            state.strstart += match_len;
                            StandardHashCalc::quick_insert_string(
                                state,
                                state.strstart + 2 - STD_MIN_MATCH,
                            );
                        }

                        if block_ready {
                            return ProduceStatus::BlockReady(BlockContinuation::None);
                        }
                        continue;
                    }
                }
                value as u8
            } else {
                state.window.filled()[state.strstart]
            };

            let block_ready = state.tally_lit(literal);
            state.lookahead -= 1;
            state.strstart += 1;
            if block_ready {
                return ProduceStatus::BlockReady(BlockContinuation::None);
            }
        }
    }

    fn finish(&mut self, state: &mut State<'_>) -> usize {
        state.strstart.min(STD_MIN_MATCH - 1)
    }
}
