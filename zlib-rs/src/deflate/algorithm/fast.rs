#![forbid(unsafe_code)]

use super::{BlockContinuation, ProduceStatus, TokenProducer, LOCAL_MATCH_SEARCH_MAX_CHAIN};
use crate::deflate::hash_calc::StandardHashCalc;
use crate::deflate::longest_match::{MatchSearchKind, MatchSearchResult, MatchSearchWork};
use crate::deflate::{State, MIN_LOOKAHEAD, STD_MIN_MATCH, WANT_MIN_MATCH};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FastPhase {
    Ready,
    AwaitingSearch {
        literal: u8,
    },
    SearchReady {
        literal: u8,
        result: MatchSearchResult,
    },
}

pub(super) struct FastProducer {
    phase: FastPhase,
}

impl FastProducer {
    pub(super) fn new() -> Self {
        Self {
            phase: FastPhase::Ready,
        }
    }
}

impl TokenProducer for FastProducer {
    fn produce(&mut self, state: &mut State<'_>, allow_short_tail: bool) -> ProduceStatus {
        loop {
            if let FastPhase::SearchReady { literal, result } = self.phase {
                self.phase = FastPhase::Ready;
                state.match_start = result.start;
                let mut match_len = result.length;

                if match_len >= WANT_MIN_MATCH {
                    let block_ready = state.tally_dist(
                        state.strstart - usize::from(result.start),
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

                let block_ready = state.tally_lit(literal);
                state.lookahead -= 1;
                state.strstart += 1;
                if block_ready {
                    return ProduceStatus::BlockReady(BlockContinuation::None);
                }
                continue;
            }

            assert_eq!(
                self.phase,
                FastPhase::Ready,
                "fast producer cannot advance while match search is outstanding"
            );

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
                    let work = MatchSearchWork::new(hash_head, MatchSearchKind::Standard);
                    if state.max_chain_length <= LOCAL_MATCH_SEARCH_MAX_CHAIN {
                        let result = work.request(state).execute();
                        self.phase = FastPhase::SearchReady {
                            literal: value as u8,
                            result,
                        };
                        continue;
                    }
                    self.phase = FastPhase::AwaitingSearch {
                        literal: value as u8,
                    };
                    return ProduceStatus::MatchSearch(work);
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

    fn complete_match_search(&mut self, _state: &mut State<'_>, result: MatchSearchResult) {
        let FastPhase::AwaitingSearch { literal } = self.phase else {
            panic!("fast producer received a match result without an outstanding search");
        };
        self.phase = FastPhase::SearchReady { literal, result };
    }

    fn finish(&mut self, state: &mut State<'_>) -> usize {
        assert_eq!(
            self.phase,
            FastPhase::Ready,
            "fast producer cannot finish while match search is outstanding"
        );
        state.strstart.min(STD_MIN_MATCH - 1)
    }
}
