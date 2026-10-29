#![forbid(unsafe_code)]

use super::{BlockContinuation, ProduceStatus, TokenProducer};
use crate::deflate::longest_match::{MatchSearchKind, MatchSearchResult, MatchSearchWork};
use crate::deflate::{State, Strategy, MIN_LOOKAHEAD, STD_MIN_MATCH, WANT_MIN_MATCH};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlowPhase {
    Ready,
    AwaitingSearch,
    SearchReady(MatchSearchResult),
}

pub(super) struct SlowProducer {
    use_longest_match_slow: bool,
    match_available: bool,
    phase: SlowPhase,
}

impl SlowProducer {
    pub(super) fn new(state: &State<'_>) -> Self {
        Self {
            use_longest_match_slow: state.max_chain_length > 1024,
            match_available: state.match_available,
            phase: SlowPhase::Ready,
        }
    }
}

impl TokenProducer for SlowProducer {
    fn produce(&mut self, state: &mut State<'_>, allow_short_tail: bool) -> ProduceStatus {
        loop {
            let match_len = match self.phase {
                SlowPhase::SearchReady(result) => {
                    self.phase = SlowPhase::Ready;
                    state.match_start = result.start;
                    let mut length = result.length;
                    if length <= 5 && state.strategy == Strategy::Filtered {
                        length = STD_MIN_MATCH - 1;
                    }
                    length
                }
                SlowPhase::Ready => {
                    if state.lookahead == 0
                        || (!allow_short_tail && state.lookahead < MIN_LOOKAHEAD)
                    {
                        return ProduceStatus::NeedInput;
                    }

                    let hash_head = if state.lookahead >= WANT_MIN_MATCH {
                        state.quick_insert_string(state.strstart)
                    } else {
                        0
                    };

                    state.prev_match = state.match_start;
                    let distance = state.strstart as isize - hash_head as isize;
                    if (1..=state.max_dist() as isize).contains(&distance)
                        && state.prev_length < state.max_lazy_match
                        && hash_head != 0
                    {
                        let kind = if self.use_longest_match_slow {
                            MatchSearchKind::Slow
                        } else {
                            MatchSearchKind::Standard
                        };
                        self.phase = SlowPhase::AwaitingSearch;
                        return ProduceStatus::MatchSearch(MatchSearchWork::new(hash_head, kind));
                    }

                    STD_MIN_MATCH - 1
                }
                SlowPhase::AwaitingSearch => {
                    panic!("slow producer cannot advance while match search is outstanding")
                }
            };

            if usize::from(state.prev_length) >= STD_MIN_MATCH
                && match_len <= usize::from(state.prev_length)
            {
                let max_insert = state.strstart + state.lookahead - STD_MIN_MATCH;
                let block_ready = state.tally_dist(
                    state.strstart - 1 - usize::from(state.prev_match),
                    usize::from(state.prev_length) - STD_MIN_MATCH,
                );

                state.prev_length -= 1;
                state.lookahead -= usize::from(state.prev_length);

                let move_forward = usize::from(state.prev_length) - 1;
                if max_insert > state.strstart {
                    let insert_count = move_forward.min(max_insert - state.strstart);
                    state.insert_string(state.strstart + 1, insert_count);
                }
                state.prev_length = 0;
                state.match_available = false;
                self.match_available = false;
                state.strstart += move_forward + 1;

                if block_ready {
                    return ProduceStatus::BlockReady(BlockContinuation::None);
                }
            } else if self.match_available {
                let literal = state.window.filled()[state.strstart - 1];
                let block_ready = state.tally_lit(literal);
                if block_ready {
                    // zlib encodes this full block before advancing the lazy-match cursor. Keep that
                    // ordering explicit so stored-block length/accounting remains bit-for-bit stable.
                    return ProduceStatus::BlockReady(BlockContinuation::AdvanceLazyMatch {
                        match_len: match_len as u16,
                    });
                }

                state.prev_length = match_len as u16;
                state.strstart += 1;
                state.lookahead -= 1;
            } else {
                state.prev_length = match_len as u16;
                state.match_available = true;
                self.match_available = true;
                state.strstart += 1;
                state.lookahead -= 1;
            }
        }
    }

    fn complete_match_search(&mut self, _state: &mut State<'_>, result: MatchSearchResult) {
        assert_eq!(
            self.phase,
            SlowPhase::AwaitingSearch,
            "slow producer received a match result without an outstanding search"
        );
        self.phase = SlowPhase::SearchReady(result);
    }

    fn finish(&mut self, state: &mut State<'_>) -> usize {
        assert_eq!(
            self.phase,
            SlowPhase::Ready,
            "slow producer cannot finish while match search is outstanding"
        );
        if state.match_available {
            let literal = state.window.filled()[state.strstart - 1];
            let _ = state.tally_lit(literal);
            state.match_available = false;
            self.match_available = false;
        }
        state.strstart.min(STD_MIN_MATCH - 1)
    }
}
