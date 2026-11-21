#![forbid(unsafe_code)]

use super::{BlockContinuation, MatchSourceKind, ProduceStatus, TokenProducer};
use crate::deflate::hash_calc::StandardHashCalc;
use crate::deflate::longest_match::{
    MatchSearchError, MatchSearchKind, MatchSearchResult, MatchSearchWork,
};
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
    match_source: MatchSourceKind,
}

impl FastProducer {
    pub(super) const fn new(match_source: MatchSourceKind) -> Self {
        Self {
            phase: FastPhase::Ready,
            match_source,
        }
    }
}

impl FastProducer {
    fn produce_inner<const LOCAL: bool>(
        &mut self,
        state: &mut State<'_>,
        allow_short_tail: bool,
    ) -> ProduceStatus {
        loop {
            let (literal, result) = match self.phase {
                FastPhase::SearchReady { literal, result } if !LOCAL => {
                    self.phase = FastPhase::Ready;
                    (literal, Some(result))
                }
                _ => {
                    if !LOCAL && self.phase != FastPhase::Ready {
                        return ProduceStatus::MatchSearchError(
                            MatchSearchError::OutstandingSearch,
                        );
                    }
                    if state.lookahead == 0
                        || (!allow_short_tail && state.lookahead < MIN_LOOKAHEAD)
                    {
                        return ProduceStatus::NeedInput;
                    }

                    if state.lookahead >= WANT_MIN_MATCH {
                        if !LOCAL && self.match_source == MatchSourceKind::External {
                            let literal = state.window.filled()[state.strstart];
                            self.phase = FastPhase::AwaitingSearch { literal };
                            return ProduceStatus::MatchSearch(MatchSearchWork::external());
                        }
                        let value = StandardHashCalc::first_four_bytes(
                            &state.window.filled()[state.strstart..],
                        );
                        let literal = value.to_le_bytes()[0];
                        let hash_head =
                            StandardHashCalc::quick_insert_value(state, state.strstart, value);
                        let result = if hash_head != 0
                            && state.recent_hash_distance(hash_head).is_some()
                        {
                            let work = MatchSearchWork::new(hash_head, MatchSearchKind::Standard);
                            if LOCAL
                                || self
                                    .match_source
                                    .should_search_locally(state.max_chain_length)
                            {
                                match work.request(state) {
                                    Ok(request) => Some(request.execute()),
                                    Err(error) => return ProduceStatus::MatchSearchError(error),
                                }
                            } else {
                                self.phase = FastPhase::AwaitingSearch { literal };
                                return ProduceStatus::MatchSearch(work);
                            }
                        } else {
                            None
                        };
                        (literal, result)
                    } else {
                        (state.window.filled()[state.strstart], None)
                    }
                }
            };

            if let Some(result) = result {
                state.match_start = result.start;
                let mut match_len = usize::from(result.length);
                if match_len >= WANT_MIN_MATCH {
                    let block_ready = state.tally_dist(
                        state.strstart - usize::from(result.start),
                        match_len - STD_MIN_MATCH,
                    );
                    state.lookahead -= match_len;
                    if LOCAL || self.match_source.maintains_hash_chain() {
                        if match_len <= usize::from(state.max_insert_length)
                            && state.lookahead >= WANT_MIN_MATCH
                        {
                            match_len -= 1;
                            state.strstart += 1;
                            state.insert_string(state.strstart, match_len);
                            state.strstart += match_len;
                        } else {
                            state.strstart += match_len;
                            // Preserve the normal hash insert when a complete
                            // tail probe is available.
                            if state.lookahead >= STD_MIN_MATCH {
                                StandardHashCalc::quick_insert_string(
                                    state,
                                    state.strstart + 2 - STD_MIN_MATCH,
                                );
                            }
                        }
                    } else {
                        state.strstart += match_len;
                    }
                    if block_ready {
                        return ProduceStatus::BlockReady(BlockContinuation::None);
                    }
                    continue;
                }
            }
            let block_ready = state.tally_lit(literal);
            state.lookahead -= 1;
            state.strstart += 1;
            if block_ready {
                return ProduceStatus::BlockReady(BlockContinuation::None);
            }
        }
    }
}

impl TokenProducer for FastProducer {
    fn produce(&mut self, state: &mut State<'_>, allow_short_tail: bool) -> ProduceStatus {
        if matches!(self.match_source, MatchSourceKind::LocalHashChain) {
            if self.phase != FastPhase::Ready {
                return ProduceStatus::MatchSearchError(MatchSearchError::OutstandingSearch);
            }
            self.produce_inner::<true>(state, allow_short_tail)
        } else {
            self.produce_inner::<false>(state, allow_short_tail)
        }
    }

    fn complete_match_search(
        &mut self,
        _state: &mut State<'_>,
        result: MatchSearchResult,
    ) -> Result<(), MatchSearchError> {
        let FastPhase::AwaitingSearch { literal } = self.phase else {
            return Err(MatchSearchError::UnexpectedMatchResult);
        };
        self.phase = FastPhase::SearchReady { literal, result };
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn match_result_completion_rejects_wrong_phase_without_mutating_state(
    ) -> Result<(), crate::ReturnCode> {
        use crate::deflate::{DeflateConfig, DeflateStream};
        let mut stream = DeflateStream::new(DeflateConfig::new(2))?;
        let mut producer = FastProducer::new(MatchSourceKind::HashChain);
        let result = MatchSearchResult::new(4, 1);
        assert_eq!(
            producer.complete_match_search(&mut stream.state, result),
            Err(MatchSearchError::UnexpectedMatchResult)
        );
        assert_eq!(producer.phase, FastPhase::Ready);

        producer.phase = FastPhase::AwaitingSearch { literal: b'a' };
        assert_eq!(
            producer.complete_match_search(&mut stream.state, result),
            Ok(())
        );
        assert!(matches!(
            producer.phase,
            FastPhase::SearchReady { literal: b'a', result: found } if found == result
        ));
        assert_eq!(
            producer.complete_match_search(&mut stream.state, result),
            Err(MatchSearchError::UnexpectedMatchResult)
        );
        Ok(())
    }

    #[test]
    fn unfinished_match_search_is_a_typed_error() -> Result<(), crate::ReturnCode> {
        use crate::deflate::{DeflateConfig, DeflateStream};

        let mut stream = DeflateStream::new(DeflateConfig::new(2))?;
        let mut producer = FastProducer::new(MatchSourceKind::HashChain);
        producer.phase = FastPhase::AwaitingSearch { literal: b'a' };
        assert!(matches!(
            producer.produce(&mut stream.state, false),
            ProduceStatus::MatchSearchError(MatchSearchError::OutstandingSearch)
        ));
        Ok(())
    }
}
