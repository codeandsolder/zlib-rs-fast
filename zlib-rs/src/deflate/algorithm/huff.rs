#![forbid(unsafe_code)]

use super::{BlockContinuation, ProduceStatus, TokenProducer};
use crate::deflate::State;

pub(super) struct HuffmanProducer;

impl TokenProducer for HuffmanProducer {
    fn preferred_lookahead(&self) -> usize {
        1
    }

    fn produce(&mut self, state: &mut State<'_>, _allow_short_tail: bool) -> ProduceStatus {
        loop {
            if state.lookahead == 0 {
                return ProduceStatus::NeedInput;
            }

            let literal = state.window.filled()[state.strstart];
            let block_ready = state.tally_lit(literal);
            state.lookahead -= 1;
            state.strstart += 1;
            if block_ready {
                return ProduceStatus::BlockReady(BlockContinuation::None);
            }
        }
    }

    fn finish(&mut self, _state: &mut State<'_>) -> usize {
        0
    }
}
