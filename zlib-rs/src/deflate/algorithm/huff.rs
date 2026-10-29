#![forbid(unsafe_code)]

use super::{drive_standard, ProduceStatus, TokenProducer};
use crate::{
    deflate::{BlockState, Compressor, State},
    DeflateFlush,
};

pub fn deflate_huff(stream: &mut Compressor, flush: DeflateFlush) -> BlockState {
    drive_standard(stream, flush, &mut HuffmanProducer)
}

struct HuffmanProducer;

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
                return ProduceStatus::BlockReady;
            }
        }
    }

    fn finish(&mut self, _state: &mut State<'_>) -> usize {
        0
    }
}
