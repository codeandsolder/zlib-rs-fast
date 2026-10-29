#![forbid(unsafe_code)]

use super::{drive_standard, ensure_input, InputStatus, ProduceStatus, TokenProducer};
use crate::{
    deflate::{BlockState, Compressor},
    DeflateFlush,
};

pub fn deflate_huff(stream: &mut Compressor, flush: DeflateFlush) -> BlockState {
    drive_standard(stream, flush, &mut HuffmanProducer)
}

struct HuffmanProducer;

impl TokenProducer for HuffmanProducer {
    fn produce(&mut self, stream: &mut Compressor<'_, '_>, flush: DeflateFlush) -> ProduceStatus {
        loop {
            match ensure_input(stream, 1, flush) {
                InputStatus::Ready => {}
                InputStatus::NeedMore => return ProduceStatus::NeedInput,
                InputStatus::Exhausted => return ProduceStatus::Finished { insert: 0 },
            }

            let state = &mut stream.state;
            let literal = state.window.filled()[state.strstart];
            let block_ready = state.tally_lit(literal);
            state.lookahead -= 1;
            state.strstart += 1;
            if block_ready {
                return ProduceStatus::BlockReady;
            }
        }
    }
}
