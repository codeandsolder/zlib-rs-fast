#![forbid(unsafe_code)]

use super::{ensure_input, finish_standard, InputStatus};
use crate::{
    deflate::{BlockState, Compressor},
    DeflateFlush,
};

pub fn deflate_huff(stream: &mut Compressor, flush: DeflateFlush) -> BlockState {
    loop {
        match ensure_input(stream, 1, flush) {
            InputStatus::Ready => {}
            InputStatus::NeedMore => return BlockState::NeedMore,
            InputStatus::Exhausted => break,
        }

        /* Output a literal byte */
        let state = &mut stream.state;
        let lc = state.window.filled()[state.strstart];
        let bflush = state.tally_lit(lc);
        state.lookahead -= 1;
        state.strstart += 1;
        if bflush {
            if let Some(blocked) = stream.flush_block(false) {
                return blocked;
            }
        }
    }

    finish_standard(stream, flush, 0)
}
