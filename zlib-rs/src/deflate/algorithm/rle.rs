#![forbid(unsafe_code)]

use super::{drive_standard, ensure_input, InputStatus, ProduceStatus, TokenProducer};
use crate::{
    deflate::{
        compare256::compare256_rle_slice, BlockState, Compressor, MIN_LOOKAHEAD, STD_MAX_MATCH,
        STD_MIN_MATCH,
    },
    DeflateFlush,
};

pub fn deflate_rle(stream: &mut Compressor, flush: DeflateFlush) -> BlockState {
    drive_standard(stream, flush, &mut RleProducer)
}

struct RleProducer;

impl TokenProducer for RleProducer {
    fn produce(&mut self, stream: &mut Compressor<'_, '_>, flush: DeflateFlush) -> ProduceStatus {
        loop {
            match ensure_input(stream, MIN_LOOKAHEAD, flush) {
                InputStatus::Ready => {}
                InputStatus::NeedMore => return ProduceStatus::NeedInput,
                InputStatus::Exhausted => return ProduceStatus::Finished { insert: 0 },
            }

            let state = &mut stream.state;
            let mut match_len = 0;
            if state.lookahead >= STD_MIN_MATCH && state.strstart > 0 {
                let scan = &state.window.filled()[state.strstart - 1..][..3 + 256];
                if scan[0] == scan[1] && scan[1] == scan[2] {
                    match_len = compare256_rle_slice(scan[0], &scan[3..]) + 2;
                    match_len = match_len.min(state.lookahead).min(STD_MAX_MATCH);
                }
                assert!(
                    state.strstart - 1 + match_len < state.window_size,
                    "wild scan"
                );
            }

            let block_ready = if match_len >= STD_MIN_MATCH {
                let ready = state.tally_dist(1, match_len - STD_MIN_MATCH);
                state.lookahead -= match_len;
                state.strstart += match_len;
                ready
            } else {
                let literal = state.window.filled()[state.strstart];
                let ready = state.tally_lit(literal);
                state.lookahead -= 1;
                state.strstart += 1;
                ready
            };

            if block_ready {
                return ProduceStatus::BlockReady;
            }
        }
    }
}
