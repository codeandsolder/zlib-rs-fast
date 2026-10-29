#![forbid(unsafe_code)]

use super::{BlockContinuation, ProduceStatus, TokenProducer};
use crate::deflate::{
    compare256::compare256_rle_slice, State, MIN_LOOKAHEAD, STD_MAX_MATCH, STD_MIN_MATCH,
};

pub(super) struct RleProducer;

impl TokenProducer for RleProducer {
    fn produce(&mut self, state: &mut State<'_>, allow_short_tail: bool) -> ProduceStatus {
        loop {
            if state.lookahead == 0 || (!allow_short_tail && state.lookahead < MIN_LOOKAHEAD) {
                return ProduceStatus::NeedInput;
            }

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
                return ProduceStatus::BlockReady(BlockContinuation::None);
            }
        }
    }

    fn finish(&mut self, _state: &mut State<'_>) -> usize {
        0
    }
}
