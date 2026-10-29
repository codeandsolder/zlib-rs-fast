#![forbid(unsafe_code)]

use super::{ensure_input, InputStatus};
use crate::deflate::hash_calc::StandardHashCalc;
use crate::{
    deflate::{
        BlockState, BlockType, Compressor, State, StaticTreeDesc, MIN_LOOKAHEAD, STD_MAX_MATCH,
        STD_MIN_MATCH, WANT_MIN_MATCH,
    },
    DeflateFlush,
};

pub fn deflate_quick(stream: &mut Compressor, flush: DeflateFlush) -> BlockState {
    let last = flush == DeflateFlush::Finish;

    // A finishing call must move any open non-final block out of the way and explicitly open the
    // final static block, even when there is no input left to encode.
    if last && stream.state.block_open != 2 {
        if let Some(blocked) = end_block(stream, false) {
            return blocked;
        }
        start_block(stream.state, true);
    }

    loop {
        if pending_needs_drain(stream.state) {
            stream.flush_pending();
            if stream.output.avail_out == 0 {
                return if last
                    && stream.input.avail_in == 0
                    && stream.state.bit_writer.bits_valid == 0
                    && stream.state.block_open == 0
                {
                    BlockState::FinishStarted
                } else {
                    BlockState::NeedMore
                };
            }
        }

        match ensure_input(stream, MIN_LOOKAHEAD, flush) {
            InputStatus::Ready => {}
            InputStatus::NeedMore => return BlockState::NeedMore,
            InputStatus::Exhausted => break,
        }

        // Do not emit an empty non-final block. A final empty block, when needed, was opened above.
        if stream.state.block_open == 0 {
            start_block(stream.state, last);
        }

        emit_next(stream.state);
    }

    stream.state.insert = Ord::min(stream.state.strstart, STD_MIN_MATCH - 1);

    if let Some(blocked) = end_block(stream, last) {
        return blocked;
    }

    if last {
        BlockState::FinishDone
    } else {
        BlockState::BlockDone
    }
}

#[inline]
fn start_block(state: &mut State<'_>, last: bool) {
    state.bit_writer.emit_tree(BlockType::StaticTrees, last);
    state.block_open = 1 + u8::from(last);
    state.block_start = state.strstart as isize;
}

fn end_block(stream: &mut Compressor<'_, '_>, last: bool) -> Option<BlockState> {
    if stream.state.block_open == 0 {
        return None;
    }

    {
        let state = &mut stream.state;
        state
            .bit_writer
            .emit_end_block_and_align(&StaticTreeDesc::L.static_tree, last);
        state.block_open = 0;
        state.block_start = state.strstart as isize;
    }

    stream.flush_pending();
    (stream.output.avail_out == 0).then_some(if last {
        BlockState::FinishStarted
    } else {
        BlockState::NeedMore
    })
}

#[inline]
fn pending_needs_drain(state: &State<'_>) -> bool {
    state.bit_writer.pending.pending + usize::from(State::BIT_BUF_SIZE.div_ceil(8))
        >= state.pending_buf_size()
}

fn emit_next(state: &mut State<'_>) {
    if state.lookahead >= WANT_MIN_MATCH {
        let str_val = first_four_bytes(&state.window.filled()[state.strstart..]);
        let hash_head = StandardHashCalc::quick_insert_value(state, state.strstart, str_val);
        let distance = state.strstart as isize - hash_head as isize;

        if (1..=state.max_dist() as isize).contains(&distance) {
            let match_start = &state.window.filled()[usize::from(hash_head)..];
            if str_val == first_four_bytes(match_start) {
                let mut match_len = crate::deflate::compare256::compare256_slice(
                    &state.window.filled()[state.strstart + 2..],
                    &match_start[2..],
                ) + 2;

                if match_len >= WANT_MIN_MATCH {
                    match_len = match_len.min(state.lookahead).min(STD_MAX_MATCH);
                    let distance = u16::try_from(distance)
                        .expect("DEFLATE window distances always fit in 16 bits");
                    state
                        .bit_writer
                        .emit_dist_static((match_len - STD_MIN_MATCH) as u8, distance);
                    state.lookahead -= match_len;
                    state.strstart += match_len;
                    return;
                }
            }
        }

        state
            .bit_writer
            .emit_lit(StaticTreeDesc::L.static_tree, str_val as u8);
    } else {
        let literal = state.window.filled()[state.strstart];
        state
            .bit_writer
            .emit_lit(StaticTreeDesc::L.static_tree, literal);
    }

    state.strstart += 1;
    state.lookahead -= 1;
}

#[inline]
fn first_four_bytes(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(
        bytes[..4]
            .try_into()
            .expect("quick deflate only probes four initialized bytes"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_four_bytes_is_little_endian() {
        assert_eq!(first_four_bytes(&[1, 2, 3, 4, 5]), 0x0403_0201);
    }
}
