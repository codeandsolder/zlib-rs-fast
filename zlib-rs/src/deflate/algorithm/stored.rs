#![forbid(unsafe_code)]

use crate::{
    deflate::{drain_pending_output, zng_tr_stored_block, BlockState, Compressor, MAX_STORED},
    DeflateFlush,
};

pub fn deflate_stored(stream: &mut Compressor, flush: DeflateFlush) -> BlockState {
    // Smallest worthy block size when not flushing or finishing. By default
    // this is 32K. This can be as small as 507 bytes for memLevel == 1. For
    // large input and output buffers, the stored block size will be larger.
    let min_block = Ord::min(
        stream.state.bit_writer.pending.capacity() - 5,
        stream.state.w_size,
    );

    // Copy as many min_block or larger stored blocks directly to next_out as
    // possible. If flushing, copy the remaining available input to next_out as
    // stored blocks, if there is enough space.

    // unsigned len, left, have, last = 0;
    let mut have;
    let mut last = false;
    let mut used = stream.input.avail_in;
    loop {
        // maximum deflate stored block length
        let mut len = MAX_STORED;

        // number of header bytes
        have = ((stream.state.bit_writer.bits_valid + 42) / 8) as usize;

        // we need room for at least the header
        if stream.output.avail_out < have as u32 {
            break;
        }

        let left = stream.state.strstart as isize - stream.state.block_start;
        let left = Ord::max(0, left) as usize;

        have = stream.output.avail_out as usize - have;

        if len > left + stream.input.avail_in as usize {
            // limit len to the input
            len = left + stream.input.avail_in as usize;
        }

        len = Ord::min(len, have);

        // If the stored block would be less than min_block in length, or if
        // unable to copy all of the available input when flushing, then try
        // copying to the window and the pending buffer instead. Also don't
        // write an empty block when flushing -- deflate() does that.
        if len < min_block
            && ((len == 0 && flush != DeflateFlush::Finish)
                || flush == DeflateFlush::NoFlush
                || len != left + stream.input.avail_in as usize)
        {
            break;
        }

        // Make a dummy stored block in pending to get the header bytes,
        // including any pending bits. This also updates the debugging counts.
        last = flush == DeflateFlush::Finish && len == left + stream.input.avail_in as usize;
        zng_tr_stored_block(stream.state, 0..0, last);

        /* Replace the lengths in the dummy stored block with len. */
        stream.state.bit_writer.pending.truncate_tail(4);
        stream
            .state
            .bit_writer
            .pending
            .extend(&(len as u16).to_le_bytes());
        stream
            .state
            .bit_writer
            .pending
            .extend(&(!len as u16).to_le_bytes());

        // Write the stored block header bytes.
        let _ = drain_pending_output(stream.state, &mut stream.output);

        // Update debugging counts for the data about to be copied.
        stream.state.bit_writer.cmpr_bits_add(len << 3);
        stream.state.bit_writer.sent_bits_add(len << 3);

        if left > 0 {
            let left = left.min(len);
            let start = stream.state.block_start as usize;
            stream.copy_window_to_output(start..start + left);
            stream.state.block_start += left as isize;
            len -= left;
        }

        // Copy uncompressed bytes directly from input to output, updating the check value.
        if len > 0 {
            stream.copy_input_to_output(len);
        }

        if last {
            break;
        }
    }

    // Update the sliding window with the last s->w_size bytes of the copied
    // data, or append all of the copied data to the existing window if less
    // than s->w_size bytes were copied. Also update the number of bytes to
    // insert in the hash tables, in the event that deflateParams() switches to
    // a non-zero compression level.
    used -= stream.input.avail_in; /* number of input bytes directly copied */

    if used > 0 {
        let used = used as usize;
        if used >= stream.state.w_size {
            // Supplant the previous history with the last full window of directly copied input.
            let w_size = stream.state.w_size;
            stream.state.matches = 2; // clear hash on a later compressed-level transition
            stream.copy_recent_input_to_window(0..w_size, w_size);
            stream.state.strstart = w_size;
            stream.state.insert = w_size;
        } else {
            if stream.state.window_size - stream.state.strstart <= used {
                // Slide the initialized portion of the history window down.
                let state = &mut stream.state;
                state.strstart -= state.w_size;
                state.window.slide_down(state.w_size);
                state.matches = (state.matches + 1).min(2); // pending slide_hash
                state.insert = state.insert.min(state.strstart);
            }

            let start = stream.state.strstart;
            stream.copy_recent_input_to_window(start..start + used, used);
            let state = &mut stream.state;
            state.strstart += used;
            state.insert += used.min(state.w_size - state.insert);
        }
        stream.state.block_start = stream.state.strstart as isize;
    }

    // If the last block was written to next_out, then done.
    if last {
        stream.state.bit_writer.bits_used = 8;
        return BlockState::FinishDone;
    }

    // If flushing and all input has been consumed, then done.
    if flush != DeflateFlush::NoFlush
        && flush != DeflateFlush::Finish
        && stream.input.avail_in == 0
        && stream.state.strstart as isize == stream.state.block_start
    {
        return BlockState::BlockDone;
    }

    // Fill the window with any remaining input
    let mut have = stream.state.window_size - stream.state.strstart;
    if stream.input.avail_in as usize > have
        && stream.state.block_start >= stream.state.w_size as isize
    {
        // slide the window down
        let state = &mut stream.state;
        state.block_start -= state.w_size as isize;
        state.strstart -= state.w_size;

        // Preserve only the logically initialized upper-half history.
        state.window.slide_down(state.w_size);

        if state.matches < 2 {
            // add a pending slide_hash
            state.matches += 1;
        }

        have += state.w_size; // more space now
        state.insert = Ord::min(state.insert, state.strstart);
    }

    let have = Ord::min(have, stream.input.avail_in as usize);
    if have > 0 {
        stream.read_buf_window(stream.state.strstart, have);

        let state = &mut stream.state;
        state.strstart += have;
        state.insert += Ord::min(have, state.w_size - state.insert);
    }

    // There was not enough avail_out to write a complete worthy or flushed
    // stored block to next_out. Write a stored block to pending instead, if we
    // have enough input for a worthy block, or if flushing and there is enough
    // room for the remaining input as a stored block in the pending buffer.

    // number of header bytes
    let state = &mut stream.state;
    let have = ((state.bit_writer.bits_valid + 42) >> 3) as usize;

    // maximum stored block length that will fit in pending:
    let have = Ord::min(state.bit_writer.pending.capacity() - have, MAX_STORED);
    let min_block = Ord::min(have, state.w_size);
    let left = state.strstart as isize - state.block_start;

    if left >= min_block as isize
        || ((left > 0 || flush == DeflateFlush::Finish)
            && flush != DeflateFlush::NoFlush
            && stream.input.avail_in == 0
            && left <= have as isize)
    {
        // The branch predicate above requires `left >= 0`: either it is at least
        // `min_block`, or the flush arm requires `left > 0`. `deflateParams()`
        // also flushes and resolves any algorithm transition before entering stored mode.
        debug_assert!(left >= 0);
        let len = Ord::min(left as usize, have);
        last =
            flush == DeflateFlush::Finish && stream.input.avail_in == 0 && len == (left as usize);

        let range = state.block_start as usize..state.block_start as usize + len;
        zng_tr_stored_block(state, range, last);

        state.block_start += len as isize;
        let _ = drain_pending_output(stream.state, &mut stream.output);
    }

    // We've done all we can with the available input and output.
    if last {
        stream.state.bit_writer.bits_used = 8;
        BlockState::FinishStarted
    } else {
        BlockState::NeedMore
    }
}
