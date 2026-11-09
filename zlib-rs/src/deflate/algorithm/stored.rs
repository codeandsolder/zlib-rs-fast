#![forbid(unsafe_code)]

use crate::{
    deflate::{
        drain_pending_output, emit_stored_block_header, unflushed_block_span, zng_tr_stored_block,
        BlockState, Compressor, State, MAX_STORED,
    },
    DeflateFlush,
};

/// Stored output may reference only bytes still resident in the window.
///
/// A negative DEFLATE block origin is valid for an already-compressed block,
/// but stored-mode copying must never reinterpret it as an unsigned offset.
fn resident_origin(state: &State<'_>) -> usize {
    let (origin, _) = unflushed_block_span(state.strstart, state.block_start);
    let Some(origin) = origin else {
        unreachable!("stored block start must be present in the sliding window");
    };
    origin
}

/// Convert only at the State boundary, after validating the next resident
/// offset against the current initialized position.
fn set_resident_origin(state: &mut State<'_>, next: usize) {
    assert!(
        next <= state.strstart,
        "stored block origin moved beyond the window cursor"
    );
    let Ok(next) = isize::try_from(next) else {
        unreachable!("stored block origin exceeds signed index range");
    };
    state.block_start = next;
}

fn advance_resident_origin(state: &mut State<'_>, count: usize) {
    let Some(next) = resident_origin(state).checked_add(count) else {
        unreachable!("stored block origin arithmetic overflow");
    };
    set_resident_origin(state, next);
}

/// Number of output bytes needed for a stored-block header, accounting for
/// buffered prefix bits, the three-bit block marker, byte alignment and LEN/NLEN.
fn stored_header_size(bits_valid: u8) -> usize {
    assert!(bits_valid <= 64, "invalid bit reservoir occupancy");
    (usize::from(bits_valid) + 42) / 8
}

pub fn deflate_stored(stream: &mut Compressor, flush: DeflateFlush) -> BlockState {
    // Stored-mode output must be backed by bytes still in the sliding window.
    // The compressed-level transition path flushes before switching modes;
    // negative block_start is only meaningful for already-flushed history.
    let _ = resident_origin(stream.state);
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
        have = stored_header_size(stream.state.bit_writer.bits_valid);

        // we need room for at least the header
        if (stream.output.avail_out as usize) < have {
            break;
        }

        let left = stream.state.strstart - resident_origin(stream.state);
        // avail_in is a C uInt. On a 32-bit target its maximum can be
        // usize::MAX, so adding buffered bytes must not wrap the total.
        let available = left.saturating_add(stream.input.avail_in as usize);

        have = stream.output.avail_out as usize - have;

        if len > available {
            // limit len to the input
            len = available;
        }

        len = Ord::min(len, have);

        // If the stored block would be less than min_block in length, or if
        // unable to copy all of the available input when flushing, then try
        // copying to the window and the pending buffer instead. Also don't
        // write an empty block when flushing -- deflate() does that.
        if len < min_block
            && ((len == 0 && flush != DeflateFlush::Finish)
                || flush == DeflateFlush::NoFlush
                || len != available)
        {
            break;
        }

        // Emit the actual stored header directly, preserving pending bits
        // and debug accounting without constructing and rewriting a dummy block.
        last = flush == DeflateFlush::Finish && len == available;
        emit_stored_block_header(stream.state, len, last);

        // Write the stored block header bytes.
        let _ = drain_pending_output(stream.state, &mut stream.output);

        // Update debugging counts for the data about to be copied.
        #[cfg(feature = "ZLIB_DEBUG")]
        {
            stream.state.bit_writer.cmpr_bits_add(len << 3);
            stream.state.bit_writer.sent_bits_add(len << 3);
        }

        if left > 0 {
            let left = left.min(len);
            let start = resident_origin(stream.state);
            stream.copy_window_to_output(start..start + left);
            advance_resident_origin(stream.state, left);
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
        set_resident_origin(stream.state, stream.state.strstart);
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
        && stream.state.strstart == resident_origin(stream.state)
    {
        return BlockState::BlockDone;
    }

    // Fill the window with any remaining input
    let mut have = stream.state.window_size - stream.state.strstart;
    if stream.input.avail_in as usize > have && resident_origin(stream.state) >= stream.state.w_size
    {
        // slide the window down
        let state = &mut stream.state;
        let next_origin = resident_origin(state) - state.w_size;
        state.strstart -= state.w_size;
        set_resident_origin(state, next_origin);

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
    let have = stored_header_size(state.bit_writer.bits_valid);

    // maximum stored block length that will fit in pending:
    let have = Ord::min(state.bit_writer.pending.capacity() - have, MAX_STORED);
    let min_block = Ord::min(have, state.w_size);
    let left = state.strstart - resident_origin(state);

    if left >= min_block
        || ((left > 0 || flush == DeflateFlush::Finish)
            && flush != DeflateFlush::NoFlush
            && stream.input.avail_in == 0
            && left <= have)
    {
        let len = left.min(have);
        last = flush == DeflateFlush::Finish && stream.input.avail_in == 0 && len == left;

        let start = resident_origin(state);
        zng_tr_stored_block(state, start..start + len, last);

        advance_resident_origin(state, len);
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
