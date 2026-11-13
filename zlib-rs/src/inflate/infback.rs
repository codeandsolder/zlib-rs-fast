#![forbid(unsafe_code)]

use core::marker::PhantomData;

use crate::allocate::Allocator;
use crate::c_api::internal_state;
use crate::inflate::bitreader::BitReader;
use crate::inflate::{
    back_io::{BackInput, BackOutput},
    BackWindow, CodeLengthRepeat, Codes, Flags, InflateAllocOffsets, InflateConfig,
    InflateStateKind, InflateStream, Mode, State, Table, Window, INFLATE_FAST_MIN_HAVE,
    INFLATE_FAST_MIN_LEFT, INFLATE_STRICT, MAX_BITS,
};
use crate::traceln;
use crate::{c_api::z_stream, inflate::writer::BackWriter, ReturnCode};
use crate::{MAX_WBITS, MIN_WBITS};

/// Initialize callback-driven inflate using caller-owned window storage.
pub fn back_init(stream: &mut z_stream, config: InflateConfig, window: Window) -> ReturnCode {
    back_init_window(stream, config, BackWindow::from_window_storage(window))
}

pub(super) fn back_init_window(
    stream: &mut z_stream,
    config: InflateConfig,
    window: BackWindow<'_>,
) -> ReturnCode {
    if !(MIN_WBITS..=MAX_WBITS).contains(&config.window_bits) {
        return ReturnCode::StreamError;
    }
    if window.buffer_size() != 1usize << config.window_bits {
        return ReturnCode::StreamError;
    }

    stream.msg = core::ptr::null_mut();

    // for safety we must really make sure that alloc and free are consistent
    // this is a (slight) deviation from stock zlib. In this crate we pick the rust
    // allocator as the default, but `libz-rs-sys` configures the C allocator
    #[cfg(feature = "rust-allocator")]
    if stream.zalloc.is_none() || stream.zfree.is_none() {
        stream.configure_default_rust_allocator();
    }

    #[cfg(feature = "c-allocator")]
    if stream.zalloc.is_none() || stream.zfree.is_none() {
        stream.configure_default_c_allocator();
    }

    if stream.zalloc.is_none() || stream.zfree.is_none() {
        return ReturnCode::StreamError;
    }

    // Back-window geometry was validated before this conversion. Keep
    // the state field native u8 and report failure instead of narrowing.
    let Ok(wbits) = u8::try_from(config.window_bits) else {
        return ReturnCode::StreamError;
    };
    let (Some(zalloc), Some(zfree)) = (stream.zalloc, stream.zfree) else {
        return ReturnCode::StreamError;
    };

    let mut state = State::new(InflateStateKind::Back);
    // inflateBack uses the caller-provided window directly, so the allocation contains only State
    // plus alignment headroom.
    state.back_window = window;
    state.wbits = wbits;
    state.flags.update(Flags::SANE, true);

    let alloc = Allocator {
        zalloc,
        zfree,
        opaque: stream.opaque,
        _marker: PhantomData,
    };
    let allocs = InflateAllocOffsets::state_only();
    let Some(state_allocation) = allocs.allocate_state(&alloc, state) else {
        return ReturnCode::MemError;
    };
    stream.state = state_allocation.as_ptr().cast::<internal_state>();

    ReturnCode::Ok
}

/// Safe `inflateBack` state-machine core. Raw callback and initial-span contracts are bound before
/// entry; decoding itself only manipulates checked cursors, state, and the exclusive `BackWriter`.
pub(super) fn decode_back(
    strm: &mut InflateStream,
    mut input: BackInput,
    mut output: BackOutput,
) -> ReturnCode {
    let mut ret;

    /* Reset the state */
    strm.msg = core::ptr::null_mut();
    strm.state.mode = Mode::Type;
    strm.state.flags.update(Flags::IS_LAST_BLOCK, false);

    // Detach the caller-owned output/history window from `State` for the whole decode. Temporary
    // `Writer` views can then never alias a window still reachable through the simultaneously
    // borrowed `&mut State`. The single exit path below restores it before returning.
    let state = &mut *strm.state;
    let mut window = core::mem::replace(&mut state.back_window, BackWindow::empty());
    window.clear();

    'inf_leave: loop {
        match state.mode {
            Mode::Type => {
                if state.flags.contains(Flags::IS_LAST_BLOCK) {
                    input.next_byte_boundary();
                    state.mode = Mode::Done;
                    continue;
                }

                if let Err(code) = input.need_bits(3) {
                    ret = code;
                    break 'inf_leave;
                }

                let last = input.bits(1) != 0;
                state.flags.update(Flags::IS_LAST_BLOCK, last);
                input.drop_bits(1);

                match input.bits(2) {
                    0b00 => {
                        traceln!("inflate:     stored block (last = {last})");

                        input.drop_bits(2);
                        state.mode = Mode::Stored;
                        continue;
                    }
                    0b01 => {
                        traceln!("inflate:     fixed codes block (last = {last})");

                        state.len_table = Table {
                            codes: Codes::Fixed,
                            bits: 9,
                        };

                        state.dist_table = Table {
                            codes: Codes::Fixed,
                            bits: 5,
                        };

                        input.drop_bits(2);
                        state.mode = Mode::Len;
                        continue;
                    }
                    0b10 => {
                        traceln!("inflate:     dynamic codes block (last = {last})");

                        input.drop_bits(2);
                        state.mode = Mode::Table;
                        continue;
                    }
                    0b11 => {
                        traceln!("inflate:     invalid block type");

                        input.drop_bits(2);
                        state.mode = Mode::Bad;
                        state.bad("invalid block type\0");
                        continue;
                    }
                    _ => {
                        // LLVM will optimize this branch away
                        unreachable!("BitReader::bits(2) only yields a value of two bits, so this match is already exhaustive")
                    }
                }
            }
            Mode::Stored => {
                input.next_byte_boundary();
                if let Err(code) = input.need_bits(32) {
                    ret = code;
                    break 'inf_leave;
                }

                let Some(length) = super::stored_block_length(input.bits_u32(32)) else {
                    state.mode = Mode::Bad;
                    state.bad("invalid stored block lengths\0");
                    continue;
                };

                state.length = usize::from(length);
                traceln!("inflate:     stored length {}", state.length);

                input.init_bits();

                /* copy stored block from input to output */
                while state.length != 0 {
                    let mut copy = state.length;

                    if let Err(code) = input.ensure_available() {
                        ret = code;
                        break 'inf_leave;
                    }
                    if let Err(code) = output.flush_if_full(&mut window) {
                        ret = code;
                        break 'inf_leave;
                    }

                    copy = Ord::min(copy, input.available());
                    copy = Ord::min(copy, output.remaining(&window));
                    output.extend(&mut window, &input.as_slice()[..copy]);
                    input.advance(copy);
                    state.length -= copy;
                }

                state.mode = Mode::Type;
                continue;
            }
            Mode::Table => {
                if let Err(code) = input.need_bits(14) {
                    ret = code;
                    break 'inf_leave;
                }
                state.nlen = input.bits_usize(5) + 257;
                input.drop_bits(5);
                state.ndist = input.bits_usize(5) + 1;
                input.drop_bits(5);
                state.ncode = input.bits_usize(4) + 4;
                input.drop_bits(4);

                // TODO pkzit_bug_workaround
                if state.nlen > 286 || state.ndist > 30 {
                    state.mode = Mode::Bad;
                    state.bad("too many length or distance symbols\0");
                    continue;
                }

                traceln!("inflate:       table sizes ok");
                state.have = 0;

                while state.have < state.ncode {
                    if let Err(code) = input.need_bits(3) {
                        ret = code;
                        break 'inf_leave;
                    }
                    state.append_code_length_alphabet(input.bits_u16(3));
                    input.drop_bits(3);
                }

                if let Err(msg) = state.finish_code_length_alphabet() {
                    state.mode = Mode::Bad;
                    state.bad(msg);
                    continue;
                }

                traceln!("inflate:       table sizes ok");

                while state.have < state.nlen + state.ndist {
                    let here = loop {
                        let here = state.len_table_get(input.bits_usize(state.len_table.bits));
                        if here.bits <= input.bits_in_buffer() {
                            break here;
                        }

                        if let Err(code) = input.pull_byte() {
                            ret = code;

                            break 'inf_leave;
                        }
                    };

                    let here_bits = here.bits;

                    match here.val {
                        0..=15 => {
                            input.drop_bits(here_bits);
                            state.lens[state.have] = here.val;
                            state.have += 1;
                        }
                        symbol => {
                            let Some(repeat) = CodeLengthRepeat::from_symbol(symbol) else {
                                state.mode = Mode::Bad;
                                state.bad("invalid bit length repeat\0");
                                continue 'inf_leave;
                            };
                            if let Err(code) = input
                                .need_bits(usize::from(here_bits) + usize::from(repeat.extra_bits))
                            {
                                ret = code;
                                break 'inf_leave;
                            }
                            input.drop_bits(here_bits);
                            let extra = input.bits(usize::from(repeat.extra_bits));
                            input.drop_bits(repeat.extra_bits);
                            if let Err(msg) = state.append_code_length_repeat(repeat, extra) {
                                state.mode = Mode::Bad;
                                state.bad(msg);
                                continue 'inf_leave;
                            }
                        }
                    }
                }

                if let Err(msg) = state.finish_dynamic_trees() {
                    state.mode = Mode::Bad;
                    state.bad(msg);
                    continue 'inf_leave;
                }

                state.mode = Mode::Len;
            }
            Mode::Len => {
                if input.available() >= INFLATE_FAST_MIN_HAVE
                    && output.remaining(&window) >= INFLATE_FAST_MIN_LEFT
                {
                    let input_len = input.available();
                    let mut bit_reader =
                        BitReader::from_buffer(input.bit_buffer(), input.as_slice());
                    output.with_writer(&mut window, |writer| {
                        super::inflate_fast_help_impl::<false, _>(state, &mut bit_reader, writer);
                    });

                    let remaining = bit_reader.bytes_remaining();
                    let bits = bit_reader.into_buffer();
                    input.advance(input_len - remaining);
                    input.set_bit_buffer(bits);

                    continue 'inf_leave;
                }

                let len_table = match state.len_table.codes {
                    Codes::Fixed => &crate::inflate::inffixed_tbl::LENFIX[..],
                    Codes::CodeLengths => &state.codes_codes,
                    Codes::Len => &state.len_codes,
                    Codes::Dist => &state.dist_codes,
                };

                // get a literal, length, or end-of-block code
                let mut here;
                loop {
                    here = len_table[input.bits_usize(state.len_table.bits)];

                    if here.bits <= input.bits_in_buffer() {
                        break;
                    }

                    if let Err(code) = input.pull_byte() {
                        ret = code;

                        break 'inf_leave;
                    }
                }

                if here.op != 0 && here.op & 0xf0 == 0 {
                    let last = here;
                    loop {
                        let tmp = input.bits_u16(usize::from(last.bits + last.op));
                        here = len_table[usize::from(last.val + (tmp >> last.bits))];
                        if last.bits + here.bits <= input.bits_in_buffer() {
                            break;
                        }

                        if let Err(code) = input.pull_byte() {
                            ret = code;

                            break 'inf_leave;
                        }
                    }

                    input.drop_bits(last.bits);
                }

                input.drop_bits(here.bits);
                state.length = here.val as usize;

                if here.op == 0 {
                    if here.val >= 0x20 && here.val < 0x7f {
                        traceln!("inflate:         literal '{}'", here.val as u8 as char);
                    } else {
                        traceln!("inflate:         literal {:#04x}", here.val);
                    }
                    if let Err(code) = output.flush_if_full(&mut window) {
                        ret = code;
                        break 'inf_leave;
                    }
                    output.push(&mut window, state.length as u8);

                    state.mode = Mode::Len;
                    continue;
                } else if here.op & 32 != 0 {
                    // end of block

                    traceln!("inflate:         end of block");

                    state.mode = Mode::Type;
                    continue;
                } else if here.op & 64 != 0 {
                    state.mode = Mode::Bad;
                    state.bad("invalid literal/length code\0");
                    continue;
                } else {
                    // length code
                    state.extra = here.op & MAX_BITS;
                }

                // get extra bits, if any
                if state.extra != 0 {
                    if let Err(code) = input.need_bits(usize::from(state.extra)) {
                        ret = code;
                        break 'inf_leave;
                    }
                    state.length += input.bits_usize(usize::from(state.extra));
                    input.drop_bits(state.extra);
                }
                traceln!("inflate:         length {}", state.length);

                // get distance code
                let mut here;
                loop {
                    here = state.dist_table_get(input.bits_usize(state.dist_table.bits));
                    if here.bits <= input.bits_in_buffer() {
                        break;
                    }

                    if let Err(code) = input.pull_byte() {
                        ret = code;

                        break 'inf_leave;
                    }
                }

                if here.op & 0xf0 == 0 {
                    let last = here;

                    loop {
                        here = state.dist_table_get(
                            usize::from(last.val)
                                + (input.bits_usize(usize::from(last.bits + last.op)) >> last.bits),
                        );

                        if last.bits + here.bits <= input.bits_in_buffer() {
                            break;
                        }

                        if let Err(code) = input.pull_byte() {
                            ret = code;

                            break 'inf_leave;
                        }
                    }

                    input.drop_bits(last.bits);
                }

                input.drop_bits(here.bits);

                if here.op & 64 != 0 {
                    state.mode = Mode::Bad;
                    state.bad("invalid distance code\0");
                    continue 'inf_leave;
                }

                state.offset = here.val as usize;

                state.extra = here.op & MAX_BITS;

                let extra = state.extra;

                if extra > 0 {
                    if let Err(code) = input.need_bits(usize::from(extra)) {
                        ret = code;
                        break 'inf_leave;
                    }
                    state.offset += input.bits_usize(usize::from(extra));
                    input.drop_bits(extra);
                }

                if INFLATE_STRICT && state.offset > output.history_available(&window) {
                    state.mode = Mode::Bad;
                    state.bad("invalid distance too far back\0");
                    continue 'inf_leave;
                }

                traceln!("inflate:         distance {}", state.offset);

                loop {
                    if let Err(code) = output.flush_if_full(&mut window) {
                        ret = code;
                        break 'inf_leave;
                    }
                    let copy = match output.copy_match(
                        &mut window,
                        state.flags.contains(Flags::SANE),
                        state.offset,
                        state.length,
                    ) {
                        Ok(count) => count,
                        Err(msg) => {
                            state.mode = Mode::Bad;
                            state.bad(msg);
                            continue 'inf_leave;
                        }
                    };
                    state.length -= copy;
                    if state.length == 0 {
                        break;
                    }
                }

                continue 'inf_leave;
            }
            Mode::Done => {
                ret = ReturnCode::StreamEnd;
                break 'inf_leave;
            }
            Mode::Bad => {
                ret = ReturnCode::DataError;
                break 'inf_leave;
            }

            Mode::Head
            | Mode::Flags
            | Mode::Time
            | Mode::Os
            | Mode::ExLen
            | Mode::Extra
            | Mode::Name
            | Mode::Comment
            | Mode::HCrc
            | Mode::Sync
            | Mode::Mem
            | Mode::Length
            | Mode::TypeDo
            | Mode::CopyBlock
            | Mode::Check
            | Mode::Len_
            | Mode::Lit
            | Mode::LenExt
            | Mode::Dist
            | Mode::DistExt
            | Mode::Match
            | Mode::LenLens
            | Mode::CodeLens
            | Mode::DictId
            | Mode::Dict => {
                // All other states should be unreachable, and return StreamError.
                ret = ReturnCode::StreamError;
                break 'inf_leave;
            }
        }
    }

    if output.flush_remaining(&mut window).is_err() && ret == ReturnCode::StreamEnd {
        ret = ReturnCode::BufError;
    }

    let Ok(available) = u32::try_from(input.available()) else {
        // Preserve the caller's history window on every exit path.
        state.back_window = window;
        return ReturnCode::StreamError;
    };
    strm.next_in = input.next_ptr().cast_mut();
    strm.avail_in = available;
    state.back_window = window;

    ret
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn back_init_rejects_invalid_window_bits_before_geometry() {
        let mut storage = [0u8; 1];
        let window = Window::from_initialized_slice(&mut storage);
        let mut stream = z_stream::default();

        assert_eq!(
            back_init(
                &mut stream,
                InflateConfig {
                    window_bits: MIN_WBITS - 1,
                },
                window,
            ),
            ReturnCode::StreamError,
        );
    }

    #[test]
    fn back_init_rejects_mismatched_window_size() {
        let mut storage = [0u8; 1];
        let window = Window::from_initialized_slice(&mut storage);
        let mut stream = z_stream::default();

        assert_eq!(
            back_init(
                &mut stream,
                InflateConfig {
                    window_bits: MIN_WBITS,
                },
                window,
            ),
            ReturnCode::StreamError,
        );
    }

    #[test]
    fn back_writer_match_crosses_previous_window_into_current_output() -> Result<(), &'static str> {
        let mut storage = core::array::from_fn::<_, 32, _>(|i| i as u8);
        // The current callback window has already overwritten its first four bytes.
        storage[..4].copy_from_slice(&[100, 101, 102, 103]);
        let mut window = BackWindow::from_initialized_slice(&mut storage);
        window.mark_full_history();

        {
            let mut writer = BackWriter::new(&mut window, 4);
            writer.copy_match(true, 6, 8)?;
            assert_eq!(writer.len(), 12);
        }

        assert_eq!(&storage[4..12], &[30, 31, 100, 101, 102, 103, 30, 31]);
        Ok(())
    }

    #[test]
    fn back_writer_rejects_distance_beyond_configured_window() {
        let mut storage = [0u8; 32];
        let mut window = BackWindow::from_initialized_slice(&mut storage);
        window.mark_full_history();

        let mut writer = BackWriter::new(&mut window, 4);
        assert_eq!(
            writer.copy_match(true, 33, 3),
            Err("invalid distance too far back\0")
        );
    }

    #[test]
    fn back_writer_rejects_history_before_first_full_window() {
        let mut storage = [0u8; 32];
        storage[..4].copy_from_slice(&[1, 2, 3, 4]);
        let mut window = BackWindow::from_initialized_slice(&mut storage);

        let mut writer = BackWriter::new(&mut window, 4);
        assert_eq!(
            writer.copy_match(true, 5, 3),
            Err("invalid distance too far back\0")
        );
    }
}

pub fn back_end<'a>(strm: &'a mut InflateStream<'a>) {
    // With infback the window is user-supplied, so we mustn't try to free it.
    let _ = core::mem::replace(&mut strm.state.back_window, BackWindow::empty());
    crate::inflate::end(strm);
}
