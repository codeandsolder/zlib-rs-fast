#![forbid(unsafe_code)]

use core::marker::PhantomData;

use crate::allocate::Allocator;
use crate::c_api::internal_state;
use crate::inflate::bitreader::BitReader;
use crate::inflate::inftrees::{inflate_table, CodeType, InflateTable};
use crate::inflate::{
    back_io::{BackInput, BackOutput},
    BackWindow, Codes, Flags, InflateAllocOffsets, InflateConfig, InflateStateKind, InflateStream,
    Mode, State, Table, Window, INFLATE_FAST_MIN_HAVE, INFLATE_FAST_MIN_LEFT, INFLATE_STRICT,
    MAX_BITS, MAX_DIST_EXTRA_BITS,
};
use crate::traceln;
use crate::{c_api::z_stream, inflate::writer::BackWriter, ReturnCode};
use crate::{MAX_WBITS, MIN_WBITS};

/// Initialize callback-driven inflate using an initialized caller-owned window.
pub fn back_init(stream: &mut z_stream, config: InflateConfig, window: Window) -> ReturnCode {
    back_init_window(stream, config, BackWindow::from_initialized_window(window))
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
        stream.configure_default_rust_allocator()
    }

    #[cfg(feature = "c-allocator")]
    if stream.zalloc.is_none() || stream.zfree.is_none() {
        stream.configure_default_c_allocator()
    }

    if stream.zalloc.is_none() || stream.zfree.is_none() {
        return ReturnCode::StreamError;
    }

    let mut state = State::new(InflateStateKind::Back);
    // inflateBack uses the caller-provided window directly, so the allocation contains only State
    // plus alignment headroom.
    state.back_window = window;
    state.wbits = config.window_bits as u8;
    state.flags.update(Flags::SANE, true);

    let alloc = Allocator {
        zalloc: stream.zalloc.unwrap(),
        zfree: stream.zfree.unwrap(),
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

                if input.hold() as u16 != !((input.hold() >> 16) as u16) {
                    state.mode = Mode::Bad;
                    state.bad("invalid stored block lengths\0");
                    continue;
                }

                state.length = input.hold() as usize & 0xFFFF;
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
                state.nlen = input.bits(5) as usize + 257;
                input.drop_bits(5);
                state.ndist = input.bits(5) as usize + 1;
                input.drop_bits(5);
                state.ncode = input.bits(4) as usize + 4;
                input.drop_bits(4);

                // TODO pkzit_bug_workaround
                if state.nlen > 286 || state.ndist > 30 {
                    state.mode = Mode::Bad;
                    state.bad("too many length or distance symbols\0");
                    continue;
                }

                traceln!("inflate:       table sizes ok");
                state.have = 0;

                // permutation of code lengths ;
                const ORDER: [u8; 19] = [
                    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
                ];

                while state.have < state.ncode {
                    if let Err(code) = input.need_bits(3) {
                        ret = code;
                        break 'inf_leave;
                    }
                    state.lens[usize::from(ORDER[state.have])] = input.bits(3) as u16;
                    state.have += 1;
                    input.drop_bits(3);
                }

                while state.have < 19 {
                    state.lens[usize::from(ORDER[state.have])] = 0;
                    state.have += 1;
                }

                let InflateTable::Success { root, used } = inflate_table(
                    CodeType::Codes,
                    &state.lens[..19],
                    &mut state.codes_codes,
                    7,
                    &mut state.work,
                ) else {
                    state.mode = Mode::Bad;
                    state.bad("invalid code lengths set\0");
                    continue;
                };

                state.next = used;
                state.len_table.codes = Codes::Codes;
                state.len_table.bits = root;

                traceln!("inflate:       table sizes ok");
                state.have = 0;

                while state.have < state.nlen + state.ndist {
                    let here = loop {
                        let here = state.len_table_get(input.bits(state.len_table.bits) as usize);
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
                        16 => {
                            if let Err(code) = input.need_bits(usize::from(here_bits) + 2) {
                                ret = code;
                                break 'inf_leave;
                            }
                            input.drop_bits(here_bits);
                            if state.have == 0 {
                                state.mode = Mode::Bad;
                                state.bad("invalid bit length repeat\0");
                                continue 'inf_leave;
                            }

                            let len = state.lens[state.have - 1];
                            let copy = 3 + input.bits(2) as usize;
                            input.drop_bits(2);

                            if state.have + copy > state.nlen + state.ndist {
                                state.mode = Mode::Bad;
                                state.bad("invalid bit length repeat\0");
                                continue 'inf_leave;
                            }

                            state.lens[state.have..][..copy].fill(len);
                            state.have += copy;
                        }
                        17 => {
                            if let Err(code) = input.need_bits(usize::from(here_bits) + 3) {
                                ret = code;
                                break 'inf_leave;
                            }
                            input.drop_bits(here_bits);
                            let copy = 3 + input.bits(3) as usize;
                            input.drop_bits(3);

                            if state.have + copy > state.nlen + state.ndist {
                                state.mode = Mode::Bad;
                                state.bad("invalid bit length repeat\0");
                                continue 'inf_leave;
                            }

                            state.lens[state.have..][..copy].fill(0);
                            state.have += copy;
                        }
                        18.. => {
                            if let Err(code) = input.need_bits(usize::from(here_bits) + 7) {
                                ret = code;
                                break 'inf_leave;
                            }
                            input.drop_bits(here_bits);
                            let copy = 11 + input.bits(7) as usize;
                            input.drop_bits(7);

                            if state.have + copy > state.nlen + state.ndist {
                                state.mode = Mode::Bad;
                                state.bad("invalid bit length repeat\0");
                                continue 'inf_leave;
                            }

                            state.lens[state.have..][..copy].fill(0);
                            state.have += copy;
                        }
                    }
                }

                // check for end-of-block code (better have one)
                if state.lens[256] == 0 {
                    state.mode = Mode::Bad;
                    state.bad("invalid code -- missing end-of-block\0");
                    continue 'inf_leave;
                }

                // build code tables

                let InflateTable::Success { root, used } = inflate_table(
                    CodeType::Lens,
                    &state.lens[..state.nlen],
                    &mut state.len_codes,
                    10,
                    &mut state.work,
                ) else {
                    state.mode = Mode::Bad;
                    state.bad("invalid literal/lengths set\0");
                    continue 'inf_leave;
                };

                state.len_table.codes = Codes::Len;
                state.len_table.bits = root;
                state.next = used;

                let InflateTable::Success { root, used } = inflate_table(
                    CodeType::Dists,
                    &state.lens[state.nlen..][..state.ndist],
                    &mut state.dist_codes,
                    9,
                    &mut state.work,
                ) else {
                    state.mode = Mode::Bad;
                    state.bad("invalid distances set\0");
                    continue 'inf_leave;
                };

                state.dist_table.bits = root;
                state.dist_table.codes = Codes::Dist;
                state.next += used;

                state.mode = Mode::Len;
            }
            Mode::Len => {
                if input.available() >= INFLATE_FAST_MIN_HAVE
                    && output.remaining(&window) >= INFLATE_FAST_MIN_LEFT
                {
                    let input_len = input.available();
                    let mut bit_reader =
                        BitReader::from_buffer(input.bit_buffer(), input.as_slice());
                    let mut writer = BackWriter::new(&mut window, output.len());
                    inflate_fast_back(state, &mut bit_reader, &mut writer);

                    let remaining = bit_reader.bytes_remaining();
                    let bits = bit_reader.into_buffer();
                    input.advance(input_len - remaining);
                    input.set_bit_buffer(bits);
                    let filled = writer.len();
                    drop(writer);
                    output.set_len(filled, &window);

                    continue 'inf_leave;
                }

                let len_table = match state.len_table.codes {
                    Codes::Fixed => &crate::inflate::inffixed_tbl::LENFIX[..],
                    Codes::Codes => &state.codes_codes,
                    Codes::Len => &state.len_codes,
                    Codes::Dist => &state.dist_codes,
                };

                // get a literal, length, or end-of-block code
                let mut here;
                loop {
                    here = len_table[input.bits(state.len_table.bits) as usize];

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
                        let tmp = input.bits((last.bits + last.op) as usize) as u16;
                        here = len_table[(last.val + (tmp >> last.bits)) as usize];
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
                    state.extra = (here.op & MAX_BITS) as usize;
                }

                // get extra bits, if any
                if state.extra != 0 {
                    if let Err(code) = input.need_bits(state.extra) {
                        ret = code;
                        break 'inf_leave;
                    }
                    state.length += input.bits(state.extra) as usize;
                    input.drop_bits(state.extra as u8);
                }
                traceln!("inflate:         length {}", state.length);

                // get distance code
                let mut here;
                loop {
                    here = state.dist_table_get(input.bits(state.dist_table.bits) as usize);
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
                            last.val as usize
                                + ((input.bits((last.bits + last.op) as usize) as usize)
                                    >> last.bits),
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

                state.extra = (here.op & MAX_BITS) as usize;

                let extra = state.extra;

                if extra > 0 {
                    if let Err(code) = input.need_bits(extra) {
                        ret = code;
                        break 'inf_leave;
                    }
                    state.offset += input.bits(extra) as usize;
                    input.drop_bits(extra as u8);
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
                    let mut writer = BackWriter::new(&mut window, output.len());
                    let copy = Ord::min(state.length, writer.remaining());
                    if let Err(msg) =
                        writer.copy_match(state.flags.contains(Flags::SANE), state.offset, copy)
                    {
                        state.mode = Mode::Bad;
                        state.bad(msg);
                        continue 'inf_leave;
                    }
                    let filled = writer.len();
                    drop(writer);
                    output.set_len(filled, &window);
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

    strm.next_in = input.next_ptr().cast_mut();
    strm.avail_in = input.available() as u32;
    state.back_window = window;

    ret
}

#[inline(always)]
fn inflate_fast_back(
    state: &mut State,
    bit_reader: &mut BitReader<'_>,
    writer: &mut BackWriter<'_>,
) {
    debug_assert!(bit_reader.bytes_remaining() >= 15);

    let lcode = state.len_table_ref();
    let dcode = state.dist_table_ref();

    // IDEA: use const generics for the bits here?
    let lmask = (1u64 << state.len_table.bits) - 1;
    let dmask = (1u64 << state.dist_table.bits) - 1;

    let mut bad = None;

    if bit_reader.bits_in_buffer() < 10 {
        debug_assert!(bit_reader.bytes_remaining() >= 15);
        // The fast-path entry invariant guarantees at least eight bytes remain for `refill`.
        bit_reader.refill();
    }
    // We had at least 15 bytes in the slice, plus whatever was in the buffer. After filling the
    // buffer from the slice, we now have at least 8 bytes remaining in the slice, plus a full buffer.
    debug_assert!(
        bit_reader.bytes_remaining() >= 8 && bit_reader.bytes_remaining_including_buffer() >= 15
    );

    'outer: loop {
        // This condition is ensured above for the first iteration of the `outer` loop. For
        // subsequent iterations, the loop continuation condition is
        // `bit_reader.bytes_remaining_including_buffer() > 15`. And because the buffer
        // contributes at most 7 bytes to the result of bit_reader.bytes_remaining_including_buffer(),
        // that means that the slice contains at least 8 bytes.
        debug_assert!(
            bit_reader.bytes_remaining() >= 8
                && bit_reader.bytes_remaining_including_buffer() >= 15
        );

        let mut here = {
            let bits = bit_reader.bits_in_buffer();
            let hold = bit_reader.hold();

            // Safety: As described in the comments for the debug_assert at the start of
            // the `outer` loop, it is guaranteed that `bit_reader.bytes_remaining() >= 8` here,
            // which satisfies `refill`'s eight-byte requirement. And, because the total
            // number of bytes in `bit_reader`'s buffer plus its slice is at least 15, and
            // `refill` moves at most 7 bytes from the slice to the buffer, the slice will still
            // contain at least 8 bytes after this `refill` call.
            bit_reader.refill();
            // After the refill, there will be at least 8 bytes left in the bit_reader's slice.
            debug_assert!(bit_reader.bytes_remaining() >= 8);

            // in most cases, the read can be interleaved with the logic
            // based on benchmarks this matters in practice. wild.
            if bits as usize >= state.len_table.bits {
                lcode[(hold & lmask) as usize]
            } else {
                lcode[(bit_reader.hold() & lmask) as usize]
            }
        };

        if here.op == 0 {
            writer.push(here.val as u8);
            bit_reader.drop_bits(here.bits);
            here = lcode[(bit_reader.hold() & lmask) as usize];

            if here.op == 0 {
                writer.push(here.val as u8);
                bit_reader.drop_bits(here.bits);
                here = lcode[(bit_reader.hold() & lmask) as usize];
            }
        }

        'dolen: loop {
            bit_reader.drop_bits(here.bits);
            let op = here.op;

            if op == 0 {
                writer.push(here.val as u8);
            } else if op & 16 != 0 {
                let op = op & MAX_BITS;
                let len = here.val + bit_reader.bits(op as usize) as u16;
                bit_reader.drop_bits(op);

                here = dcode[(bit_reader.hold() & dmask) as usize];

                // we have two fast-path loads: 10+10 + 15+5 = 40,
                // but we may need to refill here in the worst case
                if bit_reader.bits_in_buffer() < MAX_BITS + MAX_DIST_EXTRA_BITS {
                    debug_assert!(bit_reader.bytes_remaining() >= 8);
                    // Safety: On the first iteration of the `dolen` loop, we can rely on the
                    // invariant documented for the previous `refill` call above: after that
                    // operation, `bit_reader.bytes_remining >= 8`, which satisfies the safety
                    // requirement for this call. For subsequent iterations, this invariant
                    // remains true because nothing else within the `dolen` loop consumes data
                    // from the slice.
                    bit_reader.refill();
                }

                'dodist: loop {
                    bit_reader.drop_bits(here.bits);
                    let op = here.op;

                    if op & 16 != 0 {
                        let op = op & MAX_BITS;
                        let dist = here.val + bit_reader.bits(op as usize) as u16;

                        if INFLATE_STRICT && dist as usize > state.dmax {
                            bad = Some("invalid distance too far back\0");
                            state.mode = Mode::Bad;
                            break 'outer;
                        }

                        bit_reader.drop_bits(op);

                        if let Err(msg) = writer.copy_match(
                            state.flags.contains(Flags::SANE),
                            dist as usize,
                            len as usize,
                        ) {
                            bad = Some(msg);
                            state.mode = Mode::Bad;
                            break 'outer;
                        }
                    } else if (op & 64) == 0 {
                        // 2nd level distance code
                        here = dcode[(here.val + bit_reader.bits(op as usize) as u16) as usize];
                        continue 'dodist;
                    } else {
                        bad = Some("invalid distance code\0");
                        state.mode = Mode::Bad;
                        break 'outer;
                    }

                    break 'dodist;
                }
            } else if (op & 64) == 0 {
                // 2nd level length code
                here = lcode[(here.val + bit_reader.bits(op as usize) as u16) as usize];
                continue 'dolen;
            } else if op & 32 != 0 {
                // end of block
                state.mode = Mode::Type;
                break 'outer;
            } else {
                bad = Some("invalid literal/length code\0");
                state.mode = Mode::Bad;
                break 'outer;
            }

            break 'dolen;
        }

        // For normal `inflate`, include the bits in the bit_reader buffer in the count of available bytes.
        let remaining = bit_reader.bytes_remaining();
        if remaining >= INFLATE_FAST_MIN_HAVE && writer.remaining() >= INFLATE_FAST_MIN_LEFT {
            continue;
        }

        break 'outer;
    }

    // return unused bytes (on entry, bits < 8, so in won't go too far back)
    bit_reader.return_unused_bytes();

    if let Some(error_message) = bad {
        debug_assert!(matches!(state.mode, Mode::Bad));
        state.bad(error_message);
    }
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
    fn back_writer_match_crosses_previous_window_into_current_output() {
        let mut storage = core::array::from_fn::<_, 32, _>(|i| i as u8);
        // The current callback window has already overwritten its first four bytes.
        storage[..4].copy_from_slice(&[100, 101, 102, 103]);
        let mut window = BackWindow::from_initialized_slice(&mut storage);
        window.mark_full_history();

        let mut writer = BackWriter::new(&mut window, 4);
        writer.copy_match(true, 6, 8).unwrap();
        assert_eq!(writer.len(), 12);
        drop(writer);

        assert_eq!(&storage[4..12], &[30, 31, 100, 101, 102, 103, 30, 31]);
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
