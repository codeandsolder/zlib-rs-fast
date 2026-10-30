use core::ffi::c_void;
use core::marker::PhantomData;

use crate::allocate::Allocator;
use crate::c_api::{in_func, internal_state, out_func};
use crate::inflate::bitreader::{BitBuffer, BitReader};
use crate::inflate::inftrees::{inflate_table, CodeType, InflateTable};
use crate::inflate::{
    Codes, Flags, InflateAllocOffsets, InflateConfig, InflateStream, Mode, State, Table, Window,
    INFLATE_FAST_MIN_HAVE, INFLATE_FAST_MIN_LEFT, INFLATE_STRICT, MAX_BITS, MAX_DIST_EXTRA_BITS,
};
use crate::traceln;
use crate::{c_api::z_stream, inflate::writer::Writer, ReturnCode};
use crate::{MAX_WBITS, MIN_WBITS};

/// Callback-fed input cursor for `inflateBack`.
///
/// The callback may replace the backing buffer whenever the current span is exhausted, so unlike
/// ordinary inflate this cursor cannot own one long-lived borrowed slice. Raw pointer lifetime is
/// confined here; decoding sees ordinary byte/bit operations and the shared [`BitBuffer`].
struct BackInput {
    next: *const u8,
    remaining: usize,
    bits: BitBuffer,
}

impl BackInput {
    /// Bind the initial zlib input span.
    ///
    /// # Safety
    /// A non-null `next` must be readable for `available` bytes for the duration of this call.
    unsafe fn new(next: *const u8, available: u32) -> Self {
        Self {
            next,
            remaining: if next.is_null() {
                0
            } else {
                available as usize
            },
            bits: BitBuffer::default(),
        }
    }

    #[inline(always)]
    fn available(&self) -> usize {
        self.remaining
    }

    #[inline(always)]
    fn next_ptr(&self) -> *const u8 {
        self.next
    }

    #[inline(always)]
    fn as_slice(&self) -> &[u8] {
        if self.remaining == 0 {
            &[]
        } else {
            // SAFETY: maintained by `new`, `ensure_available`, and `advance`.
            unsafe { core::slice::from_raw_parts(self.next, self.remaining) }
        }
    }

    #[inline(always)]
    fn bit_buffer(&self) -> BitBuffer {
        self.bits
    }

    #[inline(always)]
    fn set_bit_buffer(&mut self, bits: BitBuffer) {
        self.bits = bits;
    }

    #[inline(always)]
    fn bits_in_buffer(&self) -> u8 {
        self.bits.bits_in_buffer()
    }

    #[inline(always)]
    fn bits(&self, count: usize) -> u64 {
        self.bits.bits(count)
    }

    #[inline(always)]
    fn hold(&self) -> u64 {
        self.bits.hold()
    }

    #[inline(always)]
    fn drop_bits(&mut self, count: u8) {
        self.bits.drop_bits(count);
    }

    #[inline(always)]
    fn init_bits(&mut self) {
        self.bits.init_bits();
    }

    #[inline(always)]
    fn next_byte_boundary(&mut self) {
        self.bits.next_byte_boundary();
    }

    #[inline(always)]
    fn advance(&mut self, count: usize) {
        assert!(
            count <= self.remaining,
            "inflateBack input advance stays in-bounds"
        );
        self.remaining -= count;
        self.next = self.next.wrapping_add(count);
    }

    /// Ensure that at least one callback-owned byte is available.
    ///
    /// # Safety
    /// `in_` must obey the zlib `in_func` contract: a non-zero count must return a pointer readable
    /// for that many bytes until the next callback invocation.
    #[inline(always)]
    unsafe fn ensure_available(
        &mut self,
        in_: in_func,
        in_desc: *mut c_void,
    ) -> Result<(), ReturnCode> {
        if self.remaining != 0 {
            return Ok(());
        }

        let mut next = core::ptr::null();
        let available = unsafe { in_(in_desc, &mut next) } as usize;
        if available == 0 {
            self.next = core::ptr::null();
            return Err(ReturnCode::BufError);
        }

        debug_assert!(
            !next.is_null(),
            "inflateBack callback returned bytes without a pointer"
        );
        self.next = next;
        self.remaining = available;
        Ok(())
    }

    /// Pull one byte from the callback-fed stream into the shared DEFLATE reservoir.
    ///
    /// # Safety
    /// Same callback contract as [`Self::ensure_available`].
    #[inline(always)]
    unsafe fn pull_byte(&mut self, in_: in_func, in_desc: *mut c_void) -> Result<(), ReturnCode> {
        unsafe { self.ensure_available(in_, in_desc)? };
        // SAFETY: `ensure_available` established one readable byte.
        let byte = unsafe { *self.next };
        self.advance(1);
        self.bits.prime(8, byte as u64);
        Ok(())
    }

    /// Fill the shared reservoir to `count` bits, asking the callback for another span as needed.
    ///
    /// # Safety
    /// Same callback contract as [`Self::ensure_available`].
    #[inline(always)]
    unsafe fn need_bits(
        &mut self,
        count: usize,
        in_: in_func,
        in_desc: *mut c_void,
    ) -> Result<(), ReturnCode> {
        while usize::from(self.bits.bits_in_buffer()) < count {
            unsafe { self.pull_byte(in_, in_desc)? };
        }
        Ok(())
    }
}

/// Cursor over the caller-provided `inflateBack` output window.
///
/// The window storage itself remains owned by `State::window`; this cursor owns only the initialized
/// prefix length. Writes go through `Writer`, so the uninitialized tail is never materialized as a
/// typed `u8` slice.
#[derive(Debug, Default)]
struct BackOutput {
    filled: usize,
}

impl BackOutput {
    #[inline(always)]
    fn len(&self) -> usize {
        self.filled
    }

    #[inline(always)]
    fn set_len(&mut self, filled: usize, window: &Window<'_>) {
        assert!(
            filled <= window.buffer_size(),
            "inflateBack output stays in-bounds"
        );
        self.filled = filled;
    }

    #[inline(always)]
    fn remaining(&self, window: &Window<'_>) -> usize {
        window.buffer_size() - self.filled
    }

    #[inline(always)]
    fn history_available(&self, window: &Window<'_>) -> usize {
        if window.have() < window.buffer_size() {
            self.filled
        } else {
            window.buffer_size()
        }
    }

    #[inline(always)]
    fn push(&mut self, window: &mut Window<'_>, byte: u8) {
        let capacity = window.buffer_size();
        assert!(
            self.filled < capacity,
            "inflateBack output has room for literal"
        );
        // SAFETY: `filled <= capacity`, the caller-owned window remains valid for the whole
        // inflateBack call, and Writer tracks initialization of exactly this prefix.
        let mut writer =
            unsafe { Writer::new_uninit_raw(window.as_mut_ptr(), self.filled, capacity) };
        writer.push(byte);
        self.filled = writer.len();
    }

    #[inline(always)]
    fn extend(&mut self, window: &mut Window<'_>, bytes: &[u8]) {
        let capacity = window.buffer_size();
        assert!(
            bytes.len() <= capacity - self.filled,
            "inflateBack output copy fits"
        );
        // SAFETY: same window/prefix invariant as `push`; Writer writes only `bytes.len()` bytes.
        let mut writer =
            unsafe { Writer::new_uninit_raw(window.as_mut_ptr(), self.filled, capacity) };
        writer.extend(bytes);
        self.filled = writer.len();
    }

    /// Flush a full output window to the zlib callback and reset the cursor for reuse.
    ///
    /// # Safety
    /// `out` must obey the zlib `out_func` callback contract for the supplied window.
    #[inline(always)]
    unsafe fn flush_if_full(
        &mut self,
        window: &mut Window<'_>,
        out: out_func,
        out_desc: *mut c_void,
    ) -> Result<(), ReturnCode> {
        if self.filled != window.buffer_size() {
            return Ok(());
        }

        // The whole window is initialized before it is exposed to the callback or later history
        // reads.
        unsafe { window.set_have(self.filled) };
        if unsafe { out(out_desc, window.as_mut_ptr(), self.filled as u32) } != 0 {
            return Err(ReturnCode::BufError);
        }
        self.filled = 0;
        Ok(())
    }
}

/// Copy one DEFLATE back-reference into the current inflateBack output prefix.
///
/// History may start in the previous full callback window and continue into bytes already emitted in
/// the current window. Keeping this in one place makes the slow and fast decoders share those wrap
/// and overlap semantics.
#[inline(always)]
fn copy_back_match(
    window: &Window<'_>,
    sane: bool,
    writer: &mut Writer<'_>,
    distance: usize,
    length: usize,
) -> Result<(), &'static str> {
    let written = writer.len();
    if distance > written {
        let mut from_end = distance - written;
        if from_end > window.have() {
            if sane {
                return Err("invalid distance too far back\0");
            }
            panic!("INFLATE_ALLOW_INVALID_DISTANCE_TOOFAR_ARRR")
        }

        let window_size = window.buffer_size();
        let window_next = window.next();
        let from = if window_next == 0 {
            window_size - from_end
        } else if window_next >= from_end {
            window_next - from_end
        } else {
            from_end -= window_next;
            let from = window_size - from_end;
            if from_end < length {
                writer.extend_from_window_back(window, from..from + from_end);
                let remaining = length - from_end;
                let copy = Ord::min(window_next, remaining);
                writer.extend_from_window_back(window, 0..copy);
                if copy < remaining {
                    writer.copy_match_back(distance, remaining - copy);
                }
                return Ok(());
            }
            from
        };

        let copy = Ord::min(from_end, length);
        writer.extend_from_window_back(window, from..from + copy);
        if copy < length {
            writer.copy_match_back(distance, length - copy);
        }
    } else {
        writer.copy_match_back(distance, length);
    }
    Ok(())
}

/// Initialize the stream in an inflate state
pub fn back_init(stream: &mut z_stream, config: InflateConfig, window: Window) -> ReturnCode {
    assert_eq!(1 << config.window_bits, window.buffer_size());

    if !(MIN_WBITS..=MAX_WBITS).contains(&config.window_bits) {
        return ReturnCode::StreamError as _;
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

    let mut state = State::new();

    let alloc = Allocator {
        zalloc: stream.zalloc.unwrap(),
        zfree: stream.zfree.unwrap(),
        opaque: stream.opaque,
        _marker: PhantomData,
    };
    let allocs = InflateAllocOffsets::new();

    let Some(allocation_start) = alloc.allocate_slice_raw::<u8>(allocs.total_size) else {
        return ReturnCode::MemError;
    };

    let address = allocation_start.as_ptr() as usize;
    let align_offset = address.next_multiple_of(64) - address;
    let buf = unsafe { allocation_start.as_ptr().add(align_offset) };

    // NOTE: the window part of the allocation is ignored in this case.
    state.window = window;

    let state_allocation = unsafe { buf.add(allocs.state_pos).cast::<State>() };
    unsafe { state_allocation.write(state) };
    stream.state = state_allocation.cast::<internal_state>();

    // SAFETY: we've correctly initialized the stream to be an InflateStream
    let Some(stream) = (unsafe { InflateStream::from_stream_mut(stream) }) else {
        return ReturnCode::StreamError;
    };

    stream.state.allocation_start = allocation_start.as_ptr();
    stream.state.total_allocation_size = allocs.total_size;

    stream.state.wbits = config.window_bits as u8;
    stream.state.flags.update(Flags::SANE, true);

    ReturnCode::Ok
}

pub unsafe fn back(
    strm: &mut InflateStream,
    in_: in_func,
    in_desc: *mut c_void,
    out: out_func,
    out_desc: *mut c_void,
) -> ReturnCode {
    let mut ret;

    /* Reset the state */
    strm.msg = core::ptr::null_mut();
    strm.state.mode = Mode::Type;
    strm.state.flags.update(Flags::IS_LAST_BLOCK, false);
    strm.state.window.clear();

    // SAFETY: `inflateBack` requires the caller-provided initial span to be valid for `avail_in`.
    let mut input = unsafe { BackInput::new(strm.next_in.cast_const(), strm.avail_in) };
    let mut output = BackOutput::default();

    let state = &mut *strm.state;

    'inf_leave: loop {
        match state.mode {
            Mode::Type => {
                if state.flags.contains(Flags::IS_LAST_BLOCK) {
                    input.next_byte_boundary();
                    state.mode = Mode::Done;
                    continue;
                }

                if let Err(code) = unsafe { input.need_bits(3, in_, in_desc) } {
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
                if let Err(code) = unsafe { input.need_bits(32, in_, in_desc) } {
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

                    if let Err(code) = unsafe { input.ensure_available(in_, in_desc) } {
                        ret = code;
                        break 'inf_leave;
                    }
                    if let Err(code) =
                        unsafe { output.flush_if_full(&mut state.window, out, out_desc) }
                    {
                        ret = code;
                        break 'inf_leave;
                    }

                    copy = Ord::min(copy, input.available());
                    copy = Ord::min(copy, output.remaining(&state.window));
                    output.extend(&mut state.window, &input.as_slice()[..copy]);
                    input.advance(copy);
                    state.length -= copy;
                }

                state.mode = Mode::Type;
                continue;
            }
            Mode::Table => {
                if let Err(code) = unsafe { input.need_bits(14, in_, in_desc) } {
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
                    if let Err(code) = unsafe { input.need_bits(3, in_, in_desc) } {
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

                        if let Err(code) = unsafe { input.pull_byte(in_, in_desc) } {
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
                            if let Err(code) =
                                unsafe { input.need_bits(usize::from(here_bits) + 2, in_, in_desc) }
                            {
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
                            if let Err(code) =
                                unsafe { input.need_bits(usize::from(here_bits) + 3, in_, in_desc) }
                            {
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
                            if let Err(code) =
                                unsafe { input.need_bits(usize::from(here_bits) + 7, in_, in_desc) }
                            {
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
                    && output.remaining(&state.window) >= INFLATE_FAST_MIN_LEFT
                {
                    let input_len = input.available();
                    let mut bit_reader =
                        BitReader::from_buffer(input.bit_buffer(), input.as_slice());
                    let capacity = state.window.buffer_size();
                    // SAFETY: BackOutput guarantees its initialized prefix is in-bounds; the
                    // writer is call-local and dropped before the output cursor or callback is used.
                    let mut writer = unsafe {
                        Writer::new_uninit_raw(state.window.as_mut_ptr(), output.len(), capacity)
                    };

                    unsafe { inflate_fast_back(state, &mut bit_reader, &mut writer) };

                    let remaining = bit_reader.bytes_remaining();
                    let bits = bit_reader.into_buffer();
                    input.advance(input_len - remaining);
                    input.set_bit_buffer(bits);
                    output.set_len(writer.len(), &state.window);

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

                    if let Err(code) = unsafe { input.pull_byte(in_, in_desc) } {
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

                        if let Err(code) = unsafe { input.pull_byte(in_, in_desc) } {
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
                    if let Err(code) =
                        unsafe { output.flush_if_full(&mut state.window, out, out_desc) }
                    {
                        ret = code;
                        break 'inf_leave;
                    }
                    output.push(&mut state.window, state.length as u8);

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
                    if let Err(code) = unsafe { input.need_bits(state.extra, in_, in_desc) } {
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

                    if let Err(code) = unsafe { input.pull_byte(in_, in_desc) } {
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

                        if let Err(code) = unsafe { input.pull_byte(in_, in_desc) } {
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
                    if let Err(code) = unsafe { input.need_bits(extra, in_, in_desc) } {
                        ret = code;
                        break 'inf_leave;
                    }
                    state.offset += input.bits(extra) as usize;
                    input.drop_bits(extra as u8);
                }

                if INFLATE_STRICT && state.offset > output.history_available(&state.window) {
                    state.mode = Mode::Bad;
                    state.bad("invalid distance too far back\0");
                    continue 'inf_leave;
                }

                traceln!("inflate:         distance {}", state.offset);

                loop {
                    if let Err(code) =
                        unsafe { output.flush_if_full(&mut state.window, out, out_desc) }
                    {
                        ret = code;
                        break 'inf_leave;
                    }
                    let copy = Ord::min(state.length, output.remaining(&state.window));
                    let capacity = state.window.buffer_size();
                    // SAFETY: BackOutput owns the initialized-prefix length and `copy` is bounded
                    // by the remaining window capacity.
                    let mut writer = unsafe {
                        Writer::new_uninit_raw(state.window.as_mut_ptr(), output.len(), capacity)
                    };
                    if let Err(msg) = copy_back_match(
                        &state.window,
                        state.flags.contains(Flags::SANE),
                        &mut writer,
                        state.offset,
                        copy,
                    ) {
                        state.mode = Mode::Bad;
                        state.bad(msg);
                        continue 'inf_leave;
                    }
                    output.set_len(writer.len(), &state.window);
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

    if output.len() != 0
        && unsafe { out(out_desc, state.window.as_mut_ptr(), output.len() as u32) } != 0
        && ret == ReturnCode::StreamEnd
    {
        ret = ReturnCode::BufError;
    }

    strm.next_in = input.next_ptr().cast_mut();
    strm.avail_in = input.available() as u32;

    ret
}

#[inline(always)]
unsafe fn inflate_fast_back(
    state: &mut State,
    bit_reader: &mut BitReader<'_>,
    writer: &mut Writer<'_>,
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

                        if let Err(msg) = copy_back_match(
                            &state.window,
                            state.flags.contains(Flags::SANE),
                            writer,
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

pub fn back_end<'a>(strm: &'a mut InflateStream<'a>) {
    // With infback the window is user-supplied, so we mustn't try to free it.
    let _ = core::mem::replace(&mut strm.state.window, Window::empty());
    crate::inflate::end(strm);
}
