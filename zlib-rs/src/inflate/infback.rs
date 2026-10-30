use core::ffi::c_void;
use core::marker::PhantomData;

use crate::allocate::Allocator;
use crate::c_api::{in_func, internal_state, out_func};
use crate::inflate::bitreader::{BitBuffer, BitReader};
use crate::inflate::inftrees::{inflate_table, CodeType, InflateTable};
use crate::inflate::{
    Codes, Flags, InflateAllocOffsets, InflateConfig, InflateStateKind, InflateStream, Mode, State,
    Table, Window, INFLATE_FAST_MIN_HAVE, INFLATE_FAST_MIN_LEFT, INFLATE_STRICT, MAX_BITS,
    MAX_DIST_EXTRA_BITS,
};
use crate::traceln;
use crate::{c_api::z_stream, inflate::writer::BackWriter, ReturnCode};
use crate::{MAX_WBITS, MIN_WBITS};

/// Callback-fed input cursor for `inflateBack`.
///
/// The callback may replace the backing buffer whenever the current span is exhausted, so unlike
/// ordinary inflate this cursor cannot own one long-lived borrowed slice. Raw pointer lifetime and
/// callback unsafety are confined here; decoding sees ordinary byte/bit operations and the shared
/// [`BitBuffer`].
struct BackInput {
    next: *const u8,
    remaining: usize,
    bits: BitBuffer,
    callback: in_func,
    descriptor: *mut c_void,
}

impl BackInput {
    /// Bind the initial zlib input span and callback contract.
    ///
    /// # Safety
    /// A non-null `next` must be readable for `available` bytes for the duration of this call.
    /// `callback` must obey the zlib `in_func` contract: a non-zero count must provide a non-null
    /// pointer readable for that many bytes, and that span must remain unchanged until the next
    /// callback invocation or until `inflateBack` returns.
    unsafe fn new(
        next: *const u8,
        available: u32,
        callback: in_func,
        descriptor: *mut c_void,
    ) -> Self {
        Self {
            next,
            remaining: if next.is_null() {
                0
            } else {
                available as usize
            },
            bits: BitBuffer::default(),
            callback,
            descriptor,
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
            // SAFETY: maintained by `new`, `ensure_available`, and `advance`. A mutable borrow of
            // this cursor is required before the callback can replace the span, so this shared
            // slice cannot overlap a later callback invocation.
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
        if count != 0 {
            // SAFETY: the cursor invariant guarantees `next..next + remaining + count` belongs to
            // the current callback-owned span, and `count` has just been checked in-bounds.
            self.next = unsafe { self.next.add(count) };
        }
    }

    /// Ensure that at least one callback-owned byte is available.
    #[inline(always)]
    fn ensure_available(&mut self) -> Result<(), ReturnCode> {
        if self.remaining != 0 {
            return Ok(());
        }

        let mut next = core::ptr::null();
        // SAFETY: the callback contract is established once by `BackInput::new` and retained as a
        // cursor invariant. No shared slice from the previous span can exist through this `&mut`.
        let available = unsafe { (self.callback)(self.descriptor, &mut next) } as usize;
        if available == 0 || next.is_null() {
            self.next = core::ptr::null();
            self.remaining = 0;
            return Err(ReturnCode::BufError);
        }

        self.next = next;
        self.remaining = available;
        Ok(())
    }

    /// Pull one byte from the callback-fed stream into the shared DEFLATE reservoir.
    #[inline(always)]
    fn pull_byte(&mut self) -> Result<(), ReturnCode> {
        self.ensure_available()?;
        // SAFETY: `ensure_available` established one readable byte in the current callback span.
        let byte = unsafe { *self.next };
        self.advance(1);
        self.bits.prime(8, byte as u64);
        Ok(())
    }

    /// Fill the shared reservoir to `count` bits, asking the callback for another span as needed.
    #[inline(always)]
    fn need_bits(&mut self, count: usize) -> Result<(), ReturnCode> {
        while usize::from(self.bits.bits_in_buffer()) < count {
            self.pull_byte()?;
        }
        Ok(())
    }
}

/// Cursor over the caller-provided `inflateBack` output window.
///
/// The window storage itself remains owned by `State::window`; this cursor owns only the initialized
/// prefix length and the callback contract. Writes go through `Writer`, so the uninitialized tail is
/// never materialized as a typed `u8` slice.
struct BackOutput {
    filled: usize,
    callback: out_func,
    descriptor: *mut c_void,
}

impl BackOutput {
    /// Bind the zlib output callback contract to this cursor.
    ///
    /// # Safety
    /// `callback` must obey the zlib `out_func` contract for the caller-supplied window: it may read
    /// the initialized prefix passed to it, but must not mutate or invalidate that window while
    /// `inflateBack` is running.
    unsafe fn new(callback: out_func, descriptor: *mut c_void) -> Self {
        Self {
            filled: 0,
            callback,
            descriptor,
        }
    }

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
        let mut writer = BackWriter::new(window, self.filled);
        assert!(
            writer.remaining() != 0,
            "inflateBack output has room for literal"
        );
        writer.push(byte);
        self.filled = writer.len();
    }

    #[inline(always)]
    fn extend(&mut self, window: &mut Window<'_>, bytes: &[u8]) {
        let mut writer = BackWriter::new(window, self.filled);
        assert!(
            bytes.len() <= writer.remaining(),
            "inflateBack output copy fits"
        );
        writer.extend(bytes);
        self.filled = writer.len();
    }

    /// Flush a full output window to the zlib callback and reset the cursor for reuse.
    #[inline(always)]
    fn flush_if_full(&mut self, window: &mut Window<'_>) -> Result<(), ReturnCode> {
        if self.filled != window.buffer_size() {
            return Ok(());
        }

        // The whole window is initialized before it is exposed to the callback or later history
        // reads.
        window.set_have(self.filled);
        // SAFETY: `BackOutput::new` established the callback contract, and `filled == capacity`
        // proves the complete window is initialized for this callback.
        if unsafe { (self.callback)(self.descriptor, window.as_mut_ptr(), self.filled as u32) } != 0
        {
            return Err(ReturnCode::BufError);
        }
        self.filled = 0;
        Ok(())
    }

    /// Deliver any final partial window. This intentionally does not change `window.have()`: no
    /// later decode step consumes history after `inflateBack` starts returning.
    #[inline(always)]
    fn flush_remaining(&mut self, window: &mut Window<'_>) -> Result<(), ReturnCode> {
        if self.filled == 0 {
            return Ok(());
        }

        // SAFETY: `BackOutput::new` established the callback contract and `filled` tracks exactly
        // the initialized prefix of the caller-owned window.
        if unsafe { (self.callback)(self.descriptor, window.as_mut_ptr(), self.filled as u32) } != 0
        {
            return Err(ReturnCode::BufError);
        }
        Ok(())
    }
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

    let mut state = State::new(InflateStateKind::Back);
    // inflateBack uses the caller-provided window directly, so the allocation contains only State
    // plus alignment headroom.
    state.window = window;

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

    // SAFETY: `allocate_state` initialized a Back state behind the ABI state pointer.
    let Some(stream) = (unsafe { InflateStream::from_back_stream_mut(stream) }) else {
        return ReturnCode::StreamError;
    };

    stream.state.wbits = config.window_bits as u8;
    stream.state.flags.update(Flags::SANE, true);

    ReturnCode::Ok
}

/// Run callback-driven raw DEFLATE decompression.
///
/// # Safety
/// The initial `strm.next_in` span must be readable for `strm.avail_in` bytes when non-null.
/// `in_` must provide readable, unchanged spans until its next invocation or this function returns.
/// `out` must not mutate or invalidate the caller-provided output window while this function runs.
/// Both descriptors must remain valid for their callback's use for the duration of the call.
pub unsafe fn back(
    strm: &mut InflateStream,
    in_: in_func,
    in_desc: *mut c_void,
    out: out_func,
    out_desc: *mut c_void,
) -> ReturnCode {
    if strm.state.kind != InflateStateKind::Back {
        return ReturnCode::StreamError;
    }

    // SAFETY: `back` inherits zlib's input-span and callback contracts from its caller for this
    // invocation. Once bound, `decode_back` uses only safe cursor operations.
    let input = unsafe { BackInput::new(strm.next_in.cast_const(), strm.avail_in, in_, in_desc) };
    // SAFETY: same contract for the output callback and caller-owned window.
    let output = unsafe { BackOutput::new(out, out_desc) };

    decode_back(strm, input, output)
}

/// Safe `inflateBack` state-machine core. Raw callback and initial-span contracts are bound before
/// entry; decoding itself only manipulates checked cursors, state, and the exclusive `BackWriter`.
fn decode_back(
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
    let mut window = core::mem::replace(&mut state.window, Window::empty());
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
    state.window = window;

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
    fn back_writer_match_crosses_previous_window_into_current_output() {
        let mut storage = core::array::from_fn::<_, 32, _>(|i| i as u8);
        // The current callback window has already overwritten its first four bytes.
        storage[..4].copy_from_slice(&[100, 101, 102, 103]);
        let mut window = unsafe { Window::from_raw_parts(storage.as_mut_ptr(), storage.len()) };
        window.set_have(storage.len());

        let mut writer = BackWriter::new(&mut window, 4);
        writer.copy_match(true, 6, 8).unwrap();
        assert_eq!(writer.len(), 12);
        drop(writer);

        assert_eq!(&storage[4..12], &[30, 31, 100, 101, 102, 103, 30, 31]);
    }

    #[test]
    fn back_writer_rejects_distance_beyond_configured_window() {
        let mut storage = [0u8; 32];
        let mut window = unsafe { Window::from_raw_parts(storage.as_mut_ptr(), storage.len()) };
        window.set_have(storage.len());

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
        let mut window = unsafe { Window::from_raw_parts(storage.as_mut_ptr(), storage.len()) };

        let mut writer = BackWriter::new(&mut window, 4);
        assert_eq!(
            writer.copy_match(true, 5, 3),
            Err("invalid distance too far back\0")
        );
    }
}

pub fn back_end<'a>(strm: &'a mut InflateStream<'a>) {
    // With infback the window is user-supplied, so we mustn't try to free it.
    let _ = core::mem::replace(&mut strm.state.window, Window::empty());
    crate::inflate::end(strm);
}
