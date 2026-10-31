use core::ffi::c_void;

use crate::c_api::{in_func, out_func};
use crate::inflate::bitreader::BitBuffer;
use crate::inflate::window::BackWindow;
use crate::inflate::writer::BackWriter;
use crate::ReturnCode;

/// Callback-fed input cursor for `inflateBack`.
///
/// The callback may replace the backing buffer whenever the current span is exhausted, so unlike
/// ordinary inflate this cursor cannot own one long-lived borrowed slice. Raw pointer lifetime and
/// callback unsafety are confined here; decoding sees ordinary byte/bit operations and the shared
/// [`BitBuffer`].
pub(super) struct BackInput {
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
    pub(super) unsafe fn new(
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
    pub(super) fn available(&self) -> usize {
        self.remaining
    }

    #[inline(always)]
    pub(super) fn next_ptr(&self) -> *const u8 {
        self.next
    }

    #[inline(always)]
    pub(super) fn as_slice(&self) -> &[u8] {
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
    pub(super) fn bit_buffer(&self) -> BitBuffer {
        self.bits
    }

    #[inline(always)]
    pub(super) fn set_bit_buffer(&mut self, bits: BitBuffer) {
        self.bits = bits;
    }

    #[inline(always)]
    pub(super) fn bits_in_buffer(&self) -> u8 {
        self.bits.bits_in_buffer()
    }

    #[inline(always)]
    pub(super) fn bits(&self, count: usize) -> u64 {
        self.bits.bits(count)
    }

    #[inline(always)]
    pub(super) fn hold(&self) -> u64 {
        self.bits.hold()
    }

    #[inline(always)]
    pub(super) fn drop_bits(&mut self, count: u8) {
        self.bits.drop_bits(count);
    }

    #[inline(always)]
    pub(super) fn init_bits(&mut self) {
        self.bits.init_bits();
    }

    #[inline(always)]
    pub(super) fn next_byte_boundary(&mut self) {
        self.bits.next_byte_boundary();
    }

    #[inline(always)]
    pub(super) fn advance(&mut self, count: usize) {
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
    pub(super) fn ensure_available(&mut self) -> Result<(), ReturnCode> {
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
    pub(super) fn pull_byte(&mut self) -> Result<(), ReturnCode> {
        self.ensure_available()?;
        // SAFETY: `ensure_available` established one readable byte in the current callback span.
        let byte = unsafe { *self.next };
        self.advance(1);
        self.bits.prime(8, byte as u64);
        Ok(())
    }

    /// Fill the shared reservoir to `count` bits, asking the callback for another span as needed.
    #[inline(always)]
    pub(super) fn need_bits(&mut self, count: usize) -> Result<(), ReturnCode> {
        while usize::from(self.bits.bits_in_buffer()) < count {
            self.pull_byte()?;
        }
        Ok(())
    }
}

/// Cursor over the caller-provided `inflateBack` output window.
///
/// The window storage itself remains owned by `State::back_window`; this cursor owns only the initialized
/// prefix length and the callback contract. Writes go through `Writer`, so the uninitialized tail is
/// never materialized as a typed `u8` slice.
pub(super) struct BackOutput {
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
    pub(super) unsafe fn new(callback: out_func, descriptor: *mut c_void) -> Self {
        Self {
            filled: 0,
            callback,
            descriptor,
        }
    }

    #[inline(always)]
    pub(super) fn len(&self) -> usize {
        self.filled
    }

    #[inline(always)]
    pub(super) fn set_len(&mut self, filled: usize, window: &BackWindow<'_>) {
        assert!(
            filled <= window.buffer_size(),
            "inflateBack output stays in-bounds"
        );
        self.filled = filled;
    }

    #[inline(always)]
    pub(super) fn remaining(&self, window: &BackWindow<'_>) -> usize {
        window.buffer_size() - self.filled
    }

    #[inline(always)]
    pub(super) fn history_available(&self, window: &BackWindow<'_>) -> usize {
        if window.history_len() == 0 {
            self.filled
        } else {
            window.buffer_size()
        }
    }

    #[inline(always)]
    pub(super) fn push(&mut self, window: &mut BackWindow<'_>, byte: u8) {
        let mut writer = BackWriter::new(window, self.filled);
        assert!(
            writer.remaining() != 0,
            "inflateBack output has room for literal"
        );
        writer.push(byte);
        self.filled = writer.len();
    }

    #[inline(always)]
    pub(super) fn extend(&mut self, window: &mut BackWindow<'_>, bytes: &[u8]) {
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
    pub(super) fn flush_if_full(&mut self, window: &mut BackWindow<'_>) -> Result<(), ReturnCode> {
        if self.filled != window.buffer_size() {
            return Ok(());
        }

        // The whole window is initialized before it is exposed to the callback or later history
        // reads.
        window.mark_full_history();
        // SAFETY: `BackOutput::new` established the callback contract, and `filled == capacity`
        // proves the complete window is initialized for this callback.
        if unsafe {
            (self.callback)(
                self.descriptor,
                window.as_mut_ptr().cast::<u8>(),
                self.filled as u32,
            )
        } != 0
        {
            return Err(ReturnCode::BufError);
        }
        self.filled = 0;
        Ok(())
    }

    /// Deliver any final partial window. This intentionally does not mark a full-history window:
    /// no later decode step consumes history after `inflateBack` starts returning.
    #[inline(always)]
    pub(super) fn flush_remaining(
        &mut self,
        window: &mut BackWindow<'_>,
    ) -> Result<(), ReturnCode> {
        if self.filled == 0 {
            return Ok(());
        }

        // SAFETY: `BackOutput::new` established the callback contract and `filled` tracks exactly
        // the initialized prefix of the caller-owned window.
        if unsafe {
            (self.callback)(
                self.descriptor,
                window.as_mut_ptr().cast::<u8>(),
                self.filled as u32,
            )
        } != 0
        {
            return Err(ReturnCode::BufError);
        }
        Ok(())
    }
}
