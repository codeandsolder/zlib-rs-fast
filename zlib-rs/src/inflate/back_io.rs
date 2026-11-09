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
        let available = unsafe { (self.callback)(self.descriptor, &raw mut next) } as usize;
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

    /// Give the decoder one exclusive, initialized-prefix-aware writer view,
    /// then commit its new length before the window can be borrowed again.
    #[inline]
    pub(super) fn with_writer<R>(
        &mut self,
        window: &mut BackWindow<'_>,
        operation: impl FnOnce(&mut BackWriter<'_>) -> R,
    ) -> R {
        let mut writer = BackWriter::new(window, self.filled);
        let result = operation(&mut writer);
        self.filled = writer.len();
        result
    }

    /// Copy at most the current window's remaining capacity. The calling
    /// decoder handles continuation after a full-window callback.
    pub(super) fn copy_match(
        &mut self,
        window: &mut BackWindow<'_>,
        sane: bool,
        distance: usize,
        length: usize,
    ) -> Result<usize, &'static str> {
        self.with_writer(window, |writer| {
            let count = length.min(writer.remaining());
            writer.copy_match(sane, distance, count)?;
            Ok(count)
        })
    }

    #[inline(always)]
    pub(super) fn push(&mut self, window: &mut BackWindow<'_>, byte: u8) {
        self.with_writer(window, |writer| {
            assert_ne!(
                writer.remaining(),
                0,
                "inflateBack output has room for literal"
            );
            writer.push(byte);
        });
    }

    #[inline(always)]
    pub(super) fn extend(&mut self, window: &mut BackWindow<'_>, bytes: &[u8]) {
        self.with_writer(window, |writer| {
            assert!(
                bytes.len() <= writer.remaining(),
                "inflateBack output copy fits"
            );
            writer.extend(bytes);
        });
    }

    /// Expose only the initialized output prefix to the caller callback.
    /// Both full-window and final-partial-window deliveries use this boundary.
    #[inline]
    fn deliver(&mut self, window: &mut BackWindow<'_>) -> Result<(), ReturnCode> {
        debug_assert!(self.filled > 0 && self.filled <= window.buffer_size());
        // The window size was validated at inflateBackInit to be at most 32768.
        let count = self.filled as u32;
        // SAFETY: BackOutput::new established the callback contract. Every
        // byte in 0..filled was initialized by BackWriter, and the callback
        // may read that prefix but may not invalidate the window storage.
        let result = unsafe {
            (self.callback)(
                self.descriptor,
                window.as_uninit_slice().as_mut_ptr().cast::<u8>(),
                count,
            )
        };
        if result != 0 {
            return Err(ReturnCode::BufError);
        }
        Ok(())
    }

    /// Flush a full output window and reset the prefix for the next chunk.
    #[inline(always)]
    pub(super) fn flush_if_full(&mut self, window: &mut BackWindow<'_>) -> Result<(), ReturnCode> {
        if self.filled != window.buffer_size() {
            return Ok(());
        }

        // Only a completely initialized window becomes previous history.
        window.mark_full_history();
        self.deliver(window)?;
        self.filled = 0;
        Ok(())
    }

    /// Deliver a final partial prefix, without marking incomplete history.
    #[inline(always)]
    pub(super) fn flush_remaining(
        &mut self,
        window: &mut BackWindow<'_>,
    ) -> Result<(), ReturnCode> {
        if self.filled == 0 {
            return Ok(());
        }
        self.deliver(window)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CallbackData {
        chunks: Vec<Vec<u8>>,
        reject: bool,
    }

    unsafe extern "C" fn record_output(descriptor: *mut c_void, bytes: *mut u8, len: u32) -> i32 {
        // SAFETY: test passes the exclusive address of CallbackData as descriptor.
        let data = unsafe { &mut *descriptor.cast::<CallbackData>() };
        if data.reject {
            return 1;
        }

        // SAFETY: BackOutput delivers only the initialized prefix of its borrowed
        // scratch window. The callback copies it before that window is reused.
        let prefix = unsafe { core::slice::from_raw_parts(bytes, len as usize) };
        data.chunks.push(prefix.to_vec());
        0
    }

    #[test]
    fn callback_output_commits_history_and_delivers_only_initialized_prefixes() {
        let mut storage = [0xa5; 8];
        let mut window = BackWindow::from_initialized_slice(&mut storage);
        let mut data = CallbackData {
            chunks: Vec::new(),
            reject: false,
        };
        // SAFETY: the descriptor refers to live, uniquely borrowed test state,
        // and record_output only reads initialized output bytes before returning.
        let mut output =
            unsafe { BackOutput::new(record_output, (&raw mut data).cast::<c_void>()) };

        output.extend(&mut window, &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(output.len(), 8);
        assert_eq!(output.flush_if_full(&mut window), Ok(()));
        assert_eq!(output.len(), 0);
        assert_eq!(output.history_available(&window), 8);
        assert_eq!(data.chunks, vec![vec![1, 2, 3, 4, 5, 6, 7, 8]]);

        assert_eq!(output.copy_match(&mut window, true, 8, 5), Ok(5));
        assert_eq!(output.len(), 5);
        assert_eq!(output.flush_if_full(&mut window), Ok(()));
        assert_eq!(data.chunks.len(), 1);
        assert_eq!(output.flush_remaining(&mut window), Ok(()));
        assert_eq!(data.chunks[1], [1, 2, 3, 4, 5]);
        data.reject = true;
        assert_eq!(
            output.flush_remaining(&mut window),
            Err(ReturnCode::BufError)
        );
        assert_eq!(output.len(), 5);
        assert_eq!(data.chunks.len(), 2);
        // No part of the unwritten tail is overwritten by the partial match.
        assert_eq!(&storage[5..], &[6, 7, 8]);
    }
}
