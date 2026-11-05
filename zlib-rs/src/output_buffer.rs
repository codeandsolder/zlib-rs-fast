use core::mem::MaybeUninit;

/// Writable byte storage used by Rust-facing compression/decompression helpers.
///
/// Both initialized and potentially-uninitialized byte slices are valid destinations. Keeping
/// their types distinct until the caller-visible boundary avoids manufacturing a
/// `MaybeUninit` view over memory that Rust already knows is initialized.
pub trait WritableBytes {
    fn byte_ptr(&mut self) -> *mut u8;
    fn len(&self) -> usize;
}

impl WritableBytes for [u8] {
    #[inline]
    fn byte_ptr(&mut self) -> *mut u8 {
        self.as_mut_ptr()
    }

    #[inline]
    fn len(&self) -> usize {
        <[u8]>::len(self)
    }
}

impl WritableBytes for [MaybeUninit<u8>] {
    #[inline]
    fn byte_ptr(&mut self) -> *mut u8 {
        self.as_mut_ptr().cast()
    }

    #[inline]
    fn len(&self) -> usize {
        <[MaybeUninit<u8>]>::len(self)
    }
}

/// Number of bytes written by a chunked zlib stream into a caller-owned output slice.
///
/// `not_issued` is output capacity not yet handed to zlib, and `available` is the
/// unused portion of the last issued chunk. Unlike `z_stream.total_out`, these
/// counters preserve full `usize` precision when C `unsigned long` is 32-bit.
///
/// # Panics
///
/// Panics if the unissued or available output spans exceed the caller's storage.
pub const fn written_from_budget(total: usize, not_issued: usize, available: u32) -> usize {
    assert!(
        not_issued <= total,
        "unissued output exceeds caller storage"
    );
    let issued = total - not_issued;
    assert!(
        available as usize <= issued,
        "zlib output cursor exceeds issued storage"
    );
    issued - available as usize
}

/// Expose exactly the prefix that a writer has proven initialized.
///
/// # Safety
///
/// Every element of `output[..initialized]` must have been initialized as a valid `u8`.
#[inline]
pub unsafe fn assume_init_prefix(output: &mut [MaybeUninit<u8>], initialized: usize) -> &mut [u8] {
    assert!(
        initialized <= output.len(),
        "initialized prefix exceeds output storage"
    );
    // SAFETY: guaranteed by the caller; the untouched tail remains MaybeUninit and is not exposed.
    unsafe { output[..initialized].assume_init_mut() }
}

#[cfg(test)]
mod budget_tests {
    use super::written_from_budget;

    #[test]
    fn output_budget_counts_partial_and_multi_chunk_writes() {
        assert_eq!(written_from_budget(100, 0, 0), 100);
        assert_eq!(written_from_budget(100, 30, 20), 50);
        assert_eq!(written_from_budget(100, 0, 100), 0);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn output_budget_does_not_wrap_at_32_bit_counter_boundary() {
        let total = u32::MAX as usize + 1_024;
        assert_eq!(written_from_budget(total, 0, 0), total);
        assert_eq!(written_from_budget(total, 512, 256), total - 768);
    }
}
