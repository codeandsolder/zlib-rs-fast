use core::mem::MaybeUninit;

/// Writable byte storage used by Rust-facing compression/decompression helpers.
///
/// Both initialized and potentially-uninitialized byte slices are valid destinations. Keeping
/// their types distinct until the caller-visible boundary avoids manufacturing a
/// `MaybeUninit` view over memory that Rust already knows is initialized.
pub(crate) trait WritableBytes {
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

/// Expose exactly the prefix that a writer has proven initialized.
///
/// # Safety
///
/// Every element of `output[..initialized]` must have been initialized as a valid `u8`.
#[inline]
pub(crate) unsafe fn assume_init_prefix(
    output: &mut [MaybeUninit<u8>],
    initialized: usize,
) -> &mut [u8] {
    assert!(
        initialized <= output.len(),
        "initialized prefix exceeds output storage"
    );
    // SAFETY: guaranteed by the caller; the untouched tail remains MaybeUninit and is not exposed.
    unsafe { output[..initialized].assume_init_mut() }
}
