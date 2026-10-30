use core::{marker::PhantomData, mem::MaybeUninit};

use crate::weak_slice::WeakSliceMut;

pub struct Pending<'a> {
    /// start of the allocation
    buf: WeakSliceMut<'a, MaybeUninit<u8>>,
    /// next pending byte to output to the stream
    out: usize,
    /// number of bytes in the pending buffer
    pending: usize,
    /// semantically we're storing a mutable slice of bytes
    _marker: PhantomData<&'a mut [u8]>,
}

impl<'a> Pending<'a> {
    pub fn reset_keep(&mut self) {
        // Discard pending bytes and rewind the output cursor just like zlib's
        // `pending = 0; pending_out = pending_buf`.
        self.out = 0;
        self.pending = 0;
    }

    pub fn pending(&self) -> &[u8] {
        let slice = &self.buf.as_slice()[self.out..][..self.pending];
        // SAFETY: `out..out + pending` is exactly the initialized pending-output range.
        unsafe { slice.assume_init_ref() }
    }

    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        self.pending
    }

    #[inline(always)]
    pub(crate) fn is_empty(&self) -> bool {
        self.pending == 0
    }

    /// Number of bytes that can be added to the pending buffer until it is full
    pub(crate) fn remaining(&self) -> usize {
        self.buf.len() - (self.out + self.pending)
    }

    /// Total number of bytes that can be stored in the pending buffer
    pub(crate) fn capacity(&self) -> usize {
        self.buf.len()
    }

    #[inline(always)]
    #[track_caller]
    /// Mark a number of pending bytes as no longer pending
    pub fn advance(&mut self, number_of_bytes: usize) {
        assert!(
            number_of_bytes <= self.pending,
            "advancing past pending output"
        );

        self.out += number_of_bytes;
        self.pending -= number_of_bytes;

        if self.pending == 0 {
            self.out = 0;
        }
    }

    #[inline(always)]
    #[track_caller]
    pub fn truncate_tail(&mut self, n: usize) {
        assert!(n <= self.pending, "truncating past pending output");

        self.pending -= n;

        if self.pending == 0 {
            self.out = 0;
        }
    }

    #[inline(always)]
    #[track_caller]
    pub fn extend(&mut self, buf: &[u8]) {
        assert!(
            self.remaining() >= buf.len(),
            "buf.len() must fit in remaining()"
        );

        self.buf.as_mut_slice()[self.out + self.pending..][..buf.len()].write_copy_of_slice(buf);

        self.pending += buf.len();
    }

    pub(crate) unsafe fn from_raw_parts(ptr: *mut MaybeUninit<u8>, len: usize) -> Self {
        let buf = unsafe { WeakSliceMut::from_raw_parts_mut(ptr, len) };

        Self {
            buf,
            out: 0,
            pending: 0,
            _marker: PhantomData,
        }
    }

    /// Clone this pending-buffer state into independent raw storage without reading its
    /// uninitialized tail.
    ///
    /// # Safety
    ///
    /// `ptr` must be valid for writes of `self.buf.len()` bytes, must not overlap the source, and
    /// must remain alive and uniquely owned for the returned destination lifetime.
    pub(crate) unsafe fn clone_to<'dst>(&self, ptr: *mut u8) -> Pending<'dst> {
        let mut cloned =
            unsafe { Pending::from_raw_parts(ptr.cast::<MaybeUninit<u8>>(), self.buf.len()) };
        let initialized = self.out + self.pending;
        cloned.buf.as_mut_slice()[..initialized]
            .copy_from_slice(&self.buf.as_slice()[..initialized]);
        cloned.out = self.out;
        cloned.pending = self.pending;
        cloned
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clone_copies_initialized_prefix_and_preserves_cursor() {
        let mut source_storage = [MaybeUninit::<u8>::uninit(); 16];
        let mut destination_storage = [MaybeUninit::<u8>::uninit(); 16];

        let mut source =
            unsafe { Pending::from_raw_parts(source_storage.as_mut_ptr(), source_storage.len()) };
        source.extend(&[1, 2, 3, 4, 5, 6]);
        source.advance(2);

        assert_eq!(source.pending(), &[3, 4, 5, 6]);
        assert_eq!(source.remaining(), 10);

        let cloned = unsafe { source.clone_to(destination_storage.as_mut_ptr().cast::<u8>()) };

        assert_eq!(cloned.pending(), &[3, 4, 5, 6]);
        assert_eq!(cloned.len(), 4);
        assert_eq!(cloned.remaining(), source.remaining());
    }
}
