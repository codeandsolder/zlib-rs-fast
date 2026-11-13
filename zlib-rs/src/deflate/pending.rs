use core::mem::MaybeUninit;

use crate::weak_slice::WeakSliceMut;

pub struct Pending<'a> {
    /// start of the allocation
    buf: WeakSliceMut<'a, MaybeUninit<u8>>,
    /// next pending byte to output to the stream
    out: usize,
    /// Number of unread bytes beginning at `out`; `out + count` bounds initialized storage.
    count: usize,
}

impl Pending<'_> {
    pub const fn reset_keep(&mut self) {
        // Discard pending bytes and rewind the output cursor just like zlib's
        // `pending = 0; pending_out = pending_buf`.
        self.out = 0;
        self.count = 0;
    }

    pub fn pending(&self) -> &[u8] {
        let slice = &self.buf.as_slice()[self.out..][..self.count];
        // SAFETY: `out..out + count` is exactly the initialized pending-output range.
        unsafe { slice.assume_init_ref() }
    }

    #[inline]
    pub(crate) const fn len(&self) -> usize {
        self.count
    }

    #[inline]
    pub(crate) const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Number of bytes that can be added to the pending buffer until it is full
    pub(crate) const fn remaining(&self) -> usize {
        self.buf.len() - (self.out + self.count)
    }

    /// Total number of bytes that can be stored in the pending buffer
    pub(crate) const fn capacity(&self) -> usize {
        self.buf.len()
    }

    #[inline]
    #[track_caller]
    /// Mark a number of pending bytes as no longer pending
    pub fn advance(&mut self, number_of_bytes: usize) {
        assert!(
            number_of_bytes <= self.count,
            "advancing past pending output"
        );

        self.out += number_of_bytes;
        self.count -= number_of_bytes;

        if self.count == 0 {
            self.out = 0;
        }
    }

    // Used only by the partial-clone invariant regression.
    #[cfg(test)]
    #[inline]
    #[track_caller]
    pub fn truncate_tail(&mut self, n: usize) {
        assert!(n <= self.count, "truncating past pending output");
        self.count -= n;
        if self.count == 0 {
            self.out = 0;
        }
    }

    #[inline]
    #[track_caller]
    pub fn extend(&mut self, buf: &[u8]) {
        assert!(
            self.remaining() >= buf.len(),
            "buf.len() must fit in remaining()"
        );

        self.buf.as_mut_slice()[self.out + self.count..][..buf.len()].write_copy_of_slice(buf);

        self.count += buf.len();
    }

    /// Bind independently owned byte storage without reading its uninitialized contents.
    ///
    /// # Safety
    ///
    /// `ptr` must be non-null, aligned, and writable for `len` bytes throughout
    /// the returned lifetime. No conflicting accesses may occur while this handle exists.
    pub(crate) const unsafe fn from_raw_parts<'a>(
        ptr: *mut MaybeUninit<u8>,
        len: usize,
    ) -> Pending<'a> {
        // SAFETY: the caller guarantees a uniquely writable and suitably aligned allocation
        // for the returned lifetime; MaybeUninit places no initialization requirement on bytes.
        let buf = unsafe { WeakSliceMut::from_raw_parts_mut(ptr, len) };

        Pending {
            buf,
            out: 0,
            count: 0,
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
        // SAFETY: clone_to provides independent writable destination storage for the
        // complete allocation; its bytes may start uninitialized.
        let mut cloned = unsafe {
            Pending::<'dst>::from_raw_parts(ptr.cast::<MaybeUninit<u8>>(), self.buf.len())
        };
        let initialized = self.out + self.count;
        cloned.buf.as_mut_slice()[..initialized]
            .copy_from_slice(&self.buf.as_slice()[..initialized]);
        cloned.out = self.out;
        cloned.count = self.count;
        cloned
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reused_storage_and_partial_clone_preserve_unread_bytes() {
        let mut source_storage = [MaybeUninit::<u8>::uninit(); 12];
        let mut clone_storage = [MaybeUninit::<u8>::uninit(); 12];
        // SAFETY: the arrays are independent and uniquely writable throughout the test.
        let mut pending =
            unsafe { Pending::from_raw_parts(source_storage.as_mut_ptr(), source_storage.len()) };

        pending.extend(&[1, 2, 3, 4, 5, 6, 7]);
        pending.advance(3);
        pending.truncate_tail(2);
        assert_eq!(pending.pending(), &[4, 5]);
        assert_eq!(pending.remaining(), 7);

        // SAFETY: clone_storage does not overlap with source_storage and has the same length.
        let mut copy = unsafe { pending.clone_to(clone_storage.as_mut_ptr().cast()) };
        assert_eq!(copy.pending(), &[4, 5]);
        copy.extend(&[8, 9]);
        assert_eq!(copy.pending(), &[4, 5, 8, 9]);

        pending.advance(2);
        assert!(pending.is_empty());
        assert_eq!(pending.remaining(), pending.capacity());
        pending.extend(&[10, 11, 12]);
        assert_eq!(pending.pending(), &[10, 11, 12]);

        copy.reset_keep();
        copy.extend(&[13, 14]);
        assert_eq!(copy.pending(), &[13, 14]);
    }

    #[test]
    fn clone_copies_initialized_prefix_and_preserves_cursor() {
        let mut source_storage = [MaybeUninit::<u8>::uninit(); 16];
        let mut destination_storage = [MaybeUninit::<u8>::uninit(); 16];

        // SAFETY: source_storage is uniquely writable and remains live for the test.
        let mut source =
            unsafe { Pending::from_raw_parts(source_storage.as_mut_ptr(), source_storage.len()) };
        source.extend(&[1, 2, 3, 4, 5, 6]);
        source.advance(2);

        assert_eq!(source.pending(), &[3, 4, 5, 6]);
        assert_eq!(source.remaining(), 10);

        // SAFETY: destination_storage is a separate, writable allocation of the same length.
        let cloned = unsafe { source.clone_to(destination_storage.as_mut_ptr().cast::<u8>()) };

        assert_eq!(cloned.pending(), &[3, 4, 5, 6]);
        assert_eq!(cloned.len(), 4);
        assert_eq!(cloned.remaining(), source.remaining());
    }
}
