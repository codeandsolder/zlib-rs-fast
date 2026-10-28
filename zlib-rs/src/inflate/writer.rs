use core::fmt;
use core::mem::MaybeUninit;
use core::ops::Range;

use crate::weak_slice::WeakSliceMut;

pub struct Writer<'a> {
    buf: WeakSliceMut<'a, MaybeUninit<u8>>,
    filled: usize,
}

impl<'a> Writer<'a> {
    /// Creates a new `Writer` from a fully initialized buffer.
    #[inline]
    pub fn new(buf: &'a mut [u8]) -> Writer<'a> {
        // SAFETY: Because buf is a slice, most of the preconditions for
        // core::slice::from_raw_parts_mut are satisfied:
        // * buf.as_mut_ptr() is non-null.
        // * The memory range is within a single allocated object.
        // * buf is mutable, so the range is valid for both reads
        //   and writes of up to buf.len() * size_of::<u8>() bytes.
        // * The contents of the slice are initialized.
        // * buf.as_mut_ptr() + buf.len() does not wrap around
        //   the end of the address space.
        // The remaining precondition is enforced by the borrow checker when this function is called:
        // * The memory range cannot be accessed through any other pointer for the
        //   duration of lifetime 'a.
        unsafe { Self::new_uninit(buf.as_mut_ptr(), buf.len()) }
    }

    /// Creates a new `Writer` from an uninitialized buffer.
    ///
    /// # Safety
    ///
    /// The arguments must satisfy the requirements of [`core::slice::from_raw_parts_mut`].
    #[inline]
    pub unsafe fn new_uninit(ptr: *mut u8, len: usize) -> Writer<'a> {
        // SAFETY: The preconditions for WeakSliceMut::from_raw_parts_mut are the same
        // as for core::slice::from_raw_parts_mut, and the caller is responsible for
        // ensuring the latter.
        let buf = unsafe { WeakSliceMut::from_raw_parts_mut(ptr as *mut MaybeUninit<u8>, len) };
        Writer { buf, filled: 0 }
    }

    #[inline]
    pub unsafe fn new_uninit_raw(ptr: *mut u8, len: usize, capacity: usize) -> Writer<'a> {
        let buf =
            unsafe { WeakSliceMut::from_raw_parts_mut(ptr as *mut MaybeUninit<u8>, capacity) };
        Writer { buf, filled: len }
    }

    /// Pointer to where the next byte will be written
    #[inline]
    pub fn next_out(&mut self) -> *mut MaybeUninit<u8> {
        self.buf.as_mut_ptr().wrapping_add(self.filled).cast()
    }

    /// Returns the total capacity of the buffer.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    /// Returns the length of the filled part of the buffer
    #[inline]
    pub fn len(&self) -> usize {
        self.filled
    }

    /// Returns a shared reference to the filled portion of the buffer.
    #[inline]
    pub fn filled(&self) -> &[u8] {
        // SAFETY: the filled area of the buffer is always initialized, and self.filled is always
        // in-bounds.
        unsafe { core::slice::from_raw_parts(self.buf.as_ptr().cast(), self.filled) }
    }

    /// Returns the number of bytes at the end of the slice that have not yet been filled.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.capacity() - self.filled
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        self.filled == self.buf.len()
    }

    pub fn push(&mut self, byte: u8) {
        self.buf.as_mut_slice()[self.filled] = MaybeUninit::new(byte);

        self.filled += 1;
    }

    /// Appends data to the buffer
    #[inline(always)]
    pub fn extend(&mut self, buf: &[u8]) {
        // using simd here (on x86_64) was not fruitful
        self.buf.as_mut_slice()[self.filled..][..buf.len()].copy_from_slice(slice_to_uninit(buf));

        self.filled += buf.len();
    }

    #[inline(always)]
    pub fn extend_from_window(&mut self, window: &super::window::Window, range: Range<usize>) {
        let src = &window.as_slice()[range];
        self.extend(src);
    }

    /// Variant of `extend_from_window` used with `inflateBack`. It does not attempt a chunked
    /// copy, because there is no padding at the end and the window and output buffer alias.
    /// So a standard `memmove` will have to do.
    #[inline(always)]
    pub fn extend_from_window_back(&mut self, window: &super::window::Window, range: Range<usize>) {
        let len = range.end - range.start;

        unsafe {
            core::ptr::copy(
                window.as_ptr().add(range.start),
                self.buf.as_mut_ptr().add(self.filled).cast(),
                len,
            );
        }

        self.filled += len;
    }

    #[inline(always)]
    pub fn copy_match(&mut self, offset_from_end: usize, length: usize) {
        let current = self.filled;
        let start = current
            .checked_sub(offset_from_end)
            .expect("match distance in bounds");
        let end = current.checked_add(length).expect("match length in bounds");
        assert!(end <= self.capacity(), "match fits output buffer");

        let buf = self.buf.as_mut_slice();
        self.filled = end;

        // DEFLATE matches may overlap their output. `copy_within` has memmove semantics,
        // which is not enough for a match longer than its distance: newly written bytes
        // become the source for later bytes. Seed at most one distance, then repeatedly
        // double the initialized prefix. This expresses that recurrence directly without
        // architecture dispatch or copying past the logical end of the match.
        if length > offset_from_end {
            let seed_len = length.min(offset_from_end);
            buf.copy_within(start..start + seed_len, current);
            let mut initialized = seed_len;
            while initialized < length {
                let copy_len = initialized.min(length - initialized);
                buf.copy_within(current..current + copy_len, current + initialized);
                initialized += copy_len;
            }
        } else {
            buf.copy_within(start..start + length, current);
        }
    }

    /// Variant used by `inflateBack`. Match expansion itself needs no padding now,
    /// so it shares the exact-copy implementation with ordinary inflate.
    #[inline(always)]
    pub fn copy_match_back(&mut self, offset_from_end: usize, length: usize) {
        self.copy_match(offset_from_end, length);
    }
}

impl fmt::Debug for Writer<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Writer")
            .field("ptr", &self.buf.as_ptr())
            .field("filled", &self.filled)
            .field("capacity", &self.capacity())
            .finish()
    }
}

fn slice_to_uninit(slice: &[u8]) -> &[MaybeUninit<u8>] {
    unsafe { &*(slice as *const [u8] as *const [MaybeUninit<u8>]) }
}

#[cfg(test)]
mod test {
    use super::*;

    const N: usize = 512;
    const M: usize = 128;

    fn test_array() -> [MaybeUninit<u8>; N] {
        core::array::from_fn(|i| MaybeUninit::new(if i < M { i as u8 } else { 0xAAu8 }))
    }

    fn test_copy_match(offset_from_end: usize, length: usize) {
        let mut buf = test_array();
        let mut writer = Writer {
            buf: unsafe { WeakSliceMut::from_raw_parts_mut(buf.as_mut_ptr(), buf.len()) },
            filled: M,
        };
        writer.copy_match(offset_from_end, length);
        assert_eq!(writer.filled, M + length);

        let mut naive = test_array();
        for i in 0..length {
            naive[M + i] = naive[M - offset_from_end + i];
        }

        let buf = unsafe { core::mem::transmute::<[MaybeUninit<u8>; N], [u8; N]>(buf) };
        let naive = unsafe { core::mem::transmute::<[MaybeUninit<u8>; N], [u8; N]>(naive) };
        assert_eq!(
            buf[M..][..length],
            naive[M..][..length],
            "{offset_from_end} {length}"
        );
    }

    #[test]
    fn copy_match_boundaries_and_overlap() {
        for &(distance, length) in &[
            (1, 1),
            (1, 64),
            (2, 63),
            (3, 64),
            (7, 31),
            (16, 16),
            (17, 33),
            (64, 64),
        ] {
            test_copy_match(distance, length);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore = "slow")]
    fn copy_match() {
        for offset_from_end in 1..=M {
            for length in 0..=258 {
                test_copy_match(offset_from_end, length)
            }
        }
    }

    #[test]
    fn copy_match_at_exact_buffer_end() {
        let mut buf = [1, 2, 3, 0xAA, 0xAA].map(MaybeUninit::new);
        let mut writer = Writer {
            buf: unsafe { WeakSliceMut::from_raw_parts_mut(buf.as_mut_ptr(), buf.len()) },
            filled: 3,
        };

        writer.copy_match(3, 2);

        assert_eq!(buf.map(|e| unsafe { e.assume_init() }), [1, 2, 3, 1, 2]);
    }
}
