use core::fmt;
use core::mem::MaybeUninit;
use core::ops::Range;

use crate::weak_slice::WeakSliceMut;

use super::window::Window;

pub struct Writer<'a> {
    buf: WeakSliceMut<'a, MaybeUninit<u8>>,
    filled: usize,
}

impl<'a> Writer<'a> {
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
    unsafe fn new_uninit_raw(ptr: *mut u8, len: usize, capacity: usize) -> Writer<'a> {
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
        let filled = &self.buf.as_slice()[..self.filled];
        // SAFETY: the filled area of the buffer is always initialized, and self.filled is always
        // in-bounds.
        unsafe { filled.assume_init_ref() }
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
        self.buf.as_mut_slice()[self.filled..][..buf.len()].write_copy_of_slice(buf);

        self.filled += buf.len();
    }

    #[inline(always)]
    pub fn extend_from_window(&mut self, window: &super::window::Window, range: Range<usize>) {
        let src = &window.as_slice()[range];
        self.extend(src);
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

    /// Copy initialized bytes from another position in this same backing allocation.
    ///
    /// # Safety
    /// `source_start..source_start + length` must contain initialized bytes. The range may overlap
    /// the destination; this uses memmove semantics.
    #[inline(always)]
    unsafe fn extend_from_backing(&mut self, source_start: usize, length: usize) {
        let source_end = source_start
            .checked_add(length)
            .expect("backing copy source range overflow");
        assert!(
            source_end <= self.capacity(),
            "backing copy source in bounds"
        );
        assert!(
            length <= self.remaining(),
            "backing copy destination in bounds"
        );

        unsafe {
            core::ptr::copy(
                self.buf.as_ptr().add(source_start),
                self.buf.as_mut_ptr().add(self.filled),
                length,
            );
        }
        self.filled += length;
    }
}

/// Exclusive writer over the caller-owned `inflateBack` callback window.
///
/// The mutable borrow prevents the decoder from separately accessing `Window` while output bytes
/// are being written. `previous_history` snapshots how much of the previous callback window is
/// valid; current output always starts again at index zero after a flush, so Back mode does not need
/// the ordinary circular-window `next` cursor.
pub(crate) struct BackWriter<'a> {
    writer: Writer<'a>,
    previous_history: usize,
}

impl<'a> BackWriter<'a> {
    #[inline(always)]
    pub(crate) fn new(window: &'a mut Window<'_>, filled: usize) -> Self {
        let capacity = window.buffer_size();
        assert!(
            filled <= capacity,
            "inflateBack output prefix stays in-bounds"
        );
        let previous_history = window.have();
        let ptr = window.as_mut_ptr();

        // SAFETY: `Window` owns a writable backing allocation of `capacity` bytes for this borrow,
        // and `filled` is exactly the initialized current-output prefix tracked by `BackOutput`.
        let writer = unsafe { Writer::new_uninit_raw(ptr, filled, capacity) };
        Self {
            writer,
            previous_history,
        }
    }

    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        self.writer.len()
    }

    #[inline(always)]
    pub(crate) fn remaining(&self) -> usize {
        self.writer.remaining()
    }

    #[inline(always)]
    pub(crate) fn push(&mut self, byte: u8) {
        self.writer.push(byte);
    }

    #[inline(always)]
    pub(crate) fn extend(&mut self, bytes: &[u8]) {
        self.writer.extend(bytes);
    }

    /// Copy one DEFLATE back-reference into the current output prefix.
    ///
    /// If the distance reaches before the current prefix, history begins in the untouched tail of
    /// the previous full callback window. Once that tail is consumed, ordinary overlapping match
    /// expansion continues from the current prefix.
    #[inline(always)]
    pub(crate) fn copy_match(
        &mut self,
        sane: bool,
        distance: usize,
        length: usize,
    ) -> Result<(), &'static str> {
        if distance > self.writer.capacity() {
            if sane {
                return Err("invalid distance too far back\0");
            }
            panic!("INFLATE_ALLOW_INVALID_DISTANCE_TOOFAR_ARRR")
        }

        let written = self.writer.len();
        if distance > written {
            let from_previous = distance - written;
            if from_previous > self.previous_history {
                if sane {
                    return Err("invalid distance too far back\0");
                }
                panic!("INFLATE_ALLOW_INVALID_DISTANCE_TOOFAR_ARRR")
            }

            let copy = Ord::min(from_previous, length);
            let source_start = self.writer.capacity() - from_previous;
            // SAFETY: `previous_history` describes initialized bytes from the preceding full
            // callback window. Back output overwrites from index zero, so the still-needed history
            // is exactly the untouched tail ending at capacity. `copy <= from_previous` keeps the
            // source within that initialized tail; the Writer bounds-checks the destination.
            unsafe { self.writer.extend_from_backing(source_start, copy) };

            if copy < length {
                // `copy < length` implies `copy == from_previous`, so after consuming the previous
                // tail the remaining match begins at index zero of current output.
                self.writer.copy_match(distance, length - copy);
            }
        } else {
            self.writer.copy_match(distance, length);
        }
        Ok(())
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
