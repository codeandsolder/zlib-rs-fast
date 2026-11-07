use core::fmt;
use core::mem::MaybeUninit;
use core::ops::Range;

use crate::weak_slice::WeakSliceMut;

use super::window::BackWindow;

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

    /// Reborrow a window with a prefix already produced by the decoder.
    ///
    /// # Safety
    /// The first filled bytes of buf must have been initialized.
    #[inline]
    unsafe fn from_initialized_prefix(buf: &'a mut [MaybeUninit<u8>], filled: usize) -> Self {
        assert!(
            filled <= buf.len(),
            "initialized output prefix stays in bounds"
        );
        Self {
            buf: WeakSliceMut::from_mut_slice(buf),
            filled,
        }
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

    /// Materialize missing history as initialized zero bytes, as requested by
    /// inflateUndermine. Never expose the uninitialized contents of a window.
    #[inline]
    pub fn fill_zero(&mut self, len: usize) {
        self.buf.as_mut_slice()[self.filled..][..len].fill(MaybeUninit::new(0));
        self.filled += len;
    }

    /// Appends data to the buffer
    #[inline]
    pub fn extend(&mut self, buf: &[u8]) {
        // using simd here (on x86_64) was not fruitful
        self.buf.as_mut_slice()[self.filled..][..buf.len()].write_copy_of_slice(buf);

        self.filled += buf.len();
    }

    #[inline]
    pub fn extend_from_window(&mut self, window: &super::window::Window, range: Range<usize>) {
        let src = &window.as_slice()[range];
        self.extend(src);
    }

    #[inline]
    pub fn copy_match(&mut self, offset_from_end: usize, length: usize) {
        assert_ne!(
            offset_from_end, 0,
            "DEFLATE match distance must be positive"
        );
        let current = self.filled;
        assert!(offset_from_end <= current, "match distance in bounds");
        assert!(length <= self.remaining(), "match fits output buffer");
        let start = current - offset_from_end;
        let end = current + length;

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
    #[inline]
    unsafe fn extend_from_backing(&mut self, source_start: usize, length: usize) {
        assert!(
            source_start <= self.capacity() && length <= self.capacity() - source_start,
            "backing copy source in bounds"
        );
        let source_end = source_start + length;
        assert!(
            length <= self.remaining(),
            "backing copy destination in bounds"
        );

        // Copying MaybeUninit elements is safe, but the initialization
        // contract above still matters: the source becomes filled output.
        self.buf
            .as_mut_slice()
            .copy_within(source_start..source_end, self.filled);
        self.filled += length;
    }
}

/// Exclusive writer over the caller-owned `inflateBack` callback window.
///
/// The mutable borrow prevents the decoder from separately accessing `Window` while output bytes
/// are being written. `previous_history` snapshots how much of the previous callback window is
/// valid; current output always starts again at index zero after a flush, so Back mode does not need
/// the ordinary circular-window `next` cursor.
pub struct BackWriter<'a> {
    writer: Writer<'a>,
    previous_history: usize,
}

impl<'a> BackWriter<'a> {
    #[inline]
    pub(crate) fn new(window: &'a mut BackWindow<'_>, filled: usize) -> Self {
        let previous_history = window.history_len();

        // SAFETY: BackWindow uniquely owns the storage for this borrow and
        // BackOutput tracks the initialized prefix of the caller-owned window.
        let writer = unsafe { Writer::from_initialized_prefix(window.as_uninit_slice(), filled) };
        Self {
            writer,
            previous_history,
        }
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.writer.len()
    }

    #[inline]
    pub(crate) fn remaining(&self) -> usize {
        self.writer.remaining()
    }

    #[inline]
    pub(crate) fn push(&mut self, byte: u8) {
        self.writer.push(byte);
    }

    #[inline]
    pub(crate) fn extend(&mut self, bytes: &[u8]) {
        self.writer.extend(bytes);
    }

    /// Copy one DEFLATE back-reference into the current output prefix.
    ///
    /// If the distance reaches before the current prefix, history begins in the untouched tail of
    /// the previous full callback window. Once that tail is consumed, ordinary overlapping match
    /// expansion continues from the current prefix.
    #[inline]
    pub(crate) fn copy_match(
        &mut self,
        sane: bool,
        distance: usize,
        length: usize,
    ) -> Result<(), &'static str> {
        if length == 0 {
            return Ok(());
        }
        if distance == 0 {
            return Err("invalid distance too far back\0");
        }
        if distance > self.writer.capacity() {
            if sane {
                return Err("invalid distance too far back\0");
            }
            // A distance larger than the history window cannot name an
            // available byte; never read overwritten callback storage.
            self.writer.fill_zero(length);
            return Ok(());
        }

        let written = self.writer.len();
        if distance > written {
            let from_previous = distance - written;
            if from_previous > self.previous_history {
                if sane {
                    return Err("invalid distance too far back\0");
                }
                // The callback's previous window has no initialized bytes
                // this far back. Synthesize only the missing prefix, then
                // resume the same match using the now-initialized output.
                // A distance bounded by capacity can lack history only
                // before the first complete callback window.
                debug_assert_eq!(self.previous_history, 0);
                let zeros = from_previous.min(length);
                self.writer.fill_zero(zeros);
                if zeros < length {
                    self.writer.copy_match(distance, length - zeros);
                }
                return Ok(());
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
    fn callback_writer_non_strict_missing_history_zero_fills_only_unknown_prefix() {
        let mut storage = [0xa5u8; 16];
        let mut window = super::super::window::BackWindow::from_initialized_slice(&mut storage);
        let mut writer = BackWriter::new(&mut window, 0);
        assert_eq!(
            writer.copy_match(true, 5, 11),
            Err("invalid distance too far back\0")
        );
        assert_eq!(writer.len(), 0);
        assert_eq!(writer.copy_match(false, 5, 11), Ok(()));
        assert_eq!(writer.len(), 11);
        assert_eq!(&storage[..11], &[0; 11]);
        assert_eq!(&storage[11..], &[0xa5; 5]);
    }

    #[test]
    fn callback_writer_ignores_unreachable_history_beyond_window() {
        let mut storage = [0xa5u8; 16];
        let mut window = super::super::window::BackWindow::from_initialized_slice(&mut storage);
        window.mark_full_history();
        let mut writer = BackWriter::new(&mut window, 0);
        assert_eq!(
            writer.copy_match(true, 20, 8),
            Err("invalid distance too far back\0")
        );
        assert_eq!(writer.copy_match(false, 20, 8), Ok(()));
        assert_eq!(writer.len(), 8);
        assert_eq!(&storage[..8], &[0; 8]);
        assert_eq!(&storage[8..], &[0xa5; 8]);
    }

    #[test]
    fn callback_writer_rejects_zero_distance_in_both_modes() {
        let mut storage = [0xa5u8; 16];
        let mut window = super::super::window::BackWindow::from_initialized_slice(&mut storage);
        let mut writer = BackWriter::new(&mut window, 0);
        assert_eq!(
            writer.copy_match(true, 0, 3),
            Err("invalid distance too far back\0")
        );
        assert_eq!(
            writer.copy_match(false, 0, 3),
            Err("invalid distance too far back\0")
        );
        assert_eq!(writer.len(), 0);
        assert_eq!(storage, [0xa5; 16]);
    }

    #[test]
    fn callback_writer_uses_only_initialized_previous_history() {
        let mut storage = [0, 1, 2, 3, 4, 5, 6, 7];
        let mut window = super::super::window::BackWindow::from_initialized_slice(&mut storage);
        window.mark_full_history();
        let mut writer = BackWriter::new(&mut window, 0);
        assert_eq!(writer.copy_match(false, 6, 8), Ok(()));
        assert_eq!(writer.len(), 8);
        assert_eq!(storage, [2, 3, 4, 5, 6, 7, 2, 3]);
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
