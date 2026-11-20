use core::fmt;
use core::mem::MaybeUninit;
use core::ops::Range;

use crate::weak_slice::WeakSliceMut;

use super::window::BackWindow;

// A short overlapping match is cheaper to advance byte-by-byte than to
// seed and repeatedly double an initialized prefix. Longer matches use
// disjoint block copies instead of doing one load/store per output byte.
const SCALAR_OVERLAP_COPY_LIMIT: usize = 32;
// Endpoint values cover periods smaller than twice the largest 32-byte
// fixed value. For non-power-of-two periods, cap the number of stores:
// the measured long distance-2/3/4 recurrences favor prefix doubling.
const MAX_REPLICATED_PERIOD: usize = 64;
const MAX_PERIOD_REPETITIONS: usize = 64;
// A fixed-size byte-array copy lets LLVM choose efficient native vector
// loads and stores without handwritten SIMD intrinsics.
#[cfg(target_arch = "x86_64")]
const MATCH_COPY_WIDTH: usize = 32;
#[cfg(target_arch = "aarch64")]
const MATCH_COPY_WIDTH: usize = 16;
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
const MATCH_COPY_WIDTH: usize = 8;

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
    pub const unsafe fn new_uninit(ptr: *mut u8, len: usize) -> Self {
        // SAFETY: The preconditions for WeakSliceMut::from_raw_parts_mut are the same
        // as for core::slice::from_raw_parts_mut, and the caller is responsible for
        // ensuring the latter.
        let buf = unsafe { WeakSliceMut::from_raw_parts_mut(ptr.cast(), len) };
        Self { buf, filled: 0 }
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
    pub const fn next_out(&mut self) -> *mut MaybeUninit<u8> {
        self.buf.as_mut_ptr().wrapping_add(self.filled).cast()
    }

    /// Returns the total capacity of the buffer.
    #[inline]
    pub const fn capacity(&self) -> usize {
        self.buf.len()
    }

    /// Returns the length of the filled part of the buffer
    #[inline]
    pub const fn len(&self) -> usize {
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
    pub const fn remaining(&self) -> usize {
        self.capacity() - self.filled
    }

    #[inline]
    pub const fn is_full(&self) -> bool {
        self.filled == self.buf.len()
    }

    pub const fn push(&mut self, byte: u8) {
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

    #[expect(
        clippy::inline_always,
        reason = "inlining the fast DEFLATE back-reference copy improves RT99 paired PDF decoding by 5.45%"
    )]
    #[inline(always)]
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

        // DEFLATE matches may overlap their output. `copy_within` has memmove semantics,
        // which is not enough for a match longer than its distance: newly written bytes
        // become the source for later bytes. Short overlaps copy forward;
        // longer matches seed one distance and then double the initialized
        // prefix using disjoint copies, without architecture-specific dispatch.
        if length > offset_from_end {
            if offset_from_end == 1 {
                // A distance-one match repeats a single initialized byte.
                // Filling the span avoids up to eight overlapping memmoves.
                let repeated = buf[start];
                buf[current..end].fill(repeated);
            } else if length <= SCALAR_OVERLAP_COPY_LIMIT {
                // Short overlaps are a forward recurrence: each source byte is
                // initialized history or was written by an earlier iteration.
                for i in 0..length {
                    buf[current + i] = buf[start + i];
                }
            } else if offset_from_end < MAX_REPLICATED_PERIOD
                && (offset_from_end.is_power_of_two()
                    || length <= MAX_PERIOD_REPETITIONS * offset_from_end)
            {
                // SAFETY: the validated distance and output capacity
                // establish the initialized period and writable span.
                unsafe { self.copy_short_period(offset_from_end, length) };
            } else {
                // Seed one distance from disjoint history, then repeatedly
                // double the initialized output prefix with disjoint copies.
                let (previous, output) = buf.split_at_mut(current);
                output[..offset_from_end].copy_from_slice(&previous[start..current]);
                let mut initialized = offset_from_end;
                while initialized < length {
                    let copy_len = initialized.min(length - initialized);
                    let (prefix, tail) = buf.split_at_mut(current + initialized);
                    tail[..copy_len].copy_from_slice(&prefix[current..current + copy_len]);
                    initialized += copy_len;
                }
            }
        } else {
            // Non-overlapping matches are disjoint by definition.
            let (previous, output) = buf.split_at_mut(current);
            output[..length].copy_from_slice(&previous[start..start + length]);
        }
        // Commit the initialized extent only after every copy succeeds.
        self.filled = end;
    }

    /// Repeat a short initialized period without reading newly written
    /// output. Fixed endpoint values cover the whole period.
    ///
    /// # Safety
    /// N <= distance < 2*N, distance bytes of initialized history precede
    /// current, and length bytes from current are writable.
    #[inline]
    unsafe fn repeat_period<const N: usize>(
        ptr: *mut MaybeUninit<u8>,
        current: usize,
        distance: usize,
        length: usize,
    ) {
        debug_assert!(distance >= N && distance < 2 * N);
        let start = current - distance;
        let tail = distance - N;
        // SAFETY: both loads lie in the initialized history period.
        let first = unsafe { ptr.add(start).cast::<[u8; N]>().read_unaligned() };
        // SAFETY: the second endpoint ends exactly at current.
        let last = unsafe { ptr.add(start + tail).cast::<[u8; N]>().read_unaligned() };
        let mut written = 0;
        while length - written >= distance {
            // SAFETY: the two endpoint stores cover exactly one complete
            // period within the validated output range.
            unsafe {
                ptr.add(current + written)
                    .cast::<[u8; N]>()
                    .write_unaligned(first);
                ptr.add(current + written + tail)
                    .cast::<[u8; N]>()
                    .write_unaligned(last);
            }
            written += distance;
        }
        // SAFETY: the remaining prefix of the period is initialized history,
        // disjoint from all output, and the destination has exactly this space.
        unsafe {
            core::ptr::copy_nonoverlapping(
                ptr.add(start),
                ptr.add(current + written),
                length - written,
            );
        }
    }

    /// Repeat an aligned power-of-two period using an expanded 32-byte
    /// chunk. This stays separate from arbitrary-period endpoint copying.
    ///
    /// # Safety
    /// N is a power of two <= 32; N initialized bytes precede current,
    /// and length writable bytes follow current.
    #[inline(never)]
    unsafe fn repeat_aligned_period<const N: usize>(
        ptr: *mut MaybeUninit<u8>,
        current: usize,
        length: usize,
    ) {
        // SAFETY: the fixed source period is initialized.
        let seed = unsafe { ptr.add(current - N).cast::<[u8; N]>().read_unaligned() };
        let chunk: [u8; 32] = core::array::from_fn(|i| seed[i % N]);
        let mut written = 0;
        while length - written >= chunk.len() {
            // SAFETY: the next full chunk lies in the writable output span.
            unsafe {
                ptr.add(current + written)
                    .cast::<[u8; 32]>()
                    .write_unaligned(chunk)
            };
            written += chunk.len();
        }
        // SAFETY: the remaining prefix is fully initialized within chunk,
        // and the destination has precisely the remaining writable bytes.
        unsafe {
            core::ptr::copy_nonoverlapping(
                chunk.as_ptr().cast(),
                ptr.add(current + written),
                length - written,
            );
        }
    }

    /// Replicate one initialized short period without reading new output.
    ///
    /// # Safety
    /// Distance is positive and less than MAX_REPLICATED_PERIOD, the
    /// preceding distance bytes are initialized, and length bytes fit
    /// in the output capacity. The caller commits the final extent.
    #[cold]
    #[inline(never)]
    unsafe fn copy_short_period(&mut self, distance: usize, length: usize) {
        let current = self.filled;
        let ptr = self.buf.as_mut_ptr();
        // SAFETY: the caller guarantees the initialized source period
        // and writable destination. Width is the largest power of two
        // not exceeding distance, which is less than 64.
        unsafe {
            match distance {
                2 => return Self::repeat_aligned_period::<2>(ptr, current, length),
                4 => return Self::repeat_aligned_period::<4>(ptr, current, length),
                8 => return Self::repeat_aligned_period::<8>(ptr, current, length),
                16 => return Self::repeat_aligned_period::<16>(ptr, current, length),
                32 => return Self::repeat_aligned_period::<32>(ptr, current, length),
                _ => {}
            }
            if distance >= 32 {
                Self::repeat_period::<32>(ptr, current, distance, length);
            } else if distance >= 16 {
                Self::repeat_period::<16>(ptr, current, distance, length);
            } else if distance >= 8 {
                Self::repeat_period::<8>(ptr, current, distance, length);
            } else if distance >= 4 {
                Self::repeat_period::<4>(ptr, current, distance, length);
            } else if distance >= 2 {
                Self::repeat_period::<2>(ptr, current, distance, length);
            } else {
                Self::repeat_period::<1>(ptr, current, distance, length);
            }
        }
    }

    /// Copy one fixed-size initialized chunk between disjoint buffers.
    ///
    /// # Safety
    /// Both source and destination N-byte spans must lie in the allocation,
    /// source bytes must be initialized, and the spans must not overlap.
    #[inline]
    const unsafe fn copy_disjoint_chunk<const N: usize>(
        ptr: *mut MaybeUninit<u8>,
        src: usize,
        dst: usize,
    ) {
        // SAFETY: source and destination validity, initialization, and
        // disjointness are provided by the caller. Byte arrays have no
        // alignment requirement; LLVM can generate an unaligned vector move.
        let value = unsafe { ptr.add(src).cast::<[u8; N]>().read_unaligned() };
        // SAFETY: the caller guaranteed this is writable and disjoint
        // from the initialized source; N bytes are within the allocation.
        unsafe { ptr.add(dst).cast::<[u8; N]>().write_unaligned(value) };
    }

    /// Copy a disjoint DEFLATE match in exact N-byte chunks, using a final
    /// overlapping *destination* chunk anchored at the logical end.
    ///
    /// # Safety
    /// The full length must be at least N, the entire source span is
    /// initialized before dst, and the destination span has length bytes.
    #[inline]
    unsafe fn copy_disjoint_chunks<const N: usize>(
        ptr: *mut MaybeUninit<u8>,
        src: usize,
        dst: usize,
        length: usize,
    ) {
        debug_assert!(length >= N);
        let mut offset = 0;
        while offset + N < length {
            // SAFETY: offset + N < length; both ranges fit entirely within
            // the disjoint initialized source and writable destination.
            unsafe { Self::copy_disjoint_chunk::<N>(ptr, src + offset, dst + offset) };
            offset += N;
        }
        // SAFETY: length >= N, so this last chunk lies exactly within the
        // logical source/output spans even when it overlaps an earlier
        // destination chunk. It cannot affect bytes after the reported end.
        unsafe { Self::copy_disjoint_chunk::<N>(ptr, src + length - N, dst + length - N) };
    }

    /// Copy a span whose size is in [N, 2*N). Both endpoint loads happen
    /// before either store; no length-dependent loop or branch is needed.
    ///
    /// # Safety
    /// The source and destination spans of length bytes are disjoint,
    /// initialized/readable and writable respectively, and N <= length < 2*N.
    #[inline]
    unsafe fn copy_disjoint_endpoints<const N: usize>(
        ptr: *mut MaybeUninit<u8>,
        src: usize,
        dst: usize,
        length: usize,
    ) {
        debug_assert!(length >= N && length < 2 * N);
        let tail = length - N;
        // SAFETY: both endpoint ranges lie within the disjoint validated spans.
        let first = unsafe { ptr.add(src).cast::<[u8; N]>().read_unaligned() };
        // SAFETY: tail + N == length, so the second load is also initialized.
        let last = unsafe { ptr.add(src + tail).cast::<[u8; N]>().read_unaligned() };
        // SAFETY: both endpoint stores fit the exact destination span.
        unsafe {
            ptr.add(dst).cast::<[u8; N]>().write_unaligned(first);
            ptr.add(dst + tail).cast::<[u8; N]>().write_unaligned(last);
        }
    }

    /// Fast ordinary-inflate copy for a non-overlapping DEFLATE reference.
    /// The callback writer uses the original, exact match copy instead.
    // The large fast-path dispatch needs to be inlined at its one-shot
    // decoder call site. A plain inline hint loses 10–15% end-to-end
    // decode throughput on the tested Zen2/Zen5 PDFs.
    #[expect(
        clippy::inline_always,
        reason = "measured fast-inflater throughput requires inlining this dispatch"
    )]
    #[inline(always)]
    pub(super) fn copy_match_fast(&mut self, distance: usize, length: usize) {
        assert_ne!(distance, 0, "DEFLATE match distance must be positive");
        assert!(distance <= self.filled, "match distance in bounds");
        assert!(length <= self.remaining(), "match fits output buffer");

        if length == 0 {
            return;
        }
        if length > distance {
            self.copy_match(distance, length);
            return;
        }

        let start = self.filled - distance;
        let current = self.filled;
        let ptr = self.buf.as_mut_ptr();

        // The full source span lies in the initialized prefix and ends
        // before current. Every store is strictly within the logical
        // destination [current, current+length); C ABI output canaries and
        // caller-owned bytes after the reported output are untouched.
        // The width is known at compile time so LLVM may choose native
        // vector moves without handwritten architecture intrinsics.
        // SAFETY: distance >= length, so the complete source is in the
        // initialized prefix before current. Each chosen chunk stays
        // entirely inside the source and logical destination spans.
        unsafe {
            if length >= MATCH_COPY_WIDTH {
                Self::copy_disjoint_chunks::<MATCH_COPY_WIDTH>(ptr, start, current, length);
            } else if length >= 16 {
                Self::copy_disjoint_endpoints::<16>(ptr, start, current, length);
            } else if length >= 8 {
                Self::copy_disjoint_endpoints::<8>(ptr, start, current, length);
            } else if length >= 4 {
                Self::copy_disjoint_endpoints::<4>(ptr, start, current, length);
            } else if length >= 2 {
                Self::copy_disjoint_endpoints::<2>(ptr, start, current, length);
            } else {
                Self::copy_disjoint_endpoints::<1>(ptr, start, current, length);
            }
        }

        self.filled = current + length;
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
    pub(crate) const fn len(&self) -> usize {
        self.writer.len()
    }

    #[inline]
    pub(crate) const fn remaining(&self) -> usize {
        self.writer.remaining()
    }

    #[inline]
    pub(crate) const fn push(&mut self, byte: u8) {
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

    fn test_array() -> [u8; N] {
        core::array::from_fn(|i| {
            if i < M {
                u8::try_from(i).unwrap_or(0xaa)
            } else {
                0xaa
            }
        })
    }

    fn test_copy_match(offset_from_end: usize, length: usize) {
        let mut buf = test_array().map(MaybeUninit::new);
        let mut writer = Writer {
            buf: WeakSliceMut::from_mut_slice(&mut buf),
            filled: M,
        };
        writer.copy_match(offset_from_end, length);
        assert_eq!(writer.len(), M + length);

        let mut expected = test_array();
        for i in 0..length {
            expected[M + i] = expected[M - offset_from_end + i];
        }

        assert_eq!(
            &writer.filled()[M..],
            &expected[M..M + length],
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
                test_copy_match(offset_from_end, length);
            }
        }
    }

    #[test]
    fn copy_match_initializes_uninitialized_output_tail() {
        // Copy the same references into genuinely uninitialized backing
        // storage, not merely a buffer pre-filled with known sentinel bytes.
        // Only the decoded prefix may ever be materialized as typed bytes.
        let prefix: [u8; 64] = *b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for distance in [1, 2, 3, 4, 8, 16, 32, 64] {
            for length in [0, 1, distance - 1, distance, distance + 1, 258] {
                let mut storage = [MaybeUninit::uninit(); 64 + 258];
                for (slot, &byte) in storage.iter_mut().zip(prefix.iter()) {
                    slot.write(byte);
                }
                let mut writer = Writer {
                    buf: WeakSliceMut::from_mut_slice(&mut storage),
                    filled: prefix.len(),
                };

                let mut expected = prefix.to_vec();
                for _ in 0..length {
                    expected.push(expected[expected.len() - distance]);
                }

                writer.copy_match(distance, length);
                assert_eq!(writer.len(), expected.len());
                assert_eq!(
                    writer.filled(),
                    expected,
                    "distance {distance}, length {length}"
                );
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore = "exhaustive copy-length and slack matrix")]
    fn fast_copy_match_preserves_logical_bytes_and_allocation_canaries() {
        // Include a genuinely uninitialized output tail, unaligned lengths,
        // every RFC 1951 match length, and both vector and boundary paths.
        let prefix: [u8; 64] = core::array::from_fn(|i| u8::try_from(i + 1).unwrap_or(0));
        for distance in 1..=prefix.len() {
            for length in 0..=258 {
                for slack in [
                    0,
                    1,
                    7,
                    MATCH_COPY_WIDTH - 1,
                    MATCH_COPY_WIDTH,
                    MATCH_COPY_WIDTH + 1,
                ] {
                    let capacity = prefix.len() + length + slack;
                    let mut storage = vec![MaybeUninit::<u8>::uninit(); capacity + 32];
                    for (slot, byte) in storage.iter_mut().zip(prefix) {
                        slot.write(byte);
                    }
                    for slot in &mut storage[capacity..] {
                        slot.write(0xa5);
                    }
                    for slot in &mut storage[prefix.len() + length..capacity] {
                        slot.write(0x5a);
                    }

                    let mut expected = prefix.to_vec();
                    for _ in 0..length {
                        let source = expected.len() - distance;
                        expected.push(expected[source]);
                    }

                    {
                        let mut writer = Writer {
                            buf: WeakSliceMut::from_mut_slice(&mut storage[..capacity]),
                            filled: prefix.len(),
                        };
                        writer.copy_match_fast(distance, length);
                        assert_eq!(writer.len(), expected.len());
                        assert_eq!(
                            writer.filled(),
                            expected,
                            "distance {distance}, length {length}, slack {slack}"
                        );
                    }
                    for (i, slot) in storage[prefix.len() + length..capacity].iter().enumerate() {
                        // SAFETY: slack canaries were initialized above.
                        assert_eq!(
                            {
                                // SAFETY: slack canaries were
                                // initialized before the tested copy.
                                unsafe { slot.assume_init() }
                            },
                            0x5a,
                            "unexpected write after logical output at {i}, distance {distance}, length {length}, slack {slack}"
                        );
                    }
                    for (i, slot) in storage[capacity..].iter().enumerate() {
                        // SAFETY: the allocation canaries were initialized above
                        // and the copy routine must not write beyond capacity.
                        assert_eq!(
                            {
                                // SAFETY: allocation guard canaries were
                                // initialized before the tested copy.
                                unsafe { slot.assume_init() }
                            },
                            0xa5,
                            "out-of-bounds write at {i}, distance {distance}, length {length}, slack {slack}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn callback_writer_preserves_unused_previous_window_history() {
        let mut storage = core::array::from_fn::<u8, 128, _>(|i| u8::try_from(i).unwrap_or(0));
        {
            let mut window = super::super::window::BackWindow::from_initialized_slice(&mut storage);
            window.mark_full_history();
            let mut writer = BackWriter::new(&mut window, 0);
            assert_eq!(writer.copy_match(true, 64, 1), Ok(()));
        }
        // A padded 32-byte store here would destroy previous-window bytes
        // that subsequent inflateBack callbacks might need.
        assert_eq!(storage[0], 64);
        assert_eq!(&storage[1..64], &(1u8..64).collect::<Vec<_>>());
    }

    #[test]
    fn copy_match_at_exact_buffer_end() {
        let mut buf = [1, 2, 3, 0xAA, 0xAA].map(MaybeUninit::new);
        let mut writer = Writer {
            buf: WeakSliceMut::from_mut_slice(&mut buf),
            filled: 3,
        };

        writer.copy_match(3, 2);

        assert_eq!(writer.filled(), &[1, 2, 3, 1, 2]);
    }
}
