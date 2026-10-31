use core::mem::MaybeUninit;

use crate::{
    adler32::{adler32, adler32_fold_copy},
    crc32::Crc32Fold,
    weak_slice::WeakSliceMut,
};

// translation guide:
//
// wsize -> buf.capacity()
// wnext -> buf.ptr
// whave -> buf.filled.len()
#[derive(Debug)]
pub struct Window<'a> {
    buf: WeakSliceMut<'a, u8>,

    have: usize, // number of initialized/logically available bytes, bounded by the backing buffer
    next: usize, // write head
}

impl<'a> Window<'a> {
    pub fn into_raw_parts(self) -> (*mut u8, usize) {
        self.buf.into_raw_parts()
    }

    /// Bind fully initialized caller-owned byte storage.
    ///
    /// # Safety
    ///
    /// `ptr` must address `len` initialized, writable bytes and remain uniquely owned for the
    /// returned lifetime.
    pub unsafe fn from_raw_parts(ptr: *mut u8, len: usize) -> Self {
        Self {
            buf: unsafe { WeakSliceMut::from_raw_parts_mut(ptr, len) },
            have: 0,
            next: 0,
        }
    }

    #[cfg(test)]
    pub(crate) fn from_initialized_slice(storage: &'a mut [u8]) -> Self {
        // SAFETY: a mutable byte slice is fully initialized, writable, and uniquely borrowed for
        // the returned window lifetime.
        unsafe { Self::from_raw_parts(storage.as_mut_ptr(), storage.len()) }
    }

    /// Initialize an owned raw window before exposing it through typed `u8` slices.
    ///
    /// # Safety
    ///
    /// `ptr` must be writable for `len` bytes and remain uniquely owned for the returned lifetime.
    pub unsafe fn from_uninitialized_raw_parts(ptr: *mut u8, len: usize) -> Self {
        unsafe { ptr.write_bytes(0, len) };
        unsafe { Self::from_raw_parts(ptr, len) }
    }

    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }

    pub fn size(&self) -> usize {
        // `self.len == 0` is used for uninitialized buffers
        assert!(self.buf.is_empty() || self.buf.len() >= Self::padding());
        self.buf.len().saturating_sub(Self::padding())
    }

    /// number of bytes in the window. Saturates at `Self::capacity`.
    pub fn have(&self) -> usize {
        self.have
    }

    /// Position where the next byte will be written
    pub fn next(&self) -> usize {
        self.next
    }

    pub fn empty() -> Self {
        Self {
            buf: WeakSliceMut::empty(),
            have: 0,
            next: 0,
        }
    }

    pub fn clear(&mut self) {
        self.have = 0;
        self.next = 0;
    }

    pub fn as_slice(&self) -> &[u8] {
        debug_assert!(self.have <= self.buf.len());
        // SAFETY: `have` tracks exactly the initialized prefix of the backing window. In
        // particular, inflateBack's caller-provided tail may still be uninitialized, so never
        // materialize a typed slice over the full capacity before trimming it.
        unsafe { core::slice::from_raw_parts(self.buf.as_ptr(), self.have) }
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.buf.as_ptr()
    }

    /// Return the logical inflate dictionary as at most two contiguous segments, oldest first.
    ///
    /// Before the circular window fills, all history is one initialized prefix. Once full, `next`
    /// points at the oldest byte, so the logical dictionary is the tail followed by the head.
    pub fn dictionary_segments(&self) -> (&[u8], &[u8]) {
        let initialized = self.as_slice();
        if self.have < self.size() {
            debug_assert_eq!(self.next, self.have);
            (initialized, &[])
        } else {
            debug_assert!(self.next <= self.have);
            let (head, tail) = initialized.split_at(self.next);
            (tail, head)
        }
    }

    #[cfg(test)]
    fn extend_adler32(&mut self, slice: &[u8], checksum: &mut u32) {
        self.extend(slice, 0, true, checksum, &mut Crc32Fold::new());
    }

    pub(crate) fn extend(
        &mut self,
        slice: &[u8],
        flags: i32,
        update_checksum: bool,
        checksum: &mut u32,
        crc_fold: &mut Crc32Fold,
    ) {
        let len = slice.len();
        let wsize = self.size();

        if len >= wsize {
            // We have to split the checksum over non-copied and copied bytes
            let pos = len.saturating_sub(self.size());
            let (non_window_slice, window_slice) = slice.split_at(pos);

            if update_checksum {
                if flags != 0 {
                    crc_fold.fold(non_window_slice);
                    crc_fold.fold_copy(&mut self.buf.as_mut_slice()[..wsize], window_slice);
                } else {
                    *checksum = adler32(*checksum, non_window_slice);
                    *checksum = adler32_fold_copy(*checksum, self.buf.as_mut_slice(), window_slice);
                }
            } else {
                self.buf.as_mut_slice()[..wsize].copy_from_slice(window_slice);
            }

            self.next = 0;
            self.have = self.size();
        } else {
            let dist = Ord::min(wsize - self.next, slice.len());

            // the end part goes onto the end of the window. The start part wraps around and is
            // written to the start of the window.
            let (end_part, start_part) = slice.split_at(dist);

            if update_checksum {
                let dst = &mut self.buf.as_mut_slice()[self.next..][..end_part.len()];
                if flags != 0 {
                    crc_fold.fold_copy(dst, end_part);
                } else {
                    *checksum = adler32_fold_copy(*checksum, dst, end_part);
                }
            } else {
                self.buf.as_mut_slice()[self.next..][..end_part.len()].copy_from_slice(end_part);
            }

            if !start_part.is_empty() {
                let dst = &mut self.buf.as_mut_slice()[..start_part.len()];

                if update_checksum {
                    if flags != 0 {
                        crc_fold.fold_copy(dst, start_part);
                    } else {
                        *checksum = adler32_fold_copy(*checksum, dst, start_part);
                    }
                } else {
                    dst.copy_from_slice(start_part);
                }

                self.next = start_part.len();
                self.have = self.size();
            } else {
                self.next += dist;
                if self.next == self.size() {
                    self.next = 0;
                }
                if self.have < self.size() {
                    self.have += dist;
                }
            }
        }
    }

    #[cfg(test)]
    pub fn new_in(alloc: &crate::inflate::Allocator<'a>, window_bits: usize) -> Option<Self> {
        let len = (1 << window_bits) + Self::padding();
        let ptr = alloc.allocate_zeroed_buffer(len)?;

        Some(Self {
            buf: unsafe { WeakSliceMut::from_raw_parts_mut(ptr.as_ptr(), len) },
            have: 0,
            next: 0,
        })
    }

    /// Clone this complete backing buffer into independent raw storage.
    ///
    /// # Safety
    /// `ptr` must be valid and uniquely writable for `self.buf.len()` bytes, and must not overlap
    /// this window's backing allocation.
    pub(super) unsafe fn clone_to(&self, ptr: *mut u8) -> Self {
        let len = self.buf.len();
        unsafe { core::ptr::copy_nonoverlapping(self.buf.as_ptr(), ptr, len) };

        Self {
            buf: unsafe { WeakSliceMut::from_raw_parts_mut(ptr, len) },
            have: self.have,
            next: self.next,
        }
    }

    // padding required so that SIMD operations going out-of-bounds are not a problem
    pub fn padding() -> usize {
        64 // very conservative
    }
}

/// Caller-owned `inflateBack` output/history storage.
///
/// Unlike ordinary inflate's [`Window`], this backing allocation is allowed to start completely
/// uninitialized. Keeping it as `MaybeUninit<u8>` means callback mode cannot accidentally
/// materialize a typed byte slice over unwritten tail storage. History is represented only as
/// "no previous full window" or "one complete previous full window"; the current output prefix is
/// owned by `BackOutput`.
#[derive(Debug)]
pub(crate) struct BackWindow<'a> {
    buf: WeakSliceMut<'a, MaybeUninit<u8>>,
    has_full_history: bool,
}

impl<'a> BackWindow<'a> {
    /// Bind caller-owned scratch storage without reading or initializing it.
    ///
    /// # Safety
    ///
    /// `ptr` must be writable for `len` bytes and remain uniquely owned for the returned lifetime.
    pub unsafe fn from_raw_parts(ptr: *mut u8, len: usize) -> Self {
        Self {
            buf: unsafe { WeakSliceMut::from_raw_parts_mut(ptr.cast(), len) },
            has_full_history: false,
        }
    }

    #[cfg(test)]
    pub(crate) fn from_initialized_slice(storage: &'a mut [u8]) -> Self {
        // SAFETY: representing initialized writable bytes as `MaybeUninit<u8>` only weakens the
        // initialization guarantee while preserving the exclusive borrow.
        unsafe { Self::from_raw_parts(storage.as_mut_ptr(), storage.len()) }
    }

    pub(crate) fn from_initialized_window(window: Window<'a>) -> Self {
        let (ptr, len) = window.into_raw_parts();
        // SAFETY: `Window` is only constructible over initialized writable storage that remains
        // uniquely owned for its lifetime; weakening that storage to MaybeUninit is sound.
        unsafe { Self::from_raw_parts(ptr, len) }
    }

    pub(crate) fn empty() -> Self {
        Self {
            buf: WeakSliceMut::empty(),
            has_full_history: false,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub(crate) fn buffer_size(&self) -> usize {
        self.buf.len()
    }

    pub(crate) fn history_len(&self) -> usize {
        if self.has_full_history {
            self.buf.len()
        } else {
            0
        }
    }

    pub(crate) fn mark_full_history(&mut self) {
        self.has_full_history = true;
    }

    pub(crate) fn clear(&mut self) {
        self.has_full_history = false;
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut MaybeUninit<u8> {
        self.buf.as_mut_ptr()
    }
}

#[cfg(all(test, feature = "rust-allocator"))]
mod test {
    use super::*;

    fn alloc_window(window_bits_log2: usize) -> Window<'static> {
        let mut window = Window::new_in(&crate::allocate::RUST, window_bits_log2).unwrap();
        window.have = 0;
        window.next = 0;
        window
    }

    #[test]
    fn back_window_tracks_only_complete_previous_history() {
        let mut storage = [MaybeUninit::<u8>::uninit(); 8];
        let mut window =
            unsafe { BackWindow::from_raw_parts(storage.as_mut_ptr().cast(), storage.len()) };

        assert_eq!(window.history_len(), 0);
        window.mark_full_history();
        assert_eq!(window.history_len(), storage.len());
        window.clear();
        assert_eq!(window.history_len(), 0);
    }

    #[test]
    fn owned_raw_constructor_initializes_backing_storage() {
        use core::mem::MaybeUninit;

        let mut storage = [MaybeUninit::<u8>::uninit(); 80];
        let window = unsafe {
            Window::from_uninitialized_raw_parts(storage.as_mut_ptr().cast::<u8>(), storage.len())
        };

        assert!(window.buf.as_slice().iter().all(|byte| *byte == 0));
    }

    #[test]
    fn window_init() {
        let window = alloc_window(2);
        assert_eq!(window.size(), 4);
        assert_eq!(window.have(), 0);
        assert!(!window.is_empty());
        let start = window.as_ptr();
        let size = window.size();
        let buffer_size = window.buf.len();
        let (ptr, len) = window.into_raw_parts();
        assert_eq!(ptr.cast_const(), start);
        assert!(len >= size); // >= because the impl is allowed to add padding to the internal buffer
        unsafe { crate::allocate::RUST.deallocate(ptr, buffer_size) }
    }

    #[test]
    fn extend_in_bounds() {
        let mut checksum = 0;

        let mut window = alloc_window(4);

        window.extend_adler32(&[1; 5], &mut checksum);
        assert_eq!(window.have, 5);
        assert_eq!(window.next, 5);

        let slice = &window.buf.as_slice()[..window.size()];
        assert_eq!(&[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], slice);

        window.extend_adler32(&[2; 7], &mut checksum);
        assert_eq!(window.have, 12);
        assert_eq!(window.next, 12);

        let slice = &window.buf.as_slice()[..window.size()];
        assert_eq!(&[1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0], slice);

        assert_eq!(checksum, 6946835);

        unsafe {
            crate::allocate::RUST.deallocate(
                window.buf.as_mut_slice().as_mut_ptr(),
                window.buf.as_slice().len(),
            )
        }
    }

    #[test]
    fn extend_crosses_bounds() {
        let mut checksum = 0;

        let mut window = alloc_window(2);

        window.extend_adler32(&[1; 3], &mut checksum);
        assert_eq!(window.have, 3);
        assert_eq!(window.next, 3);

        let slice = &window.buf.as_slice()[..window.size()];
        assert_eq!(&[1, 1, 1, 0], slice);

        window.extend_adler32(&[2; 3], &mut checksum);
        assert_eq!(window.have, 4);
        assert_eq!(window.next, 2);

        let slice = &window.buf.as_slice()[..window.size()];
        assert_eq!(&[2, 2, 1, 2], slice);

        assert_eq!(checksum, 1769481);

        unsafe {
            crate::allocate::RUST.deallocate(
                window.buf.as_mut_slice().as_mut_ptr(),
                window.buf.as_slice().len(),
            )
        }
    }

    #[test]
    fn dictionary_segments_preserve_logical_history_order() {
        let mut checksum = 0;
        let mut window = alloc_window(2);

        window.extend_adler32(&[1, 2, 3], &mut checksum);
        let (first, second) = window.dictionary_segments();
        assert_eq!(first, &[1, 2, 3]);
        assert!(second.is_empty());

        window.extend_adler32(&[4, 5, 6], &mut checksum);
        let (first, second) = window.dictionary_segments();
        assert_eq!(first, &[3, 4]);
        assert_eq!(second, &[5, 6]);

        unsafe {
            crate::allocate::RUST
                .deallocate(window.buf.as_mut_slice().as_mut_ptr(), window.buf.len())
        }
    }

    #[test]
    fn extend_out_of_bounds() {
        let mut checksum = 0;

        let mut window = alloc_window(3);

        // adds 9 numbers, that won't fit into a window of size 8
        window.extend_adler32(&[1, 2, 3, 4, 5, 6, 7, 8, 9], &mut checksum);
        assert_eq!(window.have, 8);
        assert_eq!(window.next, 0);

        let slice = &window.as_slice()[..window.size()];
        assert_eq!(&[2, 3, 4, 5, 6, 7, 8, 9], slice);

        assert_eq!(checksum, 10813485);

        unsafe {
            crate::allocate::RUST
                .deallocate(window.buf.as_mut_slice().as_mut_ptr(), window.buf.len())
        }
    }
}
