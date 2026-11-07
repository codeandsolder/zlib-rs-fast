use core::mem::MaybeUninit;

use crate::{adler32::adler32, crc32::Crc32Fold, weak_slice::WeakSliceMut};

// translation guide:
//
// wsize -> buf.capacity()
// wnext -> buf.ptr
// whave -> have
#[derive(Debug)]
pub struct Window<'a> {
    buf: WeakSliceMut<'a, MaybeUninit<u8>>,

    have: usize, // number of initialized/logically available bytes, bounded by the backing buffer
    next: usize, // write head
}

impl Window<'_> {
    #[must_use]
    pub const fn into_raw_parts(self) -> (*mut u8, usize) {
        let (ptr, len) = self.buf.into_raw_parts();
        (ptr.cast(), len)
    }

    /// Bind caller-owned history storage without assuming any byte is initialized yet.
    ///
    /// # Safety
    ///
    /// `ptr` must address `len` writable bytes and remain uniquely owned for the returned lifetime.
    pub const unsafe fn from_raw_parts(ptr: *mut u8, len: usize) -> Self {
        Self {
            // SAFETY: the constructor contract grants this exact writable span uniquely.
            buf: unsafe { WeakSliceMut::from_raw_parts_mut(ptr.cast(), len) },
            have: 0,
            next: 0,
        }
    }

    #[cfg(test)]
    pub(crate) const fn from_initialized_slice<'a>(storage: &'a mut [u8]) -> Window<'a> {
        // SAFETY: a mutable byte slice is fully initialized, writable, and uniquely borrowed for
        // the returned window lifetime.
        unsafe { Window::<'a>::from_raw_parts(storage.as_mut_ptr(), storage.len()) }
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.size() == 0
    }

    #[must_use]
    pub const fn size(&self) -> usize {
        self.buf.len()
    }

    /// Number of initialized logical history bytes, saturating at `size()`.
    #[must_use]
    pub const fn have(&self) -> usize {
        self.have
    }

    /// Position where the next byte will be written
    #[must_use]
    pub const fn next(&self) -> usize {
        self.next
    }

    #[must_use]
    pub const fn empty() -> Self {
        Self {
            buf: WeakSliceMut::empty(),
            have: 0,
            next: 0,
        }
    }

    pub const fn clear(&mut self) {
        self.have = 0;
        self.next = 0;
    }

    ///
    /// # Panics
    ///
    /// Panics if the logical history length exceeds storage or a partially filled
    /// window's write cursor differs from the initialized prefix length.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        assert!(
            self.have <= self.buf.len(),
            "inflate history stays in-bounds"
        );
        if self.have < self.buf.len() {
            assert_eq!(self.next, self.have, "partial inflate history is a prefix");
        }
        // SAFETY: before the circular window fills, `extend` initializes exactly `0..have` and
        // keeps `next == have`. Once `have == len`, every byte has been initialized.
        unsafe { self.buf.as_slice()[..self.have].assume_init_ref() }
    }

    #[must_use]
    pub const fn as_ptr(&self) -> *const u8 {
        self.buf.as_ptr().cast()
    }

    /// Return the logical inflate dictionary as at most two contiguous segments, oldest first.
    ///
    /// Before the circular window fills, all history is one initialized prefix. Once full, `next`
    /// points at the oldest byte, so the logical dictionary is the tail followed by the head.
    #[must_use]
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
            let pos = len - wsize;
            let (non_window_slice, window_slice) = slice.split_at(pos);

            if update_checksum {
                if flags != 0 {
                    crc_fold.fold(non_window_slice);
                    crc_fold.fold(window_slice);
                } else {
                    *checksum = adler32(*checksum, non_window_slice);
                    *checksum = adler32(*checksum, window_slice);
                }
            }
            self.buf.as_mut_slice()[..wsize].write_copy_of_slice(window_slice);
            self.next = 0;
            self.have = wsize;
            return;
        }

        let end_len = Ord::min(wsize - self.next, len);
        let (end_part, start_part) = slice.split_at(end_len);

        if update_checksum {
            if flags != 0 {
                crc_fold.fold(end_part);
            } else {
                *checksum = adler32(*checksum, end_part);
            }
        }
        self.buf.as_mut_slice()[self.next..][..end_part.len()].write_copy_of_slice(end_part);

        if start_part.is_empty() {
            self.next += end_len;
            if self.next == wsize {
                self.next = 0;
            }
            if self.have < wsize {
                self.have += end_len;
            }
        } else {
            if update_checksum {
                if flags != 0 {
                    crc_fold.fold(start_part);
                } else {
                    *checksum = adler32(*checksum, start_part);
                }
            }
            self.buf.as_mut_slice()[..start_part.len()].write_copy_of_slice(start_part);
            self.next = start_part.len();
            self.have = wsize;
        }
    }

    #[cfg(test)]
    #[must_use]
    pub fn new_in<'a>(
        alloc: &crate::inflate::Allocator<'a>,
        window_bits: usize,
    ) -> Option<Window<'a>> {
        let len = 1 << window_bits;
        let ptr = alloc.allocate_slice_raw::<u8>(len)?;

        // SAFETY: the allocator returned a unique writable allocation of exactly len bytes.
        Some(unsafe { Window::<'a>::from_raw_parts(ptr.as_ptr(), len) })
    }

    /// Clone only proven-initialized history into independent raw storage.
    ///
    /// # Safety
    /// `ptr` must be valid and uniquely writable for `self.buf.len()` bytes, and must not overlap
    /// this window's backing allocation.
    pub(super) unsafe fn clone_to(&self, ptr: *mut u8) -> Self {
        let len = self.buf.len();
        let initialized = self.as_slice();
        // SAFETY: initialized covers only live history; clone_to guarantees non-overlap,
        // writable destination storage and enough space for its full extent.
        unsafe { core::ptr::copy_nonoverlapping(initialized.as_ptr(), ptr, initialized.len()) };

        Self {
            // SAFETY: clone_to guarantees the destination is uniquely writable for len bytes.
            buf: unsafe { WeakSliceMut::from_raw_parts_mut(ptr.cast(), len) },
            have: self.have,
            next: self.next,
        }
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
pub struct BackWindow<'a> {
    buf: WeakSliceMut<'a, MaybeUninit<u8>>,
    has_full_history: bool,
}

impl<'a> BackWindow<'a> {
    /// Bind caller-owned scratch storage without reading or initializing it.
    ///
    /// # Safety
    ///
    /// `ptr` must be writable for `len` bytes and remain uniquely owned for the returned lifetime.
    pub const unsafe fn from_raw_parts(ptr: *mut u8, len: usize) -> Self {
        Self {
            // SAFETY: the constructor contract grants unique mutable access to the full span.
            buf: unsafe { WeakSliceMut::from_raw_parts_mut(ptr.cast(), len) },
            has_full_history: false,
        }
    }

    #[cfg(test)]
    pub(crate) const fn from_initialized_slice(storage: &'a mut [u8]) -> Self {
        // SAFETY: representing initialized writable bytes as `MaybeUninit<u8>` only weakens the
        // initialization guarantee while preserving the exclusive borrow.
        unsafe { Self::from_raw_parts(storage.as_mut_ptr(), storage.len()) }
    }

    pub(crate) const fn from_window_storage(window: Window<'a>) -> Self {
        let (ptr, len) = window.into_raw_parts();
        // SAFETY: `Window` owns this writable storage uniquely for its lifetime. Its initialized
        // history is tracked logically, while both window types keep the backing bytes as
        // `MaybeUninit<u8>`, so transferring the raw storage does not strengthen initialization.
        unsafe { Self::from_raw_parts(ptr, len) }
    }

    pub(crate) const fn empty() -> Self {
        Self {
            buf: WeakSliceMut::empty(),
            has_full_history: false,
        }
    }

    pub(crate) const fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub(crate) const fn buffer_size(&self) -> usize {
        self.buf.len()
    }

    pub(crate) const fn history_len(&self) -> usize {
        if self.has_full_history {
            self.buf.len()
        } else {
            0
        }
    }

    pub(crate) const fn mark_full_history(&mut self) {
        self.has_full_history = true;
    }

    pub(crate) const fn clear(&mut self) {
        self.has_full_history = false;
    }

    pub(crate) const fn as_mut_ptr(&mut self) -> *mut MaybeUninit<u8> {
        self.buf.as_mut_ptr()
    }

    /// Borrow storage without asserting that its unwritten tail is initialized.
    pub(crate) const fn as_uninit_slice(&mut self) -> &mut [MaybeUninit<u8>] {
        self.buf.as_mut_slice()
    }
}

#[cfg(all(test, feature = "rust-allocator"))]
mod test {
    use super::*;

    fn alloc_window(window_bits_log2: usize) -> Result<Window<'static>, &'static str> {
        Window::new_in(&crate::allocate::RUST, window_bits_log2)
            .ok_or("inflate window test allocation failed")
    }

    /// Release the single allocation that belongs to a test-created window.
    fn free_window(window: Window<'static>) {
        let (ptr, size) = window.into_raw_parts();
        // SAFETY: alloc_window created this allocation with RUST. It is now
        // consumed and no references into its backing storage survive.
        unsafe { crate::allocate::RUST.deallocate(ptr, size) };
    }

    #[test]
    fn back_window_tracks_only_complete_previous_history() {
        let mut storage = [MaybeUninit::<u8>::uninit(); 8];
        // SAFETY: storage is writable for len bytes and stays borrowed for the whole test.
        let mut window =
            unsafe { BackWindow::from_raw_parts(storage.as_mut_ptr().cast(), storage.len()) };

        assert_eq!(window.history_len(), 0);
        window.mark_full_history();
        assert_eq!(window.history_len(), storage.len());
        window.clear();
        assert_eq!(window.history_len(), 0);
    }

    #[test]
    fn raw_constructor_does_not_initialize_backing_storage() {
        use core::mem::MaybeUninit;

        let mut storage = [MaybeUninit::new(0xa5u8); 80];
        // SAFETY: storage is uniquely writable for its full length in this test scope.
        let window = unsafe { Window::from_raw_parts(storage.as_mut_ptr().cast(), storage.len()) };
        assert_eq!(window.have(), 0);
        assert_eq!(window.as_slice(), &[]);

        let (ptr, len) = window.into_raw_parts();
        // SAFETY: this test initialized every backing byte to 0xa5 before constructing the window.
        let bytes = unsafe { core::slice::from_raw_parts(ptr, len) };
        assert!(bytes.iter().all(|&byte| byte == 0xa5));
    }

    #[test]
    fn clone_copies_only_initialized_history_prefix() {
        use core::mem::MaybeUninit;

        let mut source_storage = [MaybeUninit::<u8>::uninit(); 16];
        let mut destination_storage = [MaybeUninit::new(0x5au8); 16];
        // SAFETY: source_storage is uniquely writable for its entire test lifetime.
        let mut source = unsafe {
            Window::from_raw_parts(source_storage.as_mut_ptr().cast(), source_storage.len())
        };
        let mut checksum = 0;
        source.extend_adler32(&[1, 2, 3, 4, 5], &mut checksum);

        // SAFETY: destination_storage is independently writable, initialized, and at least
        // source.size() bytes long; neither storage span overlaps the other.
        let cloned = unsafe { source.clone_to(destination_storage.as_mut_ptr().cast()) };
        assert_eq!(cloned.have(), 5);
        assert_eq!(cloned.next(), 5);
        assert_eq!(cloned.size(), 16);
        assert_eq!(cloned.as_slice(), &[1, 2, 3, 4, 5]);

        // SAFETY: destination_storage was initialized to 0x5a before clone_to. The clone must not
        // touch the uninitialized logical tail, so those initialized sentinel bytes remain intact.
        let destination = unsafe { destination_storage.assume_init_ref() };
        assert_eq!(&destination[..5], &[1, 2, 3, 4, 5]);
        assert!(destination[5..].iter().all(|&byte| byte == 0x5a));
    }

    #[test]
    fn window_init() -> Result<(), &'static str> {
        let window = alloc_window(2)?;
        assert_eq!(window.size(), 4);
        assert_eq!(window.have(), 0);
        assert!(!window.is_empty());
        let start = window.as_ptr();
        let size = window.size();
        let (ptr, len) = window.into_raw_parts();
        assert_eq!(ptr.cast_const(), start);
        assert_eq!(len, size);
        // SAFETY: this allocation came from RUST and into_raw_parts consumed its sole owner.
        unsafe { crate::allocate::RUST.deallocate(ptr, len) };
        Ok(())
    }

    #[test]
    fn extend_in_bounds() -> Result<(), &'static str> {
        let mut checksum = 0;

        let mut window = alloc_window(4)?;

        window.extend_adler32(&[1; 5], &mut checksum);
        assert_eq!(window.have, 5);
        assert_eq!(window.next, 5);

        let slice = window.as_slice();
        assert_eq!(&[1, 1, 1, 1, 1], slice);

        window.extend_adler32(&[2; 7], &mut checksum);
        assert_eq!(window.have, 12);
        assert_eq!(window.next, 12);

        let slice = window.as_slice();
        assert_eq!(&[1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2], slice);

        assert_eq!(checksum, 6_946_835);

        free_window(window);
        Ok(())
    }

    #[test]
    fn extend_crosses_bounds() -> Result<(), &'static str> {
        let mut checksum = 0;

        let mut window = alloc_window(2)?;

        window.extend_adler32(&[1; 3], &mut checksum);
        assert_eq!(window.have, 3);
        assert_eq!(window.next, 3);

        let slice = window.as_slice();
        assert_eq!(&[1, 1, 1], slice);

        window.extend_adler32(&[2; 3], &mut checksum);
        assert_eq!(window.have, 4);
        assert_eq!(window.next, 2);

        let slice = window.as_slice();
        assert_eq!(&[2, 2, 1, 2], slice);

        assert_eq!(checksum, 1_769_481);

        free_window(window);
        Ok(())
    }

    #[test]
    fn dictionary_segments_preserve_logical_history_order() -> Result<(), &'static str> {
        let mut checksum = 0;
        let mut window = alloc_window(2)?;

        window.extend_adler32(&[1, 2, 3], &mut checksum);
        let (first, second) = window.dictionary_segments();
        assert_eq!(first, &[1, 2, 3]);
        assert_eq!(second, &[]);

        window.extend_adler32(&[4, 5, 6], &mut checksum);
        let (first, second) = window.dictionary_segments();
        assert_eq!(first, &[3, 4]);
        assert_eq!(second, &[5, 6]);

        free_window(window);
        Ok(())
    }

    #[test]
    fn extend_out_of_bounds() -> Result<(), &'static str> {
        let mut checksum = 0;

        let mut window = alloc_window(3)?;

        // adds 9 numbers, that won't fit into a window of size 8
        window.extend_adler32(&[1, 2, 3, 4, 5, 6, 7, 8, 9], &mut checksum);
        assert_eq!(window.have, 8);
        assert_eq!(window.next, 0);

        let slice = window.as_slice();
        assert_eq!(&[2, 3, 4, 5, 6, 7, 8, 9], slice);

        assert_eq!(checksum, 10_813_485);

        free_window(window);
        Ok(())
    }
}
