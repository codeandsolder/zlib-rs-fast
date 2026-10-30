use core::mem::MaybeUninit;

use crate::weak_slice::WeakSliceMut;

#[derive(Debug)]
pub struct Window<'a> {
    // Raw backing storage. Only `buf[..initialized]` may be observed as bytes.
    buf: WeakSliceMut<'a, MaybeUninit<u8>>,
    initialized: usize,
}

impl<'a> Window<'a> {
    /// Construct an empty logical window over raw writable storage.
    ///
    /// # Safety
    ///
    /// `ptr` must be valid for writes of `2 * (1 << window_bits)` bytes for the returned lifetime
    /// and must not alias another live mutable view. The backing bytes may be uninitialized.
    pub unsafe fn from_raw_parts(ptr: *mut u8, window_bits: usize) -> Self {
        let len = (1 << window_bits) * 2;
        let buf = unsafe { WeakSliceMut::from_raw_parts_mut(ptr.cast::<MaybeUninit<u8>>(), len) };

        Self {
            buf,
            initialized: 0,
        }
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.buf.as_ptr().cast()
    }

    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    /// Return at most `max_len` bytes of history ending at `end`, in logical input order.
    pub fn recent_history(&self, end: usize, max_len: usize) -> &[u8] {
        let window = self.filled();
        assert!(
            end <= window.len(),
            "history end stays inside initialized deflate window"
        );
        &window[end.saturating_sub(max_len)..end]
    }

    /// Returns the logically initialized prefix of the window.
    #[inline]
    pub fn filled(&self) -> &[u8] {
        let slice = &self.buf.as_slice()[..self.initialized];
        // SAFETY: `initialized` advances only after those bytes are written, and may shrink without
        // exposing any previously invalid byte again.
        unsafe { slice.assume_init_ref() }
    }

    /// Returns a mutable view of the logically initialized prefix of the window.
    #[inline]
    pub fn filled_mut(&mut self) -> &mut [u8] {
        let slice = &mut self.buf.as_mut_slice()[..self.initialized];
        // SAFETY: same initialized-prefix invariant as `filled()`; the borrow is exclusive.
        unsafe { slice.assume_init_mut() }
    }

    /// Forget all logical window contents without touching the backing allocation.
    #[inline]
    pub fn clear(&mut self) {
        self.initialized = 0;
    }

    /// Move the initialized bytes at `offset..` to the start of the window and discard the prefix.
    #[inline]
    pub fn slide_down(&mut self, offset: usize) {
        assert!(
            offset <= self.initialized,
            "window slide exceeds initialized history"
        );
        let retained = self.initialized - offset;
        if retained != 0 {
            self.buf
                .as_mut_slice()
                .copy_within(offset..offset + retained, 0);
        }
        self.initialized = retained;
    }

    /// Copy bytes into the logical window and extend its initialized prefix.
    ///
    /// # Safety
    ///
    /// `src` must point to `range.len()` readable initialized bytes. The caller-input contract may
    /// permit concurrent mutation, so this operation intentionally copies from the raw pointer
    /// without manufacturing an immutable source slice.
    pub unsafe fn copy_and_initialize(&mut self, range: core::ops::Range<usize>, src: *const u8) {
        assert!(range.start <= range.end, "invalid window write range");
        assert!(
            range.end <= self.buf.len(),
            "window write exceeds allocation"
        );
        assert!(
            range.start <= self.initialized,
            "window writes must keep the initialized prefix contiguous"
        );

        let len = range.len();
        if len != 0 {
            let dst = self.buf.as_mut_slice()[range.clone()]
                .as_mut_ptr()
                .cast::<u8>();
            unsafe { core::ptr::copy_nonoverlapping(src, dst, len) };
        }
        self.initialized = self.initialized.max(range.end);
    }

    /// Clone the logical window into independent raw storage without reading the uninitialized tail.
    ///
    /// # Safety
    ///
    /// `ptr` must satisfy the same storage/lifetime requirements as `from_raw_parts`, must not
    /// overlap this window, and must remain uniquely owned by the returned window.
    pub(crate) unsafe fn clone_to<'dst>(&self, ptr: *mut u8) -> Window<'dst> {
        let mut cloned = Window {
            buf: unsafe {
                WeakSliceMut::from_raw_parts_mut(ptr.cast::<MaybeUninit<u8>>(), self.buf.len())
            },
            initialized: 0,
        };
        if self.initialized != 0 {
            unsafe {
                core::ptr::copy_nonoverlapping(self.filled().as_ptr(), ptr, self.initialized)
            };
            cloned.initialized = self.initialized;
        }
        cloned
    }
}

#[cfg(test)]
mod tests {
    use super::Window;
    use core::mem::MaybeUninit;

    #[test]
    fn raw_window_tracks_only_the_written_prefix() {
        let mut storage = [MaybeUninit::<u8>::uninit(); 16];
        let mut window = unsafe { Window::from_raw_parts(storage.as_mut_ptr().cast(), 3) };
        assert!(window.filled().is_empty());

        let bytes = [1, 2, 3, 4, 5, 6];
        unsafe { window.copy_and_initialize(0..bytes.len(), bytes.as_ptr()) };
        assert_eq!(window.filled(), &bytes);

        window.clear();
        assert!(window.filled().is_empty());
        let replacement = [9, 8, 7];
        unsafe { window.copy_and_initialize(0..replacement.len(), replacement.as_ptr()) };
        assert_eq!(window.filled(), &replacement);
    }

    #[test]
    fn slide_discards_only_the_logical_prefix() {
        let mut storage = [MaybeUninit::<u8>::uninit(); 16];
        let mut window = unsafe { Window::from_raw_parts(storage.as_mut_ptr().cast(), 3) };
        let bytes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        unsafe { window.copy_and_initialize(0..bytes.len(), bytes.as_ptr()) };

        window.slide_down(4);
        assert_eq!(window.filled(), &[4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn recent_history_is_checked_and_window_limited() {
        let mut storage = [MaybeUninit::<u8>::uninit(); 16];
        let mut window = unsafe { Window::from_raw_parts(storage.as_mut_ptr().cast(), 3) };
        let bytes = core::array::from_fn::<_, 10, _>(|index| index as u8);
        unsafe { window.copy_and_initialize(0..bytes.len(), bytes.as_ptr()) };

        assert_eq!(window.recent_history(6, 4), &[2, 3, 4, 5]);
        assert_eq!(window.recent_history(3, 8), &[0, 1, 2]);
        assert_eq!(window.recent_history(0, 8), &[]);
    }
}
