use core::marker::PhantomData;

/// a mutable "slice" (bundle of pointer and length). The main goal of this type is passing MIRI
/// with stacked borrows. In particular, storing a standard slice in data structures violates the
/// stacked borrows rule when that slice is deallocated. By only materializing the slice when
/// needed for data access, hence bounding the lifetime more tightly, this restriction is circumvented.
#[derive(Debug)]
pub struct WeakSliceMut<'a, T> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<&'a mut [T]>,
}

impl<T> WeakSliceMut<'_, T> {
    /// Retain a mutable slice's exclusive lifetime without permanently
    /// materializing a Rust reference to its backing allocation.
    pub(crate) const fn from_mut_slice(slice: &mut [T]) -> WeakSliceMut<'_, T> {
        WeakSliceMut {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
            _marker: PhantomData,
        }
    }

    /// # Safety
    ///
    /// The pointer must be non-null, properly aligned, and point into one live allocation
    /// containing at least len initialized elements of T (even if len is zero, the pointer
    /// must still be non-null and aligned). The allocation must remain alive for the handle
    /// lifetime. The caller must prevent conflicting accesses whenever a shared or mutable
    /// reference is materialized. Unlike a permanent Rust slice, this handle creates
    /// references only during each borrow; it does not relax their aliasing requirements.
    pub(crate) const unsafe fn from_raw_parts_mut(ptr: *mut T, len: usize) -> Self {
        Self {
            ptr,
            len,
            _marker: PhantomData,
        }
    }

    pub(crate) const fn into_raw_parts(self) -> (*mut T, usize) {
        (self.ptr, self.len)
    }

    pub(crate) const fn as_slice(&self) -> &[T] {
        // Keep the materialized Rust reference tied to this borrow. Returning the backing
        // allocation lifetime here would allow a shared slice to outlive `&self` and overlap a
        // later mutable materialization from the same raw pointer.
        // SAFETY: from_raw_parts_mut requires valid aligned storage covering len elements;
        // the shared view is borrowed from self, excluding a concurrent mutable view.
        unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub(crate) const fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: the unsafe constructor guarantees writable aligned storage covering len
        // elements; exclusive access is enforced by the borrow of self.
        unsafe { core::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub(crate) const fn as_ptr(&self) -> *const T {
        self.ptr
    }

    #[expect(
        clippy::needless_pass_by_ref_mut,
        reason = "mutable pointer access requires exclusive borrowing of the backing slice handle"
    )]
    pub(crate) const fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    pub(crate) const fn len(&self) -> usize {
        self.len
    }

    pub(crate) const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) const fn empty() -> Self {
        let buf = &mut [];
        Self {
            ptr: buf.as_mut_ptr(),
            len: buf.len(),
            _marker: PhantomData,
        }
    }
}

#[derive(Debug)]
pub struct WeakArrayMut<'a, T, const N: usize> {
    ptr: *mut [T; N],
    _marker: PhantomData<&'a mut [T; N]>,
}

impl<T, const N: usize> WeakArrayMut<'_, T, N> {
    /// # Safety
    ///
    /// The pointer must be non-null, properly aligned, and valid for a live writable
    /// array of N initialized elements for the handle lifetime. The caller must prevent
    /// overlapping conflicting accesses whenever a reference is materialized.
    pub(crate) const unsafe fn from_ptr(ptr: *mut [T; N]) -> Self {
        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    pub(crate) const fn as_slice(&self) -> &[T] {
        // As above, the reference must not outlive the borrow used to materialize it.
        // SAFETY: from_ptr requires a live, aligned [T; N] allocation, and this shared
        // materialization cannot outlive the borrow of the handle.
        unsafe { core::slice::from_raw_parts(self.ptr.cast(), N) }
    }

    pub(crate) const fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: from_ptr requires valid writable [T; N] storage, and the mutable borrow
        // of the handle excludes an overlapping materialization from this handle.
        unsafe { core::slice::from_raw_parts_mut(self.ptr.cast(), N) }
    }

    pub(crate) const fn as_ptr(&self) -> *const [T; N] {
        self.ptr
    }
}
