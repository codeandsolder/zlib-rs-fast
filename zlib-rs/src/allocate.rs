#[cfg(unix)]
use core::ffi::c_int;
use core::{
    alloc::Layout,
    ffi::{c_uint, c_void},
    marker::PhantomData,
    mem,
    ptr::NonNull,
};

#[expect(
    non_camel_case_types,
    reason = "size_t is the standard C allocator ABI type"
)]
type size_t = usize;

const ALIGN: usize = 64;

/// One cache-line of typed allocation storage, shared by Deflate and Inflate.
/// The allocator enforces this alignment even for custom C callbacks.
#[repr(align(64))]
pub(crate) struct AlignedAllocationBlock {
    _bytes: [u8; ALIGN],
}
pub(crate) const ALLOCATION_BLOCK_BYTES: usize = mem::size_of::<AlignedAllocationBlock>();

const _: () = assert!(mem::align_of::<AlignedAllocationBlock>() == ALIGN);
const _: () = assert!(ALLOCATION_BLOCK_BYTES == ALIGN);

// posix_memalign requires that the alignment be a power of two and a multiple of sizeof(void*).
const _: () = assert!(ALIGN.is_power_of_two());
const _: () = assert!(ALIGN.is_multiple_of(mem::size_of::<*mut c_void>()));

/// Check the C callback's multiplication without wrapping to an undersized allocation.
fn allocation_size(items: c_uint, size: c_uint) -> Option<usize> {
    usize::try_from(items)
        .ok()?
        .checked_mul(usize::try_from(size).ok()?)
}

/// # Safety
///
/// This function is safe, but must have this type signature to be used elsewhere in the library
#[cfg(unix)]
unsafe extern "C" fn zalloc_c(opaque: *mut c_void, items: c_uint, size: c_uint) -> *mut c_void {
    extern "C" {
        fn posix_memalign(memptr: *mut *mut c_void, align: size_t, size: size_t) -> c_int;
    }

    let _ = opaque;
    let mut ptr = core::ptr::null_mut();
    let Some(size) = allocation_size(items, size) else {
        return core::ptr::null_mut();
    };
    if size == 0 {
        return ptr;
    }
    // SAFETY: ALIGN is a power of 2 and multiple of sizeof(void*), as required by posix_memalign.
    // In addition, since posix_memalign is allowed to return a unique but non-null pointer when
    // called with a size of zero, we returned above if the size was zero.
    match unsafe { posix_memalign(&raw mut ptr, ALIGN, size) } {
        0 => ptr,
        _ => core::ptr::null_mut(),
    }
}

/// # Safety
///
/// This function is safe, but must have this type signature to be used elsewhere in the library
#[cfg(not(unix))]
unsafe extern "C" fn zalloc_c(opaque: *mut c_void, items: c_uint, size: c_uint) -> *mut c_void {
    let _ = opaque;

    let Some(size) = allocation_size(items, size) else {
        return core::ptr::null_mut();
    };
    if size == 0 {
        return core::ptr::null_mut();
    }

    extern "C" {
        fn malloc(size: size_t) -> *mut c_void;
    }

    // SAFETY: malloc is allowed to return a unique but non-null pointer when given a size
    // of zero. To prevent potentially undefined behavior from such a pointer reaching Rust code
    // and being dereferenced, we handled the zero-size case separately above.
    unsafe { malloc(size) }
}

/// # Safety
///
/// The `ptr` must be allocated with the allocator that is used internally by `zcfree`
unsafe extern "C" fn zfree_c(opaque: *mut c_void, ptr: *mut c_void) {
    extern "C" {
        fn free(p: *mut c_void);
    }

    let _ = opaque;
    // SAFETY: The caller ensured that ptr was obtained from the same allocator. Also,
    // free properly handles the case where ptr == NULL.
    unsafe { free(ptr) }
}

/// # Safety
///
/// This function is safe to call.
#[cfg(feature = "rust-allocator")]
unsafe extern "C" fn zalloc_rust(_opaque: *mut c_void, count: c_uint, size: c_uint) -> *mut c_void {
    let Some(size) = allocation_size(count, size) else {
        return core::ptr::null_mut();
    };
    if size == 0 {
        return core::ptr::null_mut();
    }

    // internally, we want to align allocations to 64 bytes (in part for SIMD reasons)
    let Ok(layout) = Layout::from_size_align(size, ALIGN) else {
        return core::ptr::null_mut();
    };

    // SAFETY: alloc requires that the layout have a nonzero size, so we return null
    // above (and never reach this call) if the requested count * size is zero.
    let ptr = unsafe { alloc::alloc::alloc(layout) };

    ptr.cast()
}

/// # Safety
///
/// - `ptr` must be allocated with the rust `alloc::alloc` allocator
/// - `opaque` is a `&usize` that represents the size of the allocation
#[cfg(feature = "rust-allocator")]
unsafe extern "C" fn zfree_rust(opaque: *mut c_void, ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }

    // we can't really do much else. Deallocating with an invalid layout is UB.
    debug_assert!(!opaque.is_null());
    if opaque.is_null() {
        return;
    }

    // SAFETY: The caller ensured that *opaque is valid to dereference.
    let size = unsafe { *opaque.cast::<usize>() };

    // zalloc_rust bypasses the Rust allocator and just returns null when asked to
    // allocate something of zero size. So if a caller tries to
    // free something of zero size, we return here rather than trying to call the
    // Rust deallocator.
    if size == 0 {
        return;
    }

    let Ok(layout) = Layout::from_size_align(size, ALIGN) else {
        return;
    };

    // SAFETY: The caller ensured that ptr was allocated with the `alloc` allocator,
    // and the size check above ensures that we are not trying to use a zero-size layout
    // that would produce undefined behavior in the allocator.
    unsafe { alloc::alloc::dealloc(ptr.cast(), layout) };
}

#[cfg(test)]
const unsafe extern "C" fn zalloc_fail(_: *mut c_void, _: c_uint, _: c_uint) -> *mut c_void {
    core::ptr::null_mut()
}

#[cfg(test)]
const unsafe extern "C" fn zfree_fail(_: *mut c_void, _: *mut c_void) {
    // do nothing
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Allocator<'a> {
    pub zalloc: crate::c_api::alloc_func,
    pub zfree: crate::c_api::free_func,
    pub opaque: crate::c_api::voidpf,
    pub _marker: PhantomData<&'a ()>,
}

// These are copyable callback configurations, not globally shared mutable allocator objects.
// Keeping them const avoids claiming that arbitrary user-supplied opaque callbacks are Sync.
#[cfg(feature = "rust-allocator")]
pub const RUST: Allocator<'static> = Allocator {
    zalloc: zalloc_rust,
    zfree: zfree_rust,
    opaque: core::ptr::null_mut(),
    _marker: PhantomData,
};

#[cfg(feature = "c-allocator")]
pub const C: Allocator<'static> = Allocator {
    zalloc: zalloc_c,
    zfree: zfree_c,
    opaque: core::ptr::null_mut(),
    _marker: PhantomData,
};

#[cfg(test)]
const FAIL: Allocator<'static> = Allocator {
    zalloc: zalloc_fail,
    zfree: zfree_fail,
    opaque: core::ptr::null_mut(),
    _marker: PhantomData,
};

impl Allocator<'_> {
    fn allocate_layout(&self, layout: Layout) -> *mut c_void {
        assert!(layout.align() <= ALIGN);

        // Special case for the Rust `alloc` backed allocator
        #[cfg(feature = "rust-allocator")]
        if core::ptr::fn_addr_eq(self.zalloc, RUST.zalloc) {
            // The callback takes c_uint even on a 64-bit host. Narrowing here would expose a
            // small allocation through a pointer the caller treats as layout.size() bytes.
            let Ok(size) = c_uint::try_from(layout.size()) else {
                return core::ptr::null_mut();
            };
            // SAFETY: the requested allocation size is exactly representable in the C ABI.
            let ptr = unsafe { (RUST.zalloc)(self.opaque, size, 1) };

            debug_assert_eq!(ptr as usize % layout.align(), 0);

            return ptr;
        }

        // General case for c-style allocation

        // We cannot rely on the allocator giving properly aligned allocations and have to fix that ourselves.
        //
        // The general approach is to allocate a bit more than the layout needs, so that we can
        // give the application a properly aligned address and also store the real allocation
        // pointer in the allocation so that `free` can free the real allocation pointer.
        //
        //
        // Example: The layout represents `(u32, u32)`, with an alignment of 4 bytes and a
        // total size of 8 bytes.
        //
        // Assume that the allocator will give us address `0x07`. We need that to be a multiple
        // of the alignment, so that shifts the starting position to `0x08`. Then we also need
        // to store the pointer to the start of the allocation so that `free` can free that
        // pointer, bumping to `0x10`. The `0x10` pointer is then the pointer that the application
        // deals with. When free'ing, the original allocation pointer can be read from `0x10 - size_of::<*const c_void>()`.
        //
        // Of course there does need to be enough space in the allocation such that when we
        // shift the start forwards, the end is still within the allocation. Hence we allocate
        // `extra_space` bytes: enough for a full alignment plus a pointer.

        // we need at least
        //
        // - `align` extra space so that no matter what pointer we get from zalloc, we can shift the start of the
        //      allocation by at most `align - 1` so that `ptr as usize % align == 0
        // - `size_of::<*mut _>` extra space so that after aligning to `align`,
        //      there is `size_of::<*mut _>` space to store the pointer to the allocation.
        //      This pointer is then retrieved in `free`
        let extra_space = core::mem::size_of::<*mut c_void>() + layout.align();

        // Safety: we assume allocating works correctly in the safety assumptions on
        // `DeflateStream` and `InflateStream`.
        let Some(required_size) = layout.size().checked_add(extra_space) else {
            return core::ptr::null_mut();
        };
        let Ok(allocation_size) = c_uint::try_from(required_size) else {
            return core::ptr::null_mut();
        };
        // SAFETY: the C allocator callback contract promises a writable allocation of
        // exactly allocation_size bytes on success. The caller owns its lifetime.
        let ptr = unsafe { (self.zalloc)(self.opaque, allocation_size, 1) };

        if ptr.is_null() {
            return ptr;
        }

        // Calculate return pointer address with space enough to store original pointer
        let align_diff = (ptr as usize).next_multiple_of(layout.align()) - (ptr as usize);

        // Safety: offset is smaller than 64, and we allocated 64 extra bytes in the allocation
        let mut return_ptr = unsafe { ptr.cast::<u8>().add(align_diff) };

        // if there is not enough space to store a pointer we need to make more
        if align_diff < core::mem::size_of::<*mut c_void>() {
            // # Safety
            //
            // - `return_ptr` is well-aligned, therefore `return_ptr + align` is also well-aligned
            // - we reserve `size_of::<*mut _> + align` extra space in the allocation, so
            //      `ptr + align_diff + align` is still valid for (at least) `layout.size` bytes
            let offset = Ord::max(core::mem::size_of::<*mut c_void>(), layout.align());
            // SAFETY: extra_space reserves alignment and pointer-metadata headroom;
            // offset advances within the same allocation with the requested payload remaining.
            return_ptr = unsafe { return_ptr.add(offset) };
        }

        // Store the original pointer for free()
        //
        // Safety: `align >= size_of::<*mut _>`, so there is now space for a pointer before `return_ptr`
        // in the allocation
        unsafe {
            let original_ptr = return_ptr.sub(core::mem::size_of::<*mut c_void>());
            core::ptr::write_unaligned(original_ptr.cast::<*mut c_void>(), ptr);
        };

        // Return properly aligned pointer in allocation
        let ptr = return_ptr.cast::<c_void>();

        debug_assert_eq!(ptr as usize % layout.align(), 0);

        ptr
    }

    #[must_use]
    pub fn allocate_raw<T>(&self) -> Option<NonNull<T>> {
        NonNull::new(self.allocate_layout(Layout::new::<T>()).cast())
    }

    #[must_use]
    pub fn allocate_slice_raw<T>(&self, len: usize) -> Option<NonNull<T>> {
        NonNull::new(self.allocate_layout(Layout::array::<T>(len).ok()?).cast())
    }

    /// # Panics
    ///
    /// - when `len` is 0
    ///
    /// # Safety
    ///
    /// - `ptr` must be allocated with this allocator
    /// - `len` must be the number of `T`s that are in this allocation
    pub unsafe fn deallocate<T>(&self, ptr: *mut T, len: usize) {
        #[cfg(not(feature = "rust-allocator"))]
        let _ = len;
        if !ptr.is_null() {
            // Special case for the Rust `alloc` backed allocator
            #[cfg(feature = "rust-allocator")]
            if core::ptr::fn_addr_eq(self.zfree, RUST.zfree) {
                assert_ne!(len, 0, "invalid size for {ptr:?}");
                let mut size = core::mem::size_of::<T>() * len;
                // SAFETY: The caller ensured that ptr was allocated with this allocator, and
                // we initialized size above.
                return unsafe { (RUST.zfree)((&raw mut size).cast(), ptr.cast()) };
            }

            // General case for c-style allocation
            // SAFETY: allocate_layout allocates extra space at the start so that *ptr is preceded
            // by a pointer holding the pointer to the actual allocation. Therefore, it is safe to
            // subtract size_of::<*const c_void> from pointer, dereference the resulting address,
            // and use that as the argument to the low-level free function.
            unsafe {
                let original_ptr = ptr.cast::<u8>().sub(core::mem::size_of::<*const c_void>());
                let free_ptr = core::ptr::read_unaligned(original_ptr.cast::<*mut c_void>());

                (self.zfree)(self.opaque, free_ptr);
            }
        }
    }
}

/// Reusable test-only C allocator simulating every raw pointer residue.
/// The per-stream context avoids global callbacks and cross-test contention.
#[cfg(test)]
use std::{boxed::Box, vec, vec::Vec};

#[cfg(test)]
pub(super) struct MisalignedAlloc {
    pub(super) remainder: usize,
    pub(super) allocations: usize,
    pub(super) frees: usize,
    pub(super) buffers: Vec<(*mut core::ffi::c_void, Box<[u8]>)>,
}

#[cfg(test)]
pub(super) unsafe extern "C" fn misaligned_alloc_callback(
    opaque: *mut core::ffi::c_void,
    items: crate::c_api::uInt,
    size: crate::c_api::uInt,
) -> *mut core::ffi::c_void {
    // SAFETY: init and copy receive a live pointer to our stack-local allocator context.
    let Some(ctx) = (unsafe { opaque.cast::<MisalignedAlloc>().as_mut() }) else {
        return core::ptr::null_mut();
    };
    let Some(length) = usize::try_from(items)
        .ok()
        .and_then(|count| usize::try_from(size).ok()?.checked_mul(count))
        .and_then(|size| size.checked_add(ALLOCATION_BLOCK_BYTES))
    else {
        return core::ptr::null_mut();
    };
    let mut data = vec![0u8; length].into_boxed_slice();
    let base = data.as_mut_ptr() as usize;
    let delta = (ctx.remainder + ALLOCATION_BLOCK_BYTES - base % ALLOCATION_BLOCK_BYTES)
        % ALLOCATION_BLOCK_BYTES;
    // SAFETY: the allocation reserves one full alignment unit beyond the requested
    // size, so the shifted pointer still has all requested bytes available.
    let ptr = unsafe { data.as_mut_ptr().add(delta) }.cast();
    ctx.buffers.push((ptr, data));
    ctx.allocations += 1;
    ptr
}

#[cfg(test)]
pub(super) unsafe extern "C" fn misaligned_free_callback(
    opaque: *mut core::ffi::c_void,
    ptr: *mut core::ffi::c_void,
) {
    // SAFETY: zfree is invoked before the stack-local context is destroyed, and
    // receives the exact pointer previously returned by the matching allocator.
    if let Some(ctx) = unsafe { opaque.cast::<MisalignedAlloc>().as_mut() } {
        if let Some(index) = ctx.buffers.iter().position(|(address, _)| *address == ptr) {
            ctx.buffers.swap_remove(index);
            ctx.frees += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use core::sync::atomic::{AtomicPtr, Ordering};
    use std::ptr;
    use std::sync::Mutex;

    use super::*;

    #[test]
    fn allocation_size_rejects_unrepresentable_products() {
        assert_eq!(allocation_size(0, c_uint::MAX), Some(0));
        assert_eq!(allocation_size(17, 19), Some(323));
        let max_product = u64::from(c_uint::MAX) * u64::from(c_uint::MAX);
        assert_eq!(
            allocation_size(c_uint::MAX, c_uint::MAX),
            usize::try_from(max_product).ok()
        );
    }

    #[cfg(all(feature = "rust-allocator", target_pointer_width = "64"))]
    #[test]
    fn rust_allocator_rejects_oversized_requests_before_narrowing() {
        let beyond_c_uint = u64::from(c_uint::MAX) + 16;
        assert!(usize::try_from(beyond_c_uint)
            .is_ok_and(|len| RUST.allocate_slice_raw::<u8>(len).is_none()));

        // A representable usize can still exceed Layout's ISIZE_MAX limit. The callback must
        // reject it rather than panic across an extern "C" boundary.
        // SAFETY: the callback ignores its opaque pointer.
        assert!(unsafe { zalloc_rust(ptr::null_mut(), c_uint::MAX, c_uint::MAX) }.is_null());
    }

    static PTR: AtomicPtr<c_void> = AtomicPtr::new(core::ptr::null_mut());
    static MUTEX: Mutex<()> = Mutex::new(());

    unsafe extern "C" fn unaligned_alloc(
        _opaque: *mut c_void,
        _items: c_uint,
        _size: c_uint,
    ) -> *mut c_void {
        PTR.load(Ordering::Relaxed)
    }

    unsafe extern "C" fn unaligned_free(_opaque: *mut c_void, ptr: *mut c_void) {
        let expected = PTR.load(Ordering::Relaxed);
        assert_eq!(expected, ptr);
    }

    fn unaligned_allocator_help<T>() -> Result<(), &'static str> {
        let mut buf = [0u8; 1024];

        // Only one test may install the global mock allocator callback at a time.
        let _guard = MUTEX.lock().map_err(|_| "allocator test mutex poisoned")?;

        for i in 0..64 {
            let ptr = buf[i..].as_mut_ptr().cast();
            PTR.store(ptr, Ordering::Relaxed);

            let allocator = Allocator {
                zalloc: unaligned_alloc,
                zfree: unaligned_free,
                opaque: core::ptr::null_mut(),
                _marker: PhantomData,
            };

            let ptr = allocator
                .allocate_raw::<T>()
                .ok_or("mock allocator failed to return an aligned object")?
                .as_ptr();
            assert_eq!(ptr as usize % core::mem::align_of::<T>(), 0);
            // SAFETY: the allocation was obtained from allocator for exactly one T.
            unsafe { allocator.deallocate(ptr, 1) };

            let ptr = allocator
                .allocate_slice_raw::<T>(10)
                .ok_or("mock allocator failed to return an aligned slice")?
                .as_ptr();
            assert_eq!(ptr as usize % core::mem::align_of::<T>(), 0);
            // SAFETY: the allocation was obtained from allocator for ten Ts.
            unsafe { allocator.deallocate(ptr, 10) };
        }
        Ok(())
    }

    #[test]
    fn unaligned_allocator_0() -> Result<(), &'static str> {
        unaligned_allocator_help::<()>()
    }

    #[test]
    fn unaligned_allocator_1() -> Result<(), &'static str> {
        unaligned_allocator_help::<u8>()
    }

    #[test]
    fn unaligned_allocator_2() -> Result<(), &'static str> {
        unaligned_allocator_help::<u16>()
    }
    #[test]
    fn unaligned_allocator_4() -> Result<(), &'static str> {
        unaligned_allocator_help::<u32>()
    }
    #[test]
    fn unaligned_allocator_8() -> Result<(), &'static str> {
        unaligned_allocator_help::<u64>()
    }
    #[test]
    fn unaligned_allocator_16() -> Result<(), &'static str> {
        unaligned_allocator_help::<u128>()
    }

    #[test]
    fn unaligned_allocator_32() -> Result<(), &'static str> {
        #[repr(C, align(32))]
        struct Align32(u8);

        unaligned_allocator_help::<Align32>()
    }

    #[test]
    fn unaligned_allocator_64() -> Result<(), &'static str> {
        #[repr(C, align(64))]
        struct Align64(u8);

        unaligned_allocator_help::<Align64>()
    }

    fn test_allocate_raw_help(allocator: Allocator<'_>) -> Result<(), &'static str> {
        #[repr(C, align(64))]
        struct Align64(u8);

        let ptr = allocator
            .allocate_raw::<Align64>()
            .ok_or("failed to allocate 64-byte aligned object")?;
        // SAFETY: ptr was allocated by allocator for one Align64 and is owned by this test.
        unsafe { allocator.deallocate(ptr.as_ptr(), 1) };
        Ok(())
    }

    #[test]
    fn test_allocate_raw() -> Result<(), &'static str> {
        #[cfg(feature = "rust-allocator")]
        test_allocate_raw_help(RUST)?;

        #[cfg(feature = "c-allocator")]
        test_allocate_raw_help(C)?;

        assert!(FAIL.allocate_raw::<u128>().is_none());
        Ok(())
    }

    #[test]
    fn test_deallocate_null() {
        // SAFETY: all three allocator callback implementations accept a null pointer
        // without consulting their opaque argument or releasing any allocation.
        unsafe {
            #[cfg(feature = "rust-allocator")]
            (RUST.zfree)(core::ptr::null_mut(), core::ptr::null_mut());

            #[cfg(feature = "c-allocator")]
            (C.zfree)(core::ptr::null_mut(), core::ptr::null_mut());

            (FAIL.zfree)(core::ptr::null_mut(), core::ptr::null_mut());
        }
    }

    #[test]
    fn test_allocate_zero_size() {
        // Verify that zero-size allocation requests return a null pointer.
        // SAFETY: zero-sized requests are handled explicitly by these C callbacks;
        // no caller memory or opaque state is read or written.
        unsafe {
            assert!(zalloc_c(ptr::null_mut(), 1, 0).is_null());
            assert!(zalloc_c(ptr::null_mut(), 0, 1).is_null());
            #[cfg(feature = "rust-allocator")]
            assert!(zalloc_rust(ptr::null_mut(), 1, 0).is_null());
            #[cfg(feature = "rust-allocator")]
            assert!(zalloc_rust(ptr::null_mut(), 0, 1).is_null());
        }
    }

    #[test]
    fn allocate_more_than_c_uint_max() {
        unsafe extern "C" fn small_alloc(
            _opaque: *mut c_void,
            _items: c_uint,
            _size: c_uint,
        ) -> *mut c_void {
            unreachable!("we should return NULL before this");
        }

        unsafe extern "C" fn small_free(_opaque: *mut c_void, _ptr: *mut c_void) {}

        let mut storage = [0u8; 64];
        let allocator = Allocator {
            zalloc: small_alloc,
            zfree: small_free,
            opaque: storage.as_mut_ptr().cast(),
            _marker: PhantomData,
        };

        let Some(len) = (c_uint::MAX as usize).checked_add(1) else {
            return;
        };
        assert!(allocator.allocate_slice_raw::<u8>(len).is_none());
    }
}
