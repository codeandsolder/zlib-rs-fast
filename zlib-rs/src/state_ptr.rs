use core::ops::{Deref, DerefMut};

/// Nullable raw state pointer used at the zlib ABI boundary.
///
/// `z_stream.state` is a C pointer: it is null before initialization/after `*End`, and copyable
/// as a pointer value. Keeping that representation here avoids pretending the ABI shell owns a
/// long-lived Rust `&mut` reference. Rust references are materialized only for the duration of a
/// borrow through `Deref`/`DerefMut`.
#[repr(transparent)]
pub struct StatePtr<T>(*mut T);

impl<T> StatePtr<T> {
    /// # Safety
    ///
    /// A non-null pointer must point to a live, correctly aligned T whenever the handle
    /// is dereferenced. The caller must also uphold shared/exclusive reference aliasing
    /// rules for each Deref/DerefMut access. Null may be stored but not dereferenced.
    pub(crate) const unsafe fn from_raw(ptr: *mut T) -> Self {
        Self(ptr)
    }

    #[inline]
    pub(crate) const fn as_ptr(&self) -> *mut T {
        self.0
    }
}

impl<T> Deref for StatePtr<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        // SAFETY: construction requires every non-null pointer to be valid while dereferenced.
        unsafe { self.0.as_ref() }.unwrap_or_else(|| unreachable!("zlib stream state is null"))
    }
}

impl<T> DerefMut for StatePtr<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: mutable access requires `&mut self`, so safe callers cannot materialize two
        // mutable references through the same handle at once.
        unsafe { self.0.as_mut() }.unwrap_or_else(|| unreachable!("zlib stream state is null"))
    }
}

const _: () = {
    assert!(core::mem::size_of::<StatePtr<u8>>() == core::mem::size_of::<*mut u8>());
    assert!(core::mem::align_of::<StatePtr<u8>>() == core::mem::align_of::<*mut u8>());
};

#[cfg(test)]
mod tests {
    use super::StatePtr;

    #[test]
    fn null_is_a_valid_abi_state() {
        // SAFETY: null is a representable zlib ABI state and is never dereferenced here.
        let state = unsafe { StatePtr::<u8>::from_raw(core::ptr::null_mut()) };
        assert!(state.as_ptr().is_null());
    }
}
