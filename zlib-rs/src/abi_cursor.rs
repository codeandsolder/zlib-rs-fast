use crate::c_api::{uInt, z_size, Bytef};

/// Advance a live caller buffer cursor and both public counters together.
/// Length is bounded by the caller-provided available byte count. Zlib totals
/// use modular arithmetic in the native unsigned C ABI width.
#[inline]
pub(crate) fn advance_abi_bytes(
    pointer: &mut *mut Bytef,
    available: &mut uInt,
    total: &mut z_size,
    len: usize,
) {
    debug_assert!(len <= *available as usize);
    let Ok(consumed) = uInt::try_from(len) else {
        unreachable!("ABI byte progress cannot exceed caller availability");
    };
    *pointer = pointer.wrapping_add(len);
    *available -= consumed;
    *total = total.wrapping_add(z_size::from(consumed));
}
