use crate::c_api::{uInt, z_size, Bytef};

/// Advance a live caller buffer cursor and both public counters together.
/// Length is bounded by the caller-provided available byte count. Zlib totals
/// use modular arithmetic in the native unsigned C ABI width.
#[inline]
pub fn advance_abi_bytes(
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

/// Tracks whole-buffer Rust slices offered to the C ABI in uInt-sized chunks.
/// Pending bytes exclude the currently issued chunk, so the number processed
/// remains full-width even when public C totals wrap.
pub struct SliceBudget {
    total: usize,
    pending: usize,
}

impl SliceBudget {
    pub const fn new(total: usize) -> Self {
        Self {
            total,
            pending: total,
        }
    }

    pub fn refill(&mut self, available: &mut uInt) {
        if *available == 0 {
            let issued = uInt::try_from(self.pending).unwrap_or(uInt::MAX);
            *available = issued;
            self.pending -= issued as usize;
        }
    }

    pub const fn pending(&self) -> usize {
        self.pending
    }

    pub const fn processed(&self, available: uInt) -> usize {
        assert!(
            self.pending <= self.total,
            "unissued bytes exceed caller storage"
        );
        let issued = self.total - self.pending;
        assert!(
            available as usize <= issued,
            "ABI cursor exceeds issued storage"
        );
        issued - available as usize
    }
}

#[cfg(test)]
mod slice_budget_tests {
    use super::*;

    #[test]
    fn refills_only_drained_chunks_and_counts_partial_progress() {
        let mut budget = SliceBudget::new(19);
        let mut available = 0;
        budget.refill(&mut available);
        assert_eq!(available, 19);
        available -= 7;
        budget.refill(&mut available);
        assert_eq!(available, 12);
        assert_eq!(budget.processed(available), 7);
        available = 0;
        budget.refill(&mut available);
        assert_eq!(budget.processed(available), 19);
    }

    #[test]
    fn empty_budget_stays_empty() {
        let mut budget = SliceBudget::new(0);
        let mut available = 0;
        budget.refill(&mut available);
        assert_eq!(available, 0);
        assert_eq!(budget.processed(available), 0);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn splits_large_slices_at_u_int_max() {
        let total = uInt::MAX as usize + 4097;
        let mut budget = SliceBudget::new(total);
        let mut available = 0;
        budget.refill(&mut available);
        assert_eq!(available, uInt::MAX);
        assert_eq!(budget.pending(), 4097);
        available = 0;
        budget.refill(&mut available);
        assert_eq!(available, 4097);
        assert_eq!(budget.processed(available), uInt::MAX as usize);
        available -= 11;
        assert_eq!(budget.processed(available), uInt::MAX as usize + 11);
    }
}
