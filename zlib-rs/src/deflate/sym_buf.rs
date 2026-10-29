// taken from https://docs.rs/tokio/latest/src/tokio/io/read_buf.rs.html#23-27
// based on https://rust-lang.github.io/rfcs/2930-read-buf.html

use crate::weak_slice::WeakSliceMut;

pub(crate) struct SymBuf<'a> {
    buf: WeakSliceMut<'a, u8>,
    filled: usize,
}

impl<'a> SymBuf<'a> {
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (u16, u8)> + '_ {
        self.buf.as_slice()[..self.filled]
            .chunks_exact(3)
            .map(|chunk| match *chunk {
                [dist_low, dist_high, lc] => (u16::from_le_bytes([dist_low, dist_high]), lc),
                _ => unreachable!("chunks are exactly 3 elements wide"),
            })
    }

    #[inline]
    pub fn should_flush_block(&self) -> bool {
        self.filled == self.buf.len() - 3
    }

    /// Returns true if there are no bytes in this ReadBuf
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.filled == 0
    }

    /// Clears the buffer, resetting the filled region to empty.
    ///
    /// The number of initialized bytes is not changed, and the contents of the buffer are not modified.
    #[inline]
    pub fn clear(&mut self) {
        self.buf.as_mut_slice().fill(0);
        self.filled = 0;
    }

    #[inline(always)]
    pub fn push_lit(&mut self, byte: u8) {
        // NOTE: we rely on the buffer being zeroed here!
        self.buf.as_mut_slice()[self.filled + 2] = byte;

        self.filled += 3;
    }

    #[inline(always)]
    pub fn push_dist(&mut self, dist: u16, len: u8) {
        let buf = &mut self.buf.as_mut_slice()[self.filled..][..3];

        // Write both dist bytes as a single 2-byte store. This avoids the
        // `movb %ch, [mem]` instruction pattern (store from high-byte register
        // alias) that LLVM otherwise emits when dist arrives as a wide register.
        // That pattern triggers the Intel Raptor Lake CPU errata, causing silent
        // 2-byte stores that corrupt the adjacent `len` byte.
        // SAFETY: buf has at least 3 bytes (guaranteed above), so the unaligned
        // 2-byte write at offset 0 stays within bounds.
        unsafe {
            core::ptr::write_unaligned(buf.as_mut_ptr().cast::<[u8; 2]>(), dist.to_le_bytes())
        };
        buf[2] = len;

        self.filled += 3;
    }

    /// Construct a symbol buffer over raw writable storage.
    ///
    /// # Safety
    ///
    /// `ptr` must be valid for writes of `lit_bufsize * 3` bytes for the lifetime of the returned
    /// buffer and must not alias another live reference. The storage may be uninitialized: this
    /// constructor zeroes it before creating the typed `u8` view.
    pub(crate) unsafe fn from_raw_parts(ptr: *mut u8, lit_bufsize: usize) -> Self {
        let len = lit_bufsize
            .checked_mul(3)
            .expect("symbol buffer size overflow");

        // SAFETY: the caller guarantees `ptr` is writable for `len` bytes. Raw byte writes do not
        // require the destination to contain initialized `u8` values beforehand.
        unsafe { ptr.write_bytes(0, len) };

        Self {
            // SAFETY: the write above initialized all `len` bytes before the typed view is created.
            buf: unsafe { WeakSliceMut::from_raw_parts_mut(ptr, len) },
            filled: 0,
        }
    }

    pub(crate) unsafe fn clone_to(&self, ptr: *mut u8) -> Self {
        unsafe { ptr.copy_from_nonoverlapping(self.buf.as_ptr(), self.buf.len()) };
        Self {
            buf: unsafe { WeakSliceMut::from_raw_parts_mut(ptr, self.buf.len()) },
            filled: self.filled,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::MaybeUninit;

    #[test]
    fn raw_constructor_initializes_symbol_storage() {
        const LIT_BUFSIZE: usize = 4;
        let mut storage = [MaybeUninit::<u8>::uninit(); LIT_BUFSIZE * 3];

        let mut sym_buf =
            unsafe { SymBuf::from_raw_parts(storage.as_mut_ptr().cast::<u8>(), LIT_BUFSIZE) };

        assert_eq!(sym_buf.buf.as_slice(), &[0; LIT_BUFSIZE * 3]);

        // A literal writes only the third byte of its 3-byte record. The distance bytes must stay
        // zero because that is how the token iterator distinguishes literals from matches.
        sym_buf.push_lit(0xa5);
        assert_eq!(&sym_buf.buf.as_slice()[..3], &[0, 0, 0xa5]);
    }
}
