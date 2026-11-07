// taken from https://docs.rs/tokio/latest/src/tokio/io/read_buf.rs.html#23-27
// based on https://rust-lang.github.io/rfcs/2930-read-buf.html

use core::mem::MaybeUninit;

use crate::weak_slice::WeakSliceMut;

pub struct SymBuf<'a> {
    buf: WeakSliceMut<'a, MaybeUninit<u8>>,
    filled: usize,
}

impl SymBuf<'_> {
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (u16, u8)> + '_ {
        // SAFETY: each append initializes all three bytes before advancing filled;
        // clear only decreases filled, so the entire logical prefix is initialized.
        let initialized = unsafe { self.buf.as_slice()[..self.filled].assume_init_ref() };
        initialized
            .as_chunks::<3>()
            .0
            .iter()
            .map(|&[dist_low, dist_high, lc]| (u16::from_le_bytes([dist_low, dist_high]), lc))
    }

    #[inline]
    pub const fn should_flush_block(&self) -> bool {
        self.filled == self.buf.len() - 3
    }

    /// Returns true if there are no initialized tokens in this symbol buffer.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.filled == 0
    }

    /// Clears the logical token buffer without touching its initialized backing storage.
    #[inline]
    pub const fn clear(&mut self) {
        self.filled = 0;
    }

    #[inline(always)]
    pub fn push_lit(&mut self, byte: u8) {
        self.buf.as_mut_slice()[self.filled..][..3].write_copy_of_slice(&[0, 0, byte]);
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
            core::ptr::write_unaligned(buf.as_mut_ptr().cast::<[u8; 2]>(), dist.to_le_bytes());
        }
        buf[2].write(len);

        self.filled += 3;
    }

    /// Bind a symbol buffer over raw writable storage without initializing unused tokens.
    ///
    /// # Safety
    ///
    /// `ptr` must be non-null, aligned, and uniquely writable for `byte_len` bytes
    /// throughout the returned lifetime. `byte_len` must be a multiple of three.
    /// The bytes may initially be uninitialized.
    pub(crate) const unsafe fn from_raw_parts<'a>(ptr: *mut u8, byte_len: usize) -> SymBuf<'a> {
        SymBuf {
            // SAFETY: the caller guarantees a uniquely writable byte span of byte_len;
            // MaybeUninit does not require its contents to be initialized.
            buf: unsafe { WeakSliceMut::from_raw_parts_mut(ptr.cast(), byte_len) },
            filled: 0,
        }
    }

    /// Clone only initialized tokens into independent raw storage.
    ///
    /// # Safety
    ///
    /// `ptr` must be valid for writes of `self.buf.len()` bytes, must not overlap the source, and
    /// must remain alive and uniquely owned for the returned destination lifetime.
    pub(crate) const unsafe fn clone_to<'dst>(&self, ptr: *mut u8) -> SymBuf<'dst> {
        // SAFETY: filled is an initialized prefix; the caller supplies disjoint writable
        // destination storage with at least the source allocation's capacity.
        unsafe { core::ptr::copy_nonoverlapping(self.buf.as_ptr(), ptr.cast(), self.filled) };
        SymBuf {
            // SAFETY: clone_to's contract ensures ptr owns an independent byte span
            // valid for this destination lifetime, including the still-uninitialized tail.
            buf: unsafe { WeakSliceMut::from_raw_parts_mut(ptr.cast(), self.buf.len()) },
            filled: self.filled,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::MaybeUninit;

    #[test]
    fn clear_reuses_storage_without_leaking_stale_distance_into_literals() {
        const LIT_BUFSIZE: usize = 4;
        let mut storage = [MaybeUninit::<u8>::uninit(); LIT_BUFSIZE * 3];
        // SAFETY: the stack allocation is uniquely writable for its full byte extent.
        let mut sym_buf =
            unsafe { SymBuf::from_raw_parts(storage.as_mut_ptr().cast::<u8>(), LIT_BUFSIZE * 3) };

        sym_buf.push_dist(0x1234, 7);
        sym_buf.clear();
        sym_buf.push_lit(0xa5);

        assert_eq!(sym_buf.iter().collect::<Vec<_>>(), vec![(0, 0xa5)]);
        // SAFETY: push_lit initialized the full 3-byte record.
        let bytes = unsafe { sym_buf.buf.as_slice()[..3].assume_init_ref() };
        assert_eq!(bytes, &[0, 0, 0xa5]);
    }

    #[test]
    fn clone_copies_only_complete_tokens_not_unused_tail() {
        let mut source_storage = [MaybeUninit::<u8>::uninit(); 15];
        let mut destination_storage = [MaybeUninit::new(0x5au8); 15];

        // SAFETY: these arrays are disjoint, uniquely writable and live for the test scope.
        let mut source = unsafe { SymBuf::from_raw_parts(source_storage.as_mut_ptr().cast(), 15) };
        source.push_dist(0x1234, 7);
        source.push_lit(0xa5);

        // SAFETY: destination_storage is separate and large enough for the source capacity.
        let copied = unsafe { source.clone_to(destination_storage.as_mut_ptr().cast()) };
        assert_eq!(
            copied.iter().collect::<Vec<_>>(),
            vec![(0x1234, 7), (0, 0xa5)]
        );

        // SAFETY: all destination bytes were initialized to 0x5a before cloning,
        // and the clone overwrote only the first six with initialized token data.
        let backing = unsafe { copied.buf.as_slice().assume_init_ref() };
        assert_eq!(&backing[..6], &[0x34, 0x12, 7, 0, 0, 0xa5]);
        assert!(backing[6..].iter().all(|&byte| byte == 0x5a));
    }

    #[test]
    fn raw_constructor_leaves_unused_storage_untouched() {
        const LIT_BUFSIZE: usize = 4;
        let mut storage = [MaybeUninit::new(0x5au8); LIT_BUFSIZE * 3];

        // SAFETY: the stack allocation is uniquely writable for its full byte extent.
        let mut sym_buf =
            unsafe { SymBuf::from_raw_parts(storage.as_mut_ptr().cast::<u8>(), LIT_BUFSIZE * 3) };

        // SAFETY: the test intentionally initialized every backing byte with the sentinel.
        let backing = unsafe { sym_buf.buf.as_slice().assume_init_ref() };
        assert!(backing.iter().all(|&byte| byte == 0x5a));

        // Each literal initializes its entire record, including the zero distance bytes.
        sym_buf.push_lit(0xa5);
        assert_eq!(sym_buf.iter().collect::<Vec<_>>(), vec![(0, 0xa5)]);
        // SAFETY: the test started with initialized sentinel bytes and every token
        // append initializes its entire record; the full backing storage is initialized.
        let backing = unsafe { sym_buf.buf.as_slice().assume_init_ref() };
        assert_eq!(&backing[..3], &[0, 0, 0xa5]);
        assert!(backing[3..].iter().all(|&byte| byte == 0x5a));
    }
}
