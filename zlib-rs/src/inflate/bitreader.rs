use core::ptr::NonNull;

use crate::ReturnCode;

#[derive(Debug, Clone, Copy)]
pub(crate) struct BitReader {
    ptr: *const u8,
    end: *const u8,
    bit_buffer: u64,
    bits_used: u8,
}

impl BitReader {
    /// Construct a reader with no caller input attached.
    pub fn new() -> Self {
        let ptr = Self::detached_ptr();
        Self {
            ptr,
            end: ptr,
            bit_buffer: 0,
            bits_used: 0,
        }
    }

    #[inline(always)]
    fn detached_ptr() -> *const u8 {
        NonNull::<u8>::dangling().as_ptr().cast_const()
    }

    /// Attach caller input while preserving the persistent bit buffer.
    ///
    /// # Safety
    ///
    /// If `len` is non-zero, `ptr` and `len` must satisfy the requirements of
    /// [`core::slice::from_raw_parts`] for the duration of all reads performed before
    /// [`Self::detach_input`]. A zero-length input may use a null pointer, as zlib permits.
    #[inline(always)]
    pub unsafe fn attach_input(&mut self, ptr: *const u8, len: usize) {
        if len == 0 {
            self.detach_input();
        } else {
            self.ptr = ptr;
            self.end = ptr.wrapping_add(len);
        }
    }

    /// Drop the caller-input cursor while preserving buffered bits across inflate calls.
    #[inline(always)]
    pub fn detach_input(&mut self) {
        let ptr = Self::detached_ptr();
        self.ptr = ptr;
        self.end = ptr;
    }

    #[inline(always)]
    pub fn advance(&mut self, bytes: usize) {
        self.ptr = Ord::min(self.ptr.wrapping_add(bytes), self.end);
    }

    #[inline(always)]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr as *mut u8
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[u8] {
        let len = self.bytes_remaining();
        // SAFETY: condition of constructing this struct.
        unsafe { core::slice::from_raw_parts(self.ptr, len) }
    }

    #[inline(always)]
    pub fn bits_in_buffer(&self) -> u8 {
        self.bits_used
    }

    #[inline(always)]
    pub fn hold(&self) -> u64 {
        self.bit_buffer
    }

    #[inline(always)]
    pub fn bytes_remaining(&self) -> usize {
        self.end as usize - self.ptr as usize
    }

    #[inline(always)]
    pub fn bytes_remaining_including_buffer(&self) -> usize {
        (self.end as usize - self.ptr as usize) + (self.bits_used as usize >> 3)
    }

    #[inline(always)]
    pub fn need_bits(&mut self, n: usize) -> Result<(), ReturnCode> {
        while (self.bits_used as usize) < n {
            self.pull_byte()?;
        }

        Ok(())
    }

    /// Remove zero to seven bits as needed to go to a byte boundary
    #[inline(always)]
    pub fn next_byte_boundary(&mut self) {
        self.bit_buffer >>= self.bits_used & 0b0111;
        self.bits_used -= self.bits_used & 0b0111;
    }

    #[inline(always)]
    pub fn pull_byte(&mut self) -> Result<u8, ReturnCode> {
        // SAFETY: bounds checking.
        if self.ptr == self.end {
            return Err(ReturnCode::Ok);
        }
        let byte = unsafe { *self.ptr };
        self.ptr = unsafe { self.ptr.add(1) };

        self.bit_buffer |= (byte as u64) << self.bits_used;
        self.bits_used += 8;

        Ok(byte)
    }

    #[inline(always)]
    /// Copy enough bytes from the BitReader's underlying slice to fill the internal
    /// bit buffer with 7 bytes of data.
    ///
    /// Safety:
    ///
    /// `self.ptr` must point to at least 8 readable bytes, as indicated by `bytes_remaining()`
    pub unsafe fn refill(&mut self) {
        debug_assert!(self.bytes_remaining() >= 8);

        // SAFETY: caller ensures we have 8 bytes to read for a u64.
        let read = unsafe { core::ptr::read_unaligned(self.ptr.cast::<u64>()) }.to_le();
        self.bit_buffer |= read << self.bits_used;
        // this xor was previously a subtraction but was changed for performance reasons.
        // for bits_used between 0 and 63 (inclusive), it will always have the same behavior.
        let increment = (63 ^ self.bits_used) >> 3;
        self.ptr = self.ptr.wrapping_add(increment as usize);
        self.bits_used |= 56;
    }

    #[inline(always)]
    pub fn bits(&mut self, n: usize) -> u64 {
        // debug_assert!( n <= self.bits_used, "{n} bits requested, but only {} avaliable", self.bits_used);

        let lowest_n_bits = (1 << n) - 1;
        self.bit_buffer & lowest_n_bits
    }

    #[inline(always)]
    pub fn drop_bits(&mut self, n: u8) {
        self.bit_buffer >>= n;
        self.bits_used -= n;
    }

    #[inline(always)]
    pub fn start_sync_search(&mut self) -> ([u8; 4], usize) {
        let mut buf = [0u8; 4];

        self.bit_buffer <<= self.bits_used & 7;
        self.bits_used -= self.bits_used & 7;

        let mut len = 0;
        while self.bits_used >= 8 {
            buf[len] = self.bit_buffer as u8;
            len += 1;
            self.bit_buffer >>= 8;
            self.bits_used -= 8;
        }

        (buf, len)
    }

    #[inline(always)]
    pub fn init_bits(&mut self) {
        self.bit_buffer = 0;
        self.bits_used = 0;
    }

    #[inline(always)]
    pub fn prime(&mut self, bits: u8, value: u64) {
        let value = value & ((1 << bits) - 1);
        self.bit_buffer += value << self.bits_used;
        self.bits_used += bits;
    }

    #[inline(always)]
    pub fn return_unused_bytes(&mut self) {
        let len = self.bits_used >> 3;
        // SAFETY: ptr is advanced whenever bits_used is incremented by 8, so this sub is always
        // in bounds.
        self.ptr = unsafe { self.ptr.sub(len as usize) };
        self.bits_used -= len << 3;
        self.bit_buffer &= (1u64 << self.bits_used) - 1u64;

        assert!(self.bits_used <= 32);
    }
}

#[cfg(feature = "std")]
impl std::io::Read for BitReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        assert_eq!(self.bits_used, 0, "bit buffer not cleared before read");

        let number_of_bytes = Ord::min(buf.len(), self.bytes_remaining());

        // SAFETY: `buf` is a mutable (exclusive) reference, so it cannot overlap the memory that
        // the reader contains
        unsafe { core::ptr::copy_nonoverlapping(self.ptr, buf.as_mut_ptr(), number_of_bytes) }

        self.ptr = unsafe { self.ptr.add(number_of_bytes) };
        Ok(number_of_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::BitReader;

    #[test]
    fn detach_input_preserves_bits_but_not_caller_slice() {
        let input = [0xabu8, 0xcd];
        let mut reader = BitReader::new();
        reader.prime(3, 0b101);

        unsafe { reader.attach_input(input.as_ptr(), input.len()) };
        assert_eq!(reader.bytes_remaining(), 2);
        assert_eq!(reader.bits_in_buffer(), 3);
        assert_eq!(reader.hold() & 0b111, 0b101);

        reader.detach_input();

        assert_eq!(reader.bytes_remaining(), 0);
        assert!(reader.as_slice().is_empty());
        assert_eq!(reader.bits_in_buffer(), 3);
        assert_eq!(reader.hold() & 0b111, 0b101);

        unsafe { reader.attach_input(core::ptr::null(), 0) };
        assert_eq!(reader.bytes_remaining(), 0);
        assert!(reader.as_slice().is_empty());
        assert_eq!(reader.bits_in_buffer(), 3);
    }
}
