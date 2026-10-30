#![forbid(unsafe_code)]

use crate::ReturnCode;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct BitBuffer {
    bit_buffer: u64,
    bits_used: u8,
}

impl BitBuffer {
    #[inline(always)]
    pub fn bits_in_buffer(&self) -> u8 {
        self.bits_used
    }

    #[inline(always)]
    pub fn hold(&self) -> u64 {
        self.bit_buffer
    }

    #[inline(always)]
    pub fn bits(&self, n: usize) -> u64 {
        let lowest_n_bits = (1 << n) - 1;
        self.bit_buffer & lowest_n_bits
    }

    #[inline(always)]
    pub fn drop_bits(&mut self, n: u8) {
        self.bit_buffer >>= n;
        self.bits_used -= n;
    }

    #[inline(always)]
    pub fn next_byte_boundary(&mut self) {
        self.drop_bits(self.bits_used & 0b0111);
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
    pub fn start_sync_search(&mut self) -> ([u8; 4], usize) {
        assert!(
            self.bits_used <= 32,
            "persistent inflate reservoir fits sync-search scratch storage"
        );
        let mut buf = [0u8; 4];

        self.next_byte_boundary();

        let mut len = 0;
        while self.bits_used >= 8 {
            buf[len] = self.bit_buffer as u8;
            len += 1;
            self.bit_buffer >>= 8;
            self.bits_used -= 8;
        }

        (buf, len)
    }
}

/// A call-local cursor over one borrowed input buffer plus the DEFLATE bit reservoir.
///
/// Only [`BitBuffer`] is persisted in `State`; the input slice and cursor position die at the end
/// of each public inflate call. This keeps caller-buffer lifetime and pointer arithmetic out of the
/// decoder state machine.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BitReader<'input> {
    input: &'input [u8],
    position: usize,
    bits: BitBuffer,
}

impl<'input> BitReader<'input> {
    pub fn from_buffer(bits: BitBuffer, input: &'input [u8]) -> Self {
        Self {
            input,
            position: 0,
            bits,
        }
    }

    pub fn into_buffer(self) -> BitBuffer {
        self.bits
    }

    #[inline(always)]
    pub fn advance(&mut self, bytes: usize) {
        self.position = self.position.saturating_add(bytes).min(self.input.len());
    }

    #[inline(always)]
    pub fn next_input_ptr(&self) -> *const u8 {
        self.input.as_ptr().wrapping_add(self.position)
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[u8] {
        &self.input[self.position..]
    }

    #[inline(always)]
    pub fn bits_in_buffer(&self) -> u8 {
        self.bits.bits_used
    }

    #[inline(always)]
    pub fn hold(&self) -> u64 {
        self.bits.hold()
    }

    #[inline(always)]
    pub fn bytes_remaining(&self) -> usize {
        self.input.len() - self.position
    }

    #[inline(always)]
    pub fn bytes_remaining_including_buffer(&self) -> usize {
        self.bytes_remaining() + (self.bits.bits_used as usize >> 3)
    }

    #[inline(always)]
    pub fn need_bits(&mut self, n: usize) -> Result<(), ReturnCode> {
        while (self.bits.bits_used as usize) < n {
            self.pull_byte()?;
        }
        Ok(())
    }

    #[inline(always)]
    pub fn next_byte_boundary(&mut self) {
        self.bits.next_byte_boundary();
    }

    #[inline(always)]
    pub fn pull_byte(&mut self) -> Result<u8, ReturnCode> {
        let Some(&byte) = self.input.get(self.position) else {
            return Err(ReturnCode::Ok);
        };
        self.position += 1;
        self.bits.prime(8, byte as u64);
        Ok(byte)
    }

    /// Refill the hot decoder reservoir from eight readable bytes.
    ///
    /// The fast decoder may load one byte beyond the logically consumed prefix; `position` advances
    /// only by the whole bytes that actually become valid reservoir bits, and
    /// [`Self::return_unused_bytes`] gives any remaining whole bytes back before the call returns.
    #[inline(always)]
    pub fn refill(&mut self) {
        assert!(
            self.bytes_remaining() >= 8,
            "bit-reader refill requires eight input bytes"
        );
        let bytes: [u8; 8] = self.as_slice()[..8]
            .try_into()
            .expect("refill length was checked above");
        let read = u64::from_le_bytes(bytes);
        self.bits.bit_buffer |= read << self.bits.bits_used;
        let increment = (63 ^ self.bits.bits_used) >> 3;
        self.position += increment as usize;
        self.bits.bits_used |= 56;
    }

    #[inline(always)]
    pub fn bits(&self, n: usize) -> u64 {
        self.bits.bits(n)
    }

    #[inline(always)]
    pub fn drop_bits(&mut self, n: u8) {
        self.bits.drop_bits(n);
    }

    #[inline(always)]
    pub fn init_bits(&mut self) {
        self.bits.init_bits();
    }

    #[inline(always)]
    pub fn return_unused_bytes(&mut self) {
        let bytes = usize::from(self.bits.bits_used >> 3);
        self.position = self
            .position
            .checked_sub(bytes)
            .expect("buffered whole bytes came from this input cursor");
        self.bits.bits_used -= (bytes as u8) << 3;
        self.bits.bit_buffer &= (1u64 << self.bits.bits_used) - 1u64;
        assert!(self.bits.bits_used <= 32);
    }
}

#[cfg(feature = "std")]
impl std::io::Read for BitReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        assert_eq!(self.bits.bits_used, 0, "bit buffer not cleared before read");
        let count = Ord::min(buf.len(), self.bytes_remaining());
        buf[..count].copy_from_slice(&self.as_slice()[..count]);
        self.position += count;
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::{BitBuffer, BitReader};

    #[test]
    fn sync_search_discards_partial_low_bits_before_scanning_bytes() {
        let mut bits = BitBuffer::default();
        bits.prime(3, 0b101);
        bits.prime(8, 0x00);
        bits.prime(8, 0x00);

        let (buf, len) = bits.start_sync_search();
        assert_eq!(len, 2);
        assert_eq!(&buf[..len], &[0x00, 0x00]);
        assert_eq!(bits.bits_in_buffer(), 0);
    }

    #[test]
    fn bit_buffer_owns_reservoir_operations() {
        let mut bits = BitBuffer::default();
        bits.prime(3, 0b101);
        bits.prime(8, 0xab);

        assert_eq!(bits.bits_in_buffer(), 11);
        assert_eq!(bits.bits(3), 0b101);
        assert_eq!(bits.hold(), 0b101 | (0xabu64 << 3));

        bits.next_byte_boundary();
        assert_eq!(bits.bits_in_buffer(), 8);
        assert_eq!(bits.bits(8), 0xab);

        bits.drop_bits(4);
        assert_eq!(bits.bits_in_buffer(), 4);
        assert_eq!(bits.bits(4), 0x0a);
    }

    #[test]
    fn reader_round_trip_persists_bits_without_caller_input() {
        let input = [0xabu8, 0xcd];
        let mut bits = BitBuffer::default();
        bits.prime(3, 0b101);
        let reader = BitReader::from_buffer(bits, &input);
        assert_eq!(reader.bytes_remaining(), 2);
        assert_eq!(reader.bits_in_buffer(), 3);
        assert_eq!(reader.hold() & 0b111, 0b101);

        let bits = reader.into_buffer();
        assert_eq!(bits.bits_in_buffer(), 3);
        assert_eq!(bits.bit_buffer & 0b111, 0b101);

        let reader = BitReader::from_buffer(bits, &[]);
        assert_eq!(reader.bytes_remaining(), 0);
        assert!(reader.as_slice().is_empty());
        assert_eq!(reader.bits_in_buffer(), 3);
    }

    #[test]
    fn cursor_never_advances_past_its_input() {
        let input = [1u8, 2, 3];
        let mut reader = BitReader::from_buffer(BitBuffer::default(), &input);
        reader.advance(2);
        assert_eq!(reader.as_slice(), &[3]);
        reader.advance(usize::MAX);
        assert!(reader.as_slice().is_empty());
        assert_eq!(reader.bytes_remaining(), 0);
    }

    #[test]
    fn refill_and_rewind_preserve_the_unconsumed_input() {
        let input = [0x10u8, 0x21, 0x32, 0x43, 0x54, 0x65, 0x76, 0x87, 0x98];
        let mut reader = BitReader::from_buffer(BitBuffer::default(), &input);
        reader.refill();
        assert_eq!(reader.bits_in_buffer(), 56);
        assert_eq!(reader.bytes_remaining(), 2);
        assert_eq!(
            reader.hold(),
            u64::from_le_bytes(input[..8].try_into().unwrap())
        );

        reader.return_unused_bytes();
        assert_eq!(reader.bits_in_buffer(), 0);
        assert_eq!(reader.as_slice(), &input);
    }
}
