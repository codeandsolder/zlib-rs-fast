#![forbid(unsafe_code)]

use crate::ReturnCode;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BitBuffer {
    bit_buffer: u64,
    bits_used: u8,
}

impl BitBuffer {
    #[inline]
    pub const fn bits_in_buffer(&self) -> u8 {
        self.bits_used
    }

    #[inline]
    pub const fn hold(&self) -> u64 {
        self.bit_buffer
    }

    #[inline]
    pub const fn bits(&self, n: usize) -> u64 {
        let lowest_n_bits = (1 << n) - 1;
        self.bit_buffer & lowest_n_bits
    }

    /// Extract a DEFLATE code or extra field as its native bounded
    /// representation. Huffman lookup widths and extra fields are at most
    /// 15 bits; accepting 16 also covers the gzip XLEN field.
    ///
    /// The check prevents silently losing high bits if a decoder ever
    /// passes a wider width. Extracting the two low bytes is portable to
    /// 32-bit hosts and avoids narrowing a general 64-bit reservoir value.
    #[inline]
    pub const fn bits_u16(&self, n: usize) -> u16 {
        assert!(n <= 16, "DEFLATE code/field exceeds 16 bits");
        let bytes = self.bits(n).to_le_bytes();
        u16::from_le_bytes([bytes[0], bytes[1]])
    }

    /// Bounded bit extraction for Rust slice and lookup-table indices.
    #[inline]
    pub fn bits_usize(&self, n: usize) -> usize {
        usize::from(self.bits_u16(n))
    }

    /// Decode a complete 32-bit wire field (CRC, Adler checksum, ISIZE,
    /// gzip timestamp, stored-block LEN/NLEN) without truncating a u64
    /// reservoir or relying on the host's usize width.
    #[inline]
    pub const fn bits_u32(&self, n: usize) -> u32 {
        assert!(n <= 32, "DEFLATE wire field exceeds 32 bits");
        let bytes = self.bits(n).to_le_bytes();
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }

    #[inline]
    pub const fn drop_bits(&mut self, n: u8) {
        self.bit_buffer >>= n;
        self.bits_used -= n;
    }

    #[inline]
    pub const fn next_byte_boundary(&mut self) {
        self.drop_bits(self.bits_used & 0b0111);
    }

    #[inline]
    pub const fn init_bits(&mut self) {
        self.bit_buffer = 0;
        self.bits_used = 0;
    }

    #[inline]
    pub const fn prime(&mut self, bits: u8, value: u64) {
        let value = value & ((1 << bits) - 1);
        self.bit_buffer += value << self.bits_used;
        self.bits_used += bits;
    }

    /// Validate a Huffman lookup width once per fast-decoder invocation,
    /// instead of rechecking its range for every decoded symbol.
    #[inline]
    pub(super) fn huffman_mask(width: usize) -> Option<u16> {
        let width = u32::try_from(width).ok()?;
        1u16.checked_shl(width).map(|shifted| shifted - 1)
    }

    /// Index a validated 15-bit table mask against a reservoir snapshot.
    /// The fast decoder also uses snapshots captured before a refill.
    #[inline]
    pub(super) fn lookup_index(hold: u64, mask: u16) -> usize {
        let bytes = hold.to_le_bytes();
        usize::from(u16::from_le_bytes([bytes[0], bytes[1]]) & mask)
    }

    /// Return the number of bytes contributed to the reservoir by an
    /// eight-byte lookahead. When 56..=63 bits are already buffered,
    /// this is a speculative peek that consumes no new input bytes.
    ///
    /// A refill may *inspect* eight bytes, but only the whole bytes actually
    /// incorporated into the reservoir advance the input cursor.
    #[inline]
    fn refill_from_word(&mut self, lookahead: u64) -> usize {
        let previous_bits = self.bits_used;
        debug_assert!(previous_bits < 64);
        let loaded_bytes = usize::from(7 - (previous_bits >> 3));
        // The high byte of the lookahead may be speculative. It becomes
        // valid only after the next refill, and return_unused_bytes masks it.
        self.bit_buffer |= lookahead << previous_bits;
        self.bits_used = 0b111_000 | (previous_bits & 0b111);
        loaded_bytes
    }

    #[inline]
    pub fn start_sync_search(&mut self) -> ([u8; 4], usize) {
        assert!(
            self.bits_used <= 32,
            "persistent inflate reservoir fits sync-search scratch storage"
        );
        let mut buf = [0u8; 4];

        self.next_byte_boundary();

        let mut len = 0;
        while self.bits_used >= 8 {
            // Sync search consumes the low byte of the little-endian reservoir.
            buf[len] = self.bit_buffer.to_le_bytes()[0];
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
pub struct BitReader<'input> {
    input: &'input [u8],
    position: usize,
    bits: BitBuffer,
    // Trailing bytes pulled into this call's reservoir after the last direct input skip.
    // On entry the reservoir may already hold whole bytes supplied by inflatePrime, which
    // must never be refunded to the current input pointer. Direct input skips also break
    // the contiguous suffix of bytes eligible for a rewind.
    loaded_input_bytes: usize,
}

impl<'input> BitReader<'input> {
    pub const fn from_buffer(bits: BitBuffer, input: &'input [u8]) -> Self {
        Self {
            input,
            position: 0,
            bits,
            loaded_input_bytes: 0,
        }
    }

    pub const fn into_buffer(self) -> BitBuffer {
        self.bits
    }

    #[inline]
    pub fn advance(&mut self, bytes: usize) {
        let next = self.position.saturating_add(bytes).min(self.input.len());
        if next != self.position {
            // Bytes buffered before a direct skip are no longer a contiguous suffix of
            // the input. Keep their bits resident rather than rewinding over skipped bytes.
            self.loaded_input_bytes = 0;
            self.position = next;
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.input[self.position..]
    }

    #[inline]
    pub const fn bits_in_buffer(&self) -> u8 {
        self.bits.bits_used
    }

    #[inline]
    pub const fn hold(&self) -> u64 {
        self.bits.hold()
    }

    #[inline]
    pub const fn bytes_remaining(&self) -> usize {
        self.input.len() - self.position
    }

    #[inline]
    pub const fn bytes_remaining_including_buffer(&self) -> usize {
        self.bytes_remaining() + (self.bits.bits_used as usize >> 3)
    }

    #[inline]
    pub fn need_bits(&mut self, n: usize) -> Result<(), ReturnCode> {
        while (self.bits.bits_used as usize) < n {
            self.pull_byte()?;
        }
        Ok(())
    }

    #[inline]
    pub const fn next_byte_boundary(&mut self) {
        self.bits.next_byte_boundary();
    }

    #[inline]
    pub fn pull_byte(&mut self) -> Result<u8, ReturnCode> {
        let Some(&byte) = self.input.get(self.position) else {
            return Err(ReturnCode::Ok);
        };
        self.position += 1;
        self.loaded_input_bytes += 1;
        self.bits.prime(8, u64::from(byte));
        Ok(byte)
    }

    /// Refill the hot decoder reservoir from eight readable bytes.
    ///
    /// The fast decoder may load one byte beyond the logically consumed prefix; `position` advances
    /// only by the whole bytes that actually become valid reservoir bits, and
    /// [`Self::return_unused_bytes`] gives any remaining whole bytes back before the call returns.
    #[inline]
    pub fn refill(&mut self) {
        assert!(
            self.bytes_remaining() >= 8,
            "bit-reader refill requires eight input bytes"
        );
        // Fixed-size indexing below is safe because we checked that eight bytes
        // are available. A stack array makes the endian conversion explicit
        // without an additional fallible conversion or unchecked pointer read.
        let bytes = self.as_slice();
        let lookahead = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        let loaded_bytes = self.bits.refill_from_word(lookahead);
        self.position += loaded_bytes;
        self.loaded_input_bytes += loaded_bytes;
    }

    #[inline]
    pub const fn bits(&self, n: usize) -> u64 {
        self.bits.bits(n)
    }

    #[inline]
    pub const fn bits_u16(&self, n: usize) -> u16 {
        self.bits.bits_u16(n)
    }

    #[inline]
    pub fn bits_usize(&self, n: usize) -> usize {
        self.bits.bits_usize(n)
    }

    #[inline]
    pub const fn bits_u32(&self, n: usize) -> u32 {
        self.bits.bits_u32(n)
    }

    #[inline]
    pub const fn drop_bits(&mut self, n: u8) {
        self.bits.drop_bits(n);
    }

    #[inline]
    pub const fn init_bits(&mut self) {
        self.bits.init_bits();
    }

    /// Return only whole buffered bytes actually loaded from this call's input slice.
    ///
    /// The low reservoir bits may predate this input (including bits injected with
    /// `inflatePrime`), and `advance()` may consume input without loading it into the reservoir.
    /// Neither source is eligible for a rewind.
    #[inline]
    pub fn return_unused_bytes(&mut self) {
        let bytes = usize::from(self.bits.bits_used >> 3).min(self.loaded_input_bytes);
        self.position -= bytes;
        self.loaded_input_bytes -= bytes;
        // At most seven whole bytes are refundable. Account for them in the
        // reservoir's native bit-count domain rather than narrowing a host index.
        for _ in 0..bytes {
            self.bits.bits_used -= 8;
        }
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
        // Direct reads have the same rewind boundary as any other direct skip.
        self.advance(count);
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
    fn table_lookup_index_preserves_all_15_code_bits_on_wide_reservoirs() -> Result<(), &'static str>
    {
        for width in 0..=15 {
            let mask = BitBuffer::huffman_mask(width).ok_or("valid Huffman width was rejected")?;
            for value in [0, 1, 0xffff, 0xffff_0000, 0x1234_abcd_5678_ef01, u64::MAX] {
                assert_eq!(
                    u64::try_from(BitBuffer::lookup_index(value, mask)),
                    Ok(value & u64::from(mask)),
                    "DEFLATE lookup bits={width}, value={value:x}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn table_lookup_mask_rejects_widths_outside_deflate() {
        assert_eq!(BitBuffer::huffman_mask(15), Some(0x7fff));
        assert_eq!(BitBuffer::huffman_mask(16), None);
        assert_eq!(BitBuffer::huffman_mask(usize::MAX), None);
    }

    #[test]
    fn full_word_extraction_is_exact_for_every_width_and_portable_to_32_bit() {
        let mut bits = BitBuffer::default();
        bits.prime(32, 0xa5bf_c4ed);
        bits.prime(16, 0x7e39);
        let reader = BitReader::from_buffer(bits, &[]);
        for width in 0..=32 {
            assert_eq!(
                u64::from(reader.bits_u32(width)),
                reader.bits(width),
                "32-bit wire field width {width}"
            );
        }
        assert_eq!(reader.bits_u32(32), 0xa5bf_c4ed);
    }

    #[test]
    #[should_panic(expected = "DEFLATE wire field exceeds 32 bits")]
    fn full_word_extraction_rejects_truncating_widths() {
        let reader = BitReader::from_buffer(BitBuffer::default(), &[]);
        let _ = reader.bits_u32(33);
    }

    #[test]
    fn short_code_extraction_is_exact_for_every_width_and_portable_to_32_bit() {
        let mut bits = BitBuffer::default();
        bits.prime(32, 0x9abc_def0);
        let reader = BitReader::from_buffer(bits, &[]);
        for width in 0..=16 {
            assert_eq!(
                u64::from(reader.bits_u16(width)),
                reader.bits(width),
                "DEFLATE field width {width}"
            );
            assert_eq!(
                reader.bits_usize(width),
                usize::from(reader.bits_u16(width))
            );
        }
        assert_eq!(reader.bits_u16(16), 0xdef0);
    }

    #[test]
    #[should_panic(expected = "DEFLATE code/field exceeds 16 bits")]
    fn short_code_extraction_rejects_widths_that_would_truncate() {
        let reader = BitReader::from_buffer(BitBuffer::default(), &[]);
        let _ = reader.bits_u16(17);
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
        assert_eq!(reader.as_slice(), []);
        assert_eq!(reader.bits_in_buffer(), 3);
    }

    #[test]
    fn cursor_never_advances_past_its_input() {
        let input = [1u8, 2, 3];
        let mut reader = BitReader::from_buffer(BitBuffer::default(), &input);
        reader.advance(2);
        assert_eq!(reader.as_slice(), &[3]);
        reader.advance(usize::MAX);
        assert_eq!(reader.as_slice(), []);
        assert_eq!(reader.bytes_remaining(), 0);
    }

    #[test]
    fn rewind_does_not_rewind_primed_bits_from_a_previous_call() {
        let mut original = BitBuffer::default();
        original.prime(16, 0xbeef);

        // inflatePrime has already provided whole bytes before any caller input exists.
        // They must be retained even if the fast decoder exits without pulling any bytes.
        let mut reader = BitReader::from_buffer(original, &[]);
        reader.return_unused_bytes();
        assert_eq!(reader.into_buffer(), original);

        // Only the one byte pulled from this particular input span can be returned.
        let input = [0x73, 0x9c, 0xae];
        let mut reader = BitReader::from_buffer(original, &input);
        assert_eq!(reader.pull_byte(), Ok(0x73));
        reader.return_unused_bytes();
        assert_eq!(reader.into_buffer(), original);
        assert_eq!(reader.as_slice(), &input);

        // Skipping raw input must not make primed reservoir bits refundable.
        let mut reader = BitReader::from_buffer(original, &input);
        reader.advance(1);
        reader.return_unused_bytes();
        assert_eq!(reader.as_slice(), &input[1..]);
        assert_eq!(reader.into_buffer(), original);

        // A direct skip after a buffered read also prevents rewinding through the skip.
        let mut reader = BitReader::from_buffer(BitBuffer::default(), &input);
        assert_eq!(reader.pull_byte(), Ok(0x73));
        reader.advance(1);
        reader.return_unused_bytes();
        assert_eq!(reader.as_slice(), &input[2..]);
        assert_eq!(reader.bits_in_buffer(), 8);
        assert_eq!(reader.hold(), 0x73);

        // A partial use of primed bits leaves only the remaining primed bits resident.
        let mut partial = BitBuffer::default();
        partial.prime(11, 0x5ab);
        let mut reader = BitReader::from_buffer(partial, &input);
        assert_eq!(reader.pull_byte(), Ok(0x73));
        reader.drop_bits(3);
        reader.return_unused_bytes();
        partial.drop_bits(3);
        assert_eq!(reader.into_buffer(), partial);
        assert_eq!(reader.as_slice(), &input);

        // Fast refill observes eight bytes but logically loads fewer. It must refund
        // only the actual input bytes, not the preexisting four primed bytes.
        let mut max_prime = BitBuffer::default();
        max_prime.prime(16, 0xabcd);
        max_prime.prime(16, 0x1234);
        let mut reader = BitReader::from_buffer(max_prime, &[0x42; 16]);
        reader.refill();
        reader.return_unused_bytes();
        assert_eq!(reader.into_buffer(), max_prime);
        assert_eq!(reader.bytes_remaining(), 16);
    }

    #[test]
    fn refill_rewind_retains_only_valid_bits_for_every_alignment() {
        let input = [0x31, 0x82, 0x73, 0x14, 0xf5, 0xa6, 0x47, 0x98, 0x39];
        let input_word = u64::from_le_bytes([
            input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7],
        ]);
        for initial_bits in 0..=63 {
            let mut bits = BitBuffer::default();
            bits.prime(initial_bits, u64::from(u32::MAX));
            let original = (u64::from(u32::MAX) & ((1_u64 << initial_bits) - 1))
                | (input_word << initial_bits);
            let loaded_bytes = usize::from(7 - (initial_bits >> 3));
            for consumed in 0..=32u8 {
                let mut reader = BitReader::from_buffer(bits, &input);
                reader.refill();
                assert_eq!(reader.bytes_remaining(), input.len() - loaded_bytes);
                assert_eq!(reader.bits_in_buffer(), 0b111_000 | (initial_bits & 0b111));
                assert_eq!(reader.hold(), original);

                reader.drop_bits(consumed);
                let remaining = usize::from(reader.bits_in_buffer());
                let refundable = (remaining / 8).min(loaded_bytes);
                let retained_bits = remaining - refundable * 8;
                // The fast decoder exits only after it can persist at most
                // 32 leftover bits. Larger states are valid *during* refill.
                if retained_bits > 32 {
                    continue;
                }
                reader.return_unused_bytes();

                let retained_mask = (1_u64 << retained_bits) - 1;
                assert_eq!(usize::from(reader.bits_in_buffer()), retained_bits);
                assert_eq!(reader.hold(), (original >> consumed) & retained_mask);
                assert_eq!(
                    reader.bytes_remaining(),
                    input.len() - loaded_bytes + refundable
                );
                if consumed == 0 {
                    assert_eq!(reader.into_buffer(), bits);
                    assert_eq!(reader.as_slice(), &input);
                }
            }
        }
    }

    #[test]
    fn refill_and_rewind_preserve_the_unconsumed_input() {
        let input = [0x10u8, 0x21, 0x32, 0x43, 0x54, 0x65, 0x76, 0x87, 0x98];
        let mut reader = BitReader::from_buffer(BitBuffer::default(), &input);
        reader.refill();
        assert_eq!(reader.bits_in_buffer(), 56);
        assert_eq!(reader.bytes_remaining(), 2);
        assert_eq!(reader.hold(), 0x8776_6554_4332_2110);

        reader.return_unused_bytes();
        assert_eq!(reader.bits_in_buffer(), 0);
        assert_eq!(reader.as_slice(), &input);
    }
}
