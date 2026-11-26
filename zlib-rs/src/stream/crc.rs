//! Streaming CRC using the native optimized checksum.

use crate::stream::io;
use crate::stream::io::{BufRead, Read, Write};

/// The CRC calculated by a [`CrcReader`].
#[derive(Debug, Default)]
pub struct Crc {
    consumed: u64,
    state: u32,
}

impl Crc {
    /// Create a new CRC.
    #[must_use = "The returned value conveys the requested stream state or configured adapter."]
    pub const fn new() -> Self {
        Self {
            consumed: 0,
            state: 0,
        }
    }
    /// Returns the current crc32 checksum.
    #[must_use = "The returned value conveys the requested stream state or configured adapter."]
    pub const fn sum(&self) -> u32 {
        self.state
    }
    /// Number of processed bytes modulo 2^32 (the gzip ISIZE field).
    #[must_use = "The returned value conveys the requested stream state or configured adapter."]
    pub const fn amount(&self) -> u32 {
        let [a, b, c, d, ..] = self.consumed.to_le_bytes();
        u32::from_le_bytes([a, b, c, d])
    }
    /// Update the checksum.
    pub fn update(&mut self, data: &[u8]) {
        self.consumed = self.consumed.wrapping_add(data.len() as u64);
        self.state = crate::crc32::crc32(self.state, data);
    }
    /// Reset to the initial state.
    pub const fn reset(&mut self) {
        *self = Self::new();
    }
    /// Combine the CRC with the checksum of the subsequent block.
    pub const fn combine(&mut self, additional_crc: &Self) {
        self.consumed = self.consumed.wrapping_add(additional_crc.consumed);
        self.state =
            crate::crc32::crc32_combine(self.state, additional_crc.state, additional_crc.consumed);
    }
}

/// A wrapper around a [`Read`] that calculates the CRC.
///
/// [`Read`]: https://doc.rust-lang.org/std/io/trait.Read.html
#[derive(Debug)]
pub struct CrcReader<R> {
    inner: R,
    crc: Crc,
}

impl<R: Read> CrcReader<R> {
    /// Create a new `CrcReader`.
    pub const fn new(r: R) -> Self {
        Self {
            inner: r,
            crc: Crc::new(),
        }
    }
}

impl<R> CrcReader<R> {
    /// Get the Crc for this `CrcReader`.
    pub const fn crc(&self) -> &Crc {
        &self.crc
    }

    /// Get the reader that is wrapped by this `CrcReader`.
    pub fn into_inner(self) -> R {
        self.inner
    }

    /// Get the reader that is wrapped by this `CrcReader` by reference.
    pub const fn get_ref(&self) -> &R {
        &self.inner
    }

    /// Get a mutable reference to the reader that is wrapped by this `CrcReader`.
    pub const fn get_mut(&mut self) -> &mut R {
        &mut self.inner
    }

    /// Reset the Crc in this `CrcReader`.
    pub const fn reset(&mut self) {
        self.crc.reset();
    }
}

impl<R: Read> Read for CrcReader<R> {
    fn read(&mut self, into: &mut [u8]) -> io::Result<usize> {
        let amt = self.inner.read(into)?;
        self.crc.update(&into[..amt]);
        Ok(amt)
    }
}

impl<R: BufRead> BufRead for CrcReader<R> {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        self.inner.fill_buf()
    }
    fn consume(&mut self, amt: usize) {
        if let Ok(data) = self.inner.fill_buf() {
            self.crc.update(&data[..amt]);
        }
        self.inner.consume(amt);
    }
}

/// A wrapper around a [`Write`] that calculates the CRC.
///
/// [`Write`]: https://doc.rust-lang.org/std/io/trait.Write.html
#[derive(Debug)]
pub struct CrcWriter<W> {
    inner: W,
    crc: Crc,
}

impl<W> CrcWriter<W> {
    /// Get the Crc for this `CrcWriter`.
    pub const fn crc(&self) -> &Crc {
        &self.crc
    }

    /// Get the writer that is wrapped by this `CrcWriter`.
    pub fn into_inner(self) -> W {
        self.inner
    }

    /// Get the writer that is wrapped by this `CrcWriter` by reference.
    pub const fn get_ref(&self) -> &W {
        &self.inner
    }

    /// Get a mutable reference to the writer that is wrapped by this `CrcWriter`.
    pub const fn get_mut(&mut self) -> &mut W {
        &mut self.inner
    }

    /// Reset the Crc in this `CrcWriter`.
    pub const fn reset(&mut self) {
        self.crc.reset();
    }
}

impl<W: Write> CrcWriter<W> {
    /// Create a new `CrcWriter`.
    pub const fn new(w: W) -> Self {
        Self {
            inner: w,
            crc: Crc::new(),
        }
    }
}

impl<W: Write> Write for CrcWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let amt = self.inner.write(buf)?;
        self.crc.update(&buf[..amt]);
        Ok(amt)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::Crc;

    fn crc_of(data: &[u8]) -> Crc {
        let mut c = Crc::new();
        c.update(data);
        c
    }

    fn sum_of(data: &[u8]) -> u32 {
        crc_of(data).sum()
    }

    #[test]
    fn new_is_empty() {
        let c = Crc::new();
        assert_eq!(c.amount(), 0);
        assert_eq!(c.sum(), 0);
    }

    #[test]
    fn known_vector_hello() {
        assert_eq!(sum_of(b"hello"), 0x3610_A686);
    }

    #[test]
    fn known_vector_quick_brown_fox() {
        assert_eq!(
            sum_of(b"The quick brown fox jumps over the lazy dog"),
            0x414F_A339
        );
    }

    #[test]
    fn update_is_streaming() {
        let mut c = Crc::new();
        c.update(b"hello");
        c.update(b" ");
        c.update(b"world");

        assert_eq!(c.amount(), 11);
        assert_eq!(c.sum(), sum_of(b"hello world"));
    }

    #[test]
    fn reset_restores_initial_state() {
        let mut c = Crc::new();
        c.update(b"abc");
        assert_ne!(c.sum(), 0);
        assert_eq!(c.amount(), 3);

        c.reset();
        assert_eq!(c.amount(), 0);
        assert_eq!(c.sum(), 0);
    }

    #[test]
    fn combine_matches_concatenation() {
        let a = b"hello ";
        let b = b"world";

        let mut ca = crc_of(a);
        let cb = crc_of(b);

        ca.combine(&cb);

        assert_eq!(ca.amount(), 11);
        assert_eq!(ca.sum(), sum_of(b"hello world"));
    }
}
