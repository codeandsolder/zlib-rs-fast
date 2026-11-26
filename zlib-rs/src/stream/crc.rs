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

/// An I/O adapter that calculates the CRC of transferred bytes.
///
/// [`Read`]: https://doc.rust-lang.org/std/io/trait.Read.html
#[derive(Debug)]
pub struct CrcIo<R> {
    inner: R,
    crc: Crc,
}

impl<R> CrcIo<R> {
    /// Create a checksum-tracking I/O adapter.
    pub const fn new(r: R) -> Self {
        Self {
            inner: r,
            crc: Crc::new(),
        }
    }
}

impl<R> CrcIo<R> {
    /// Get the Crc for this `CrcIo`.
    pub const fn crc(&self) -> &Crc {
        &self.crc
    }

    /// Get the reader that is wrapped by this `CrcIo`.
    pub fn into_inner(self) -> R {
        self.inner
    }

    /// Get the reader that is wrapped by this `CrcIo` by reference.
    pub const fn get_ref(&self) -> &R {
        &self.inner
    }

    /// Get a mutable reference to the reader that is wrapped by this `CrcIo`.
    pub const fn get_mut(&mut self) -> &mut R {
        &mut self.inner
    }

    /// Reset the Crc in this `CrcIo`.
    pub const fn reset(&mut self) {
        self.crc.reset();
    }
}

/// Checksum-tracking reader.
pub type CrcReader<R> = CrcIo<R>;
/// Checksum-tracking writer.
pub type CrcWriter<W> = CrcIo<W>;

impl<R: Read> Read for CrcIo<R> {
    fn read(&mut self, into: &mut [u8]) -> io::Result<usize> {
        let amt = self.inner.read(into)?;
        self.crc.update(&into[..amt]);
        Ok(amt)
    }
}

impl<R: BufRead> BufRead for CrcIo<R> {
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

impl<W: Write> Write for CrcIo<W> {
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
#[path = "tests/crc_tests.rs"]
mod tests;
