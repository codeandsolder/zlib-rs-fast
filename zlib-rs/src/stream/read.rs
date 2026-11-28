//! Streaming adapters over unbuffered readers.
//!
//! Read adapters may consume bytes after the end of a compressed member.
//! Use the corresponding bufread adapter when those bytes must be retained.

use super::{bufread, bufreader::BufReader, zio, Compress, Compression, Decompress};
use core::marker::PhantomData;
use std::io::{self, Read, Write};

/// Streaming codec with an owned input buffer.
#[derive(Debug)]
pub struct Reader<R, S> {
    pub(super) inner: S,
    marker: PhantomData<fn() -> R>,
}

impl<R, S> Reader<R, S> {
    pub(super) const fn from_buffered(inner: S) -> Self {
        Self {
            inner,
            marker: PhantomData,
        }
    }
}

/// Access to the underlying buffer shared by the streaming codecs.
pub trait BufferedSource<R> {
    /// Borrow the input buffer.
    fn buffer(&self) -> &BufReader<R>;
    /// Mutably borrow the input buffer.
    fn buffer_mut(&mut self) -> &mut BufReader<R>;
    /// Recover the input buffer.
    fn into_buffer(self) -> BufReader<R>;
}

impl<R, D, const ZLIB: bool> BufferedSource<R> for bufread::Stream<BufReader<R>, D, ZLIB> {
    fn buffer(&self) -> &BufReader<R> {
        self.get_ref()
    }
    fn buffer_mut(&mut self) -> &mut BufReader<R> {
        self.get_mut()
    }
    fn into_buffer(self) -> BufReader<R> {
        self.into_inner()
    }
}

/// Native stream with an owned input buffer.
pub type Stream<R, D, const ZLIB: bool> = Reader<R, bufread::Stream<BufReader<R>, D, ZLIB>>;
/// A buffered compressor over a reader.
pub type Encoder<R, const ZLIB: bool> = Stream<R, Compress, ZLIB>;
/// A buffered decompressor over a reader.
pub type Decoder<R, const ZLIB: bool> = Stream<R, Decompress, ZLIB>;

impl<R: Read, const ZLIB: bool> Encoder<R, ZLIB> {
    /// Construct a compressor.
    pub fn new(r: R, level: Compression) -> Self {
        Self::new_with_compress(r, Compress::new(level, ZLIB))
    }

    /// Construct a compressor using supplied codec state.
    pub fn new_with_compress(r: R, compression: Compress) -> Self {
        Self::from_buffered(bufread::Encoder::new_with_compress(
            BufReader::new(r),
            compression,
        ))
    }
}

impl<R: Read, const ZLIB: bool> Decoder<R, ZLIB> {
    /// Construct a decompressor.
    pub fn new(r: R) -> Self {
        Self::new_with_decompress(r, Decompress::new(ZLIB))
    }

    /// Construct a decompressor with an explicit input buffer.
    pub fn new_with_buf(r: R, buf: Vec<u8>) -> Self {
        Self::new_with_decompress_and_buf(r, buf, Decompress::new(ZLIB))
    }

    /// Construct a decompressor with supplied codec state.
    pub fn new_with_decompress(r: R, decompression: Decompress) -> Self {
        Self::from_buffered(bufread::Decoder::new_with_decompress(
            BufReader::new(r),
            decompression,
        ))
    }

    /// Construct a decompressor with supplied input buffer and codec state.
    pub fn new_with_decompress_and_buf(r: R, buf: Vec<u8>, decompression: Decompress) -> Self {
        Self::from_buffered(bufread::Decoder::new_with_decompress(
            BufReader::with_buf(buf, r),
            decompression,
        ))
    }
}

impl<R, S: BufferedSource<R>> Reader<R, S> {
    /// Borrow the reader.
    pub fn get_ref(&self) -> &R {
        self.inner.buffer().get_ref()
    }

    /// Borrow the reader mutably without discarding buffered input.
    pub fn get_mut(&mut self) -> &mut R {
        self.inner.buffer_mut().get_mut()
    }

    /// Return the reader, discarding any unread buffered bytes.
    pub fn into_inner(self) -> R {
        self.inner.into_buffer().into_inner()
    }
}

impl<R, D: zio::Ops, const ZLIB: bool> Stream<R, D, ZLIB> {
    /// Reset the stream, discard buffered input, and replace the reader.
    pub fn reset(&mut self, r: R) -> R {
        self.inner.reset_data();
        self.inner.get_mut().reset(r)
    }

    /// Bytes consumed by the codec, excluding buffered input.
    #[must_use = "The counter reports codec progress."]
    pub fn total_in(&self) -> u64 {
        self.inner.total_in()
    }

    /// Bytes produced by the codec.
    #[must_use = "The counter reports codec progress."]
    pub fn total_out(&self) -> u64 {
        self.inner.total_out()
    }
}

impl<R, S: Read> Read for Reader<R, S> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }
}

impl<R: Read + Write, S: BufferedSource<R> + Read> Write for Reader<R, S> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.get_mut().write(buf)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.get_mut().flush()
    }
}

super::framing_aliases!();

/// Gzip-framed I/O adapters.
pub use super::gz::read::{GzDecoder, GzEncoder, MultiGzDecoder};
