//! Streaming adapters for buffered I/O.

use super::{zio, Compress, Compression, Decompress};
use core::mem;
use std::io::{self, BufRead, Read, Write};

/// Streaming adapter over a caller-supplied buffered reader.
/// The codec type determines compression or decompression, while ZLIB
/// selects the wire framing. Decoders stop after the first member and
/// leave following bytes in the underlying reader.
#[derive(Debug)]
pub struct Stream<R, D, const ZLIB: bool> {
    obj: R,
    data: D,
}

/// Buffered compression stream.
pub type Encoder<R, const ZLIB: bool> = Stream<R, Compress, ZLIB>;
/// Buffered decompression stream.
pub type Decoder<R, const ZLIB: bool> = Stream<R, Decompress, ZLIB>;

impl<R: BufRead, const ZLIB: bool> Encoder<R, ZLIB> {
    /// Construct a compressor at the given level.
    pub fn new(r: R, level: Compression) -> Self {
        Self::new_with_compress(r, Compress::new(level, ZLIB))
    }

    /// Construct a compressor with caller-configured codec state.
    pub const fn new_with_compress(r: R, compression: Compress) -> Self {
        Self {
            obj: r,
            data: compression,
        }
    }
}

impl<R: BufRead, const ZLIB: bool> Decoder<R, ZLIB> {
    /// Construct a decompressor.
    pub fn new(r: R) -> Self {
        Self::new_with_decompress(r, Decompress::new(ZLIB))
    }

    /// Construct a decompressor with caller-configured codec state.
    pub const fn new_with_decompress(r: R, decompression: Decompress) -> Self {
        Self {
            obj: r,
            data: decompression,
        }
    }
}

impl<R, D, const ZLIB: bool> Stream<R, D, ZLIB> {
    /// Borrow the underlying reader.
    pub const fn get_ref(&self) -> &R {
        &self.obj
    }

    /// Borrow the reader mutably. Preserve unread data and its position until
    /// the compressed member has ended; use reset to start a new stream.
    pub const fn get_mut(&mut self) -> &mut R {
        &mut self.obj
    }

    /// Return the underlying reader, retaining unread input.
    pub fn into_inner(self) -> R {
        self.obj
    }
}

impl<R, D: zio::Ops, const ZLIB: bool> Stream<R, D, ZLIB> {
    /// Reset the codec while retaining the reader.
    pub fn reset_data(&mut self) {
        self.data.reset_stream(ZLIB);
    }

    /// Reset the codec and replace the reader.
    pub fn reset(&mut self, r: R) -> R {
        self.reset_data();
        mem::replace(&mut self.obj, r)
    }

    /// Bytes consumed by the codec.
    #[must_use = "The counter reports codec progress."]
    pub fn total_in(&self) -> u64 {
        self.data.total_in()
    }

    /// Bytes produced by the codec.
    #[must_use = "The counter reports codec progress."]
    pub fn total_out(&self) -> u64 {
        self.data.total_out()
    }
}

impl<R: BufRead, D: zio::Ops, const ZLIB: bool> Read for Stream<R, D, ZLIB> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        zio::read(&mut self.obj, &mut self.data, buf)
    }
}

impl<R: BufRead + Write, D: zio::Ops, const ZLIB: bool> Write for Stream<R, D, ZLIB> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.obj.write(buf)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.obj.flush()
    }
}

/// Raw DEFLATE encoder.
pub type DeflateEncoder<T> = Encoder<T, false>;
/// Raw DEFLATE decoder.
pub type DeflateDecoder<T> = Decoder<T, false>;
/// Zlib-framed encoder.
pub type ZlibEncoder<T> = Encoder<T, true>;
/// Zlib-framed decoder.
pub type ZlibDecoder<T> = Decoder<T, true>;

#[cfg(test)]
#[path = "tests/bufread_zlib_tests.rs"]
mod zlib_tests;

#[cfg(test)]
#[path = "tests/bufread_raw_tests.rs"]
mod raw_tests;

/// Gzip-framed I/O adapters.
pub use super::gz::bufread::{GzDecoder, GzEncoder, MultiGzDecoder};
