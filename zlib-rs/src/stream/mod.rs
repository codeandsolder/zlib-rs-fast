//! Streaming raw DEFLATE, zlib and gzip adapters over the native codec.
//! Requires the standard library.

use alloc::vec::Vec;
use std::{error, io};

pub use crate::stream::crc::{Crc, CrcReader, CrcWriter};
pub use crate::stream::gz::GzBuilder;
pub use crate::stream::gz::GzHeader;
pub use crate::stream::mem::{Compress, CompressError, Decompress, DecompressError};
pub use crate::stream::mem::{FlushCompress, FlushDecompress};
pub use crate::Status;

mod bufreader;
mod codec;
mod crc;
#[cfg(test)]
#[path = "tests/deflate_tests.rs"]
mod deflate_tests;
mod gz;
mod mem;
mod zio;
#[cfg(test)]
#[path = "tests/zlib_tests.rs"]
mod zlib_tests;

/// Types which operate over [`Read`] streams, both encoders and decoders for
/// various formats.
///
/// Note that the `read` decoder types may read past the end of the compressed
/// data while decoding. If the caller requires subsequent reads to start
/// immediately following the compressed data wrap the `Read` type in a
/// [`BufReader`] and use the `BufReader` with the equivalent decoder from the
/// `bufread` module and also for the subsequent reads.
///
/// [`Read`]: https://doc.rust-lang.org/std/io/trait.Read.html
/// [`BufReader`]: https://doc.rust-lang.org/std/io/struct.BufReader.html
pub mod read;

/// Types which operate over [`Write`] streams, both encoders and decoders for
/// various formats.
///
/// [`Write`]: https://doc.rust-lang.org/std/io/trait.Write.html
pub mod write;

/// Types which operate over [`BufRead`] streams, both encoders and decoders for
/// various formats.
///
/// [`BufRead`]: https://doc.rust-lang.org/std/io/trait.BufRead.html
pub mod bufread;

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<read::DeflateEncoder<&[u8]>>();
    assert_send_sync::<read::DeflateDecoder<&[u8]>>();
    assert_send_sync::<read::ZlibEncoder<&[u8]>>();
    assert_send_sync::<read::ZlibDecoder<&[u8]>>();
    assert_send_sync::<read::GzEncoder<&[u8]>>();
    assert_send_sync::<read::GzDecoder<&[u8]>>();
    assert_send_sync::<read::MultiGzDecoder<&[u8]>>();
    assert_send_sync::<write::DeflateEncoder<Vec<u8>>>();
    assert_send_sync::<write::DeflateDecoder<Vec<u8>>>();
    assert_send_sync::<write::ZlibEncoder<Vec<u8>>>();
    assert_send_sync::<write::ZlibDecoder<Vec<u8>>>();
    assert_send_sync::<write::GzEncoder<Vec<u8>>>();
    assert_send_sync::<write::GzDecoder<Vec<u8>>>();
};

/// When compressing data, the compression level can be specified by a value in
/// this struct.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Compression(u32);

impl Compression {
    /// Creates a new description of the compression level with an explicitly
    /// specified integer.
    ///
    /// The integer here is typically on a scale of 0-9 where 0 means "no
    /// compression" and 9 means "take as long as you'd like".
    #[must_use = "The returned value conveys the requested stream state or configured adapter."]
    pub const fn new(level: u32) -> Self {
        Self(level)
    }

    /// No compression is to be performed, this may actually inflate data
    /// slightly when encoding.
    #[must_use = "The returned value conveys the requested stream state or configured adapter."]
    pub const fn none() -> Self {
        Self(0)
    }

    /// Optimize for the best speed of encoding.
    #[must_use = "The returned value conveys the requested stream state or configured adapter."]
    pub const fn fast() -> Self {
        Self(1)
    }

    /// Optimize for the size of data being encoded.
    #[must_use = "The returned value conveys the requested stream state or configured adapter."]
    pub const fn best() -> Self {
        Self(9)
    }

    /// Returns an integer representing the compression level, typically on a
    /// scale of 0-9. See [`new`](Self::new) for details about compression levels.
    #[must_use = "The returned value conveys the requested stream state or configured adapter."]
    pub const fn level(&self) -> u32 {
        self.0
    }
}

impl Default for Compression {
    fn default() -> Self {
        Self(6)
    }
}

#[cfg(test)]
fn random_bytes() -> impl Iterator<Item = u8> {
    use rand::{rngs::SmallRng, RngExt, SeedableRng};
    SmallRng::seed_from_u64(0x7a6c_6962).random_iter()
}
