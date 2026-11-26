use crate::stream::error::Error;
use crate::stream::io;
use alloc::vec::Vec;
use core::fmt;
use core::mem::MaybeUninit;

use super::codec::{self, Deflate, ErrorMessage, Inflate};
use crate::stream::Compression;
use crate::Status;

/// Raw in-memory compression stream for blocks of data.
///
/// This type is the building block for the I/O streams in the rest of this
/// crate. It requires more management than the [`Read`]/[`Write`] API but is
/// maximally flexible in terms of accepting input from any source and being
/// able to produce output to any memory location.
///
/// It is recommended to use the I/O stream adaptors over this type as they're
/// easier to use.
///
/// [`Read`]: https://doc.rust-lang.org/std/io/trait.Read.html
/// [`Write`]: https://doc.rust-lang.org/std/io/trait.Write.html
#[derive(Debug)]
pub struct Compress {
    inner: Deflate,
}

/// Raw in-memory decompression stream for blocks of data.
///
/// This type is the building block for the I/O streams in the rest of this
/// crate. It requires more management than the [`Read`]/[`Write`] API but is
/// maximally flexible in terms of accepting input from any source and being
/// able to produce output to any memory location.
///
/// It is recommended to use the I/O stream adaptors over this type as they're
/// easier to use.
///
/// [`Read`]: https://doc.rust-lang.org/std/io/trait.Read.html
/// [`Write`]: https://doc.rust-lang.org/std/io/trait.Write.html
#[derive(Debug)]
pub struct Decompress {
    inner: Inflate,
}

/// Values which indicate the form of flushing to be used when compressing
/// in-memory data.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[non_exhaustive]
pub enum FlushCompress {
    /// A typical parameter for passing to compression/decompression functions,
    /// this indicates that the underlying stream to decide how much data to
    /// accumulate before producing output in order to maximize compression.
    None = codec::MZ_NO_FLUSH,

    /// All pending output is flushed to the output buffer, but the output is
    /// not aligned to a byte boundary.
    ///
    /// All input data so far will be available to the decompressor (as with
    /// `Flush::Sync`). This completes the current deflate block and follows it
    /// with an empty fixed codes block that is 10 bits long, and it assures
    /// that enough bytes are output in order for the decompressor to finish the
    /// block before the empty fixed code block.
    Partial = codec::MZ_PARTIAL_FLUSH,

    /// All pending output is flushed to the output buffer and the output is
    /// aligned on a byte boundary so that the decompressor can get all input
    /// data available so far.
    ///
    /// Flushing may degrade compression for some compression algorithms and so
    /// it should only be used when necessary. This will complete the current
    /// deflate block and follow it with an empty stored block.
    Sync = codec::MZ_SYNC_FLUSH,

    /// All output is flushed as with `Flush::Sync` and the compression state is
    /// reset so decompression can restart from this point if previous
    /// compressed data has been damaged or if random access is desired.
    ///
    /// Using this option too often can seriously degrade compression.
    Full = codec::MZ_FULL_FLUSH,

    /// Pending input is processed and pending output is flushed.
    ///
    /// The return value may indicate that the stream is not yet done and more
    /// data has yet to be processed.
    Finish = codec::MZ_FINISH,
}

/// Values which indicate the form of flushing to be used when
/// decompressing in-memory data.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[non_exhaustive]
pub enum FlushDecompress {
    /// A typical parameter for passing to compression/decompression functions,
    /// this indicates that the underlying stream to decide how much data to
    /// accumulate before producing output in order to maximize compression.
    None = codec::MZ_NO_FLUSH,

    /// All pending output is flushed to the output buffer and the output is
    /// aligned on a byte boundary so that the decompressor can get all input
    /// data available so far.
    ///
    /// Flushing may degrade compression for some compression algorithms and so
    /// it should only be used when necessary. This will complete the current
    /// deflate block and follow it with an empty stored block.
    Sync = codec::MZ_SYNC_FLUSH,

    /// Pending input is processed and pending output is flushed.
    ///
    /// The return value may indicate that the stream is not yet done and more
    /// data has yet to be processed.
    Finish = codec::MZ_FINISH,
}

/// The inner state for an error when decompressing
#[derive(Clone, Debug)]
pub(super) enum DecompressErrorInner {
    General { msg: ErrorMessage },
    NeedsDictionary(u32),
}

/// Error returned when a decompression object finds that the input stream of
/// bytes was not a valid input stream of bytes.
#[derive(Clone, Debug)]
pub struct DecompressError(pub(super) DecompressErrorInner);

impl DecompressError {
    /// Indicates whether decompression failed due to requiring a dictionary.
    ///
    /// The resulting integer is the Adler-32 checksum of the dictionary
    /// required.
    #[must_use = "The returned value conveys the requested stream state or configured adapter."]
    pub const fn needs_dictionary(&self) -> Option<u32> {
        match self.0 {
            DecompressErrorInner::NeedsDictionary(adler) => Some(adler),
            DecompressErrorInner::General { .. } => None,
        }
    }
}

#[inline]
pub(super) const fn decompress_failed<T>(msg: ErrorMessage) -> Result<T, DecompressError> {
    Err(DecompressError(DecompressErrorInner::General { msg }))
}

#[inline]
pub(super) const fn decompress_need_dict<T>(adler: u32) -> Result<T, DecompressError> {
    Err(DecompressError(DecompressErrorInner::NeedsDictionary(
        adler,
    )))
}

/// Error returned when a compression object is used incorrectly or otherwise
/// generates an error.
#[derive(Clone, Debug)]
pub struct CompressError {
    pub(super) msg: ErrorMessage,
}

#[inline]
pub(super) const fn compress_failed<T>(msg: ErrorMessage) -> Result<T, CompressError> {
    Err(CompressError { msg })
}

// Both codec directions expose exactly the same bounded-output contract.
macro_rules! stream_methods {
    ($run:ident, $uninit:ident, $vec:ident, $flush:ty, $error:ty) => {
        /// Process as much input as the output buffer permits. Inspect the
        /// stream's total_in and total_out counters for exact progress.
        ///
        /// # Errors
        /// Returns a codec error for invalid input, configuration, or state.
        pub fn $run(&mut self, input: &[u8], output: &mut [u8], flush: $flush) -> Result<Status, $error> {
            self.inner.$run(input, output, flush)
        }

        /// Process input into possibly uninitialized output memory.
        ///
        /// # Errors
        /// Returns a codec error for invalid input, configuration, or state.
        pub fn $uninit(&mut self, input: &[u8], output: &mut [MaybeUninit<u8>], flush: $flush) -> Result<Status, $error> {
            self.inner.$uninit(input, output, flush)
        }

        /// Append output into the vector's existing spare capacity.
        /// Does not allocate or extend capacity. Inspect stream counters
        /// to determine how much input was consumed.
        ///
        /// # Errors
        /// Returns a codec error for invalid input, configuration, or state.
        pub fn $vec(&mut self, input: &[u8], output: &mut Vec<u8>, flush: $flush) -> Result<Status, $error> {
            let before = self.total_out();
            // SAFETY: native codec counters track the initialized output prefix.
            unsafe {
                write_to_spare_capacity_of_vec(output, |out| {
                    let result = self.$uninit(input, out, flush);
                    (self.total_out() - before, result)
                })
            }
        }
    };
}

impl Compress {
    /// Creates a new object ready for compressing data that it's given.
    ///
    /// The `level` argument here indicates what level of compression is going
    /// to be performed, and the `zlib_header` argument indicates whether the
    /// output data should have a zlib header or not.
    #[must_use = "The returned value conveys the requested stream state or configured adapter."]
    pub fn new(level: Compression, zlib_header: bool) -> Self {
        Self {
            inner: Deflate::make(level, zlib_header, codec::MZ_DEFAULT_WINDOW_BITS),
        }
    }

    /// Creates a new object ready for compressing data that it's given.
    ///
    /// The `level` argument here indicates what level of compression is going
    /// to be performed, and the `zlib_header` argument indicates whether the
    /// output data should have a zlib header or not. The `window_bits` parameter
    /// indicates the base-2 logarithm of the sliding window size and must be
    /// between 9 and 15.
    ///
    /// # Panics
    ///
    /// If `window_bits` does not fall into the range 9 ..= 15,
    /// this function will panic.
    #[must_use = "The returned value conveys the requested stream state or configured adapter."]
    pub fn new_with_window_bits(level: Compression, zlib_header: bool, window_bits: u8) -> Self {
        assert!(
            window_bits > 8 && window_bits < 16,
            "window_bits must be within 9 ..= 15"
        );
        Self {
            inner: Deflate::make(level, zlib_header, window_bits),
        }
    }

    /// Creates a new object ready for compressing data that it's given.
    ///
    /// The `level` argument here indicates what level of compression is going
    /// to be performed.
    ///
    /// The Compress object produced by this constructor outputs gzip headers
    /// for the compressed data.
    ///
    /// # Panics
    ///
    /// If `window_bits` does not fall into the range 9 ..= 15,
    /// this function will panic.
    #[must_use = "The returned value conveys the requested stream state or configured adapter."]
    pub fn new_gzip(level: Compression, window_bits: u8) -> Self {
        assert!(
            window_bits > 8 && window_bits < 16,
            "window_bits must be within 9 ..= 15"
        );
        Self {
            inner: Deflate::make(level, true, window_bits + 16),
        }
    }

    /// Returns the total number of input bytes which have been processed by
    /// this compression object.
    #[must_use = "The returned value conveys the requested stream state or configured adapter."]
    pub const fn total_in(&self) -> u64 {
        self.inner.total_in()
    }

    /// Returns the total number of output bytes which have been produced by
    /// this compression object.
    #[must_use = "The returned value conveys the requested stream state or configured adapter."]
    pub const fn total_out(&self) -> u64 {
        self.inner.total_out()
    }

    /// Specifies the compression dictionary to use.
    ///
    /// Returns the Adler-32 checksum of the dictionary.
    ///
    /// # Errors
    ///
    /// The dictionary is invalid for the current stream state.
    pub fn set_dictionary(&mut self, dictionary: &[u8]) -> Result<u32, CompressError> {
        self.inner.set_dictionary(dictionary)
    }

    /// Quickly resets this compressor without having to reallocate anything.
    ///
    /// This is equivalent to dropping this object and then creating a new one.
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// Dynamically updates the compression level.
    ///
    /// This can be used to switch between compression levels for different
    /// kinds of data, or it can be used in conjunction with a call to reset
    /// to reuse the compressor.
    ///
    /// This may return an error if there wasn't enough output space to complete
    /// the compression of the available input data before changing the
    /// compression level. Flushing the stream before calling this method
    /// ensures that the function will succeed on the first call.
    ///
    /// # Errors
    ///
    /// The level is invalid or the codec needs more output space.
    pub fn set_level(&mut self, level: Compression) -> Result<(), CompressError> {
        self.inner.set_level(level)
    }

    stream_methods!(compress, compress_uninit, compress_vec, FlushCompress, CompressError);
}

impl Decompress {
    /// Creates a new object ready for decompressing data that it's given.
    ///
    /// The `zlib_header` argument indicates whether the input data is expected
    /// to have a zlib header or not.
    #[must_use = "The returned value conveys the requested stream state or configured adapter."]
    pub fn new(zlib_header: bool) -> Self {
        Self {
            inner: Inflate::make(zlib_header, codec::MZ_DEFAULT_WINDOW_BITS),
        }
    }

    /// Creates a new object ready for decompressing data that it's given.
    ///
    /// The `zlib_header` argument indicates whether the input data is expected
    /// to have a zlib header or not. The `window_bits` parameter indicates the
    /// base-2 logarithm of the sliding window size and must be between 9 and 15.
    ///
    /// # Panics
    ///
    /// If `window_bits` does not fall into the range 9 ..= 15,
    /// this function will panic.
    #[must_use = "The returned value conveys the requested stream state or configured adapter."]
    pub fn new_with_window_bits(zlib_header: bool, window_bits: u8) -> Self {
        assert!(
            window_bits > 8 && window_bits < 16,
            "window_bits must be within 9 ..= 15"
        );
        Self {
            inner: Inflate::make(zlib_header, window_bits),
        }
    }

    /// Creates a new object ready for decompressing data that it's given.
    ///
    /// The Decompress object produced by this constructor expects gzip headers
    /// for the compressed data.
    ///
    /// # Panics
    ///
    /// If `window_bits` does not fall into the range 9 ..= 15,
    /// this function will panic.
    #[must_use = "The returned value conveys the requested stream state or configured adapter."]
    pub fn new_gzip(window_bits: u8) -> Self {
        assert!(
            window_bits > 8 && window_bits < 16,
            "window_bits must be within 9 ..= 15"
        );
        Self {
            inner: Inflate::make(true, window_bits + 16),
        }
    }

    /// Returns the total number of input bytes which have been processed by
    /// this decompression object.
    #[must_use = "The returned value conveys the requested stream state or configured adapter."]
    pub const fn total_in(&self) -> u64 {
        self.inner.total_in()
    }

    /// Returns the total number of output bytes which have been produced by
    /// this decompression object.
    #[must_use = "The returned value conveys the requested stream state or configured adapter."]
    pub const fn total_out(&self) -> u64 {
        self.inner.total_out()
    }

    stream_methods!(decompress, decompress_uninit, decompress_vec, FlushDecompress, DecompressError);
    
    /// Specifies the decompression dictionary to use.
    ///
    /// # Errors
    ///
    /// The dictionary is invalid for the current stream state.
    pub fn set_dictionary(&mut self, dictionary: &[u8]) -> Result<u32, DecompressError> {
        self.inner.set_dictionary(dictionary)
    }

    /// Performs the equivalent of replacing this decompression state with a
    /// freshly allocated copy.
    ///
    /// This function may not allocate memory, though, and attempts to reuse any
    /// previously existing resources.
    ///
    /// The argument provided here indicates whether the reset state will
    /// attempt to decode a zlib header first or not.
    pub fn reset(&mut self, zlib_header: bool) {
        self.inner.reset(zlib_header);
    }
}

impl Error for DecompressError {}

impl DecompressError {
    /// Retrieve the implementation's message about why the operation failed, if one exists.
    #[must_use = "The returned value conveys the requested stream state or configured adapter."]
    pub const fn message(&self) -> Option<&str> {
        match &self.0 {
            DecompressErrorInner::General { msg } => msg.get(),
            DecompressErrorInner::NeedsDictionary { .. } => None,
        }
    }
}

impl From<DecompressError> for io::Error {
    fn from(data: DecompressError) -> Self {
        Self::other(data)
    }
}

impl fmt::Display for DecompressError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let msg = match &self.0 {
            DecompressErrorInner::General { msg } => msg.get(),
            DecompressErrorInner::NeedsDictionary { .. } => Some("requires a dictionary"),
        };
        match msg {
            Some(msg) => write!(f, "deflate decompression error: {msg}"),
            None => write!(f, "deflate decompression error"),
        }
    }
}

impl Error for CompressError {}

impl CompressError {
    /// Retrieve the implementation's message about why the operation failed, if one exists.
    #[must_use = "The returned value conveys the requested stream state or configured adapter."]
    pub const fn message(&self) -> Option<&str> {
        self.msg.get()
    }
}

impl From<CompressError> for io::Error {
    fn from(data: CompressError) -> Self {
        Self::other(data)
    }
}

impl fmt::Display for CompressError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.msg.get() {
            Some(msg) => write!(f, "deflate compression error: {msg}"),
            None => write!(f, "deflate compression error"),
        }
    }
}

/// Allows `writer` to write data into the spare capacity of the `output` vector.
/// This will not reallocate the vector provided or attempt to grow it, so space
/// for the `output` must be reserved by the caller before calling this
/// function.
///
/// `writer` needs to return the number of bytes written (and can also return
/// another arbitrary return value).
///
/// # Safety:
///
/// The length returned by the `writer` must be equal to actual number of bytes written
/// to the uninitialized slice passed in and initialized.
unsafe fn write_to_spare_capacity_of_vec<T>(
    output: &mut Vec<u8>,
    writer: impl FnOnce(&mut [MaybeUninit<u8>]) -> (u64, T),
) -> T {
    let len = output.len();
    let spare = output.capacity() - len;

    let (bytes_written, ret) = writer(output.spare_capacity_mut());
    let written = usize::try_from(bytes_written).unwrap_or(spare).min(spare);
    // SAFETY: the caller guarantees the reported initialized prefix. written is bounded by spare capacity.
    unsafe { output.set_len(len + written) };

    ret
}

#[cfg(test)]
#[path = "tests/mem_tests.rs"]
mod tests;
