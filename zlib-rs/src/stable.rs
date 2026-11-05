use core::{ffi::c_uint, mem::MaybeUninit};

use crate::deflate::DeflateConfig;
use crate::inflate::InflateConfig;
use crate::output_buffer::WritableBytes;
use crate::ReturnCode;
pub use crate::{DeflateFlush, InflateFlush};

/// Possible status results of compressing some data or successfully
/// decompressing a block of data.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Status {
    /// Indicates success.
    ///
    /// Means that more input may be needed but isn't available
    /// and/or there's more output to be written but the output buffer is full.
    Ok,

    /// Indicates that forward progress is not possible due to input or output
    /// buffers being empty.
    ///
    /// For compression it means the input buffer needs some more data or the
    /// output buffer needs to be freed up before trying again.
    ///
    /// For decompression this means that more input is needed to continue or
    /// the output buffer isn't large enough to contain the result. The function
    /// can be called again after fixing both.
    BufError,

    /// Indicates that all input has been consumed and all output bytes have
    /// been written. Decompression/compression should not be called again.
    ///
    /// For decompression with zlib streams the adler-32 of the decompressed
    /// data has also been verified.
    StreamEnd,
}

/// Errors that can occur when decompressing.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(i32)]
pub enum InflateError {
    /// Decompressing this input requires a dictionary.
    NeedDict { dict_id: u32 } = 2,
    /// The [`Inflate`] is in an inconsistent state, most likely
    /// due to an invalid configuration parameter.
    StreamError = -2,
    /// The input is not a valid deflate stream.
    DataError = -3,
    /// A memory allocation failed.
    MemError = -4,
}

impl From<InflateError> for ReturnCode {
    fn from(value: InflateError) -> Self {
        match value {
            InflateError::NeedDict { .. } => ReturnCode::NeedDict,
            InflateError::StreamError => ReturnCode::StreamError,
            InflateError::DataError => ReturnCode::DataError,
            InflateError::MemError => ReturnCode::MemError,
        }
    }
}

impl InflateError {
    pub fn as_str(self) -> &'static str {
        ReturnCode::from(self).error_message_str()
    }
}

/// The state that is used to decompress an input.
pub struct Inflate {
    inner: crate::inflate::InflateStream<'static>,
    total_in: u64,
    total_out: u64,
}

impl Inflate {
    /// The amount of bytes consumed from the input so far.
    pub fn total_in(&self) -> u64 {
        self.total_in
    }

    /// The amount of decompressed bytes that have been written to the output thus far.
    pub fn total_out(&self) -> u64 {
        self.total_out
    }

    /// The error message if the previous operation failed.
    pub fn error_message(&self) -> Option<&'static str> {
        if self.inner.msg.is_null() {
            None
        } else {
            unsafe { core::ffi::CStr::from_ptr(self.inner.msg).to_str() }.ok()
        }
    }

    /// Create a new instance. This function allocates, and so it is recommended to re-use this
    /// state when possible, using [`Inflate::reset`] as needed.
    ///
    /// This function will:
    ///
    /// - decode a raw deflate stream when `expect_header = false` and `window_bits` is in the
    ///   range `8..=15`
    /// - decode a zlib header followed by a deflate stream when `expect_header = true` and
    ///   `window_bits` is in the range `8..=15`
    /// - decode a gzip header followed by a deflate stream when `expect_header = true` and
    ///   `window_bits` is in the range `16 + 8..=16 + 15`
    /// - decode either a zlib or a gzip header, followed by a deflate stream when
    ///   `expect_header = true` and `window_bits` is in the range `32 + 8..=32 + 15`
    ///
    /// `window_bits` can also be 0 to request that inflate use the window size in the
    /// zlib header of the compressed stream when using zlib.
    ///
    /// Note that when deflating a value of `window_bits = 8` is silently converted to
    /// `window_bits = 9` in most zlib implementations, and hence should be inflated using
    /// `window_bits = 9`.
    ///
    /// # Panics
    ///
    /// This function may panic when the `window_bits` and `expect_header` have values not listed above.
    pub fn new(expect_header: bool, window_bits: u8) -> Self {
        let config = InflateConfig {
            window_bits: if expect_header {
                i32::from(window_bits)
            } else {
                -i32::from(window_bits)
            },
        };

        Self {
            inner: crate::inflate::InflateStream::new(config),
            total_in: 0,
            total_out: 0,
        }
    }

    /// Reset the state to allow handling a new stream.
    pub fn reset(&mut self, zlib_header: bool) {
        let mut config = InflateConfig::default();

        if !zlib_header {
            config.window_bits = -config.window_bits;
        }

        self.total_in = 0;
        self.total_out = 0;

        crate::inflate::reset_with_config(&mut self.inner, config);
    }

    /// Decompress `input` and write all decompressed bytes into `output`,
    /// with `flush` defining some details about this.
    pub fn decompress(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        flush: InflateFlush,
    ) -> Result<Status, InflateError> {
        self.decompress_into(input, output, flush)
    }

    /// Decompress `input` and write all decompressed bytes into a potentially uninitialized `output`,
    /// with `flush` defining some details about this.
    pub fn decompress_uninit(
        &mut self,
        input: &[u8],
        output: &mut [MaybeUninit<u8>],
        flush: InflateFlush,
    ) -> Result<Status, InflateError> {
        self.decompress_into(input, output, flush)
    }

    fn decompress_into<O: WritableBytes + ?Sized>(
        &mut self,
        input: &[u8],
        output: &mut O,
        flush: InflateFlush,
    ) -> Result<Status, InflateError> {
        // Limit the length of the input and output to the maximum value of a c_uint. For larger
        // inputs, this will either complete or signal that more input and output is needed. The
        // caller should be able to handle this regardless.
        self.inner.avail_in = Ord::min(input.len(), c_uint::MAX as usize) as c_uint;
        self.inner.avail_out = Ord::min(output.len(), c_uint::MAX as usize) as c_uint;

        // zlib's ABI uses a mutable input pointer even though ordinary inflate only reads it.
        self.inner.next_in = input.as_ptr().cast_mut();
        self.inner.next_out = output.byte_ptr();

        let start_in = self.inner.next_in;
        let start_out = self.inner.next_out;

        // SAFETY: the inflate state was properly initialized and the output capability supplies a
        // writable span of `avail_out` bytes.
        let ret = unsafe { crate::inflate::inflate(&mut self.inner, flush) };

        self.total_in += (self.inner.next_in as usize - start_in as usize) as u64;
        self.total_out += (self.inner.next_out as usize - start_out as usize) as u64;

        // The caller's input and output borrows end at this method boundary.
        // Keep internal inflate state, but never retain pointers to their buffers.
        self.inner.next_in = core::ptr::null_mut();
        self.inner.next_out = core::ptr::null_mut();
        self.inner.avail_in = 0;
        self.inner.avail_out = 0;

        match ret {
            ReturnCode::Ok => Ok(Status::Ok),
            ReturnCode::StreamEnd => Ok(Status::StreamEnd),
            ReturnCode::NeedDict => Err(InflateError::NeedDict {
                dict_id: self.inner.adler as u32,
            }),
            ReturnCode::ErrNo => unreachable!("the rust API does not use files"),
            ReturnCode::StreamError => Err(InflateError::StreamError),
            ReturnCode::DataError => Err(InflateError::DataError),
            ReturnCode::MemError => Err(InflateError::MemError),
            ReturnCode::BufError => Ok(Status::BufError),
            ReturnCode::VersionError => unreachable!("the rust API does not use the version"),
        }
    }

    pub fn set_dictionary(&mut self, dictionary: &[u8]) -> Result<u32, InflateError> {
        match crate::inflate::set_dictionary(&mut self.inner, dictionary) {
            ReturnCode::Ok => Ok(self.inner.adler as u32),
            ReturnCode::StreamError => Err(InflateError::StreamError),
            ReturnCode::DataError => Err(InflateError::DataError),
            other => unreachable!("set_dictionary does not return {other:?}"),
        }
    }
}

impl Drop for Inflate {
    fn drop(&mut self) {
        let _ = crate::inflate::end(&mut self.inner);
    }
}

/// Errors that can occur when compressing.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum DeflateError {
    /// The [`Deflate`] is in an inconsistent state, most likely
    /// due to an invalid configuration parameter.
    StreamError = -2,
    /// The input is not a valid deflate stream.
    DataError = -3,
    /// A memory allocation failed.
    MemError = -4,
}

impl From<DeflateError> for ReturnCode {
    fn from(value: DeflateError) -> Self {
        match value {
            DeflateError::StreamError => ReturnCode::StreamError,
            DeflateError::DataError => ReturnCode::DataError,
            DeflateError::MemError => ReturnCode::MemError,
        }
    }
}

impl DeflateError {
    pub fn as_str(self) -> &'static str {
        ReturnCode::from(self).error_message_str()
    }
}

impl From<ReturnCode> for Result<Status, DeflateError> {
    fn from(value: ReturnCode) -> Self {
        match value {
            ReturnCode::Ok => Ok(Status::Ok),
            ReturnCode::StreamEnd => Ok(Status::StreamEnd),
            ReturnCode::NeedDict => unreachable!("compression does not use dictionary"),
            ReturnCode::ErrNo => unreachable!("the rust API does not use files"),
            ReturnCode::StreamError => Err(DeflateError::StreamError),
            ReturnCode::DataError => Err(DeflateError::DataError),
            ReturnCode::MemError => Err(DeflateError::MemError),
            ReturnCode::BufError => Ok(Status::BufError),
            ReturnCode::VersionError => unreachable!("the rust API does not use the version"),
        }
    }
}

/// The state that is used to compress an input.
pub struct Deflate {
    inner: crate::deflate::DeflateStream<'static>,
    total_in: u64,
    total_out: u64,
}

impl Deflate {
    /// The number of bytes that were read from the input.
    pub fn total_in(&self) -> u64 {
        self.total_in
    }

    /// The number of compressed bytes that were written to the output.
    pub fn total_out(&self) -> u64 {
        self.total_out
    }

    /// The error message if the previous operation failed.
    pub fn error_message(&self) -> Option<&'static str> {
        if self.inner.msg.is_null() {
            None
        } else {
            unsafe { core::ffi::CStr::from_ptr(self.inner.msg).to_str() }.ok()
        }
    }

    /// Create a new instance - this allocates so should be done with care.
    ///
    /// The `window_bits` must be in the range `8..=15`, with `15` being most common.
    pub fn new(level: i32, zlib_header: bool, window_bits: u8) -> Self {
        let config = DeflateConfig {
            window_bits: if zlib_header {
                i32::from(window_bits)
            } else {
                -i32::from(window_bits)
            },
            level,
            ..DeflateConfig::default()
        };

        Self::new_with_config(config)
    }

    /// Create a new instance with the provided [`DeflateConfig`].
    ///
    /// In most cases it is recommended to use the standard [`Deflate::new`] constructor unless
    /// tweaking `mem_level` or `strategy` is desired.
    ///
    /// This allocates, so should be done with care.
    pub fn new_with_config(config: DeflateConfig) -> Self {
        Self {
            inner: crate::deflate::DeflateStream::new(config),
            total_in: 0,
            total_out: 0,
        }
    }

    /// Prepare the instance for a new stream.
    pub fn reset(&mut self) {
        self.total_in = 0;
        self.total_out = 0;

        crate::deflate::reset(&mut self.inner);
    }

    /// Compress `input` and write compressed bytes to `output`,
    /// with `flush` controlling additional characteristics.
    pub fn compress(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        flush: DeflateFlush,
    ) -> Result<Status, DeflateError> {
        self.compress_into(input, output, flush)
    }

    /// Compress `input` and write compressed bytes to a potentially uninitialized `output`,
    /// with `flush` controlling additional characteristics.
    pub fn compress_uninit(
        &mut self,
        input: &[u8],
        output: &mut [MaybeUninit<u8>],
        flush: DeflateFlush,
    ) -> Result<Status, DeflateError> {
        self.compress_into(input, output, flush)
    }

    fn compress_into<O: WritableBytes + ?Sized>(
        &mut self,
        input: &[u8],
        output: &mut O,
        flush: DeflateFlush,
    ) -> Result<Status, DeflateError> {
        // Limit the length of the input and output to the maximum value of a c_uint. For larger
        // inputs, this will either complete or signal that more input and output is needed. The
        // caller should be able to handle this regardless.
        self.inner.avail_in = Ord::min(input.len(), c_uint::MAX as usize) as c_uint;
        self.inner.avail_out = Ord::min(output.len(), c_uint::MAX as usize) as c_uint;

        // zlib's ABI uses a mutable input pointer even though deflate only reads this input span.
        self.inner.next_in = input.as_ptr().cast_mut();
        self.inner.next_out = output.byte_ptr();

        let start_in = self.inner.next_in;
        let start_out = self.inner.next_out;

        let ret = crate::deflate::deflate(&mut self.inner, flush).into();

        self.total_in += (self.inner.next_in as usize - start_in as usize) as u64;
        self.total_out += (self.inner.next_out as usize - start_out as usize) as u64;

        // Clear these pointers so there can be no use after the caller's buffers go out of scope.
        self.inner.next_in = core::ptr::null_mut();
        self.inner.next_out = core::ptr::null_mut();
        self.inner.avail_in = 0;
        self.inner.avail_out = 0;

        ret
    }

    /// Specifies the compression dictionary to use.
    ///
    /// Returns the Adler-32 checksum of the dictionary.
    pub fn set_dictionary(&mut self, dictionary: &[u8]) -> Result<u32, DeflateError> {
        match crate::deflate::set_dictionary(&mut self.inner, dictionary) {
            ReturnCode::Ok => Ok(self.inner.adler as u32),
            ReturnCode::StreamError => Err(DeflateError::StreamError),
            other => unreachable!("set_dictionary does not return {other:?}"),
        }
    }

    /// Dynamically updates the compression level.
    ///
    /// This can be used to switch between compression levels for different
    /// kinds of data, or it can be used in conjunction with a call to [`Deflate::reset`]
    /// to reuse the compressor.
    ///
    /// This may return an error if there wasn't enough output space to complete
    /// the compression of the available input data before changing the
    /// compression level. Flushing the stream before calling this method
    /// ensures that the function will succeed on the first call.
    pub fn set_level(&mut self, level: i32) -> Result<Status, DeflateError> {
        // Clear these pointers so there can be no use after free.
        self.inner.next_in = core::ptr::null_mut();
        self.inner.next_out = core::ptr::null_mut();

        self.inner.avail_in = 0;
        self.inner.avail_out = 0;

        match crate::deflate::params(&mut self.inner, level, Default::default()) {
            ReturnCode::Ok => Ok(Status::Ok),
            ReturnCode::StreamError => Err(DeflateError::StreamError),
            ReturnCode::BufError => Ok(Status::BufError),
            other => unreachable!("set_level does not return {other:?}"),
        }
    }
}

impl Drop for Deflate {
    fn drop(&mut self) {
        let _ = crate::deflate::end(&mut self.inner);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_no_caller_buffers_attached(inflate: &Inflate) {
        assert!(inflate.inner.next_in.is_null());
        assert!(inflate.inner.next_out.is_null());
        assert_eq!(inflate.inner.avail_in, 0);
        assert_eq!(inflate.inner.avail_out, 0);
    }

    #[test]
    fn safe_inflate_detaches_buffers_with_bounded_output() -> Result<(), &'static str> {
        let source = b"quick brown fox jumps over the lazy dog; ".repeat(36);
        let mut encoder = Deflate::new(6, true, 15);
        let mut compressed = vec![0u8; source.len() + 128];
        assert_eq!(
            encoder.compress(&source, &mut compressed, DeflateFlush::Finish),
            Ok(Status::StreamEnd)
        );
        let compressed_bytes =
            usize::try_from(encoder.total_out()).map_err(|_| "compressed size exceeds usize")?;
        compressed.truncate(compressed_bytes);

        let mut decoder = Inflate::new(true, 15);
        let mut recovered = vec![0u8; source.len()];
        let mut reached_end = false;
        for _ in 0..source.len() {
            let prev_in = decoder.total_in();
            let prev_out = decoder.total_out();
            let old_in = usize::try_from(prev_in).map_err(|_| "input offset exceeds usize")?;
            let old_out = usize::try_from(prev_out).map_err(|_| "output offset exceeds usize")?;
            let end_out = (old_out + 7).min(recovered.len());
            let result = decoder.decompress(
                &compressed[old_in..],
                &mut recovered[old_out..end_out],
                InflateFlush::NoFlush,
            );
            assert_no_caller_buffers_attached(&decoder);
            assert!(matches!(
                result,
                Ok(Status::Ok | Status::BufError | Status::StreamEnd)
            ));
            if result == Ok(Status::StreamEnd) {
                reached_end = true;
                break;
            }
            assert!(
                decoder.total_in() != prev_in || decoder.total_out() != prev_out,
                "decompression made no progress with available input and output"
            );
        }
        assert!(
            reached_end,
            "decompression never reported the end of stream"
        );
        assert_eq!(
            decoder.total_in(),
            u64::try_from(compressed.len()).map_err(|_| "compressed length exceeds u64")?
        );
        assert_eq!(
            decoder.total_out(),
            u64::try_from(source.len()).map_err(|_| "plain length exceeds u64")?
        );
        assert_eq!(recovered, source);
        Ok(())
    }

    #[test]
    fn safe_inflate_dictionary_resume_after_detaching_buffers() -> Result<(), &'static str> {
        let dictionary = b"recurring input history - useful to decoder";
        let source = b"recurring input history - useful to decoder, again";
        let mut encoder = Deflate::new(6, true, 15);
        assert!(encoder.set_dictionary(dictionary).is_ok());
        let mut compressed_stream = [0u8; 256];
        assert_eq!(
            encoder.compress(source, &mut compressed_stream, DeflateFlush::Finish),
            Ok(Status::StreamEnd)
        );
        let compressed_len = usize::try_from(encoder.total_out())
            .map_err(|_| "compressed_stream size exceeds usize")?;

        let mut decoder = Inflate::new(true, 15);
        let mut recovered = [0u8; 128];
        let first = decoder.decompress(
            &compressed_stream[..compressed_len],
            &mut recovered,
            InflateFlush::NoFlush,
        );
        assert!(matches!(first, Err(InflateError::NeedDict { .. })));
        assert_no_caller_buffers_attached(&decoder);

        // The decompressor retains its format/history state, not a reference
        // to the old caller input or output slices.
        assert!(decoder.set_dictionary(dictionary).is_ok());
        let consumed = usize::try_from(decoder.total_in())
            .map_err(|_| "dictionary-header consumption exceeds usize")?;
        assert_eq!(
            decoder.decompress(
                &compressed_stream[consumed..compressed_len],
                &mut recovered,
                InflateFlush::Finish,
            ),
            Ok(Status::StreamEnd)
        );
        assert_no_caller_buffers_attached(&decoder);
        assert_eq!(&recovered[..source.len()], source);
        Ok(())
    }

    #[test]
    fn safe_inflate_drops_caller_buffer_pointers_between_calls() {
        // RFC1950 empty stream. A partial feed must not leave pointers to the
        // temporary caller slices alive while waiting for the next chunk.
        let compressed = [0x78u8, 0x9c, 0x03, 0, 0, 0, 0, 1];
        let mut inflate = Inflate::new(true, 15);
        let mut output = [0u8; 8];

        let partial = inflate.decompress(&compressed[..2], &mut output, InflateFlush::NoFlush);
        assert!(matches!(partial, Ok(Status::Ok | Status::BufError)));
        assert_no_caller_buffers_attached(&inflate);
        assert_eq!(inflate.total_in(), 2);

        assert_eq!(
            inflate.decompress(&compressed[2..], &mut output, InflateFlush::Finish),
            Ok(Status::StreamEnd),
        );
        assert_no_caller_buffers_attached(&inflate);
        assert_eq!(inflate.total_in(), 8);
        assert_eq!(inflate.total_out(), 0);

        inflate.reset(true);
        let mut bad_output = [MaybeUninit::<u8>::uninit(); 8];
        assert_eq!(
            inflate.decompress_uninit(&[0xff, 0xff], &mut bad_output, InflateFlush::NoFlush),
            Err(InflateError::DataError),
        );
        assert_no_caller_buffers_attached(&inflate);
    }
}
