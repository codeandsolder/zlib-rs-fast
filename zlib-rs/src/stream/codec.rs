//! Concrete adapter between the high-level I/O API and native codec state.
use core::fmt;

use crate::{DeflateFlush, InflateError, InflateFlush};

pub(super) const MZ_NO_FLUSH: isize = DeflateFlush::NoFlush as isize;
pub(super) const MZ_PARTIAL_FLUSH: isize = DeflateFlush::PartialFlush as isize;
pub(super) const MZ_SYNC_FLUSH: isize = DeflateFlush::SyncFlush as isize;
pub(super) const MZ_FULL_FLUSH: isize = DeflateFlush::FullFlush as isize;
pub(super) const MZ_FINISH: isize = DeflateFlush::Finish as isize;

pub(super) const MZ_DEFAULT_WINDOW_BITS: u8 = 15;

use super::mem::{CompressError, DecompressError};
use super::{Compression, FlushCompress, FlushDecompress, Status};
use crate::stream::mem::{compress_failed, decompress_failed};
use core::mem::MaybeUninit;

#[derive(Clone, Default)]
pub(super) struct ErrorMessage(Option<&'static str>);

impl ErrorMessage {
    pub(super) const fn get(&self) -> Option<&str> {
        self.0
    }
}

pub(super) struct Inflate {
    inner: crate::Inflate,
    window_bits: u8,
}

impl fmt::Debug for Inflate {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            f,
            "zlib_rs inflate internal state. total_in: {}, total_out: {}",
            self.total_in(),
            self.total_out(),
        )
    }
}

impl From<FlushDecompress> for InflateFlush {
    fn from(value: FlushDecompress) -> Self {
        match value {
            FlushDecompress::None => Self::NoFlush,
            FlushDecompress::Sync => Self::SyncFlush,
            FlushDecompress::Finish => Self::Finish,
        }
    }
}

impl Inflate {
    pub(super) fn make(zlib_header: bool, window_bits: u8) -> Self {
        Self {
            inner: crate::Inflate::new(zlib_header, window_bits),
            window_bits,
        }
    }

    pub(super) fn decompress(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        flush: FlushDecompress,
    ) -> Result<Status, DecompressError> {
        let flush = flush.into();

        let result = self.inner.decompress(input, output, flush);

        match result {
            Ok(status) => Ok(status),
            Err(InflateError::NeedDict { dict_id }) => {
                crate::stream::mem::decompress_need_dict(dict_id)
            }
            Err(_) => self.decompress_error(),
        }
    }

    pub(super) fn decompress_uninit(
        &mut self,
        input: &[u8],
        output: &mut [MaybeUninit<u8>],
        flush: FlushDecompress,
    ) -> Result<Status, DecompressError> {
        let flush = flush.into();

        let result = self.inner.decompress_uninit(input, output, flush);

        match result {
            Ok(status) => Ok(status),
            Err(InflateError::NeedDict { dict_id }) => {
                crate::stream::mem::decompress_need_dict(dict_id)
            }
            Err(_) => self.decompress_error(),
        }
    }

    pub(super) fn reset(&mut self, zlib_header: bool) {
        if self.window_bits == MZ_DEFAULT_WINDOW_BITS {
            self.inner.reset(zlib_header);
        } else {
            // zlib_rs::Inflate::reset always restores the default
            // configuration, which would discard the window size and gzip
            // mode selected at construction, so re-create the state instead.
            // Keep only the base-2 logarithm of the window size when
            // resetting a gzip decoder (window_bits + 16) to a raw stream.
            let window_bits = if zlib_header {
                self.window_bits
            } else {
                self.window_bits & 15
            };
            self.inner = crate::Inflate::new(zlib_header, window_bits);
        }
    }
}

impl Inflate {
    #[inline]
    pub(super) const fn total_in(&self) -> u64 {
        self.inner.total_in()
    }

    #[inline]
    pub(super) const fn total_out(&self) -> u64 {
        self.inner.total_out()
    }
}

impl Inflate {
    fn decompress_error<T>(&self) -> Result<T, DecompressError> {
        decompress_failed(ErrorMessage(self.inner.error_message()))
    }

    pub(super) fn set_dictionary(&mut self, dictionary: &[u8]) -> Result<u32, DecompressError> {
        self.inner.set_dictionary(dictionary).or_else(|_| self.decompress_error())
    }
}

pub(super) struct Deflate {
    inner: crate::Deflate,
}

impl fmt::Debug for Deflate {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            f,
            "zlib_rs deflate internal state. total_in: {}, total_out: {}",
            self.total_in(),
            self.total_out(),
        )
    }
}

impl From<FlushCompress> for DeflateFlush {
    fn from(value: FlushCompress) -> Self {
        match value {
            FlushCompress::None => Self::NoFlush,
            FlushCompress::Partial => Self::PartialFlush,
            FlushCompress::Sync => Self::SyncFlush,
            FlushCompress::Full => Self::FullFlush,
            FlushCompress::Finish => Self::Finish,
        }
    }
}

impl Deflate {
    pub(super) fn make(level: Compression, zlib_header: bool, window_bits: u8) -> Self {
        Self {
            inner: crate::Deflate::new(
                i32::try_from(level.level()).unwrap_or(i32::MAX),
                zlib_header,
                window_bits,
            ),
        }
    }

    pub(super) fn compress(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        flush: FlushCompress,
    ) -> Result<Status, CompressError> {
        let flush = flush.into();

        let result = self.inner.compress(input, output, flush);

        result.or_else(|_| self.compress_error())
    }

    pub(super) fn compress_uninit(
        &mut self,
        input: &[u8],
        output: &mut [MaybeUninit<u8>],
        flush: FlushCompress,
    ) -> Result<Status, CompressError> {
        let flush = flush.into();

        let result = self.inner.compress_uninit(input, output, flush);

        result.or_else(|_| self.compress_error())
    }

    pub(super) fn reset(&mut self) {
        self.inner.reset();
    }
}

impl Deflate {
    #[inline]
    pub(super) const fn total_in(&self) -> u64 {
        self.inner.total_in()
    }

    #[inline]
    pub(super) const fn total_out(&self) -> u64 {
        self.inner.total_out()
    }
}

impl Deflate {
    fn compress_error<T>(&self) -> Result<T, CompressError> {
        compress_failed(ErrorMessage(self.inner.error_message()))
    }

    pub(super) fn set_dictionary(&mut self, dictionary: &[u8]) -> Result<u32, CompressError> {
        self.inner.set_dictionary(dictionary).or_else(|_| self.compress_error())
    }

    pub(super) fn set_level(&mut self, level: Compression) -> Result<(), CompressError> {
        use crate::Status;

        let result = self.inner.set_level(i32::try_from(level.level()).unwrap_or(i32::MAX));
        match result {
            Ok(Status::Ok) => Ok(()),
            Ok(Status::BufError) => compress_failed(ErrorMessage(Some("insufficient space"))),
            Ok(Status::StreamEnd) => compress_failed(ErrorMessage(Some("unexpected stream end"))),
            Err(_) => self.compress_error(),
        }
    }
}

impl fmt::Debug for ErrorMessage {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.get().fmt(f)
    }
}
