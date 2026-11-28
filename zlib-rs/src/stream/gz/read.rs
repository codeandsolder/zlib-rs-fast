//! Gzip readers use the same owned-buffer adapter as raw DEFLATE and zlib.

use super::{bufread, GzBuilder, GzHeader};
use crate::stream::{
    bufreader::BufReader,
    read::{BufferedSource, Reader},
    Compression,
};
use std::io::Read;

/// Gzip compressor over a reader.
pub type GzEncoder<R> = Reader<R, bufread::GzEncoder<BufReader<R>>>;
/// Gzip decoder over a reader, with configurable member handling.
pub type Decoder<R, const MULTI: bool> = Reader<R, bufread::Decoder<BufReader<R>, MULTI>>;
/// Gzip decoder that stops after a single member.
pub type GzDecoder<R> = Decoder<R, false>;
/// Gzip decoder that consumes all members.
pub type MultiGzDecoder<R> = Decoder<R, true>;

pub const fn gz_encoder<R: Read>(inner: bufread::GzEncoder<BufReader<R>>) -> GzEncoder<R> {
    Reader::from_buffered(inner)
}

impl<R: Read> GzEncoder<R> {
    /// Construct a gzip encoder with a default header.
    pub fn new(r: R, level: Compression) -> Self {
        GzBuilder::new().read(r, level)
    }
}

impl<R: Read, const MULTI: bool> Decoder<R, MULTI> {
    /// Construct a gzip decoder, parsing the first header if available.
    pub fn new(r: R) -> Self {
        Self::from_buffered(bufread::Decoder::new(BufReader::new(r)))
    }
}

impl<R, const MULTI: bool> Decoder<R, MULTI> {
    /// Return the currently decoded gzip header.
    pub const fn header(&self) -> Option<&GzHeader> {
        self.inner.header()
    }

    /// Reset the decoder and replace the underlying reader.
    pub fn reset(&mut self, r: R) -> R {
        bufread::reset_decoder_data(&mut self.inner);
        self.inner.get_mut().reset(r)
    }
}

impl<R> BufferedSource<R> for bufread::GzEncoder<BufReader<R>> {
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

impl<R, const MULTI: bool> BufferedSource<R> for bufread::Decoder<BufReader<R>, MULTI> {
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

#[cfg(test)]
#[path = "../tests/gz_read_tests.rs"]
mod tests;
