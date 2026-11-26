use crate::stream::io;
use crate::stream::io::{BufRead, Read, Write};
use alloc::vec::Vec;
use core::cmp;
use core::mem;

use super::{corrupt, read_into, GzBuilder, GzHeader, GzHeaderParser};
use crate::stream::bufread;
use crate::stream::crc::CrcReader;
use crate::stream::Compression;

fn copy(into: &mut [u8], from: &[u8], pos: &mut usize) -> usize {
    let min = cmp::min(into.len(), from.len() - *pos);
    into[..min].copy_from_slice(&from[*pos..*pos + min]);
    *pos += min;
    min
}

/// A gzip streaming encoder
///
/// This structure implements a [`Read`] interface. When read from, it reads
/// uncompressed data from the underlying [`BufRead`] and provides the compressed data.
///
/// [`Read`]: https://doc.rust-lang.org/std/io/trait.Read.html
/// [`BufRead`]: https://doc.rust-lang.org/std/io/trait.BufRead.html
///
/// # Examples
///
/// ```
/// use std::io::prelude::*;
/// use std::io;
/// use zlib_rs::stream::Compression;
/// use zlib_rs::stream::bufread::GzEncoder;
/// use std::fs::File;
/// use std::io::BufReader;
///
/// // Opens sample file, compresses the contents and returns a Vector or error
/// // File wrapped in a BufReader implements BufRead
///
/// fn open_hello_world() -> io::Result<Vec<u8>> {
///     let f = File::open("examples/hello_world.txt")?;
///     let b = BufReader::new(f);
///     let mut gz = GzEncoder::new(b, Compression::fast());
///     let mut buffer = Vec::new();
///     gz.read_to_end(&mut buffer)?;
///     Ok(buffer)
/// }
/// ```
#[derive(Debug)]
pub struct GzEncoder<R> {
    inner: bufread::DeflateEncoder<CrcReader<R>>,
    header: Vec<u8>,
    pos: usize,
    eof: bool,
}

pub fn gz_encoder<R: BufRead>(header: Vec<u8>, r: R, lvl: Compression) -> GzEncoder<R> {
    let crc = CrcReader::new(r);
    GzEncoder {
        inner: bufread::DeflateEncoder::new(crc, lvl),
        header,
        pos: 0,
        eof: false,
    }
}

impl<R: BufRead> GzEncoder<R> {
    /// Creates a new encoder which will use the given compression level.
    ///
    /// The encoder is not configured specially for the emitted header. For
    /// header configuration, see the `GzBuilder` type.
    ///
    /// The data read from the stream `r` will be compressed and available
    /// through the returned reader.
    pub fn new(r: R, level: Compression) -> Self {
        GzBuilder::new().buf_read(r, level)
    }

    fn read_footer(&mut self, into: &mut [u8]) -> usize {
        copy(
            into,
            &super::footer(self.inner.get_ref().crc()),
            &mut self.pos,
        )
    }
}

impl<R> GzEncoder<R> {
    /// Acquires a reference to the underlying reader.
    pub const fn get_ref(&self) -> &R {
        self.inner.get_ref().get_ref()
    }

    /// Acquires a mutable reference to the underlying reader.
    ///
    /// The underlying reader may be mutated as long as its unread input and
    /// current position are preserved for subsequent reads by this encoder.
    ///
    /// To process a new stream, wait for this encoder to reach EOF and create a
    /// new encoder; replacing the reader directly does not reset it.
    pub const fn get_mut(&mut self) -> &mut R {
        self.inner.get_mut().get_mut()
    }

    /// Returns the underlying stream, consuming this encoder
    pub fn into_inner(self) -> R {
        self.inner.into_inner().into_inner()
    }
}

#[inline]
const fn finish(buf: [u8; 8]) -> (u32, u32) {
    let crc = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    let amt = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
    (crc, amt)
}

impl<R: BufRead> Read for GzEncoder<R> {
    fn read(&mut self, into: &mut [u8]) -> io::Result<usize> {
        if into.is_empty() {
            return Ok(0);
        }
        if self.eof {
            return Ok(self.read_footer(into));
        } else if self.pos < self.header.len() {
            // Report header progress before attempting a fallible input read.
            return Ok(copy(into, &self.header, &mut self.pos));
        }
        match self.inner.read(into) {
            Ok(0) => {
                self.eof = true;
                self.pos = 0;
                Ok(self.read_footer(into))
            }
            res => res,
        }
    }
}

impl<R: BufRead + Write> Write for GzEncoder<R> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.get_mut().write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.get_mut().flush()
    }
}

/// A decoder for a single member of a [gzip file].
///
/// This structure implements a [`Read`] interface. When read from, it reads
/// compressed data from the underlying [`BufRead`] and provides the uncompressed data.
///
/// After reading a single member of the gzip data this reader will return
/// Ok(0) even if there are more bytes available in the underlying reader.
/// If you need the following bytes, call `into_inner()` after Ok(0) to
/// recover the underlying reader.
///
/// To handle gzip files that may have multiple members, see [`MultiGzDecoder`]
/// or read more
/// [in the introduction](../index.html#about-multi-member-gzip-files).
///
/// [gzip file]: https://www.rfc-editor.org/rfc/rfc1952#page-5
/// [`Read`]: https://doc.rust-lang.org/std/io/trait.Read.html
/// [`BufRead`]: https://doc.rust-lang.org/std/io/trait.BufRead.html
///
/// # Examples
///
/// ```
/// use std::io::prelude::*;
/// use std::io;
/// # use zlib_rs::stream::Compression;
/// # use zlib_rs::stream::write::GzEncoder;
/// use zlib_rs::stream::bufread::GzDecoder;
///
/// # fn main() {
/// #   let mut e = GzEncoder::new(Vec::new(), Compression::default());
/// #   e.write_all(b"Hello World").unwrap();
/// #   let bytes = e.finish().unwrap();
/// #   println!("{}", decode_reader(bytes).unwrap());
/// # }
/// #
/// // Uncompresses a Gz Encoded vector of bytes and returns a string or error
/// // Here &[u8] implements BufRead
///
/// fn decode_reader(bytes: Vec<u8>) -> io::Result<String> {
///    let mut gz = GzDecoder::new(&bytes[..]);
///    let mut s = String::new();
///    gz.read_to_string(&mut s)?;
///    Ok(s)
/// }
/// ```
#[derive(Debug)]
pub struct Decoder<R, const MULTI: bool> {
    state: GzState,
    reader: CrcReader<bufread::DeflateDecoder<R>>,
}

#[derive(Debug)]
enum GzState {
    Header(GzHeaderParser),
    Body(GzHeader),
    Finished(GzHeader, usize, [u8; 8]),
    Err(io::Error),
    End(Option<GzHeader>),
}

pub fn reset_decoder_data<R, const MULTI: bool>(decoder: &mut Decoder<R, MULTI>) {
    decoder.state = GzState::Header(GzHeaderParser::new());
    decoder.reader.reset(); // reset CrcReader
    decoder.reader.get_mut().reset_data(); // reset DeflateDecoder
}

impl<R: BufRead, const MULTI: bool> Decoder<R, MULTI> {
    /// Creates a new decoder from the given reader, immediately parsing the
    /// gzip header.
    pub fn new(mut r: R) -> Self {
        let mut header_parser = GzHeaderParser::new();

        let state = match header_parser.parse(&mut r) {
            Ok(()) => GzState::Body(GzHeader::from(header_parser)),
            Err(ref err) if io::ErrorKind::WouldBlock == err.kind() => {
                GzState::Header(header_parser)
            }
            Err(err) => GzState::Err(err),
        };

        Self {
            state,
            reader: CrcReader::new(bufread::DeflateDecoder::new(r)),
        }
    }
}

impl<R, const MULTI: bool> Decoder<R, MULTI> {
    /// Returns the header associated with this stream, if it was valid
    pub const fn header(&self) -> Option<&GzHeader> {
        match &self.state {
            GzState::Body(header) | GzState::Finished(header, _, _) => Some(header),
            GzState::End(header) => header.as_ref(),
            _ => None,
        }
    }

    /// Acquires a reference to the underlying reader.
    pub const fn get_ref(&self) -> &R {
        self.reader.get_ref().get_ref()
    }

    /// Acquires a mutable reference to the underlying stream.
    ///
    /// The underlying reader may be mutated as long as its unread input and
    /// current position are preserved for subsequent reads by this decoder.
    ///
    /// To process a new stream, wait for this decoder to reach EOF and use
    /// [`reset`](Self::reset); replacing the reader directly does not reset it.
    pub const fn get_mut(&mut self) -> &mut R {
        self.reader.get_mut().get_mut()
    }

    /// Consumes this decoder, returning the underlying reader.
    pub fn into_inner(self) -> R {
        self.reader.into_inner().into_inner()
    }

    /// Resets the state of this decoder entirely, swapping out the input
    /// stream for another.
    ///
    /// This will reset the internal state of this decoder and replace the
    /// input stream with the one provided, returning the previous input
    /// stream. Future data read from this decoder will be the decompressed
    /// version of `r`'s data.
    pub fn reset(&mut self, r: R) -> R {
        reset_decoder_data(self);
        self.reader.get_mut().reset(r)
    }
}

impl<R: BufRead, const MULTI: bool> Read for Decoder<R, MULTI> {
    fn read(&mut self, into: &mut [u8]) -> io::Result<usize> {
        loop {
            match &mut self.state {
                GzState::Header(parser) => {
                    parser.parse(self.reader.get_mut().get_mut())?;
                    self.state = GzState::Body(GzHeader::from(mem::take(parser)));
                }
                GzState::Body(header) => {
                    if into.is_empty() {
                        return Ok(0);
                    }
                    match self.reader.read(into)? {
                        0 => {
                            self.state = GzState::Finished(mem::take(header), 0, [0; 8]);
                        }
                        n => {
                            return Ok(n);
                        }
                    }
                }
                GzState::Finished(header, pos, buf) => {
                    if *pos < buf.len() {
                        *pos += read_into(self.reader.get_mut().get_mut(), &mut buf[*pos..])?;
                    } else {
                        let (crc, amt) = finish(*buf);

                        if crc != self.reader.crc().sum() || amt != self.reader.crc().amount() {
                            self.state = GzState::End(Some(mem::take(header)));
                            return Err(corrupt());
                        } else if MULTI {
                            let is_eof = self
                                .reader
                                .get_mut()
                                .get_mut()
                                .fill_buf()
                                .map(<[u8]>::is_empty)?;

                            if is_eof {
                                self.state = GzState::End(Some(mem::take(header)));
                            } else {
                                self.reader.reset();
                                self.reader.get_mut().reset_data();
                                self.state = GzState::Header(GzHeaderParser::new());
                            }
                        } else {
                            self.state = GzState::End(Some(mem::take(header)));
                        }
                    }
                }
                GzState::Err(err) => {
                    let result = Err(mem::replace(err, io::ErrorKind::Other.into()));
                    self.state = GzState::End(None);
                    return result;
                }
                GzState::End(_) => return Ok(0),
            }
        }
    }
}

impl<R: BufRead + Write, const MULTI: bool> Write for Decoder<R, MULTI> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.get_mut().write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.get_mut().flush()
    }
}

/// Gzip decoder that consumes every concatenated member.
pub type GzDecoder<R> = Decoder<R, false>;
/// Gzip decoder that consumes every concatenated member.
pub type MultiGzDecoder<R> = Decoder<R, true>;

#[cfg(test)]
#[path = "../tests/gz_bufread_test.rs"]
mod test;
