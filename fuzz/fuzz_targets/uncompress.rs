//! Exercise the public inflate C ABI with fragmented input and growable output.
//!
//! Checksums are deliberately disabled to explore parser/error paths, not to
//! validate a decompressor oracle. Run using `--features disable-checksum`.
//! Seed `corpus/uncompress` with zlib/gzip streams to explore valid paths too.
//!
//! `cargo fuzz run uncompress --features disable-checksum`
//! Miri can examine an existing corpus with:
//! `MIRIFLAGS=-Zmiri-disable-isolation cargo miri nextest run --bin uncompress --features disable-checksum`
#![cfg_attr(not(any(miri, test)), no_main)]

use core::ffi::{c_int, c_uint};
use libfuzzer_sys::{fuzz_target, Corpus};
use libz_rs_sys::{
    gz_header, inflate, inflateEnd, inflateGetHeader, inflateInit2_, z_stream, zlibVersion,
};
use zlib_rs::{InflateFlush, ReturnCode};

fuzz_target!(|input: &[u8]| -> Corpus { run(input) });

fn run(input: &[u8]) -> Corpus {
    if input.is_empty() {
        return Corpus::Reject;
    }

    let (_, code) = inflate_payload(input);
    if code == ReturnCode::StreamEnd || cfg!(feature = "keep-invalid-in-corpus") {
        Corpus::Keep
    } else {
        Corpus::Reject
    }
}

/// Decode without depending on the width of the ABI's total_out counter.
/// Keep previously written output initialized, even across Vec reallocations.
fn inflate_payload(input: &[u8]) -> (Vec<u8>, ReturnCode) {
    let mut stream = z_stream::default();

    // SAFETY: z_stream is initialized, version and ABI size match the callee.
    let err = unsafe {
        inflateInit2_(
            &raw mut stream,
            15 + 32, // accept both zlib and gzip
            zlibVersion(),
            c_int::try_from(size_of::<z_stream>()).unwrap_or(c_int::MAX),
        )
    };
    assert_eq!(ReturnCode::from(err), ReturnCode::Ok);

    let mut extra = [0u8; 64];
    let mut name = [0u8; 64];
    let mut comment = [0u8; 64];
    let mut header = gz_header {
        text: 0,
        time: 0,
        xflags: 0,
        os: 0,
        extra: extra.as_mut_ptr(),
        extra_len: 0,
        extra_max: 64,
        name: name.as_mut_ptr(),
        name_max: 64,
        comment: comment.as_mut_ptr(),
        comm_max: 64,
        hcrc: 0,
        done: 0,
    };
    // SAFETY: the header and nested writable arrays remain alive for the
    // entire decoder lifetime, including all partial-header calls.
    let err = unsafe { inflateGetHeader(&raw mut stream, &raw mut header) };
    assert_eq!(ReturnCode::from(err), ReturnCode::Ok);

    let mut output = Vec::<u8>::new();
    let mut outcome = ReturnCode::BufError;

    'decompression: for chunk in input.chunks(64) {
        debug_assert_eq!(stream.avail_in, 0);
        stream.next_in = chunk.as_ptr().cast_mut();
        stream.avail_in = c_uint::try_from(chunk.len()).unwrap_or(c_uint::MAX);

        loop {
            if stream.avail_out == 0 {
                // Reallocation is legal only after the decoder has exhausted
                // its old output span. Vec::reserve uses len, not capacity.
                if output.len() == output.capacity()
                    && output.try_reserve(output.capacity().max(1024)).is_err()
                {
                    outcome = ReturnCode::MemError;
                    break 'decompression;
                }

                let spare = output.capacity() - output.len();
                stream.avail_out = c_uint::try_from(spare).unwrap_or(c_uint::MAX);
                // SAFETY: len <= capacity, and the next span is uniquely
                // writable for exactly avail_out bytes until the next call.
                stream.next_out = unsafe { output.as_mut_ptr().add(output.len()) };
            }

            let before_in = stream.avail_in;
            let before_out = stream.avail_out;

            // SAFETY: next_in refers to this live input chunk; next_out refers
            // to spare output capacity. Neither span moves during inflate.
            let code =
                ReturnCode::from(unsafe { inflate(&raw mut stream, InflateFlush::NoFlush as _) });

            let written = usize::try_from(before_out - stream.avail_out)
                .expect("the ABI output count fits in Rust addressable memory");
            // SAFETY: inflate initialized precisely the bytes removed from
            // avail_out. Growing len exposes only those initialized bytes.
            unsafe { output.set_len(output.len() + written) };

            match code {
                ReturnCode::StreamEnd => {
                    outcome = code;
                    break 'decompression;
                }
                ReturnCode::Ok | ReturnCode::BufError => {
                    // Drain any pending output even if this input chunk ended.
                    if stream.avail_in == 0 && stream.avail_out != 0 {
                        break;
                    }
                    if before_in == stream.avail_in && written == 0 && stream.avail_out != 0 {
                        // No input/output progress despite available spans:
                        // malformed input, not a reason to grow output.
                        outcome = ReturnCode::BufError;
                        break 'decompression;
                    }
                }
                _ => {
                    outcome = code;
                    break 'decompression;
                }
            }
        }
    }

    // End does not need the caller buffers. Do not leave pointers into a
    // potentially reallocated Vec in the stream while tearing it down.
    stream.next_in = core::ptr::null_mut();
    stream.avail_in = 0;
    stream.next_out = core::ptr::null_mut();
    stream.avail_out = 0;
    // SAFETY: successful init created one live state, released exactly once.
    let err = unsafe { inflateEnd(&raw mut stream) };
    assert_eq!(ReturnCode::from(err), ReturnCode::Ok);

    (output, outcome)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn growing_output_preserves_all_decoded_bytes() {
        let mut plain = Vec::new();
        for i in 0..120_000usize {
            plain.push(((i / 521) ^ (i / 2_131)).to_le_bytes()[0]);
        }
        for window_bits in [15, 31] {
            let config = zlib_rs::deflate::DeflateConfig {
                window_bits,
                ..Default::default()
            };
            let mut storage = vec![0; zlib_rs::deflate::compress_bound(plain.len())];
            let (compressed, code) = zlib_rs::deflate::compress_slice(&mut storage, &plain, config);
            assert_eq!(code, ReturnCode::Ok);
            assert!(compressed.len() < plain.len());

            let (decoded, status) = inflate_payload(compressed);
            assert_eq!(
                status,
                ReturnCode::StreamEnd,
                "window_bits={window_bits}, compressed_len={}",
                compressed.len()
            );
            assert_eq!(decoded, plain);
        }
    }

    #[test]
    fn truncated_stream_does_not_count_as_complete() {
        let plain = b"a short payload";
        let mut storage = vec![0; zlib_rs::deflate::compress_bound(plain.len())];
        let (compressed, code) = zlib_rs::deflate::compress_slice(
            &mut storage,
            plain,
            zlib_rs::deflate::DeflateConfig::default(),
        );
        assert_eq!(code, ReturnCode::Ok);
        assert_eq!(inflate_payload(compressed).1, ReturnCode::StreamEnd);
        assert_ne!(
            inflate_payload(&compressed[..compressed.len() - 1]).1,
            ReturnCode::StreamEnd
        );
        assert_ne!(inflate_payload(&[0xff, 0xff]).1, ReturnCode::StreamEnd);
    }

    #[test]
    fn disabled_checksum_still_consumes_trailers_and_checks_gzip_size() {
        // Disabling checksum *verification* must not drop the four bytes
        // needed to align the following gzip ISIZE field.
        let plain = b"payload with repeated characters ".repeat(3_000);
        for window_bits in [15, 31] {
            let mut storage = vec![0; zlib_rs::deflate::compress_bound(plain.len())];
            let config = zlib_rs::deflate::DeflateConfig {
                window_bits,
                ..Default::default()
            };
            let (compressed, code) = zlib_rs::deflate::compress_slice(&mut storage, &plain, config);
            assert_eq!(code, ReturnCode::Ok);

            let mut corrupted_crc = compressed.to_vec();
            let trailer_length = if window_bits == 31 { 8 } else { 4 };
            corrupted_crc[compressed.len() - trailer_length] ^= 0x80;
            let (decoded, status) = inflate_payload(&corrupted_crc);
            assert_eq!(status, ReturnCode::StreamEnd, "wrapper={window_bits}");
            assert_eq!(decoded, plain);

            if window_bits == 31 {
                let mut corrupted_size = compressed.to_vec();
                let last = corrupted_size.len() - 1;
                corrupted_size[last] ^= 0x80;
                assert_eq!(
                    inflate_payload(&corrupted_size).1,
                    ReturnCode::DataError,
                    "a valid gzip ISIZE is still required even without CRC validation",
                );
            }
        }
    }

    #[cfg(miri)]
    use {
        rstest::rstest,
        std::{fs::File, io::Read, path::PathBuf},
    };

    #[rstest]
    #[cfg(miri)]
    fn miri_corpus(#[files("${ZLIB_RS_CORPUS_DIR:-corpus/uncompress}/*")] path: PathBuf) {
        let mut input = File::open(path).unwrap();
        let mut buf = Vec::new();
        input.read_to_end(&mut buf).unwrap();
        run(&buf);
    }
}
