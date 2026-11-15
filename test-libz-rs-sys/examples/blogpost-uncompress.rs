//! Native zlib-rs versus zlib-ng decompression/checksum comparison.
//! Keep C ABI entry points visible for assembly analysis and equivalence tests.

use std::error::Error;
use std::ffi::{c_uint, c_ulong};

use libz_sys as libz_ng_sys;

#[path = "shared/native_uncompress.rs"]
mod native_uncompress;

use native_uncompress::{uncompress_ng, uncompress_rs};

type CliResult<T> = Result<T, Box<dyn Error>>;

const OUTPUT_CAPACITY: usize = 1 << 28;
const SILESIA_ZLIB: &[u8] = include_bytes!("../../silesia-small.tar.gz");

macro_rules! backend {
    ($api:ident, $chunked:ident, $adler:ident) => {
        fn $chunked(output: &mut [u8], input: &[u8], chunk_size: usize) -> CliResult<usize> {
            use $api as ffi;

            if chunk_size == 0 {
                return Err("input chunk size must be positive".into());
            }
            // Preflight all fallible C ABI capacities before acquiring native state.
            let _ = c_uint::try_from(chunk_size)?;
            let output_capacity = c_uint::try_from(output.len())?;
            let stream_size = i32::try_from(core::mem::size_of::<ffi::z_stream>())?;

            let mut storage = core::mem::MaybeUninit::<ffi::z_stream>::zeroed();
            // SAFETY: live writable z_stream storage and the same library's
            // version/structure size; only successful initialization is dereferenced.
            let init = unsafe {
                ffi::inflateInit2_(
                    storage.as_mut_ptr(),
                    zlib_rs::inflate::InflateConfig::default().window_bits,
                    ffi::zlibVersion(),
                    stream_size,
                )
            };
            if init != ffi::Z_OK {
                return Err(format!("native inflate initialization failed: {init}").into());
            }
            // SAFETY: successful inflateInit2_ populated the stream structure.
            let stream = unsafe { storage.assume_init_mut() };
            stream.next_out = output.as_mut_ptr();
            stream.avail_out = output_capacity;
            // The input trailer can remain unread after the advertised output is
            // exactly full. A live scratch byte permits completion without
            // handing the native decoder any extra caller-owned output space.
            let mut scratch = [0u8; 1];
            let mut using_scratch = false;

            let result = (|| -> Result<usize, String> {
                let mut finished = false;
                for chunk in input.chunks(chunk_size) {
                    stream.next_in = chunk.as_ptr().cast_mut();
                    stream.avail_in = c_uint::try_from(chunk.len())
                        .map_err(|_| "input chunk exceeds native C unsigned count")?;
                    loop {
                        if stream.avail_out == 0 && !using_scratch {
                            stream.next_out = scratch.as_mut_ptr();
                            stream.avail_out = 1;
                            using_scratch = true;
                        }
                        let before = (stream.avail_in, stream.avail_out);
                        // SAFETY: the input chunk and output slice remain live and disjoint,
                        // native counts describe their exact remaining spans, and the C
                        // inflater exclusively owns the initialized stream until teardown.
                        let code = unsafe {
                            ffi::inflate(&raw mut *stream, zlib_rs::InflateFlush::NoFlush as i32)
                        };
                        if using_scratch && stream.avail_out == 0 {
                            return Err("decompressed output exceeds destination capacity".into());
                        }
                        if code == ffi::Z_STREAM_END {
                            finished = true;
                            break;
                        }
                        if code != ffi::Z_OK {
                            return Err(format!("native inflate failed with status {code}"));
                        }
                        if stream.avail_in == 0 {
                            break;
                        }
                        if before == (stream.avail_in, stream.avail_out) {
                            return Err("native inflate made no input/output progress".into());
                        }
                    }
                    if finished {
                        break;
                    }
                }
                if !finished {
                    return Err("compressed input ended without a complete zlib stream".into());
                }
                let produced = usize::try_from(stream.total_out)
                    .map_err(|_| "native output counter exceeds pointer width")?;
                if produced > output.len() {
                    return Err(
                        "native inflater output counter exceeds destination capacity".into(),
                    );
                }
                Ok(produced)
            })();

            // SAFETY: successful initialization above allocated the native state;
            // teardown is called exactly once, including when decoding fails.
            let end = unsafe { ffi::inflateEnd(&raw mut *stream) };
            if end != ffi::Z_OK {
                return Err(format!("native inflate teardown failed: {end}").into());
            }
            result.map_err(Into::into)
        }

        fn $adler(input: &[u8], chunk_size: usize) -> CliResult<c_ulong> {
            use $api as ffi;

            if chunk_size == 0 {
                return Err("checksum chunk size must be positive".into());
            }
            let _ = c_uint::try_from(chunk_size)?;
            // SAFETY: zlib defines adler32(0, NULL, 0) as the initializer.
            let mut checksum = unsafe { ffi::adler32(0, core::ptr::null(), 0) };
            for chunk in input.chunks(chunk_size) {
                let len = c_uint::try_from(chunk.len())?;
                // SAFETY: each input chunk is live, readable for len bytes,
                // and the cumulative checksum is a plain native scalar.
                checksum = unsafe { ffi::adler32(checksum, chunk.as_ptr(), len) };
            }
            Ok(checksum)
        }
    };
}

backend!(libz_ng_sys, chunked_ng, adler_ng);
backend!(libz_rs_sys, chunked_rs, adler_rs);

fn parse_chunk_size(value: &str) -> CliResult<usize> {
    let log: u32 = value.parse()?;
    let size = 1usize
        .checked_shl(log)
        .ok_or("chunk-size exponent exceeds the host pointer width")?;
    let _ = c_uint::try_from(size)?;
    Ok(size)
}

fn run() -> CliResult<()> {
    let mut args = std::env::args().skip(1);
    let mode = args.next().ok_or(
        "usage: blogpost-uncompress <ng|rs|ng-chunked|rs-chunked|ng-adler|rs-adler> [chunk-log] [input]",
    )?;

    match mode.as_str() {
        "ng" | "rs" => {
            let path = args.next().ok_or("missing compressed input path")?;
            let input = std::fs::read(path)?;
            let mut output = vec![0u8; OUTPUT_CAPACITY];
            let produced = if mode == "ng" {
                uncompress_ng(&mut output, &input)?
            } else {
                uncompress_rs(&mut output, &input)?
            };
            println!("decoded {produced} bytes");
        }
        "ng-chunked" | "rs-chunked" => {
            let log = args.next().ok_or("missing chunk-size exponent")?;
            let size = parse_chunk_size(&log)?;
            let mut output = vec![0u8; OUTPUT_CAPACITY];
            let produced = if mode == "ng-chunked" {
                chunked_ng(&mut output, SILESIA_ZLIB, size)?
            } else {
                chunked_rs(&mut output, SILESIA_ZLIB, size)?
            };
            println!("decoded {produced} bytes");
        }
        "ng-adler" | "rs-adler" => {
            let log = args.next().ok_or("missing chunk-size exponent")?;
            let size = parse_chunk_size(&log)?;
            let path = args.next().ok_or("missing checksum input path")?;
            let input = std::fs::read(path)?;
            let checksum = if mode == "ng-adler" {
                adler_ng(&input, size)?
            } else {
                adler_rs(&input, size)?
            };
            println!("adler32: {checksum:08x}");
        }
        _ => return Err(format!("unknown comparison backend or operation: {mode}").into()),
    }
    if args.next().is_some() {
        return Err("unexpected additional arguments".into());
    }
    Ok(())
}

fn main() -> CliResult<()> {
    run()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compressed_fixture() -> CliResult<(Vec<u8>, Vec<u8>)> {
        let original = b"native ABI input refill/output cursor regression".repeat(7);
        let mut storage = vec![0u8; zlib_rs::deflate::compress_bound(original.len())];
        let (encoded, status) = zlib_rs::compress_slice(
            &mut storage,
            &original,
            zlib_rs::deflate::DeflateConfig::default(),
        );
        if status != zlib_rs::ReturnCode::Ok {
            return Err(format!("fixture compression failed: {status:?}").into());
        }
        Ok((encoded.to_vec(), original))
    }

    #[test]
    fn native_reference_streams_complete_at_every_input_chunk_boundary() -> CliResult<()> {
        let (compressed, original) = compressed_fixture()?;
        for size in [1, 2, 7, 31, compressed.len()] {
            for decode in [chunked_ng, chunked_rs] {
                let mut output = vec![0xa5u8; original.len() + 16];
                let produced = decode(&mut output[..original.len()], &compressed, size)?;
                assert_eq!(produced, original.len());
                assert_eq!(&output[..produced], original.as_slice());
                assert_eq!(&output[original.len()..], &[0xa5; 16]);

                assert!(decode(&mut output, &compressed[..compressed.len() - 1], size).is_err());
                assert!(decode(&mut output[..3], &compressed, size).is_err());
            }
        }
        Ok(())
    }

    #[test]
    fn native_one_shot_and_adler_match_reference() -> CliResult<()> {
        let (compressed, original) = compressed_fixture()?;
        for decode in [uncompress_ng, uncompress_rs] {
            let mut output = vec![0xa5u8; original.len() + 16];
            let produced = decode(&mut output[..original.len()], &compressed)?;
            assert_eq!(&output[..produced], original.as_slice());
            assert_eq!(&output[original.len()..], &[0xa5; 16]);
            assert!(decode(&mut output[..3], &compressed).is_err());
        }
        for size in [1, 7, original.len()] {
            assert_eq!(adler_ng(&original, size)?, adler_rs(&original, size)?);
        }
        Ok(())
    }

    #[test]
    fn rejects_oversized_or_zero_chunk_sizes() {
        assert!(parse_chunk_size("999").is_err());
        assert!(chunked_ng(&mut [0u8; 5], &[], 0).is_err());
        assert!(adler_rs(&[], 0).is_err());
    }
}
