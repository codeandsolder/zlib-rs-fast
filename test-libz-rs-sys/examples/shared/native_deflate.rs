//! Shared direct zlib-compatible C ABI driver for native compressor comparisons.

use std::error::Error;
use std::ffi::{c_int, c_uint};

use libz_sys as libz_ng_sys;

pub type NativeResult<T> = Result<T, Box<dyn Error>>;

fn offer_native_chunk(remaining: &mut usize, limit: c_uint) -> NativeResult<c_uint> {
    let offered = c_uint::try_from(*remaining)
        .unwrap_or(c_uint::MAX)
        .min(limit);
    *remaining -= usize::try_from(offered)?;
    Ok(offered)
}

macro_rules! define_native_compressor {
    ($name:ident, $with_limit:ident, $api:ident, $alloc:expr, $free:expr) => {
        pub fn $name(output: &mut [u8], input: &[u8], level: i32) -> NativeResult<usize> {
            $with_limit(output, input, level, c_uint::MAX)
        }

        fn $with_limit(
            output: &mut [u8],
            input: &[u8],
            level: i32,
            transfer_limit: c_uint,
        ) -> NativeResult<usize> {
            use $api as ffi;
            if transfer_limit == 0 {
                return Err("native transfer limit must be positive".into());
            }
            let stream_size = c_int::try_from(core::mem::size_of::<ffi::z_stream>())?;
            let mut stream = ffi::z_stream {
                next_in: input.as_ptr().cast_mut(),
                avail_in: 0,
                total_in: 0,
                next_out: output.as_mut_ptr(),
                avail_out: 0,
                total_out: 0,
                msg: core::ptr::null_mut(),
                state: core::ptr::null_mut(),
                zalloc: $alloc,
                zfree: $free,
                opaque: core::ptr::null_mut(),
                data_type: 0,
                adler: 0,
                reserved: 0,
            };
            // SAFETY: writable live C structure, exact checked ABI size and
            // version from the matching backend; both I/O buffers remain live.
            let init = unsafe {
                ffi::deflateInit2_(
                    &raw mut stream,
                    level,
                    zlib_rs::c_api::Z_DEFLATED,
                    15,
                    8,
                    zlib_rs::c_api::Z_DEFAULT_STRATEGY,
                    ffi::zlibVersion(),
                    stream_size,
                )
            };
            if init != ffi::Z_OK {
                return Err(format!("native deflate initialization failed: {init}").into());
            }

            let mut output_left = output.len();
            let mut input_left = input.len();
            let encoded = (|| -> NativeResult<usize> {
                loop {
                    if stream.avail_out == 0 {
                        stream.avail_out =
                            offer_native_chunk(&mut output_left, transfer_limit)?;
                    }
                    if stream.avail_in == 0 {
                        stream.avail_in = offer_native_chunk(&mut input_left, transfer_limit)?;
                    }
                    let flush = if input_left == 0 {
                        zlib_rs::DeflateFlush::Finish
                    } else {
                        zlib_rs::DeflateFlush::NoFlush
                    };
                    let before = (stream.avail_in, stream.avail_out);
                    // SAFETY: input/output spans are disjoint and stay allocated;
                    // native avail counts never exceed the original Rust spans.
                    // The stream was initialized by deflateInit2_ above.
                    let code = unsafe { ffi::deflate(&raw mut stream, flush as c_int) };
                    if code == ffi::Z_STREAM_END {
                        if input_left != 0 || stream.avail_in != 0 {
                            return Err("native compressor ended before consuming all input".into());
                        }
                        let unused = usize::try_from(stream.avail_out)?;
                        return output
                            .len()
                            .checked_sub(output_left)
                            .and_then(|value| value.checked_sub(unused))
                            .ok_or_else(|| "native output cursors exceeded destination bounds".into());
                    }
                    if code != ffi::Z_OK {
                        return Err(format!("native deflate failed with status {code}").into());
                    }
                    if output_left == 0 && stream.avail_out == 0 {
                        return Err("native deflate exhausted output capacity".into());
                    }
                    if before == (stream.avail_in, stream.avail_out) {
                        return Err("native deflate made no input/output progress".into());
                    }
                }
            })();
            // SAFETY: successful deflateInit2_ installed the initialized C state;
            // this matching release happens exactly once on every exit path.
            let end = unsafe { ffi::deflateEnd(&raw mut stream) };
            if end != ffi::Z_OK && encoded.is_ok() {
                return Err(format!("native deflate teardown failed: {end}").into());
            }
            encoded
        }
    };
}

define_native_compressor!(
    compress_ng,
    compress_ng_with_limit,
    libz_ng_sys,
    zlib_rs::allocate::C.zalloc,
    zlib_rs::allocate::C.zfree
);
define_native_compressor!(
    compress_rs,
    compress_rs_with_limit,
    libz_rs_sys,
    Some(zlib_rs::allocate::C.zalloc),
    Some(zlib_rs::allocate::C.zfree)
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_native_compressors_round_trip_at_all_transfer_boundaries() -> NativeResult<()> {
        let input: Vec<u8> = (0..3_001u32)
            .map(|i| i.wrapping_mul(37).to_le_bytes()[0])
            .collect();
        for limit in [1, 7, 255, c_uint::MAX] {
            for compress in [compress_ng_with_limit, compress_rs_with_limit] {
                let capacity = zlib_rs::deflate::compress_bound(input.len());
                let mut storage = vec![0xa5; capacity + 16];
                let produced = compress(&mut storage[..capacity], &input, 6, limit)?;
                let mut decoded = vec![0u8; input.len()];
                let (plaintext, status) = zlib_rs::decompress_slice(
                    &mut decoded,
                    &storage[..produced],
                    zlib_rs::inflate::InflateConfig::default(),
                );
                assert_eq!(status, zlib_rs::ReturnCode::Ok);
                assert_eq!(plaintext, input);
                assert_eq!(&storage[capacity..], &[0xa5; 16]);

                assert!(compress(&mut storage[..3], &input, 6, limit).is_err());
                assert!(compress(&mut storage[..capacity], &input, -99, limit).is_err());
                assert!(compress(&mut storage[..capacity], &input, 6, 0).is_err());
            }
        }
        Ok(())
    }

    #[test]
    fn empty_input_creates_valid_zlib_stream() -> NativeResult<()> {
        for compress in [compress_rs, compress_ng] {
            let mut compressed = [0xa5u8; 64];
            let written = compress(&mut compressed, &[], 6)?;
            let mut restored = [0u8; 1];
            let (plain, status) = zlib_rs::decompress_slice(
                &mut restored,
                &compressed[..written],
                zlib_rs::inflate::InflateConfig::default(),
            );
            assert_eq!(status, zlib_rs::ReturnCode::Ok);
            assert_eq!(plain, &[]);
        }
        Ok(())
    }
}
