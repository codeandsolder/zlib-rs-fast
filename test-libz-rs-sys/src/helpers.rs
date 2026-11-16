use zlib_rs::{
    deflate::DeflateConfig, inflate::InflateConfig, DeflateFlush, InflateFlush, ReturnCode,
};

// we use the libz_sys but configure zlib-ng in zlib compat mode
use libz_sys as libz_ng_sys;

use core::ffi::{c_int, c_uint};

/// Cap each offered C-ABI span at both the caller's budget and `uInt::MAX`.
/// The consumed Rust budget is exact, even if `usize` exceeds native `uInt`.
fn next_c_transfer(remaining: &mut usize, limit: c_uint) -> c_uint {
    let count = c_uint::try_from(*remaining)
        .unwrap_or(c_uint::MAX)
        .min(limit);
    // A C uInt fits usize on supported std targets; count <= remaining.
    *remaining -= count as usize;
    count
}

pub fn compress_slice_ng<'a>(
    output: &'a mut [u8],
    input: &[u8],
    config: DeflateConfig,
) -> (&'a mut [u8], Result<ReturnCode, i32>) {
    compress_slice_with_flush_ng(output, input, config, DeflateFlush::Finish)
}

pub fn compress_slice_with_flush_ng<'a>(
    output: &'a mut [u8],
    input: &[u8],
    config: DeflateConfig,
    final_flush: DeflateFlush,
) -> (&'a mut [u8], Result<ReturnCode, i32>) {
    compress_slice_with_flush_ng_with_limit(output, input, config, final_flush, c_uint::MAX)
}

fn compress_slice_with_flush_ng_with_limit<'a>(
    output: &'a mut [u8],
    input: &[u8],
    config: DeflateConfig,
    final_flush: DeflateFlush,
    transfer_limit: c_uint,
) -> (&'a mut [u8], Result<ReturnCode, i32>) {
    assert_ne!(
        transfer_limit, 0,
        "C-ABI transfers must have positive capacity"
    );
    let Ok(stream_size) = c_int::try_from(core::mem::size_of::<libz_ng_sys::z_stream>()) else {
        return (&mut [], Ok(ReturnCode::StreamError));
    };
    let mut stream = libz_ng_sys::z_stream {
        next_in: input.as_ptr().cast_mut(),
        avail_in: 0, // for special logic in the first  iteration
        total_in: 0,
        next_out: output.as_mut_ptr(),
        avail_out: 0, // for special logic on the first iteration
        total_out: 0,
        msg: core::ptr::null_mut(),
        state: core::ptr::null_mut(),
        zalloc: zlib_rs::allocate::C.zalloc,
        zfree: zlib_rs::allocate::C.zfree,
        opaque: core::ptr::null_mut(),
        data_type: 0,
        adler: 0,
        reserved: 0,
    };

    // SAFETY: live writable native stream, exact checked ABI size and
    // the version string from the same library.
    let err = unsafe {
        libz_ng_sys::deflateInit2_(
            &raw mut stream,
            config.level,
            config.method as i32,
            config.window_bits,
            config.mem_level,
            config.strategy as i32,
            libz_ng_sys::zlibVersion(),
            stream_size,
        )
    };

    if err != libz_ng_sys::Z_OK {
        return (&mut [], ReturnCode::try_from(err));
    }

    let mut left = output.len();
    let mut source_len = input.len();

    let err = loop {
        if stream.avail_out == 0 {
            stream.avail_out = next_c_transfer(&mut left, transfer_limit);
        }

        if stream.avail_in == 0 {
            stream.avail_in = next_c_transfer(&mut source_len, transfer_limit);
        }

        let flush = if source_len > 0 {
            DeflateFlush::NoFlush
        } else {
            final_flush
        };

        // SAFETY: live disjoint readable/writable buffers, native lengths
        // bounded by each Rust span and the native C ABI transfer count.
        let err = unsafe { libz_ng_sys::deflate(&raw mut stream, flush as i32) };

        if err != libz_ng_sys::Z_OK {
            break err;
        }
    };

    // The C total_out counter can wrap on Windows while usize is 64-bit.
    let produced = output.len() - left - stream.avail_out as usize;

    // Preserve the primary compressor result, but also check the matching
    // release when compression actually reached a complete stream.
    // SAFETY: the initialized stream is released once while all borrowed
    // buffers remain alive and stable.
    let end_code = unsafe { libz_ng_sys::deflateEnd(&raw mut stream) };
    let return_code = if err == libz_ng_sys::Z_STREAM_END {
        ReturnCode::try_from(end_code)
    } else {
        ReturnCode::try_from(err)
    };

    (&mut output[..produced], return_code)
}

pub fn uncompress_slice_ng<'a>(
    output: &'a mut [u8],
    input: &[u8],
    config: InflateConfig,
) -> (&'a mut [u8], Result<ReturnCode, i32>) {
    uncompress_slice_ng_with_limit(output, input, config, c_uint::MAX)
}

/// Force C-ABI transfer boundaries at small sizes in tests without allocating
/// multi-gigabyte input or output buffers.
fn uncompress_slice_ng_with_limit<'a>(
    output: &'a mut [u8],
    input: &[u8],
    config: InflateConfig,
    transfer_limit: c_uint,
) -> (&'a mut [u8], Result<ReturnCode, i32>) {
    assert_ne!(
        transfer_limit, 0,
        "C-ABI transfers must have positive capacity"
    );

    let Ok(stream_size) = c_int::try_from(core::mem::size_of::<libz_ng_sys::z_stream>()) else {
        return (&mut [], Ok(ReturnCode::StreamError));
    };
    let output_len = output.len();
    let mut scratch = [0u8];
    let destination = if output_len == 0 {
        scratch.as_mut_ptr()
    } else {
        output.as_mut_ptr()
    };
    let mut stream = libz_ng_sys::z_stream {
        next_in: input.as_ptr().cast_mut(),
        avail_in: 0,
        total_in: 0,
        next_out: destination,
        avail_out: 0,
        total_out: 0,
        msg: core::ptr::null_mut(),
        state: core::ptr::null_mut(),
        zalloc: zlib_rs::allocate::C.zalloc,
        zfree: zlib_rs::allocate::C.zfree,
        opaque: core::ptr::null_mut(),
        data_type: 0,
        adler: 0,
        reserved: 0,
    };

    // SAFETY: stream is a valid writable z_stream, using the library's own
    // version and exact ABI structure size.
    let code = unsafe {
        libz_ng_sys::inflateInit2_(
            &raw mut stream,
            config.window_bits,
            libz_ng_sys::zlibVersion(),
            stream_size,
        )
    };
    if code != libz_ng_sys::Z_OK {
        return (&mut [], ReturnCode::try_from(code));
    }

    let mut output_left = if output_len == 0 { 1 } else { output_len };
    let mut input_left = input.len();

    let code = loop {
        if stream.avail_out == 0 {
            stream.avail_out = next_c_transfer(&mut output_left, transfer_limit);
        }
        if stream.avail_in == 0 {
            stream.avail_in = next_c_transfer(&mut input_left, transfer_limit);
        }

        // SAFETY: the zlib-ng implementation advances next_in/next_out only
        // within the supplied input/output spans; each refill is bounded by
        // both the remaining Rust slice budget and c_uint::MAX.
        let code = ReturnCode::try_from(unsafe {
            libz_ng_sys::inflate(&raw mut stream, InflateFlush::NoFlush as c_int)
        });
        if code != Ok(ReturnCode::Ok) {
            break code;
        }
    };

    // Do not use the C ABI total_out counter to slice Rust buffers: it may
    // wrap on targets where c_ulong is narrower than usize.
    let produced = if output_len == 0 {
        0
    } else {
        output_len - output_left - stream.avail_out as usize
    };

    // The one-byte scratch distinguishes a zero-length, valid decompression
    // from a stream that cannot fit in the caller's zero-length destination.
    if output_len == 0 && stream.total_out != 0 && code == Ok(ReturnCode::BufError) {
        output_left = 1;
    }

    let remaining_output = output_left + stream.avail_out as usize;
    // SAFETY: successful inflateInit2_ created a live zlib-ng inflate state;
    // this is the matching release, performed exactly once.
    let end_code = unsafe { libz_ng_sys::inflateEnd(&raw mut stream) };
    assert_eq!(ReturnCode::try_from(end_code), Ok(ReturnCode::Ok));

    let result = match code {
        Ok(ReturnCode::StreamEnd) => Ok(ReturnCode::Ok),
        Ok(ReturnCode::NeedDict) => Ok(ReturnCode::DataError),
        Ok(ReturnCode::BufError) if remaining_output != 0 => Ok(ReturnCode::DataError),
        other => other,
    };
    (&mut output[..produced], result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zlib_ng_reference_transfer_chunk_preserves_exact_budget() {
        for budget in [0, 1, 7, 255, 65_535, usize::MAX] {
            for limit in [1, 7, 255, c_uint::MAX] {
                let mut left = budget;
                let offered = next_c_transfer(&mut left, limit);
                assert!(offered <= limit);
                let offered_usize = usize::try_from(offered).unwrap_or(usize::MAX);
                assert!(offered_usize <= budget);
                assert_eq!(left.checked_add(offered_usize), Some(budget));
                if budget == 0 {
                    assert_eq!(offered, 0);
                } else {
                    assert!(offered > 0);
                }
            }
        }
    }

    #[test]
    fn zlib_ng_reference_zero_capacity_matches_plaintext_semantics() {
        for source in [&[][..], &b"nonempty output requires space"[..]] {
            let mut storage = vec![0u8; zlib_rs::deflate::compress_bound(source.len())];
            let (encoded, status) =
                zlib_rs::compress_slice(&mut storage, source, DeflateConfig::default());
            assert_eq!(status, ReturnCode::Ok);
            for limit in [1, 7, 255, c_uint::MAX] {
                let mut destination: [u8; 0] = [];
                let (decoded, status) = uncompress_slice_ng_with_limit(
                    &mut destination,
                    encoded,
                    InflateConfig::default(),
                    limit,
                );
                assert_eq!(decoded, []);
                assert_eq!(
                    status,
                    Ok(if source.is_empty() {
                        ReturnCode::Ok
                    } else {
                        ReturnCode::DataError
                    }),
                    "input size {}, limit={limit}",
                    source.len()
                );
            }
        }
    }

    #[test]
    fn zlib_ng_reference_inflate_refills_both_c_abi_spans() {
        let mut seed = 0x96ea_42f5u32;
        let input: Vec<u8> = (0..72_313)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                seed.to_le_bytes()[0]
            })
            .collect();
        let mut compressed_storage = vec![0; zlib_rs::deflate::compress_bound(input.len())];
        let (compressed, code) =
            compress_slice_ng(&mut compressed_storage, &input, DeflateConfig::default());
        assert_eq!(code, Ok(ReturnCode::Ok));
        assert!(
            compressed.len() > 70_000,
            "regression must span many C-ABI chunks"
        );

        for limit in [1, 7, 255, c_uint::MAX] {
            let mut decoded = vec![0xa5u8; input.len() + 16];
            let (result, status) = uncompress_slice_ng_with_limit(
                &mut decoded[..input.len()],
                compressed,
                InflateConfig::default(),
                limit,
            );
            assert_eq!(status, Ok(ReturnCode::Ok), "transfer_limit={limit}");
            assert_eq!(result, &input, "transfer_limit={limit}");
            assert_eq!(
                &decoded[input.len()..],
                &[0xa5; 16],
                "decoder wrote beyond the advertised output span, limit={limit}"
            );
        }
    }

    #[test]
    fn zlib_ng_reference_inflate_errors_are_chunking_independent() {
        let input = b"some data to test truncated streams and insufficient output";
        let mut compressed_storage = vec![0; zlib_rs::deflate::compress_bound(input.len())];
        let (compressed, code) =
            compress_slice_ng(&mut compressed_storage, input, DeflateConfig::default());
        assert_eq!(code, Ok(ReturnCode::Ok));

        for limit in [1, 3, 17, c_uint::MAX] {
            let mut too_short = [0xa5; 7 + 16];
            let (_, capacity_status) = uncompress_slice_ng_with_limit(
                &mut too_short[..7],
                compressed,
                InflateConfig::default(),
                limit,
            );
            assert_eq!(
                capacity_status,
                Ok(ReturnCode::BufError),
                "transfer_limit={limit}"
            );
            assert_eq!(&too_short[7..], &[0xa5; 16]);

            let mut output = vec![0xa5; input.len() + 1 + 16];
            let (_, truncated_status) = uncompress_slice_ng_with_limit(
                &mut output[..=input.len()],
                &compressed[..compressed.len() - 2],
                InflateConfig::default(),
                limit,
            );
            assert_eq!(
                truncated_status,
                Ok(ReturnCode::DataError),
                "transfer_limit={limit}"
            );
            assert_eq!(&output[input.len() + 1..], &[0xa5; 16]);
        }
    }

    #[test]
    fn zlib_ng_reference_deflate_refills_both_c_abi_spans() {
        // The same data must round-trip regardless of how zlib-ng is fed or
        // drained. Different output streams are allowed: chunking may change
        // compression decisions, but must never change decoded content.
        let mut seed = 0x25df_19a3u32;
        let input: Vec<u8> = (0..18_321)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                seed.to_le_bytes()[0]
            })
            .collect();

        for limit in [1, 7, 255, c_uint::MAX] {
            let capacity = zlib_rs::deflate::compress_bound(input.len());
            let mut storage = vec![0xa5; capacity + 16];
            let (compressed, code) = compress_slice_with_flush_ng_with_limit(
                &mut storage[..capacity],
                &input,
                DeflateConfig::default(),
                DeflateFlush::Finish,
                limit,
            );
            assert_eq!(code, Ok(ReturnCode::Ok), "transfer_limit={limit}");

            let mut decoded = vec![0u8; input.len()];
            let (roundtrip, code) = zlib_rs::inflate::decompress_slice(
                &mut decoded,
                compressed,
                InflateConfig::default(),
            );
            assert_eq!(code, ReturnCode::Ok, "transfer_limit={limit}");
            assert_eq!(roundtrip, &input, "transfer_limit={limit}");
            assert_eq!(
                &storage[capacity..],
                &[0xa5; 16],
                "encoder wrote beyond the advertised C output span, limit={limit}"
            );
        }
    }
}
