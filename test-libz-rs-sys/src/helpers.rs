use zlib_rs::{
    deflate::DeflateConfig, inflate::InflateConfig, DeflateFlush, InflateFlush, ReturnCode,
};

// we use the libz_sys but configure zlib-ng in zlib compat mode
use libz_sys as libz_ng_sys;

use core::ffi::{c_int, c_uint};

pub fn compress_slice_ng<'a>(
    output: &'a mut [u8],
    input: &[u8],
    config: DeflateConfig,
) -> (&'a mut [u8], ReturnCode) {
    compress_slice_with_flush_ng(output, input, config, DeflateFlush::Finish)
}

pub fn compress_slice_with_flush_ng<'a>(
    output: &'a mut [u8],
    input: &[u8],
    config: DeflateConfig,
    final_flush: DeflateFlush,
) -> (&'a mut [u8], ReturnCode) {
    compress_slice_with_flush_ng_with_limit(output, input, config, final_flush, c_uint::MAX)
}

fn compress_slice_with_flush_ng_with_limit<'a>(
    output: &'a mut [u8],
    input: &[u8],
    config: DeflateConfig,
    final_flush: DeflateFlush,
    transfer_limit: c_uint,
) -> (&'a mut [u8], ReturnCode) {
    assert_ne!(
        transfer_limit, 0,
        "C-ABI transfers must have positive capacity"
    );
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

    let err = unsafe {
        libz_ng_sys::deflateInit2_(
            &raw mut stream,
            config.level,
            config.method as i32,
            config.window_bits,
            config.mem_level,
            config.strategy as i32,
            libz_ng_sys::zlibVersion(),
            core::mem::size_of::<libz_ng_sys::z_stream>() as c_int,
        )
    };

    if err != libz_ng_sys::Z_OK {
        return (&mut [], ReturnCode::from(err));
    }

    let max = transfer_limit as usize;

    let mut left = output.len();
    let mut source_len = input.len();

    let err = loop {
        if stream.avail_out == 0 {
            stream.avail_out = Ord::min(left, max) as _;
            left -= stream.avail_out as usize;
        }

        if stream.avail_in == 0 {
            stream.avail_in = Ord::min(source_len, max) as _;
            source_len -= stream.avail_in as usize;
        }

        let flush = if source_len > 0 {
            DeflateFlush::NoFlush
        } else {
            final_flush
        };

        let err = unsafe { libz_ng_sys::deflate(&raw mut stream, flush as i32) };

        if err != libz_ng_sys::Z_OK {
            break err;
        }
    };

    // The C total_out counter can wrap on Windows while usize is 64-bit.
    let produced = output.len() - left - stream.avail_out as usize;

    // may DataError if there was insufficient output space
    let _ = unsafe { libz_ng_sys::deflateEnd(&raw mut stream) };
    let return_code: ReturnCode = match ReturnCode::from(err) {
        ReturnCode::StreamEnd => ReturnCode::Ok,
        other => other,
    };

    (&mut output[..produced], return_code)
}

pub fn uncompress_slice_ng<'a>(
    output: &'a mut [u8],
    input: &[u8],
    config: InflateConfig,
) -> (&'a mut [u8], ReturnCode) {
    uncompress_slice_ng_with_limit(output, input, config, c_uint::MAX)
}

/// Force C-ABI transfer boundaries at small sizes in tests without allocating
/// multi-gigabyte input or output buffers.
fn uncompress_slice_ng_with_limit<'a>(
    output: &'a mut [u8],
    input: &[u8],
    config: InflateConfig,
    transfer_limit: c_uint,
) -> (&'a mut [u8], ReturnCode) {
    assert_ne!(
        transfer_limit, 0,
        "C-ABI transfers must have positive capacity"
    );

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
            core::mem::size_of::<libz_ng_sys::z_stream>() as c_int,
        )
    };
    if code != libz_ng_sys::Z_OK {
        return (&mut [], ReturnCode::from(code));
    }

    let max_transfer = transfer_limit as usize;
    let mut output_left = if output_len == 0 { 1 } else { output_len };
    let mut input_left = input.len();

    let code = loop {
        if stream.avail_out == 0 {
            let count = output_left.min(max_transfer);
            stream.avail_out = c_uint::try_from(count).unwrap_or(c_uint::MAX);
            output_left -= count;
        }
        if stream.avail_in == 0 {
            let count = input_left.min(max_transfer);
            stream.avail_in = c_uint::try_from(count).unwrap_or(c_uint::MAX);
            input_left -= count;
        }

        // SAFETY: the zlib-ng implementation advances next_in/next_out only
        // within the supplied input/output spans; each refill is bounded by
        // both the remaining Rust slice budget and c_uint::MAX.
        let code = ReturnCode::from(unsafe {
            libz_ng_sys::inflate(&raw mut stream, InflateFlush::NoFlush as c_int)
        });
        if code != ReturnCode::Ok {
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
    if output_len == 0 && stream.total_out != 0 && code == ReturnCode::BufError {
        output_left = 1;
    }

    let remaining_output = output_left + stream.avail_out as usize;
    // SAFETY: successful inflateInit2_ created a live zlib-ng inflate state;
    // this is the matching release, performed exactly once.
    let end_code = unsafe { libz_ng_sys::inflateEnd(&raw mut stream) };
    assert_eq!(ReturnCode::from(end_code), ReturnCode::Ok);

    let result = match code {
        ReturnCode::StreamEnd => ReturnCode::Ok,
        ReturnCode::NeedDict => ReturnCode::DataError,
        ReturnCode::BufError if remaining_output != 0 => ReturnCode::DataError,
        other => other,
    };
    (&mut output[..produced], result)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(code, ReturnCode::Ok);
        assert!(
            compressed.len() > 70_000,
            "regression must span many C-ABI chunks"
        );

        for limit in [1, 7, 255, c_uint::MAX] {
            let mut decoded = vec![0u8; input.len()];
            let (result, status) = uncompress_slice_ng_with_limit(
                &mut decoded,
                compressed,
                InflateConfig::default(),
                limit,
            );
            assert_eq!(status, ReturnCode::Ok, "transfer_limit={limit}");
            assert_eq!(result, &input, "transfer_limit={limit}");
        }
    }

    #[test]
    fn zlib_ng_reference_inflate_errors_are_chunking_independent() {
        let input = b"some data to test truncated streams and insufficient output";
        let mut compressed_storage = vec![0; zlib_rs::deflate::compress_bound(input.len())];
        let (compressed, code) =
            compress_slice_ng(&mut compressed_storage, input, DeflateConfig::default());
        assert_eq!(code, ReturnCode::Ok);

        for limit in [1, 3, 17, c_uint::MAX] {
            let mut too_short = [0; 7];
            let (_, capacity_status) = uncompress_slice_ng_with_limit(
                &mut too_short,
                compressed,
                InflateConfig::default(),
                limit,
            );
            assert_eq!(
                capacity_status,
                ReturnCode::BufError,
                "transfer_limit={limit}"
            );

            let mut output = vec![0; input.len() + 1];
            let (_, truncated_status) = uncompress_slice_ng_with_limit(
                &mut output,
                &compressed[..compressed.len() - 2],
                InflateConfig::default(),
                limit,
            );
            assert_eq!(
                truncated_status,
                ReturnCode::DataError,
                "transfer_limit={limit}"
            );
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
            let mut storage = vec![0; zlib_rs::deflate::compress_bound(input.len())];
            let (compressed, code) = compress_slice_with_flush_ng_with_limit(
                &mut storage,
                &input,
                DeflateConfig::default(),
                DeflateFlush::Finish,
                limit,
            );
            assert_eq!(code, ReturnCode::Ok, "transfer_limit={limit}");

            let mut decoded = vec![0u8; input.len()];
            let (roundtrip, code) = zlib_rs::inflate::decompress_slice(
                &mut decoded,
                compressed,
                InflateConfig::default(),
            );
            assert_eq!(code, ReturnCode::Ok, "transfer_limit={limit}");
            assert_eq!(roundtrip, &input, "transfer_limit={limit}");
        }
    }
}
