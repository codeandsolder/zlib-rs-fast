//! Cross-check zlib-ng compression against streaming zlib-rs C-ABI inflate.
//!
//! Input and output are independently fragmented, including a one-byte window.
//! Stream completion is checked explicitly, so correct output without a final
//! checksum/trailer is not enough to pass.

use core::ffi::c_uint;
use zlib_rs::{deflate::DeflateConfig, ReturnCode};

use crate::helpers::compress_slice_ng;

/// Compress arbitrary bytes with zlib-ng, then decode with the public zlib-rs C ABI.
///
/// Chunk sizes of zero exercise the same one-byte path as chunk sizes of one;
/// C ABI transfer widths are bounded to `c_uint::MAX`.
pub fn cross_decode_chunked(data: &[u8], input_chunk: usize, output_chunk: usize) {
    let mut compressed_storage = vec![0; zlib_rs::deflate::compress_bound(data.len())];
    let (compressed, code) = compress_slice_ng(
        &mut compressed_storage,
        data,
        DeflateConfig::best_compression(),
    );
    assert_eq!(code, ReturnCode::Ok, "zlib-ng compression failed");

    let mut stream = libz_rs_sys::z_stream::default();
    // SAFETY: the stream is a valid writable z_stream, and we pass the
    // library's own ABI version and exact structure size.
    let code = unsafe {
        libz_rs_sys::inflateInit2_(
            &raw mut stream,
            15,
            libz_rs_sys::zlibVersion(),
            core::mem::size_of::<libz_rs_sys::z_stream>() as i32,
        )
    };
    assert_eq!(ReturnCode::from(code), ReturnCode::Ok);

    // The spare byte lets inflate consume an end marker/trailer after producing
    // exactly `data.len()` bytes, including the empty-stream case.
    let mut decoded = vec![0xa5; data.len().saturating_add(1)];
    let input_chunk = input_chunk.max(1).min(c_uint::MAX as usize);
    let output_chunk = output_chunk.max(1).min(c_uint::MAX as usize);

    let mut supplied = 0usize;
    let mut produced = 0usize;

    loop {
        if stream.avail_in == 0 && supplied < compressed.len() {
            let len = input_chunk.min(compressed.len() - supplied);
            stream.next_in = compressed[supplied..].as_ptr().cast_mut();
            stream.avail_in = c_uint::try_from(len).unwrap_or(c_uint::MAX);
            supplied += len;
        }

        if stream.avail_out == 0 {
            let remaining = decoded.len() - produced;
            assert_ne!(remaining, 0, "inflate produced more than expected");
            let len = output_chunk.min(remaining);
            stream.next_out = decoded[produced..].as_mut_ptr();
            stream.avail_out = c_uint::try_from(len).unwrap_or(c_uint::MAX);
        }

        let before_in = stream.avail_in;
        let before_out = stream.avail_out;

        // SAFETY: next_in and next_out point to live, nonoverlapping spans of
        // exactly avail_in/avail_out bytes; neither vector reallocates here.
        let code = unsafe { libz_rs_sys::inflate(&raw mut stream, libz_rs_sys::Z_NO_FLUSH) };
        produced += (before_out - stream.avail_out) as usize;

        match ReturnCode::from(code) {
            ReturnCode::StreamEnd => {
                break;
            }
            ReturnCode::Ok | ReturnCode::BufError => {
                if before_in == stream.avail_in && before_out == stream.avail_out {
                    // A no-progress result is only legitimate at a boundary
                    // where the next iteration can provide new input/output.
                    assert!(
                        (stream.avail_in == 0 && supplied < compressed.len())
                            || (stream.avail_out == 0 && produced < decoded.len()),
                        "inflate stalled without remaining input or output capacity"
                    );
                }
            }
            other => panic!("chunked inflate failed: {other:?}"),
        }
    }

    // SAFETY: a successful inflateInit2_ owns a live state here. This call
    // releases it exactly once after completing the loop.
    let end_code = unsafe { libz_rs_sys::inflateEnd(&raw mut stream) };
    assert_eq!(ReturnCode::from(end_code), ReturnCode::Ok);
    assert_eq!(
        supplied - stream.avail_in as usize,
        compressed.len(),
        "unconsumed compressed input"
    );
    assert_eq!(produced, data.len(), "incorrect decompressed length");
    assert_eq!(&decoded[..produced], data);
    assert_eq!(
        decoded[produced], 0xa5,
        "output beyond decoded length was modified"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_and_small_inputs_with_fragmented_output() {
        for payload in [b"".as_slice(), b"a", b"abababababab", b"gzip decoded data"] {
            for (input_chunk, output_chunk) in
                [(0, 0), (1, 1), (2, 7), (7, 2), (usize::MAX, usize::MAX)]
            {
                cross_decode_chunked(payload, input_chunk, output_chunk);
            }
        }
    }

    #[test]
    fn incompressible_input_exceeds_old_fixed_fuzzer_buffers() {
        // The old target allocated 8 KiB compressed and 32 KiB decoded
        // buffers and called inflate just once per input chunk.
        let mut seed = 0x1234_5678u32;
        let data: Vec<u8> = (0..40_123)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                seed.to_le_bytes()[0]
            })
            .collect();
        for (input_chunk, output_chunk) in [(1, 1), (13, 7), (4_096, 255)] {
            cross_decode_chunked(&data, input_chunk, output_chunk);
        }
    }
}
