//! Cross-check zlib-ng compression against streaming zlib-rs C-ABI inflate.
//!
//! Input and output are independently fragmented, including one-byte windows.
//! Only a completed stream with a validated trailer is accepted.

use core::ffi::c_uint;
use zlib_rs::{deflate::DeflateConfig, ReturnCode};

use crate::helpers::compress_slice_ng;

/// Holds the initialized native inflater at a stable heap address and
/// finalizes it on every return path, including unwinding during a test.
struct NativeInflate {
    stream: Box<libz_rs_sys::z_stream>,
    active: bool,
}

impl NativeInflate {
    fn new() -> Result<Self, String> {
        let stream_size = i32::try_from(core::mem::size_of::<libz_rs_sys::z_stream>())
            .map_err(|_| "native inflate ABI struct size exceeds c_int")?;
        let mut owned = Self {
            stream: Box::new(libz_rs_sys::z_stream::default()),
            active: false,
        };
        // SAFETY: a stable boxed stream is writable, and the ABI version and
        // checked structure size originate from this exact native library.
        let code = unsafe {
            libz_rs_sys::inflateInit2_(
                &raw mut *owned.stream,
                15,
                libz_rs_sys::zlibVersion(),
                stream_size,
            )
        };
        if code != libz_rs_sys::Z_OK {
            return Err(format!("native inflate initialization failed: {code}"));
        }
        owned.active = true;
        Ok(owned)
    }

    fn end(&mut self) -> Result<(), String> {
        self.active = false;
        // SAFETY: successful initialization owns this exclusive native state.
        // Ending detaches the state once; Drop will not call end again.
        let code = unsafe { libz_rs_sys::inflateEnd(&raw mut *self.stream) };
        if code != libz_rs_sys::Z_OK {
            return Err(format!("native inflate teardown failed: {code}"));
        }
        Ok(())
    }
}

impl Drop for NativeInflate {
    fn drop(&mut self) {
        if self.active {
            // SAFETY: the guard is the only owner of an initialized stream;
            // this path handles errors and assertion unwinding.
            let _ = unsafe { libz_rs_sys::inflateEnd(&raw mut *self.stream) };
        }
    }
}

/// Decode one complete zlib frame with independently fragmented input/output.
///
/// Errors include incomplete trailers, bad native statuses, stalled progress,
/// insufficient destination capacity and inconsistent byte counts.
fn decode_chunks(
    compressed: &[u8],
    expected: &[u8],
    input_chunk: usize,
    output_chunk: usize,
) -> Result<(), String> {
    let mut state = NativeInflate::new()?;
    let stream = &mut *state.stream;

    // A spare canary byte lets the decoder consume an end marker/trailer
    // even when output is exactly the expected length, including empty input.
    let capacity = expected
        .len()
        .checked_add(1)
        .ok_or("decoded output capacity overflow")?;
    let mut decoded = vec![0xa5; capacity];
    let input_chunk = input_chunk
        .max(1)
        .min(usize::try_from(c_uint::MAX).unwrap_or(usize::MAX));
    let output_chunk = output_chunk
        .max(1)
        .min(usize::try_from(c_uint::MAX).unwrap_or(usize::MAX));
    let mut supplied = 0usize;
    let mut produced = 0usize;

    loop {
        if stream.avail_in == 0 && supplied < compressed.len() {
            let len = input_chunk.min(compressed.len() - supplied);
            stream.next_in = compressed[supplied..].as_ptr().cast_mut();
            stream.avail_in = c_uint::try_from(len).map_err(|_| "input chunk exceeds c_uint")?;
            supplied += len;
        }
        if stream.avail_out == 0 {
            let available = decoded
                .len()
                .checked_sub(produced)
                .ok_or("decoded output count exceeds buffer")?;
            if available == 0 {
                return Err("decoded output exceeds expected length".into());
            }
            let len = output_chunk.min(available);
            stream.next_out = decoded[produced..].as_mut_ptr();
            stream.avail_out = c_uint::try_from(len).map_err(|_| "output chunk exceeds c_uint")?;
        }

        let before_in = stream.avail_in;
        let before_out = stream.avail_out;

        // SAFETY: next_in and next_out point to disjoint live buffers
        // advertised with checked C transfer lengths; neither reallocates.
        let code = unsafe { libz_rs_sys::inflate(&raw mut *stream, libz_rs_sys::Z_NO_FLUSH) };
        let consumed = before_in
            .checked_sub(stream.avail_in)
            .ok_or("native inflate increased available input")?;
        let wrote = before_out
            .checked_sub(stream.avail_out)
            .ok_or("native inflate increased available output")?;
        produced = produced
            .checked_add(usize::try_from(wrote).map_err(|_| "output progress exceeds host width")?)
            .ok_or("decoded output count overflow")?;

        match ReturnCode::try_from(code) {
            Ok(ReturnCode::StreamEnd) => break,
            Ok(ReturnCode::Ok | ReturnCode::BufError) => {
                if consumed == 0 && wrote == 0 {
                    // A no-progress return is acceptable only when the next
                    // loop can provide a *different* live input/output span.
                    let can_refill = (stream.avail_in == 0 && supplied < compressed.len())
                        || (stream.avail_out == 0 && produced < decoded.len());
                    if !can_refill {
                        return Err("native inflate stalled without further input/output".into());
                    }
                }
            }
            other => return Err(format!("native inflate returned invalid status: {other:?}")),
        }
    }

    // Capture native counters *before* inflateEnd, which may invalidate fields.
    let unconsumed = usize::try_from(stream.avail_in)
        .map_err(|_| "native remaining input exceeds host width")?;
    let received = supplied
        .checked_sub(unconsumed)
        .ok_or("native input count exceeds supplied data")?;
    if received != compressed.len()
        || usize::try_from(stream.total_in).map_err(|_| "native total input too large")? != received
    {
        return Err("native inflater did not consume the entire zlib frame".into());
    }
    if usize::try_from(stream.total_out).map_err(|_| "native total output too large")? != produced {
        return Err("native total output differs from the output cursor".into());
    }
    state.end()?;

    if produced != expected.len() || decoded[..produced] != *expected {
        return Err("decoded bytes do not equal original input".into());
    }
    if decoded[produced] != 0xa5 {
        return Err("output canary was overwritten".into());
    }
    Ok(())
}

/// Compress with zlib-ng, then verify exact zlib-rs C-ABI roundtrip.
///
/// Zero-sized requested chunks deliberately exercise one-byte transfers.
/// The function returns an error rather than panicking on malformed native
/// results so the fuzz target can report the failure with its input intact.
///
/// # Errors
///
/// Returns an error if the reference encoder fails, the native decoder cannot
/// initialize or finish, byte counts exceed the advertised spans, output is
/// incorrect, or the stream ends before its checksum and trailer are verified.
pub fn cross_decode_chunked(
    data: &[u8],
    input_chunk: usize,
    output_chunk: usize,
) -> Result<(), String> {
    let mut compressed_storage = vec![0; zlib_rs::deflate::compress_bound(data.len())];
    let (compressed, code) = compress_slice_ng(
        &mut compressed_storage,
        data,
        DeflateConfig::best_compression(),
    );
    if code != Ok(ReturnCode::Ok) {
        return Err(format!("zlib-ng compression failed: {code:?}"));
    }
    decode_chunks(compressed, data, input_chunk, output_chunk)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_and_small_inputs_with_fragmented_output() -> Result<(), String> {
        for payload in [b"".as_slice(), b"a", b"abababababab", b"gzip decoded data"] {
            for (input_chunk, output_chunk) in
                [(0, 0), (1, 1), (2, 7), (7, 2), (usize::MAX, usize::MAX)]
            {
                cross_decode_chunked(payload, input_chunk, output_chunk)?;
            }
        }
        Ok(())
    }

    #[test]
    fn incompressible_input_exceeds_old_fixed_fuzzer_buffers() -> Result<(), String> {
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
            cross_decode_chunked(&data, input_chunk, output_chunk)?;
        }
        Ok(())
    }

    #[test]
    fn rejects_missing_checksum_and_extra_trailing_bytes() -> Result<(), String> {
        let source = b"the complete zlib trailer must be read";
        let mut storage = vec![0; zlib_rs::deflate::compress_bound(source.len())];
        let (compressed, status) =
            compress_slice_ng(&mut storage, source, DeflateConfig::best_compression());
        if status != Ok(ReturnCode::Ok) {
            return Err(format!("zlib-ng fixture compression failed: {status:?}"));
        }
        assert!(decode_chunks(compressed, source, 1, 1).is_ok());
        assert!(decode_chunks(&compressed[..compressed.len() - 1], source, 1, 1).is_err());
        let mut trailing = compressed.to_vec();
        trailing.push(0x99);
        assert!(decode_chunks(&trailing, source, 1, 1).is_err());
        Ok(())
    }
}
