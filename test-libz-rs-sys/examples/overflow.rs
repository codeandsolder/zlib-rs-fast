//! Exercise streaming totals beyond the 32-bit native zlib ABI counter range.
//! The full 5 GiB run is manual; unit tests cover the same control flow cheaply.

use zlib_rs::{Deflate, DeflateFlush, Inflate, InflateFlush, Status};

const TOTAL: u64 = 5 * 1024 * 1024 * 1024;
const IN_CHUNK: usize = 1 << 20;
const OUT_CHUNK: usize = 1 << 20;

/// Native aggregate totals are 64-bit; validate a single call's transfer
/// against the exact Rust span offered to the streaming API.
fn checked_progress(after: u64, before: u64, offered: usize) -> Result<usize, &'static str> {
    let delta = after
        .checked_sub(before)
        .ok_or("aggregate stream counter moved backwards")?;
    let count = usize::try_from(delta).map_err(|_| "transfer does not fit a host pointer")?;
    if count > offered {
        return Err("stream transferred more bytes than the caller offered");
    }
    Ok(count)
}

fn exercise(total: u64) -> Result<(), &'static str> {
    let zeros = vec![0u8; IN_CHUNK];
    let mut output = vec![0u8; OUT_CHUNK];

    let mut compressor =
        Deflate::new(1, true, 15).map_err(|_| "failed to initialize deflate compressor")?;
    let mut compressed = Vec::new();

    let chunk_budget = u64::try_from(IN_CHUNK).map_err(|_| "input chunk exceeds 64-bit count")?;
    let mut left = total;
    while left != 0 {
        let n = usize::try_from(left.min(chunk_budget))
            .map_err(|_| "remaining input chunk exceeds host pointer width")?;
        let mut input_pos = 0;
        while input_pos < n {
            let span = &zeros[input_pos..n];
            let input_before = compressor.total_in();
            let output_before = compressor.total_out();
            let status = compressor
                .compress(span, &mut output, DeflateFlush::NoFlush)
                .map_err(|_| "deflate failed while accepting input")?;
            if status == Status::StreamEnd {
                return Err("deflate ended before Finish");
            }

            let consumed = checked_progress(compressor.total_in(), input_before, span.len())?;
            let produced = checked_progress(compressor.total_out(), output_before, output.len())?;
            if consumed == 0 && produced == 0 {
                return Err("deflate made no progress while input remained");
            }
            input_pos += consumed;
            compressed.extend_from_slice(&output[..produced]);
        }
        left -= u64::try_from(n).map_err(|_| "input offset exceeds 64-bit counter")?;
    }

    loop {
        let output_before = compressor.total_out();
        let status = compressor
            .compress(&[], &mut output, DeflateFlush::Finish)
            .map_err(|_| "deflate failed while finishing")?;
        let produced = checked_progress(compressor.total_out(), output_before, output.len())?;
        compressed.extend_from_slice(&output[..produced]);
        if status == Status::StreamEnd {
            break;
        }
        // Including Status::BufError: retrying without consuming or producing
        // cannot change the state and otherwise loops forever.
        if produced == 0 {
            return Err("deflate finish made no progress");
        }
    }

    if compressor.total_in() != total {
        return Err("deflate total input counter mismatch");
    }
    let total_compressed = compressor.total_out();
    if usize::try_from(total_compressed).ok() != Some(compressed.len()) {
        return Err("deflate aggregate output disagrees with collected data");
    }

    let mut decompressor =
        Inflate::new(true, 15).map_err(|_| "failed to initialize inflate decoder")?;
    let mut input_pos = 0usize;
    loop {
        let input = compressed
            .get(input_pos..)
            .ok_or("inflate input cursor exceeded compressed data")?;
        let input_before = decompressor.total_in();
        let output_before = decompressor.total_out();
        let status = decompressor
            .decompress(input, &mut output, InflateFlush::NoFlush)
            .map_err(|_| "inflate failed")?;
        let consumed = checked_progress(decompressor.total_in(), input_before, input.len())?;
        let produced = checked_progress(decompressor.total_out(), output_before, output.len())?;
        input_pos += consumed;

        if output[..produced].iter().any(|&byte| byte != 0) {
            return Err("inflated output contained a nonzero byte");
        }
        if status == Status::StreamEnd {
            break;
        }
        if consumed == 0 && produced == 0 {
            return Err("inflate made no progress before stream end");
        }
    }

    if decompressor.total_in() != total_compressed || input_pos != compressed.len() {
        return Err("inflate consumed an unexpected number of compressed bytes");
    }
    if decompressor.total_out() != total {
        return Err("inflate aggregate output counter mismatch");
    }
    Ok(())
}

pub fn main() -> Result<(), &'static str> {
    exercise(TOTAL)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_progress_rejects_wrapping_and_out_of_bounds_totals() {
        assert_eq!(checked_progress(0, 0, 0), Ok(0));
        assert_eq!(checked_progress(8, 5, 3), Ok(3));
        assert!(checked_progress(4, 5, usize::MAX).is_err());
        assert!(checked_progress(8, 5, 2).is_err());
        assert!(checked_progress(u64::MAX, 0, 2).is_err());
    }

    #[test]
    fn small_stream_covers_input_finish_and_inflate_replay() -> Result<(), &'static str> {
        for n in [0, 1, IN_CHUNK - 1, IN_CHUNK, IN_CHUNK + 1] {
            let len = u64::try_from(n).map_err(|_| "test input exceeds 64-bit counter")?;
            exercise(len)?;
        }
        Ok(())
    }
}
