//! Compare zlib-rs and zlib-ng native inflate over the captured gix-blame corpus.
//!
//! A small number of recorded streams end before their zlib trailer. Treat
//! their valid decoded prefixes as *partial*, never as completed frames.

use std::{error::Error, ffi::c_uint, mem::MaybeUninit};

use libz_sys as libz_ng_sys;

const CORPUS: &[u8] = include_bytes!("../src/test-data/gix-blame-readme.bin");
const OUTPUT_CAPACITY: usize = 1 << 20;

type BenchResult<T> = Result<T, Box<dyn Error>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DecodeSummary {
    complete: usize,
    partial: usize,
    output_bytes: usize,
    checksum: u32,
}

/// Parse the length-prefixed compressed frames without accepting a truncated
/// length prefix or a frame that extends past the source buffer.
fn parse_records(mut corpus: &[u8]) -> BenchResult<Vec<&[u8]>> {
    let mut records = Vec::new();
    while !corpus.is_empty() {
        let (length, rest) = corpus
            .split_at_checked(4)
            .ok_or("truncated gix-blame record length")?;
        let bytes: [u8; 4] = length.try_into()?;
        let length = usize::try_from(u32::from_le_bytes(bytes))?;
        let (frame, tail) = rest
            .split_at_checked(length)
            .ok_or("truncated gix-blame compressed frame")?;
        records.push(frame);
        corpus = tail;
    }
    Ok(records)
}

macro_rules! native_decoder {
    ($api:ident, $guard:ident, $decode:ident) => {
        struct $guard {
            raw: Box<MaybeUninit<$api::z_stream>>,
            live: bool,
        }

        impl $guard {
            fn end(mut self) -> BenchResult<()> {
                // SAFETY: successful native initialization established a live
                // state in stable boxed C storage, and this ends it once.
                let status = unsafe { $api::inflateEnd(self.raw.as_mut_ptr()) };
                self.live = false;
                if status != $api::Z_OK {
                    return Err(format!("native inflate teardown failed: {status}").into());
                }
                Ok(())
            }
        }

        impl Drop for $guard {
            fn drop(&mut self) {
                if self.live {
                    // SAFETY: the guard owns the initialized C state, including
                    // on early error or panic; no other owner can end it.
                    let _ = unsafe { $api::inflateEnd(self.raw.as_mut_ptr()) };
                }
            }
        }

        fn $decode(records: &[&[u8]]) -> BenchResult<DecodeSummary> {
            let output_capacity = c_uint::try_from(OUTPUT_CAPACITY)?;
            let stream_size = i32::try_from(core::mem::size_of::<$api::z_stream>())?;
            let mut raw = Box::new(MaybeUninit::<$api::z_stream>::zeroed());
            // SAFETY: boxed writable stream storage is stable and the native
            // version and ABI size originate from this exact backend.
            let status = unsafe {
                $api::inflateInit2_(raw.as_mut_ptr(), 15, $api::zlibVersion(), stream_size)
            };
            if status != $api::Z_OK {
                return Err(format!("native inflateInit2_ failed: {status}").into());
            }

            let mut guard = $guard { raw, live: true };
            // SAFETY: successful inflateInit2_ fully initialized native storage.
            let stream = unsafe { guard.raw.assume_init_mut() };
            let mut output = vec![0u8; OUTPUT_CAPACITY];
            let mut checksum = crc32fast::Hasher::new();
            let mut complete = 0usize;
            let mut partial = 0usize;
            let mut output_bytes = 0usize;

            for (index, input) in records.iter().enumerate() {
                stream.next_in = input.as_ptr().cast_mut();
                stream.avail_in = c_uint::try_from(input.len())?;
                stream.next_out = output.as_mut_ptr();
                stream.avail_out = output_capacity;

                let ended = loop {
                    let old_input = stream.avail_in;
                    let old_output = stream.avail_out;
                    // SAFETY: these are disjoint live input/output slices with
                    // exactly the advertised C counts; reset separates frames.
                    let code = unsafe { $api::inflate(stream, $api::Z_NO_FLUSH) };
                    if stream.avail_in > old_input || stream.avail_out > old_output {
                        return Err(format!("frame {index}: native cursor moved backward").into());
                    }
                    if code == $api::Z_STREAM_END {
                        break true;
                    }
                    if code != $api::Z_OK && code != $api::Z_BUF_ERROR {
                        return Err(format!("frame {index}: invalid inflate status {code}").into());
                    }
                    if stream.avail_out == 0 {
                        return Err(format!("frame {index}: output exceeds buffer").into());
                    }
                    if stream.avail_in == 0 {
                        // The corpus includes partial streams with valid prefix
                        // output. Do not pretend that their trailers were checked.
                        break false;
                    }
                    if (old_input, old_output) == (stream.avail_in, stream.avail_out)
                        || code == $api::Z_BUF_ERROR
                    {
                        return Err(format!("frame {index}: native inflater stalled").into());
                    }
                };

                if stream.avail_in != 0 || usize::try_from(stream.total_in)? != input.len() {
                    return Err(format!("frame {index}: unconsumed input").into());
                }
                let written = usize::try_from(stream.total_out)?;
                let remaining = usize::try_from(stream.avail_out)?;
                if written > output.len() || written != output.len() - remaining {
                    return Err(format!("frame {index}: inconsistent output length").into());
                }
                complete += usize::from(ended);
                partial += usize::from(!ended);
                output_bytes = output_bytes
                    .checked_add(written)
                    .ok_or("decoded byte total overflow")?;
                checksum.update(&[u8::from(ended)]);
                checksum.update(&written.to_le_bytes());
                checksum.update(&output[..written]);

                // SAFETY: this stream was successfully initialized and remains
                // exclusively borrowed; reset is valid after completed/partial
                // data, provided the next frame starts from a new zlib header.
                let status = unsafe { $api::inflateReset(stream) };
                if status != $api::Z_OK {
                    return Err(format!("frame {index}: inflateReset failed: {status}").into());
                }
            }

            guard.end()?;
            Ok(DecodeSummary {
                complete,
                partial,
                output_bytes,
                checksum: checksum.finalize(),
            })
        }
    };
}

native_decoder!(libz_ng_sys, NgStream, decode_ng);
native_decoder!(libz_rs_sys, RsStream, decode_rs);

fn run() -> BenchResult<()> {
    let mut args = std::env::args().skip(1);
    let mode = args
        .next()
        .ok_or("usage: gix-blame <ng|rs> [repetitions]")?;
    let reps = args.next().map_or(Ok(100), |s| s.parse::<usize>())?;
    if reps == 0 || args.next().is_some() {
        return Err("repetitions must be positive, with no extra arguments".into());
    }
    let records = parse_records(CORPUS)?;
    let decode: fn(&[&[u8]]) -> BenchResult<DecodeSummary> = match mode.as_str() {
        "ng" => decode_ng,
        "rs" => decode_rs,
        _ => return Err("expected ng or rs backend".into()),
    };
    let mut previous = None;
    for _ in 0..reps {
        let summary = decode(&records)?;
        if let Some(expected) = previous {
            if summary != expected {
                return Err("non-deterministic decoding across repetitions".into());
            }
        }
        previous = Some(summary);
    }
    println!("{mode}: {} records x {reps}; {previous:?}", records.len());
    Ok(())
}

fn main() -> BenchResult<()> {
    run()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn framing_rejects_both_truncated_lengths_and_records() {
        assert!(parse_records(&[]).is_ok());
        assert!(parse_records(&[1, 0, 0]).is_err());
        assert!(parse_records(&[5, 0, 0, 0, 1, 2]).is_err());
        assert_eq!(
            parse_records(&[2, 0, 0, 0, 0xaa, 0xbb])
                .map(|frames| frames.len())
                .ok(),
            Some(1)
        );
    }

    #[test]
    fn captured_corpus_distinguishes_complete_and_partial_streams() -> BenchResult<()> {
        let records = parse_records(CORPUS)?;
        assert_eq!(records.len(), 4_674);
        let ng = decode_ng(&records)?;
        let rs = decode_rs(&records)?;
        assert_eq!(ng, rs, "native backends decoded different record contents");
        assert_eq!(ng.complete, 4_656);
        assert_eq!(ng.partial, 18);
        assert_eq!(ng.output_bytes, 3_067_074);
        Ok(())
    }
}
