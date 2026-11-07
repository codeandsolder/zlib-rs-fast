//! Focused real-corpus throughput benchmark for ordinary and callback-based inflate.
//!
//! Usage:
//!
//!     cargo run --release -p test-libz-rs-sys --example inflate_fast_bench --
//!       /path/to/uncompressed-corpus [repetitions] [regular|callback-N]
//!
//! The input corpus is compressed once outside the timed region. Each sample
//! initializes a fresh raw-DEFLATE decoder; correctness is checked after every
//! timed run. No files or benchmark output are written by this program.
use core::ffi::{c_int, c_uint, c_void};
use std::{hint::black_box, mem::MaybeUninit, time::Instant};

use libz_rs_sys::{
    inflate, inflateBack, inflateBackEnd, inflateBackInit_, inflateEnd, inflateInit2_, z_stream,
    zlibVersion, Z_FINISH, Z_OK, Z_STREAM_END,
};
use zlib_rs::{
    deflate::{compress_bound, compress_slice, DeflateConfig},
    ReturnCode,
};

const WINDOW_BITS: i32 = 15;
const WINDOW_SIZE: usize = 1 << WINDOW_BITS;

struct CallbackContext {
    compressed: *const u8,
    compressed_len: usize,
    next: usize,
    chunk_size: usize,
    output: *mut u8,
    output_len: usize,
    written: usize,
}

unsafe extern "C" fn input_callback(desc: *mut c_void, next: *mut *const u8) -> c_uint {
    // SAFETY: inflateBack invokes callbacks synchronously with the exclusive
    // descriptor pointer supplied by the benchmark, while the input is live.
    let ctx = unsafe { &mut *desc.cast::<CallbackContext>() };
    let count = (ctx.compressed_len - ctx.next).min(ctx.chunk_size);
    // SAFETY: offset remains within the owned compressed Vec allocation and
    // the callback's output-pointer slot is writable.
    unsafe { *next = ctx.compressed.add(ctx.next) };
    ctx.next += count;
    c_uint::try_from(count).unwrap_or_default()
}

unsafe extern "C" fn output_callback(desc: *mut c_void, src: *mut u8, count: c_uint) -> c_int {
    // SAFETY: the output callback is synchronous, with an exclusive descriptor
    // and a readable initialized prefix of the decode window.
    let ctx = unsafe { &mut *desc.cast::<CallbackContext>() };
    let count = usize::try_from(count).unwrap_or(usize::MAX);
    assert!(count <= ctx.output_len - ctx.written);
    // SAFETY: source is valid for count bytes for this callback, and the
    // destination is a disjoint range within the preallocated output buffer.
    unsafe { core::ptr::copy_nonoverlapping(src, ctx.output.add(ctx.written), count) };
    ctx.written += count;
    0
}

type BenchResult<T> = Result<T, Box<dyn std::error::Error>>;

fn decode_regular(compressed: &[u8], output: &mut [u8]) -> BenchResult<usize> {
    let mut raw = MaybeUninit::<z_stream>::zeroed();
    let stream_size = c_int::try_from(core::mem::size_of::<z_stream>())?;
    // SAFETY: the stream allocation, ABI version and struct size are valid.
    let status =
        unsafe { inflateInit2_(raw.as_mut_ptr(), -WINDOW_BITS, zlibVersion(), stream_size) };
    assert_eq!(status, Z_OK);
    // SAFETY: successful init initialized the C-compatible stream.
    let stream = unsafe { raw.assume_init_mut() };
    stream.next_in = compressed.as_ptr().cast_mut();
    stream.avail_in = c_uint::try_from(compressed.len())?;
    stream.next_out = output.as_mut_ptr();
    stream.avail_out = c_uint::try_from(output.len())?;
    // SAFETY: both disjoint buffer spans live through inflate and inflateEnd.
    let status = unsafe { inflate(stream, Z_FINISH) };
    assert_eq!(status, Z_STREAM_END);
    let produced = usize::try_from(stream.total_out)?;
    // SAFETY: the stream was successfully initialized above and ends exactly once.
    let status = unsafe { inflateEnd(stream) };
    assert_eq!(status, Z_OK);
    Ok(produced)
}

fn decode_callback(
    compressed: &[u8],
    output: &mut [u8],
    window: &mut [u8],
    chunk_size: usize,
) -> BenchResult<usize> {
    assert_eq!(window.len(), WINDOW_SIZE);
    let mut raw = MaybeUninit::<z_stream>::zeroed();
    let stream_size = c_int::try_from(core::mem::size_of::<z_stream>())?;
    // SAFETY: the stream and scratch window are live and uniquely writable.
    let status = unsafe {
        inflateBackInit_(
            raw.as_mut_ptr(),
            WINDOW_BITS,
            window.as_mut_ptr(),
            zlibVersion(),
            stream_size,
        )
    };
    assert_eq!(status, Z_OK);
    // SAFETY: successful init initialized the stream.
    let stream = unsafe { raw.assume_init_mut() };
    let first = chunk_size.min(compressed.len());
    stream.next_in = compressed.as_ptr().cast_mut();
    stream.avail_in = c_uint::try_from(first)?;

    let mut context = CallbackContext {
        compressed: compressed.as_ptr(),
        compressed_len: compressed.len(),
        next: first,
        chunk_size,
        output: output.as_mut_ptr(),
        output_len: output.len(),
        written: 0,
    };
    let desc = (&raw mut context).cast::<c_void>();
    // SAFETY: the descriptors and their input/output spans remain valid and
    // non-overlapping until the synchronous inflateBack invocation returns.
    let status = unsafe {
        inflateBack(
            stream,
            Some(input_callback),
            desc,
            Some(output_callback),
            desc,
        )
    };
    assert_eq!(status, Z_STREAM_END);
    // SAFETY: the stream was successfully initialized above and ends exactly once.
    let status = unsafe { inflateBackEnd(stream) };
    assert_eq!(status, Z_OK);
    Ok(context.written)
}

fn main() -> BenchResult<()> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: inflate_fast_bench CORPUS [REPETITIONS] [regular|callback-N]")?;
    let repetitions: usize = args.next().map_or(Ok(15), |arg| arg.parse())?;
    assert!(repetitions > 0);
    // Choose any callback input chunk size without rebuilding the benchmark.
    // With no selector, cover both the hot path and deliberately fragmented input.
    let cases = match args.next() {
        None => vec![
            ("regular".to_owned(), None),
            ("callback-65536".to_owned(), Some(65536)),
            ("callback-7".to_owned(), Some(7)),
        ],
        Some(name) if name == "regular" => vec![(name, None)],
        Some(name) => {
            let chunk = name
                .strip_prefix("callback-")
                .ok_or("mode must be regular or callback-N")?
                .parse::<usize>()?;
            assert!(
                (1..=usize::try_from(c_uint::MAX)?).contains(&chunk),
                "callback chunk must fit a nonzero c_uint"
            );
            vec![(name, Some(chunk))]
        }
    };
    let original = std::fs::read(path)?;
    assert!(!original.is_empty(), "benchmark requires nonempty input");
    assert!(original.len() <= usize::try_from(c_uint::MAX)?);
    let mut compressed = vec![0u8; compress_bound(original.len())];
    let (bytes, result) = compress_slice(
        &mut compressed,
        &original,
        DeflateConfig {
            window_bits: -WINDOW_BITS,
            ..DeflateConfig::default()
        },
    );
    assert_eq!(result, ReturnCode::Ok);
    let compressed = bytes.to_vec();
    assert!(compressed.len() <= usize::try_from(c_uint::MAX)?);

    println!(
        "corpus={} compressed={} repetitions={}",
        original.len(),
        compressed.len(),
        repetitions
    );
    let mut output = vec![0u8; original.len()];
    let mut window = vec![0u8; WINDOW_SIZE];
    for (name, chunk) in cases {
        // Warm up the decoder and instruction cache without including setup.
        for _ in 0..3 {
            let written = match chunk {
                None => decode_regular(&compressed, &mut output)?,
                Some(size) => decode_callback(&compressed, &mut output, &mut window, size)?,
            };
            assert_eq!(written, original.len());
            assert_eq!(output, original);
        }

        let mut durations = Vec::with_capacity(repetitions);
        for _ in 0..repetitions {
            let start = Instant::now();
            let written = match chunk {
                None => decode_regular(&compressed, &mut output)?,
                Some(size) => decode_callback(&compressed, &mut output, &mut window, size)?,
            };
            durations.push(start.elapsed().as_secs_f64());
            black_box(written);
            assert_eq!(written, original.len());
            assert_eq!(output, original);
        }
        durations.sort_by(f64::total_cmp);
        let median = durations[durations.len() / 2];
        println!(
            "{name}: median_ms={:.4} throughput_MiBps={:.2}",
            median * 1e3,
            (f64::from(c_uint::try_from(original.len())?) / 1_048_576.0) / median
        );
    }
    Ok(())
}
