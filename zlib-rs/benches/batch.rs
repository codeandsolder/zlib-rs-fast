use std::{
    env,
    hint::black_box,
    time::{Duration, Instant},
};

#[cfg(feature = "parallel")]
use zlib_rs::compress_slices_batch_parallel;
use zlib_rs::{
    compress_bound, compress_slice, compress_slices_batch, compress_slices_batch_with_backend,
    BatchCompressJob, BatchMatchSearchBackend, DeflateConfig, ReturnCode,
};

const MIB: f64 = (1024 * 1024) as f64;
const QUICK_SAMPLE_TARGET: Duration = Duration::from_millis(80);
const FULL_SAMPLE_TARGET: Duration = Duration::from_millis(220);
const MAX_SAMPLE_ITERS: usize = 128;
const QUICK_SAMPLES: usize = 3;
const FULL_SAMPLES: usize = 7;

#[derive(Clone, Copy, Debug)]
enum Corpus {
    Repetitive,
    Structured,
    PseudoRandom,
}

#[derive(Clone, Copy, Debug)]
struct Case {
    name: &'static str,
    corpus: Corpus,
    streams: usize,
    bytes_per_stream: usize,
    level: i32,
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    Independent,
    BatchScalar,
    Threads(usize),
    #[cfg(feature = "parallel")]
    ParallelStreams,
}

impl Mode {
    fn label(self) -> String {
        match self {
            Self::Independent => "independent".into(),
            Self::BatchScalar => "batch-scalar".into(),
            Self::Threads(workers) => format!("threads-{workers}"),
            #[cfg(feature = "parallel")]
            Self::ParallelStreams => "parallel-streams".into(),
        }
    }
}

fn main() {
    let args = env::args().collect::<Vec<_>>();
    let full = args.iter().any(|arg| arg == "--full");
    let include_threads = args.iter().any(|arg| arg == "--threads");
    let cases = if full { full_cases() } else { quick_cases() };

    println!("zlib-rs batch scheduler benchmark");
    let (sample_target, samples) = benchmark_shape(full);
    println!(
        "logical CPUs: {} | mode: {} | sample target: {:?} x {} samples",
        std::thread::available_parallelism().map_or(1, usize::from),
        if full { "full" } else { "quick" },
        sample_target,
        samples
    );
    println!(
        "synthetic inputs; threaded backend: {} | use these measurements for scheduler/backend decisions, not general zlib claims\n",
        if include_threads { "included" } else { "skipped (pass --threads)" }
    );

    for case in cases {
        run_case(case, full, include_threads);
    }
}

fn benchmark_shape(full: bool) -> (Duration, usize) {
    if full {
        (FULL_SAMPLE_TARGET, FULL_SAMPLES)
    } else {
        (QUICK_SAMPLE_TARGET, QUICK_SAMPLES)
    }
}

fn quick_cases() -> Vec<Case> {
    vec![
        Case {
            name: "rep-64k-x8-l6",
            corpus: Corpus::Repetitive,
            streams: 8,
            bytes_per_stream: 64 * 1024,
            level: 6,
        },
        Case {
            name: "structured-256k-x8-l6",
            corpus: Corpus::Structured,
            streams: 8,
            bytes_per_stream: 256 * 1024,
            level: 6,
        },
        Case {
            name: "random-256k-x8-l6",
            corpus: Corpus::PseudoRandom,
            streams: 8,
            bytes_per_stream: 256 * 1024,
            level: 6,
        },
        Case {
            name: "structured-256k-x8-l9",
            corpus: Corpus::Structured,
            streams: 8,
            bytes_per_stream: 256 * 1024,
            level: 9,
        },
    ]
}

fn full_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for streams in [1, 2, 4, 8, 16] {
        cases.push(Case {
            name: match streams {
                1 => "structured-256k-x1-l6",
                2 => "structured-256k-x2-l6",
                4 => "structured-256k-x4-l6",
                8 => "structured-256k-x8-l6",
                16 => "structured-256k-x16-l6",
                _ => unreachable!(),
            },
            corpus: Corpus::Structured,
            streams,
            bytes_per_stream: 256 * 1024,
            level: 6,
        });
    }
    cases.extend([
        Case {
            name: "rep-64k-x16-l6",
            corpus: Corpus::Repetitive,
            streams: 16,
            bytes_per_stream: 64 * 1024,
            level: 6,
        },
        Case {
            name: "structured-1m-x4-l6",
            corpus: Corpus::Structured,
            streams: 4,
            bytes_per_stream: 1024 * 1024,
            level: 6,
        },
        Case {
            name: "random-256k-x8-l6",
            corpus: Corpus::PseudoRandom,
            streams: 8,
            bytes_per_stream: 256 * 1024,
            level: 6,
        },
        Case {
            name: "structured-256k-x8-l9",
            corpus: Corpus::Structured,
            streams: 8,
            bytes_per_stream: 256 * 1024,
            level: 9,
        },
    ]);
    cases
}

fn run_case(case: Case, full: bool, include_threads: bool) {
    let config = DeflateConfig::new(case.level);
    let inputs = (0..case.streams)
        .map(|stream| make_input(case.corpus, case.bytes_per_stream, stream as u64))
        .collect::<Vec<_>>();
    let reference = compress_reference(&inputs, config);
    let total_input = case.streams * case.bytes_per_stream;
    let total_output: usize = reference.iter().map(Vec::len).sum();

    println!(
        "{}: {:?}, {} streams x {} KiB, level {}, ratio {:.3}",
        case.name,
        case.corpus,
        case.streams,
        case.bytes_per_stream / 1024,
        case.level,
        total_output as f64 / total_input.max(1) as f64
    );

    let mut modes = vec![Mode::Independent, Mode::BatchScalar];
    if include_threads {
        if full {
            modes.extend([Mode::Threads(2), Mode::Threads(4), Mode::Threads(8)]);
        } else {
            modes.push(Mode::Threads(4));
        }
    }
    #[cfg(feature = "parallel")]
    modes.push(Mode::ParallelStreams);
    let mut rows = Vec::new();
    for mode in modes {
        rows.push((
            mode.label(),
            bench_mode(mode, &inputs, &reference, config, full),
        ));
    }

    let independent = rows[0].1.median;
    for (label, stats) in rows {
        let mib_per_s = total_input as f64 / MIB / stats.median.as_secs_f64();
        let relative = independent.as_secs_f64() / stats.median.as_secs_f64();
        println!(
            "  {label:>12}: {:8.1} MiB/s | {:5.2}x independent | median {:8.3} ms | min {:8.3} ms | iters/sample {}",
            mib_per_s,
            relative,
            stats.median.as_secs_f64() * 1e3,
            stats.min.as_secs_f64() * 1e3,
            stats.iters
        );
    }
    println!();
}

#[derive(Clone, Copy)]
struct Stats {
    median: Duration,
    min: Duration,
    iters: usize,
}

fn bench_mode(
    mode: Mode,
    inputs: &[Vec<u8>],
    reference: &[Vec<u8>],
    config: DeflateConfig,
    full: bool,
) -> Stats {
    let capacities = inputs
        .iter()
        .map(|input| compress_bound(input.len()) + 64)
        .collect::<Vec<_>>();
    let mut outputs = capacities
        .iter()
        .map(|&capacity| vec![0u8; capacity])
        .collect::<Vec<_>>();

    match mode {
        Mode::Independent => {
            let mut validate = || {
                let mut sink = 0usize;
                for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
                    let (written, code) = compress_slice(output, input, config);
                    assert_eq!(code, ReturnCode::Ok);
                    sink ^= written.len();
                }
                sink
            };
            black_box(validate());
            drop(validate);
            assert_outputs(mode, &outputs, reference);

            let mut run = || {
                let mut sink = 0usize;
                for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
                    let (written, code) = compress_slice(output, input, config);
                    assert_eq!(code, ReturnCode::Ok);
                    sink ^= written.len();
                }
                sink
            };
            let stats = measure(&mut run, full);
            drop(run);
            black_box(&outputs);
            stats
        }
        _ => {
            let mut jobs = inputs
                .iter()
                .zip(outputs.iter_mut())
                .map(|(input, output)| BatchCompressJob::new(input, output))
                .collect::<Vec<_>>();
            let mut run = || {
                let results = match mode {
                    Mode::BatchScalar => compress_slices_batch(&mut jobs, config),
                    Mode::Threads(workers) => compress_slices_batch_with_backend(
                        &mut jobs,
                        config,
                        BatchMatchSearchBackend::ScopedThreads { workers },
                    ),
                    #[cfg(feature = "parallel")]
                    Mode::ParallelStreams => compress_slices_batch_parallel(&mut jobs, config),
                    Mode::Independent => unreachable!(),
                };
                results.into_iter().fold(0usize, |sink, result| {
                    assert_eq!(result.code, ReturnCode::Ok);
                    sink ^ result.written
                })
            };
            black_box(run());
            drop(run);
            drop(jobs);
            assert_outputs(mode, &outputs, reference);

            // Rebind jobs after validation so the timed closure owns the same pre-built caller job
            // descriptors for every iteration. Only allocations performed inside the batch API are
            // charged to the batch modes.
            let mut jobs = inputs
                .iter()
                .zip(outputs.iter_mut())
                .map(|(input, output)| BatchCompressJob::new(input, output))
                .collect::<Vec<_>>();
            let mut run = || {
                let results = match mode {
                    Mode::BatchScalar => compress_slices_batch(&mut jobs, config),
                    Mode::Threads(workers) => compress_slices_batch_with_backend(
                        &mut jobs,
                        config,
                        BatchMatchSearchBackend::ScopedThreads { workers },
                    ),
                    #[cfg(feature = "parallel")]
                    Mode::ParallelStreams => compress_slices_batch_parallel(&mut jobs, config),
                    Mode::Independent => unreachable!(),
                };
                results.into_iter().fold(0usize, |sink, result| {
                    assert_eq!(result.code, ReturnCode::Ok);
                    sink ^ result.written
                })
            };
            let stats = measure(&mut run, full);
            drop(run);
            drop(jobs);
            black_box(&outputs);
            stats
        }
    }
}

fn measure(run: &mut impl FnMut() -> usize, full: bool) -> Stats {
    let start = Instant::now();
    black_box(run());
    let first = start.elapsed();
    let (sample_target, sample_count) = benchmark_shape(full);
    let iters = iterations_for(first, sample_target);
    let mut samples = Vec::with_capacity(sample_count);
    for _ in 0..sample_count {
        let start = Instant::now();
        let mut sink = 0usize;
        for _ in 0..iters {
            sink ^= black_box(run());
        }
        black_box(sink);
        samples.push(start.elapsed() / iters as u32);
    }
    samples.sort_unstable();

    Stats {
        median: samples[samples.len() / 2],
        min: samples[0],
        iters,
    }
}

fn iterations_for(one: Duration, sample_target: Duration) -> usize {
    if one.is_zero() {
        return MAX_SAMPLE_ITERS;
    }
    let target = sample_target.as_nanos();
    let one = one.as_nanos();
    ((target / one).max(1) as usize).min(MAX_SAMPLE_ITERS)
}

fn assert_outputs(mode: Mode, outputs: &[Vec<u8>], reference: &[Vec<u8>]) {
    for (index, (output, expected)) in outputs.iter().zip(reference).enumerate() {
        assert_eq!(
            &output[..expected.len()],
            expected,
            "{mode:?} changed compressed output for stream {index}"
        );
    }
}

fn compress_reference(inputs: &[Vec<u8>], config: DeflateConfig) -> Vec<Vec<u8>> {
    inputs
        .iter()
        .map(|input| {
            let mut output = vec![0u8; compress_bound(input.len()) + 64];
            let (written, code) = compress_slice(&mut output, input, config);
            assert_eq!(code, ReturnCode::Ok);
            let len = written.len();
            output.truncate(len);
            output
        })
        .collect()
}

fn make_input(corpus: Corpus, len: usize, seed: u64) -> Vec<u8> {
    match corpus {
        Corpus::Repetitive => {
            const PATTERN: &[u8] = b"zlib-rs portable SIMD batch scheduling / ";
            (0..len)
                .map(|i| PATTERN[(i + seed as usize * 7) % PATTERN.len()])
                .collect()
        }
        Corpus::Structured => {
            let mut out = Vec::with_capacity(len);
            let mut row = seed;
            while out.len() < len {
                let line = format!(
                    "{{\"stream\":{},\"row\":{},\"sensor\":{},\"value\":{},\"status\":\"{}\"}}\n",
                    seed,
                    row,
                    row % 37,
                    (row.wrapping_mul(2_654_435_761) >> 11) % 100_000,
                    if row % 11 == 0 { "warn" } else { "ok" }
                );
                let take = (len - out.len()).min(line.len());
                out.extend_from_slice(&line.as_bytes()[..take]);
                row = row.wrapping_add(1);
            }
            out
        }
        Corpus::PseudoRandom => {
            let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
            (0..len)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    state as u8
                })
                .collect()
        }
    }
}
