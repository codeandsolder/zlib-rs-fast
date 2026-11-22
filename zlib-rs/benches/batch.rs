use std::{
    env,
    hint::black_box,
    time::{Duration, Instant},
};

#[cfg(feature = "parallel")]
use zlib_rs::BatchCodec;
#[path = "shared/measurement.rs"]
mod measurement;
use measurement::{approximately_f64, iterations_for, repeating_phase};

use zlib_rs::{
    compress_bound, compress_slice, compress_slices_batch, BatchJob, DeflateConfig, ReturnCode,
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
    #[cfg(feature = "parallel")]
    ParallelStreams(usize),
}

impl Mode {
    fn label(self) -> String {
        match self {
            Self::Independent => "independent".into(),
            Self::BatchScalar => "batch-scalar".into(),
            #[cfg(feature = "parallel")]
            Self::ParallelStreams(workers) => format!("parallel-streams-{workers}"),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = env::args().collect::<Vec<_>>();
    if args.len() == 2 && args[1] == "--self-test" {
        self_test()?;
        println!("batch benchmark invariants passed");
        return Ok(());
    }
    let full = args.iter().any(|arg| arg == "--full");
    let cases = if full { full_cases() } else { quick_cases() };

    println!("zlib-rs batch scheduler benchmark");
    let (sample_target, samples) = benchmark_shape(full);
    println!(
        "logical CPUs: {} | mode: {} | sample target: {sample_target:?} x {samples} samples",
        std::thread::available_parallelism().map_or(1, usize::from),
        if full { "full" } else { "quick" },
    );
    println!("synthetic inputs; compare whole-stream CPU batch execution modes");
    for case in cases {
        run_case(case, full)?;
    }
    Ok(())
}

const fn benchmark_shape(full: bool) -> (Duration, usize) {
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

fn run_case(case: Case, full: bool) -> Result<(), Box<dyn std::error::Error>> {
    let config = DeflateConfig::new(case.level);
    let inputs = (0..case.streams)
        .map(|stream| {
            Ok(make_input(
                case.corpus,
                case.bytes_per_stream,
                u64::try_from(stream)?,
            ))
        })
        .collect::<Result<Vec<_>, std::num::TryFromIntError>>()?;
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
        approximately_f64(total_output) / approximately_f64(total_input.max(1))
    );

    let modes = vec![Mode::Independent, Mode::BatchScalar];
    #[cfg(feature = "parallel")]
    let modes = {
        let mut modes = modes;
        if full {
            modes.extend([
                Mode::ParallelStreams(2),
                Mode::ParallelStreams(4),
                Mode::ParallelStreams(8),
            ]);
        } else {
            modes.push(Mode::ParallelStreams(4));
        }
        modes
    };
    let mut rows = Vec::new();
    for mode in modes {
        rows.push((
            mode.label(),
            bench_mode(mode, &inputs, &reference, config, full)?,
        ));
    }

    let independent = rows[0].1.median;
    for (label, stats) in rows {
        let mib_per_s = approximately_f64(total_input) / MIB / stats.median.as_secs_f64();
        let relative = independent.as_secs_f64() / stats.median.as_secs_f64();
        let median_ms = stats.median.as_secs_f64() * 1e3;
        let min_ms = stats.min.as_secs_f64() * 1e3;
        let iters = stats.iters;
        println!(
            "  {label:>12}: {mib_per_s:8.1} MiB/s | {relative:5.2}x independent | median {median_ms:8.3} ms | min {min_ms:8.3} ms | iters/sample {iters}"
        );
    }
    println!();
    Ok(())
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
) -> Result<Stats, Box<dyn std::error::Error>> {
    let capacities = inputs
        .iter()
        .map(|input| compress_bound(input.len()) + 64)
        .collect::<Vec<_>>();
    let mut outputs = capacities
        .iter()
        .map(|&capacity| vec![0u8; capacity])
        .collect::<Vec<_>>();
    if matches!(mode, Mode::Independent) {
        bench_independent(inputs, reference, config, full, &mut outputs)
    } else {
        bench_batched(mode, inputs, reference, config, full, &mut outputs)
    }
}

fn bench_independent(
    inputs: &[Vec<u8>],
    reference: &[Vec<u8>],
    config: DeflateConfig,
    full: bool,
    outputs: &mut [Vec<u8>],
) -> Result<Stats, Box<dyn std::error::Error>> {
    let mut written_lengths = Vec::with_capacity(inputs.len());
    {
        let mut validate = || {
            let mut sink = 0usize;
            for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
                let (written, code) = compress_slice(output, input, config);
                assert_eq!(code, ReturnCode::Ok);
                written_lengths.push(written.len());
                sink ^= written.len();
            }
            sink
        };
        black_box(validate());
    }
    assert_outputs(Mode::Independent, outputs, reference, &written_lengths);

    let stats = {
        let mut run = || {
            let mut sink = 0usize;
            for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
                let (written, code) = compress_slice(output, input, config);
                assert_eq!(code, ReturnCode::Ok);
                sink ^= written.len();
            }
            sink
        };
        measure(&mut run, full)?
    };
    black_box(&outputs);
    Ok(stats)
}

enum BatchEngine {
    Scalar,
    #[cfg(feature = "parallel")]
    Parallel(BatchCodec),
}

impl BatchEngine {
    fn new(mode: Mode) -> Result<Self, Box<dyn std::error::Error>> {
        match mode {
            Mode::BatchScalar => Ok(Self::Scalar),
            #[cfg(feature = "parallel")]
            Mode::ParallelStreams(workers) => {
                let workers = std::num::NonZeroUsize::new(workers)
                    .ok_or("parallel benchmark needs a nonzero worker count")?;
                Ok(Self::Parallel(BatchCodec::new(workers)?))
            }
            Mode::Independent => Err("independent mode is not a batch engine".into()),
        }
    }

    fn compress(
        &self,
        jobs: &mut [BatchJob<'_>],
        config: DeflateConfig,
    ) -> Vec<zlib_rs::BatchCompressResult> {
        match self {
            Self::Scalar => compress_slices_batch(jobs, config),
            #[cfg(feature = "parallel")]
            Self::Parallel(codec) => codec.compress(jobs, config),
        }
    }
}

fn bench_batched(
    mode: Mode,
    inputs: &[Vec<u8>],
    reference: &[Vec<u8>],
    config: DeflateConfig,
    full: bool,
    outputs: &mut [Vec<u8>],
) -> Result<Stats, Box<dyn std::error::Error>> {
    let engine = BatchEngine::new(mode)?;

    let mut written_lengths = Vec::with_capacity(inputs.len());
    {
        let mut jobs = inputs
            .iter()
            .zip(outputs.iter_mut())
            .map(|(input, output)| BatchJob::new(input, output))
            .collect::<Vec<_>>();
        let mut run = || {
            let results = engine.compress(&mut jobs, config);
            results.into_iter().fold(0usize, |sink, result| {
                assert_eq!(result.code, ReturnCode::Ok);
                written_lengths.push(result.written);
                sink ^ result.written
            })
        };
        black_box(run());
    }
    assert_outputs(mode, outputs, reference, &written_lengths);

    // Rebind jobs after validation so the timed closure owns the same pre-built caller job
    // descriptors for every iteration. Only allocations performed inside the batch API are
    // charged to the batch modes.
    let stats = {
        let mut jobs = inputs
            .iter()
            .zip(outputs.iter_mut())
            .map(|(input, output)| BatchJob::new(input, output))
            .collect::<Vec<_>>();
        let mut run = || {
            let results = engine.compress(&mut jobs, config);
            results.into_iter().fold(0usize, |sink, result| {
                assert_eq!(result.code, ReturnCode::Ok);
                sink ^ result.written
            })
        };
        measure(&mut run, full)?
    };
    black_box(&outputs);
    Ok(stats)
}

fn measure(
    run: &mut impl FnMut() -> usize,
    full: bool,
) -> Result<Stats, std::num::TryFromIntError> {
    let start = Instant::now();
    black_box(run());
    let first = start.elapsed();
    let (sample_target, sample_count) = benchmark_shape(full);
    let iters = iterations_for(first, sample_target, MAX_SAMPLE_ITERS)?;
    let divisor = u32::try_from(iters)?;
    let mut samples = Vec::with_capacity(sample_count);
    for _ in 0..sample_count {
        let start = Instant::now();
        let mut sink = 0usize;
        for _ in 0..iters {
            sink ^= black_box(run());
        }
        black_box(sink);
        samples.push(start.elapsed() / divisor);
    }
    samples.sort_unstable();

    Ok(Stats {
        median: samples[samples.len() / 2],
        min: samples[0],
        iters,
    })
}

fn assert_outputs(
    mode: Mode,
    outputs: &[Vec<u8>],
    reference: &[Vec<u8>],
    written_lengths: &[usize],
) {
    assert_eq!(outputs.len(), reference.len());
    assert_eq!(written_lengths.len(), reference.len());
    for (index, ((output, expected), &written)) in outputs
        .iter()
        .zip(reference)
        .zip(written_lengths)
        .enumerate()
    {
        assert_eq!(
            written,
            expected.len(),
            "{mode:?} changed reported compressed length for stream {index}"
        );
        assert_eq!(
            &output[..written],
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
            let phase = repeating_phase(seed);
            (0..len)
                .map(|i| PATTERN[i.wrapping_add(phase) % PATTERN.len()])
                .collect()
        }
        Corpus::Structured => {
            let mut out = Vec::with_capacity(len);
            let mut row = seed;
            while out.len() < len {
                let sensor = row % 37;
                let value = (row.wrapping_mul(2_654_435_761) >> 11) % 100_000;
                let status = if row.is_multiple_of(11) { "warn" } else { "ok" };
                let line = format!(
                    "{{\"stream\":{seed},\"row\":{row},\"sensor\":{sensor},\"value\":{value},\"status\":\"{status}\"}}\n"
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
                    state.to_le_bytes()[0]
                })
                .collect()
        }
    }
}

/// Cargo harness=false runs this bench directly; invariants must be callable
/// without relying on otherwise-unregistered test attributes.
fn self_test() -> Result<(), std::num::TryFromIntError> {
    let target = Duration::from_millis(80);
    assert_eq!(
        iterations_for(Duration::ZERO, target, MAX_SAMPLE_ITERS)?,
        MAX_SAMPLE_ITERS
    );
    assert_eq!(
        iterations_for(Duration::from_millis(20), target, MAX_SAMPLE_ITERS)?,
        4
    );
    assert_eq!(
        iterations_for(Duration::from_millis(30), target, MAX_SAMPLE_ITERS)?,
        3
    );
    assert_eq!(
        approximately_f64(1_048_576).to_bits(),
        1_048_576.0f64.to_bits()
    );
    for corpus in [Corpus::Repetitive, Corpus::Structured, Corpus::PseudoRandom] {
        for seed in [0, 1, u64::MAX] {
            let short = make_input(corpus, 17, seed);
            let long = make_input(corpus, 131, seed);
            assert_eq!(short, long[..short.len()]);
        }
    }
    let expected = vec![b'x', b'y', b'z'];
    let outputs = vec![expected.clone()];
    let reference = vec![expected];
    assert_outputs(Mode::Independent, &outputs, &reference, &[3]);
    Ok(())
}
