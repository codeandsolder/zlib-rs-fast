use std::{
    env,
    hint::black_box,
    time::{Duration, Instant},
};

#[path = "shared/measurement.rs"]
mod measurement;

use measurement::{approximately_f64, iterations_for, repeating_phase};
use zlib_rs::{compress_bound, compress_slice, DeflateConfig, ReturnCode};

const MIB: f64 = (1024 * 1024) as f64;
const QUICK_SAMPLE_TARGET: Duration = Duration::from_millis(100);
const FULL_SAMPLE_TARGET: Duration = Duration::from_millis(250);
const MAX_SAMPLE_ITERS: usize = 256;
const QUICK_SAMPLES: usize = 5;
const FULL_SAMPLES: usize = 9;

#[derive(Clone, Copy, Debug)]
enum Corpus {
    Repetitive,
    Structured,
    PseudoRandom,
}

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    corpus: Corpus,
    bytes: usize,
    level: i32,
}

#[derive(Clone, Copy)]
struct Stats {
    median: Duration,
    min: Duration,
    iters: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = env::args().collect::<Vec<_>>();
    if args.len() == 2 && args[1] == "--self-test" {
        self_test()?;
        println!("single-stream benchmark invariants passed");
        return Ok(());
    }
    let full = args.iter().any(|arg| arg == "--full");
    let sweep = args.iter().any(|arg| arg == "--sweep");
    let profile = args.iter().any(|arg| arg == "--profile");
    let profile_iters = args
        .iter()
        .find_map(|arg| arg.strip_prefix("--profile-iters="))
        .map_or(Ok(1usize), str::parse)?;
    if profile_iters == 0 {
        return Err("--profile-iters must be nonzero".into());
    }
    let mut cases = if sweep {
        sweep_cases()
    } else if full {
        full_cases()
    } else {
        quick_cases()
    };
    if let Some(only) = args.iter().find_map(|arg| arg.strip_prefix("--only=")) {
        let filters = only.split(',').collect::<Vec<_>>();
        cases.retain(|case| filters.iter().any(|filter| case.name.contains(filter)));
    }
    if profile {
        println!("zlib-rs single-stream profile workload");
        for case in cases {
            profile_case(case, profile_iters);
        }
        return Ok(());
    }

    let (target, samples) = benchmark_shape(full);

    println!("zlib-rs single-stream benchmark");
    println!(
        "mode: {} | sample target: {target:?} x {samples} samples | synthetic inputs",
        if full { "full" } else { "quick" },
    );

    for case in cases {
        run_case(case, full)?;
    }
    Ok(())
}

fn quick_cases() -> Vec<Case> {
    vec![
        Case {
            name: "rep-64k-l1",
            corpus: Corpus::Repetitive,
            bytes: 64 * 1024,
            level: 1,
        },
        Case {
            name: "rep-64k-l6",
            corpus: Corpus::Repetitive,
            bytes: 64 * 1024,
            level: 6,
        },
        Case {
            name: "structured-256k-l6",
            corpus: Corpus::Structured,
            bytes: 256 * 1024,
            level: 6,
        },
        Case {
            name: "structured-256k-l7",
            corpus: Corpus::Structured,
            bytes: 256 * 1024,
            level: 7,
        },
        Case {
            name: "random-256k-l6",
            corpus: Corpus::PseudoRandom,
            bytes: 256 * 1024,
            level: 6,
        },
        Case {
            name: "random-256k-l9",
            corpus: Corpus::PseudoRandom,
            bytes: 256 * 1024,
            level: 9,
        },
        Case {
            name: "structured-256k-l9",
            corpus: Corpus::Structured,
            bytes: 256 * 1024,
            level: 9,
        },
    ]
}

fn sweep_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for level in [6, 9] {
        for kib in [64, 128, 256, 384, 512, 768, 1024, 1536, 2048] {
            cases.push(Case {
                name: "structured-sweep",
                corpus: Corpus::Structured,
                bytes: kib * 1024,
                level,
            });
        }
    }
    cases
}

fn full_cases() -> Vec<Case> {
    let mut cases = quick_cases();
    cases.extend([
        Case {
            name: "structured-64k-l6",
            corpus: Corpus::Structured,
            bytes: 64 * 1024,
            level: 6,
        },
        Case {
            name: "structured-1m-l6",
            corpus: Corpus::Structured,
            bytes: 1024 * 1024,
            level: 6,
        },
        Case {
            name: "random-1m-l6",
            corpus: Corpus::PseudoRandom,
            bytes: 1024 * 1024,
            level: 6,
        },
        Case {
            name: "structured-1m-l9",
            corpus: Corpus::Structured,
            bytes: 1024 * 1024,
            level: 9,
        },
    ]);
    cases
}

fn profile_case(case: Case, iters: usize) {
    let input = make_input(case.corpus, case.bytes, 0x1234_5678_9abc_def0);
    let config = DeflateConfig::new(case.level);
    let mut output = vec![0u8; compress_bound(input.len()) + 64];

    let mut written_len = 0usize;
    for _ in 0..iters {
        let (written, code) = compress_slice(&mut output, &input, config);
        assert_eq!(code, ReturnCode::Ok);
        written_len = black_box(written.len());
    }
    black_box((&input, &output, written_len));

    println!(
        "{:<22} {:?} {:>4} KiB L{} | {written_len} bytes ({ratio:.4}) | {iters} compression iteration(s)",
        case.name,
        case.corpus,
        case.bytes / 1024,
        case.level,
        ratio = approximately_f64(written_len) / approximately_f64(input.len()),
    );
}

fn run_case(case: Case, full: bool) -> Result<(), std::num::TryFromIntError> {
    let input = make_input(case.corpus, case.bytes, 0x1234_5678_9abc_def0);
    let config = DeflateConfig::new(case.level);
    let mut output = vec![0u8; compress_bound(input.len()) + 64];

    let (reference_len, ratio) = {
        let (written, code) = compress_slice(&mut output, &input, config);
        assert_eq!(code, ReturnCode::Ok);
        (
            written.len(),
            approximately_f64(written.len()) / approximately_f64(input.len().max(1)),
        )
    };

    let mut run = || {
        let (written, code) = compress_slice(&mut output, &input, config);
        assert_eq!(code, ReturnCode::Ok);
        assert_eq!(written.len(), reference_len);
        written.len()
    };
    let stats = measure(&mut run, full)?;
    drop(run);
    black_box(&output);

    let mib_per_s = approximately_f64(case.bytes) / MIB / stats.median.as_secs_f64();
    println!(
        "{:<22} {:?} {:>4} KiB L{} ratio {ratio:.3} | {mib_per_s:8.1} MiB/s | median {:8.3} ms | min {:8.3} ms | iters/sample {}",
        case.name,
        case.corpus,
        case.bytes / 1024,
        case.level,
        stats.median.as_secs_f64() * 1e3,
        stats.min.as_secs_f64() * 1e3,
        stats.iters,
    );
    Ok(())
}

const fn benchmark_shape(full: bool) -> (Duration, usize) {
    if full {
        (FULL_SAMPLE_TARGET, FULL_SAMPLES)
    } else {
        (QUICK_SAMPLE_TARGET, QUICK_SAMPLES)
    }
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

fn make_input(corpus: Corpus, len: usize, seed: u64) -> Vec<u8> {
    match corpus {
        Corpus::Repetitive => {
            const PATTERN: &[u8] = b"zlib-rs portable SIMD single stream / ";
            // Match the previous release-mode low-word truncation and wrapping
            // arithmetic on both 32/64-bit targets without unchecked casts.
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
                let status = if row % 11 == 0 { "warn" } else { "ok" };
                let line = format!(
                    "{{\"row\":{row},\"sensor\":{sensor},\"value\":{value},\"status\":\"{status}\"}}\n"
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

/// This target uses Cargo harness=false, so these regressions must be run
/// through the executable instead of silently ignored #[test] functions.
fn self_test() -> Result<(), std::num::TryFromIntError> {
    let target = Duration::from_millis(100);
    assert_eq!(
        iterations_for(Duration::ZERO, target, MAX_SAMPLE_ITERS)?,
        MAX_SAMPLE_ITERS
    );
    assert_eq!(
        iterations_for(Duration::from_nanos(1), target, MAX_SAMPLE_ITERS)?,
        MAX_SAMPLE_ITERS
    );
    assert_eq!(
        iterations_for(Duration::from_millis(25), target, MAX_SAMPLE_ITERS)?,
        4
    );
    assert_eq!(
        iterations_for(Duration::from_millis(30), target, MAX_SAMPLE_ITERS)?,
        4
    );
    assert_eq!(
        iterations_for(Duration::from_secs(1), target, MAX_SAMPLE_ITERS)?,
        1
    );
    assert_eq!(approximately_f64(0), 0.0);
    assert_eq!(approximately_f64(1_048_576), 1_048_576.0);
    for corpus in [Corpus::Repetitive, Corpus::Structured, Corpus::PseudoRandom] {
        for seed in [0, 1, 0x1234_5678_9abc_def0, u64::MAX] {
            let short = make_input(corpus, 31, seed);
            let long = make_input(corpus, 129, seed);
            assert_eq!(short, long[..short.len()]);
            assert_eq!(make_input(corpus, 0, seed), Vec::<u8>::new());
        }
    }
    Ok(())
}
