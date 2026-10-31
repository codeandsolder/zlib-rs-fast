use std::{
    env,
    hint::black_box,
    time::{Duration, Instant},
};

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

fn main() {
    let args = env::args().collect::<Vec<_>>();
    let full = args.iter().any(|arg| arg == "--full");
    let sweep = args.iter().any(|arg| arg == "--sweep");
    let profile = args.iter().any(|arg| arg == "--profile");
    let profile_iters = args
        .iter()
        .find_map(|arg| arg.strip_prefix("--profile-iters="))
        .map_or(Ok(1usize), str::parse)
        .expect("--profile-iters must be a positive integer");
    assert!(profile_iters != 0, "--profile-iters must be nonzero");
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
        return;
    }

    let (target, samples) = benchmark_shape(full);

    println!("zlib-rs single-stream benchmark");
    println!(
        "mode: {} | sample target: {:?} x {} samples | synthetic inputs",
        if full { "full" } else { "quick" },
        target,
        samples
    );

    for case in cases {
        run_case(case, full);
    }
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
        "{:<22} {:?} {:>4} KiB L{} | {} bytes ({:.4}) | {} compression iteration(s)",
        case.name,
        case.corpus,
        case.bytes / 1024,
        case.level,
        written_len,
        written_len as f64 / input.len() as f64,
        iters,
    );
}

fn run_case(case: Case, full: bool) {
    let input = make_input(case.corpus, case.bytes, 0x1234_5678_9abc_def0);
    let config = DeflateConfig::new(case.level);
    let mut output = vec![0u8; compress_bound(input.len()) + 64];

    let (reference_len, ratio) = {
        let (written, code) = compress_slice(&mut output, &input, config);
        assert_eq!(code, ReturnCode::Ok);
        (
            written.len(),
            written.len() as f64 / input.len().max(1) as f64,
        )
    };

    let mut run = || {
        let (written, code) = compress_slice(&mut output, &input, config);
        assert_eq!(code, ReturnCode::Ok);
        assert_eq!(written.len(), reference_len);
        written.len()
    };
    let stats = measure(&mut run, full);
    drop(run);
    black_box(&output);

    let mib_per_s = case.bytes as f64 / MIB / stats.median.as_secs_f64();
    println!(
        "{:<22} {:?} {:>4} KiB L{} ratio {:.3} | {:8.1} MiB/s | median {:8.3} ms | min {:8.3} ms | iters/sample {}",
        case.name,
        case.corpus,
        case.bytes / 1024,
        case.level,
        ratio,
        mib_per_s,
        stats.median.as_secs_f64() * 1e3,
        stats.min.as_secs_f64() * 1e3,
        stats.iters,
    );
}

fn benchmark_shape(full: bool) -> (Duration, usize) {
    if full {
        (FULL_SAMPLE_TARGET, FULL_SAMPLES)
    } else {
        (QUICK_SAMPLE_TARGET, QUICK_SAMPLES)
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
    ((sample_target.as_nanos() / one.as_nanos()).max(1) as usize).min(MAX_SAMPLE_ITERS)
}

fn make_input(corpus: Corpus, len: usize, seed: u64) -> Vec<u8> {
    match corpus {
        Corpus::Repetitive => {
            const PATTERN: &[u8] = b"zlib-rs portable SIMD single stream / ";
            (0..len)
                .map(|i| PATTERN[(i + seed as usize * 7) % PATTERN.len()])
                .collect()
        }
        Corpus::Structured => {
            let mut out = Vec::with_capacity(len);
            let mut row = seed;
            while out.len() < len {
                let line = format!(
                    "{{\"row\":{},\"sensor\":{},\"value\":{},\"status\":\"{}\"}}\n",
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
