use std::{
    hint::black_box,
    time::{Duration, Instant},
};

use zlib_rs::adler32::adler32;

const MIB: f64 = 1_048_576.0;
const SAMPLE_TARGET: Duration = Duration::from_millis(250);
const SAMPLES: usize = 9;
const MAX_ITERS: usize = 4096;
const CALIBRATION_CALLS: u32 = 8;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    match args.as_slice() {
        [] => {}
        [mode] if mode == "--self-test" => {
            self_test();
            println!("Adler benchmark invariants passed");
            return Ok(());
        }
        _ => return Err("usage: adler [--self-test]".into()),
    }

    for bytes in [64 * 1024, 1024 * 1024] {
        let input = make_input(bytes);
        let expected = adler32(1, &input);
        let iters = calibrate(&input, expected);
        let divisor = u32::try_from(iters)?;
        let mut samples = Vec::with_capacity(SAMPLES);

        for _ in 0..SAMPLES {
            let start = Instant::now();
            for _ in 0..iters {
                black_box(adler32(1, black_box(&input)));
            }
            samples.push(start.elapsed() / divisor);
        }

        samples.sort_unstable();
        let median = samples[SAMPLES / 2];
        let min = samples[0];
        let bytes_exact = f64::from(u32::try_from(bytes)?);
        let mib_per_second = bytes_exact / MIB / median.as_secs_f64();
        println!(
            "Adler32 {:4} KiB | {mib_per_second:9.1} MiB/s | median {:8.3} us | min {:8.3} us | iters/sample {iters}",
            bytes / 1024,
            median.as_secs_f64() * 1e6,
            min.as_secs_f64() * 1e6,
        );
    }
    Ok(())
}

/// Scale eight measured checksum calls to the requested sample budget using
/// integer nanoseconds. Saturate at the explicit iteration cap for an
/// unmeasurably short call or a count beyond host addressable range.
fn iteration_budget(elapsed: Duration) -> usize {
    let observed_ns = elapsed.as_nanos();
    if observed_ns == 0 {
        return MAX_ITERS;
    }
    let target_ns = SAMPLE_TARGET.as_nanos() * u128::from(CALIBRATION_CALLS);
    usize::try_from(target_ns.div_ceil(observed_ns))
        .unwrap_or(MAX_ITERS)
        .clamp(1, MAX_ITERS)
}

fn calibrate(input: &[u8], expected: u32) -> usize {
    let start = Instant::now();
    let mut checksum = 0;
    for _ in 0..CALIBRATION_CALLS {
        checksum ^= black_box(adler32(1, black_box(input)));
    }
    black_box(checksum ^ expected);
    iteration_budget(start.elapsed())
}

fn make_input(bytes: usize) -> Vec<u8> {
    let mut state = 0x1234_5678_9abc_def0u64;
    (0..bytes)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state.to_le_bytes()[0]
        })
        .collect()
}

/// The benchmark uses Cargo harness=false: keep its invariants
/// explicitly executable rather than attaching unreachable #[test] functions.
fn self_test() {
    assert_eq!(iteration_budget(Duration::ZERO), MAX_ITERS);
    assert_eq!(iteration_budget(Duration::from_nanos(1)), MAX_ITERS);
    assert_eq!(iteration_budget(Duration::from_millis(1)), 2000);
    assert_eq!(iteration_budget(Duration::from_millis(250)), 8);
    assert_eq!(iteration_budget(Duration::from_secs(1)), 2);
    assert_eq!(iteration_budget(Duration::from_secs(4)), 1);

    let short = make_input(64);
    let long = make_input(257);
    assert_eq!(short.as_slice(), &long[..short.len()]);
    assert!(short.windows(2).any(|pair| pair[0] != pair[1]));
    assert_eq!(make_input(0), Vec::<u8>::new());
}
