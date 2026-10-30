use std::{
    hint::black_box,
    time::{Duration, Instant},
};

use zlib_rs::adler32::adler32;

const MIB: f64 = (1024 * 1024) as f64;
const SAMPLE_TARGET: Duration = Duration::from_millis(250);
const SAMPLES: usize = 9;
const MAX_ITERS: usize = 4096;

fn main() {
    for bytes in [64 * 1024, 1024 * 1024] {
        let input = make_input(bytes);
        let expected = adler32(1, &input);
        let iters = calibrate(&input, expected);
        let mut samples = Vec::with_capacity(SAMPLES);

        for _ in 0..SAMPLES {
            let start = Instant::now();
            for _ in 0..iters {
                black_box(adler32(1, black_box(&input)));
            }
            samples.push(start.elapsed() / iters as u32);
        }

        samples.sort_unstable();
        let median = samples[SAMPLES / 2];
        let min = samples[0];
        let mib_per_second = bytes as f64 / MIB / median.as_secs_f64();
        println!(
            "Adler32 {:4} KiB | {:9.1} MiB/s | median {:8.3} us | min {:8.3} us | iters/sample {}",
            bytes / 1024,
            mib_per_second,
            median.as_secs_f64() * 1e6,
            min.as_secs_f64() * 1e6,
            iters,
        );
    }
}

fn calibrate(input: &[u8], expected: u32) -> usize {
    let start = Instant::now();
    let mut checksum = 0;
    for _ in 0..8 {
        checksum ^= black_box(adler32(1, black_box(input)));
    }
    black_box(checksum ^ expected);
    let per_call = start.elapsed().as_secs_f64() / 8.0;
    if per_call == 0.0 {
        return MAX_ITERS;
    }
    ((SAMPLE_TARGET.as_secs_f64() / per_call).ceil() as usize).clamp(1, MAX_ITERS)
}

fn make_input(bytes: usize) -> Vec<u8> {
    let mut state = 0x1234_5678_9abc_def0u64;
    (0..bytes)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u8
        })
        .collect()
}
