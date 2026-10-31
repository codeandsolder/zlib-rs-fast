use std::{env, hint::black_box, time::Instant};

use zlib_rs::{
    compress_bound, compress_slice, compress_slices_batch_gpu, decompress_slice, BatchCompressJob,
    DeflateConfig, GpuMatch, GpuMatchSearcher, InflateConfig, ReturnCode,
};

const HASH_SIZE: usize = 1 << 16;
const MAX_MATCH: usize = 258;
const MAX_DIST: usize = 32_768 - (258 + 3 + 1);
const DEFAULT_SEARCH_BUDGET: u32 = 4_096;

#[derive(Clone, Copy, Debug)]
enum Corpus {
    Structured,
    Repetitive,
    PseudoRandom,
}

struct HashIndex {
    offsets: Vec<u32>,
    positions: Vec<u32>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = env::args().collect::<Vec<_>>();
    let bytes = option_value(&args, "--bytes=").map_or(Ok(64 * 1024usize), str::parse)?;
    let search_budget =
        option_value(&args, "--budget=").map_or(Ok(DEFAULT_SEARCH_BUDGET), str::parse)?;
    if search_budget == 0 {
        return Err("--budget must be nonzero".into());
    }
    let corpus = match option_value(&args, "--corpus=").unwrap_or("structured") {
        "structured" => Corpus::Structured,
        "repetitive" => Corpus::Repetitive,
        "random" => Corpus::PseudoRandom,
        other => return Err(format!("unknown corpus {other:?}").into()),
    };
    let adapter_filter = option_value(&args, "--adapter=");
    let level = option_value(&args, "--level=").map_or(Ok(9), str::parse)?;

    if bytes < 3 {
        return Err("--bytes must be at least 3".into());
    }
    let _: u32 = bytes.try_into()?;

    let input = make_input(corpus, bytes);
    let index = timed("cpu CSR index", || build_hash_index(&input));
    let expected = timed("cpu match reference", || {
        match_reference(&input, &index, search_budget)
    });

    let searcher = timed("persistent GPU initialization", || {
        GpuMatchSearcher::new(adapter_filter)
    })?;
    eprintln!("selected adapter: {}", searcher.adapter_name());

    // First search proves the production match-table implementation against the independent CPU
    // oracle. A second timed search shows steady-state cost with device and pipeline already alive.
    let first = timed("GPU first table", || searcher.search(&input, search_budget))?;
    assert_eq!(first.as_slice(), expected.as_slice());
    let steady = timed("GPU steady table", || {
        searcher.search(&input, search_budget)
    })?;
    assert_eq!(steady.as_slice(), expected.as_slice());

    let config = DeflateConfig::new(level);
    let mut scalar_compressed = vec![0u8; compress_bound(input.len()) + 64];
    let scalar_started = Instant::now();
    let (scalar_written, scalar_code) = compress_slice(&mut scalar_compressed, &input, config);
    let scalar_elapsed = scalar_started.elapsed();
    assert_eq!(scalar_code, ReturnCode::Ok);
    let scalar_len = scalar_written.len();

    let mut compressed = vec![0u8; compress_bound(input.len()) + 64];
    let mut jobs = [BatchCompressJob::new(&input, &mut compressed)];
    let started = Instant::now();
    let results = compress_slices_batch_gpu(&mut jobs, config, &searcher, search_budget)?;
    let compression_elapsed = started.elapsed();
    drop(jobs);
    assert_eq!(results[0].code, ReturnCode::Ok);
    let compressed_len = results[0].written;

    let mut decoded = vec![0u8; input.len()];
    let (decoded, code) = decompress_slice(
        &mut decoded,
        &compressed[..compressed_len],
        InflateConfig::default(),
    );
    assert_eq!(code, ReturnCode::Ok);
    assert_eq!(decoded, input.as_slice());

    let matches = steady
        .as_slice()
        .iter()
        .filter(|item| item.length >= 3)
        .count();
    let matched_bytes: u64 = steady
        .as_slice()
        .iter()
        .map(|item| u64::from(item.length))
        .sum();
    println!(
        "validated {} {:?} bytes at budget {} on {}: {} matches, {} total matched bytes; level {} scalar {} bytes ({:.3}%) in {:?}; GPU-fed {} bytes ({:.3}%) in {:?}",
        input.len(),
        corpus,
        search_budget,
        searcher.adapter_name(),
        matches,
        matched_bytes,
        level,
        scalar_len,
        scalar_len as f64 * 100.0 / input.len() as f64,
        scalar_elapsed,
        compressed_len,
        compressed_len as f64 * 100.0 / input.len() as f64,
        compression_elapsed,
    );
    black_box((first, steady, scalar_len, compressed_len));
    Ok(())
}

fn option_value<'a>(args: &'a [String], prefix: &str) -> Option<&'a str> {
    args.iter().find_map(|arg| arg.strip_prefix(prefix))
}

fn timed<T>(label: &str, run: impl FnOnce() -> T) -> T {
    let started = Instant::now();
    let result = run();
    eprintln!("{label}: {:?}", started.elapsed());
    result
}

fn build_hash_index(input: &[u8]) -> HashIndex {
    let valid_positions = input.len().saturating_sub(2);
    let mut counts = vec![0u32; HASH_SIZE];
    for position in 0..valid_positions {
        counts[hash3(input, position) as usize] += 1;
    }

    let mut offsets = vec![0u32; HASH_SIZE + 1];
    for (index, count) in counts.into_iter().enumerate() {
        offsets[index + 1] = offsets[index] + count;
    }

    let mut cursors = offsets[..HASH_SIZE].to_vec();
    let mut positions = vec![0u32; valid_positions];
    for position in 0..valid_positions {
        let hash = hash3(input, position) as usize;
        let cursor = &mut cursors[hash];
        positions[*cursor as usize] = position as u32;
        *cursor += 1;
    }
    HashIndex { offsets, positions }
}

fn match_reference(input: &[u8], index: &HashIndex, search_budget: u32) -> Vec<GpuMatch> {
    let mut results = vec![
        GpuMatch {
            length: 0,
            start: 0
        };
        input.len()
    ];
    for position in 0..input.len().saturating_sub(2) {
        let hash = hash3(input, position) as usize;
        let bucket =
            &index.positions[index.offsets[hash] as usize..index.offsets[hash + 1] as usize];
        let minimum_start = if position >= MAX_DIST {
            position - MAX_DIST + 1
        } else {
            0
        };
        let first = bucket.partition_point(|&candidate| candidate < minimum_start as u32);
        let end = bucket.partition_point(|&candidate| candidate < position as u32);
        let max_len = (input.len() - position).min(MAX_MATCH);
        let mut best = GpuMatch {
            length: 0,
            start: 0,
        };

        for &candidate in bucket[first..end].iter().rev().take(search_budget as usize) {
            let candidate = candidate as usize;
            let len = input[position..position + max_len]
                .iter()
                .zip(&input[candidate..candidate + max_len])
                .position(|(left, right)| left != right)
                .unwrap_or(max_len);
            if len >= 3
                && (len > best.length as usize
                    || (len == best.length as usize && candidate > best.start as usize))
            {
                best = GpuMatch {
                    length: len as u32,
                    start: candidate as u32,
                };
            }
        }
        results[position] = best;
    }
    results
}

fn hash3(input: &[u8], position: usize) -> u32 {
    let value = u32::from(input[position])
        | (u32::from(input[position + 1]) << 8)
        | (u32::from(input[position + 2]) << 16);
    value.wrapping_mul(2_654_435_761) >> 16
}

fn make_input(corpus: Corpus, len: usize) -> Vec<u8> {
    match corpus {
        Corpus::Repetitive => {
            const PATTERN: &[u8] = b"zlib-rs-gpu-match/";
            (0..len)
                .map(|index| PATTERN[index % PATTERN.len()])
                .collect()
        }
        Corpus::Structured => {
            let mut out = Vec::with_capacity(len);
            let mut row = 0u64;
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
            let mut state = 0x1234_5678_9abc_def0u64;
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
