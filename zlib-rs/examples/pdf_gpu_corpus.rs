use std::{env, fs, num::NonZeroUsize, path::Path, time::Instant};
use zlib_rs::{
    compress_bound, decompress_slice, BatchCodec, BatchJob, DeflateConfig, GpuMatchSearcher,
    InflateConfig, ReturnCode,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let [root, workers, adapter] = args.as_slice() else {
        return Err("usage: pdf_gpu_corpus CORPUS_DIRECTORY WORKERS ADAPTER_SUBSTRING".into());
    };
    let root = Path::new(root);
    let workers = workers.parse::<NonZeroUsize>()?;
    let list = fs::read_to_string(root.join("lists/all.txt"))?;
    let mut inputs = list
        .lines()
        .map(|path| fs::read(root.join(path)))
        .collect::<Result<Vec<_>, _>>()?;
    inputs.sort_by_key(Vec::len);
    let pool = BatchCodec::new(workers)?;
    let start = Instant::now();
    let gpu = GpuMatchSearcher::new(Some(adapter))?;
    eprintln!(
        "adapter={}, init_ms={:.1}",
        gpu.adapter_name(),
        start.elapsed().as_secs_f64() * 1000.
    );
    println!(
        "level,range,streams,input_bytes,budget,cpu_ms,gpu_ms,cpu_bytes,gpu_bytes,larger_streams"
    );
    for level in [7, 9] {
        let config = DeflateConfig::new(level);
        for (lo, hi) in [
            (0, 4096),
            (4096, 65_536),
            (65_536, 262_144),
            (262_144, 1_048_576),
            (1_048_576, usize::MAX),
        ] {
            let selected = inputs
                .iter()
                .filter(|s| (lo..hi).contains(&s.len()))
                .map(Vec::as_slice)
                .collect::<Vec<_>>();
            if selected.is_empty() {
                continue;
            }
            let total = selected.iter().map(|s| s.len()).sum::<usize>();
            let (cpu_time, cpu_sizes) = run(&selected, &pool, config, None)?;
            for budget in [8, 32, 128, 512] {
                let (gpu_time, gpu_sizes) = run(&selected, &pool, config, Some((&gpu, budget)))?;
                println!(
                    "{level},{lo}-{hi},{},{total},{budget},{:.3},{:.3},{},{},{}",
                    selected.len(),
                    cpu_time * 1000.,
                    gpu_time * 1000.,
                    cpu_sizes.iter().sum::<usize>(),
                    gpu_sizes.iter().sum::<usize>(),
                    gpu_sizes
                        .iter()
                        .zip(&cpu_sizes)
                        .filter(|(g, c)| g > c)
                        .count()
                );
            }
        }
    }
    Ok(())
}
fn run(
    inputs: &[&[u8]],
    pool: &BatchCodec,
    config: DeflateConfig,
    gpu: Option<(&GpuMatchSearcher, u32)>,
) -> Result<(f64, Vec<usize>), Box<dyn std::error::Error>> {
    let mut elapsed = 0.;
    let mut sizes = Vec::with_capacity(inputs.len());
    let mut remaining = inputs;
    while !remaining.is_empty() {
        let mut bytes = 0usize;
        let count = remaining
            .iter()
            .take(64)
            .take_while(|s| {
                if bytes != 0 && bytes + s.len() > 8 * 1024 * 1024 {
                    return false;
                }
                bytes += s.len();
                true
            })
            .count();
        let (batch, rest) = remaining.split_at(count);
        remaining = rest;
        let start = Instant::now();
        let mut outputs = vec![Vec::new(); batch.len()];
        let mut jobs = batch
            .iter()
            .zip(&mut outputs)
            .map(|(&input, output)| BatchJob::with_vec(input, output, compress_bound(input.len())))
            .collect::<Result<Vec<_>, _>>()?;
        let results = match gpu {
            Some((searcher, budget)) => pool.compress_gpu(&mut jobs, config, searcher, budget)?,
            None => pool.compress(&mut jobs, config),
        };
        drop(jobs);
        elapsed += start.elapsed().as_secs_f64();
        for ((input, output), result) in batch.iter().zip(outputs).zip(results) {
            assert_eq!(result.code, ReturnCode::Ok);
            assert_eq!(result.written, output.len());
            let mut decoded = vec![0; input.len()];
            let (decoded, code) = decompress_slice(&mut decoded, &output, InflateConfig::default());
            assert_eq!(code, ReturnCode::Ok);
            assert_eq!(decoded, *input);
            sizes.push(output.len());
        }
    }
    Ok((elapsed, sizes))
}
