use std::{
    env, fs,
    num::NonZeroUsize,
    time::{Duration, Instant},
};
use zlib_rs::{
    decompress_slice, BatchCodec, BatchCompressResult, BatchJob, DeflateConfig, GpuMatchSearcher,
    InflateConfig, ReturnCode,
};

struct Sample {
    input: Vec<u8>,
    cap: usize,
}

#[derive(Clone, Copy)]
enum Mode {
    Cpu,
    Whole(u32),
    Windowed(u32, NonZeroUsize),
}

struct Measurement {
    elapsed: Duration,
    results: Vec<BatchCompressResult>,
    outputs: Vec<Vec<u8>>,
}

impl Measurement {
    fn accepted(&self) -> usize {
        self.results
            .iter()
            .filter(|result| result.code == ReturnCode::Ok)
            .count()
    }

    fn verify(&self, samples: &[Sample]) -> Result<(), Box<dyn std::error::Error>> {
        for ((sample, result), output) in samples.iter().zip(&self.results).zip(&self.outputs) {
            assert_eq!(output.len(), result.written);
            assert!(output.len() <= sample.cap);
            match result.code {
                ReturnCode::Ok => {
                    let mut decoded = vec![0; sample.input.len()];
                    let (decoded, code) =
                        decompress_slice(&mut decoded, output, InflateConfig::default());
                    assert_eq!(code, ReturnCode::Ok);
                    assert_eq!(decoded, sample.input);
                }
                ReturnCode::BufError => {}
                code => return Err(format!("unexpected codec result: {code:?}").into()),
            }
        }
        Ok(())
    }
}

fn measure(
    samples: &[Sample],
    pool: &BatchCodec,
    searcher: &GpuMatchSearcher,
    config: DeflateConfig,
    mode: Mode,
) -> Result<Measurement, Box<dyn std::error::Error>> {
    let mut elapsed = Duration::ZERO;
    let mut outputs = Vec::with_capacity(samples.len());
    let mut results = Vec::with_capacity(samples.len());
    let mut rest = samples;
    while !rest.is_empty() {
        let mut bytes = 0usize;
        let count = rest
            .iter()
            .take(64)
            .take_while(|sample| {
                if bytes != 0 && bytes.saturating_add(sample.input.len()) > 32 * 1024 * 1024 {
                    return false;
                }
                bytes = bytes.saturating_add(sample.input.len());
                true
            })
            .count();
        let (batch, remaining) = rest.split_at(count);
        rest = remaining;
        let start = Instant::now();
        let mut buffers = vec![Vec::new(); batch.len()];
        let mut jobs = batch
            .iter()
            .zip(&mut buffers)
            .map(|(sample, output)| BatchJob::with_vec(&sample.input, output, sample.cap))
            .collect::<Result<Vec<_>, _>>()?;
        let batch_results = match mode {
            Mode::Cpu => pool.compress(&mut jobs, config),
            Mode::Whole(budget) => pool.compress_gpu(&mut jobs, config, searcher, budget)?,
            Mode::Windowed(budget, window) => {
                pool.compress_gpu_windowed(&mut jobs, config, searcher, budget, window)?
            }
        };
        drop(jobs);
        elapsed += start.elapsed();
        results.extend(batch_results);
        outputs.extend(buffers);
    }
    Ok(Measurement {
        elapsed,
        results,
        outputs,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let [manifest, workers, adapter] = args.as_slice() else {
        return Err("usage: pdf_gpu_capped_corpus MANIFEST_TSV WORKERS ADAPTER".into());
    };
    let samples = fs::read_to_string(manifest)?
        .lines()
        .map(|line| {
            let (path, cap) = line.split_once('\t').ok_or("invalid manifest row")?;
            Ok(Sample {
                input: fs::read(path)?,
                cap: cap.parse()?,
            })
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    let pool = BatchCodec::new(workers.parse()?)?;
    let searcher = GpuMatchSearcher::new(Some(adapter))?;
    let config = DeflateConfig::new(9);
    let cpu = measure(&samples, &pool, &searcher, config, Mode::Cpu)?;
    cpu.verify(&samples)?;
    println!("mode,budget,window,ms,accepted,streams");
    println!(
        "cpu,0,0,{:.3},{},{}",
        cpu.elapsed.as_secs_f64() * 1000.,
        cpu.accepted(),
        samples.len()
    );
    for budget in [128, 512] {
        let whole = measure(&samples, &pool, &searcher, config, Mode::Whole(budget))?;
        whole.verify(&samples)?;
        println!(
            "whole,{budget},0,{:.3},{},{}",
            whole.elapsed.as_secs_f64() * 1000.,
            whole.accepted(),
            samples.len()
        );
        for bytes in [256 * 1024, 1024 * 1024, 4 * 1024 * 1024] {
            let window = NonZeroUsize::new(bytes).ok_or("zero window")?;
            let candidate = measure(
                &samples,
                &pool,
                &searcher,
                config,
                Mode::Windowed(budget, window),
            )?;
            assert_eq!(candidate.results, whole.results);
            assert_eq!(candidate.outputs, whole.outputs);
            println!(
                "windowed,{budget},{bytes},{:.3},{},{}",
                candidate.elapsed.as_secs_f64() * 1000.,
                candidate.accepted(),
                samples.len()
            );
        }
    }
    Ok(())
}
