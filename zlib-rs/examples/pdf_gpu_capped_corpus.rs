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
    Routed {
        budget: u32,
        window: NonZeroUsize,
        min_cap_per_mille: u16,
        overlap: bool,
        retry: bool,
    },
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

fn measure_routed(
    samples: &[Sample],
    jobs: Vec<BatchJob<'_>>,
    pool: &BatchCodec,
    searcher: &GpuMatchSearcher,
    config: DeflateConfig,
    mode: Mode,
) -> Result<Vec<BatchCompressResult>, Box<dyn std::error::Error>> {
    let Mode::Routed {
        budget,
        window,
        min_cap_per_mille,
        overlap,
        retry,
    } = mode
    else {
        return Err("routed mode required".into());
    };
    let count = jobs.len();
    let (cpu, gpu): (Vec<_>, Vec<_>) = jobs.into_iter().enumerate().partition(|(index, job)| {
        samples[*index].cap as u128 * 1000 < job.input.len() as u128 * u128::from(min_cap_per_mille)
    });
    let (cpu_indices, mut cpu_jobs): (Vec<_>, Vec<_>) = cpu.into_iter().unzip();
    let (gpu_indices, mut gpu_jobs): (Vec<_>, Vec<_>) = gpu.into_iter().unzip();
    let (cpu, mut gpu) = if overlap {
        pool.compress_mixed_gpu(
            &mut cpu_jobs,
            &mut gpu_jobs,
            config,
            searcher,
            budget,
            Some(window),
        )?
    } else {
        (
            pool.compress(&mut cpu_jobs, config),
            pool.compress_gpu_windowed(&mut gpu_jobs, config, searcher, budget, window)?,
        )
    };
    if retry {
        let (indices, mut jobs): (Vec<_>, Vec<_>) = gpu_jobs
            .into_iter()
            .enumerate()
            .filter(|(index, _)| gpu[*index].code == ReturnCode::BufError)
            .unzip();
        for (index, result) in indices.into_iter().zip(pool.compress(&mut jobs, config)) {
            gpu[index] = result;
        }
    }
    let mut results = vec![
        BatchCompressResult {
            written: 0,
            code: ReturnCode::StreamError
        };
        count
    ];
    for (index, result) in cpu_indices
        .into_iter()
        .zip(cpu)
        .chain(gpu_indices.into_iter().zip(gpu))
    {
        results[index] = result;
    }
    Ok(results)
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
            Mode::Routed { .. } => measure_routed(batch, jobs, pool, searcher, config, mode)?,
            Mode::Cpu => {
                let results = pool.compress(&mut jobs, config);
                drop(jobs);
                results
            }
            Mode::Whole(budget) => {
                let results = pool.compress_gpu(&mut jobs, config, searcher, budget)?;
                drop(jobs);
                results
            }
            Mode::Windowed(budget, window) => {
                let results =
                    pool.compress_gpu_windowed(&mut jobs, config, searcher, budget, window)?;
                drop(jobs);
                results
            }
        };
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

fn print_measurement(
    label: &str,
    measurement: &Measurement,
    samples: &[Sample],
) -> Result<(), Box<dyn std::error::Error>> {
    measurement.verify(samples)?;
    let savings: usize = samples
        .iter()
        .zip(&measurement.results)
        .filter(|(_, result)| result.code == ReturnCode::Ok)
        .map(|(sample, result)| sample.cap - result.written)
        .sum();
    println!(
        "{label},{:.3},{},{savings}",
        measurement.elapsed.as_secs_f64() * 1000.,
        measurement.accepted()
    );
    for (index, (sample, result)) in samples.iter().zip(&measurement.results).enumerate() {
        eprintln!(
            "{label},{index},{},{},{:?},{}",
            sample.input.len(),
            sample.cap,
            result.code,
            result.written
        );
    }
    Ok(())
}

fn run_policy(
    samples: &[Sample],
    pool: &BatchCodec,
    searcher: &GpuMatchSearcher,
    config: DeflateConfig,
    budget: u32,
    window: NonZeroUsize,
    min_cap_per_mille: Option<u16>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("mode,ms,accepted,bytes_saved_below_cap");
    eprintln!("mode,index,decoded,cap,code,written");
    let cpu = measure(samples, pool, searcher, config, Mode::Cpu)?;
    print_measurement("cpu", &cpu, samples)?;
    let gpu = measure(
        samples,
        pool,
        searcher,
        config,
        Mode::Windowed(budget, window),
    )?;
    print_measurement("gpu", &gpu, samples)?;
    if let Some(min_cap_per_mille) = min_cap_per_mille {
        let mut serial = None;
        for (label, overlap, retry) in [
            ("routed-serial", false, false),
            ("routed-overlap", true, false),
            ("routed-rescue", true, true),
        ] {
            let measurement = measure(
                samples,
                pool,
                searcher,
                config,
                Mode::Routed {
                    budget,
                    window,
                    min_cap_per_mille,
                    overlap,
                    retry,
                },
            )?;
            print_measurement(label, &measurement, samples)?;
            if !overlap {
                serial = Some(measurement);
            } else if !retry {
                let reference = serial.as_ref().ok_or("missing serial reference")?;
                assert_eq!(measurement.results, reference.results);
                assert_eq!(measurement.outputs, reference.outputs);
            }
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let [manifest, workers, adapter, policy @ ..] = args.as_slice() else {
        return Err("usage: pdf_gpu_capped_corpus MANIFEST_TSV WORKERS ADAPTER [BUDGET WINDOW_BYTES [MIN_CAP_PER_MILLE]]".into());
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
    match policy {
        [] => {}
        [budget, window] => {
            return run_policy(
                &samples,
                &pool,
                &searcher,
                config,
                budget.parse()?,
                window.parse()?,
                None,
            )
        }
        [budget, window, cap] => {
            return run_policy(
                &samples,
                &pool,
                &searcher,
                config,
                budget.parse()?,
                window.parse()?,
                Some(cap.parse()?),
            )
        }
        _ => return Err("expected budget, window and optional cap ratio".into()),
    }
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
