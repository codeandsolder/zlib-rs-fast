//! Whole-stream scheduling shared across the CPU and GPU-fed codecs.

use crate::{inflate::uncompress2_into, InflateConfig, ReturnCode};

/// One independent input/output pair for batch encoding or decoding.
///
/// Each job owns an exclusive destination. The corresponding result reports
/// the initialized output prefix and the same return code as an independent
/// whole-slice call.
pub struct BatchJob<'a> {
    pub input: &'a [u8],
    pub output: &'a mut [u8],
}

impl<'a> BatchJob<'a> {
    #[must_use]
    pub const fn new(input: &'a [u8], output: &'a mut [u8]) -> Self {
        Self { input, output }
    }
}

/// Result of one independently decoded stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchDecompressResult {
    /// Compressed input bytes consumed, excluding trailing bytes.
    pub read: usize,
    /// Initialized output bytes, including a partial prefix on failure.
    pub written: usize,
    pub code: ReturnCode,
}

/// Decode independent streams serially, preserving input order and returning
/// exact input consumption and written lengths for each job.
pub fn decompress_slices_batch(
    jobs: &mut [BatchJob<'_>],
    config: InflateConfig,
) -> Vec<BatchDecompressResult> {
    jobs.iter_mut()
        .map(|job| decompress_job(job, config))
        .collect()
}

fn decompress_job(job: &mut BatchJob<'_>, config: InflateConfig) -> BatchDecompressResult {
    let (read, written, code) = uncompress2_into(job.output, job.input, config);
    BatchDecompressResult {
        read,
        written,
        code,
    }
}

/// Caller-owned reusable worker pool for independent DEFLATE streams.
///
/// Both compression and decompression run complete streams on this pool.
/// CPU-only and explicitly selected GPU-assisted encoding share its worker
/// budget, including simultaneous calls from different threads; it never
/// consults or configures Rayon's global pool.
/// Requires the optional `parallel` feature.
#[cfg(feature = "parallel")]
pub struct BatchCodec {
    pool: rayon_core::ThreadPool,
}

#[cfg(feature = "parallel")]
impl BatchCodec {
    /// Build a reusable pool with exactly `workers` threads.
    ///
    /// # Errors
    /// Returns the Rayon pool-construction error if workers cannot be created.
    pub fn new(workers: core::num::NonZeroUsize) -> Result<Self, rayon_core::ThreadPoolBuildError> {
        rayon_core::ThreadPoolBuilder::new()
            .num_threads(workers.get())
            .build()
            .map(|pool| Self { pool })
    }

    /// Worker count owned by this codec.
    #[must_use]
    pub fn workers(&self) -> usize {
        self.pool.current_num_threads()
    }

    /// Decode streams concurrently. Each job's input consumption, initialized
    /// output, and return code match an independent whole-slice decode.
    pub fn decompress(
        &self,
        jobs: &mut [BatchJob<'_>],
        config: InflateConfig,
    ) -> Vec<BatchDecompressResult> {
        self.run_jobs(
            jobs,
            BatchDecompressResult {
                read: 0,
                written: 0,
                code: ReturnCode::StreamError,
            },
            |_, job| decompress_job(job, config),
        )
    }

    pub(crate) fn run_jobs<R: Copy + Send>(
        &self,
        jobs: &mut [BatchJob<'_>],
        initial: R,
        run: impl Fn(usize, &mut BatchJob<'_>) -> R + Sync,
    ) -> Vec<R> {
        if jobs.is_empty() {
            return Vec::new();
        }
        if jobs.len() == 1 || self.workers() == 1 {
            // Even single-job calls share the pool budget with concurrent
            // callers; running on their threads would exceed that budget.
            return self.pool.install(|| {
                jobs.iter_mut()
                    .enumerate()
                    .map(|(index, job)| run(index, job))
                    .collect()
            });
        }
        let mut results = vec![initial; jobs.len()];
        // Each task owns the independent input/output pair and its result slot.
        // Work stealing handles differences in stream length and complexity;
        // output order is independent of execution order.
        self.pool.scope(|scope| {
            let run = &run;
            for (index, (job, result)) in jobs.iter_mut().zip(&mut results).enumerate() {
                scope.spawn(move |_| {
                    *result = run(index, job);
                });
            }
        });
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{compress_bound, compress_slice, DeflateConfig};

    #[cfg(feature = "parallel")]
    #[test]
    fn even_single_job_runs_in_the_owned_pool() -> Result<(), &'static str> {
        for workers in [
            core::num::NonZeroUsize::MIN,
            core::num::NonZeroUsize::new(4).ok_or("worker count")?,
        ] {
            let codec = BatchCodec::new(workers).map_err(|_| "worker pool")?;
            // Separate caller threads must both run inside this pool, not
            // bypass it when submitting single-job batches.
            std::thread::scope(|scope| {
                for _ in 0..4 {
                    let codec = &codec;
                    scope.spawn(move || {
                        let mut output = [0u8; 1];
                        let mut jobs = [BatchJob::new(b"x", &mut output)];
                        let results = codec.run_jobs(&mut jobs, false, |_, _| {
                            codec.pool.current_thread_index().is_some()
                        });
                        assert_eq!(results, [true]);
                    });
                }
            });
        }
        Ok(())
    }

    #[test]
    fn serial_batch_decode_preserves_independent_results_and_canaries() {
        run_decode_cases(decompress_slices_batch);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_batch_decode_preserves_independent_results_and_canaries() -> Result<(), &'static str>
    {
        for workers in [
            core::num::NonZeroUsize::MIN,
            core::num::NonZeroUsize::new(4).ok_or("worker count")?,
        ] {
            let codec = BatchCodec::new(workers).map_err(|_| "worker pool")?;
            run_decode_cases(|jobs, config| codec.decompress(jobs, config));
            assert!(codec
                .decompress(&mut [], InflateConfig::default())
                .is_empty());
        }
        Ok(())
    }

    fn run_decode_cases(
        mut decode: impl FnMut(&mut [BatchJob<'_>], InflateConfig) -> Vec<BatchDecompressResult>,
    ) {
        let input = b"batch-decode/history-window/".repeat(8192);
        for wrapper in [-15, 15, 31] {
            let config = InflateConfig {
                window_bits: wrapper,
            };
            let encode_config = DeflateConfig {
                window_bits: wrapper,
                ..DeflateConfig::new(6)
            };
            let mut encoded = vec![0u8; compress_bound(input.len()) + 64];
            let (compressed, rc) = compress_slice(&mut encoded, &input, encode_config);
            assert_eq!(rc, ReturnCode::Ok);
            let mut trailing = compressed.to_vec();
            trailing.extend_from_slice(b"unused trailing bytes");
            let truncated = &compressed[..compressed.len() / 2];
            let mut empty_buf = [0u8; 64];
            let (empty, rc) = compress_slice(&mut empty_buf, &[], encode_config);
            assert_eq!(rc, ReturnCode::Ok);
            let mut corrupted = compressed.to_vec();
            let middle = corrupted.len() / 2;
            corrupted[middle] ^= 0x80;
            let inputs = [
                compressed,
                trailing.as_slice(),
                truncated,
                empty,
                corrupted.as_slice(),
                &[],
            ];
            // Whole buffers, short and zero destinations, valid and failed jobs.
            for capacity in [0, 1, 23, input.len(), input.len() + 31] {
                let mut expected = vec![vec![0xa5; capacity]; inputs.len()];
                let refs = inputs
                    .iter()
                    .zip(&mut expected)
                    .map(|(src, dst)| {
                        let (read, written, code) =
                            uncompress2_into(dst.as_mut_slice(), src, config);
                        BatchDecompressResult {
                            read,
                            written,
                            code,
                        }
                    })
                    .collect::<Vec<_>>();
                let mut actual = vec![vec![0xa5; capacity]; inputs.len()];
                let results = {
                    let mut jobs = inputs
                        .iter()
                        .zip(&mut actual)
                        .map(|(src, dst)| BatchJob::new(src, dst))
                        .collect::<Vec<_>>();
                    decode(&mut jobs, config)
                };
                assert_eq!(results, refs);
                assert_eq!(actual, expected);
                if capacity >= input.len() {
                    for index in 0..2 {
                        assert_eq!(results[index].code, ReturnCode::Ok);
                        assert_eq!(results[index].read, compressed.len());
                        assert_eq!(results[index].written, input.len());
                        assert_eq!(&actual[index][..input.len()], input);
                        assert!(actual[index][input.len()..].iter().all(|&b| b == 0xa5));
                    }
                }
            }
        }
    }
}
