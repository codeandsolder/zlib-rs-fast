//! Whole-stream scheduling shared across the CPU and GPU-fed codecs.

use crate::{
    inflate::{uncompress2_into, WholeDecoder},
    output_buffer::WritableBytes,
    InflateConfig, ReturnCode,
};

/// One independent input/output pair for batch encoding or decoding.
///
/// Each job owns an exclusive destination. A fixed slice preserves bytes
/// beyond its written prefix. A vector destination replaces its contents
/// with exactly that prefix, including a partial prefix on failure.
pub struct BatchJob<'a> {
    pub input: &'a [u8],
    output: BatchOutput<'a>,
}

/// Both destinations expose only the capacity explicitly selected by the caller.
pub struct BatchOutput<'a>(Destination<'a>);

enum Destination<'a> {
    Slice(&'a mut [u8]),
    Vector {
        output: &'a mut Vec<u8>,
        limit: usize,
    },
}

impl WritableBytes for BatchOutput<'_> {
    #[inline]
    fn byte_ptr(&mut self) -> *mut u8 {
        match &mut self.0 {
            Destination::Slice(output) => output.as_mut_ptr(),
            Destination::Vector { output, limit } => {
                output.spare_capacity_mut()[..*limit].as_mut_ptr().cast()
            }
        }
    }

    #[inline]
    fn len(&self) -> usize {
        match &self.0 {
            Destination::Slice(output) => output.len(),
            Destination::Vector { limit, .. } => *limit,
        }
    }
}

impl<'a> BatchJob<'a> {
    /// Write into a fixed initialized slice, retaining its untouched tail.
    #[must_use]
    pub const fn new(input: &'a [u8], output: &'a mut [u8]) -> Self {
        Self {
            input,
            output: BatchOutput(Destination::Slice(output)),
        }
    }

    /// Replace a caller-owned vector with an output prefix of at most
    /// `limit` bytes. Capacity is reserved before execution, without
    /// initializing the spare storage. Existing capacity can be reused;
    /// the vector's previous contents are discarded when the job runs.
    ///
    /// # Errors
    /// Returns a reservation error if the requested capacity cannot be allocated.
    pub fn with_vec(
        input: &'a [u8],
        output: &'a mut Vec<u8>,
        limit: usize,
    ) -> Result<Self, std::collections::TryReserveError> {
        output.try_reserve_exact(limit.saturating_sub(output.len()))?;
        Ok(Self {
            input,
            output: BatchOutput(Destination::Vector { output, limit }),
        })
    }

    /// Run a codec against the bounded destination and commit its reported prefix.
    ///
    /// # Safety
    /// The operation must return the exact initialized prefix length, no larger
    /// than the destination's declared capacity. It must not retain the buffer.
    pub(crate) unsafe fn write<R>(
        &mut self,
        operation: impl FnOnce(&[u8], &mut BatchOutput<'_>) -> (usize, R),
    ) -> R {
        if let Destination::Vector { output, .. } = &mut self.output.0 {
            output.clear();
        }
        let (written, result) = operation(self.input, &mut self.output);
        assert!(
            written <= self.output.len(),
            "codec prefix exceeds destination limit"
        );
        if let Destination::Vector { output, .. } = &mut self.output.0 {
            // SAFETY: reservation establishes capacity >= limit; the codec
            // operation reports only its initialized prefix. The exclusive
            // borrow prevents changing capacity during codec execution.
            unsafe { output.set_len(written) };
        }
        result
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
    if jobs.len() <= 1 {
        return jobs
            .iter_mut()
            .map(|job| decompress_job(job, |output, input| uncompress2_into(output, input, config)))
            .collect();
    }
    let mut decoder = WholeDecoder::new(config);
    jobs.iter_mut()
        .map(|job| decompress_reused_job(job, decoder.as_mut()))
        .collect()
}

fn decompress_reused_job(
    job: &mut BatchJob<'_>,
    decoder: Option<&mut WholeDecoder>,
) -> BatchDecompressResult {
    decompress_job(job, |output, input| {
        decoder.map_or((0, 0, ReturnCode::StreamError), |decoder| {
            decoder.decode(output, input)
        })
    })
}

fn decompress_job(
    job: &mut BatchJob<'_>,
    decode: impl FnOnce(&mut BatchOutput<'_>, &[u8]) -> (usize, usize, ReturnCode),
) -> BatchDecompressResult {
    // SAFETY: decode reports only initialized output bytes, bounded
    // by its destination's declared capacity, on every return path.
    unsafe {
        job.write(|input, output| {
            let (read, written, code) = decode(output, input);
            (
                written,
                BatchDecompressResult {
                    read,
                    written,
                    code,
                },
            )
        })
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
        if jobs.len() <= 1 {
            return self.pool.install(|| decompress_slices_batch(jobs, config));
        }
        self.run_with_state(
            jobs,
            || BatchDecompressResult {
                read: 0,
                written: 0,
                code: ReturnCode::StreamError,
            },
            || WholeDecoder::new(config),
            |decoder, _, job| decompress_reused_job(job, decoder.as_mut()),
        )
    }

    /// Stateful workers own their allocation locally, including non-Send
    /// state. Claim jobs dynamically from a short-lived shared iterator;
    /// only the disjoint job/result references leave the critical section.
    pub(crate) fn run_with_state<T: Send, R: Send, S>(
        &self,
        jobs: &mut [T],
        initial: impl FnMut() -> R,
        make: impl Fn() -> S + Sync,
        run: impl Fn(&mut S, usize, &mut T) -> R + Sync,
    ) -> Vec<R> {
        let workers = self.workers().min(jobs.len());
        if workers == 0 {
            return Vec::new();
        }
        if workers == 1 {
            return self.pool.install(|| {
                let mut state = make();
                jobs.iter_mut()
                    .enumerate()
                    .map(|(index, job)| run(&mut state, index, job))
                    .collect()
            });
        }
        let mut results: Vec<R> = core::iter::repeat_with(initial).take(jobs.len()).collect();
        let queue = std::sync::Mutex::new(jobs.iter_mut().zip(&mut results).enumerate());
        self.pool.scope(|scope| {
            for _ in 0..workers {
                let (queue, make, run) = (&queue, &make, &run);
                scope.spawn(move |_| {
                    let mut state = make();
                    loop {
                        let next = queue
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .next();
                        let Some((index, (job, result))) = next else {
                            break;
                        };
                        *result = run(&mut state, index, job);
                    }
                });
            }
        });
        results
    }

    #[cfg(any(feature = "gpu", test))]
    pub(crate) fn run_items<T: Send, R: Send>(
        &self,
        jobs: &mut [T],
        initial: impl FnMut() -> R,
        run: impl Fn(usize, &mut T) -> R + Sync,
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
        let mut results: Vec<R> = core::iter::repeat_with(initial).take(jobs.len()).collect();
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
                        let results = codec.run_items(
                            &mut jobs,
                            || false,
                            |_, _| codec.pool.current_thread_index().is_some(),
                        );
                        assert_eq!(results, [true]);
                    });
                }
            });
        }
        Ok(())
    }

    #[test]
    fn serial_batch_decode_preserves_independent_results_and_canaries(
    ) -> Result<(), std::collections::TryReserveError> {
        run_decode_cases(decompress_slices_batch)
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
            run_decode_cases(|jobs, config| codec.decompress(jobs, config))
                .map_err(|_| "vector output reservation")?;
            assert!(codec
                .decompress(&mut [], InflateConfig::default())
                .is_empty());
        }
        Ok(())
    }

    #[test]
    fn reuse_handles_different_headers_and_partial_streams(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let input = b"decoder reuse: different wrapper windows and block modes/".repeat(128);
        let mut streams = Vec::new();
        for window_bits in [9, 12, 15, 25, 31] {
            for level in [0, 1, 6, 9] {
                let mut compressed = vec![0; compress_bound(input.len()) + 64];
                let (encoded, code) = compress_slice(
                    &mut compressed,
                    &input,
                    DeflateConfig {
                        window_bits,
                        ..DeflateConfig::new(level)
                    },
                );
                assert_eq!(code, ReturnCode::Ok);
                let len = encoded.len();
                compressed.truncate(len);
                streams.push(compressed.clone());
                compressed.extend_from_slice(b"trailing input");
                streams.push(compressed.clone());
                compressed.truncate(len / 2);
                streams.push(compressed);
            }
        }
        streams.push(vec![0xff; 17]);
        streams.push(Vec::new());
        let mut seed = 0x1234_5678u32;
        for _ in 0..128 {
            let bytes = (0..32)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    seed.to_le_bytes()[0]
                })
                .collect::<Vec<_>>();
            streams.push(bytes);
        }
        for window_bits in [-15, 0, 15, 31, 32, 47, i32::MAX] {
            let config = InflateConfig { window_bits };
            let capacities = (0..streams.len())
                .map(|i| [0, 1, 23, input.len(), input.len() + 31][i % 5])
                .collect::<Vec<_>>();
            let mut expected = capacities
                .iter()
                .map(|&cap| vec![0xa5; cap])
                .collect::<Vec<_>>();
            let references = streams
                .iter()
                .zip(&mut expected)
                .map(|(src, dst)| {
                    let (read, written, code) = uncompress2_into(dst.as_mut_slice(), src, config);
                    BatchDecompressResult {
                        read,
                        written,
                        code,
                    }
                })
                .collect::<Vec<_>>();
            let check = |run: &dyn Fn(&mut [BatchJob<'_>]) -> Vec<BatchDecompressResult>| -> Result<(), std::collections::TryReserveError> {
                let mut slice_outputs = capacities.iter().map(|&cap| vec![0xa5; cap]).collect::<Vec<_>>();
                let mut jobs = streams.iter().zip(&mut slice_outputs).map(|(src,dst)| BatchJob::new(src,dst)).collect::<Vec<_>>();
                assert_eq!(run(&mut jobs), references);
                drop(jobs);
                assert_eq!(slice_outputs, expected);
                let mut vectors = vec![vec![0x79; 3]; streams.len()];
                let mut jobs = streams.iter().zip(&mut vectors).zip(&capacities)
                    .map(|((src,dst),&cap)| BatchJob::with_vec(src,dst,cap)).collect::<Result<Vec<_>,_>>()?;
                assert_eq!(run(&mut jobs), references);
                drop(jobs);
                for ((actual, expected), result) in vectors.iter().zip(&expected).zip(&references) {
                    assert_eq!(actual, &expected[..result.written]);
                }
                Ok(())
            };
            check(&|jobs| decompress_slices_batch(jobs, config))?;
            #[cfg(feature = "parallel")]
            for workers in [
                core::num::NonZeroUsize::MIN,
                core::num::NonZeroUsize::new(4).ok_or("workers")?,
            ] {
                let codec = BatchCodec::new(workers)?;
                check(&|jobs| codec.decompress(jobs, config))?;
            }
        }
        Ok(())
    }

    #[test]
    fn vector_reservation_failure_preserves_previous_contents() {
        let mut output = b"prior contents".to_vec();
        assert!(BatchJob::with_vec(b"input", &mut output, usize::MAX).is_err());
        assert_eq!(output, b"prior contents");
    }

    fn verify_vector_decode(
        decode: &mut impl FnMut(&mut [BatchJob<'_>], InflateConfig) -> Vec<BatchDecompressResult>,
        inputs: &[&[u8]],
        capacity: usize,
        config: InflateConfig,
        expected: &[Vec<u8>],
        refs: &[BatchDecompressResult],
    ) -> Result<(), std::collections::TryReserveError> {
        let mut vectors = vec![vec![0x79; 3]; inputs.len()];
        let mut jobs = inputs
            .iter()
            .zip(&mut vectors)
            .map(|(src, dst)| BatchJob::with_vec(src, dst, capacity))
            .collect::<Result<Vec<_>, _>>()?;
        // Repeated execution replaces the initialized extent rather than
        // appending or exposing the previous contents.
        for _ in 0..2 {
            assert_eq!(decode(&mut jobs, config), refs);
        }
        drop(jobs);
        for (vector, (reference, result)) in vectors.iter().zip(expected.iter().zip(refs)) {
            assert_eq!(vector, &reference[..result.written]);
        }
        Ok(())
    }

    fn run_decode_cases(
        mut decode: impl FnMut(&mut [BatchJob<'_>], InflateConfig) -> Vec<BatchDecompressResult>,
    ) -> Result<(), std::collections::TryReserveError> {
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
                verify_vector_decode(&mut decode, &inputs, capacity, config, &expected, &refs)?;
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
        Ok(())
    }
}
