#[cfg(any(feature = "gpu", test))]
use core::marker::PhantomData;
use std::vec::Vec;

#[cfg(test)]
use super::compress_slice;
#[cfg(feature = "gpu")]
use super::gpu_match::{GpuMatchError, GpuMatchSearcher, GpuMatchTable};
#[cfg(feature = "gpu")]
use super::STD_MIN_MATCH;
#[cfg(any(feature = "gpu", test))]
use super::{
    algorithm::{
        Algorithm, EndBlockKind, MatchSourceKind, StandardDriver, StandardSegmentMode,
        StandardTurnOutcome,
    },
    deflate,
    longest_match::{MatchSearchError, MatchSearchResult, MatchSearchWork},
    CallCursors, State, Status,
};
use super::{
    compress_into_stream, reset_impl, DeflateConfig, DeflateFlush, DeflateStream, HashHistoryReset,
    ReturnCode,
};
#[cfg(any(feature = "gpu", test))]
use crate::abi_cursor::SliceBudget;
#[cfg(any(feature = "gpu", test))]
use crate::output_buffer::WritableBytes;
#[cfg(feature = "parallel")]
use crate::BatchCodec;
use crate::BatchJob;
#[cfg(feature = "gpu")]
use crate::{MAX_WBITS, MIN_WBITS};

/// Result for one job passed to [`compress_slices_batch`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchCompressResult {
    pub written: usize,
    pub code: ReturnCode,
}

impl BatchJob<'_> {
    /// Compress this one independent stream without allocating a batch
    /// result vector. The destination follows the same bounded-output
    /// contract as [`compress_slices_batch`].
    pub fn compress(&mut self, config: DeflateConfig) -> BatchCompressResult {
        CpuEncoder::new(config).compress(self)
    }
}

/// Compress several independent buffers with scalar per-stream compression.
///
/// Every job uses the same configuration and produces the same bytes as an independent
/// [`super::compress_slice`] call. Each stream is driven to completion to preserve cache locality.
/// For concurrent CPU compression use `BatchCodec` with the `parallel` feature; GPU
/// compression has a separate whole-input match-table API.
pub fn compress_slices_batch(
    jobs: &mut [BatchJob<'_>],
    config: DeflateConfig,
) -> Vec<BatchCompressResult> {
    compress_synchronously(jobs, config)
}

#[cfg(feature = "parallel")]
impl BatchCodec {
    /// Compress several independent buffers concurrently on this codec's worker pool.
    pub fn compress(
        &self,
        jobs: &mut [BatchJob<'_>],
        config: DeflateConfig,
    ) -> Vec<BatchCompressResult> {
        self.run_with_state(
            jobs,
            || BatchCompressResult {
                written: 0,
                code: ReturnCode::StreamError,
            },
            || CpuEncoder::new(config),
            |encoder, _, job| encoder.compress(job),
        )
    }

    /// Search all jobs on the GPU, then encode independent streams concurrently on this pool.
    ///
    /// Input preparation and encoding use this pool; each encoder owns its stream and
    /// reads only its corresponding immutable match table. Results retain input order and
    /// match the sequential GPU-assisted API byte-for-byte.
    ///
    /// # Errors
    ///
    /// Returns the same GPU search errors as [`compress_slices_batch_gpu`].
    #[cfg(feature = "gpu")]
    pub fn compress_gpu(
        &self,
        jobs: &mut [BatchJob<'_>],
        config: DeflateConfig,
        searcher: &GpuMatchSearcher,
        search_budget: u32,
    ) -> Result<Vec<BatchCompressResult>, GpuMatchError> {
        let Some((algorithm, tables)) = prepare_gpu_matches(jobs, config, |inputs, max_dist| {
            searcher.search_batch_with_pool(inputs, search_budget, max_dist, self)
        })?
        else {
            return Ok(self.compress(jobs, config));
        };
        Ok(self.run_items(
            jobs,
            || BatchCompressResult {
                written: 0,
                code: ReturnCode::StreamError,
            },
            |job_index, job| {
                compress_external_job(
                    job_index,
                    job,
                    config,
                    algorithm,
                    &mut |index, state, position, work| {
                        resolve_table_match(index, state, position, work, &tables)
                    },
                )
            },
        ))
    }
}

/// Compress complete input buffers using a persistent GPU whole-input match table.
///
/// The GPU device and compute pipeline belong to `searcher` and are reused across calls. Each input
/// receives one whole-input table before compression begins; when the ordinary producer yields a
/// detached match-search request, the driver resolves it from that table instead of walking the
/// hash chain on the CPU. Framing, parsing, token emission, Huffman coding, and output draining stay
/// on the existing CPU path.
///
/// # Errors
///
/// Returns an error if the GPU match search cannot accept the workload,
/// allocate, dispatch, or retrieve its results.
#[cfg(feature = "gpu")]
pub fn compress_slices_batch_gpu(
    jobs: &mut [BatchJob<'_>],
    config: DeflateConfig,
    searcher: &GpuMatchSearcher,
    search_budget: u32,
) -> Result<Vec<BatchCompressResult>, GpuMatchError> {
    let Some((algorithm, tables)) = prepare_gpu_matches(jobs, config, |inputs, max_dist| {
        searcher.search_batch_with_max_dist(inputs, search_budget, max_dist)
    })?
    else {
        return Ok(compress_synchronously(jobs, config));
    };
    Ok(compress_external(
        jobs,
        config,
        algorithm,
        |job_index, state, position, search| {
            resolve_table_match(job_index, state, position, search, &tables)
        },
    ))
}

#[cfg(feature = "gpu")]
fn prepare_gpu_matches(
    jobs: &[BatchJob<'_>],
    config: DeflateConfig,
    search: impl FnOnce(&[&[u8]], u32) -> Result<Vec<GpuMatchTable>, GpuMatchError>,
) -> Result<Option<(Algorithm, Vec<GpuMatchTable>)>, GpuMatchError> {
    let Some(algorithm) =
        configured_algorithm(config).filter(|&algorithm| uses_match_search(algorithm))
    else {
        return Ok(None);
    };
    let Some(max_dist) = configured_max_dist(config) else {
        return Ok(None);
    };
    let inputs = jobs.iter().map(|job| job.input).collect::<Vec<_>>();
    let tables = search(&inputs, max_dist)?;
    Ok(Some((algorithm, tables)))
}

#[cfg(feature = "gpu")]
fn resolve_table_match(
    job_index: usize,
    state: &State<'_>,
    absolute_position: Option<usize>,
    search: MatchSearchWork,
    tables: &[GpuMatchTable],
) -> Result<MatchSearchResult, MatchSearchError> {
    if !search.is_external() {
        return Err(MatchSearchError::WrongBackend);
    }
    Ok(match_from_table(
        state,
        absolute_position,
        &tables[job_index],
    ))
}

#[cfg(any(feature = "gpu", test))]
const fn uses_match_search(algorithm: Algorithm) -> bool {
    matches!(
        algorithm,
        Algorithm::Fast | Algorithm::Medium | Algorithm::Slow
    )
}

#[cfg(feature = "gpu")]
fn configured_max_dist(config: DeflateConfig) -> Option<u32> {
    let mut window_bits = config.window_bits;
    let zlib_wrapper = if window_bits < 0 {
        if window_bits < -MAX_WBITS {
            return None;
        }
        window_bits = -window_bits;
        false
    } else if window_bits > MAX_WBITS {
        window_bits -= 16;
        false
    } else {
        true
    };

    if !(MIN_WBITS..=MAX_WBITS).contains(&window_bits) || (window_bits == 8 && !zlib_wrapper) {
        return None;
    }
    let effective_bits = if window_bits == 8 {
        9
    } else {
        u32::try_from(window_bits).ok()?
    };
    // The GPU uses u32 positions; compute the exact zlib distance bound in
    // that representation, or decline GPU acceleration if it cannot fit.
    (1u32.checked_shl(effective_bits)?).checked_sub(u32::try_from(super::MIN_LOOKAHEAD).ok()?)
}

/// Match tables are complete before this stage. There is no further parallel search to
/// rendezvous: run each independent producer to completion, resolving matches locally.
#[cfg(any(feature = "gpu", test))]
fn compress_external(
    jobs: &mut [BatchJob<'_>],
    config: DeflateConfig,
    algorithm: Algorithm,
    mut resolve: impl FnMut(
        usize,
        &State<'_>,
        Option<usize>,
        MatchSearchWork,
    ) -> Result<MatchSearchResult, MatchSearchError>,
) -> Vec<BatchCompressResult> {
    jobs.iter_mut()
        .enumerate()
        .map(|(job_index, job)| {
            compress_external_job(job_index, job, config, algorithm, &mut resolve)
        })
        .collect()
}

#[cfg(any(feature = "gpu", test))]
fn compress_external_job(
    job_index: usize,
    job: &mut BatchJob<'_>,
    config: DeflateConfig,
    algorithm: Algorithm,
    resolve: &mut impl FnMut(
        usize,
        &State<'_>,
        Option<usize>,
        MatchSearchWork,
    ) -> Result<MatchSearchResult, MatchSearchError>,
) -> BatchCompressResult {
    // SAFETY: BatchRuntime accounts for only initialized output bytes via
    // the full-width output budget, including all failure paths.
    unsafe {
        job.write(|input, output| {
            let result = match BatchRuntime::new(job_index, input, output, config, algorithm) {
                Ok(mut runtime) => runtime.run(resolve),
                Err(result) => result,
            };
            (result.written, result)
        })
    }
}

/// One exclusively owned allocation reused only within this batch worker.
/// A reset after either success or output exhaustion starts a new stream.
struct CpuEncoder {
    config: DeflateConfig,
    stream: Option<DeflateStream<'static>>,
    previous_input_len: usize,
}

impl CpuEncoder {
    const fn new(config: DeflateConfig) -> Self {
        Self {
            config,
            stream: None,
            previous_input_len: 0,
        }
    }

    fn compress(&mut self, job: &mut BatchJob<'_>) -> BatchCompressResult {
        // SAFETY: compress_into_stream reports exactly the initialized
        // prefix, and clears the borrowed buffer pointers before returning.
        unsafe {
            job.write(|input, output| {
                let (written, code) = self.encode(input, output);
                (written, BatchCompressResult { written, code })
            })
        }
    }

    fn encode(
        &mut self,
        input: &[u8],
        output: &mut crate::batch::BatchOutput<'_>,
    ) -> (usize, ReturnCode) {
        let stream = if let Some(stream) = self.stream.as_mut() {
            let code = reset_impl(
                stream,
                HashHistoryReset::Independent(self.previous_input_len),
            );
            if code != ReturnCode::Ok {
                self.stream = None;
                return (0, code);
            }
            stream
        } else {
            let stream = match DeflateStream::new(self.config) {
                Ok(stream) => stream,
                Err(code) => return (0, code),
            };
            self.stream.insert(stream)
        };
        self.previous_input_len = input.len();
        compress_into_stream(stream, output, input, DeflateFlush::Finish)
    }
}

fn compress_synchronously(
    jobs: &mut [BatchJob<'_>],
    config: DeflateConfig,
) -> Vec<BatchCompressResult> {
    let mut encoder = CpuEncoder::new(config);
    jobs.iter_mut().map(|job| encoder.compress(job)).collect()
}

#[cfg(any(feature = "gpu", test))]
fn configured_algorithm(config: DeflateConfig) -> Option<Algorithm> {
    let level = match config.level {
        crate::c_api::Z_DEFAULT_COMPRESSION => 6,
        0..=9 => i8::try_from(config.level).ok()?,
        _ => return None,
    };
    Algorithm::for_stream(level, config.strategy)
}

#[cfg(any(feature = "gpu", test))]
struct BatchRuntime<'a> {
    job_index: usize,
    stream: DeflateStream<'static>,
    driver: StandardDriver,
    input_budget: SliceBudget,
    output_budget: SliceBudget,
    _buffers: PhantomData<(&'a [u8], &'a mut [core::mem::MaybeUninit<u8>])>,
}

#[cfg(any(feature = "gpu", test))]
impl<'a> BatchRuntime<'a> {
    fn new(
        job_index: usize,
        input: &'a [u8],
        output: &'a mut (impl WritableBytes + ?Sized),
        config: DeflateConfig,
        algorithm: Algorithm,
    ) -> Result<Self, BatchCompressResult> {
        let mut stream =
            DeflateStream::new(config).map_err(|code| BatchCompressResult { written: 0, code })?;
        stream.next_out = output.byte_ptr();
        // The temporary Rust owner releases its State if construction of a
        // compatible token driver fails; no partially initialized C shell.
        let Some(driver) = StandardDriver::for_algorithm_with_match_source(
            algorithm,
            &stream.state,
            MatchSourceKind::External,
        ) else {
            return Err(BatchCompressResult {
                written: 0,
                code: ReturnCode::StreamError,
            });
        };

        let mut runtime = Self {
            job_index,
            stream,
            driver,
            input_budget: SliceBudget::new(input.len()),
            output_budget: SliceBudget::new(output.len()),
            _buffers: PhantomData,
        };

        if !runtime.offer_output() {
            return Err(runtime.finish_error(ReturnCode::BufError));
        }

        // Emit raw/zlib/gzip framing through the ordinary compatibility state machine. No input is
        // attached yet, so the algorithm body cannot run in this phase.
        loop {
            let stream = &mut runtime.stream;
            stream.next_in = core::ptr::null_mut();
            stream.avail_in = 0;
            let code = deflate(stream, DeflateFlush::NoFlush);
            if code != ReturnCode::Ok {
                return Err(runtime.finish_error(code));
            }

            if stream.state.status == Status::Busy && stream.state.bit_writer.pending.is_empty() {
                break;
            }

            if stream.avail_out == 0 && !runtime.offer_output() {
                return Err(runtime.finish_error(ReturnCode::BufError));
            }
        }

        runtime.stream.next_in = input.as_ptr().cast_mut();
        runtime.offer_input();
        Ok(runtime)
    }

    fn offer_input(&mut self) -> bool {
        self.input_budget.refill(&mut self.stream.avail_in);
        self.stream.avail_in != 0
    }

    fn offer_output(&mut self) -> bool {
        self.output_budget.refill(&mut self.stream.avail_out);
        self.stream.avail_out != 0
    }

    /// Drive the entire externally searched body using one cursor snapshot.
    /// Slice budgets are updated against the live local cursors, including
    /// when the ABI's narrow availability/total fields wrap.
    fn run(
        &mut self,
        resolve: &mut impl FnMut(
            usize,
            &State<'_>,
            Option<usize>,
            MatchSearchWork,
        ) -> Result<MatchSearchResult, MatchSearchError>,
    ) -> BatchCompressResult {
        let Self {
            job_index,
            stream,
            driver,
            input_budget,
            output_budget,
            ..
        } = self;
        let body = CallCursors::with_state(stream, |state, cursors| loop {
            input_budget.refill(&mut cursors.input.avail_in);
            output_budget.refill(&mut cursors.output.remaining);
            if cursors.output.remaining == 0 {
                return Err(ReturnCode::BufError);
            }
            let mode = if input_budget.pending() == 0 {
                StandardSegmentMode::Final
            } else {
                StandardSegmentMode::Continue
            };
            match driver.step(
                state,
                &mut cursors.input,
                &mut cursors.output,
                &mut cursors.checksum,
                mode,
            ) {
                StandardTurnOutcome::Progress => {}
                StandardTurnOutcome::MatchSearch(work) => {
                    let position = absolute_match_position(
                        input_budget,
                        cursors.input.avail_in,
                        state.lookahead,
                    );
                    resolve(*job_index, state, position, work)
                        .and_then(|result| driver.complete_match_search(state, result))
                        .map_err(|_| ReturnCode::StreamError)?;
                }
                StandardTurnOutcome::NeedMoreCallerInput => {
                    input_budget.refill(&mut cursors.input.avail_in);
                    if cursors.input.avail_in == 0 {
                        return Err(ReturnCode::StreamError);
                    }
                }
                StandardTurnOutcome::NeedMoreOutput { .. } => {
                    output_budget.refill(&mut cursors.output.remaining);
                    if cursors.output.remaining == 0 {
                        return Err(ReturnCode::BufError);
                    }
                }
                StandardTurnOutcome::SegmentDone(EndBlockKind::Final) => return Ok(()),
                StandardTurnOutcome::MatchSearchError(_)
                | StandardTurnOutcome::InvalidSegmentMode
                | StandardTurnOutcome::SegmentDone(EndBlockKind::FlushBoundary) => {
                    return Err(ReturnCode::StreamError);
                }
            }
        });
        match body {
            Ok(()) => self.finish_stream(),
            Err(code) => self.finish_error(code),
        }
    }

    fn finish_stream(&mut self) -> BatchCompressResult {
        {
            let stream = &mut self.stream;
            debug_assert_eq!(stream.avail_in, 0);
            debug_assert!(stream.state.bit_writer.pending.is_empty());
            stream.state.status = Status::Finish;
            stream.next_in = core::ptr::null_mut();
            stream.avail_in = 0;
        }

        loop {
            if self.stream.avail_out == 0 && !self.offer_output() {
                return self.finish_error(ReturnCode::BufError);
            }

            match deflate(&mut self.stream, DeflateFlush::Finish) {
                ReturnCode::StreamEnd => {
                    return BatchCompressResult {
                        // The output slice budget remains full-width even
                        // when the C ABI's total_out counter wraps.
                        written: self.output_budget.processed(self.stream.avail_out),
                        code: ReturnCode::Ok,
                    };
                }
                ReturnCode::Ok => {}
                code => return self.finish_error(code),
            }
        }
    }

    const fn finish_error(&self, code: ReturnCode) -> BatchCompressResult {
        BatchCompressResult {
            written: self.output_budget.processed(self.stream.avail_out),
            code,
        }
    }
}

/// Full-width absolute match position, independent of wrapping C totals.
#[cfg(any(feature = "gpu", test))]
const fn absolute_match_position(
    budget: &SliceBudget,
    available: crate::c_api::uInt,
    lookahead: usize,
) -> Option<usize> {
    budget.processed(available).checked_sub(lookahead)
}

#[cfg(feature = "gpu")]
fn match_from_table(
    state: &State<'_>,
    absolute_position: Option<usize>,
    table: &GpuMatchTable,
) -> MatchSearchResult {
    let incumbent_length = if state.prev_length == 0 {
        STD_MIN_MATCH - 1
    } else {
        usize::from(state.prev_length)
    }
    .min(state.lookahead);
    let incumbent = MatchSearchResult::new(incumbent_length, state.match_start);

    let Some(absolute_position) = absolute_position else {
        return incumbent;
    };
    let Some((length, distance)) = table.length_distance(absolute_position) else {
        return incumbent;
    };
    let candidate_length = usize::from(length).min(state.lookahead);
    if candidate_length <= incumbent_length {
        return incumbent;
    }

    let distance = usize::from(distance);
    if !(1..=state.max_dist()).contains(&distance) || distance > absolute_position {
        return incumbent;
    }
    let Some(local_start) = state.strstart.checked_sub(distance) else {
        return incumbent;
    };
    let Ok(local_start) = u16::try_from(local_start) else {
        return incumbent;
    };

    MatchSearchResult::new(candidate_length, local_start)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deflate::{compress_bound, Method, Strategy};
    use crate::{decompress_slice, InflateConfig};

    fn independent(input: &[u8], config: DeflateConfig, capacity: usize) -> (Vec<u8>, ReturnCode) {
        let mut output = vec![0u8; capacity];
        let (written, code) = compress_slice(&mut output, input, config);
        let len = written.len();
        output.truncate(len);
        (output, code)
    }

    #[test]
    fn scheduled_runtime_reports_initialization_and_unsupported_driver_errors(
    ) -> Result<(), &'static str> {
        let input = b"batch setup failure";
        let mut output = [0u8; 128];

        let invalid = DeflateConfig {
            level: i32::MAX,
            ..DeflateConfig::default()
        };
        let algorithm = configured_algorithm(DeflateConfig::new(6))
            .ok_or("standard compressor must have an algorithm")?;
        assert!(matches!(
            BatchRuntime::new(0, input, &mut output[..], invalid, algorithm),
            Err(BatchCompressResult {
                code: ReturnCode::StreamError,
                written: 0
            })
        ));

        let stored = configured_algorithm(DeflateConfig::new(0))
            .ok_or("stored compressor must have an algorithm")?;
        assert!(matches!(
            BatchRuntime::new(0, input, &mut output[..], DeflateConfig::new(0), stored),
            Err(BatchCompressResult {
                code: ReturnCode::StreamError,
                written: 0
            })
        ));
        Ok(())
    }

    #[test]
    fn scheduled_result_size_uses_buffer_progress_not_wrapping_c_total() -> Result<(), &'static str>
    {
        let config = DeflateConfig::new(2);
        let algorithm = configured_algorithm(config).ok_or("valid standard deflate level")?;
        let input = b"scheduled output remains bounded by its caller buffer";
        let mut output = [0u8; 128];
        let output_capacity = output.len();
        let mut runtime = BatchRuntime::new(0, input, &mut output[..], config, algorithm)
            .map_err(|_| "batch runtime failed initialization")?;

        let actual = runtime.output_budget.processed(runtime.stream.avail_out);
        assert!(actual <= output_capacity);
        // C z_size may wrap at the ABI width; a batch job still reports the
        // exact number of bytes initialized in the caller-owned output slice.
        runtime.stream.total_out = crate::c_api::z_size::MAX;
        let failure = runtime.finish_error(ReturnCode::BufError);
        assert_eq!(failure.code, ReturnCode::BufError);
        assert_eq!(failure.written, actual);
        Ok(())
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_resolver_rejects_hash_chain_work_without_a_gpu_device() -> Result<(), &'static str> {
        let mut output = [0u8; 128];
        let runtime = BatchRuntime::new(
            0,
            b"external match table",
            &mut output[..],
            DeflateConfig::new(2),
            Algorithm::Fast,
        )
        .map_err(|_| "external runtime initialization failed")?;
        let request =
            MatchSearchWork::new(1, crate::deflate::longest_match::MatchSearchKind::Standard);
        assert_eq!(
            resolve_table_match(0, &runtime.stream.state, Some(0), request, &[]),
            Err(MatchSearchError::WrongBackend)
        );
        Ok(())
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_window_distance_matches_deflate_initialization() -> Result<(), &'static str> {
        let lookahead = u32::try_from(super::super::MIN_LOOKAHEAD)
            .map_err(|_| "DEFLATE lookahead must fit in u32")?;
        let cases = [
            (8, (1u32 << 9) - lookahead),
            (9, (1u32 << 9) - lookahead),
            (15, (1u32 << 15) - lookahead),
            (-15, (1u32 << 15) - lookahead),
            (31, (1u32 << 15) - lookahead),
        ];
        for (window_bits, expected) in cases {
            let config = DeflateConfig {
                window_bits,
                ..DeflateConfig::default()
            };
            assert_eq!(configured_max_dist(config), Some(expected));
        }

        for window_bits in [-8, 7, 32] {
            let config = DeflateConfig {
                window_bits,
                ..DeflateConfig::default()
            };
            assert_eq!(configured_max_dist(config), None);
        }
        Ok(())
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_match_position_ignores_wrapping_c_total() -> Result<(), &'static str> {
        let input = [0x43u8; 128];
        let mut output = [0u8; 512];
        let config = DeflateConfig::new(2);
        let algorithm = configured_algorithm(config).ok_or("valid GPU test algorithm missing")?;
        let mut runtime = BatchRuntime::new(0, &input, &mut output[..], config, algorithm)
            .map_err(|_| "unable to initialize test compressor")?;
        // The whole input is offered but no bytes consumed yet. The live C
        // counter may have wrapped without affecting this absolute position.
        runtime.stream.total_in = crate::c_api::z_size::MAX;
        assert_eq!(
            absolute_match_position(
                &runtime.input_budget,
                runtime.stream.avail_in,
                runtime.stream.state.lookahead
            ),
            Some(0)
        );

        // Simulate 23 bytes consumed and 11 bytes retained in lookahead.
        runtime.stream.avail_in -= 23;
        runtime.stream.state.lookahead = 11;
        assert_eq!(
            absolute_match_position(
                &runtime.input_budget,
                runtime.stream.avail_in,
                runtime.stream.state.lookahead
            ),
            Some(12)
        );
        runtime.stream.state.lookahead = 24;
        assert_eq!(
            absolute_match_position(
                &runtime.input_budget,
                runtime.stream.avail_in,
                runtime.stream.state.lookahead
            ),
            None
        );
        Ok(())
    }

    #[test]
    fn scheduled_match_search_is_only_used_by_lz_algorithms() {
        assert!(uses_match_search(Algorithm::Fast));
        assert!(uses_match_search(Algorithm::Medium));
        assert!(uses_match_search(Algorithm::Slow));
        assert!(!uses_match_search(Algorithm::Stored));
        assert!(!uses_match_search(Algorithm::Quick));
        assert!(!uses_match_search(Algorithm::Huffman));
        assert!(!uses_match_search(Algorithm::Rle));
    }

    #[test]
    fn malformed_match_request_only_fails_its_own_batch_job() {
        let inputs = [
            b"first-stream-rejected/".repeat(600),
            b"unaffected-second-stream/".repeat(600),
        ];
        let mut outputs = inputs
            .iter()
            .map(|input| vec![0u8; compress_bound(input.len()) + 64])
            .collect::<Vec<_>>();
        let results = {
            let mut jobs = inputs
                .iter()
                .zip(outputs.iter_mut())
                .map(|(input, output)| BatchJob::new(input, output))
                .collect::<Vec<_>>();
            compress_external(
                &mut jobs,
                DeflateConfig::new(2),
                Algorithm::Fast,
                |job_index, _, _, search| {
                    assert!(search.is_external());
                    if job_index == 0 {
                        Err(MatchSearchError::UninitializedWindow)
                    } else {
                        Ok(MatchSearchResult::new(0, 0))
                    }
                },
            )
        };

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].code, ReturnCode::StreamError);
        assert_eq!(results[1].code, ReturnCode::Ok);
        let mut decoded = vec![0u8; inputs[1].len()];
        let (decoded, code) = decompress_slice(
            &mut decoded,
            &outputs[1][..results[1].written],
            InflateConfig::default(),
        );
        assert_eq!(code, ReturnCode::Ok);
        assert_eq!(decoded, inputs[1]);
    }

    #[cfg(feature = "gpu")]
    #[test]
    #[ignore = "requires a Vulkan compute adapter"]
    fn gpu_batch_matches_independent_gpu_jobs() -> Result<(), Box<dyn std::error::Error>> {
        let searcher = GpuMatchSearcher::new(None)?;
        let inputs = [
            Vec::new(),
            b"abc".to_vec(),
            b"short output buffer/".repeat(50),
            b"long repeated prefix/".repeat(4_000),
            (0..70_123u32)
                .map(|i| (i.wrapping_mul(731) ^ (i / 13)).to_le_bytes()[0])
                .collect(),
        ];
        for (level, window_bits) in [(2, -15), (6, 9), (9, 31)] {
            let config = DeflateConfig {
                level,
                window_bits,
                ..DeflateConfig::default()
            };
            let capacities = inputs
                .iter()
                .enumerate()
                .map(|(index, input)| {
                    if index == 2 {
                        1
                    } else {
                        compress_bound(input.len()) + 64
                    }
                })
                .collect::<Vec<_>>();
            let mut batch_outputs = capacities.iter().map(|&n| vec![0; n]).collect::<Vec<_>>();
            let results = {
                let mut jobs = inputs
                    .iter()
                    .zip(&mut batch_outputs)
                    .map(|(input, output)| BatchJob::new(input, output))
                    .collect::<Vec<_>>();
                compress_slices_batch_gpu(&mut jobs, config, &searcher, 32)?
            };
            #[cfg(feature = "parallel")]
            for workers in [
                core::num::NonZeroUsize::MIN,
                core::num::NonZeroUsize::new(4).ok_or("four workers is nonzero")?,
            ] {
                let pool = BatchCodec::new(workers)?;
                let mut parallel_outputs =
                    capacities.iter().map(|&n| vec![0; n]).collect::<Vec<_>>();
                let mut jobs = inputs
                    .iter()
                    .zip(&mut parallel_outputs)
                    .map(|(input, output)| BatchJob::new(input, output))
                    .collect::<Vec<_>>();
                assert_eq!(
                    pool.compress_gpu(&mut jobs, config, &searcher, 32)?,
                    results
                );
                assert_eq!(parallel_outputs, batch_outputs);
            }
            check_gpu_vector_outputs(
                &inputs,
                &capacities,
                &batch_outputs,
                &results,
                config,
                &searcher,
            )?;
            for (index, ((input, &capacity), result)) in
                inputs.iter().zip(&capacities).zip(results).enumerate()
            {
                let mut expected = vec![0u8; capacity];
                let reference = compress_slices_batch_gpu(
                    &mut [BatchJob::new(input, &mut expected)],
                    config,
                    &searcher,
                    32,
                )?;
                assert_eq!(reference, [result]);
                assert_eq!(
                    expected[..result.written],
                    batch_outputs[index][..result.written]
                );
                if index == 2 {
                    assert_eq!(result.code, ReturnCode::BufError);
                    continue;
                }
                assert_eq!(result.code, ReturnCode::Ok);
                let mut decoded = vec![0u8; input.len()];
                let inflate_config = InflateConfig { window_bits };
                let (decoded, code) = decompress_slice(
                    &mut decoded,
                    &batch_outputs[index][..result.written],
                    inflate_config,
                );
                assert_eq!(code, ReturnCode::Ok, "level {level} index {index}");
                assert_eq!(decoded, input.as_slice());
            }
        }
        Ok(())
    }

    #[cfg(feature = "gpu")]
    fn check_gpu_vector_outputs(
        inputs: &[Vec<u8>],
        capacities: &[usize],
        batch_outputs: &[Vec<u8>],
        results: &[BatchCompressResult],
        config: DeflateConfig,
        searcher: &GpuMatchSearcher,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut vector_outputs = vec![vec![0x79; 3]; inputs.len()];
        let mut vector_jobs = inputs
            .iter()
            .zip(capacities)
            .zip(&mut vector_outputs)
            .map(|((input, &capacity), output)| BatchJob::with_vec(input, output, capacity))
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(
            compress_slices_batch_gpu(&mut vector_jobs, config, searcher, 32)?,
            results
        );
        #[cfg(feature = "parallel")]
        {
            let pool = BatchCodec::new(core::num::NonZeroUsize::new(4).ok_or("worker count")?)?;
            assert_eq!(
                pool.compress_gpu(&mut vector_jobs, config, searcher, 32)?,
                results
            );
        }
        drop(vector_jobs);
        for (vector, (reference, result)) in
            vector_outputs.iter().zip(batch_outputs.iter().zip(results))
        {
            assert_eq!(vector, &reference[..result.written]);
        }
        Ok(())
    }

    #[test]
    fn external_batch_finishes_each_stream_before_starting_the_next() {
        let inputs = [
            b"alpha/".repeat(2_000),
            b"skipped".repeat(2_000),
            b"beta/".repeat(3_000),
        ];
        let mut first = vec![0u8; compress_bound(inputs[0].len()) + 64];
        let mut short = [];
        let mut last = vec![0u8; compress_bound(inputs[2].len()) + 64];
        let mut observed = Vec::new();
        let results = compress_external(
            &mut [
                BatchJob::new(&inputs[0], &mut first),
                BatchJob::new(&inputs[1], &mut short),
                BatchJob::new(&inputs[2], &mut last),
            ],
            DeflateConfig::new(2),
            Algorithm::Fast,
            |job_index, _, _, work| {
                assert!(work.is_external());
                if observed.last() != Some(&job_index) {
                    observed.push(job_index);
                }
                Ok(MatchSearchResult::new(0, 0))
            },
        );
        assert_eq!(observed, [0, 2]);
        assert_eq!(
            results[1],
            BatchCompressResult {
                written: 0,
                code: ReturnCode::BufError
            }
        );
        for (index, output) in [(0, first), (2, last)] {
            assert_eq!(results[index].code, ReturnCode::Ok);
            let mut restored = vec![0; inputs[index].len()];
            let (decoded, code) = decompress_slice(
                &mut restored,
                &output[..results[index].written],
                InflateConfig::default(),
            );
            assert_eq!(code, ReturnCode::Ok);
            assert_eq!(decoded, inputs[index]);
        }
    }

    #[test]
    fn batch_runtime_propagates_unsolicited_completion_without_advancing(
    ) -> Result<(), &'static str> {
        let input = b"independent batch match completion";
        let mut output = [0u8; 256];
        let mut runtime = BatchRuntime::new(
            0,
            input,
            &mut output[..],
            DeflateConfig::new(2),
            Algorithm::Fast,
        )
        .map_err(|_| "unable to construct independent batch runtime")?;

        let before_strstart = runtime.stream.state.strstart;
        let before_lookahead = runtime.stream.state.lookahead;
        assert_eq!(
            runtime
                .driver
                .complete_match_search(&mut runtime.stream.state, MatchSearchResult::new(4, 1)),
            Err(MatchSearchError::UnexpectedMatchResult)
        );
        assert_eq!(runtime.stream.state.strstart, before_strstart);
        assert_eq!(runtime.stream.state.lookahead, before_lookahead);
        Ok(())
    }

    #[test]
    fn external_match_source_crosses_window_slides_without_hash_chain() -> Result<(), &'static str>
    {
        let input = (0..120_000)
            .map(|index| {
                u8::try_from((index * 73 + index / 19) % 251)
                    .map_err(|_| "generated external-match input byte is out of range")
            })
            .collect::<Result<Vec<_>, _>>()?;

        for level in [2, 6, 9] {
            let config = DeflateConfig::new(level);
            let algorithm = configured_algorithm(config)
                .ok_or("valid standard deflate algorithm is missing")?;
            assert!(uses_match_search(algorithm));

            let mut output = vec![0u8; compress_bound(input.len()) + 64];
            let results = {
                let mut jobs = [BatchJob::new(&input, &mut output)];
                compress_external(&mut jobs, config, algorithm, |_, _, _, search| {
                    assert!(search.is_external());
                    Ok(MatchSearchResult::new(0, 0))
                })
            };

            assert_eq!(results.len(), 1);
            assert_eq!(results[0].code, ReturnCode::Ok, "level {level}");

            let mut decoded = vec![0u8; input.len()];
            let (decoded, code) = decompress_slice(
                &mut decoded,
                &output[..results[0].written],
                InflateConfig::default(),
            );
            assert_eq!(code, ReturnCode::Ok, "level {level}");
            assert_eq!(decoded, input.as_slice(), "level {level}");
        }
        Ok(())
    }

    #[test]
    fn batch_matches_independent_standard_compression() -> Result<(), &'static str> {
        let inputs = [
            b"batch-alpha/".repeat(7_000),
            (0..90_000)
                .map(|i| {
                    u8::try_from((i * 37) % 251)
                        .map_err(|_| "generated batch input byte is out of range")
                })
                .collect::<Result<Vec<_>, _>>()?,
            b"zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz".repeat(3_000),
            Vec::new(),
        ];
        let algorithms = [
            (2, Strategy::Default),
            (6, Strategy::Default),
            (9, Strategy::Default),
            (6, Strategy::HuffmanOnly),
            (6, Strategy::Rle),
        ];
        let wrappers = [-crate::MAX_WBITS, crate::MAX_WBITS, 16 + crate::MAX_WBITS];

        for window_bits in wrappers {
            for (level, strategy) in algorithms {
                let config = DeflateConfig {
                    level,
                    method: Method::Deflated,
                    window_bits,
                    mem_level: 1,
                    strategy,
                };
                let references = inputs
                    .iter()
                    .map(|input| independent(input, config, compress_bound(input.len()) + 64))
                    .collect::<Vec<_>>();
                let mut outputs = inputs
                    .iter()
                    .map(|input| vec![0u8; compress_bound(input.len()) + 64])
                    .collect::<Vec<_>>();
                let mut jobs = inputs
                    .iter()
                    .zip(outputs.iter_mut())
                    .map(|(input, output)| BatchJob::new(input, output))
                    .collect::<Vec<_>>();

                let results = compress_slices_batch(&mut jobs, config);
                drop(jobs);

                assert_eq!(results.len(), inputs.len());
                for (index, result) in results.into_iter().enumerate() {
                    let (reference, reference_code) = &references[index];
                    assert_eq!(result.code, *reference_code);
                    assert_eq!(result.written, reference.len());
                    assert_eq!(
                        &outputs[index][..result.written],
                        reference,
                        "batch output changed for job {index}, level {level}, strategy {strategy:?}, window_bits {window_bits}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn batch_falls_back_for_stored_and_quick() {
        let inputs = [
            b"stored-or-quick".repeat(1_000),
            b"other input".repeat(2_000),
        ];
        for level in [0, 1] {
            let config = DeflateConfig {
                level,
                ..DeflateConfig::default()
            };
            let references = inputs
                .iter()
                .map(|input| independent(input, config, compress_bound(input.len())))
                .collect::<Vec<_>>();
            let mut outputs = inputs
                .iter()
                .map(|input| vec![0u8; compress_bound(input.len())])
                .collect::<Vec<_>>();
            let mut jobs = inputs
                .iter()
                .zip(outputs.iter_mut())
                .map(|(input, output)| BatchJob::new(input, output))
                .collect::<Vec<_>>();

            let results = compress_slices_batch(&mut jobs, config);
            drop(jobs);

            for (index, result) in results.into_iter().enumerate() {
                let (reference, code) = &references[index];
                assert_eq!(result.code, *code);
                assert_eq!(&outputs[index][..result.written], reference);
            }
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn explicit_single_worker_batch_matches_scalar() -> Result<(), &'static str> {
        let input = b"single-worker-batch/".repeat(4_000);
        let config = DeflateConfig::new(9);
        let (reference, code) = independent(&input, config, compress_bound(input.len()) + 64);
        let mut output = vec![0u8; compress_bound(input.len()) + 64];
        let compressor = BatchCodec::new(core::num::NonZeroUsize::MIN)
            .map_err(|_| "one-worker batch compressor failed initialization")?;
        let results = {
            let mut jobs = [BatchJob::new(&input, &mut output)];
            compressor.compress(&mut jobs, config)
        };

        assert_eq!(compressor.workers(), 1);
        assert_eq!(results[0].code, code);
        assert_eq!(results[0].written, reference.len());
        assert_eq!(&output[..results[0].written], reference);
        Ok(())
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_streams_match_independent_compression() -> Result<(), &'static str> {
        let inputs = [
            b"parallel-alpha/".repeat(8_000),
            (0..180_000)
                .map(|i| {
                    u8::try_from((i * 73 + i / 19) % 251)
                        .map_err(|_| "generated parallel input byte is out of range")
                })
                .collect::<Result<Vec<_>, _>>()?,
            b"zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz".repeat(5_000),
            Vec::new(),
        ];
        let configs = [
            DeflateConfig::new(0),
            DeflateConfig::new(1),
            DeflateConfig::new(6),
            DeflateConfig::new(9),
            DeflateConfig {
                level: 6,
                method: Method::Deflated,
                window_bits: -crate::MAX_WBITS,
                mem_level: 1,
                strategy: Strategy::Default,
            },
            DeflateConfig {
                level: 6,
                method: Method::Deflated,
                window_bits: 16 + crate::MAX_WBITS,
                mem_level: 1,
                strategy: Strategy::Default,
            },
        ];

        for config in configs {
            let references = inputs
                .iter()
                .map(|input| independent(input, config, compress_bound(input.len()) + 64))
                .collect::<Vec<_>>();
            let mut outputs = inputs
                .iter()
                .map(|input| vec![0u8; compress_bound(input.len()) + 64])
                .collect::<Vec<_>>();
            let mut jobs = inputs
                .iter()
                .zip(outputs.iter_mut())
                .map(|(input, output)| BatchJob::new(input, output))
                .collect::<Vec<_>>();

            let workers = core::num::NonZeroUsize::new(4).ok_or("four workers must be nonzero")?;
            let compressor = BatchCodec::new(workers)
                .map_err(|_| "four-worker batch compressor failed initialization")?;
            assert_eq!(compressor.workers(), 4);
            let results = compressor.compress(&mut jobs, config);
            drop(jobs);

            for (index, result) in results.into_iter().enumerate() {
                let (reference, code) = &references[index];
                assert_eq!(
                    result.code, *code,
                    "code changed for job {index}, {config:?}"
                );
                assert_eq!(result.written, reference.len());
                assert_eq!(
                    &outputs[index][..result.written],
                    reference,
                    "parallel output changed for job {index}, {config:?}"
                );
            }
        }
        Ok(())
    }

    fn check_vector_jobs(
        mut compress: impl FnMut(&mut [BatchJob<'_>], DeflateConfig) -> Vec<BatchCompressResult>,
    ) -> Result<(), std::collections::TryReserveError> {
        let input = b"vector-backed-batch/".repeat(8000);
        for level in [0, 1, 6, 9] {
            let config = DeflateConfig::new(level);
            for cap in [0, 1, 31, compress_bound(input.len())] {
                let (reference, code) = independent(&input, config, cap);
                let mut vector = Vec::with_capacity(cap.max(16));
                vector.extend_from_slice(b"previous output");
                let original = vector.as_ptr();
                let mut slice = vec![0xa5u8; cap];
                {
                    let mut jobs = [
                        BatchJob::with_vec(&input, &mut vector, cap)?,
                        BatchJob::new(&input, &mut slice),
                    ];
                    let expected = [BatchCompressResult {
                        written: reference.len(),
                        code,
                    }; 2];
                    for _ in 0..2 {
                        assert_eq!(compress(&mut jobs, config), expected);
                    }
                }
                assert_eq!(vector.as_ptr(), original);
                assert_eq!(vector, reference);
                assert_eq!(&slice[..reference.len()], reference);
                assert!(slice[reference.len()..].iter().all(|&b| b == 0xa5));
            }
        }
        Ok(())
    }

    #[test]
    fn batch_vectors_reuse_capacity_and_commit_only_written_prefix(
    ) -> Result<(), std::collections::TryReserveError> {
        check_vector_jobs(compress_slices_batch)
    }

    #[test]
    fn independent_vector_jobs_reuse_capacity_and_commit_only_written_prefix(
    ) -> Result<(), std::collections::TryReserveError> {
        check_vector_jobs(|jobs, config| jobs.iter_mut().map(|job| job.compress(config)).collect())
    }

    #[test]
    fn sparse_batch_reset_clears_every_hash_bucket() -> Result<(), Box<dyn std::error::Error>> {
        let mut seed = 0x4d89_8b27u32;
        let random = (0..1025)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                seed.to_le_bytes()[0]
            })
            .collect::<Vec<_>>();
        let repeated = b"abc0123456789a".repeat(80);
        for level in 0..=9 {
            for strategy in [
                Strategy::Default,
                Strategy::Filtered,
                Strategy::Fixed,
                Strategy::Rle,
                Strategy::HuffmanOnly,
            ] {
                for window_bits in [-9, 9, 15, 31] {
                    let config = DeflateConfig {
                        level,
                        window_bits,
                        mem_level: 1,
                        strategy,
                        ..DeflateConfig::default()
                    };
                    let mut encoder = CpuEncoder::new(config);
                    for bytes in [random.as_slice(), repeated.as_slice()] {
                        for len in [
                            0, 1, 2, 3, 4, 5, 7, 15, 31, 63, 127, 255, 511, 512, 513, 1024,
                        ] {
                            for cap in [0, 1, 11, compress_bound(len) + 64] {
                                let input = &bytes[..len];
                                let (expected, code) = independent(input, config, cap);
                                let mut output = vec![0xa5; cap];
                                let result =
                                    encoder.compress(&mut BatchJob::new(input, &mut output));
                                assert_eq!(
                                    result,
                                    BatchCompressResult {
                                        written: expected.len(),
                                        code
                                    }
                                );
                                assert_eq!(&output[..expected.len()], expected);
                                let stream = encoder.stream.as_mut().ok_or("owned stream")?;
                                assert_eq!(
                                    reset_impl(stream, HashHistoryReset::Independent(len)),
                                    ReturnCode::Ok
                                );
                                assert!(stream.state.head.as_slice().iter().all(|&head| head == 0));
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn reused_compressor_preserves_independent_streams_after_partial_output(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let big = b"quick-reset-state/".repeat(5000);
        let small = b"independent-neighbor";
        let inputs: [&[u8]; 8] = [&big, small, b"", &big, small, &big, b"", small];
        let caps = [7, 256, 0, compress_bound(big.len()), 1, 19, 64, 256];
        #[cfg(feature = "parallel")]
        let pools = [
            BatchCodec::new(core::num::NonZeroUsize::MIN)?,
            BatchCodec::new(core::num::NonZeroUsize::new(4).ok_or("workers")?)?,
        ];
        for level in 0..=9 {
            for strategy in [
                Strategy::Default,
                Strategy::Filtered,
                Strategy::Fixed,
                Strategy::Rle,
                Strategy::HuffmanOnly,
            ] {
                for window_bits in [-15, 15, 31] {
                    let config = DeflateConfig {
                        level,
                        window_bits,
                        strategy,
                        ..DeflateConfig::default()
                    };
                    let references = inputs
                        .iter()
                        .zip(caps)
                        .map(|(input, cap)| independent(input, config, cap))
                        .collect::<Vec<_>>();
                    let mut outputs = caps.map(|cap| vec![0xa5; cap]);
                    let mut jobs = inputs
                        .iter()
                        .zip(&mut outputs)
                        .map(|(input, output)| BatchJob::new(input, output))
                        .collect::<Vec<_>>();
                    let results = compress_slices_batch(&mut jobs, config);
                    for (result, (bytes, code)) in results.iter().zip(&references) {
                        assert_eq!(
                            *result,
                            BatchCompressResult {
                                written: bytes.len(),
                                code: *code
                            },
                            "level {level} strategy {strategy:?} window {window_bits}"
                        );
                    }
                    drop(jobs);
                    for (output, (bytes, _)) in outputs.iter().zip(&references) {
                        assert_eq!(&output[..bytes.len()], bytes);
                        assert!(output[bytes.len()..].iter().all(|&b| b == 0xa5));
                    }
                    let check_vectors = |run: &dyn Fn(&mut [BatchJob<'_>]) -> Vec<BatchCompressResult>| -> Result<(), std::collections::TryReserveError> {
                        let mut vectors = caps.map(Vec::with_capacity);
                        let mut jobs = inputs.iter().zip(&mut vectors).zip(caps)
                            .map(|((input, output), cap)| BatchJob::with_vec(input, output, cap))
                            .collect::<Result<Vec<_>, _>>()?;
                        assert_eq!(run(&mut jobs), results);
                        drop(jobs);
                        for (output, (bytes, _)) in vectors.iter().zip(&references) {
                            assert_eq!(output, bytes);
                        }
                        Ok(())
                    };
                    check_vectors(&|jobs| compress_slices_batch(jobs, config))?;
                    #[cfg(feature = "parallel")]
                    for pool in &pools {
                        check_vectors(&|jobs| pool.compress(jobs, config))?;
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn batch_reports_the_same_tiny_output_failure() {
        let input = b"tiny-output-batch/".repeat(4_000);
        let config = DeflateConfig::new(6);
        let (reference, reference_code) = independent(&input, config, 4);
        let mut output = [0u8; 4];
        let results = {
            let mut jobs = [BatchJob::new(&input, &mut output)];
            compress_slices_batch(&mut jobs, config)
        };

        assert_eq!(results[0].code, reference_code);
        assert_eq!(results[0].written, reference.len());
        assert_eq!(&output[..results[0].written], reference);
    }
}
