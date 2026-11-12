use core::marker::PhantomData;
use std::vec::Vec;

#[cfg(feature = "gpu")]
use super::gpu_match::{GpuMatchError, GpuMatchSearcher, GpuMatchTable};
#[cfg(feature = "gpu")]
use super::STD_MIN_MATCH;
use super::{
    algorithm::{
        Algorithm, EndBlockKind, MatchSourceKind, StandardDriver, StandardSegmentMode,
        StandardTurnOutcome,
    },
    compress_slice, deflate,
    longest_match::{
        execute_match_search_batch_with_backend, MatchSearchRequest, MatchSearchRequestError,
        MatchSearchResult, MatchSearchWork, ScopedThreadMatchSearchBackend,
    },
    CallCursors, DeflateConfig, DeflateFlush, DeflateStream, ReturnCode, Status,
};
use crate::abi_cursor::SliceBudget;
#[cfg(feature = "gpu")]
use crate::{MAX_WBITS, MIN_WBITS};

/// One independent input/output pair for [`compress_slices_batch`].
///
/// The output buffer may be smaller than the compressed stream. In that case the corresponding
/// [`BatchCompressResult`] reports [`ReturnCode::BufError`] and `written` is the initialized prefix.
pub struct BatchCompressJob<'a> {
    pub input: &'a [u8],
    pub output: &'a mut [u8],
}

impl<'a> BatchCompressJob<'a> {
    pub const fn new(input: &'a [u8], output: &'a mut [u8]) -> Self {
        Self { input, output }
    }
}

/// Result for one job passed to [`compress_slices_batch`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchCompressResult {
    pub written: usize,
    pub code: ReturnCode,
}

/// Match-search executor used by [`compress_slices_batch_with_backend`].
///
/// No automatic policy is applied: scalar remains the default, while scoped threads are an explicit
/// caller choice. A scoped-thread worker count of zero is treated as one worker. The scoped-thread
/// backend creates OS threads for each concurrent match-search batch; it is primarily a concrete
/// backend-boundary implementation, not a promise that threading will improve throughput. Measure
/// it on the intended workload before selecting it for performance.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BatchMatchSearchBackend {
    #[default]
    Scalar,
    ScopedThreads {
        workers: usize,
    },
}

/// Compress several independent buffers with scalar per-stream compression.
///
/// Every job uses the same configuration and produces the same bytes as an independent
/// [`super::compress_slice`] call. The default scalar backend deliberately drives each stream to
/// completion instead of round-robin scheduling compressor states: without a concurrent compute
/// executor there is no cross-stream work to rendezvous, and preserving per-stream cache locality is
/// both simpler and substantially cheaper.
///
/// Use [`compress_slices_batch_with_backend`] to opt into a backend that can execute detached
/// longest-match searches concurrently. Stored and Quick compression always use the ordinary
/// synchronous slice compressor because their block mechanics do not use that search backend.
pub fn compress_slices_batch(
    jobs: &mut [BatchCompressJob<'_>],
    config: DeflateConfig,
) -> Vec<BatchCompressResult> {
    compress_slices_batch_with_backend(jobs, config, BatchMatchSearchBackend::Scalar)
}

/// Reusable worker pool for whole-stream batch compression.
///
/// Each scheduled task owns one complete independent stream, preserving per-stream cache locality
/// while allowing the pool to work-steal streams with very different compression costs. The worker
/// count is explicit and belongs to this object; no process-global Rayon pool is consulted or
/// configured. Output bytes and return codes match independent [`super::compress_slice`] calls
/// job-for-job.
///
/// The pool is available only with the optional `parallel` feature. Construct it once and reuse it
/// across batches so thread creation is not part of the compression hot path.
#[cfg(feature = "parallel")]
pub struct BatchCompressor {
    pool: rayon_core::ThreadPool,
}

#[cfg(feature = "parallel")]
impl BatchCompressor {
    /// Create a reusable batch compressor with exactly `workers` worker threads.
    /// # Errors
    ///
    /// Returns a Rayon pool-construction error if the worker threads cannot be created.
    pub fn new(workers: core::num::NonZeroUsize) -> Result<Self, rayon_core::ThreadPoolBuildError> {
        rayon_core::ThreadPoolBuilder::new()
            .num_threads(workers.get())
            .build()
            .map(|pool| Self { pool })
    }

    /// Number of workers owned by this compressor.
    #[must_use]
    pub fn workers(&self) -> usize {
        self.pool.current_num_threads()
    }

    /// Compress several independent buffers concurrently on this compressor's worker pool.
    pub fn compress(
        &self,
        jobs: &mut [BatchCompressJob<'_>],
        config: DeflateConfig,
    ) -> Vec<BatchCompressResult> {
        let mut results = vec![
            BatchCompressResult {
                written: 0,
                code: ReturnCode::StreamError,
            };
            jobs.len()
        ];
        if jobs.is_empty() {
            return results;
        }
        if jobs.len() == 1 || self.workers() == 1 {
            return compress_synchronously(jobs, config);
        }

        // Independent streams vary by orders of magnitude in real workloads. Scheduling one complete
        // stream per task lets the persistent pool steal remaining work instead of pinning an
        // arbitrary equal-count chunk of streams to one worker and waiting for its largest member.
        self.pool.scope(|scope| {
            for (job, result) in jobs.iter_mut().zip(&mut results) {
                scope.spawn(move |_| {
                    let (output, code) = compress_slice(&mut *job.output, job.input, config);
                    *result = BatchCompressResult {
                        written: output.len(),
                        code,
                    };
                });
            }
        });

        results
    }
}

/// Variant of [`compress_slices_batch`] with an explicit match-search executor.
pub fn compress_slices_batch_with_backend(
    jobs: &mut [BatchCompressJob<'_>],
    config: DeflateConfig,
    backend: BatchMatchSearchBackend,
) -> Vec<BatchCompressResult> {
    // With no concurrent search executor there is no cross-stream work to rendezvous. Driving each
    // stream to completion preserves compressor-state cache locality and avoids paying scheduler
    // turns for a batch that can never execute compute concurrently.
    if matches!(
        backend,
        BatchMatchSearchBackend::Scalar | BatchMatchSearchBackend::ScopedThreads { workers: 0 | 1 }
    ) {
        return compress_synchronously(jobs, config);
    }

    let Some(algorithm) = configured_algorithm(config) else {
        return compress_synchronously(jobs, config);
    };
    if !uses_match_search(algorithm) {
        return compress_synchronously(jobs, config);
    }

    let BatchMatchSearchBackend::ScopedThreads { workers } = backend else {
        unreachable!("scalar backends returned before scheduled compression")
    };
    let mut resolver = ScopedThreadResolver::new(workers);
    compress_scheduled(
        jobs,
        config,
        algorithm,
        MatchSourceKind::HashChain,
        &mut resolver,
    )
}

/// Compress complete input buffers using a persistent GPU whole-input match table.
///
/// The GPU device and compute pipeline belong to `searcher` and are reused across calls. Each input
/// receives one whole-input table before compression begins; when the ordinary producer yields a
/// detached match-search request, the scheduler resolves it from that table instead of walking the
/// hash chain on the CPU. Framing, parsing, token emission, Huffman coding, and output draining stay
/// on the existing CPU path.
///
/// # Errors
///
/// Returns an error if the GPU match search cannot accept the workload,
/// allocate, dispatch, or retrieve its results.
#[cfg(feature = "gpu")]
pub fn compress_slices_batch_gpu(
    jobs: &mut [BatchCompressJob<'_>],
    config: DeflateConfig,
    searcher: &GpuMatchSearcher,
    search_budget: u32,
) -> Result<Vec<BatchCompressResult>, GpuMatchError> {
    let Some(algorithm) = configured_algorithm(config) else {
        return Ok(compress_synchronously(jobs, config));
    };
    if !uses_match_search(algorithm) {
        return Ok(compress_synchronously(jobs, config));
    }

    let Some(max_dist) = configured_max_dist(config) else {
        return Ok(compress_synchronously(jobs, config));
    };
    let inputs = jobs.iter().map(|job| job.input).collect::<Vec<_>>();
    let tables = searcher.search_batch_with_max_dist(&inputs, search_budget, max_dist)?;
    let mut resolver = WholeInputTableResolver { tables: &tables };
    Ok(compress_scheduled(
        jobs,
        config,
        algorithm,
        MatchSourceKind::External,
        &mut resolver,
    ))
}

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

trait MatchResolver {
    /// Return one outcome per job. A malformed request must not abort another
    /// independent stream in the same batch.
    fn resolve(
        &mut self,
        runtimes: &[BatchRuntime<'_>],
        work: &[(usize, MatchSearchWork)],
        results: &mut [Result<MatchSearchResult, MatchSearchRequestError>],
    );
}

struct ScopedThreadResolver {
    workers: usize,
    // Reuse owned result/index buffers between scheduling turns. The borrowed
    // request vector must be reconstructed for each live State snapshot.
    request_indexes: Vec<usize>,
    found: Vec<MatchSearchResult>,
}

impl ScopedThreadResolver {
    const fn new(workers: usize) -> Self {
        Self {
            workers,
            request_indexes: Vec::new(),
            found: Vec::new(),
        }
    }
}

impl MatchResolver for ScopedThreadResolver {
    fn resolve(
        &mut self,
        runtimes: &[BatchRuntime<'_>],
        work: &[(usize, MatchSearchWork)],
        results: &mut [Result<MatchSearchResult, MatchSearchRequestError>],
    ) {
        if work.len() == 1 {
            let (runtime_index, search) = work[0];
            results[0] = runtimes[runtime_index]
                .match_request(search)
                .map(|request| request.execute());
            return;
        }

        self.request_indexes.clear();
        let mut requests = Vec::<MatchSearchRequest<'_>>::with_capacity(work.len());
        for (index, (runtime_index, search)) in work.iter().copied().enumerate() {
            match runtimes[runtime_index].match_request(search) {
                Ok(request) => {
                    requests.push(request);
                    self.request_indexes.push(index);
                }
                Err(error) => results[index] = Err(error),
            }
        }
        if requests.is_empty() {
            return;
        }

        self.found.resize(
            requests.len(),
            MatchSearchResult {
                length: 0,
                start: 0,
            },
        );
        execute_match_search_batch_with_backend(
            &ScopedThreadMatchSearchBackend::new(self.workers),
            &requests,
            &mut self.found,
        );
        for (index, result) in self
            .request_indexes
            .iter()
            .copied()
            .zip(self.found.iter().copied())
        {
            results[index] = Ok(result);
        }
    }
}

#[cfg(feature = "gpu")]
struct WholeInputTableResolver<'a> {
    tables: &'a [GpuMatchTable],
}

#[cfg(feature = "gpu")]
impl MatchResolver for WholeInputTableResolver<'_> {
    fn resolve(
        &mut self,
        runtimes: &[BatchRuntime<'_>],
        work: &[(usize, MatchSearchWork)],
        results: &mut [Result<MatchSearchResult, MatchSearchRequestError>],
    ) {
        for ((runtime_index, search), result) in work.iter().zip(results.iter_mut()) {
            if !search.is_external() {
                *result = Err(MatchSearchRequestError::WrongBackend);
                continue;
            }
            let runtime = &runtimes[*runtime_index];
            *result = Ok(runtime.match_from_table(&self.tables[runtime.job_index]));
        }
    }
}

fn compress_scheduled(
    jobs: &mut [BatchCompressJob<'_>],
    config: DeflateConfig,
    algorithm: Algorithm,
    match_source: MatchSourceKind,
    resolver: &mut impl MatchResolver,
) -> Vec<BatchCompressResult> {
    let mut results = vec![
        BatchCompressResult {
            written: 0,
            code: ReturnCode::StreamError,
        };
        jobs.len()
    ];
    let mut runtimes = Vec::with_capacity(jobs.len());

    for (job_index, job) in jobs.iter_mut().enumerate() {
        match BatchRuntime::new(
            job_index,
            job.input,
            &mut *job.output,
            config,
            algorithm,
            match_source,
        ) {
            Ok(runtime) => runtimes.push(runtime),
            Err(result) => results[job_index] = result,
        }
    }

    let mut active = runtimes.len();
    let mut match_work = Vec::<(usize, MatchSearchWork)>::with_capacity(runtimes.len());
    // Every resolver must overwrite all slots for its dispatched request count.
    let mut match_results = vec![Err(MatchSearchRequestError::InvalidHashHistory); runtimes.len()];

    while active != 0 {
        match_work.clear();

        for (runtime_index, runtime) in runtimes.iter_mut().enumerate() {
            if runtime.done {
                continue;
            }

            match runtime.step() {
                RuntimeStep::Progress => {}
                RuntimeStep::MatchSearch(work) => match_work.push((runtime_index, work)),
                RuntimeStep::Done(result) => {
                    results[runtime.job_index] = result;
                    active -= 1;
                }
            }
        }

        if match_work.is_empty() {
            continue;
        }

        resolver.resolve(
            &runtimes,
            &match_work,
            &mut match_results[..match_work.len()],
        );
        for ((runtime_index, _), result) in match_work
            .iter()
            .copied()
            .zip(match_results.iter().copied())
        {
            if let Ok(match_result) = result {
                runtimes[runtime_index].complete_match_search(match_result);
            } else {
                let runtime = &mut runtimes[runtime_index];
                results[runtime.job_index] = runtime.finish_error(ReturnCode::StreamError);
                active -= 1;
            }
        }
    }

    results
}

fn compress_synchronously(
    jobs: &mut [BatchCompressJob<'_>],
    config: DeflateConfig,
) -> Vec<BatchCompressResult> {
    jobs.iter_mut()
        .map(|job| {
            let (output, code) = compress_slice(&mut *job.output, job.input, config);
            BatchCompressResult {
                written: output.len(),
                code,
            }
        })
        .collect()
}

fn configured_algorithm(config: DeflateConfig) -> Option<Algorithm> {
    let level = match config.level {
        crate::c_api::Z_DEFAULT_COMPRESSION => 6,
        0..=9 => i8::try_from(config.level).ok()?,
        _ => return None,
    };
    Some(Algorithm::for_stream(level, config.strategy))
}

struct BatchRuntime<'a> {
    job_index: usize,
    stream: DeflateStream<'static>,
    driver: StandardDriver,
    input_budget: SliceBudget,
    output_budget: SliceBudget,
    done: bool,
    _buffers: PhantomData<(&'a [u8], &'a mut [u8])>,
}

enum RuntimeStep {
    Progress,
    MatchSearch(MatchSearchWork),
    Done(BatchCompressResult),
}

impl<'a> BatchRuntime<'a> {
    fn new(
        job_index: usize,
        input: &'a [u8],
        output: &'a mut [u8],
        config: DeflateConfig,
        algorithm: Algorithm,
        match_source: MatchSourceKind,
    ) -> Result<Self, BatchCompressResult> {
        let mut stream =
            DeflateStream::new(config).map_err(|code| BatchCompressResult { written: 0, code })?;
        stream.next_out = output.as_mut_ptr();
        // The temporary Rust owner releases its State if construction of a
        // compatible token driver fails; no partially initialized C shell.
        let Some(driver) =
            StandardDriver::for_algorithm_with_match_source(algorithm, &stream.state, match_source)
        else {
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
            done: false,
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

    fn step(&mut self) -> RuntimeStep {
        if self.stream.avail_in == 0 {
            self.offer_input();
        }
        if self.stream.avail_out == 0 && !self.offer_output() {
            return RuntimeStep::Done(self.finish_error(ReturnCode::BufError));
        }

        let mode = if self.input_budget.pending() == 0 {
            StandardSegmentMode::Final
        } else {
            StandardSegmentMode::Continue
        };

        let Self { stream, driver, .. } = self;
        let outcome = CallCursors::with_state(stream, |state, cursors| {
            driver.step(
                state,
                &mut cursors.input,
                &mut cursors.output,
                &mut cursors.checksum,
                mode,
            )
        });

        match outcome {
            StandardTurnOutcome::Progress => RuntimeStep::Progress,
            StandardTurnOutcome::MatchSearch(work) => RuntimeStep::MatchSearch(work),
            StandardTurnOutcome::NeedMoreCallerInput => {
                if self.offer_input() {
                    RuntimeStep::Progress
                } else {
                    RuntimeStep::Done(self.finish_error(ReturnCode::StreamError))
                }
            }
            StandardTurnOutcome::NeedMoreOutput { .. } => {
                if self.offer_output() {
                    RuntimeStep::Progress
                } else {
                    RuntimeStep::Done(self.finish_error(ReturnCode::BufError))
                }
            }
            StandardTurnOutcome::SegmentDone(EndBlockKind::Final) => {
                RuntimeStep::Done(self.finish_stream())
            }
            StandardTurnOutcome::MatchSearchError(_)
            | StandardTurnOutcome::SegmentDone(EndBlockKind::FlushBoundary) => {
                RuntimeStep::Done(self.finish_error(ReturnCode::StreamError))
            }
        }
    }

    fn match_request(
        &self,
        work: MatchSearchWork,
    ) -> Result<MatchSearchRequest<'_>, MatchSearchRequestError> {
        work.request(&self.stream.state)
    }

    /// Absolute index in the full input slice of the next token to resolve.
    /// The raw zlib total can wrap (notably 32-bit C long on Windows), while
    /// the issued-slice budget retains the full position without changing the
    /// C stream representation.
    #[cfg(feature = "gpu")]
    fn absolute_match_position(&self) -> Option<usize> {
        self.input_budget
            .processed(self.stream.avail_in)
            .checked_sub(self.stream.state.lookahead)
    }

    #[cfg(feature = "gpu")]
    fn match_from_table(&self, table: &GpuMatchTable) -> MatchSearchResult {
        let state = &*self.stream.state;
        let incumbent_length = if state.prev_length == 0 {
            STD_MIN_MATCH - 1
        } else {
            usize::from(state.prev_length)
        }
        .min(state.lookahead);
        let incumbent = MatchSearchResult::new(incumbent_length, state.match_start);

        let Some(absolute_position) = self.absolute_match_position() else {
            return incumbent;
        };
        let Some(candidate) = table.get(absolute_position) else {
            return incumbent;
        };
        let Ok(candidate_length) = usize::try_from(candidate.length) else {
            return incumbent;
        };
        let candidate_length = candidate_length.min(state.lookahead);
        if candidate_length <= incumbent_length {
            return incumbent;
        }

        let Ok(candidate_start) = usize::try_from(candidate.start) else {
            return incumbent;
        };
        let Some(distance) = absolute_position.checked_sub(candidate_start) else {
            return incumbent;
        };
        if !(1..=state.max_dist()).contains(&distance) {
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

    fn complete_match_search(&mut self, result: MatchSearchResult) {
        let Self { stream, driver, .. } = self;
        driver.complete_match_search(&mut stream.state, result);
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
                    self.done = true;
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

    const fn finish_error(&mut self, code: ReturnCode) -> BatchCompressResult {
        self.done = true;
        BatchCompressResult {
            written: self.output_budget.processed(self.stream.avail_out),
            code,
        }
    }
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
            BatchRuntime::new(
                0,
                input,
                &mut output,
                invalid,
                algorithm,
                MatchSourceKind::HashChain,
            ),
            Err(BatchCompressResult {
                code: ReturnCode::StreamError,
                written: 0
            })
        ));

        let stored = configured_algorithm(DeflateConfig::new(0))
            .ok_or("stored compressor must have an algorithm")?;
        assert!(matches!(
            BatchRuntime::new(
                0,
                input,
                &mut output,
                DeflateConfig::new(0),
                stored,
                MatchSourceKind::HashChain,
            ),
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
        let mut runtime = BatchRuntime::new(
            0,
            input,
            &mut output,
            config,
            algorithm,
            MatchSourceKind::HashChain,
        )
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
    fn gpu_resolver_rejects_hash_chain_work_without_a_gpu_device() {
        let mut resolver = WholeInputTableResolver { tables: &[] };
        let mut outcomes = [Err(MatchSearchRequestError::InvalidHashHistory)];
        let request =
            MatchSearchWork::new(1, crate::deflate::longest_match::MatchSearchKind::Standard);
        resolver.resolve(&[], &[(0, request)], &mut outcomes);
        assert_eq!(outcomes[0], Err(MatchSearchRequestError::WrongBackend));
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
        let mut runtime = BatchRuntime::new(
            0,
            &input,
            &mut output,
            config,
            algorithm,
            MatchSourceKind::HashChain,
        )
        .map_err(|_| "unable to initialize test compressor")?;
        // The whole input is offered but no bytes consumed yet. The live C
        // counter may have wrapped without affecting this absolute position.
        runtime.stream.total_in = crate::c_api::z_size::MAX;
        assert_eq!(runtime.absolute_match_position(), Some(0));

        // Simulate 23 bytes consumed and 11 bytes retained in lookahead.
        runtime.stream.avail_in -= 23;
        runtime.stream.state.lookahead = 11;
        assert_eq!(runtime.absolute_match_position(), Some(12));
        runtime.stream.state.lookahead = 24;
        assert_eq!(runtime.absolute_match_position(), None);
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

    struct NoMatchExternalResolver;

    impl MatchResolver for NoMatchExternalResolver {
        fn resolve(
            &mut self,
            _runtimes: &[BatchRuntime<'_>],
            work: &[(usize, MatchSearchWork)],
            results: &mut [Result<MatchSearchResult, MatchSearchRequestError>],
        ) {
            assert_eq!(work.len(), results.len());
            for ((_, search), result) in work.iter().zip(results.iter_mut()) {
                assert!(search.is_external());
                *result = Ok(MatchSearchResult {
                    length: 0,
                    start: 0,
                });
            }
        }
    }

    struct RejectFirstExternalResolver;

    impl MatchResolver for RejectFirstExternalResolver {
        fn resolve(
            &mut self,
            runtimes: &[BatchRuntime<'_>],
            work: &[(usize, MatchSearchWork)],
            results: &mut [Result<MatchSearchResult, MatchSearchRequestError>],
        ) {
            for ((runtime_index, request), result) in work.iter().zip(results.iter_mut()) {
                assert!(request.is_external());
                *result = if runtimes[*runtime_index].job_index == 0 {
                    Err(MatchSearchRequestError::UninitializedWindow)
                } else {
                    Ok(MatchSearchResult::new(0, 0))
                };
            }
        }
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
                .map(|(input, output)| BatchCompressJob::new(input, output))
                .collect::<Vec<_>>();
            compress_scheduled(
                &mut jobs,
                DeflateConfig::new(2),
                Algorithm::Fast,
                MatchSourceKind::External,
                &mut RejectFirstExternalResolver,
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

    #[test]
    fn scoped_resolver_reports_invalid_state_per_job() -> Result<(), &'static str> {
        let input = b"typed search window";
        let mut first_output = [0u8; 256];
        let mut second_output = [0u8; 256];
        let config = DeflateConfig::new(2);
        let mut first = BatchRuntime::new(
            0,
            input,
            &mut first_output,
            config,
            Algorithm::Fast,
            MatchSourceKind::HashChain,
        )
        .map_err(|_| "first runtime initialization failed")?;
        let second = BatchRuntime::new(
            1,
            input,
            &mut second_output,
            config,
            Algorithm::Fast,
            MatchSourceKind::HashChain,
        )
        .map_err(|_| "second runtime initialization failed")?;
        first.stream.state.strstart = 1;
        first.stream.state.lookahead = usize::MAX;

        let runtimes = [first, second];
        let search =
            MatchSearchWork::new(1, crate::deflate::longest_match::MatchSearchKind::Standard);
        let mut results = [
            Err(MatchSearchRequestError::InvalidHashHistory),
            Err(MatchSearchRequestError::InvalidHashHistory),
        ];
        ScopedThreadResolver::new(2).resolve(&runtimes, &[(0, search), (1, search)], &mut results);
        assert_eq!(
            results[0],
            Err(MatchSearchRequestError::WindowPositionOverflow)
        );
        assert!(results[1].is_ok());
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
                let mut jobs = [BatchCompressJob::new(&input, &mut output)];
                let mut resolver = NoMatchExternalResolver;
                compress_scheduled(
                    &mut jobs,
                    config,
                    algorithm,
                    MatchSourceKind::External,
                    &mut resolver,
                )
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
                    .map(|(input, output)| BatchCompressJob::new(input, output))
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
                .map(|(input, output)| BatchCompressJob::new(input, output))
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

    #[test]
    fn scoped_thread_backend_matches_scalar_batch_output() -> Result<(), &'static str> {
        let inputs = [
            b"threaded-search-alpha/".repeat(6_000),
            b"threaded-search-beta/beta/beta/".repeat(5_000),
            (0..140_000)
                .map(|i| {
                    u8::try_from(((i / 17) * 31 + i % 7) % 251)
                        .map_err(|_| "generated threaded-search input byte is out of range")
                })
                .collect::<Result<Vec<_>, _>>()?,
            b"abcabcabcabcabcabcabcabcabcabcabcabc".repeat(4_500),
        ];
        let config = DeflateConfig {
            level: 9,
            method: Method::Deflated,
            window_bits: 16 + crate::MAX_WBITS,
            mem_level: 1,
            strategy: Strategy::Default,
        };

        let capacities = inputs
            .iter()
            .map(|input| compress_bound(input.len()) + 64)
            .collect::<Vec<_>>();
        let mut scalar_outputs = capacities
            .iter()
            .map(|capacity| vec![0u8; *capacity])
            .collect::<Vec<_>>();
        let mut scalar_jobs = inputs
            .iter()
            .zip(scalar_outputs.iter_mut())
            .map(|(input, output)| BatchCompressJob::new(input, output))
            .collect::<Vec<_>>();
        let scalar_results = compress_slices_batch(&mut scalar_jobs, config);
        drop(scalar_jobs);

        for workers in [0, 3] {
            let mut threaded_outputs = capacities
                .iter()
                .map(|capacity| vec![0u8; *capacity])
                .collect::<Vec<_>>();
            let mut threaded_jobs = inputs
                .iter()
                .zip(threaded_outputs.iter_mut())
                .map(|(input, output)| BatchCompressJob::new(input, output))
                .collect::<Vec<_>>();

            let threaded_results = compress_slices_batch_with_backend(
                &mut threaded_jobs,
                config,
                BatchMatchSearchBackend::ScopedThreads { workers },
            );
            drop(threaded_jobs);

            assert_eq!(threaded_results, scalar_results);
            for (index, result) in threaded_results.iter().enumerate() {
                assert_eq!(
                    &threaded_outputs[index][..result.written],
                    &scalar_outputs[index][..result.written],
                    "scoped-thread backend changed job {index} with workers={workers}"
                );
            }
        }
        Ok(())
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn explicit_single_worker_batch_matches_scalar() -> Result<(), &'static str> {
        let input = b"single-worker-batch/".repeat(4_000);
        let config = DeflateConfig::new(9);
        let (reference, code) = independent(&input, config, compress_bound(input.len()) + 64);
        let mut output = vec![0u8; compress_bound(input.len()) + 64];
        let compressor = BatchCompressor::new(core::num::NonZeroUsize::MIN)
            .map_err(|_| "one-worker batch compressor failed initialization")?;
        let results = {
            let mut jobs = [BatchCompressJob::new(&input, &mut output)];
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
                .map(|(input, output)| BatchCompressJob::new(input, output))
                .collect::<Vec<_>>();

            let workers = core::num::NonZeroUsize::new(4).ok_or("four workers must be nonzero")?;
            let compressor = BatchCompressor::new(workers)
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

    #[test]
    fn batch_reports_the_same_tiny_output_failure() {
        let input = b"tiny-output-batch/".repeat(4_000);
        let config = DeflateConfig::new(6);
        let (reference, reference_code) = independent(&input, config, 4);
        let mut output = [0u8; 4];
        let results = {
            let mut jobs = [BatchCompressJob::new(&input, &mut output)];
            compress_slices_batch(&mut jobs, config)
        };

        assert_eq!(results[0].code, reference_code);
        assert_eq!(results[0].written, reference.len());
        assert_eq!(&output[..results[0].written], reference);
    }
}
