use core::marker::PhantomData;
use std::vec::Vec;

use super::{
    algorithm::{
        Algorithm, EndBlockKind, StandardDriver, StandardSegmentMode, StandardTurnOutcome,
    },
    compress_slice, deflate, end, init,
    longest_match::{
        execute_match_search_batch_with_backend, MatchSearchRequest, MatchSearchResult,
        MatchSearchWork, ScopedThreadMatchSearchBackend,
    },
    DeflateConfig, DeflateFlush, DeflateStream, InputCursor, OutputCursor, ReturnCode, Status,
};
use crate::c_api::z_stream;

/// One independent input/output pair for [`compress_slices_batch`].
///
/// The output buffer may be smaller than the compressed stream. In that case the corresponding
/// [`BatchCompressResult`] reports [`ReturnCode::BufError`] and `written` is the initialized prefix.
pub struct BatchCompressJob<'a> {
    pub input: &'a [u8],
    pub output: &'a mut [u8],
}

impl<'a> BatchCompressJob<'a> {
    pub fn new(input: &'a [u8], output: &'a mut [u8]) -> Self {
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
/// caller choice. A scoped-thread worker count of zero is treated as one worker.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BatchMatchSearchBackend {
    #[default]
    Scalar,
    ScopedThreads {
        workers: usize,
    },
}

/// Compress several independent buffers while batching detached longest-match searches.
///
/// Every job uses the same configuration and produces the same bytes as an independent
/// [`super::compress_slice`] call. Fast, Medium, Slow, Huffman-only, and RLE compression share the
/// schedulable standard-driver path. The default scalar backend executes detached searches directly
/// without staging; backends that need cross-stream concurrency collect outstanding searches into a
/// batch. Stored and Quick compression have different block mechanics and therefore fall back to the
/// ordinary synchronous slice compressor.
///
/// The function is available with the `std` feature because the cross-stream scheduler owns dynamic
/// scratch vectors. Per-stream compression state still uses the crate's ordinary zlib allocator
/// initialization path.
pub fn compress_slices_batch(
    jobs: &mut [BatchCompressJob<'_>],
    config: DeflateConfig,
) -> Vec<BatchCompressResult> {
    compress_slices_batch_with_backend(jobs, config, BatchMatchSearchBackend::Scalar)
}

/// Variant of [`compress_slices_batch`] with an explicit match-search executor.
pub fn compress_slices_batch_with_backend(
    jobs: &mut [BatchCompressJob<'_>],
    config: DeflateConfig,
    backend: BatchMatchSearchBackend,
) -> Vec<BatchCompressResult> {
    let Some(algorithm) = configured_algorithm(config) else {
        return compress_synchronously(jobs, config);
    };

    if !matches!(
        algorithm,
        Algorithm::Fast | Algorithm::Medium | Algorithm::Slow | Algorithm::Huffman | Algorithm::Rle
    ) {
        return compress_synchronously(jobs, config);
    }

    let mut results = vec![
        BatchCompressResult {
            written: 0,
            code: ReturnCode::StreamError,
        };
        jobs.len()
    ];
    let mut runtimes = Vec::with_capacity(jobs.len());

    for (job_index, job) in jobs.iter_mut().enumerate() {
        match BatchRuntime::new(job_index, job.input, &mut *job.output, config, algorithm) {
            Ok(runtime) => runtimes.push(runtime),
            Err(result) => results[job_index] = result,
        }
    }

    let mut active = runtimes.len();
    let concurrent_searches = matches!(
        backend,
        BatchMatchSearchBackend::ScopedThreads { workers } if workers > 1
    );
    let mut match_work = if concurrent_searches {
        Vec::<(usize, MatchSearchWork)>::with_capacity(runtimes.len())
    } else {
        Vec::new()
    };
    let mut match_results = if concurrent_searches {
        vec![
            MatchSearchResult {
                length: 0,
                start: 0,
            };
            runtimes.len()
        ]
    } else {
        Vec::new()
    };

    while active != 0 {
        match_work.clear();

        for (runtime_index, runtime) in runtimes.iter_mut().enumerate() {
            if runtime.done {
                continue;
            }

            match runtime.step() {
                RuntimeStep::Progress => {}
                RuntimeStep::MatchSearch(work) => {
                    if concurrent_searches {
                        match_work.push((runtime_index, work));
                    } else {
                        // Scalar and one-worker execution borrow only this paused runtime and can be
                        // completed immediately, so they need no cross-stream scratch vectors.
                        let result = runtime.match_request(work).execute();
                        runtime.complete_match_search(result);
                    }
                }
                RuntimeStep::Done(result) => {
                    results[runtime.job_index] = result;
                    active -= 1;
                }
            }
        }

        if match_work.is_empty() {
            continue;
        }

        if match_work.len() == 1 {
            // A temporarily desynchronized batch has nothing to parallelize. Avoid allocating a
            // request vector just to let the threaded backend collapse back to one scalar worker.
            let (runtime_index, work) = match_work[0];
            let result = runtimes[runtime_index].match_request(work).execute();
            runtimes[runtime_index].complete_match_search(result);
            continue;
        }

        let BatchMatchSearchBackend::ScopedThreads { workers } = backend else {
            unreachable!("only concurrent backends stage match-search work");
        };

        // Concurrent execution needs all immutable requests alive at once. Keep that staging vector
        // in the shortest possible scope so no borrowed runtime survives into the resume pass.
        {
            let requests = match_work
                .iter()
                .map(|(runtime_index, work)| runtimes[*runtime_index].match_request(*work))
                .collect::<Vec<MatchSearchRequest<'_>>>();
            execute_match_search_batch_with_backend(
                &ScopedThreadMatchSearchBackend::new(workers),
                &requests,
                &mut match_results[..requests.len()],
            );
        }

        for ((runtime_index, _), result) in match_work
            .iter()
            .copied()
            .zip(match_results.iter().copied())
        {
            runtimes[runtime_index].complete_match_search(result);
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
        0..=9 => config.level as i8,
        _ => return None,
    };
    Some(Algorithm::for_stream(level, config.strategy))
}

struct BatchRuntime<'a> {
    job_index: usize,
    raw: z_stream,
    driver: StandardDriver,
    input_remaining: usize,
    output_remaining: usize,
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
    ) -> Result<Self, BatchCompressResult> {
        let mut raw = z_stream::default();
        let code = init(&mut raw, config);
        if code != ReturnCode::Ok {
            return Err(BatchCompressResult { written: 0, code });
        }

        raw.next_out = output.as_mut_ptr();
        let mut runtime = Self {
            job_index,
            raw,
            driver: {
                let stream = unsafe { DeflateStream::from_stream_ref(&raw) }
                    .expect("successful init creates a typed deflate stream");
                StandardDriver::for_algorithm(algorithm, &*stream.state)
                    .expect("batch runtime requires a standard token producer")
            },
            input_remaining: input.len(),
            output_remaining: output.len(),
            done: false,
            _buffers: PhantomData,
        };

        if !runtime.offer_output() {
            return Err(runtime.finish_error(ReturnCode::BufError));
        }

        // Emit raw/zlib/gzip framing through the ordinary compatibility state machine. No input is
        // attached yet, so the algorithm body cannot run in this phase.
        loop {
            let stream = runtime.stream_mut();
            stream.next_in = core::ptr::null_mut();
            stream.avail_in = 0;
            let code = deflate(stream, DeflateFlush::NoFlush);
            if code != ReturnCode::Ok {
                return Err(runtime.finish_error(code));
            }

            if stream.state.status == Status::Busy
                && stream.state.bit_writer.pending.pending().is_empty()
            {
                break;
            }

            if stream.avail_out == 0 && !runtime.offer_output() {
                return Err(runtime.finish_error(ReturnCode::BufError));
            }
        }

        runtime.raw.next_in = input.as_ptr() as *mut u8;
        runtime.offer_input();
        Ok(runtime)
    }

    fn stream_mut(&mut self) -> &mut DeflateStream<'_> {
        unsafe { DeflateStream::from_stream_mut(&mut self.raw) }
            .expect("live batch runtime has initialized state")
    }

    fn stream_ref(&self) -> &DeflateStream<'_> {
        unsafe { DeflateStream::from_stream_ref(&self.raw) }
            .expect("live batch runtime has initialized state")
    }

    fn offer_input(&mut self) -> bool {
        if self.raw.avail_in != 0 {
            return true;
        }
        if self.input_remaining == 0 {
            return false;
        }

        let offered = self.input_remaining.min(crate::c_api::uInt::MAX as usize);
        self.raw.avail_in = offered as crate::c_api::uInt;
        self.input_remaining -= offered;
        true
    }

    fn offer_output(&mut self) -> bool {
        if self.raw.avail_out != 0 {
            return true;
        }
        if self.output_remaining == 0 {
            return false;
        }

        let offered = self.output_remaining.min(crate::c_api::uInt::MAX as usize);
        self.raw.avail_out = offered as crate::c_api::uInt;
        self.output_remaining -= offered;
        true
    }

    fn step(&mut self) -> RuntimeStep {
        if self.raw.avail_in == 0 {
            self.offer_input();
        }
        if self.raw.avail_out == 0 && !self.offer_output() {
            return RuntimeStep::Done(self.finish_error(ReturnCode::BufError));
        }

        let mode = if self.input_remaining == 0 {
            StandardSegmentMode::Final
        } else {
            StandardSegmentMode::Continue
        };

        let Self { raw, driver, .. } = self;
        let stream = unsafe { DeflateStream::from_stream_mut(raw) }
            .expect("live batch runtime has initialized state");
        let mut input = InputCursor::from_stream(stream);
        let mut output = OutputCursor::from_stream(stream);
        let outcome = driver.step(&mut *stream.state, &mut input, &mut output, mode);
        input.apply(stream);
        output.apply(stream);

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
            StandardTurnOutcome::SegmentDone(EndBlockKind::FlushBoundary) => {
                RuntimeStep::Done(self.finish_error(ReturnCode::StreamError))
            }
        }
    }

    fn match_request(&self, work: MatchSearchWork) -> MatchSearchRequest<'_> {
        work.request(&*self.stream_ref().state)
    }

    fn complete_match_search(&mut self, result: MatchSearchResult) {
        let Self { raw, driver, .. } = self;
        let stream = unsafe { DeflateStream::from_stream_mut(raw) }
            .expect("live batch runtime has initialized state");
        driver.complete_match_search(&mut *stream.state, result);
    }

    fn finish_stream(&mut self) -> BatchCompressResult {
        {
            let stream = self.stream_mut();
            debug_assert_eq!(stream.avail_in, 0);
            debug_assert!(stream.state.bit_writer.pending.pending().is_empty());
            stream.state.status = Status::Finish;
            stream.next_in = core::ptr::null_mut();
            stream.avail_in = 0;
        }

        loop {
            if self.raw.avail_out == 0 && !self.offer_output() {
                return self.finish_error(ReturnCode::BufError);
            }

            match deflate(self.stream_mut(), DeflateFlush::Finish) {
                ReturnCode::StreamEnd => {
                    self.done = true;
                    return BatchCompressResult {
                        written: self.raw.total_out as usize,
                        code: ReturnCode::Ok,
                    };
                }
                ReturnCode::Ok => {}
                code => return self.finish_error(code),
            }
        }
    }

    fn finish_error(&mut self, code: ReturnCode) -> BatchCompressResult {
        self.done = true;
        BatchCompressResult {
            written: self.raw.total_out as usize,
            code,
        }
    }
}

impl Drop for BatchRuntime<'_> {
    fn drop(&mut self) {
        if let Some(stream) = unsafe { DeflateStream::from_stream_mut(&mut self.raw) } {
            let _ = end(stream);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deflate::{compress_bound, Method, Strategy};

    fn independent(input: &[u8], config: DeflateConfig, capacity: usize) -> (Vec<u8>, ReturnCode) {
        let mut output = vec![0u8; capacity];
        let (written, code) = compress_slice(&mut output, input, config);
        let len = written.len();
        output.truncate(len);
        (output, code)
    }

    #[test]
    fn batch_matches_independent_standard_compression() {
        let inputs = [
            b"batch-alpha/".repeat(7_000),
            (0..90_000).map(|i| ((i * 37) % 251) as u8).collect(),
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
    fn scoped_thread_backend_matches_scalar_batch_output() {
        let inputs = [
            b"threaded-search-alpha/".repeat(6_000),
            b"threaded-search-beta/beta/beta/".repeat(5_000),
            (0..140_000)
                .map(|i| (((i / 17) * 31 + i % 7) % 251) as u8)
                .collect(),
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
    }

    #[test]
    fn batch_reports_the_same_tiny_output_failure() {
        let input = b"tiny-output-batch/".repeat(4_000);
        let config = DeflateConfig::new(6);
        let (reference, reference_code) = independent(&input, config, 4);
        let mut output = [0u8; 4];
        let mut jobs = [BatchCompressJob::new(&input, &mut output)];

        let results = compress_slices_batch(&mut jobs, config);
        drop(jobs);

        assert_eq!(results[0].code, reference_code);
        assert_eq!(results[0].written, reference.len());
        assert_eq!(&output[..results[0].written], reference);
    }
}
