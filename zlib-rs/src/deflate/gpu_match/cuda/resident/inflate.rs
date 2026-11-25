//! Bounded compressed-input pipeline. Only the validated entropy index and
//! compressed source bytes are uploaded. Decoded windows never leave CUDA.
use super::super::{allocation_capacity, failure, reserve};
use super::{
    append, stream_header, Checksum, Geometry, ResidentEncoder, ResidentPool, Wrapper, MAX_WINDOW,
    WORKGROUP_SIZE,
};
use crate::{
    deflate::gpu_match::{GpuMatchError, GpuTranscodeOptions, GpuTranscodeResult},
    inflate::gpu_index::{IndexScanner, InflateSegment},
    ReturnCode,
};
use cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use std::sync::{Arc, Mutex};

const HISTORY: usize = 32_768;
const MAX_SEGMENT: usize = 128 * 1024 + 258;
const MAX_BATCH_SEGMENTS: usize = 2048;
const MAX_BATCH_INPUT: usize = 8 * 1024 * 1024;
const MAX_INDEX_SEGMENTS: usize = 1_000_000;
const WORDS_PER_SEGMENT: usize = 6;
const CARRY_BYTES: usize = HISTORY + MAX_SEGMENT;

pub(super) struct Decoder {
    stream: Arc<CudaStream>,
    kernels: Kernels,
    workspace: Mutex<Workspace>,
}

kernel_set! { inflate_symbolic, inflate_status, inflate_seed, inflate_resolve }

struct Workspace {
    input: CudaSlice<u8>,
    header: CudaSlice<u8>,
    segments: CudaSlice<u64>,
    symbolic: CudaSlice<u16>,
    seeds: CudaSlice<u8>,
    used_seeds: CudaSlice<u32>,
    status: CudaSlice<u32>,
    error: CudaSlice<u32>,
    raw: CudaSlice<u8>,
    carry: CudaSlice<u8>,
}

impl Workspace {
    fn new(stream: &Arc<CudaStream>) -> Result<Self, GpuMatchError> {
        Ok(Self {
            input: stream.alloc_zeros(1).map_err(failure)?,
            header: stream.alloc_zeros(1).map_err(failure)?,
            segments: stream.alloc_zeros(1).map_err(failure)?,
            symbolic: stream.alloc_zeros(1).map_err(failure)?,
            seeds: stream.alloc_zeros(1).map_err(failure)?,
            used_seeds: stream.alloc_zeros(1).map_err(failure)?,
            status: stream.alloc_zeros(1).map_err(failure)?,
            error: stream.alloc_zeros(1).map_err(failure)?,
            raw: stream.alloc_zeros(1).map_err(failure)?,
            carry: stream.alloc_zeros(1).map_err(failure)?,
        })
    }

    pub(super) fn upper_bound() -> usize {
        let allocation = |n: usize, width: usize| allocation_capacity(n) * width;
        allocation(MAX_BATCH_INPUT, 1)
            + allocation(4096, 1)
            + allocation(MAX_BATCH_SEGMENTS * WORDS_PER_SEGMENT, 8)
            + allocation(MAX_WINDOW + MAX_SEGMENT, 2)
            + allocation(MAX_BATCH_SEGMENTS * HISTORY, 1)
            + allocation(MAX_BATCH_SEGMENTS * HISTORY / 32, 4)
            + allocation(MAX_BATCH_SEGMENTS, 4)
            + 4
            + allocation(MAX_WINDOW + CARRY_BYTES, 1)
            + allocation(CARRY_BYTES, 1)
    }
}

#[derive(Default)]
struct Cursor {
    segment: usize,
    decoded: u32,
    origin: u32,
}

struct Group<'a> {
    segments: &'a [InflateSegment],
    payload: &'a [u8],
    header: &'a [u8],
    input_origin: u64,
    header_origin: u64,
}

impl<'a> Group<'a> {
    fn next(
        source: &'a [u8],
        segments: &'a [InflateSegment],
        target: u32,
    ) -> Result<Self, GpuMatchError> {
        let first = segments
            .first()
            .ok_or_else(|| GpuMatchError("inflate index ended before decoded bound".into()))?;
        let input_origin = first.start_bit / 8;
        let n = segments
            .iter()
            .take(MAX_BATCH_SEGMENTS)
            .take_while(|s| s.end_bit.div_ceil(8) - input_origin <= MAX_BATCH_INPUT as u64)
            .position(|s| s.end >= target)
            .map_or_else(
                || {
                    segments
                        .iter()
                        .take(MAX_BATCH_SEGMENTS)
                        .take_while(|s| {
                            s.end_bit.div_ceil(8) - input_origin <= MAX_BATCH_INPUT as u64
                        })
                        .count()
                },
                |i| i + 1,
            );
        let selected = segments
            .get(..n)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| GpuMatchError("inflate segment exceeds bounded input batch".into()))?;
        let end = selected
            .last()
            .ok_or_else(|| GpuMatchError("empty inflate batch".into()))?
            .end_bit
            .div_ceil(8);
        let header_origin = first.header_bit / 8;
        let header_end = first.header_end_bit.div_ceil(8);
        let slice = |start: u64, end: u64| -> Result<&[u8], GpuMatchError> {
            source
                .get(
                    usize::try_from(start).map_err(failure)?
                        ..usize::try_from(end).map_err(failure)?,
                )
                .ok_or_else(|| GpuMatchError("inflate index points outside source".into()))
        };
        Ok(Self {
            segments: selected,
            payload: slice(input_origin, end)?,
            header: slice(header_origin, header_end)?,
            input_origin,
            header_origin,
        })
    }
}

impl Decoder {
    pub(super) fn new(
        stream: Arc<CudaStream>,
        module: &Arc<CudaModule>,
    ) -> Result<Self, GpuMatchError> {
        Ok(Self {
            kernels: Kernels::new(module)?,
            workspace: Mutex::new(Workspace::new(&stream)?),
            stream,
        })
    }

    pub(super) fn upper_bound() -> usize {
        Workspace::upper_bound()
    }

    fn retain_history(
        &self,
        w: &mut Workspace,
        cursor: &mut Cursor,
        origin: u32,
    ) -> Result<(), GpuMatchError> {
        if cursor.origin == origin {
            return Ok(());
        }
        let offset = (origin - cursor.origin) as usize;
        let end = (cursor.decoded - cursor.origin) as usize;
        let retained = end - offset;
        if retained > CARRY_BYTES {
            return Err(GpuMatchError(
                "inflate carry exceeds bounded workspace".into(),
            ));
        }
        if retained != 0 {
            self.stream
                .memcpy_dtod(&w.raw.slice(offset..end), &mut w.carry)
                .map_err(failure)?;
            self.stream
                .memcpy_dtod(&w.carry.slice(..retained), &mut w.raw)
                .map_err(failure)?;
        }
        cursor.origin = origin;
        Ok(())
    }

    #[expect(
        unsafe_code,
        reason = "validated inflate segment ABI and exclusively stream-owned bounded buffers"
    )]
    fn decode_group(
        &self,
        w: &mut Workspace,
        group: &Group<'_>,
        cursor: &mut Cursor,
        window: u32,
    ) -> Result<(), GpuMatchError> {
        let first = group
            .segments
            .first()
            .ok_or_else(|| GpuMatchError("empty inflate group".into()))?;
        let last = group
            .segments
            .last()
            .ok_or_else(|| GpuMatchError("empty inflate group".into()))?;
        let count = u32::try_from(group.segments.len()).map_err(failure)?;
        let base = cursor.decoded;
        if first.start != base || (last.end - cursor.origin) as usize > w.raw.len() {
            return Err(GpuMatchError(
                "inflate group exceeds decoded workspace".into(),
            ));
        }
        let words = group
            .segments
            .iter()
            .flat_map(|s| {
                [
                    s.header_bit,
                    s.header_end_bit,
                    s.start_bit,
                    s.end_bit,
                    u64::from(s.start) | (u64::from(s.end) << 32),
                    u64::from(s.kind) | (u64::from(s.ends_block) << 32),
                ]
            })
            .collect::<Vec<_>>();
        let seed_count = (count * window) as usize;
        reserve(&self.stream, &mut w.input, group.payload.len())?;
        reserve(&self.stream, &mut w.header, group.header.len())?;
        reserve(&self.stream, &mut w.segments, words.len())?;
        reserve(&self.stream, &mut w.symbolic, (last.end - base) as usize)?;
        reserve(&self.stream, &mut w.seeds, seed_count)?;
        reserve(&self.stream, &mut w.used_seeds, seed_count / 32)?;
        reserve(&self.stream, &mut w.status, count as usize)?;
        self.stream
            .memcpy_htod(group.payload, &mut w.input)
            .map_err(failure)?;
        self.stream
            .memcpy_htod(group.header, &mut w.header)
            .map_err(failure)?;
        self.stream
            .memcpy_htod(&words, &mut w.segments)
            .map_err(failure)?;
        self.stream
            .memset_zeros(&mut w.used_seeds.slice_mut(..seed_count / 32))
            .map_err(failure)?;
        self.stream.memset_zeros(&mut w.error).map_err(failure)?;
        launch!(self,inflate_symbolic,(count.div_ceil(4),1,1),WORKGROUP_SIZE;
            &w.input,&w.header,&group.input_origin,&group.header_origin,&first.header_bit,
            &w.segments,&count,&mut w.symbolic,&mut w.status,&mut w.used_seeds,&window,&base);
        launch!(self,inflate_status,(count.div_ceil(WORKGROUP_SIZE),1,1),WORKGROUP_SIZE;
            &w.status,&count,&mut w.error);
        launch!(self,inflate_seed,(8,count,1),WORKGROUP_SIZE;
            &w.segments,&w.symbolic,&w.raw,&w.used_seeds,&mut w.seeds,&mut w.error,&window,&base,&cursor.origin);
        launch!(self,inflate_resolve,(32,count,1),WORKGROUP_SIZE;
            &w.segments,&w.symbolic,&w.seeds,&mut w.raw,&w.error,&window,&base,&cursor.origin);
        let error = self.stream.clone_dtoh(&w.error).map_err(failure)?;
        if error[0] != 0 {
            return Err(GpuMatchError(format!(
                "CUDA inflate segment validation failed: {}",
                error[0]
            )));
        }
        cursor.decoded = last.end;
        cursor.segment += count as usize;
        Ok(())
    }

    fn decode_to(
        &self,
        w: &mut Workspace,
        source: &[u8],
        segments: &[InflateSegment],
        window: u32,
        cursor: &mut Cursor,
        target: u32,
    ) -> Result<(), GpuMatchError> {
        while cursor.decoded < target {
            let group = Group::next(source, &segments[cursor.segment..], target)?;
            self.decode_group(w, &group, cursor, window)?;
        }
        Ok(())
    }
}

struct ScanAhead<'a> {
    scanner: IndexScanner<'a>,
    pending: Vec<InflateSegment>,
    count: usize,
}

impl<'a> ScanAhead<'a> {
    fn new(source: &'a [u8], limit: u32) -> Result<Self, ReturnCode> {
        Ok(Self {
            scanner: IndexScanner::new(source, limit)?,
            pending: Vec::new(),
            count: 0,
        })
    }

    fn fill_to(&mut self, target: usize) -> Result<(), ReturnCode> {
        // One extra segment decides BFINAL at an exact window boundary.
        while self.scanner.adler().is_none() && self.scanner.decoded() <= target {
            let segment = self.scanner.next_segment()?.ok_or(ReturnCode::DataError)?;
            self.count += 1;
            if self.count > MAX_INDEX_SEGMENTS {
                return Err(ReturnCode::BufError);
            }
            if segment.start != segment.end {
                self.pending.push(segment);
            }
        }
        Ok(())
    }

    fn result(&self, output: Vec<u8>, code: ReturnCode) -> GpuTranscodeResult {
        GpuTranscodeResult {
            output,
            decoded_len: self.scanner.adler().map(|_| self.scanner.decoded()),
            decoded_bytes_indexed: self.scanner.decoded(),
            code,
        }
    }
}

impl ResidentEncoder {
    fn transcode(
        &self,
        source: &[u8],
        mut scan: ScanAhead<'_>,
        options: GpuTranscodeOptions,
        distance: u32,
        index_tiles: u32,
    ) -> Result<GpuTranscodeResult, GpuMatchError> {
        let config = options.config;
        let wrapper = Wrapper::from_bits(config.window_bits)
            .ok_or_else(|| GpuMatchError("invalid transcode wrapper".into()))?;
        let limit = options.output_limit;
        let mut output = Vec::new();
        let (header, n) = stream_header(wrapper, config.level);
        if !append(&mut output, &header[..n], limit) {
            return Ok(scan.result(output, ReturnCode::BufError));
        }
        let mut checksum = Checksum::new(wrapper);
        let window = options.window.get().min(MAX_WINDOW);
        if let Err(code) = scan.fill_to(window) {
            return Ok(scan.result(Vec::new(), code));
        }
        if scan.scanner.decoded() == 0 && !append(&mut output, &[3, 0], limit) {
            return Ok(scan.result(output, ReturnCode::BufError));
        }
        let mut w = self.decoder.workspace.lock().map_err(failure)?;
        reserve(
            &self.stream,
            &mut w.raw,
            window.min(options.decoded_limit as usize) + CARRY_BYTES,
        )?;
        reserve(&self.stream, &mut w.carry, CARRY_BYTES)?;
        let mut cursor = Cursor::default();
        let source_window = scan.scanner.window();
        let history = (source_window as usize).max(distance as usize - 1);
        let mut b = self.buffers.lock().map_err(failure)?;
        let mut start = 0;
        while start < scan.scanner.decoded() {
            if output.len() == limit {
                return Ok(scan.result(output, ReturnCode::BufError));
            }
            let end = start.saturating_add(window).min(scan.scanner.decoded());
            let origin = start.saturating_sub(history);
            self.decoder.retain_history(
                &mut w,
                &mut cursor,
                u32::try_from(origin).map_err(failure)?,
            )?;
            self.decoder.decode_to(
                &mut w,
                source,
                &scan.pending,
                source_window,
                &mut cursor,
                u32::try_from(end).map_err(failure)?,
            )?;
            scan.pending.drain(..cursor.segment);
            cursor.segment = 0;
            let input_origin = start.saturating_sub(distance as usize - 1);
            let final_window = scan.scanner.adler().is_some() && end == scan.scanner.decoded();
            let g = Geometry::new(
                end - input_origin,
                start - input_origin,
                distance,
                options.search_budget,
                final_window,
                config,
                wrapper,
            )?
            .with_index_tiles(index_tiles);
            b.prepare(&self.stream, g)?;
            self.stream
                .memcpy_dtod(
                    &w.raw.slice(input_origin - origin..end - origin),
                    &mut b.input,
                )
                .map_err(failure)?;
            self.submit_window(&mut b, g)?;
            // The stream is now busy with index, parse and entropy kernels.
            // Discover the next window's token boundaries on the host before
            // synchronizing for the compact result. No extra thread pool.
            let next = if final_window {
                Ok(())
            } else {
                scan.fill_to(end.saturating_add(window))
            };
            let piece = self.read_window(&b, g, limit - output.len())?;
            output.extend_from_slice(&piece.output);
            if !piece.complete {
                return Ok(scan.result(output, ReturnCode::BufError));
            }
            if let Err(code) = next {
                return Ok(scan.result(Vec::new(), code));
            }
            checksum.update(&piece, end - start);
            start = end;
        }
        let (trailer, n) = checksum.trailer(start);
        if Some(u32::from_be_bytes(
            trailer[..n].try_into().map_err(failure)?,
        )) != scan.scanner.adler()
        {
            return Ok(scan.result(Vec::new(), ReturnCode::DataError));
        }
        let code = if append(&mut output, &trailer[..n], limit) {
            ReturnCode::Ok
        } else {
            ReturnCode::BufError
        };
        Ok(scan.result(output, code))
    }
}

impl ResidentPool {
    pub(in crate::deflate::gpu_match::cuda) fn recompress_zlib(
        &self,
        source: &[u8],
        options: GpuTranscodeOptions,
        distance: u32,
    ) -> Result<Option<GpuTranscodeResult>, GpuMatchError> {
        if options.minimum_decoded_len > options.decoded_limit {
            return Ok(None);
        }
        let mut scan = match ScanAhead::new(source, options.decoded_limit) {
            Ok(scan) => scan,
            Err(code) => {
                return Ok(Some(GpuTranscodeResult {
                    output: Vec::new(),
                    decoded_len: None,
                    decoded_bytes_indexed: 0,
                    code,
                }))
            }
        };
        if options.minimum_decoded_len != 0 {
            let minimum = options.minimum_decoded_len as usize;
            if let Err(code) = scan.fill_to(minimum - 1) {
                return Ok(Some(scan.result(Vec::new(), code)));
            }
            if scan.scanner.decoded() < minimum {
                return Ok(None);
            }
        }
        self.acquire()?
            .transcode(source, scan, options, distance, self.index_tiles)
            .map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DeflateConfig;
    use crate::{
        compress_bound, compress_slice, compress_slices_batch_gpu_windowed, decompress_slice,
        BatchJob, GpuMatchSearcher, InflateConfig, Strategy,
    };
    use std::num::NonZeroUsize;

    #[test]
    #[ignore = "requires CUDA hardware and NVRTC"]
    fn cuda_transcode_matches_raw_encoder_and_checks_source(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let searcher = GpuMatchSearcher::new(Some("cuda:0"))?;
        let config = DeflateConfig::new(9);
        let mut seed = 0x1234_5678u32;
        let random = (0..180_013)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                seed.to_le_bytes()[0]
            })
            .collect::<Vec<_>>();
        let mut mixed = random[..33_019].repeat(5);
        mixed.extend(b"123456789abcdef".repeat(42_000));
        let repeated = vec![0; 1_040_001];
        let inputs = [b"".as_slice(), b"a", b"abc", &random, &mixed, &repeated];
        let window = NonZeroUsize::new(65_537).ok_or("window")?;
        for input in inputs {
            let mut expected = Vec::new();
            let full = compress_slices_batch_gpu_windowed(
                &mut [BatchJob::with_vec(
                    input,
                    &mut expected,
                    compress_bound(input.len()) + 512,
                )?],
                config,
                &searcher,
                128,
                window,
            )?[0];
            assert_eq!(full.code, ReturnCode::Ok);
            for (level, strategy) in [
                (0, Strategy::Default),
                (6, Strategy::Fixed),
                (9, Strategy::Default),
            ] {
                let mut bytes = vec![0; compress_bound(input.len())];
                let (source, code) = compress_slice(
                    &mut bytes,
                    input,
                    DeflateConfig {
                        level,
                        strategy,
                        ..DeflateConfig::default()
                    },
                );
                assert_eq!(code, ReturnCode::Ok);
                let max = u32::try_from(input.len())?;
                let options = |cap, decoded_limit| GpuTranscodeOptions {
                    output_limit: cap,
                    decoded_limit,
                    minimum_decoded_len: 0,
                    config,
                    search_budget: 128,
                    window,
                };
                for cap in [0, 1, expected.len() / 2, expected.len(), expected.len() + 1] {
                    let actual = searcher
                        .recompress_zlib(source, options(cap, max))?
                        .ok_or("unexpected transcode routing")?;
                    if actual.code == ReturnCode::Ok {
                        assert_eq!(actual.decoded_len, Some(input.len()));
                    }
                    assert_eq!(actual.output, expected[..expected.len().min(cap)]);
                    assert_eq!(
                        actual.code,
                        if cap < expected.len() {
                            ReturnCode::BufError
                        } else {
                            ReturnCode::Ok
                        }
                    );
                }
                let mut corrupt = source.to_vec();
                let last = corrupt.last_mut().ok_or("trailer")?;
                *last ^= 1;
                let result = searcher
                    .recompress_zlib(&corrupt, options(expected.len() + 512, max))?
                    .ok_or("unexpected transcode routing")?;
                assert_eq!(result.code, ReturnCode::DataError);
                assert!(result.output.is_empty());
                let truncated = searcher
                    .recompress_zlib(
                        &source[..source.len() - 1],
                        options(expected.len() + 512, max),
                    )?
                    .ok_or("unexpected transcode routing")?;
                assert_eq!(truncated.code, ReturnCode::DataError);
                if max != 0 {
                    let bounded = searcher
                        .recompress_zlib(source, options(expected.len() + 512, max - 1))?
                        .ok_or("unexpected transcode routing")?;
                    assert_eq!(bounded.code, ReturnCode::BufError);
                }
            }
            let mut scratch = vec![0; input.len()];
            let (decoded, code) =
                decompress_slice(&mut scratch, &expected, InflateConfig::default());
            assert_eq!(code, ReturnCode::Ok);
            assert_eq!(decoded, input);
        }
        Ok(())
    }
}
