//! Bounded compressed-input pipeline. Only the validated entropy index and
//! compressed source bytes are uploaded. Decoded windows never leave CUDA.
use super::super::{allocation_capacity, failure, reserve};
use super::{
    append, stream_header, Checksum, DeviceInput, Geometry, ResidentEncoder, ResidentPool, Wrapper,
    MAX_WINDOW, WORKGROUP_SIZE,
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

mod index;

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
    indexer: index::Indexer,
    workspace: Mutex<Workspace>,
}

kernel_set! { inflate_symbolic, inflate_status, inflate_seed, inflate_resolve }

struct Workspace {
    index: index::Workspace,
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
            index: index::Workspace::new(stream)?,
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
        index::Workspace::upper_bound()
            + allocation(MAX_BATCH_INPUT, 1)
            + allocation(4096, 1)
            + allocation(MAX_BATCH_SEGMENTS * WORDS_PER_SEGMENT, 8)
            + allocation(MAX_WINDOW + MAX_SEGMENT, 2)
            + allocation(MAX_BATCH_SEGMENTS * HISTORY, 1)
            + allocation(MAX_BATCH_SEGMENTS * HISTORY / 32, 4)
            + allocation(MAX_BATCH_SEGMENTS, 4)
            + 4
            + allocation((MAX_WINDOW + CARRY_BYTES).next_multiple_of(4), 1)
            + allocation(CARRY_BYTES, 1)
    }
}

#[derive(Default)]
struct Cursor {
    segment: usize,
    group: usize,
    decoded: u32,
    origin: u32,
    source_bytes_uploaded: usize,
    descriptor_bytes_uploaded: usize,
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

enum DecodeBatch<'a> {
    Host(Group<'a>),
    Device(index::DeviceGroup, u32),
}

impl DecodeBatch<'_> {
    fn bounds(&self) -> Result<(u32, u32, u32), GpuMatchError> {
        match self {
            Self::Host(group) => {
                let first = group
                    .segments
                    .first()
                    .ok_or_else(|| GpuMatchError("empty inflate group".into()))?;
                let last = group
                    .segments
                    .last()
                    .ok_or_else(|| GpuMatchError("empty inflate group".into()))?;
                Ok((
                    u32::try_from(group.segments.len()).map_err(failure)?,
                    first.start,
                    last.end,
                ))
            }
            Self::Device(group, _) => Ok((group.count, group.start, group.end)),
        }
    }
}

impl Decoder {
    pub(super) fn new(
        stream: Arc<CudaStream>,
        module: &Arc<CudaModule>,
    ) -> Result<Self, GpuMatchError> {
        Ok(Self {
            kernels: Kernels::new(module)?,
            indexer: index::Indexer::new(Arc::clone(&stream), module)?,
            workspace: Mutex::new(Workspace::new(&stream)?),
            stream,
        })
    }

    pub(super) fn upper_bound() -> usize {
        Workspace::upper_bound()
    }

    fn device_index(
        &self,
        source: &[u8],
        limit: u32,
        compression_window: u32,
    ) -> Result<Option<index::DeviceIndex>, GpuMatchError> {
        let mut w = self.workspace.lock().map_err(failure)?;
        self.indexer
            .prepare(&mut w.index, source, limit, compression_window)
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
        group: DecodeBatch<'_>,
        cursor: &mut Cursor,
        window: u32,
    ) -> Result<(), GpuMatchError> {
        let (count, start, end) = group.bounds()?;
        let base = cursor.decoded;
        if start != base || (end - cursor.origin) as usize > w.raw.len() {
            return Err(GpuMatchError(
                "inflate group exceeds decoded workspace".into(),
            ));
        }
        let seed_count = (count * window) as usize;
        let word_count = count as usize * WORDS_PER_SEGMENT;
        reserve(&self.stream, &mut w.symbolic, (end - base) as usize)?;
        reserve(&self.stream, &mut w.seeds, seed_count)?;
        reserve(&self.stream, &mut w.used_seeds, seed_count / 32)?;
        reserve(&self.stream, &mut w.status, count as usize)?;
        let (input, header, input_origin, header_origin, first_header, descriptors, ranges, tokens) =
            match group {
                DecodeBatch::Device(group, tokens) => {
                    let offset = group.offset * WORDS_PER_SEGMENT;
                    cursor.group += 1;
                    (
                        &w.index.source,
                        &w.index.source,
                        0_u64,
                        0_u64,
                        0_u64,
                        w.index.segments.slice(offset..offset + word_count),
                        w.index
                            .ranges
                            .slice(group.offset * 2..(group.offset + count as usize) * 2),
                        tokens,
                    )
                }
                DecodeBatch::Host(group) => {
                    let first = group
                        .segments
                        .first()
                        .ok_or_else(|| GpuMatchError("empty inflate group".into()))?;
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
                    reserve(&self.stream, &mut w.input, group.payload.len())?;
                    reserve(&self.stream, &mut w.header, group.header.len())?;
                    reserve(&self.stream, &mut w.segments, words.len())?;
                    self.stream
                        .memcpy_htod(group.payload, &mut w.input)
                        .map_err(failure)?;
                    self.stream
                        .memcpy_htod(group.header, &mut w.header)
                        .map_err(failure)?;
                    self.stream
                        .memcpy_htod(&words, &mut w.segments)
                        .map_err(failure)?;
                    cursor.source_bytes_uploaded += group.payload.len() + group.header.len();
                    cursor.descriptor_bytes_uploaded += words.len() * core::mem::size_of::<u64>();
                    cursor.segment += count as usize;
                    (
                        &w.input,
                        &w.header,
                        group.input_origin,
                        group.header_origin,
                        first.header_bit,
                        w.segments.slice(..word_count),
                        w.index.ranges.slice(..1),
                        0_u32,
                    )
                }
            };
        self.stream
            .memset_zeros(&mut w.used_seeds.slice_mut(..seed_count / 32))
            .map_err(failure)?;
        launch!(self,inflate_symbolic,(count.div_ceil(4),1,1),WORKGROUP_SIZE;
            input,header,&input_origin,&header_origin,&first_header,
            &descriptors,&count,&w.index.tokens,&ranges,&tokens,&mut w.symbolic,&mut w.status,&mut w.used_seeds,&window,&base);
        launch!(self,inflate_status,(count.div_ceil(WORKGROUP_SIZE),1,1),WORKGROUP_SIZE;
            &w.status,&count,&mut w.error);
        launch!(self,inflate_seed,(8,count,1),WORKGROUP_SIZE;
            &descriptors,&w.symbolic,&w.raw,&w.used_seeds,&mut w.seeds,&mut w.error,&window,&base,&cursor.origin);
        launch!(self,inflate_resolve,(32,count,1),WORKGROUP_SIZE;
            &descriptors,&w.symbolic,&w.seeds,&mut w.raw,&w.error,&window,&base,&cursor.origin);
        // Do not synchronize between inflate and compression. The error
        // flag is cumulative and read with the compact compression metadata.
        // On an invalid group all later resolve kernels are suppressed;
        // the bounded raw allocation remains safe to read, but no result
        // can be published until that error flag has been checked.
        cursor.decoded = end;
        Ok(())
    }

    fn decode_to(
        &self,
        w: &mut Workspace,
        source: &[u8],
        scan: &ScanAhead<'_>,
        cursor: &mut Cursor,
        target: u32,
    ) -> Result<(), GpuMatchError> {
        while cursor.decoded < target {
            let group = match &scan.source {
                SourceIndex::Device { groups, tokens, .. } => DecodeBatch::Device(
                    *groups
                        .get(cursor.group)
                        .ok_or_else(|| GpuMatchError("CUDA decode schedule ended early".into()))?,
                    *tokens,
                ),
                SourceIndex::Host(_) => DecodeBatch::Host(Group::next(
                    source,
                    &scan.pending[cursor.segment..],
                    target,
                )?),
            };
            self.decode_group(w, group, cursor, scan.window())?;
        }
        Ok(())
    }
}

enum SourceIndex<'a> {
    Host(Box<IndexScanner<'a>>),
    Device {
        groups: Vec<index::DeviceGroup>,
        tokens: u32,
        decoded: u32,
        window: u32,
        adler: u32,
    },
}

struct ScanAhead<'a> {
    source: SourceIndex<'a>,
    pending: Vec<InflateSegment>,
    count: usize,
    processed: usize,
    source_bytes_uploaded: usize,
    descriptor_bytes_uploaded: usize,
}

impl<'a> ScanAhead<'a> {
    fn new(source: &'a [u8], limit: u32, uploaded: usize) -> Result<Self, ReturnCode> {
        Ok(Self {
            source: SourceIndex::Host(Box::new(IndexScanner::new(source, limit)?)),
            pending: Vec::new(),
            count: 0,
            processed: 0,
            source_bytes_uploaded: uploaded,
            descriptor_bytes_uploaded: 0,
        })
    }

    fn device(index: index::DeviceIndex, uploaded: usize) -> Self {
        Self {
            source: SourceIndex::Device {
                groups: index.groups,
                tokens: index.tokens,
                decoded: index.decoded,
                window: index.window,
                adler: index.adler,
            },
            count: 0,
            pending: Vec::new(),
            processed: 0,
            source_bytes_uploaded: uploaded,
            descriptor_bytes_uploaded: 0,
        }
    }

    fn decoded(&self) -> usize {
        match &self.source {
            SourceIndex::Host(scanner) => scanner.decoded(),
            SourceIndex::Device { decoded, .. } => *decoded as usize,
        }
    }

    fn adler(&self) -> Option<u32> {
        match &self.source {
            SourceIndex::Host(scanner) => scanner.adler(),
            SourceIndex::Device { adler, .. } => Some(*adler),
        }
    }

    fn window(&self) -> u32 {
        match &self.source {
            SourceIndex::Host(scanner) => scanner.window(),
            SourceIndex::Device { window, .. } => *window,
        }
    }

    const fn is_device(&self) -> bool {
        matches!(self.source, SourceIndex::Device { .. })
    }

    fn fill_to(&mut self, target: usize) -> Result<(), ReturnCode> {
        let SourceIndex::Host(scanner) = &mut self.source else {
            return Ok(());
        };
        // One extra segment decides BFINAL at an exact window boundary.
        while scanner.adler().is_none() && scanner.decoded() <= target {
            let segment = scanner.next_segment()?.ok_or(ReturnCode::DataError)?;
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
            decoded_len: self.adler().map(|_| self.decoded()),
            decoded_bytes_indexed: self.decoded(),
            decoded_bytes_processed: self.processed,
            source_bytes_uploaded: self.source_bytes_uploaded,
            descriptor_bytes_uploaded: self.descriptor_bytes_uploaded,
            device_indexed: self.is_device(),
            device_tokens: match &self.source {
                SourceIndex::Device { tokens, .. } => *tokens,
                SourceIndex::Host(_) => 0,
            },
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
        if scan.decoded() == 0 && !append(&mut output, &[3, 0], limit) {
            return Ok(scan.result(output, ReturnCode::BufError));
        }
        let mut w = self.decoder.workspace.lock().map_err(failure)?;
        reserve(
            &self.stream,
            &mut w.raw,
            (window.min(options.decoded_limit as usize) + CARRY_BYTES).next_multiple_of(4),
        )?;
        reserve(&self.stream, &mut w.carry, CARRY_BYTES)?;
        let mut cursor = Cursor {
            source_bytes_uploaded: scan.source_bytes_uploaded,
            ..Cursor::default()
        };
        self.stream.memset_zeros(&mut w.error).map_err(failure)?;
        let source_window = scan.window();
        let history = (source_window as usize).max(distance as usize - 1);
        let mut b = self.buffers.lock().map_err(failure)?;
        let mut start = 0;
        while start < scan.decoded() {
            if output.len() == limit {
                return Ok(scan.result(output, ReturnCode::BufError));
            }
            let end = start.saturating_add(window).min(scan.decoded());
            let origin = start.saturating_sub(history);
            self.decoder.retain_history(
                &mut w,
                &mut cursor,
                u32::try_from(origin).map_err(failure)?,
            )?;
            self.decoder.decode_to(
                &mut w,
                source,
                &scan,
                &mut cursor,
                u32::try_from(end).map_err(failure)?,
            )?;
            scan.source_bytes_uploaded = cursor.source_bytes_uploaded;
            scan.descriptor_bytes_uploaded = cursor.descriptor_bytes_uploaded;
            scan.pending.drain(..cursor.segment);
            cursor.segment = 0;
            let input_origin = start.saturating_sub(distance as usize - 1);
            let final_window = scan.adler().is_some() && end == scan.decoded();
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
            let input = DeviceInput::new(&w.raw, input_origin - origin, g.size)?;
            self.submit_window(&mut b, g, input)?;
            // The stream is now busy with index, parse and entropy kernels.
            // Discover the next window's token boundaries on the host before
            // synchronizing for the compact result. No extra thread pool.
            let next = if final_window {
                Ok(())
            } else {
                scan.fill_to(end.saturating_add(window))
            };
            self.stream
                .memcpy_dtod(&w.error, &mut b.meta.slice_mut(4..5))
                .map_err(failure)?;
            let piece = self.read_window(&b, g, limit - output.len(), true)?;
            scan.processed = end;
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
        drop(b);
        drop(w);
        finish_transcode(&scan, output, checksum, start, limit)
    }
}

fn finish_transcode(
    scan: &ScanAhead<'_>,
    mut output: Vec<u8>,
    checksum: Checksum,
    start: usize,
    limit: usize,
) -> Result<GpuTranscodeResult, GpuMatchError> {
    let (trailer, n) = checksum.trailer(start);
    if Some(u32::from_be_bytes(
        trailer[..n].try_into().map_err(failure)?,
    )) != scan.adler()
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
        // A DEFLATE match emits at most 258 bytes for two encoded bits;
        // no valid member can expand beyond this loose upper bound.
        if options.minimum_decoded_len as usize > source.len().saturating_mul(1032) {
            return Ok(None);
        }
        let mut uploaded = 0;
        if options.output_limit > 2 && index::eligible(source) {
            let encoder = self.acquire()?;
            uploaded = source.len();
            if let Some(index) = encoder.decoder.device_index(
                source,
                options.decoded_limit,
                u32::try_from(options.window.get().min(MAX_WINDOW)).map_err(failure)?,
            )? {
                if index.decoded < options.minimum_decoded_len {
                    return Ok(None);
                }
                return encoder
                    .transcode(
                        source,
                        ScanAhead::device(index, uploaded),
                        options,
                        distance,
                        self.index_tiles,
                    )
                    .map(Some);
            }
        }
        // Resource-bounded speculation is optional. Preserve the complete
        // streaming decoder for oversized or pathological members and its
        // cap-exhaustion/error semantics.
        let mut scan = match ScanAhead::new(source, options.decoded_limit, uploaded) {
            Ok(scan) => scan,
            Err(code) => {
                return Ok(Some(GpuTranscodeResult {
                    output: Vec::new(),
                    decoded_len: None,
                    decoded_bytes_indexed: 0,
                    decoded_bytes_processed: 0,
                    source_bytes_uploaded: uploaded,
                    descriptor_bytes_uploaded: 0,
                    device_indexed: false,
                    device_tokens: 0,
                    code,
                }))
            }
        };
        if options.minimum_decoded_len != 0 {
            let minimum = options.minimum_decoded_len as usize;
            if let Err(code) = scan.fill_to(minimum - 1) {
                return Ok(Some(scan.result(Vec::new(), code)));
            }
            if scan.decoded() < minimum {
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

    #[cfg(feature = "parallel")]
    #[test]
    #[ignore = "requires CUDA hardware and NVRTC"]
    fn cuda_transcode_batch_preserves_order_and_routing() -> Result<(), Box<dyn std::error::Error>>
    {
        let searcher = GpuMatchSearcher::new(Some("cuda:0"))?;
        let pool = crate::BatchCodec::new(NonZeroUsize::new(4).ok_or("workers")?)?;
        let input = b"independent CUDA decode lane payload".repeat(27_000);
        let mut source = vec![0; compress_bound(input.len())];
        let (source, code) = compress_slice(&mut source, &input, DeflateConfig::new(6));
        let source: &[u8] = source;
        assert_eq!(code, ReturnCode::Ok);
        let options = GpuTranscodeOptions {
            output_limit: compress_bound(input.len()),
            decoded_limit: u32::try_from(input.len())?,
            minimum_decoded_len: 0,
            config: DeflateConfig::new(9),
            search_budget: 128,
            window: NonZeroUsize::new(65_537).ok_or("window")?,
        };
        let mut jobs = [
            (source, options),
            (
                source,
                GpuTranscodeOptions {
                    output_limit: 8,
                    ..options
                },
            ),
            (
                source,
                GpuTranscodeOptions {
                    decoded_limit: u32::MAX,
                    minimum_decoded_len: options.decoded_limit + 1,
                    ..options
                },
            ),
            (&source[..source.len() - 1], options),
            (source, options),
        ];
        let expected = jobs
            .iter()
            .map(|&(source, options)| searcher.recompress_zlib(source, options))
            .collect::<Result<Vec<_>, _>>()?;
        let actual = pool.recompress_zlib_batch(&mut jobs, &searcher)?;
        for (actual, expected) in actual.into_iter().zip(expected) {
            match (actual, expected) {
                (Some(a), Some(e)) => {
                    assert_eq!(a.code, e.code);
                    assert_eq!(a.output, e.output);
                    assert_eq!(a.decoded_len, e.decoded_len);
                    assert_eq!(a.decoded_bytes_processed, e.decoded_bytes_processed);
                }
                (a, e) => assert_eq!(a.is_none(), e.is_none()),
            }
        }
        Ok(())
    }

    fn check_bad_source(
        searcher: &GpuMatchSearcher,
        source: &[u8],
        expected: &[u8],
        max: u32,
        options: impl Fn(usize, u32) -> GpuTranscodeOptions,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut corrupt = source.to_vec();
        let last = corrupt.last_mut().ok_or("trailer")?;
        *last ^= 1;
        let result = searcher
            .recompress_zlib(&corrupt, options(expected.len() + 512, max))?
            .ok_or("unexpected transcode routing")?;
        assert_eq!(result.code, ReturnCode::DataError);
        assert_eq!(result.output, [] as [u8; 0]);
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
        Ok(())
    }

    fn check_resident_index(result: &GpuTranscodeResult, source: &[u8], cap: usize) {
        if cap > 2 {
            assert!(
                result.device_indexed,
                "valid fixture must use the GPU index"
            );
            assert_eq!(result.source_bytes_uploaded, source.len());
            assert_eq!(result.descriptor_bytes_uploaded, 0);
        }
    }

    #[test]
    #[ignore = "requires CUDA hardware and NVRTC"]
    fn cuda_transcode_source_windows_and_oversize_index_fallback(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let searcher = GpuMatchSearcher::new(Some("cuda:0"))?;
        let input = b"resident token ranges with small source dictionaries".repeat(8193);
        let config = DeflateConfig::new(9);
        let window = NonZeroUsize::new(131_071).ok_or("window")?;
        let options = GpuTranscodeOptions {
            output_limit: compress_bound(input.len()),
            decoded_limit: u32::try_from(input.len())?,
            minimum_decoded_len: 0,
            config,
            search_budget: 128,
            window,
        };
        let mut expected = Vec::new();
        let result = compress_slices_batch_gpu_windowed(
            &mut [BatchJob::with_vec(
                &input,
                &mut expected,
                options.output_limit,
            )?],
            config,
            &searcher,
            options.search_budget,
            window,
        )?[0];
        assert_eq!(result.code, ReturnCode::Ok);
        for window_bits in 9..=15 {
            let mut source = vec![0; compress_bound(input.len())];
            let (source, code) = compress_slice(
                &mut source,
                &input,
                DeflateConfig {
                    window_bits,
                    ..config
                },
            );
            assert_eq!(code, ReturnCode::Ok);
            let result = searcher
                .recompress_zlib(source, options)?
                .ok_or("missing transcode")?;
            assert_eq!(result.code, ReturnCode::Ok);
            assert_eq!(result.output, expected);
            check_resident_index(&result, source, options.output_limit);
            assert_eq!(source[2] & 6, 4, "fixture requires a dynamic block");
            assert!(result.device_tokens > 0, "dynamic tokens were not retained");
        }

        let large = vec![42; index::MAX_SOURCE + 1];
        let mut source = vec![0; compress_bound(large.len())];
        let (source, code) = compress_slice(&mut source, &large, DeflateConfig::new(0));
        assert_eq!(code, ReturnCode::Ok);
        assert!(source.len() > index::MAX_SOURCE);
        let options = GpuTranscodeOptions {
            decoded_limit: u32::try_from(large.len())?,
            output_limit: compress_bound(large.len()),
            window: NonZeroUsize::new(1_048_576).ok_or("window")?,
            ..options
        };
        let result = searcher
            .recompress_zlib(source, options)?
            .ok_or("missing fallback transcode")?;
        assert_eq!(result.code, ReturnCode::Ok);
        assert!(!result.device_indexed);
        assert_eq!(result.device_tokens, 0);
        assert!(result.descriptor_bytes_uploaded > 0);
        let mut decoded = vec![0; large.len()];
        let (decoded, code) =
            decompress_slice(&mut decoded, &result.output, InflateConfig::default());
        assert_eq!(code, ReturnCode::Ok);
        assert_eq!(decoded, large);
        Ok(())
    }

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
                let skipped = searcher.recompress_zlib(
                    source,
                    GpuTranscodeOptions {
                        minimum_decoded_len: max.saturating_add(1),
                        decoded_limit: u32::MAX,
                        ..options(expected.len() + 512, max)
                    },
                )?;
                assert!(skipped.is_none());
                for cap in [0, 1, expected.len() / 2, expected.len(), expected.len() + 1] {
                    let actual = searcher
                        .recompress_zlib(source, options(cap, max))?
                        .ok_or("unexpected transcode routing")?;
                    check_resident_index(&actual, source, cap);
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
                check_bad_source(&searcher, source, &expected, max, options)?;
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
