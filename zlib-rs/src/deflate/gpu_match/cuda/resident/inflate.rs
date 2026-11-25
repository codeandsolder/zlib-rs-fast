//! Bounded compressed-input pipeline. Only the validated entropy index and
//! compressed source bytes are uploaded. Decoded windows never leave CUDA.
use super::super::{allocation_capacity, failure, reserve};
use super::{
    append, stream_header, Buffers, Checksum, Geometry, ResidentEncoder, ResidentPool, Wrapper,
    MAX_WINDOW, WORKGROUP_SIZE,
};
use crate::{
    deflate::gpu_match::{GpuMatchError, GpuTranscodeResult},
    inflate::gpu_index::{index_zlib, InflateIndex, InflateSegment},
    DeflateConfig, ReturnCode,
};
use cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use std::{
    num::NonZeroUsize,
    sync::{Arc, Mutex},
};

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
        index: &InflateIndex,
        cursor: &mut Cursor,
        target: u32,
    ) -> Result<(), GpuMatchError> {
        while cursor.decoded < target {
            let group = Group::next(source, &index.segments[cursor.segment..], target)?;
            self.decode_group(w, &group, cursor, index.window)?;
        }
        Ok(())
    }
}

impl ResidentEncoder {
    fn transcode(
        &self,
        source: &[u8],
        index: &InflateIndex,
        limit: usize,
        config: DeflateConfig,
        budget: u32,
        window: NonZeroUsize,
        distance: u32,
        index_tiles: u32,
    ) -> Result<GpuTranscodeResult, GpuMatchError> {
        let wrapper = Wrapper::from_bits(config.window_bits)
            .ok_or_else(|| GpuMatchError("invalid transcode wrapper".into()))?;
        let mut output = Vec::new();
        let (header, n) = stream_header(wrapper, config.level);
        let total = index.decoded_len as usize;
        let result = |output, code| GpuTranscodeResult {
            output,
            decoded_len: total,
            code,
        };
        if !append(&mut output, &header[..n], limit) {
            return Ok(result(output, ReturnCode::BufError));
        }
        let mut checksum = Checksum::new(wrapper);
        if total == 0 && !append(&mut output, &[3, 0], limit) {
            return Ok(result(output, ReturnCode::BufError));
        }
        let window = window.get().min(MAX_WINDOW);
        let mut w = self.decoder.workspace.lock().map_err(failure)?;
        reserve(&self.stream, &mut w.raw, window.min(total) + CARRY_BYTES)?;
        reserve(&self.stream, &mut w.carry, CARRY_BYTES)?;
        let mut cursor = Cursor::default();
        let history = (index.window as usize).max(distance as usize - 1);
        let mut b = self.buffers.lock().map_err(failure)?;
        for start in (0..total).step_by(window) {
            if output.len() == limit {
                return Ok(result(output, ReturnCode::BufError));
            }
            let end = start.saturating_add(window).min(total);
            let origin = start.saturating_sub(history);
            self.decoder.retain_history(
                &mut w,
                &mut cursor,
                u32::try_from(origin).map_err(failure)?,
            )?;
            self.decoder.decode_to(
                &mut w,
                source,
                index,
                &mut cursor,
                u32::try_from(end).map_err(failure)?,
            )?;
            let input_origin = start.saturating_sub(distance as usize - 1);
            let g = Geometry::new(
                end - input_origin,
                start - input_origin,
                distance,
                budget,
                end == total,
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
            let piece = self.window_loaded(&mut b, g, limit - output.len())?;
            output.extend_from_slice(&piece.output);
            if !piece.complete {
                return Ok(result(output, ReturnCode::BufError));
            }
            checksum.update(&piece, end - start);
        }
        let (trailer, n) = checksum.trailer(total);
        // This API emits zlib; the output Adler also authenticates the
        // original member. Never publish an accepted corrupt source.
        if trailer[..n] != index.adler.to_be_bytes() {
            return Ok(result(Vec::new(), ReturnCode::DataError));
        }
        let code = if append(&mut output, &trailer[..n], limit) {
            ReturnCode::Ok
        } else {
            ReturnCode::BufError
        };
        Ok(result(output, code))
    }
}

impl ResidentPool {
    pub(in crate::deflate::gpu_match::cuda) fn recompress_zlib(
        &self,
        source: &[u8],
        limit: usize,
        decoded_limit: u32,
        config: DeflateConfig,
        budget: u32,
        window: NonZeroUsize,
        distance: u32,
    ) -> Result<GpuTranscodeResult, GpuMatchError> {
        let mut index = match index_zlib(source, decoded_limit, MAX_INDEX_SEGMENTS) {
            Ok(index) => index,
            Err(code) => {
                return Ok(GpuTranscodeResult {
                    output: Vec::new(),
                    decoded_len: 0,
                    code,
                })
            }
        };
        // Empty blocks have no bytes to reconstruct; the structural
        // scanner has already validated their headers and EOB symbols.
        index.segments.retain(|s| s.start != s.end);
        self.acquire()?.transcode(
            source,
            &index,
            limit,
            config,
            budget,
            window,
            distance,
            self.index_tiles,
        )
    }
}
