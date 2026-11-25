//! Whole GPU-resident DEFLATE windows. The only host-device payloads are
//! decoded input, compact compressed output and four metadata words.
use super::{allocation_capacity, failure, reserve};
use crate::{
    deflate::gpu_match::{GpuMatchError, ResidentSchedule, MAX_MATCH, WORKGROUP_SIZE},
    DeflateConfig, ReturnCode,
};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use std::sync::{Arc, Condvar, Mutex};

const MIB: usize = 1024 * 1024;
pub(in crate::deflate) const DEFAULT_WINDOW: usize = 16 * MIB;
const MAX_WINDOW: usize = 64 * MIB;
const MIN_MATCH_TILE: u32 = 16 * 1024;
// Local head positions are one-based u16 values.
const MAX_MATCH_TILE: u32 = 32 * 1024;
const SEGMENT: u32 = 8 * 1024;
const HASH_COUNT: u32 = 1 << 16;
const LITERAL_SYMBOLS: u32 = 286;
const DISTANCE_SYMBOLS: u32 = 30;
const HIST_WORDS: u32 = LITERAL_SYMBOLS + DISTANCE_SYMBOLS + 1;
const BLOCK_WORDS: u32 = LITERAL_SYMBOLS + DISTANCE_SYMBOLS + 5;
const ARENA_STRIDE: u32 = SEGMENT + MAX_MATCH.next_multiple_of(4) + 12;
const MAX_BLOCK_SPAN: u32 = 4 * 1024 * 1024;
const ADLER_MOD: u64 = 65_521;

#[derive(Clone, Copy)]
enum Wrapper {
    Raw,
    Zlib(u8),
    Gzip,
}

impl Wrapper {
    fn from_bits(bits: i32) -> Option<Self> {
        match bits {
            -15..=-9 => Some(Self::Raw),
            8..=15 => Some(Self::Zlib(u8::try_from(bits.max(9)).ok()?)),
            25..=31 => Some(Self::Gzip),
            _ => None,
        }
    }

    const fn checksum_kind(self) -> u32 {
        match self {
            Self::Raw => 0,
            Self::Zlib(_) => 1,
            Self::Gzip => 2,
        }
    }
}

pub(super) fn supports(config: DeflateConfig) -> bool {
    use crate::deflate::{Method, Strategy};
    matches!(config.level, -1 | 1..=9)
        && config.method == Method::Deflated
        && (1..=9).contains(&config.mem_level)
        && matches!(
            config.strategy,
            Strategy::Default | Strategy::Filtered | Strategy::Fixed
        )
        && Wrapper::from_bits(config.window_bits).is_some()
}

macro_rules! kernel_set {
    ($($name:ident),+ $(,)?) => {
        struct Kernels { $($name: CudaFunction),+ }
        impl Kernels {
            fn new(module: &Arc<CudaModule>) -> Result<Self, GpuMatchError> {
                Ok(Self { $($name: module.load_function(stringify!($name)).map_err(failure)?),+ })
            }
        }
    };
}

kernel_set! {
    expand_aligned, resident_index, resident_index_bridge, resident_match, resident_choices,
    resident_mark_paths, resident_transitions, resident_scan_paths, resident_parse_paths,
    resident_partition, resident_histogram, resident_tree, resident_segment_bits,
    resident_bit_offsets, resident_write_segments, resident_write_stored,
    resident_adler, resident_crc, resident_parallel_offsets, resident_parallel_compact,
}

macro_rules! buffer_set {
    ($($name:ident: $ty:ty),+ $(,)?) => {
        struct Buffers { $($name: CudaSlice<$ty>),+ }
        // Keep the reservation and the accounting in one place. All products
// are bounded by the validated MAX_WINDOW and fixed kernel geometry.
macro_rules! workspace_geometry {
    ($macro:ident, $g:ident) => {
        $macro! {
            input: u8 => $g.size.next_multiple_of(4), aligned: u32 => $g.size,
            heads: u16 => $g.index_tiles * HASH_COUNT, predecessors: u16 => $g.size.next_multiple_of(2),
            matches: u32 => $g.core,
            exits: u32 => $g.segments, transitions: u16 => $g.segments * MAX_MATCH,
            entries: u32 => $g.segments + 1, counts: u32 => $g.segments,
            block_ids: u32 => $g.segments, blocks: u32 => $g.segments * BLOCK_WORDS,
            hist: u32 => $g.segments * HIST_WORDS, arena: u32 => $g.arena_bytes() / 4,
            bits: u32 => $g.segments, bit_offsets: u32 => $g.segments,
            checksums: u32 => $g.tiles * 2, offsets: u32 => $g.segments, meta: u32 => 4,
            compact: u8 => $g.arena_bytes(),
        }
    };
}

impl Buffers {
            fn new(stream: &Arc<CudaStream>) -> Result<Self, GpuMatchError> {
                Ok(Self { $($name: stream.alloc_zeros(1).map_err(failure)?),+ })
            }
        }
    };
}

buffer_set! {
    input: u8, aligned: u32, heads: u16, predecessors: u16, matches: u32,
    exits: u32, transitions: u16, entries: u32, counts: u32,
    block_ids: u32, blocks: u32, block_count: u32, hist: u32, arena: u32,
    bits: u32, bit_offsets: u32, checksums: u32, offsets: u32, meta: u32,
    compact: u8,
}

#[derive(Clone, Copy)]
struct Geometry {
    size: u32,
    core: u32,
    history: u32,
    distance: u32,
    budget: u32,
    tile: u32,
    tiles: u32,
    index_tiles: u32,
    segments: u32,
    target_tokens: u32,
    final_window: u32,
    strategy: u32,
    checksum_kind: u32,
}

impl Geometry {
    fn new(
        input_len: usize,
        history: usize,
        distance: u32,
        budget: u32,
        final_window: bool,
        config: DeflateConfig,
        wrapper: Wrapper,
    ) -> Result<Self, GpuMatchError> {
        let size = u32::try_from(input_len).map_err(failure)?;
        let history = u32::try_from(history).map_err(failure)?;
        let core = size
            .checked_sub(history)
            .filter(|&n| n != 0 && n as usize <= MAX_WINDOW)
            .ok_or_else(|| GpuMatchError("invalid resident CUDA core window".into()))?;
        if !(1..=32_768).contains(&distance) || budget == 0 {
            return Err(GpuMatchError(
                "invalid resident CUDA search parameters".into(),
            ));
        }
        Ok(Self {
            size,
            core,
            history,
            distance,
            budget,
            tile: MAX_MATCH_TILE,
            tiles: core.div_ceil(MAX_MATCH_TILE),
            index_tiles: size.div_ceil(MAX_MATCH_TILE),
            segments: core.div_ceil(SEGMENT),
            target_tokens: (1u32 << (config.mem_level + 6)) - 1,
            final_window: u32::from(final_window),
            strategy: config.strategy as u32,
            checksum_kind: wrapper.checksum_kind(),
        })
    }

    // Target enough independent index warps to distribute four per SM.
    // Limit the lower end to avoid excessive duplicated history/hash heads.
    // Geometry affects throughput only: each tile sees exact history and lookahead.
    fn with_index_tiles(mut self, target: u32) -> Self {
        self.tile = self
            .core
            .div_ceil(target.max(1))
            .next_power_of_two()
            .clamp(MIN_MATCH_TILE, MAX_MATCH_TILE);
        self.tiles = self.core.div_ceil(self.tile);
        self.index_tiles = self.size.div_ceil(self.tile);
        self
    }

    const fn arena_bytes(self) -> u32 {
        self.segments * ARENA_STRIDE
    }
}

impl Buffers {
    fn prepare(&mut self, stream: &Arc<CudaStream>, g: Geometry) -> Result<(), GpuMatchError> {
        macro_rules! grow {
            ($($field:ident : $ty:ty => $count:expr),+ $(,)?) => {
                $(reserve(stream, &mut self.$field, $count as usize)?;)+
            };
        }
        workspace_geometry!(grow, g);
        Ok(())
    }

    fn upper_bound(g: Geometry) -> usize {
        macro_rules! count {
            ($($field:ident : $ty:ty => $count:expr),+ $(,)?) => {
                0 $(+ allocation_capacity($count as usize) * core::mem::size_of::<$ty>())+
            };
        }
        workspace_geometry!(count, g)
    }
}

pub(super) struct ResidentEncoder {
    stream: Arc<CudaStream>,
    kernels: Kernels,
    buffers: Mutex<Buffers>,
}

/// Independent CUDA streams may overlap kernel execution, transfers and
/// launch latency. The lane count is bounded by the largest supported
/// workspace rather than by the host worker count.
pub(super) struct ResidentPool {
    lanes: Vec<ResidentEncoder>,
    available: Mutex<Vec<usize>>,
    ready: Condvar,
    index_tiles: u32,
}

struct Lane<'a> {
    pool: &'a ResidentPool,
    index: usize,
}

impl core::ops::Deref for Lane<'_> {
    type Target = ResidentEncoder;
    fn deref(&self) -> &Self::Target {
        &self.pool.lanes[self.index]
    }
}

impl Drop for Lane<'_> {
    fn drop(&mut self) {
        let mut available = self
            .pool
            .available
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        available.push(self.index);
        drop(available);
        self.pool.ready.notify_one();
    }
}

impl ResidentPool {
    pub(super) fn new(
        context: &Arc<CudaContext>,
        module: &Arc<CudaModule>,
        multiprocessors: u32,
    ) -> Result<Self, GpuMatchError> {
        let index_tiles = multiprocessors.saturating_mul(4).max(1);
        // The peak is the full window with maximum history, at the smallest
        // tile size that the adaptive geometry can select for that window.
        let worst = Geometry::new(
            MAX_WINDOW + 32_767,
            32_767,
            32_768,
            1,
            false,
            DeflateConfig::new(9),
            Wrapper::Zlib(15),
        )?
        .with_index_tiles(index_tiles);
        let bytes = Buffers::upper_bound(worst);
        context.bind_to_thread().map_err(failure)?;
        let (free, _) = cudarc::driver::result::mem_get_info().map_err(failure)?;
        // Leave space for display, context, other workloads and transient
        // reallocation. A single lane remains usable on small devices.
        let count = (free.saturating_sub(free / 4) / bytes).clamp(1, 4);
        let lanes = (0..count)
            .map(|_| ResidentEncoder::new(context.new_stream().map_err(failure)?, module))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            lanes,
            available: Mutex::new((0..count).rev().collect()),
            ready: Condvar::new(),
            index_tiles,
        })
    }

    fn acquire(&self) -> Result<Lane<'_>, GpuMatchError> {
        let mut available = self.available.lock().map_err(failure)?;
        loop {
            if let Some(index) = available.pop() {
                return Ok(Lane { pool: self, index });
            }
            available = self.ready.wait(available).map_err(failure)?;
        }
    }

    fn window(
        &self,
        input: &[u8],
        g: Geometry,
        limit: usize,
    ) -> Result<WindowOutput, GpuMatchError> {
        self.acquire()?.window(input, g, limit)
    }
}

// All launches within a lane share one stream and a locked buffer set. Each kernel's ABI
// is declared in resident.cu / resident_parallel.cu. Geometry::new and
// Buffers::prepare establish the exact index, token, tree and arena bounds.
// Input/output buffers use the element widths expected by those kernels.
macro_rules! launch {
    ($this:expr, $name:ident, $grid:expr, $threads:expr; $($arg:expr),* $(,)?) => {{
        let mut args = $this.stream.launch_builder(&$this.kernels.$name);
        $(args.arg($arg);)*
        // SAFETY: the phase-specific arguments match the CUDA kernel ABI;
        // all device pointers refer to the validated exclusive workspace.
        unsafe { args.launch(LaunchConfig {
            grid_dim: $grid, block_dim: ($threads, 1, 1), shared_mem_bytes: 0,
        }) }.map_err(failure)?;
    }};
}

impl ResidentEncoder {
    pub(super) fn new(
        stream: Arc<CudaStream>,
        module: &Arc<CudaModule>,
    ) -> Result<Self, GpuMatchError> {
        let buffers = Mutex::new(Buffers::new(&stream)?);
        Ok(Self {
            stream,
            kernels: Kernels::new(module)?,
            buffers,
        })
    }

    #[expect(unsafe_code, reason = "validated CUDA index and match kernel ABI")]
    fn index_and_match(&self, b: &mut Buffers, g: Geometry) -> Result<(), GpuMatchError> {
        launch!(self, expand_aligned, (g.size.div_ceil(WORKGROUP_SIZE), 1, 1), WORKGROUP_SIZE;
            &b.input, &mut b.aligned, &g.size);
        launch!(self, resident_index, (g.index_tiles.div_ceil(WORKGROUP_SIZE / 32), 1, 1), WORKGROUP_SIZE;
            &b.aligned, &mut b.heads, &mut b.predecessors, &g.size,
            &g.tile, &g.distance, &g.index_tiles);
        if g.index_tiles > 1 && g.distance > 1 {
            launch!(self, resident_index_bridge, (g.tile.min(g.distance - 1).div_ceil(WORKGROUP_SIZE), g.index_tiles, 1), WORKGROUP_SIZE;
                &b.aligned, &b.heads, &mut b.predecessors, &g.size, &g.tile, &g.distance);
        }
        launch!(self, resident_match, (g.core.div_ceil(WORKGROUP_SIZE), 1, 1), WORKGROUP_SIZE;
            &b.aligned, &b.predecessors, &mut b.matches, &g.size, &g.history,
            &g.distance, &g.budget);
        Ok(())
    }

    #[expect(
        unsafe_code,
        reason = "validated CUDA path-reconciliation and partition kernel ABI"
    )]
    fn parse(&self, b: &mut Buffers, g: Geometry) -> Result<(), GpuMatchError> {
        let segment_grid = (g.segments.div_ceil(32), 1, 1);
        // The aligned match-search input is dead after resident_match.
        // Reuse that allocation for choices; original literals, stored
        // blocks and checksum kernels read the compact immutable input.
        launch!(self, resident_choices, (g.core.div_ceil(WORKGROUP_SIZE), 1, 1), WORKGROUP_SIZE;
            &b.input, &b.matches, &mut b.aligned, &g.core, &g.history, &g.strategy);
        launch!(self, resident_mark_paths, segment_grid, 32;
            &mut b.aligned, &mut b.exits, &g.core, &SEGMENT, &g.segments);
        launch!(self, resident_transitions, (g.segments, 1, 1), 256;
            &b.aligned, &b.exits, &mut b.transitions, &g.core, &SEGMENT);
        launch!(self, resident_scan_paths, (1, 1, 1), 1;
            &b.transitions, &mut b.entries, &g.segments);
        launch!(self, resident_parse_paths, segment_grid, 32;
            &b.aligned, &mut b.matches, &b.entries, &mut b.counts, &g.core, &SEGMENT, &g.segments);
        launch!(self, resident_partition, (1, 1, 1), 1;
            &b.counts, &mut b.block_ids, &mut b.blocks, &mut b.block_count,
            &g.segments, &g.target_tokens, &(MAX_BLOCK_SPAN / SEGMENT));
        Ok(())
    }

    #[expect(
        unsafe_code,
        reason = "validated CUDA histogram, Huffman and bit emission kernel ABI"
    )]
    fn encode(&self, b: &mut Buffers, g: Geometry) -> Result<(), GpuMatchError> {
        let segment_grid = (g.segments.div_ceil(32), 1, 1);
        let all_segments = (g.segments, 1, 1);
        self.stream
            .memset_zeros(&mut b.hist.slice_mut(..(g.segments * HIST_WORDS) as usize))
            .map_err(failure)?;
        self.stream
            .memset_zeros(&mut b.arena.slice_mut(..(g.arena_bytes() / 4) as usize))
            .map_err(failure)?;
        launch!(self, resident_histogram, all_segments, WORKGROUP_SIZE;
            &b.matches, &b.counts, &b.block_ids, &mut b.hist, &SEGMENT);
        launch!(self, resident_tree, segment_grid, 32;
            &mut b.blocks, &b.block_count, &mut b.hist, &mut b.arena, &b.entries,
            &g.core, &SEGMENT, &ARENA_STRIDE, &g.final_window, &g.strategy);
        launch!(self, resident_segment_bits, all_segments, WORKGROUP_SIZE;
            &b.matches, &b.counts, &b.block_ids, &b.blocks, &mut b.bits, &SEGMENT);
        launch!(self, resident_bit_offsets, segment_grid, 32;
            &b.blocks, &b.block_count, &b.bits, &mut b.bit_offsets);
        launch!(self, resident_write_segments, segment_grid, 32;
            &b.matches, &b.counts, &b.block_ids, &b.blocks, &b.bit_offsets,
            &mut b.arena, &SEGMENT, &ARENA_STRIDE, &g.segments);
        launch!(self, resident_write_stored, all_segments, WORKGROUP_SIZE;
            &b.input, &b.block_ids, &b.blocks, &mut b.arena, &b.entries,
            &g.core, &g.history, &SEGMENT, &ARENA_STRIDE);
        Ok(())
    }

    #[expect(
        unsafe_code,
        reason = "validated CUDA checksum and output compaction kernel ABI"
    )]
    fn compact(&self, b: &mut Buffers, g: Geometry) -> Result<(), GpuMatchError> {
        if g.checksum_kind == 1 {
            launch!(self, resident_adler, (g.tiles, 1, 1), WORKGROUP_SIZE;
                &b.input, &mut b.checksums, &g.size, &g.history, &g.tile, &g.distance);
        } else if g.checksum_kind == 2 {
            launch!(self, resident_crc, (g.tiles, 1, 1), WORKGROUP_SIZE;
                &b.input, &mut b.checksums, &g.size, &g.history, &g.tile, &g.distance);
        }
        launch!(self, resident_parallel_offsets, (1, 1, 1), 1;
            &b.blocks, &b.block_count, &b.checksums, &mut b.offsets, &mut b.meta,
            &g.core, &g.tile, &g.tiles, &g.checksum_kind);
        launch!(self, resident_parallel_compact, (g.segments, 1, 1), WORKGROUP_SIZE;
            &b.blocks, &b.block_count, &b.offsets, &b.arena, &mut b.compact, &ARENA_STRIDE);
        Ok(())
    }

    fn window(
        &self,
        input: &[u8],
        g: Geometry,
        limit: usize,
    ) -> Result<WindowOutput, GpuMatchError> {
        let mut b = self.buffers.lock().map_err(failure)?;
        b.prepare(&self.stream, g)?;
        self.stream
            .memcpy_htod(input, &mut b.input)
            .map_err(failure)?;
        self.index_and_match(&mut b, g)?;
        self.parse(&mut b, g)?;
        self.encode(&mut b, g)?;
        self.compact(&mut b, g)?;
        let meta = self
            .stream
            .clone_dtoh(&b.meta.slice(..4))
            .map_err(failure)?;
        let [len, a, checksum_b, blocks] = <[u32; 4]>::try_from(meta).map_err(failure)?;
        if len > g.arena_bytes() || blocks == 0 || blocks > g.segments {
            return Err(GpuMatchError(
                "invalid resident CUDA output metadata".into(),
            ));
        }
        let count = (len as usize).min(limit);
        let output = if count == 0 {
            Vec::new()
        } else {
            self.stream
                .clone_dtoh(&b.compact.slice(..count))
                .map_err(failure)?
        };
        Ok(WindowOutput {
            output,
            complete: count == len as usize,
            a,
            b: checksum_b,
        })
    }
}

impl ResidentPool {
    pub(super) fn compress(
        &self,
        input: &[u8],
        limit: usize,
        config: DeflateConfig,
        budget: u32,
        schedule: ResidentSchedule,
        distance: u32,
    ) -> Result<(Vec<u8>, ReturnCode), GpuMatchError> {
        let wrapper = Wrapper::from_bits(config.window_bits)
            .ok_or_else(|| GpuMatchError("invalid resident CUDA wrapper".into()))?;
        let mut output = Vec::new();
        let (header, header_len) = stream_header(wrapper, config.level);
        if !append(&mut output, &header[..header_len], limit) {
            return Ok((output, ReturnCode::BufError));
        }
        let mut checksum = Checksum::new(wrapper);
        if input.is_empty() && !append(&mut output, &[3, 0], limit) {
            return Ok((output, ReturnCode::BufError));
        }
        let window = schedule.window.get().min(MAX_WINDOW);
        // Each window is an independent byte-aligned DEFLATE fragment with
        // real preceding history. Reuse the caller's Rayon pool to pipeline
        // a bounded pair, never spawn a second worker pool or threads.
        let pipeline = cfg!(feature = "parallel") && schedule.parallel && self.lanes.len() > 1;
        let mut start = 0;
        while start < input.len() {
            if output.len() == limit {
                return Ok((output, ReturnCode::BufError));
            }
            let remaining = limit - output.len();
            let end = start.saturating_add(window).min(input.len());
            let run = |start: usize, end: usize| {
                let origin = start.saturating_sub(distance as usize - 1);
                let g = Geometry::new(
                    end - origin,
                    start - origin,
                    distance,
                    budget,
                    end == input.len(),
                    config,
                    wrapper,
                )?
                .with_index_tiles(self.index_tiles);
                self.window(&input[origin..end], g, remaining)
            };
            let second_end = if pipeline {
                end.saturating_add(window).min(input.len())
            } else {
                end
            };
            #[cfg(feature = "parallel")]
            let (first, second) = if second_end > end {
                let (a, b) = rayon_core::join(|| run(start, end), || run(end, second_end));
                (a, Some(b))
            } else {
                (run(start, end), None)
            };
            #[cfg(not(feature = "parallel"))]
            let (first, second) = (run(start, end), None::<Result<WindowOutput, GpuMatchError>>);
            let first = first?;
            if !append(&mut output, &first.output, limit) || !first.complete {
                return Ok((output, ReturnCode::BufError));
            }
            checksum.update(&first, end - start);
            if let Some(second) = second {
                if output.len() == limit {
                    return Ok((output, ReturnCode::BufError));
                }
                // Commit only in logical input order. The second fragment
                // was bounded to the prior remaining capacity; truncate
                // to the new remaining capacity if the first consumed it.
                let second = second?;
                if !append(&mut output, &second.output, limit) || !second.complete {
                    return Ok((output, ReturnCode::BufError));
                }
                checksum.update(&second, second_end - end);
            }
            start = second_end;
        }
        let (trailer, trailer_len) = checksum.trailer(input.len());
        let code = if append(&mut output, &trailer[..trailer_len], limit) {
            ReturnCode::Ok
        } else {
            ReturnCode::BufError
        };
        Ok((output, code))
    }
}

struct WindowOutput {
    output: Vec<u8>,
    complete: bool,
    a: u32,
    b: u32,
}

fn append(output: &mut Vec<u8>, bytes: &[u8], limit: usize) -> bool {
    let count = bytes.len().min(limit - output.len());
    output.extend_from_slice(&bytes[..count]);
    count == bytes.len()
}

const fn stream_header(wrapper: Wrapper, level: i32) -> ([u8; 10], usize) {
    let mut header = [0; 10];
    let len = match wrapper {
        Wrapper::Raw => 0,
        Wrapper::Zlib(bits) => {
            let cmf = ((bits - 8) << 4) | 8;
            let level_bits = match level {
                0..=1 => 0,
                2..=5 => 1,
                -1 | 6 => 2,
                _ => 3,
            };
            let flg = level_bits << 6;
            let combined = u16::from_be_bytes([cmf, flg]);
            header[0] = cmf;
            header[1] = flg | ((31 - combined % 31) % 31).to_le_bytes()[0];
            2
        }
        Wrapper::Gzip => {
            header = [
                31,
                139,
                8,
                0,
                0,
                0,
                0,
                0,
                match level {
                    1 => 4,
                    9 => 2,
                    _ => 0,
                },
                255,
            ];
            10
        }
    };
    (header, len)
}

enum Checksum {
    None,
    Adler { a: u64, b: u64 },
    Crc(u32),
}

impl Checksum {
    const fn new(wrapper: Wrapper) -> Self {
        match wrapper {
            Wrapper::Raw => Self::None,
            Wrapper::Zlib(_) => Self::Adler { a: 1, b: 0 },
            Wrapper::Gzip => Self::Crc(0),
        }
    }

    fn update(&mut self, result: &WindowOutput, length: usize) {
        match self {
            Self::None => {}
            Self::Adler { a, b } => {
                *b = (*b + u64::from(result.b) + length as u64 * *a) % ADLER_MOD;
                *a = (*a + u64::from(result.a)) % ADLER_MOD;
            }
            Self::Crc(crc) => *crc = crate::crc32::crc32_combine(*crc, result.a, length as u64),
        }
    }

    fn trailer(self, len: usize) -> ([u8; 8], usize) {
        let mut bytes = [0; 8];
        let size = match self {
            Self::None => 0,
            Self::Adler { a, b } => {
                let combined = ((b << 16) | a).to_be_bytes();
                bytes[..4].copy_from_slice(&combined[4..]);
                4
            }
            Self::Crc(crc) => {
                bytes[..4].copy_from_slice(&crc.to_le_bytes());
                bytes[4..].copy_from_slice(&len.to_le_bytes()[..4]);
                8
            }
        };
        (bytes, size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deflate::gpu_match::{build_predecessors, predecessor_distance, GpuMatchSearcher};
    use std::num::NonZeroUsize;

    #[test]
    #[ignore = "requires CUDA hardware and NVRTC"]
    fn cuda_resident_concurrent_windows_match_serial() -> Result<(), Box<dyn std::error::Error>> {
        let (backend, _) = super::super::CudaBackend::new(0, true)?;
        let pool = backend.resident.as_deref().ok_or("resident encoder")?;
        let mut seed = 0x1234_5678u32;
        let random = (0..100_013)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                seed.to_le_bytes()[0]
            })
            .collect::<Vec<_>>();
        let repeated = b"PDF resident concurrent CUDA windows/".repeat(20_000);
        let periodic = random[..32_503].repeat(8);
        let inputs = [random, repeated, periodic, vec![0; 300_013]];
        let window = NonZeroUsize::new(65_537).ok_or("window")?;
        let compress = |input: &[u8]| {
            pool.compress(
                input,
                input.len() + 512,
                DeflateConfig::new(9),
                128,
                ResidentSchedule {
                    window,
                    parallel: false,
                },
                32_506,
            )
        };
        let serial = inputs
            .iter()
            .map(|input| compress(input))
            .collect::<Result<Vec<_>, _>>()?;
        let parallel = std::thread::scope(|scope| {
            // Collect the handles before joining: joining inside the spawn
            // iterator would serialize the hardware concurrency test.
            #[expect(
                clippy::needless_collect,
                reason = "spawn all workers before joining any"
            )]
            let workers = inputs
                .iter()
                .map(|input| scope.spawn(|| compress(input)))
                .collect::<Vec<_>>();
            workers
                .into_iter()
                .map(|worker| {
                    worker
                        .join()
                        .map_err(|_| "resident worker panicked")?
                        .map_err(|_| "resident compression failed")
                })
                .collect::<Result<Vec<_>, _>>()
        })?;
        assert_eq!(serial, parallel);
        #[cfg(feature = "parallel")]
        {
            let workers = rayon_core::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()?;
            for (input, (full, code)) in inputs.iter().zip(&serial) {
                assert_eq!(*code, ReturnCode::Ok);
                for cap in [0, 1, full.len() / 2, full.len(), full.len() + 1] {
                    let actual = workers.install(|| {
                        pool.compress(
                            input,
                            cap,
                            DeflateConfig::new(9),
                            128,
                            ResidentSchedule {
                                window,
                                parallel: true,
                            },
                            32_506,
                        )
                    })?;
                    assert_eq!(actual.0, full[..cap.min(full.len())]);
                    assert_eq!(
                        actual.1,
                        if cap < full.len() {
                            ReturnCode::BufError
                        } else {
                            ReturnCode::Ok
                        }
                    );
                }
            }
        }

        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA hardware and NVRTC"]
    fn cuda_resident_index_and_parse_are_exact() -> Result<(), Box<dyn std::error::Error>> {
        let (backend, _) = super::super::CudaBackend::new(0, true)?;
        let pool = backend.resident.as_deref().ok_or("resident encoder")?;
        let encoder = pool.acquire()?;
        let reference = GpuMatchSearcher::new(Some("cuda-match:0"))?;
        let mut seed = 0x9e37_79b9u32;
        let random = (0..70_013)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                seed.to_le_bytes()[0] & 31
            })
            .collect::<Vec<_>>();
        let mut input = random.repeat(3);
        input.extend(b"abc".repeat(55_013));
        input.extend(random.iter().copied());
        let history = 30_017;
        let distance = 32_506;
        let budget = 128;
        let config = DeflateConfig::new(9);
        let g = Geometry::new(
            input.len(),
            history,
            distance,
            budget,
            true,
            config,
            Wrapper::Zlib(15),
        )?
        .with_index_tiles(pool.index_tiles);
        let mut b = encoder.buffers.lock().map_err(failure)?;
        b.prepare(&encoder.stream, g)?;
        encoder
            .stream
            .memcpy_htod(&input, &mut b.input)
            .map_err(failure)?;
        encoder.index_and_match(&mut b, g)?;
        let pred = encoder
            .stream
            .clone_dtoh(&b.predecessors.slice(..g.size as usize))
            .map_err(failure)?;
        let matches = encoder
            .stream
            .clone_dtoh(&b.matches.slice(..g.core as usize))
            .map_err(failure)?;
        let expected = build_predecessors(&input, distance)?;
        for (position, &value) in pred.iter().take(input.len() - 2).enumerate() {
            assert_eq!(
                usize::from(value),
                predecessor_distance(&expected, position),
                "position {position}"
            );
        }
        let dense = reference.search_with_max_dist(&input, budget, distance)?;
        assert_eq!(matches, dense.entries[history..]);
        encoder.parse(&mut b, g)?;
        let counts = encoder
            .stream
            .clone_dtoh(&b.counts.slice(..g.segments as usize))
            .map_err(failure)?;
        let actual = encoder
            .stream
            .clone_dtoh(&b.matches.slice(..g.core as usize))
            .map_err(failure)?;
        drop(b);
        let mut expected = Vec::new();
        let mut pos = 0;
        while pos < matches.len() {
            let current = matches[pos];
            let len = current & 65535;
            let next = matches.get(pos + 1).copied().unwrap_or(0) & 65535;
            if len >= 3 && next <= len {
                expected.push(current);
                pos += len as usize;
            } else {
                expected.push(u32::from(input[history + pos]));
                pos += 1;
            }
        }
        let tokens = counts
            .iter()
            .enumerate()
            .flat_map(|(segment, &count)| {
                let start = segment * SEGMENT as usize;
                actual[start..start + count as usize].iter().copied()
            })
            .collect::<Vec<_>>();
        assert_eq!(tokens, expected);
        Ok(())
    }
}
