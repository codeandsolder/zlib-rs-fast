//! Compact-source device index. Every speculative edge is validated by the
//! exact block-chain walk before any segment is exposed to the inflater.
use super::{
    allocation_capacity, failure, reserve, GpuMatchError, InflateSegment, MAX_SEGMENT,
    WORDS_PER_SEGMENT, WORKGROUP_SIZE,
};
use cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use std::sync::Arc;

pub(super) const MAX_SOURCE: usize = 8 * 1024 * 1024;
const MAX_PRELIMINARY: usize = 1 << 20;
const MAX_HEADERS: usize = 16_384;
const TILE_BITS: usize = 512;
const MAX_TILES: usize = MAX_SOURCE * 8 / TILE_BITS + MAX_HEADERS;
const MAX_SEGMENTS: usize = 65_536;
const HEADER_CHUNK_WORDS: u32 = 128;
const TREE_BYTES: usize = 2392;
const PATH_WORDS: usize = 48 * 2;
const BLOCK_WORDS: usize = 3;
const EXTENT_WORDS: usize = 6;

kernel_set! {
    inflate_header_candidates, inflate_validate_bitmap, inflate_count_header_chunks,
    inflate_header_offsets, inflate_scatter_headers, inflate_make_tile_specs,
    inflate_make_tile_tasks, inflate_prepare_tiles, inflate_tile_paths,
    inflate_compose_segments, inflate_assemble_index,
}

pub(super) struct Indexer {
    stream: Arc<CudaStream>,
    kernels: Kernels,
}

pub(super) struct DeviceIndex {
    pub(super) segments: Vec<InflateSegment>,
    pub(super) decoded: u32,
    pub(super) window: u32,
    pub(super) adler: u32,
}

pub(super) struct Workspace {
    pub(super) source: CudaSlice<u8>,
    pub(super) segments: CudaSlice<u64>,
    preliminary: CudaSlice<u64>,
    bitmap: CudaSlice<u32>,
    offsets: CudaSlice<u32>,
    headers: CudaSlice<u64>,
    blocks: CudaSlice<u64>,
    tasks: CudaSlice<u64>,
    trees: CudaSlice<u8>,
    paths: CudaSlice<u64>,
    extents: CudaSlice<u64>,
    relative: CudaSlice<u64>,
    counts: CudaSlice<u32>,
    count: CudaSlice<u32>,
    info: CudaSlice<u32>,
}

impl Workspace {
    pub(super) fn new(stream: &Arc<CudaStream>) -> Result<Self, GpuMatchError> {
        Ok(Self {
            source: stream.alloc_zeros(1).map_err(failure)?,
            segments: stream.alloc_zeros(1).map_err(failure)?,
            preliminary: stream.alloc_zeros(1).map_err(failure)?,
            bitmap: stream.alloc_zeros(1).map_err(failure)?,
            offsets: stream.alloc_zeros(1).map_err(failure)?,
            headers: stream.alloc_zeros(1).map_err(failure)?,
            blocks: stream.alloc_zeros(1).map_err(failure)?,
            tasks: stream.alloc_zeros(1).map_err(failure)?,
            trees: stream.alloc_zeros(1).map_err(failure)?,
            paths: stream.alloc_zeros(1).map_err(failure)?,
            extents: stream.alloc_zeros(1).map_err(failure)?,
            relative: stream.alloc_zeros(1).map_err(failure)?,
            counts: stream.alloc_zeros(1).map_err(failure)?,
            count: stream.alloc_zeros(1).map_err(failure)?,
            info: stream.alloc_zeros(6).map_err(failure)?,
        })
    }

    pub(super) fn upper_bound() -> usize {
        let allocation = |n: usize, width: usize| allocation_capacity(n) * width;
        allocation(MAX_SOURCE + 16, 1)
            + allocation(MAX_SEGMENTS * WORDS_PER_SEGMENT, 8)
            + allocation(MAX_PRELIMINARY, 8)
            + allocation(MAX_SOURCE / 4, 4)
            + allocation(MAX_SOURCE / (4 * HEADER_CHUNK_WORDS as usize) + 1, 4)
            + allocation(MAX_HEADERS, 8)
            + allocation(MAX_HEADERS * BLOCK_WORDS, 8)
            + allocation(MAX_TILES, 8)
            + allocation(MAX_HEADERS * TREE_BYTES, 1)
            + allocation(MAX_TILES * PATH_WORDS, 8)
            + allocation(MAX_HEADERS * EXTENT_WORDS, 8)
            + allocation(MAX_TILES * WORDS_PER_SEGMENT, 8)
            + allocation(MAX_HEADERS, 4)
            + 28
    }
}

/// Bound the speculative workspace, and reject unsupported framing before
/// allocating or uploading anything. Other members use the streaming scanner.
pub(super) fn eligible(source: &[u8]) -> bool {
    if !(7..=MAX_SOURCE).contains(&source.len()) {
        return false;
    }
    let cmf = source[0];
    cmf & 15 == 8
        && cmf >> 4 <= 7
        && u16::from_be_bytes([cmf, source[1]]) % 31 == 0
        && source[1] & 32 == 0
}

impl Indexer {
    pub(super) fn new(
        stream: Arc<CudaStream>,
        module: &Arc<CudaModule>,
    ) -> Result<Self, GpuMatchError> {
        Ok(Self {
            stream,
            kernels: Kernels::new(module)?,
        })
    }

    #[expect(
        unsafe_code,
        reason = "padded compact source, bounded candidate arrays and validated index kernel ABI"
    )]
    fn candidates(
        &self,
        w: &mut Workspace,
        source: &[u8],
        end: u64,
    ) -> Result<Option<u32>, GpuMatchError> {
        reserve(&self.stream, &mut w.source, source.len() + 16)?;
        self.stream
            .memcpy_htod(source, &mut w.source)
            .map_err(failure)?;
        let capacity =
            u32::try_from(source.len().div_ceil(2).clamp(256, MAX_PRELIMINARY)).map_err(failure)?;
        reserve(&self.stream, &mut w.preliminary, capacity as usize)?;
        self.stream.memset_zeros(&mut w.count).map_err(failure)?;
        let grid = u32::try_from(end.div_ceil(u64::from(WORKGROUP_SIZE))).map_err(failure)?;
        launch!(self,inflate_header_candidates,(grid,1,1),WORKGROUP_SIZE;
            &w.source,&end,&mut w.preliminary,&capacity,&mut w.count);
        let count = self.stream.clone_dtoh(&w.count).map_err(failure)?[0];
        if count > capacity {
            return Ok(None);
        }
        if count == 0 {
            return Ok(Some(0));
        }
        let words = u32::try_from(end.div_ceil(32)).map_err(failure)?;
        let chunks = words.div_ceil(HEADER_CHUNK_WORDS);
        reserve(&self.stream, &mut w.bitmap, words as usize)?;
        reserve(&self.stream, &mut w.offsets, chunks as usize + 1)?;
        self.stream
            .memset_zeros(&mut w.bitmap.slice_mut(..words as usize))
            .map_err(failure)?;
        launch!(self,inflate_validate_bitmap,(count.div_ceil(WORKGROUP_SIZE),1,1),WORKGROUP_SIZE;
            &w.source,&end,&w.preliminary,&count,&mut w.bitmap);
        launch!(self,inflate_count_header_chunks,(chunks,1,1),HEADER_CHUNK_WORDS;
            &w.bitmap,&words,&mut w.offsets);
        launch!(self,inflate_header_offsets,(1,1,1),1;&mut w.offsets,&chunks);
        let valid = self
            .stream
            .clone_dtoh(&w.offsets.slice(chunks as usize..chunks as usize + 1))
            .map_err(failure)?[0];
        if valid as usize > MAX_HEADERS {
            return Ok(None);
        }
        if valid != 0 {
            reserve(&self.stream, &mut w.headers, valid as usize)?;
            launch!(self,inflate_scatter_headers,(chunks,1,1),HEADER_CHUNK_WORDS;
                &w.bitmap,&words,&w.offsets,&mut w.headers);
        }
        Ok(Some(valid))
    }

    #[expect(
        unsafe_code,
        reason = "ordered candidate ranges establish disjoint tile and relative segment capacities"
    )]
    fn extents(
        &self,
        w: &mut Workspace,
        count: u32,
        end: u64,
        window: u32,
        limit: u32,
    ) -> Result<bool, GpuMatchError> {
        if count == 0 {
            return Ok(true);
        }
        let n = count as usize;
        reserve(&self.stream, &mut w.blocks, n * BLOCK_WORDS)?;
        launch!(self,inflate_make_tile_specs,(1,1,1),1;
            &w.headers,&count,&end,&mut w.blocks,&mut w.count);
        let tiles = self.stream.clone_dtoh(&w.count).map_err(failure)?[0];
        let t = tiles as usize;
        if t == 0 || t > MAX_TILES {
            return Ok(false);
        }
        reserve(&self.stream, &mut w.tasks, t)?;
        reserve(&self.stream, &mut w.trees, n * TREE_BYTES)?;
        reserve(&self.stream, &mut w.paths, t * PATH_WORDS)?;
        reserve(&self.stream, &mut w.extents, n * EXTENT_WORDS)?;
        reserve(&self.stream, &mut w.relative, t * WORDS_PER_SEGMENT)?;
        reserve(&self.stream, &mut w.counts, n)?;
        launch!(self,inflate_make_tile_tasks,(count,1,1),WORKGROUP_SIZE;&w.blocks,&mut w.tasks);
        launch!(self,inflate_prepare_tiles,(count.div_ceil(WORKGROUP_SIZE),1,1),WORKGROUP_SIZE;
            &w.source,&w.blocks,&count,&mut w.trees);
        launch!(self,inflate_tile_paths,(tiles,1,1),WORKGROUP_SIZE;
            &w.source,&w.blocks,&w.trees,&w.tasks,&window,&mut w.paths);
        launch!(self,inflate_compose_segments,(count.div_ceil(WORKGROUP_SIZE),1,1),WORKGROUP_SIZE;
            &w.blocks,&w.trees,&w.paths,&count,&limit,&mut w.extents,&mut w.relative,&mut w.counts);
        Ok(true)
    }

    #[expect(
        unsafe_code,
        reason = "the exact chain walker validates header, history, output and descriptor bounds"
    )]
    pub(super) fn prepare(
        &self,
        w: &mut Workspace,
        source: &[u8],
        limit: u32,
    ) -> Result<Option<DeviceIndex>, GpuMatchError> {
        if !eligible(source) {
            return Ok(None);
        }
        let size = source.len() as u64;
        let end = (size - 4) * 8;
        let cmf = source[0];
        let window = 256u32 << (cmf >> 4);
        let Some(count) = self.candidates(w, source, end)? else {
            return Ok(None);
        };
        if !self.extents(w, count, end, window, limit)? {
            return Ok(None);
        }
        reserve(
            &self.stream,
            &mut w.segments,
            MAX_SEGMENTS * WORDS_PER_SEGMENT,
        )?;
        let capacity = u32::try_from(MAX_SEGMENTS).map_err(failure)?;
        launch!(self,inflate_assemble_index,(1,1,1),1;
            &w.source,&size,&w.blocks,&w.extents,&w.relative,&w.counts,&count,&limit,
            &mut w.segments,&capacity,&mut w.info);
        let info = self.stream.clone_dtoh(&w.info).map_err(failure)?;
        if info[0] != 0 {
            return Ok(None);
        }
        let n = info[1] as usize;
        let decoded = info[2];
        if n > MAX_SEGMENTS || decoded > limit || info[3] != window {
            return Err(GpuMatchError("invalid CUDA index metadata".into()));
        }
        let words = if n == 0 {
            Vec::new()
        } else {
            self.stream
                .clone_dtoh(&w.segments.slice(..n * WORDS_PER_SEGMENT))
                .map_err(failure)?
        };
        let segments = validate_segments(&words, decoded, end)?;
        Ok(Some(DeviceIndex {
            segments,
            decoded,
            window,
            adler: info[4],
        }))
    }
}

fn validate_segments(
    words: &[u64],
    decoded: u32,
    end: u64,
) -> Result<Vec<InflateSegment>, GpuMatchError> {
    let mut segments = Vec::with_capacity(words.len() / WORDS_PER_SEGMENT);
    let mut position = 0;
    let mut previous_bit = 16;
    for s in words.chunks_exact(WORDS_PER_SEGMENT) {
        let start = u32::try_from(s[4] & u64::from(u32::MAX)).map_err(failure)?;
        let finish = u32::try_from(s[4] >> 32).map_err(failure)?;
        let kind = u32::try_from(s[5] & u64::from(u32::MAX)).map_err(failure)?;
        let flags = s[5] >> 32;
        if start != position
            || finish <= start
            || finish > decoded
            || (finish - start) as usize > MAX_SEGMENT
            || s[0] < 16
            || s[0] > s[1]
            || s[1] > s[2]
            || s[2] < previous_bit
            || s[2] >= s[3]
            || s[3] > end
            || kind > 2
            || flags > 1
        {
            return Err(GpuMatchError("invalid CUDA entropy segment".into()));
        }
        segments.push(InflateSegment {
            header_bit: s[0],
            header_end_bit: s[1],
            start_bit: s[2],
            end_bit: s[3],
            start,
            end: finish,
            kind,
            ends_block: flags != 0,
        });
        position = finish;
        previous_bit = s[3];
    }
    if position != decoded || words.len() % WORDS_PER_SEGMENT != 0 {
        return Err(GpuMatchError(
            "CUDA entropy index does not cover decoded member".into(),
        ));
    }
    Ok(segments)
}
