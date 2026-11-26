//! Compact-source device index. Every speculative edge is validated by the
//! exact block-chain walk before any segment is exposed to the inflater.
use super::{
    allocation_capacity, failure, reserve, GpuMatchError, MAX_BATCH_SEGMENTS, MAX_SEGMENT,
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
const MAX_TOKENS: usize = 16 * 1024 * 1024;
const HEADER_CHUNK_WORDS: u32 = 128;
const TREE_BYTES: usize = 2392;
const PATH_WORDS: usize = 48 * 2;
const BLOCK_WORDS: usize = 3;
const EXTENT_WORDS: usize = 6;

kernel_set! {
    inflate_header_candidates, inflate_validate_bitmap, inflate_count_header_chunks,
    inflate_header_offsets, inflate_scatter_headers, inflate_make_tile_specs,
    inflate_make_tile_tasks, inflate_prepare_tiles, inflate_tile_paths,
    inflate_compose_segments, inflate_assemble_index, inflate_place_segments, inflate_emit_tokens, inflate_schedule_groups,
}

pub(super) struct Indexer {
    stream: Arc<CudaStream>,
    kernels: Kernels,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct DeviceGroup {
    pub(super) offset: usize,
    pub(super) count: u32,
    pub(super) start: u32,
    pub(super) end: u32,
}

pub(super) struct DeviceIndex {
    pub(super) groups: Vec<DeviceGroup>,
    pub(super) decoded: u32,
    pub(super) window: u32,
    pub(super) adler: u32,
    pub(super) tokens: u32,
}

pub(super) struct Workspace {
    pub(super) source: CudaSlice<u8>,
    pub(super) segments: CudaSlice<u64>,
    pub(super) ranges: CudaSlice<u32>,
    pub(super) tokens: CudaSlice<u32>,
    preliminary: CudaSlice<u64>,
    bitmap: CudaSlice<u32>,
    offsets: CudaSlice<u32>,
    headers: CudaSlice<u64>,
    blocks: CudaSlice<u64>,
    tasks: CudaSlice<u64>,
    trees: CudaSlice<u8>,
    paths: CudaSlice<u64>,
    edges: CudaSlice<u32>,
    extents: CudaSlice<u64>,
    relative: CudaSlice<u64>,
    relative_ranges: CudaSlice<u32>,
    selected: CudaSlice<u32>,
    counts: CudaSlice<u32>,
    placements: CudaSlice<u32>,
    count: CudaSlice<u32>,
    info: CudaSlice<u32>,
    groups: CudaSlice<u32>,
}

impl Workspace {
    pub(super) fn new(stream: &Arc<CudaStream>) -> Result<Self, GpuMatchError> {
        Ok(Self {
            source: stream.alloc_zeros(1).map_err(failure)?,
            segments: stream.alloc_zeros(1).map_err(failure)?,
            ranges: stream.alloc_zeros(1).map_err(failure)?,
            tokens: stream.alloc_zeros(1).map_err(failure)?,
            preliminary: stream.alloc_zeros(1).map_err(failure)?,
            bitmap: stream.alloc_zeros(1).map_err(failure)?,
            offsets: stream.alloc_zeros(1).map_err(failure)?,
            headers: stream.alloc_zeros(1).map_err(failure)?,
            blocks: stream.alloc_zeros(1).map_err(failure)?,
            tasks: stream.alloc_zeros(1).map_err(failure)?,
            trees: stream.alloc_zeros(1).map_err(failure)?,
            paths: stream.alloc_zeros(1).map_err(failure)?,
            edges: stream.alloc_zeros(1).map_err(failure)?,
            extents: stream.alloc_zeros(1).map_err(failure)?,
            relative: stream.alloc_zeros(1).map_err(failure)?,
            relative_ranges: stream.alloc_zeros(1).map_err(failure)?,
            selected: stream.alloc_zeros(1).map_err(failure)?,
            counts: stream.alloc_zeros(1).map_err(failure)?,
            placements: stream.alloc_zeros(1).map_err(failure)?,
            count: stream.alloc_zeros(1).map_err(failure)?,
            info: stream.alloc_zeros(8).map_err(failure)?,
            groups: stream.alloc_zeros(1).map_err(failure)?,
        })
    }

    pub(super) fn upper_bound(source: usize) -> usize {
        if source == 0 {
            return 4096;
        }
        let source = source.min(MAX_SOURCE);
        let preliminary = source.div_ceil(2).clamp(256, MAX_PRELIMINARY);
        let headers = preliminary.min(MAX_HEADERS);
        let tiles = (source * 8)
            .div_ceil(TILE_BITS)
            .saturating_add(headers)
            .min(MAX_TILES);
        let tokens = (source * 8).min(MAX_TOKENS);

        let allocation = |n: usize, width: usize| allocation_capacity(n) * width;
        allocation(source + 16, 1)
            + allocation(MAX_SEGMENTS * WORDS_PER_SEGMENT, 8)
            + allocation(preliminary, 8)
            + allocation(source.div_ceil(4), 4)
            + allocation(source.div_ceil(4 * HEADER_CHUNK_WORDS as usize) + 1, 4)
            + allocation(headers, 8)
            + allocation(headers * BLOCK_WORDS, 8)
            + allocation(tiles, 8)
            + allocation(headers * TREE_BYTES, 1)
            + allocation(tiles * PATH_WORDS, 8)
            + allocation(tiles * TILE_BITS, 4)
            + allocation(headers * EXTENT_WORDS, 8)
            + allocation(tiles * WORDS_PER_SEGMENT, 8)
            + allocation(headers, 4)
            + allocation(headers * 4, 4)
            + allocation(MAX_SEGMENTS * 2, 4)
            + allocation(tiles * 2, 4)
            + allocation(tiles * 4, 4)
            + allocation(tokens, 4)
            + allocation(MAX_SEGMENTS * 4, 4)
            + 36
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
            .clone_dtoh(&w.offsets.slice(chunks as usize..=chunks as usize))
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
    ) -> Result<Option<u32>, GpuMatchError> {
        if count == 0 {
            return Ok(Some(0));
        }
        let n = count as usize;
        reserve(&self.stream, &mut w.blocks, n * BLOCK_WORDS)?;
        launch!(self,inflate_make_tile_specs,(1,1,1),1;
            &w.headers,&count,&end,&mut w.blocks,&mut w.count);
        let tiles = self.stream.clone_dtoh(&w.count).map_err(failure)?[0];
        let t = tiles as usize;
        if t == 0 || t > MAX_TILES {
            return Ok(None);
        }
        reserve(&self.stream, &mut w.tasks, t)?;
        reserve(&self.stream, &mut w.trees, n * TREE_BYTES)?;
        reserve(&self.stream, &mut w.paths, t * PATH_WORDS)?;
        reserve(&self.stream, &mut w.edges, t * TILE_BITS)?;
        reserve(&self.stream, &mut w.extents, n * EXTENT_WORDS)?;
        reserve(&self.stream, &mut w.relative, t * WORDS_PER_SEGMENT)?;
        reserve(&self.stream, &mut w.counts, n)?;
        reserve(&self.stream, &mut w.relative_ranges, t * 2)?;
        reserve(&self.stream, &mut w.selected, t * 4)?;
        self.stream
            .memset_zeros(&mut w.selected.slice_mut(..t * 4))
            .map_err(failure)?;

        launch!(self,inflate_make_tile_tasks,(count,1,1),WORKGROUP_SIZE;&w.blocks,&mut w.tasks);
        launch!(self,inflate_prepare_tiles,(count.div_ceil(WORKGROUP_SIZE),1,1),WORKGROUP_SIZE;
            &w.source,&w.blocks,&count,&mut w.trees);
        launch!(self,inflate_tile_paths,(tiles,1,1),WORKGROUP_SIZE;
            &w.source,&w.blocks,&w.trees,&w.tasks,&window,&mut w.paths,&mut w.edges);
        launch!(self,inflate_compose_segments,(count.div_ceil(WORKGROUP_SIZE),1,1),WORKGROUP_SIZE;
            &w.blocks,&w.trees,&w.paths,&count,&limit,&mut w.extents,&mut w.relative,&mut w.counts,&mut w.relative_ranges,&mut w.selected);
        Ok(Some(tiles))
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
        compression_window: u32,
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
        let Some(tiles) = self.extents(w, count, end, window, limit)? else {
            return Ok(None);
        };
        reserve(
            &self.stream,
            &mut w.segments,
            MAX_SEGMENTS * WORDS_PER_SEGMENT,
        )?;
        let capacity = u32::try_from(MAX_SEGMENTS).map_err(failure)?;
        let token_capacity = u32::try_from((source.len() * 8).min(MAX_TOKENS)).map_err(failure)?;
        reserve(&self.stream, &mut w.ranges, MAX_SEGMENTS * 2)?;
        self.stream.memset_zeros(&mut w.ranges).map_err(failure)?;
        if count != 0 {
            reserve(&self.stream, &mut w.tokens, token_capacity as usize)?;
            reserve(&self.stream, &mut w.placements, count as usize * 4)?;
            self.stream
                .memset_zeros(&mut w.placements.slice_mut(..count as usize * 4))
                .map_err(failure)?;
        }
        launch!(self,inflate_assemble_index,(1,1,1),1;
            &w.source,&size,&w.blocks,&w.extents,&w.counts,&count,&limit,
            &mut w.segments,&capacity,&mut w.placements,&mut w.info);
        if count != 0 {
            launch!(self,inflate_place_segments,(count,1,1),WORKGROUP_SIZE;
                &w.blocks,&w.relative,&w.placements,&mut w.segments,&w.info,&w.relative_ranges,&mut w.ranges);
            launch!(self,inflate_emit_tokens,(tiles.div_ceil(WORKGROUP_SIZE),1,1),WORKGROUP_SIZE;
                &w.edges,&w.tasks,&w.selected,&w.placements,
                &tiles,&token_capacity,&mut w.tokens,&mut w.info,&window);
        }
        reserve(&self.stream, &mut w.groups, MAX_SEGMENTS * 4)?;
        let max_group = u32::try_from(MAX_BATCH_SEGMENTS).map_err(failure)?;
        launch!(self,inflate_schedule_groups,(1,1,1),1;
            &w.segments,&mut w.info,&compression_window,&max_group,&mut w.groups);
        let info = self.stream.clone_dtoh(&w.info).map_err(failure)?;
        if info[0] != 0 {
            return Ok(None);
        }
        let n = info[1] as usize;
        let decoded = info[2];
        let group_count = info[5] as usize;
        if n > MAX_SEGMENTS || decoded > limit || info[3] != window || group_count > n {
            return Err(GpuMatchError("invalid CUDA index metadata".into()));
        }
        let words = if group_count == 0 {
            Vec::new()
        } else {
            self.stream
                .clone_dtoh(&w.groups.slice(..group_count * 4))
                .map_err(failure)?
        };
        let groups = validate_groups(&words, n, decoded, compression_window)?;
        Ok(Some(DeviceIndex {
            groups,
            decoded,
            window,
            adler: info[4],
            tokens: if info[6] <= token_capacity {
                info[6]
            } else {
                0
            },
        }))
    }
}

fn validate_groups(
    words: &[u32],
    segments: usize,
    decoded: u32,
    window: u32,
) -> Result<Vec<DeviceGroup>, GpuMatchError> {
    let mut groups = Vec::with_capacity(words.len() / 4);
    let mut position = 0;
    let mut offset = 0;
    let (records, remainder) = words.as_chunks::<4>();
    for g in records {
        let count = g[1];
        let target = (u64::from(position) / u64::from(window) + 1) * u64::from(window);
        if g[0] as usize != offset
            || count == 0
            || count as usize > MAX_BATCH_SEGMENTS
            || g[2] != position
            || g[3] <= position
            || g[3] > decoded
            || u64::from(g[3]) > target + MAX_SEGMENT as u64
        {
            return Err(GpuMatchError("invalid CUDA decode group".into()));
        }
        groups.push(DeviceGroup {
            offset,
            count,
            start: g[2],
            end: g[3],
        });
        offset = offset
            .checked_add(count as usize)
            .ok_or_else(|| GpuMatchError("CUDA group offset overflow".into()))?;
        position = g[3];
    }
    if offset != segments || position != decoded || !remainder.is_empty() {
        return Err(GpuMatchError(
            "CUDA decode groups do not cover member".into(),
        ));
    }
    Ok(groups)
}
