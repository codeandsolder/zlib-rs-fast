#![deny(unsafe_code)]

#[cfg(feature = "cuda")]
mod cuda;

use std::{
    fmt,
    num::NonZeroU64,
    ops::Range,
    sync::{mpsc, Mutex},
};

use bytemuck::cast_slice;
use wgpu::{
    Adapter, Backends, BindGroupDescriptor, BindGroupEntry, BindingResource, Buffer, BufferBinding,
    BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, Device, DeviceDescriptor, DeviceType, ExperimentalFeatures,
    Features, Instance, InstanceDescriptor, Limits, MemoryHints, PipelineCompilationOptions, Queue,
    ShaderModuleDescriptor, Trace,
};

const HASH_SIZE: usize = 1 << 16;
const NO_PREDECESSOR: u32 = u32::MAX;
const PACKED_DISTANCE_LIMIT: u32 = u16::MAX as u32 + 1;
const WORKGROUP_SIZE: u32 = 128;
const MAX_MATCH: u32 = 258;
const COOPERATIVE_SPAN: u32 = (WORKGROUP_SIZE + MAX_MATCH - 1).next_multiple_of(4);
// The cooperative verifier assigns one word of the overlapping span per lane.
const _: () = assert!(COOPERATIVE_SPAN <= WORKGROUP_SIZE * 4);
const DEFAULT_MAX_DIST: u32 = 32_768 - (258 + 3 + 1);

const SHADER: &str = r"
struct Params {
    input_len: u32,
    max_dist: u32,
    repeat_groups: u32,
    search_budget: u32,
}

@group(0) @binding(0) var<storage, read> input_words: array<u32>;
@group(0) @binding(1) var<storage, read> predecessors: array<u32>;
@group(0) @binding(2) var<storage, read_write> results: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

fn load_byte(index: u32) -> u32 {
    if aligned_input { return input_words[index] & 0xffu; }
    let word = input_words[index >> 2u];
    return (word >> ((index & 3u) * 8u)) & 0xffu;
}

fn load_u32(index: u32) -> u32 {
    if aligned_input { return input_words[index]; }
    let word_index = index >> 2u;
    let shift = (index & 3u) * 8u;
    let low = input_words[word_index] >> shift;
    if shift == 0u {
        return low;
    }
    return low | (input_words[word_index + 1u] << (32u - shift));
}

// Each 32-bit word carries two unsigned 16-bit backward distances.
// Zero terminates a chain; a nonzero distance can never denote the current byte.
fn predecessor_distance(position: u32) -> u32 {
    return (predecessors[position >> 1u] >> ((position & 1u) * 16u)) & 0xffffu;
}

// Return as soon as an unequal word identifies the first differing byte.
// No subsequent smaller-width loop needs to repeat that comparison.
fn common_length(position: u32, candidate: u32, max_len: u32, probe: u32) -> u32 {
    var length = 0u;
    if max_len >= 4u {
        let difference = probe ^ load_u32(candidate);
        if difference != 0u { return countTrailingZeros(difference) >> 3u; }
        length = 4u;
    }
    loop {
        if max_len - length < 16u { break; }
        var difference = load_u32(position + length) ^ load_u32(candidate + length);
        if difference != 0u { return length + (countTrailingZeros(difference) >> 3u); }
        difference = load_u32(position + length + 4u) ^ load_u32(candidate + length + 4u);
        if difference != 0u { return length + 4u + (countTrailingZeros(difference) >> 3u); }
        difference = load_u32(position + length + 8u) ^ load_u32(candidate + length + 8u);
        if difference != 0u { return length + 8u + (countTrailingZeros(difference) >> 3u); }
        difference = load_u32(position + length + 12u) ^ load_u32(candidate + length + 12u);
        if difference != 0u { return length + 12u + (countTrailingZeros(difference) >> 3u); }
        length += 16u;
    }
    loop {
        if max_len - length < 4u { break; }
        let difference = load_u32(position + length) ^ load_u32(candidate + length);
        if difference != 0u { return length + (countTrailingZeros(difference) >> 3u); }
        length += 4u;
    }
    loop {
        if length >= max_len || load_byte(position + length) != load_byte(candidate + length) { break; }
        length += 1u;
    }
    return length;
}

// A longer match must contain every triple in the known matching prefix.
// Intersect three chain bounds by choosing the farthest safe predecessor.
fn shifted_offset(candidate: u32, length: u32) -> u32 {
    let tail = length - 3u;
    let middle = tail / 2u;
    var offset = 0u;
    var distance = predecessor_distance(candidate);
    if distance == 0u { return 0u; }
    let tail_distance = predecessor_distance(candidate + tail);
    if tail_distance == 0u { return tail; }
    if tail_distance > distance {
        offset = tail;
        distance = tail_distance;
    }
    let middle_distance = predecessor_distance(candidate + middle);
    if middle_distance == 0u || middle_distance > distance { return middle; }
    return offset;
}

// The entire group shares one nearest-distance candidate. Collectively
// compare the union of the 128 overlapping maximum-length matches.
// This is an exact shortcut: lanes with different predecessors still search.
var<workgroup> repeat_distance: u32;
var<workgroup> repeat_equal: atomic<u32>;

fn cooperative_distance(base: u32, lane: u32) -> u32 {
    // 128 starts with 258 bytes of lookahead span 385 bytes; round
    // to complete words so each participating lane performs one load.
    let span = cooperative_span;
    if params.repeat_groups == 0u || params.input_len - base < span {
        return 0u;
    }
    if lane == 0u {
        let first = predecessor_distance(base);
        let last = predecessor_distance(base + group_size - 1u);
        var selected = 0u;
        if first != 0u && first == last && first <= base && first < params.max_dist {
            selected = first;
        }
        repeat_distance = selected;
        atomicStore(&repeat_equal, 1u);
    }
    let distance = workgroupUniformLoad(&repeat_distance);
    if distance == 0u { return 0u; }
    if lane < span / 4u {
        let offset = base + lane * 4u;
        if load_u32(offset) != load_u32(offset - distance) {
            atomicStore(&repeat_equal, 0u);
        }
    }
    workgroupBarrier();
    if atomicLoad(&repeat_equal) == 0u { return 0u; }
    return distance;
}

@compute @workgroup_size(group_size)
fn search(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    // Flatten rows of workgroups. Guard padded groups before multiplying so
    // position arithmetic is valid even at the u32 input-length boundary.
    let group_index = group.y * grid.x + group.x;
    if group_index > (params.input_len - 1u) / group_size { return; }
    let base = group_index * group_size;
    let shared_distance = cooperative_distance(base, lane);
    let position = base + lane;
    if position >= params.input_len { return; }
    if shared_distance != 0u && predecessor_distance(position) == shared_distance {
        results[position] = (shared_distance << 16u) | max_match;
        return;
    }
    if params.input_len - position < 3u {
        results[position] = 0u;
        return;
    }

    var minimum_start = 0u;
    if position >= params.max_dist {
        // zlib's matcher uses cur_match > strstart - max_dist, not >=.
        minimum_start = position - params.max_dist + 1u;
    }

    var cursor = position;
    var chain_offset = 0u;
    var distance = predecessor_distance(cursor);
    let max_len = min(max_match, params.input_len - position);
    var probe = 0u;
    if max_len >= 4u { probe = load_u32(position); }

    var best_len = 0u;
    var best_distance = 0u;
    var best_probe = probe;
    var visited = 0u;

    // Walk newest to oldest, shifting to a more selective interior triple
    // when a longer prefix proves that the skipped candidates cannot improve it.
    loop {
        if best_len == max_len || distance == 0u || visited >= params.search_budget {
            break;
        }
        if distance > cursor { break; }
        let candidate = cursor - distance;
        if candidate < minimum_start { break; }
        cursor = candidate;
        distance = predecessor_distance(cursor + chain_offset);
        visited += 1u;

        if best_len >= 3u {
            if aligned_input {
                if best_probe != load_u32(candidate + best_len - 3u) { continue; }
            } else if load_byte(position + best_len) != load_byte(candidate + best_len) {
                continue;
            }
        }

        let length = common_length(position, candidate, max_len, probe);

        // Equal lengths keep the nearest match because predecessors are descending.
        if length >= 3u && length > best_len {
            best_len = length;
            best_distance = position - candidate;
            if best_len < max_len {
                if best_len > 3u {
                    chain_offset = shifted_offset(candidate, best_len);
                    distance = predecessor_distance(candidate + chain_offset);
                }
                // The triple spanning the first mismatching byte must also
                // occur in any longer match. Its query-side predecessor is
                // another safe bound, but only when older than the candidate.
                if distance != 0u {
                    let extension_offset = best_len - 2u;
                    let extension_distance = predecessor_distance(position + extension_offset);
                    if extension_distance == 0u {
                        distance = 0u;
                    } else if extension_distance > best_distance + distance {
                        chain_offset = extension_offset;
                        distance = extension_distance - best_distance;
                    }
                }
            }
            // A three-byte best match uses the existing first-word probe.
            // Complete matches never read a following byte.
            if aligned_input && best_len > 3u && best_len < max_len {
                best_probe = load_u32(position + best_len - 3u);
            }
        }
    }

    results[position] = (best_distance << 16u) | best_len;
}
";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuMatch {
    pub length: u32,
    pub start: u32,
}

#[derive(Debug)]
pub struct GpuMatchError(String);

impl fmt::Display for GpuMatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for GpuMatchError {}

/// Match table with one 32-bit word per input byte: length in the low
/// half-word and backward distance in the high half-word.
#[derive(Debug, Default)]
pub struct GpuMatchTable {
    entries: Vec<u32>,
}

impl GpuMatchTable {
    #[inline]
    #[must_use = "The option reports the match at the requested position."]
    pub fn get(&self, absolute_position: usize) -> Option<GpuMatch> {
        let position = u32::try_from(absolute_position).ok()?;
        self.entries
            .get(absolute_position)
            .map(|&entry| decode_match(position, entry))
    }

    /// Return the wire-domain length and backward distance directly.
    /// A zero word means no match (both components are zero).
    #[inline]
    pub(super) fn length_distance(&self, position: usize) -> Option<(u16, u16)> {
        let [a, b, c, d] = self.entries.get(position)?.to_le_bytes();
        Some((u16::from_le_bytes([a, b]), u16::from_le_bytes([c, d])))
    }

    #[must_use = "The count describes the number of matches."]
    pub const fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use = "The boolean reports whether the collection is empty."]
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[must_use = "Iterators are lazy and must be consumed to visit matches."]
    pub fn iter(&self) -> impl ExactSizeIterator<Item = GpuMatch> + '_ {
        (0..u32::MAX)
            .zip(self.entries.iter().copied())
            .map(|(position, entry)| decode_match(position, entry))
    }
}

#[inline]
const fn decode_match(position: u32, entry: u32) -> GpuMatch {
    if entry == 0 {
        GpuMatch {
            length: 0,
            start: 0,
        }
    } else {
        GpuMatch {
            length: entry & 0xffff,
            start: position - (entry >> 16),
        }
    }
}

struct ReusableBuffer {
    buffer: Buffer,
    capacity: u64,
    usage: BufferUsages,
    label: &'static str,
}

impl ReusableBuffer {
    fn new(device: &Device, label: &'static str, usage: BufferUsages) -> Self {
        let capacity = 4;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: capacity,
            usage,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            capacity,
            usage,
            label,
        }
    }

    fn ensure_capacity(
        &mut self,
        device: &Device,
        required: u64,
        max_buffer_size: u64,
    ) -> Result<(), GpuMatchError> {
        if required <= self.capacity {
            return Ok(());
        }
        if required > max_buffer_size {
            return Err(GpuMatchError(format!(
                "GPU {} buffer requires {required} bytes, device limit is {max_buffer_size}",
                self.label
            )));
        }
        let grown = required
            .checked_next_power_of_two()
            .unwrap_or(required)
            .min(max_buffer_size);
        let capacity = grown.max(required);
        self.buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(self.label),
            size: capacity,
            usage: self.usage,
            mapped_at_creation: false,
        });
        self.capacity = capacity;
        Ok(())
    }
}

struct GpuWorkspace {
    input: ReusableBuffer,
    predecessors: ReusableBuffer,
    results: ReusableBuffer,
    params: ReusableBuffer,
    download: ReusableBuffer,
}

impl GpuWorkspace {
    fn new(device: &Device) -> Self {
        Self {
            input: ReusableBuffer::new(
                device,
                "match input workspace",
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
            ),
            predecessors: ReusableBuffer::new(
                device,
                "match predecessor workspace",
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
            ),
            results: ReusableBuffer::new(
                device,
                "match result workspace",
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ),
            params: ReusableBuffer::new(
                device,
                "match parameter workspace",
                BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            ),
            download: ReusableBuffer::new(
                device,
                "match download workspace",
                BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            ),
        }
    }

    fn ensure_batch(
        &mut self,
        device: &Device,
        layout: &BatchLayout,
        max_buffer_size: u64,
    ) -> Result<(), GpuMatchError> {
        for (buffer, size) in [
            (&mut self.input, layout.input),
            (&mut self.predecessors, layout.predecessors),
            (&mut self.results, layout.results),
            (&mut self.params, layout.params),
            (&mut self.download, layout.download),
        ] {
            buffer.ensure_capacity(device, size, max_buffer_size)?;
        }
        Ok(())
    }

    fn upload(&self, queue: &Queue, dispatches: &[GpuDispatch]) {
        for dispatch in dispatches {
            queue.write_buffer(
                &self.input.buffer,
                dispatch.input_offset,
                cast_slice(&dispatch.input.input_words),
            );
            queue.write_buffer(
                &self.predecessors.buffer,
                dispatch.predecessors_offset,
                cast_slice(&dispatch.input.predecessors),
            );
            queue.write_buffer(
                &self.params.buffer,
                dispatch.params_offset,
                cast_slice(&dispatch.input.params),
            );
        }
    }

    fn readback(&self, device: &Device, batch: &mut GpuBatch) -> Result<(), GpuMatchError> {
        let download = &self.download.buffer;
        let slice = download.slice(0..batch.layout.download);
        let (sender, receiver) = mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| GpuMatchError(format!("GPU poll failed: {error}")))?;
        receiver
            .recv()
            .map_err(|_| GpuMatchError("GPU download callback was dropped".into()))?
            .map_err(|error| GpuMatchError(format!("GPU result download failed: {error}")))?;
        let mapped = slice.get_mapped_range();
        for dispatch in &batch.dispatches {
            let entries = mapped[dispatch.download.clone()]
                .as_chunks::<4>()
                .0
                .iter()
                .copied()
                .map(u32::from_le_bytes)
                .collect();
            batch.tables[dispatch.input.input_index] = GpuMatchTable { entries };
        }
        drop(mapped);
        download.unmap();
        Ok(())
    }
}

type PrepareInputs<'a> =
    dyn Fn(&[&[u8]], u32, u32, u32) -> Result<Vec<PreparedSearch>, GpuMatchError> + 'a;

const MAX_LOCAL_INPUT_BYTES: usize = 8192;

enum PreparedSearch {
    Local(GpuMatchTable),
    Gpu(PreparedGpuInput),
}

fn prepare_input(
    input_index: usize,
    input: &[u8],
    search_budget: u32,
    max_dist: u32,
    max_workgroups: u32,
    forced_layout: Option<InputLayout>,
) -> Result<PreparedSearch, GpuMatchError> {
    prepare_indexed_input(
        input_index,
        input,
        search_budget,
        max_dist,
        max_workgroups,
        forced_layout,
        build_predecessors,
    )
}

fn prepare_indexed_input(
    input_index: usize,
    input: &[u8],
    search_budget: u32,
    max_dist: u32,
    max_workgroups: u32,
    forced_layout: Option<InputLayout>,
    build_index: impl FnOnce(&[u8], u32) -> Result<Vec<u32>, GpuMatchError>,
) -> Result<PreparedSearch, GpuMatchError> {
    let input_len = u32::try_from(input.len())
        .map_err(|_| GpuMatchError("GPU match search input exceeds u32::MAX bytes".into()))?;
    // No match can start before position one and still have three bytes
    // available in an input shorter than four bytes.
    if input.len() < 4 || max_dist == 1 {
        return Ok(PreparedSearch::Local(GpuMatchTable {
            entries: vec![0; input.len()],
        }));
    }

    let predecessors = build_index(input, max_dist)?;
    // A zero chain contains no eligible candidate at any position.
    // Such an input needs neither a GPU dispatch nor a packed input copy.
    if predecessors.iter().all(|&word| word == 0) {
        return Ok(PreparedSearch::Local(GpuMatchTable {
            entries: vec![0; input.len()],
        }));
    }
    let input_layout = forced_layout
        .unwrap_or_else(|| InputLayout::select(input, &predecessors, search_budget, max_dist));
    if forced_layout.is_none()
        && input_layout == InputLayout::Packed
        && input.len() <= MAX_LOCAL_INPUT_BYTES
    {
        if let Some(matches) = resolve_bounded(input, &predecessors, search_budget, max_dist) {
            return Ok(PreparedSearch::Local(matches));
        }
    }
    let dispatch = dispatch_grid(input_len, max_workgroups)?;
    let repeat_groups = u32::from(cooperative_groups_likely(input, &predecessors));
    let input_words = input_layout.pack(input);
    Ok(PreparedSearch::Gpu(PreparedGpuInput {
        input_index,
        input_len,
        dispatch,
        input_layout,
        input_words,
        predecessors,
        params: [input_len, max_dist, repeat_groups, search_budget],
    }))
}

fn prepare_serial(
    inputs: &[&[u8]],
    search_budget: u32,
    max_dist: u32,
    max_workgroups: u32,
) -> Result<Vec<PreparedSearch>, GpuMatchError> {
    inputs
        .iter()
        .enumerate()
        .map(|(index, input)| {
            prepare_input(index, input, search_budget, max_dist, max_workgroups, None)
        })
        .collect()
}

#[cfg(feature = "parallel")]
fn prepare_parallel(
    inputs: &[&[u8]],
    search_budget: u32,
    max_dist: u32,
    max_workgroups: u32,
    pool: &crate::BatchCodec,
) -> Result<Vec<PreparedSearch>, GpuMatchError> {
    let mut inputs = inputs.to_vec();
    pool.run_items(
        &mut inputs,
        || Ok(PreparedSearch::Local(GpuMatchTable::default())),
        |index, input| {
            prepare_indexed_input(
                index,
                input,
                search_budget,
                max_dist,
                max_workgroups,
                None,
                |bytes, distance| build_predecessors_parallel(bytes, distance, pool),
            )
        },
    )
    .into_iter()
    .collect()
}

struct PreparedGpuInput {
    input_index: usize,
    input_len: u32,
    dispatch: (u32, u32),
    input_layout: InputLayout,
    input_words: Vec<u32>,
    predecessors: Vec<u32>,
    params: [u32; 4],
}

impl PreparedGpuInput {
    const fn input_size(&self) -> u64 {
        (self.input_words.len() * size_of::<u32>()) as u64
    }

    const fn predecessors_size(&self) -> u64 {
        size_of_val(self.predecessors.as_slice()) as u64
    }

    fn result_size(&self) -> u64 {
        u64::from(self.input_len) * size_of::<u32>() as u64
    }
}

/// Exact buffer extents of one submission, established before locking
/// the reusable GPU workspace. Storage and uniform offsets are independent
/// because their device alignments can differ.
#[derive(Default)]
struct BatchLayout {
    input: u64,
    predecessors: u64,
    results: u64,
    params: u64,
    download: u64,
}

struct GpuDispatch {
    input: PreparedGpuInput,
    input_offset: u64,
    predecessors_offset: u64,
    result_offset: u64,
    params_offset: u64,
    download: Range<usize>,
}

impl BatchLayout {
    fn place(
        &mut self,
        input: PreparedGpuInput,
        limits: &Limits,
    ) -> Result<GpuDispatch, GpuMatchError> {
        let storage_alignment = u64::from(limits.min_storage_buffer_offset_alignment.max(1));
        let uniform_alignment = u64::from(limits.min_uniform_buffer_offset_alignment.max(1));
        for (name, size) in [
            ("input", input.input_size()),
            ("predecessors", input.predecessors_size()),
            ("results", input.result_size()),
        ] {
            if size > limits.max_storage_buffer_binding_size {
                return Err(GpuMatchError(format!(
                    "GPU {name} binding for input {} requires {size} bytes, device limit is {}",
                    input.input_index, limits.max_storage_buffer_binding_size
                )));
            }
        }
        let params_size = size_of_val(&input.params) as u64;
        if params_size > limits.max_uniform_buffer_binding_size {
            return Err(GpuMatchError(format!(
                "GPU parameter binding requires {params_size} bytes, device limit is {}",
                limits.max_uniform_buffer_binding_size
            )));
        }
        let input_offset =
            allocate_aligned(&mut self.input, input.input_size(), storage_alignment)?;
        let predecessors_offset = allocate_aligned(
            &mut self.predecessors,
            input.predecessors_size(),
            storage_alignment,
        )?;
        let result_offset =
            allocate_aligned(&mut self.results, input.result_size(), storage_alignment)?;
        let params_offset = allocate_aligned(&mut self.params, params_size, uniform_alignment)?;
        let start = self.download;
        self.download = start
            .checked_add(input.result_size())
            .ok_or_else(|| GpuMatchError("GPU download buffer layout overflowed".into()))?;
        // Reject host-address-space overflow before mapping a GPU buffer.
        let host_index = |offset| {
            usize::try_from(offset)
                .map_err(|_| GpuMatchError("GPU readback exceeds host address space".into()))
        };
        let download = host_index(start)?..host_index(self.download)?;
        Ok(GpuDispatch {
            input,
            input_offset,
            predecessors_offset,
            result_offset,
            params_offset,
            download,
        })
    }
}

/// Locally resolved tables retain their input slots; only placed GPU
/// dispatches enter the workspace. Each dispatch already has a checked
/// host readback range before command encoding starts.
struct GpuBatch {
    layout: BatchLayout,
    dispatches: Vec<GpuDispatch>,
    tables: Vec<GpuMatchTable>,
}

impl GpuBatch {
    fn plan(prepared: Vec<PreparedSearch>, limits: &Limits) -> Result<Self, GpuMatchError> {
        let mut layout = BatchLayout::default();
        let mut tables = Vec::with_capacity(prepared.len());
        let mut dispatches = Vec::new();
        for (index, item) in prepared.into_iter().enumerate() {
            match item {
                PreparedSearch::Local(table) => tables.push(table),
                PreparedSearch::Gpu(input) => {
                    debug_assert_eq!(input.input_index, index);
                    dispatches.push(layout.place(input, limits)?);
                    tables.push(GpuMatchTable::default());
                }
            }
        }
        // Validate total buffer extents as well as individual bindings
        // before acquiring the shared workspace or issuing GPU commands.
        if [
            layout.input,
            layout.predecessors,
            layout.results,
            layout.params,
            layout.download,
        ]
        .into_iter()
        .any(|size| size > limits.max_buffer_size)
        {
            return Err(GpuMatchError(
                "GPU batch layout exceeds device buffer size".into(),
            ));
        }
        Ok(Self {
            layout,
            dispatches,
            tables,
        })
    }
}

fn allocate_aligned(cursor: &mut u64, size: u64, alignment: u64) -> Result<u64, GpuMatchError> {
    let remainder = *cursor % alignment;
    let padding = if remainder == 0 {
        0
    } else {
        alignment - remainder
    };
    let offset = cursor
        .checked_add(padding)
        .ok_or_else(|| GpuMatchError("GPU batch buffer layout overflowed".into()))?;
    *cursor = offset
        .checked_add(size)
        .ok_or_else(|| GpuMatchError("GPU batch buffer layout overflowed".into()))?;
    Ok(offset)
}

/// Construct a two-dimensional grid within the device's per-axis limit.
/// Every nonempty input byte maps to exactly one invocation; the shader
/// discards the padded final row and the padded final workgroup.
fn dispatch_grid(input_len: u32, max_dimension: u32) -> Result<(u32, u32), GpuMatchError> {
    let groups = input_len.div_ceil(WORKGROUP_SIZE);
    let columns = groups.min(max_dimension);
    if columns == 0 {
        return Err(GpuMatchError(
            "GPU dispatch requires a nonempty input and grid".into(),
        ));
    }
    let rows = groups.div_ceil(columns);
    if rows > max_dimension {
        return Err(GpuMatchError(format!(
            "GPU match dispatch requires {groups} workgroups, device grid is limited to {max_dimension} per axis"
        )));
    }
    Ok((columns, rows))
}

/// A conservative bound for each individual storage buffer and for the
/// aligned contents of any buffer in one submission. Match records (four
/// bytes per position) dominate input words, packed predecessors and readback;
/// uniforms can be larger only for very short inputs.
#[derive(Clone, Copy)]
struct SearchGeometry {
    max_part_len: usize,
    max_group_bytes: u64,
    alignment: u64,
}

impl SearchGeometry {
    fn from_limits(limits: &Limits) -> Result<Self, GpuMatchError> {
        let params_bytes = size_of::<[u32; 4]>() as u64;
        let axis = u64::from(limits.max_compute_workgroups_per_dimension);
        let grid_positions = axis
            .saturating_mul(axis)
            .saturating_mul(u64::from(WORKGROUP_SIZE));
        let binding_positions = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size)
            / size_of::<u32>() as u64;
        let max_part_len = usize::try_from(
            grid_positions
                .min(binding_positions)
                .min(u64::from(u32::MAX)),
        )
        .unwrap_or(usize::MAX);
        if max_part_len == 0
            || limits.max_buffer_size < params_bytes
            || limits.max_uniform_buffer_binding_size < params_bytes
        {
            return Err(GpuMatchError(
                "GPU limits cannot represent a match-search dispatch".into(),
            ));
        }
        Ok(Self {
            max_part_len,
            max_group_bytes: limits.max_buffer_size,
            alignment: u64::from(
                limits
                    .min_storage_buffer_offset_alignment
                    .max(limits.min_uniform_buffer_offset_alignment)
                    .max(1),
            ),
        })
    }

    fn append(self, used: u64, input_len: usize) -> Option<u64> {
        if input_len == 0 {
            return Some(used);
        }
        if input_len > self.max_part_len {
            return None;
        }
        let size = (u64::try_from(input_len).ok()? * size_of::<u32>() as u64)
            .max(size_of::<[u32; 4]>() as u64);
        let mut next = used;
        allocate_aligned(&mut next, size, self.alignment).ok()?;
        (next <= self.max_group_bytes).then_some(next)
    }

    fn core_capacity(self, input_len: usize, max_dist: u32) -> Result<usize, GpuMatchError> {
        if input_len <= self.max_part_len {
            return Ok(input_len.max(1));
        }
        self.max_part_len
            .checked_sub((max_dist - 1 + MAX_MATCH - 1) as usize)
            .filter(|&length| length != 0)
            .ok_or_else(|| {
                GpuMatchError(
                    "GPU binding cannot hold one position and its full DEFLATE history/lookahead"
                        .into(),
                )
            })
    }
}

struct SearchPart<'a> {
    input_index: usize,
    input: &'a [u8],
    keep: Range<usize>,
}

/// Persistent hardware match-search engine. Host predecessor preparation,
/// table representation and partition geometry are shared across backends.
pub struct GpuMatchSearcher {
    backend: GpuBackend,
    limits: Limits,
    adapter_name: String,
}

enum GpuBackend {
    Vulkan(VulkanBackend),
    #[cfg(feature = "cuda")]
    Cuda(cuda::CudaBackend),
}

struct VulkanBackend {
    device: Device,
    queue: Queue,
    packed_pipeline: ComputePipeline,
    aligned_pipeline: ComputePipeline,
    workspace: Mutex<GpuWorkspace>,
}

impl GpuMatchSearcher {
    /// Create a Vulkan search engine, or use `cuda:N` to explicitly select
    /// CUDA device ordinal N when compiled with the cuda feature.
    ///
    /// # Errors
    /// Returns an error for invalid device selection or initialization failure.
    pub fn new(adapter_filter: Option<&str>) -> Result<Self, GpuMatchError> {
        if let Some(ordinal) = adapter_filter.and_then(|filter| filter.strip_prefix("cuda:")) {
            #[cfg(feature = "cuda")]
            {
                let ordinal = ordinal.parse::<usize>().map_err(|error| {
                    GpuMatchError(format!("invalid CUDA device ordinal: {error}"))
                })?;
                let (backend, adapter_name) = cuda::CudaBackend::new(ordinal)?;
                return Ok(Self {
                    backend: GpuBackend::Cuda(backend),
                    limits: Limits::default(),
                    adapter_name,
                });
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = ordinal;
                return Err(GpuMatchError("CUDA requires the cuda feature".into()));
            }
        }
        let (backend, adapter_name) = VulkanBackend::new(adapter_filter)?;
        let limits = backend.device.limits();
        Ok(Self {
            backend: GpuBackend::Vulkan(backend),
            limits,
            adapter_name,
        })
    }

    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// Search a complete input using the default DEFLATE distance.
    ///
    /// # Errors
    /// Returns an error for invalid search geometry or budget, or if
    /// GPU preparation, submission, or readback fails.
    pub fn search(&self, input: &[u8], search_budget: u32) -> Result<GpuMatchTable, GpuMatchError> {
        self.search_with_max_dist(input, search_budget, DEFAULT_MAX_DIST)
    }

    /// Search several independent complete inputs, returning tables in input order.
    ///
    /// All inputs use the same search budget and DEFLATE default maximum distance.
    /// Dispatches share a submission when they fit the device buffers; larger workloads
    /// are partitioned while retaining exact history and lookahead at each boundary.
    ///
    /// # Errors
    /// Returns an error for invalid search geometry or budget, or if
    /// preparation, GPU submission, or readback fails.
    pub fn search_batch(
        &self,
        inputs: &[&[u8]],
        search_budget: u32,
    ) -> Result<Vec<GpuMatchTable>, GpuMatchError> {
        self.search_batch_with_max_dist(inputs, search_budget, DEFAULT_MAX_DIST)
    }

    pub(crate) fn search_with_max_dist(
        &self,
        input: &[u8],
        search_budget: u32,
        max_dist: u32,
    ) -> Result<GpuMatchTable, GpuMatchError> {
        let mut tables = self.search_batch_with_max_dist(&[input], search_budget, max_dist)?;
        tables.pop().ok_or_else(|| {
            GpuMatchError("GPU search did not produce a result for its input".into())
        })
    }

    pub(crate) fn search_batch_with_max_dist(
        &self,
        inputs: &[&[u8]],
        search_budget: u32,
        max_dist: u32,
    ) -> Result<Vec<GpuMatchTable>, GpuMatchError> {
        self.search_batch_with_preparation(inputs, search_budget, max_dist, &prepare_serial)
    }

    #[cfg(feature = "parallel")]
    pub(crate) fn search_batch_with_pool(
        &self,
        inputs: &[&[u8]],
        search_budget: u32,
        max_dist: u32,
        pool: &crate::BatchCodec,
    ) -> Result<Vec<GpuMatchTable>, GpuMatchError> {
        self.search_batch_with_preparation(
            inputs,
            search_budget,
            max_dist,
            &|inputs, budget, distance, workgroups| {
                prepare_parallel(inputs, budget, distance, workgroups, pool)
            },
        )
    }

    #[cfg(feature = "parallel")]
    pub(crate) fn search_with_pool(
        &self,
        input: &[u8],
        search_budget: u32,
        max_dist: u32,
        pool: &crate::BatchCodec,
    ) -> Result<GpuMatchTable, GpuMatchError> {
        let mut tables = self.search_batch_with_pool(&[input], search_budget, max_dist, pool)?;
        tables.pop().ok_or_else(|| {
            GpuMatchError("GPU search did not produce a result for its input".into())
        })
    }

    fn search_batch_with_preparation(
        &self,
        inputs: &[&[u8]],
        search_budget: u32,
        max_dist: u32,
        prepare: &PrepareInputs<'_>,
    ) -> Result<Vec<GpuMatchTable>, GpuMatchError> {
        validate_search_parameters(search_budget, max_dist)?;

        self.search_with_geometry(
            inputs,
            search_budget,
            max_dist,
            SearchGeometry::from_limits(&self.limits)?,
            prepare,
        )
    }

    fn search_with_geometry(
        &self,
        inputs: &[&[u8]],
        search_budget: u32,
        max_dist: u32,
        geometry: SearchGeometry,
        prepare: &PrepareInputs<'_>,
    ) -> Result<Vec<GpuMatchTable>, GpuMatchError> {
        for input in inputs {
            u32::try_from(input.len()).map_err(|_| {
                GpuMatchError("GPU match search input exceeds u32::MAX bytes".into())
            })?;
        }
        // Common case: retain the existing single-submission path, without
        // intermediate tables or segment metadata.
        if inputs
            .iter()
            .try_fold(0, |used, input| geometry.append(used, input.len()))
            .is_some()
        {
            return self.search_group(inputs, search_budget, max_dist, prepare);
        }

        let mut tables = inputs
            .iter()
            .map(|_| GpuMatchTable::default())
            .collect::<Vec<_>>();
        let mut pending = Vec::new();
        let mut used = 0;
        let history = (max_dist - 1) as usize;
        let lookahead = (MAX_MATCH - 1) as usize;
        for (input_index, &input) in inputs.iter().enumerate() {
            let core_capacity = geometry.core_capacity(input.len(), max_dist)?;
            if core_capacity < input.len() {
                tables[input_index]
                    .entries
                    .try_reserve_exact(input.len())
                    .map_err(|error| {
                        GpuMatchError(format!("could not reserve GPU match table: {error}"))
                    })?;
            }
            for core_start in (0..input.len()).step_by(core_capacity) {
                let core_end = core_start.saturating_add(core_capacity).min(input.len());
                let start = core_start.saturating_sub(history);
                let end = core_end.saturating_add(lookahead).min(input.len());
                let part = SearchPart {
                    input_index,
                    input: &input[start..end],
                    keep: core_start - start..core_end - start,
                };
                if geometry.append(used, part.input.len()).is_none() {
                    self.finish_group(&mut pending, &mut tables, search_budget, max_dist, prepare)?;
                    used = 0;
                }
                used = geometry.append(used, part.input.len()).ok_or_else(|| {
                    GpuMatchError("GPU part does not fit a single submission".into())
                })?;
                pending.push(part);
            }
        }
        self.finish_group(&mut pending, &mut tables, search_budget, max_dist, prepare)?;
        Ok(tables)
    }

    fn finish_group(
        &self,
        pending: &mut Vec<SearchPart<'_>>,
        tables: &mut [GpuMatchTable],
        search_budget: u32,
        max_dist: u32,
        prepare: &PrepareInputs<'_>,
    ) -> Result<(), GpuMatchError> {
        if pending.is_empty() {
            return Ok(());
        }
        let inputs = pending.iter().map(|part| part.input).collect::<Vec<_>>();
        let results = self.search_group(&inputs, search_budget, max_dist, prepare)?;
        for (part, result) in pending.drain(..).zip(results) {
            let target = &mut tables[part.input_index];
            if part.keep == (0..result.len()) && target.is_empty() {
                // An unsplit input retains its table without a copy.
                *target = result;
            } else {
                // Relative distances are independent of each part's origin.
                target.entries.extend_from_slice(&result.entries[part.keep]);
            }
        }
        Ok(())
    }

    fn search_group(
        &self,
        inputs: &[&[u8]],
        search_budget: u32,
        max_dist: u32,
        prepare: &PrepareInputs<'_>,
    ) -> Result<Vec<GpuMatchTable>, GpuMatchError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let limits = &self.limits;
        let prepared = prepare(
            inputs,
            search_budget,
            max_dist,
            limits.max_compute_workgroups_per_dimension,
        )?;
        debug_assert_eq!(prepared.len(), inputs.len());
        let mut batch = GpuBatch::plan(prepared, limits)?;
        if batch.dispatches.is_empty() {
            return Ok(batch.tables);
        }
        match &self.backend {
            GpuBackend::Vulkan(backend) => backend.execute(&mut batch, limits.max_buffer_size)?,
            #[cfg(feature = "cuda")]
            GpuBackend::Cuda(backend) => backend.execute(&mut batch)?,
        }
        Ok(batch.tables)
    }
}

impl VulkanBackend {
    fn new(adapter_filter: Option<&str>) -> Result<(Self, String), GpuMatchError> {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::VULKAN,
            ..InstanceDescriptor::new_without_display_handle()
        });
        let adapter = select_adapter(&instance, adapter_filter)?;
        let adapter_name = adapter.get_info().name;
        let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
            required_features: Features::default(),
            required_limits: Limits::default(),
            memory_hints: MemoryHints::default(),
            experimental_features: ExperimentalFeatures::disabled(),
            trace: Trace::default(),
            label: Some("zlib-rs gpu match search"),
        }))
        .map_err(|error| GpuMatchError(format!("failed to create GPU device: {error}")))?;

        let create_pipeline = |aligned: bool| {
            // Generate a constant-specialized module, rather than leaving
            // per-candidate layout branches or driver override resolution.
            let module = device.create_shader_module(ShaderModuleDescriptor {
                label: Some("zlib whole-input match search"),
                source: wgpu::ShaderSource::Wgsl(
                    format!("const aligned_input: bool = {aligned};\nconst max_match: u32 = {MAX_MATCH}u;\nconst group_size: u32 = {WORKGROUP_SIZE}u;\nconst cooperative_span: u32 = {COOPERATIVE_SPAN}u;\n{SHADER}").into(),
                ),
            });
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("whole-input match search"),
                layout: None,
                module: &module,
                entry_point: Some("search"),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let packed_pipeline = create_pipeline(false);
        let aligned_pipeline = create_pipeline(true);

        let workspace = Mutex::new(GpuWorkspace::new(&device));
        Ok((
            Self {
                device,
                queue,
                packed_pipeline,
                aligned_pipeline,
                workspace,
            },
            adapter_name,
        ))
    }

    fn execute(&self, batch: &mut GpuBatch, max_buffer_size: u64) -> Result<(), GpuMatchError> {
        let mut workspace = self
            .workspace
            .lock()
            .map_err(|_| GpuMatchError("GPU match workspace lock was poisoned".into()))?;
        workspace.ensure_batch(&self.device, &batch.layout, max_buffer_size)?;
        workspace.upload(&self.queue, &batch.dispatches);
        self.dispatch_batch(&workspace, &batch.dispatches);
        workspace.readback(&self.device, batch)?;
        drop(workspace);
        Ok(())
    }

    fn dispatch_batch(&self, workspace: &GpuWorkspace, dispatches: &[GpuDispatch]) {
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("whole-input match batch command encoder"),
            });
        let packed_layout = self.packed_pipeline.get_bind_group_layout(0);
        let aligned_layout = self.aligned_pipeline.get_bind_group_layout(0);
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("whole-input match batch"),
            timestamp_writes: None,
        });
        for dispatch in dispatches {
            let item = &dispatch.input;
            let (pipeline, layout) = match item.input_layout {
                InputLayout::Packed => (&self.packed_pipeline, &packed_layout),
                InputLayout::Aligned => (&self.aligned_pipeline, &aligned_layout),
            };
            let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                label: Some("whole-input packed match buffers"),
                layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: &workspace.input.buffer,
                            offset: dispatch.input_offset,
                            size: NonZeroU64::new(item.input_size()),
                        }),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: &workspace.predecessors.buffer,
                            offset: dispatch.predecessors_offset,
                            size: NonZeroU64::new(item.predecessors_size()),
                        }),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: &workspace.results.buffer,
                            offset: dispatch.result_offset,
                            size: NonZeroU64::new(item.result_size()),
                        }),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: &workspace.params.buffer,
                            offset: dispatch.params_offset,
                            size: NonZeroU64::new(size_of_val(&item.params) as u64),
                        }),
                    },
                ],
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(item.dispatch.0, item.dispatch.1, 1);
        }
        drop(pass);
        for dispatch in dispatches {
            encoder.copy_buffer_to_buffer(
                &workspace.results.buffer,
                dispatch.result_offset,
                &workspace.download.buffer,
                dispatch.download.start as u64,
                dispatch.input.result_size(),
            );
        }
        self.queue.submit(Some(encoder.finish()));
    }
}

pub(super) fn validate_search_parameters(
    search_budget: u32,
    max_dist: u32,
) -> Result<(), GpuMatchError> {
    if !(1..=PACKED_DISTANCE_LIMIT).contains(&max_dist) {
        return Err(GpuMatchError(format!(
            "GPU match distance must be between 1 and {PACKED_DISTANCE_LIMIT}"
        )));
    }
    if search_budget == 0 {
        return Err(GpuMatchError(
            "GPU match search budget must be nonzero".into(),
        ));
    }
    Ok(())
}

fn select_adapter(instance: &Instance, filter: Option<&str>) -> Result<Adapter, GpuMatchError> {
    let adapters = pollster::block_on(instance.enumerate_adapters(Backends::VULKAN));
    let adapter = if let Some(filter) = filter {
        let filter = filter.to_ascii_lowercase();
        adapters.into_iter().find(|adapter| {
            adapter
                .get_info()
                .name
                .to_ascii_lowercase()
                .contains(&filter)
        })
    } else {
        adapters
            .into_iter()
            .max_by_key(|adapter| device_type_rank(adapter.get_info().device_type))
    };
    adapter.ok_or_else(|| {
        GpuMatchError(format!(
            "no Vulkan adapter matched {}",
            filter.unwrap_or("the default selection")
        ))
    })
}

const fn device_type_rank(device_type: DeviceType) -> u8 {
    match device_type {
        DeviceType::DiscreteGpu => 5,
        DeviceType::IntegratedGpu => 4,
        DeviceType::VirtualGpu => 3,
        DeviceType::Cpu => 2,
        DeviceType::Other => 1,
    }
}

/// Estimate the fraction of workgroups that can use the exact cooperative
/// full-match shortcut. Only enable synchronization when at least a quarter
/// of the sampled groups have a verified full-length common predecessor.
fn cooperative_groups_likely(input: &[u8], predecessors: &[u32]) -> bool {
    const SAMPLES: usize = 64;
    const GROUP: usize = WORKGROUP_SIZE as usize;
    const SPAN: usize = COOPERATIVE_SPAN as usize;
    let groups = input.len().saturating_sub(SPAN) / GROUP;
    if groups < SAMPLES {
        return false;
    }
    let mut repeated = 0;
    for sample in 0..SAMPLES {
        let group = sample * groups / SAMPLES;
        let base = group * GROUP;
        let distance = predecessor_distance(predecessors, base);
        if distance != 0
            && distance == predecessor_distance(predecessors, base + GROUP - 1)
            && distance <= base
            && input[base..base + SPAN] == input[base - distance..base - distance + SPAN]
        {
            repeated += 1;
        }
    }
    repeated >= SAMPLES / 4
}

/// Choose a compact byte-packed layout for cheap searches, and one
/// overlapping word per byte when enough history traversal is expected
/// to amortize the additional input upload. Both produce identical tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InputLayout {
    Packed,
    Aligned,
}

impl InputLayout {
    // Bound the classifier independently of input length and search budget.
    const SAMPLES: usize = 64;
    const MAX_SAMPLE_VISITS: usize = 16;
    const MIN_MEAN_VISITS: u32 = 8;
    const MIN_EXPANDED_BYTES: usize = 16 * 1024;

    fn select(input: &[u8], predecessors: &[u32], budget: u32, max_dist: u32) -> Self {
        if input.len() < Self::MIN_EXPANDED_BYTES || budget < Self::MIN_MEAN_VISITS {
            return Self::Packed;
        }
        // Stratified samples with a different phase in each stratum avoid
        // synchronizing the sample positions with a periodic input.
        let stride = (input.len() - 2) / Self::SAMPLES;
        let visits = (0..Self::SAMPLES)
            .map(|sample| {
                let phase = (sample * 37) % Self::SAMPLES;
                let offset = (stride / Self::SAMPLES) * phase;
                let position = sample * stride + offset;
                Self::sample_visits(input, predecessors, position, budget, max_dist)
            })
            .sum::<usize>();
        if visits >= Self::SAMPLES * Self::MIN_MEAN_VISITS as usize {
            Self::Aligned
        } else {
            Self::Packed
        }
    }

    fn sample_visits(
        input: &[u8],
        predecessors: &[u32],
        position: usize,
        budget: u32,
        max_dist: u32,
    ) -> usize {
        let minimum = position.saturating_sub(max_dist as usize - 1);
        let length = (input.len() - position).min(MAX_MATCH as usize);
        let max_visits = (budget as usize).min(Self::MAX_SAMPLE_VISITS);
        let mut cursor = position;
        for visited in 0..max_visits {
            let distance = predecessor_distance(predecessors, cursor);
            if distance == 0 {
                return visited;
            }
            cursor -= distance;
            if cursor < minimum {
                return visited;
            }
            // A complete match ends the shader search immediately, even
            // when the remaining chain contains thousands of predecessors.
            if input[position..position + length] == input[cursor..cursor + length] {
                return visited + 1;
            }
        }
        max_visits
    }

    fn pack(self, input: &[u8]) -> Vec<u32> {
        match self {
            Self::Packed => pack_words(input),
            Self::Aligned => {
                let mut words = Vec::with_capacity(input.len());
                words.extend(
                    input
                        .windows(4)
                        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])),
                );
                // Only byte comparisons can use the final three positions.
                words.extend(
                    input[input.len().saturating_sub(3)..]
                        .iter()
                        .copied()
                        .map(u32::from),
                );
                words
            }
        }
    }
}

/// Pick the first, last or middle triple giving the tightest upper bound
/// on the next candidate's start. A missing predecessor proves exhaustion.
const fn shifted_offset(predecessors: &[u32], candidate: usize, length: usize) -> usize {
    let tail = length - 3;
    let middle = tail / 2;
    let mut offset = 0;
    let mut distance = predecessor_distance(predecessors, candidate);
    if distance == 0 {
        return 0;
    }
    let tail_distance = predecessor_distance(predecessors, candidate + tail);
    if tail_distance == 0 {
        return tail;
    }
    if tail_distance > distance {
        offset = tail;
        distance = tail_distance;
    }
    let middle_distance = predecessor_distance(predecessors, candidate + middle);
    if middle_distance == 0 || middle_distance > distance {
        return middle;
    }
    offset
}

/// Resolve the same nearest-first, budgeted search as the shader. Return
/// None when the bounded local effort would be exceeded; a caller can then
/// dispatch the original unmodified index on the GPU.
fn resolve_bounded(
    input: &[u8],
    predecessors: &[u32],
    budget: u32,
    max_dist: u32,
) -> Option<GpuMatchTable> {
    let mut work_left = input.len().saturating_mul(16).saturating_add(256);
    let mut entries = vec![0; input.len()];
    for (position, entry) in entries
        .iter_mut()
        .enumerate()
        .take(input.len().saturating_sub(2))
    {
        let mut distance = predecessor_distance(predecessors, position);
        if distance == 0 {
            continue;
        }
        let max_len = (input.len() - position).min(MAX_MATCH as usize);
        let min_start = position.saturating_sub(max_dist as usize - 1);
        let query = &input[position..position + max_len];
        let mut cursor = position;
        let mut chain_offset = 0;
        let mut best_len = 0;
        let mut best_distance = 0;
        for _ in 0..budget {
            if distance == 0 || best_len == max_len {
                break;
            }
            if distance > cursor {
                break;
            }
            cursor -= distance;
            if cursor < min_start {
                break;
            }
            distance = predecessor_distance(predecessors, cursor + chain_offset);
            work_left = work_left.checked_sub(1)?;
            if best_len >= 3 && query[best_len] != input[cursor + best_len] {
                continue;
            }
            let candidate = &input[cursor..cursor + max_len];
            if query[..3] != candidate[..3] {
                continue;
            }
            let len = 3 + super::match_compare::common_prefix_len(&query[3..], &candidate[3..]);
            work_left = work_left.checked_sub(len / 32)?;
            if len > best_len {
                best_len = len;
                best_distance = position - cursor;
                if len < max_len {
                    if len > 3 {
                        work_left = work_left.checked_sub(3)?;
                        chain_offset = shifted_offset(predecessors, cursor, len);
                        distance = predecessor_distance(predecessors, cursor + chain_offset);
                    }
                    if distance != 0 {
                        work_left = work_left.checked_sub(1)?;
                        let extension_offset = len - 2;
                        let extension_distance =
                            predecessor_distance(predecessors, position + extension_offset);
                        if extension_distance == 0 {
                            distance = 0;
                        } else if extension_distance > best_distance + distance {
                            chain_offset = extension_offset;
                            distance = extension_distance - best_distance;
                        }
                    }
                }
            }
        }
        *entry = (u32::try_from(best_distance).ok()? << 16) | u32::try_from(best_len).ok()?;
    }
    Some(GpuMatchTable { entries })
}

fn pack_words(input: &[u8]) -> Vec<u32> {
    input
        .chunks(4)
        .map(|chunk| {
            let mut bytes = [0u8; 4];
            bytes[..chunk.len()].copy_from_slice(chunk);
            u32::from_le_bytes(bytes)
        })
        .collect()
}

/// Encode the preceding same-bucket position as a backward distance.
/// An out-of-window predecessor terminates the chain: all older positions
/// are necessarily outside the window as well.
fn build_predecessors(input: &[u8], max_dist: u32) -> Result<Vec<u32>, GpuMatchError> {
    let valid_positions = input.len().saturating_sub(2);
    let positions = u32::try_from(valid_positions)
        .map_err(|_| GpuMatchError("GPU hash index exceeds u32::MAX positions".into()))?;
    // For short inputs the dense 65,536-entry head array dominates
    // preparation. A small exact-key table retains the same bucket
    // identity (including collisions in hash3) with at most half occupancy.
    if valid_positions <= SMALL_HEADS_CAPACITY / 2 {
        let mut heads = SmallHeads::default();
        Ok(build_predecessor_words(
            input,
            positions,
            max_dist,
            |hash, position| heads.replace(hash, position),
        ))
    } else {
        let mut heads = vec![NO_PREDECESSOR; HASH_SIZE];
        Ok(build_predecessor_words(
            input,
            positions,
            max_dist,
            |hash, position| core::mem::replace(&mut heads[hash], position),
        ))
    }
}

// At most 256 distinct keys in a 512-slot table. Position+1 fits in
// the low 16 bits and reserves zero for an unused slot, even for hash 0.
const SMALL_HEADS_CAPACITY: usize = 512;
struct SmallHeads([u32; SMALL_HEADS_CAPACITY]);

impl Default for SmallHeads {
    fn default() -> Self {
        Self([0; SMALL_HEADS_CAPACITY])
    }
}

impl SmallHeads {
    #[inline]
    fn replace(&mut self, hash: usize, position: u32) -> u32 {
        let mask = SMALL_HEADS_CAPACITY - 1;
        let Ok(key) = u16::try_from(hash) else {
            unreachable!("GPU hash key exceeds its 16-bit domain");
        };
        let key = u32::from(key) << 16;
        debug_assert!(position as usize <= SMALL_HEADS_CAPACITY / 2);
        let mut slot = hash & mask;
        loop {
            let old = self.0[slot];
            if old == 0 || old & 0xffff_0000 == key {
                self.0[slot] = key | (position + 1);
                return if old == 0 {
                    NO_PREDECESSOR
                } else {
                    (old & 0xffff) - 1
                };
            }
            slot = (slot + 1) & mask;
        }
    }
}

fn build_predecessor_words(
    input: &[u8],
    positions: u32,
    max_dist: u32,
    replace_head: impl FnMut(usize, u32) -> u32,
) -> Vec<u32> {
    // Nonempty short inputs still require a nonzero-sized array.
    let mut predecessors = vec![0; (positions as usize).div_ceil(2).max(1)];
    fill_predecessor_words(
        input,
        0..positions,
        max_dist,
        &mut predecessors,
        replace_head,
    );
    predecessors
}

/// Warm up only the history that can contain a valid predecessor. Each shard
/// starts on an even position, so packed words never straddle worker ownership.
fn fill_predecessor_words(
    input: &[u8],
    positions: Range<u32>,
    max_dist: u32,
    predecessors: &mut [u32],
    mut replace_head: impl FnMut(usize, u32) -> u32,
) {
    let start = positions.start;
    let history_start = start.saturating_sub(max_dist - 1);
    for (position, bytes) in (history_start..start).zip(input[history_start as usize..].windows(3))
    {
        replace_head(hash3(bytes), position);
    }
    for (position, bytes) in positions.zip(input[start as usize..].windows(3)) {
        let previous = replace_head(hash3(bytes), position);
        if previous != NO_PREDECESSOR {
            let distance = position - previous;
            if distance < max_dist {
                let local = position - start;
                predecessors[local as usize / 2] |= distance << ((local & 1) * 16);
            }
        }
    }
}

#[cfg(feature = "parallel")]
fn build_predecessors_parallel(
    input: &[u8],
    max_dist: u32,
    pool: &crate::BatchCodec,
) -> Result<Vec<u32>, GpuMatchError> {
    const MIN_SHARD_WORDS: usize = 128 * 1024;
    let positions = input.len().saturating_sub(2);
    u32::try_from(positions)
        .map_err(|_| GpuMatchError("GPU hash index exceeds u32::MAX positions".into()))?;
    let words = positions.div_ceil(2);
    // Bound duplicated warmup and dense head-table allocation. Divide large
    // streams among the existing pool, without splitting the GPU dispatch
    // or duplicating its input/result transfers.
    let shard_words = words.div_ceil(pool.workers()).max(MIN_SHARD_WORDS);
    if words <= shard_words {
        return build_predecessors(input, max_dist);
    }
    let mut predecessors = vec![0; words];
    let mut shards = predecessors.chunks_mut(shard_words).collect::<Vec<_>>();
    pool.run_items(
        &mut shards,
        || Ok(()),
        |index, shard| {
            let start = index * shard_words * 2;
            let end = start.saturating_add(shard.len() * 2).min(positions);
            let to_position = |offset| {
                u32::try_from(offset)
                    .map_err(|_| GpuMatchError("GPU hash shard exceeds u32::MAX positions".into()))
            };
            let mut heads = vec![NO_PREDECESSOR; HASH_SIZE];
            fill_predecessor_words(
                input,
                to_position(start)?..to_position(end)?,
                max_dist,
                shard,
                |hash, position| core::mem::replace(&mut heads[hash], position),
            );
            Ok(())
        },
    )
    .into_iter()
    .collect::<Result<(), _>>()?;
    Ok(predecessors)
}

const fn predecessor_distance(words: &[u32], position: usize) -> usize {
    ((words[position / 2] >> ((position & 1) * 16)) & 0xffff) as usize
}

fn hash3(bytes: &[u8]) -> usize {
    let value = u32::from(bytes[0]) | (u32::from(bytes[1]) << 8) | (u32::from(bytes[2]) << 16);
    (value.wrapping_mul(2_654_435_761) >> 16) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_predecessors_preserve_every_serial_word() -> Result<(), Box<dyn std::error::Error>>
    {
        let mut seed = 0x1234_5678u32;
        let random = (0..1_200_003)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                seed.to_le_bytes()[0]
            })
            .collect::<Vec<_>>();
        let periodic = random[..32_505].repeat(38);
        let repetitive = vec![b'a'; 700_001];
        for workers in [1, 2, 4] {
            let pool = crate::BatchCodec::new(
                core::num::NonZeroUsize::new(workers)
                    .ok_or_else(|| GpuMatchError("zero workers".into()))?,
            )?;
            for input in [&random, &periodic, &repetitive] {
                for distance in [1, 17, DEFAULT_MAX_DIST, PACKED_DISTANCE_LIMIT] {
                    assert_eq!(
                        build_predecessors_parallel(input, distance, &pool)?,
                        build_predecessors(input, distance)?,
                        "workers={workers}, len={}, distance={distance}",
                        input.len()
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn batch_plan_retains_local_slots_and_aligned_gpu_ranges() -> Result<(), GpuMatchError> {
        let limits = Limits {
            min_storage_buffer_offset_alignment: 256,
            min_uniform_buffer_offset_alignment: 512,
            ..Limits::default()
        };
        let make = || -> Result<Vec<PreparedSearch>, GpuMatchError> {
            Ok(vec![
                prepare_input(0, b"", 32, DEFAULT_MAX_DIST, 65535, None)?,
                prepare_input(
                    1,
                    &[b'a'; 127],
                    32,
                    DEFAULT_MAX_DIST,
                    65535,
                    Some(InputLayout::Packed),
                )?,
                prepare_input(2, b"abc", 32, DEFAULT_MAX_DIST, 65535, None)?,
                prepare_input(
                    3,
                    &[b'b'; 131],
                    32,
                    DEFAULT_MAX_DIST,
                    65535,
                    Some(InputLayout::Aligned),
                )?,
            ])
        };
        let batch = GpuBatch::plan(make()?, &limits)?;
        assert_eq!(batch.tables.len(), 4);
        assert_eq!(batch.tables[2].entries, [0; 3]);
        assert_eq!(batch.dispatches.len(), 2);
        let first = &batch.dispatches[0];
        let second = &batch.dispatches[1];
        assert_eq!(first.input.input_index, 1);
        assert_eq!(second.input.input_index, 3);
        assert_eq!(first.download, 0..127 * 4);
        assert_eq!(second.download, 127 * 4..258 * 4);
        assert_eq!(batch.layout.download, 258 * 4);
        for offset in [
            second.input_offset,
            second.predecessors_offset,
            second.result_offset,
        ] {
            assert_ne!(offset, 0);
            assert_eq!(offset % 256, 0);
        }
        assert_eq!(second.params_offset, 512);
        assert!(batch.layout.input >= second.input_offset + second.input.input_size());
        assert!(batch.layout.results >= second.result_offset + second.input.result_size());
        assert!(GpuBatch::plan(
            make()?,
            &Limits {
                max_buffer_size: 512,
                ..limits
            }
        )
        .is_err());
        assert!(GpuBatch::plan(
            make()?,
            &Limits {
                max_storage_buffer_binding_size: 64,
                ..limits
            }
        )
        .is_err());
        assert!(GpuBatch::plan(
            make()?,
            &Limits {
                max_uniform_buffer_binding_size: 8,
                ..limits
            }
        )
        .is_err());
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires CUDA hardware and NVRTC"]
    fn cuda_matches_independent_oracle() -> Result<(), GpuMatchError> {
        let searcher = GpuMatchSearcher::new(Some("cuda:0"))?;
        let mut seed = 0x1234_5678u32;
        let random = (0..4_097)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                seed.to_le_bytes()[0] & 15
            })
            .collect::<Vec<_>>();
        let mut repeated = b"PDF-stream-pattern-0123456789".repeat(600);
        repeated[8193] = b'!';
        let inputs = [&random[..], &repeated[..]];
        let geometry = SearchGeometry::from_limits(&searcher.limits)?;
        for layout in [InputLayout::Packed, InputLayout::Aligned] {
            for budget in [1, 32, 128] {
                let tables = searcher.search_with_geometry(
                    &inputs,
                    budget,
                    128,
                    geometry,
                    &|inputs, budget, distance, groups| {
                        inputs
                            .iter()
                            .enumerate()
                            .map(|(index, input)| {
                                prepare_input(index, input, budget, distance, groups, Some(layout))
                            })
                            .collect()
                    },
                )?;
                for (input, table) in inputs.iter().zip(tables) {
                    for (position, actual) in table.iter().enumerate() {
                        assert_eq!(
                            actual,
                            oracle_match(input, position, 128, budget)?,
                            "pos={position}, budget={budget}"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires a Vulkan compute adapter"]
    fn cooperative_groups_preserve_exact_match_tables() -> Result<(), GpuMatchError> {
        let searcher = GpuMatchSearcher::new(None)?;
        let mut periodic = b"PDF-stream-pattern-0123456789".repeat(600);
        periodic[12_001] = b'!';
        periodic[12_060] = b'?';
        let repeated = vec![b'a'; 16_513];
        let inputs = [&periodic[..], &repeated[..]];
        let geometry = SearchGeometry::from_limits(&searcher.limits)?;
        for layout in [InputLayout::Packed, InputLayout::Aligned] {
            for max_dist in [64, DEFAULT_MAX_DIST] {
                let run = |enabled| {
                    searcher.search_with_geometry(
                        &inputs,
                        32,
                        max_dist,
                        geometry,
                        &|inputs, budget, distance, groups| {
                            inputs
                                .iter()
                                .enumerate()
                                .map(|(index, input)| {
                                    let mut prepared = prepare_input(
                                        index,
                                        input,
                                        budget,
                                        distance,
                                        groups,
                                        Some(layout),
                                    )?;
                                    if let PreparedSearch::Gpu(item) = &mut prepared {
                                        assert_eq!(item.params[2], 1);
                                        item.params[2] = u32::from(enabled);
                                    }
                                    Ok(prepared)
                                })
                                .collect()
                        },
                    )
                };
                let ordinary = run(false)?;
                let cooperative = run(true)?;
                for (a, b) in ordinary.iter().zip(cooperative) {
                    assert_eq!(a.entries, b.entries);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn adaptive_layout_uses_bounded_search_samples() -> Result<(), GpuMatchError> {
        let mut seed = 0x4265_7391u32;
        let random = (0..65536)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                seed.to_le_bytes()[0]
            })
            .collect::<Vec<_>>();
        let periodic = b"abc01234567".repeat(6000);
        // Many same-bucket candidates but no full match: different fourth
        // bytes recur, followed by independent random data.
        let deep = random
            .chunks(1)
            .flat_map(|tail| {
                b"abcdefghijklmno"
                    .iter()
                    .copied()
                    .chain(tail.iter().copied())
            })
            .collect::<Vec<_>>();
        for (input, expected) in [
            (random.as_slice(), InputLayout::Packed),
            (periodic.as_slice(), InputLayout::Packed),
            (deep.as_slice(), InputLayout::Aligned),
        ] {
            let pred = build_predecessors(input, DEFAULT_MAX_DIST)?;
            assert_eq!(
                InputLayout::select(input, &pred, 512, DEFAULT_MAX_DIST),
                expected
            );
            assert_eq!(
                InputLayout::select(input, &pred, 1, DEFAULT_MAX_DIST),
                InputLayout::Packed
            );
        }
        for len in 0..=12 {
            let input = &random[..len];
            let words = InputLayout::Aligned.pack(input);
            assert_eq!(words.len(), len);
            for (position, word) in words.iter().enumerate() {
                assert_eq!(word.to_le_bytes()[0], input[position]);
                if position + 4 <= len {
                    assert_eq!(word.to_le_bytes(), input[position..position + 4]);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn compact_heads_preserve_dense_chain_semantics() -> Result<(), GpuMatchError> {
        let mut seed = 12345u32;
        let mut random = [0; 260];
        for value in &mut random {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            *value = seed.to_le_bytes()[0];
        }
        let repeats = b"abc01234".repeat(33);
        for input in [&random[..], repeats.as_slice()] {
            for len in 0..=input.len() {
                let input = &input[..len];
                let positions = u32::try_from(input.len().saturating_sub(2))
                    .map_err(|e| GpuMatchError(e.to_string()))?;
                for max_dist in [1, 2, 15, 64, DEFAULT_MAX_DIST] {
                    let mut dense = vec![NO_PREDECESSOR; HASH_SIZE];
                    let expected =
                        build_predecessor_words(input, positions, max_dist, |hash, position| {
                            core::mem::replace(&mut dense[hash], position)
                        });
                    assert_eq!(
                        build_predecessors(input, max_dist)?,
                        expected,
                        "len={len}, dist={max_dist}"
                    );
                }
            }
        }
        // All keys probe the same initial slot; exact identities and
        // replacement values remain distinct at maximum occupancy.
        let mut heads = SmallHeads::default();
        for i in 0..128usize {
            for key in [i * 512, i * 512 + 1] {
                let pos = u32::try_from(i).map_err(|e| GpuMatchError(e.to_string()))?;
                assert_eq!(heads.replace(key, pos), NO_PREDECESSOR);
                assert_eq!(heads.replace(key, 256), pos);
            }
        }
        Ok(())
    }

    #[test]
    fn predecessors_match_reverse_bucket_prefixes() -> Result<(), GpuMatchError> {
        let mut random = 0x618d_0e32u32;
        let input = (0..2048)
            .map(|i| {
                random ^= random << 13;
                random ^= random >> 17;
                random ^= random << 5;
                if i % 7 > 2 {
                    b"abc"[i % 3]
                } else {
                    random.to_le_bytes()[0]
                }
            })
            .collect::<Vec<_>>();
        let predecessors = build_predecessors(&input, DEFAULT_MAX_DIST)?;
        let mut found_collision = false;
        for position in 0..input.len() - 2 {
            let bytes = &input[position..position + 3];
            let hash = hash3(bytes);
            let expected = (0..position)
                .rev()
                .filter(|&candidate| hash3(&input[candidate..candidate + 3]) == hash)
                .collect::<Vec<_>>();
            let mut actual = Vec::new();
            let mut cursor = position;
            loop {
                let distance = predecessor_distance(&predecessors, cursor);
                if distance == 0 {
                    break;
                }
                cursor -= distance;
                assert!(cursor < position);
                found_collision |= &input[cursor..cursor + 3] != bytes;
                actual.push(cursor);
            }
            assert_eq!(actual, expected);
            // Also preserve finite-budget and limited-history traversal.
            for (minimum, budget) in [(0, 1), (position / 2, 7), (position.saturating_sub(32), 64)]
            {
                assert_eq!(
                    actual
                        .iter()
                        .copied()
                        .take_while(|&candidate| candidate >= minimum)
                        .take(budget)
                        .collect::<Vec<_>>(),
                    expected
                        .iter()
                        .copied()
                        .filter(|&candidate| candidate >= minimum)
                        .take(budget)
                        .collect::<Vec<_>>(),
                );
            }
        }
        assert!(found_collision);
        Ok(())
    }

    #[test]
    #[ignore = "requires a Vulkan compute adapter"]
    fn gpu_predecessor_tables_match_budgeted_cpu_oracle() -> Result<(), GpuMatchError> {
        let searcher = GpuMatchSearcher::new(None)?;
        let mut seed = 0x42bd_9431u32;
        let mixed = (0..512)
            .map(|index| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                if index % 7 > 1 {
                    b"abc"[index % 3]
                } else {
                    seed.to_le_bytes()[0]
                }
            })
            .collect::<Vec<_>>();
        let repetitive = b"abc".repeat(100);
        let inputs = [b"".as_slice(), b"a", b"ab", b"abc", &mixed, &repetitive];
        for (max_dist, budget) in [(1, 1), (8, 7), (64, 1), (64, 31), (DEFAULT_MAX_DIST, 64)] {
            let geometry = SearchGeometry::from_limits(&searcher.limits)?;
            for layout in [InputLayout::Packed, InputLayout::Aligned] {
                let tables = searcher.search_with_geometry(
                    &inputs,
                    budget,
                    max_dist,
                    geometry,
                    &|inputs, budget, distance, groups| {
                        inputs
                            .iter()
                            .enumerate()
                            .map(|(index, input)| {
                                prepare_input(index, input, budget, distance, groups, Some(layout))
                            })
                            .collect()
                    },
                )?;
                assert_eq!(tables.len(), inputs.len());
                for (input, table) in inputs.iter().zip(tables) {
                    assert_eq!(table.len(), input.len());
                    for (position, actual) in table.iter().enumerate() {
                        let expected = oracle_match(input, position, max_dist, budget)?;
                        assert_eq!(
                            actual, expected,
                            "max_dist={max_dist}, budget={budget}, position={position}"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn short_inputs_have_a_valid_storage_binding() -> Result<(), GpuMatchError> {
        for input in [b"".as_slice(), b"a", b"ab", b"abc"] {
            assert_eq!(build_predecessors(input, DEFAULT_MAX_DIST)?, [0]);
        }
        Ok(())
    }

    #[test]
    fn packed_distances_preserve_zero_and_window_boundaries() -> Result<(), GpuMatchError> {
        // The first triple starts at zero, then repeat at distances straddling the window.
        let mut input = b"abc".to_vec();
        input.resize(32_505, 0);
        input.extend_from_slice(b"abc");
        input.resize(65_011, 0);
        input.extend_from_slice(b"abc");
        let words = build_predecessors(&input, DEFAULT_MAX_DIST)?;
        assert_eq!(predecessor_distance(&words, 0), 0);
        assert_eq!(predecessor_distance(&words, 32_505), 32_505);
        assert_eq!(predecessor_distance(&words, 65_011), 0);
        assert_eq!(words.len(), (input.len() - 2).div_ceil(2));
        let result = GpuMatchTable {
            entries: vec![0, (1 << 16) | 3, 0],
        };
        assert_eq!(result.len(), 3);
        assert_eq!(result.get(0), Some(GpuMatch::default()));
        assert_eq!(
            result.get(1),
            Some(GpuMatch {
                length: 3,
                start: 0
            })
        );
        assert_eq!(result.get(3), None);
        assert_eq!(
            result.iter().collect::<Vec<_>>(),
            (0..result.len())
                .filter_map(|position| result.get(position))
                .collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn group_geometry_counts_alignment_and_uniforms() -> Result<(), GpuMatchError> {
        let geometry = SearchGeometry {
            max_part_len: 1024,
            max_group_bytes: 4096,
            alignment: 256,
        };
        assert_eq!(geometry.append(0, 0), Some(0));
        assert_eq!(geometry.append(0, 1), Some(16));
        assert_eq!(geometry.append(16, 1), Some(272));
        assert_eq!(geometry.append(0, 1024), Some(4096));
        assert_eq!(geometry.append(16, 1024), None);
        assert_eq!(geometry.append(0, 1025), None);
        assert_eq!(geometry.core_capacity(2048, 64)?, 704);
        assert!(geometry.core_capacity(2048, DEFAULT_MAX_DIST).is_err());
        Ok(())
    }

    fn oracle_match(
        input: &[u8],
        pos: usize,
        max_dist: u32,
        budget: u32,
    ) -> Result<GpuMatch, GpuMatchError> {
        let mut best = GpuMatch::default();
        let len = (input.len() - pos).min(MAX_MATCH as usize);
        if len < 3 {
            return Ok(best);
        }
        let minimum = pos.saturating_sub(max_dist as usize - 1);
        let mut cursor = pos;
        let mut offset = 0;
        for _ in 0..budget {
            let hash = hash3(&input[pos + offset..pos + offset + 3]);
            let Some(start) = (minimum..cursor)
                .rev()
                .find(|&start| hash3(&input[start + offset..start + offset + 3]) == hash)
            else {
                break;
            };
            cursor = start;
            let n = input[pos..pos + len]
                .iter()
                .zip(&input[start..start + len])
                .take_while(|(a, b)| a == b)
                .count();
            if n >= 3 && n > best.length as usize {
                best = GpuMatch {
                    length: u32::try_from(n).map_err(|e| GpuMatchError(e.to_string()))?,
                    start: u32::try_from(start).map_err(|e| GpuMatchError(e.to_string()))?,
                };
                if n == len {
                    break;
                }
                {
                    let mut selected = 0;
                    let mut nearest = (minimum..start).rev().find(|&candidate| {
                        hash3(&input[candidate..candidate + 3]) == hash3(&input[pos..pos + 3])
                    });
                    for probe_offset in [n - 3, (n - 3) / 2] {
                        let key = hash3(&input[pos + probe_offset..pos + probe_offset + 3]);
                        let next = (minimum..start).rev().find(|&candidate| {
                            hash3(&input[candidate + probe_offset..candidate + probe_offset + 3])
                                == key
                        });
                        if next < nearest {
                            nearest = next;
                            selected = probe_offset;
                        }
                    }
                    if nearest.is_some() {
                        let probe_offset = n - 2;
                        let key = hash3(&input[pos + probe_offset..pos + probe_offset + 3]);
                        let next = (minimum..pos).rev().find(|&candidate| {
                            hash3(&input[candidate + probe_offset..candidate + probe_offset + 3])
                                == key
                        });
                        if next < nearest {
                            selected = probe_offset;
                        }
                    }
                    offset = selected;
                }
            }
        }
        Ok(best)
    }

    #[test]
    #[ignore = "requires a Vulkan compute adapter"]
    fn shifted_chains_skip_only_impossible_improvements() -> Result<(), GpuMatchError> {
        let mut input = b"abcDEFGHIJKLMNOPQRSTUVWXYZ".to_vec();
        for index in (0..100u8).filter(|&index| index != b'E') {
            input.extend_from_slice(&[b'a', b'b', b'c', b'D', index, b'!']);
        }
        input.extend_from_slice(b"abcDE!");
        input.extend_from_slice(b"abcDEFGHIJKLMNOPQRSTUVWXYZ");
        let position = input.len() - 26;
        let predecessors = build_predecessors(&input, DEFAULT_MAX_DIST)?;
        let table = resolve_bounded(&input, &predecessors, 4, DEFAULT_MAX_DIST)
            .ok_or_else(|| GpuMatchError("bounded test exceeded work".into()))?;
        assert_eq!(table.get(position).map(|entry| entry.length), Some(26));
        let mut short_prefix = b"abcDEFGHIJKLMNOPQRSTUVWXYZ".to_vec();
        for byte in 0..100u8 {
            if byte != b'D' {
                short_prefix.extend_from_slice(&[b'a', b'b', b'c', byte, b'!']);
            }
        }
        short_prefix.extend_from_slice(b"abcDEFGHIJKLMNOPQRSTUVWXYZ");
        let query = short_prefix.len() - 26;
        let index = build_predecessors(&short_prefix, DEFAULT_MAX_DIST)?;
        let resolved = resolve_bounded(&short_prefix, &index, 4, DEFAULT_MAX_DIST)
            .ok_or_else(|| GpuMatchError("bounded extension test exceeded work".into()))?;
        assert_eq!(resolved.get(query).map(|entry| entry.length), Some(26));

        let searcher = GpuMatchSearcher::new(None)?;
        let geometry = SearchGeometry::from_limits(&searcher.limits)?;
        let tables = searcher.search_with_geometry(
            &[&input],
            DEFAULT_MAX_DIST,
            DEFAULT_MAX_DIST,
            geometry,
            &|inputs, budget, distance, groups| {
                inputs
                    .iter()
                    .enumerate()
                    .map(|(index, input)| {
                        prepare_input(
                            index,
                            input,
                            budget,
                            distance,
                            groups,
                            Some(InputLayout::Aligned),
                        )
                    })
                    .collect()
            },
        )?;
        let table = tables
            .first()
            .ok_or_else(|| GpuMatchError("missing exhaustive table".into()))?;
        for (pos, actual) in table.iter().enumerate() {
            let length = (input.len() - pos).min(MAX_MATCH as usize);
            let mut best = GpuMatch::default();
            for start in (0..pos).rev() {
                let n = input[pos..pos + length]
                    .iter()
                    .zip(&input[start..start + length])
                    .take_while(|(a, b)| a == b)
                    .count();
                if n >= 3 && n > best.length as usize {
                    best = GpuMatch {
                        length: u32::try_from(n).map_err(|e| GpuMatchError(e.to_string()))?,
                        start: u32::try_from(start).map_err(|e| GpuMatchError(e.to_string()))?,
                    };
                }
            }
            assert_eq!(actual, best, "pos={pos}");
        }
        Ok(())
    }

    #[test]
    fn bounded_local_resolution_matches_nearest_first_oracle() -> Result<(), GpuMatchError> {
        let mut seed = 0x87c4_238bu32;
        let random = (0..513)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                seed.to_le_bytes()[0]
            })
            .collect::<Vec<_>>();
        let patterned = b"abc0123abcXYZabc".repeat(33);
        let mut completed = 0;
        for bytes in [&random[..], &patterned[..]] {
            for len in [3, 4, 5, 15, 31, 63, 128, 255, 512] {
                let input = &bytes[..len];
                for max_dist in [1, 2, 8, 64, DEFAULT_MAX_DIST] {
                    let pred = build_predecessors(input, max_dist)?;
                    for budget in [1, 2, 8, 64, 512] {
                        let Some(table) = resolve_bounded(input, &pred, budget, max_dist) else {
                            continue;
                        };
                        completed += 1;
                        for (pos, actual) in table.iter().enumerate() {
                            let expected = oracle_match(input, pos, max_dist, budget)?;
                            assert_eq!(
                                actual, expected,
                                "pos={pos} budget={budget} max_dist={max_dist}"
                            );
                        }
                    }
                }
            }
        }
        assert!(completed > 300);
        // Dense, irregular two-symbol input retains many candidates even
        // with shifted and extension bounds. The work cap must still abort.
        let repeated = (0..8192)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                b'a' + (seed.to_le_bytes()[0] & 1)
            })
            .collect::<Vec<_>>();
        let pred = build_predecessors(&repeated, DEFAULT_MAX_DIST)?;
        assert!(resolve_bounded(&repeated, &pred, 512, DEFAULT_MAX_DIST).is_none());
        Ok(())
    }

    #[test]
    fn preparation_handles_local_results_and_forced_gpu() -> Result<(), GpuMatchError> {
        for input in [b"".as_slice(), b"a", b"ab", b"abc", b"abcdef"] {
            let PreparedSearch::Local(table) =
                prepare_input(0, input, 1, DEFAULT_MAX_DIST, 65535, None)?
            else {
                return Err(GpuMatchError("empty candidate chain dispatched".into()));
            };
            assert_eq!(table.entries, vec![0; input.len()]);
        }
        assert!(matches!(
            prepare_input(0, b"aaaaaaaa", 1, 1, 65535, None)?,
            PreparedSearch::Local(_)
        ));
        assert!(matches!(
            prepare_input(0, b"aaaaaaaa", 1, 2, 65535, Some(InputLayout::Packed))?,
            PreparedSearch::Gpu(_)
        ));
        assert!(matches!(
            prepare_input(
                0,
                b"abcabc",
                1,
                DEFAULT_MAX_DIST,
                65535,
                Some(InputLayout::Packed)
            )?,
            PreparedSearch::Gpu(_)
        ));
        Ok(())
    }

    #[test]
    #[ignore = "requires a Vulkan compute adapter"]
    fn candidate_free_inputs_preserve_zero_match_table_lengths() -> Result<(), GpuMatchError> {
        let searcher = GpuMatchSearcher::new(None)?;
        let inputs = [
            b"".as_slice(),
            b"a",
            b"abcdef",
            b"abcabc",
            b"abc",
            b"aaaaaaaa",
        ];
        for max_dist in [1, DEFAULT_MAX_DIST] {
            let tables = searcher.search_batch_with_max_dist(&inputs, 7, max_dist)?;
            assert_eq!(tables.len(), inputs.len());
            for (index, table) in tables.iter().enumerate() {
                assert_eq!(table.len(), inputs[index].len());
                if max_dist == 1 || [0, 1, 2, 4].contains(&index) {
                    assert!(table.iter().all(|m| m == GpuMatch::default()));
                } else {
                    assert!(table.iter().any(|m| m.length >= 3));
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn pooled_preparation_preserves_order_and_contents() -> Result<(), Box<dyn std::error::Error>> {
        let pool = crate::BatchCodec::new(core::num::NonZeroUsize::new(4).ok_or("workers")?)?;
        let long = b"abc".repeat(MAX_LOCAL_INPUT_BYTES / 3 + 1);
        let inputs = [
            b"".as_slice(),
            b"a",
            b"abcabcabc",
            b"",
            b"abc",
            b"aaaaaaaa",
            &long,
        ];
        let expected = prepare_serial(&inputs, 7, 64, 65535)?;
        let actual = prepare_parallel(&inputs, 7, 64, 65535, &pool)?;
        assert_eq!(actual.len(), expected.len());
        for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
            let (a, b) = match (a, b) {
                (PreparedSearch::Local(a), PreparedSearch::Local(b)) => {
                    assert_eq!(a.entries, b.entries);
                    continue;
                }
                (PreparedSearch::Gpu(a), PreparedSearch::Gpu(b)) => (a, b),
                _ => return Err("different preparation policy".into()),
            };
            assert_eq!(a.input_index, index);
            assert_eq!(a.input_index, b.input_index);
            assert_eq!(a.input_len, b.input_len);
            assert_eq!(a.dispatch, b.dispatch);
            assert_eq!(a.input_layout, b.input_layout);
            assert_eq!(a.input_words, b.input_words);
            assert_eq!(a.predecessors, b.predecessors);
            assert_eq!(a.params, b.params);
        }
        assert!(prepare_parallel(&[], 7, 64, 65535, &pool)?.is_empty());
        Ok(())
    }

    #[test]
    #[ignore = "requires a Vulkan compute adapter"]
    fn partitioned_search_preserves_full_history_and_lookahead(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let searcher = GpuMatchSearcher::new(None)?;
        let actual_geometry = SearchGeometry::from_limits(&searcher.limits)?;
        let mut mixed = b"abc0123456".repeat(210);
        for position in (0..mixed.len()).step_by(119) {
            mixed[position] = b'X';
        }
        let mut long = b"abc0123456789abcdefghijk".repeat(4_600);
        long[32_505] ^= 0x80;
        long[65_010] ^= 0x40;
        for (input, max_dist, budget, part_size) in [
            (mixed.as_slice(), 64, 7, 512),
            (long.as_slice(), DEFAULT_MAX_DIST, 4, 40_000),
        ] {
            let inputs = [b"".as_slice(), b"a", input, b"abcabc", input, b"", b"ab"];
            let expected = searcher.search_batch_with_max_dist(&inputs, budget, max_dist)?;
            let geometry = SearchGeometry {
                max_part_len: part_size,
                max_group_bytes: (part_size as u64 * 4).max(actual_geometry.alignment * 2),
                alignment: actual_geometry.alignment,
            };
            let actual = searcher.search_with_geometry(
                &inputs,
                budget,
                max_dist,
                geometry,
                &prepare_serial,
            )?;
            #[cfg(feature = "parallel")]
            {
                let pool =
                    crate::BatchCodec::new(core::num::NonZeroUsize::new(4).ok_or("workers")?)?;
                let pooled = searcher.search_with_geometry(
                    &inputs,
                    budget,
                    max_dist,
                    geometry,
                    &|inputs, budget, distance, groups| {
                        prepare_parallel(inputs, budget, distance, groups, &pool)
                    },
                )?;
                assert_eq!(pooled.len(), expected.len());
                for (a, b) in pooled.iter().zip(&expected) {
                    assert!(a.iter().eq(b.iter()));
                }
            }
            assert_eq!(actual.len(), inputs.len());
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                assert!(
                    actual.iter().eq(expected.iter()),
                    "input {index}, window {max_dist}"
                );
                assert_eq!(actual.len(), inputs[index].len());
            }
        }
        Ok(())
    }

    #[test]
    fn dispatch_grid_covers_workgroup_boundaries() -> Result<(), GpuMatchError> {
        let group = WORKGROUP_SIZE;
        for (length, x, y) in [
            (1, 1, 1),
            (group, 1, 1),
            (group + 1, 2, 1),
            (group * 7, 7, 1),
            (group * 7 + 1, 7, 2),
            (group * 49, 7, 7),
        ] {
            assert_eq!(dispatch_grid(length, 7)?, (x, y));
        }
        assert!(dispatch_grid(0, 7).is_err());
        assert!(dispatch_grid(1, 0).is_err());
        assert!(dispatch_grid(group * 49 + 1, 7).is_err());
        let (x, y) = dispatch_grid(u32::MAX, u16::MAX.into())?;
        assert!(u16::try_from(x).is_ok() && u16::try_from(y).is_ok());
        assert!(x * y >= u32::MAX.div_ceil(group));
        assert!((x - 1) + (y - 1) * x < u32::MAX / group + x);
        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan and crosses the actual storage-binding limit"]
    fn search_crosses_actual_binding_limit() -> Result<(), GpuMatchError> {
        let searcher = GpuMatchSearcher::new(None)?;
        let geometry = SearchGeometry::from_limits(&searcher.limits)?;
        let length = geometry.max_part_len + 17;
        verify_periodic_search(&searcher, length)
    }

    #[test]
    #[ignore = "requires Vulkan and dispatches over eight MiB"]
    fn gpu_search_crosses_single_axis_limit() -> Result<(), GpuMatchError> {
        let searcher = GpuMatchSearcher::new(None)?;
        let limit = searcher.limits.max_compute_workgroups_per_dimension;
        let length = usize::try_from(limit)
            .map_err(|_| GpuMatchError("grid limit too large".into()))?
            * WORKGROUP_SIZE as usize
            + 17;
        verify_periodic_search(&searcher, length)
    }

    fn verify_periodic_search(
        searcher: &GpuMatchSearcher,
        length: usize,
    ) -> Result<(), GpuMatchError> {
        // Three unique cyclic triples; at budget one the nearest predecessor is at distance three.
        let input = (0..length).map(|i| b"abc"[i % 3]).collect::<Vec<_>>();
        let table = searcher.search(&input, 1)?;
        assert_eq!(table.len(), length);
        for (position, actual) in table.iter().enumerate() {
            let expected = if position < 3 || length - position < 3 {
                GpuMatch::default()
            } else {
                GpuMatch {
                    start: u32::try_from(position - 3)
                        .map_err(|_| GpuMatchError("test position overflow".into()))?,
                    length: u32::try_from((length - position).min(MAX_MATCH as usize))
                        .map_err(|_| GpuMatchError("test length overflow".into()))?,
                }
            };
            assert_eq!(actual, expected, "position {position}");
        }
        Ok(())
    }
}
