#![forbid(unsafe_code)]

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
const DEFAULT_MAX_DIST: u32 = 32_768 - (258 + 3 + 1);

const SHADER: &str = r#"
struct Params {
    input_len: u32,
    max_dist: u32,
    max_match: u32,
    search_budget: u32,
}

@group(0) @binding(0) var<storage, read> input_words: array<u32>;
@group(0) @binding(1) var<storage, read> predecessors: array<u32>;
@group(0) @binding(2) var<storage, read_write> results: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

fn load_byte(index: u32) -> u32 {
    let word = input_words[index >> 2u];
    return (word >> ((index & 3u) * 8u)) & 0xffu;
}

fn load_u32(index: u32) -> u32 {
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

@compute @workgroup_size(128)
fn search(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    // Flatten rows of workgroups. Guard padded groups before multiplying so
    // position arithmetic is valid even at the u32 input-length boundary.
    let group_index = group.y * grid.x + group.x;
    if group_index > (params.input_len - 1u) / 128u { return; }
    let position = group_index * 128u + lane;
    if position >= params.input_len { return; }
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
    var distance = predecessor_distance(cursor);
    let max_len = min(params.max_match, params.input_len - position);
    var probe = 0u;
    if max_len >= 4u { probe = load_u32(position); }

    var best_len = 0u;
    var best_distance = 0u;
    var visited = 0u;

    // The predecessor index walks the same hash bucket newest to oldest.
    // Once one entry is outside the history bound, all later entries are too.
    loop {
        if best_len == max_len || distance == 0u || visited >= params.search_budget {
            break;
        }
        let candidate = cursor - distance;
        if candidate < minimum_start { break; }
        cursor = candidate;
        distance = predecessor_distance(cursor);
        visited += 1u;

        if best_len >= 3u && load_byte(position + best_len) != load_byte(candidate + best_len) {
            continue;
        }

        let length = common_length(position, candidate, max_len, probe);

        // Equal lengths keep the nearest match because predecessors are descending.
        if length >= 3u && length > best_len {
            best_len = length;
            best_distance = position - candidate;
        }
    }

    results[position] = (best_distance << 16u) | best_len;
}
"#;

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

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

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
}

type PrepareInputs<'a> =
    dyn Fn(&[&[u8]], u32, u32, u32) -> Result<Vec<PreparedGpuInput>, GpuMatchError> + 'a;

fn prepare_input(
    input_index: usize,
    input: &[u8],
    search_budget: u32,
    max_dist: u32,
    max_workgroups: u32,
) -> Result<Option<PreparedGpuInput>, GpuMatchError> {
    let input_len = u32::try_from(input.len())
        .map_err(|_| GpuMatchError("GPU match search input exceeds u32::MAX bytes".into()))?;
    if input.is_empty() {
        return Ok(None);
    }

    let dispatch = dispatch_grid(input_len, max_workgroups)?;
    let predecessors = build_predecessors(input, max_dist)?;
    let input_words = pack_words(input);
    Ok(Some(PreparedGpuInput {
        input_index,
        input_len,
        dispatch,
        input_words,
        predecessors,
        params: [input_len, max_dist, MAX_MATCH, search_budget],
        input_offset: 0,
        predecessors_offset: 0,
        result_offset: 0,
        params_offset: 0,
        download_offset: 0,
    }))
}

fn prepare_serial(
    inputs: &[&[u8]],
    search_budget: u32,
    max_dist: u32,
    max_workgroups: u32,
) -> Result<Vec<PreparedGpuInput>, GpuMatchError> {
    inputs
        .iter()
        .enumerate()
        .filter_map(|(index, input)| {
            prepare_input(index, input, search_budget, max_dist, max_workgroups).transpose()
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
) -> Result<Vec<PreparedGpuInput>, GpuMatchError> {
    let mut inputs = inputs.to_vec();
    pool.run_items(
        &mut inputs,
        || Ok(None),
        |index, input| prepare_input(index, input, search_budget, max_dist, max_workgroups),
    )
    .into_iter()
    .filter_map(Result::transpose)
    .collect()
}

struct PreparedGpuInput {
    input_index: usize,
    input_len: u32,
    dispatch: (u32, u32),
    input_words: Vec<u32>,
    predecessors: Vec<u32>,
    params: [u32; 4],
    input_offset: u64,
    predecessors_offset: u64,
    result_offset: u64,
    params_offset: u64,
    download_offset: u64,
}

impl PreparedGpuInput {
    fn input_size(&self) -> u64 {
        (self.input_words.len() * size_of::<u32>()) as u64
    }

    fn predecessors_size(&self) -> u64 {
        size_of_val(self.predecessors.as_slice()) as u64
    }

    fn result_size(&self) -> u64 {
        u64::from(self.input_len) * size_of::<u32>() as u64
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

/// Persistent Vulkan whole-input match-search engine.
///
/// Device selection, shader compilation, and compute-pipeline creation happen once in [`Self::new`].
/// Individual searches build a direct predecessor index, upload one input, dispatch the already-compiled
/// kernel, and read back one packed length/distance record per input byte.
pub struct GpuMatchSearcher {
    device: Device,
    queue: Queue,
    pipeline: ComputePipeline,
    adapter_name: String,
    workspace: Mutex<GpuWorkspace>,
}

impl GpuMatchSearcher {
    pub fn new(adapter_filter: Option<&str>) -> Result<Self, GpuMatchError> {
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

        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("zlib whole-input match search"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("whole-input match search"),
            layout: None,
            module: &module,
            entry_point: Some("search"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        let workspace = Mutex::new(GpuWorkspace::new(&device));
        Ok(Self {
            device,
            queue,
            pipeline,
            adapter_name,
            workspace,
        })
    }

    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    pub fn search(&self, input: &[u8], search_budget: u32) -> Result<GpuMatchTable, GpuMatchError> {
        self.search_with_max_dist(input, search_budget, DEFAULT_MAX_DIST)
    }

    /// Search several independent complete inputs, returning tables in input order.
    ///
    /// All inputs use the same search budget and DEFLATE default maximum distance.
    /// Dispatches share a submission when they fit the device buffers; larger workloads
    /// are partitioned while retaining exact history and lookahead at each boundary.
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

    fn search_batch_with_preparation(
        &self,
        inputs: &[&[u8]],
        search_budget: u32,
        max_dist: u32,
        prepare: &PrepareInputs<'_>,
    ) -> Result<Vec<GpuMatchTable>, GpuMatchError> {
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

        self.search_with_geometry(
            inputs,
            search_budget,
            max_dist,
            SearchGeometry::from_limits(&self.device.limits())?,
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
        let mut tables = inputs
            .iter()
            .map(|_| GpuMatchTable::default())
            .collect::<Vec<_>>();
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let limits = self.device.limits();
        let storage_alignment = u64::from(limits.min_storage_buffer_offset_alignment.max(1));
        let uniform_alignment = u64::from(limits.min_uniform_buffer_offset_alignment.max(1));
        let max_storage_binding_size = limits.max_storage_buffer_binding_size;
        let max_uniform_binding_size = limits.max_uniform_buffer_binding_size;

        let mut input_cursor = 0u64;
        let mut predecessors_cursor = 0u64;
        let mut result_cursor = 0u64;
        let mut params_cursor = 0u64;
        let mut download_cursor = 0u64;
        let mut prepared = prepare(
            inputs,
            search_budget,
            max_dist,
            limits.max_compute_workgroups_per_dimension,
        )?;

        for item in &mut prepared {
            let input_index = item.input_index;
            for (name, size) in [
                ("input", item.input_size()),
                ("predecessors", item.predecessors_size()),
                ("results", item.result_size()),
            ] {
                if size > max_storage_binding_size {
                    return Err(GpuMatchError(format!(
                        "GPU {name} binding for input {input_index} requires {size} bytes, device limit is {max_storage_binding_size}"
                    )));
                }
            }
            let params_size = size_of_val(&item.params) as u64;
            if params_size > max_uniform_binding_size {
                return Err(GpuMatchError(format!(
                    "GPU parameter binding requires {params_size} bytes, device limit is {max_uniform_binding_size}"
                )));
            }

            item.input_offset =
                allocate_aligned(&mut input_cursor, item.input_size(), storage_alignment)?;
            item.predecessors_offset = allocate_aligned(
                &mut predecessors_cursor,
                item.predecessors_size(),
                storage_alignment,
            )?;
            item.result_offset =
                allocate_aligned(&mut result_cursor, item.result_size(), storage_alignment)?;
            item.params_offset =
                allocate_aligned(&mut params_cursor, params_size, uniform_alignment)?;
            item.download_offset = download_cursor;
            download_cursor = download_cursor
                .checked_add(item.result_size())
                .ok_or_else(|| GpuMatchError("GPU download buffer layout overflowed".into()))?;
        }

        if prepared.is_empty() {
            return Ok(tables);
        }

        let mut workspace = self
            .workspace
            .lock()
            .map_err(|_| GpuMatchError("GPU match workspace lock was poisoned".into()))?;
        let max_buffer_size = limits.max_buffer_size;
        workspace
            .input
            .ensure_capacity(&self.device, input_cursor, max_buffer_size)?;
        workspace.predecessors.ensure_capacity(
            &self.device,
            predecessors_cursor,
            max_buffer_size,
        )?;
        workspace
            .results
            .ensure_capacity(&self.device, result_cursor, max_buffer_size)?;
        workspace
            .params
            .ensure_capacity(&self.device, params_cursor, max_buffer_size)?;
        workspace
            .download
            .ensure_capacity(&self.device, download_cursor, max_buffer_size)?;

        for item in &prepared {
            self.queue.write_buffer(
                &workspace.input.buffer,
                item.input_offset,
                cast_slice(&item.input_words),
            );
            self.queue.write_buffer(
                &workspace.predecessors.buffer,
                item.predecessors_offset,
                cast_slice(&item.predecessors),
            );
            self.queue.write_buffer(
                &workspace.params.buffer,
                item.params_offset,
                cast_slice(&item.params),
            );
        }

        let layout = self.pipeline.get_bind_group_layout(0);
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("whole-input match batch command encoder"),
            });

        for item in &prepared {
            let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                label: Some("whole-input packed match buffers"),
                layout: &layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: &workspace.input.buffer,
                            offset: item.input_offset,
                            size: NonZeroU64::new(item.input_size()),
                        }),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: &workspace.predecessors.buffer,
                            offset: item.predecessors_offset,
                            size: NonZeroU64::new(item.predecessors_size()),
                        }),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: &workspace.results.buffer,
                            offset: item.result_offset,
                            size: NonZeroU64::new(item.result_size()),
                        }),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: &workspace.params.buffer,
                            offset: item.params_offset,
                            size: NonZeroU64::new(size_of_val(&item.params) as u64),
                        }),
                    },
                ],
            });

            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("whole-input match search"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(item.dispatch.0, item.dispatch.1, 1);
            }
            encoder.copy_buffer_to_buffer(
                &workspace.results.buffer,
                item.result_offset,
                &workspace.download.buffer,
                item.download_offset,
                item.result_size(),
            );
        }

        self.queue.submit(Some(encoder.finish()));

        let (sender, receiver) = mpsc::sync_channel(1);
        workspace
            .download
            .buffer
            .slice(0..download_cursor)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| GpuMatchError(format!("GPU poll failed: {error}")))?;
        receiver
            .recv()
            .map_err(|_| GpuMatchError("GPU download callback was dropped".into()))?
            .map_err(|error| GpuMatchError(format!("GPU result download failed: {error}")))?;

        let mapped = workspace
            .download
            .buffer
            .slice(0..download_cursor)
            .get_mapped_range();
        for item in &prepared {
            let start = item.download_offset as usize;
            let end = start + item.result_size() as usize;
            let entries = mapped[start..end]
                .as_chunks::<4>()
                .0
                .iter()
                .copied()
                .map(u32::from_le_bytes)
                .collect();
            tables[item.input_index] = GpuMatchTable { entries };
        }
        drop(mapped);
        workspace.download.buffer.unmap();

        Ok(tables)
    }
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
    let mut heads = vec![NO_PREDECESSOR; HASH_SIZE];
    // Nonempty short inputs still need a nonzero-sized storage binding.
    let mut predecessors = vec![0; valid_positions.div_ceil(2).max(1)];
    for (position, bytes) in (0..positions).zip(input.windows(3)) {
        let previous = core::mem::replace(&mut heads[hash3(bytes)], position);
        if previous != NO_PREDECESSOR {
            let distance = position - previous;
            if distance < max_dist {
                predecessors[position as usize / 2] |= distance << ((position & 1) * 16);
            }
        }
    }
    Ok(predecessors)
}

#[cfg(test)]
fn predecessor_distance(words: &[u32], position: usize) -> usize {
    ((words[position / 2] >> ((position & 1) * 16)) & 0xffff) as usize
}

fn hash3(bytes: &[u8]) -> usize {
    let value = u32::from(bytes[0]) | (u32::from(bytes[1]) << 8) | (u32::from(bytes[2]) << 16);
    (value.wrapping_mul(2_654_435_761) >> 16) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

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
            let tables = searcher.search_batch_with_max_dist(&inputs, budget, max_dist)?;
            assert_eq!(tables.len(), inputs.len());
            for (input, table) in inputs.iter().zip(tables) {
                assert_eq!(table.len(), input.len());
                for (position, actual) in table.iter().enumerate() {
                    let mut expected = GpuMatch::default();
                    let max_len = (input.len() - position).min(MAX_MATCH as usize);
                    if max_len >= 3 {
                        let hash = hash3(&input[position..position + 3]);
                        let first = if position >= max_dist as usize {
                            position - max_dist as usize + 1
                        } else {
                            0
                        };
                        for candidate in (first..position)
                            .rev()
                            .filter(|&candidate| hash3(&input[candidate..candidate + 3]) == hash)
                            .take(budget as usize)
                        {
                            let len = input[position..position + max_len]
                                .iter()
                                .zip(&input[candidate..candidate + max_len])
                                .take_while(|(left, right)| left == right)
                                .count();
                            if len >= 3 && len > expected.length as usize {
                                expected = GpuMatch {
                                    length: len as u32,
                                    start: candidate as u32,
                                };
                            }
                        }
                    }
                    assert_eq!(
                        actual, expected,
                        "max_dist={max_dist}, budget={budget}, position={position}"
                    );
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

    #[cfg(feature = "parallel")]
    #[test]
    fn pooled_preparation_preserves_order_and_contents() -> Result<(), Box<dyn std::error::Error>> {
        let pool = crate::BatchCodec::new(core::num::NonZeroUsize::new(4).ok_or("workers")?)?;
        let inputs = [b"".as_slice(), b"a", b"abcabcabc", b"", b"abc", b"aaaaaaaa"];
        let expected = prepare_serial(&inputs, 7, 64, 65535)?;
        let actual = prepare_parallel(&inputs, 7, 64, 65535, &pool)?;
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.iter().zip(expected) {
            assert_eq!(a.input_index, b.input_index);
            assert_eq!(a.input_len, b.input_len);
            assert_eq!(a.dispatch, b.dispatch);
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
        let actual_geometry = SearchGeometry::from_limits(&searcher.device.limits())?;
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
        assert!(x <= u32::from(u16::MAX) && y <= u32::from(u16::MAX));
        assert!(x * y >= u32::MAX.div_ceil(group));
        assert!((x - 1) + (y - 1) * x < u32::MAX / group + x);
        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan and crosses the actual storage-binding limit"]
    fn search_crosses_actual_binding_limit() -> Result<(), GpuMatchError> {
        let searcher = GpuMatchSearcher::new(None)?;
        let geometry = SearchGeometry::from_limits(&searcher.device.limits())?;
        let length = geometry.max_part_len + 17;
        verify_periodic_search(&searcher, length)
    }

    #[test]
    #[ignore = "requires Vulkan and dispatches over eight MiB"]
    fn gpu_search_crosses_single_axis_limit() -> Result<(), GpuMatchError> {
        let searcher = GpuMatchSearcher::new(None)?;
        let limit = searcher
            .device
            .limits()
            .max_compute_workgroups_per_dimension;
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
