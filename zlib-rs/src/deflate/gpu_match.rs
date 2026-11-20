#![forbid(unsafe_code)]

use std::{
    fmt,
    num::NonZeroU64,
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
@group(0) @binding(2) var<storage, read_write> results: array<vec2<u32>>;
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
        results[position] = vec2<u32>(0u, 0u);
        return;
    }

    var minimum_start = 0u;
    if position >= params.max_dist {
        // zlib's matcher uses cur_match > strstart - max_dist, not >=.
        minimum_start = position - params.max_dist + 1u;
    }

    var cursor = predecessors[position];
    let max_len = min(params.max_match, params.input_len - position);

    var best_len = 0u;
    var best_start = 0u;
    var visited = 0u;

    // The predecessor index walks the same hash bucket newest to oldest.
    // Once one entry is outside the history bound, all later entries are too.
    loop {
        if best_len == max_len || cursor == 0xffffffffu || cursor < minimum_start || visited >= params.search_budget {
            break;
        }
        let candidate = cursor;
        cursor = predecessors[candidate];
        visited += 1u;

        if best_len >= 3u && load_byte(position + best_len) != load_byte(candidate + best_len) {
            continue;
        }

        var length = 0u;
        loop {
            if length + 16u > max_len { break; }
            var difference = load_u32(position + length) ^ load_u32(candidate + length);
            if difference != 0u {
                length += countTrailingZeros(difference) >> 3u;
                break;
            }
            difference = load_u32(position + length + 4u) ^ load_u32(candidate + length + 4u);
            if difference != 0u {
                length += 4u + (countTrailingZeros(difference) >> 3u);
                break;
            }
            difference = load_u32(position + length + 8u) ^ load_u32(candidate + length + 8u);
            if difference != 0u {
                length += 8u + (countTrailingZeros(difference) >> 3u);
                break;
            }
            difference = load_u32(position + length + 12u) ^ load_u32(candidate + length + 12u);
            if difference != 0u {
                length += 12u + (countTrailingZeros(difference) >> 3u);
                break;
            }
            length += 16u;
        }
        loop {
            if length + 4u > max_len { break; }
            let difference = load_u32(position + length) ^ load_u32(candidate + length);
            if difference != 0u {
                length += countTrailingZeros(difference) >> 3u;
                break;
            }
            length += 4u;
        }
        loop {
            if length >= max_len || load_byte(position + length) != load_byte(candidate + length) {
                break;
            }
            length += 1u;
        }

        if length >= 3u && (length > best_len || (length == best_len && candidate > best_start)) {
            best_len = length;
            best_start = candidate;
        }
    }

    results[position] = vec2<u32>(best_len, best_start);
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

#[derive(Debug)]
pub struct GpuMatchTable {
    entries: Vec<GpuMatch>,
}

impl GpuMatchTable {
    #[inline]
    pub fn get(&self, absolute_position: usize) -> Option<GpuMatch> {
        self.entries.get(absolute_position).copied()
    }

    pub fn as_slice(&self) -> &[GpuMatch] {
        &self.entries
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
        u64::from(self.input_len) * 2 * size_of::<u32>() as u64
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

/// Persistent Vulkan whole-input match-search engine.
///
/// Device selection, shader compilation, and compute-pipeline creation happen once in [`Self::new`].
/// Individual searches build a direct predecessor index, upload one input, dispatch the already-compiled
/// kernel, and read back one `(length, absolute_start)` entry per input byte.
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

    /// Search several independent complete inputs in one GPU submission.
    ///
    /// All inputs use the same search budget and DEFLATE default maximum distance. Dispatches and
    /// result copies are encoded together, then all readbacks complete behind one device poll.
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
        if search_budget == 0 {
            return Err(GpuMatchError(
                "GPU match search budget must be nonzero".into(),
            ));
        }

        let mut tables = (0..inputs.len()).map(|_| None).collect::<Vec<_>>();
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let limits = self.device.limits();
        let storage_alignment = u64::from(limits.min_storage_buffer_offset_alignment.max(1));
        let uniform_alignment = u64::from(limits.min_uniform_buffer_offset_alignment.max(1));
        let max_storage_binding_size = u64::from(limits.max_storage_buffer_binding_size);
        let max_uniform_binding_size = u64::from(limits.max_uniform_buffer_binding_size);

        let mut input_cursor = 0u64;
        let mut predecessors_cursor = 0u64;
        let mut result_cursor = 0u64;
        let mut params_cursor = 0u64;
        let mut download_cursor = 0u64;
        let mut prepared = Vec::with_capacity(inputs.len());

        for (input_index, input) in inputs.iter().copied().enumerate() {
            let input_len = u32::try_from(input.len()).map_err(|_| {
                GpuMatchError("GPU match search input exceeds u32::MAX bytes".into())
            })?;
            if input.is_empty() {
                tables[input_index] = Some(GpuMatchTable {
                    entries: Vec::new(),
                });
                continue;
            }

            let dispatch = dispatch_grid(input_len, limits.max_compute_workgroups_per_dimension)?;
            let predecessors = build_predecessors(input)?;
            let input_words = pack_words(input);
            let mut item = PreparedGpuInput {
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
            };

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
            prepared.push(item);
        }

        if prepared.is_empty() {
            return Ok(tables
                .into_iter()
                .map(|table| {
                    table.ok_or_else(|| GpuMatchError("GPU empty-input result missing".into()))
                })
                .collect::<Result<Vec<_>, _>>()?);
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
                .chunks_exact(8)
                .map(|bytes| GpuMatch {
                    length: u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                    start: u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
                })
                .collect();
            tables[item.input_index] = Some(GpuMatchTable { entries });
        }
        drop(mapped);
        workspace.download.buffer.unmap();

        Ok(tables
            .into_iter()
            .enumerate()
            .map(|(index, table)| {
                table.ok_or_else(|| GpuMatchError(format!("GPU result missing for input {index}")))
            })
            .collect::<Result<Vec<_>, _>>()?)
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

/// Store the immediately preceding position in the same three-byte hash bucket.
/// A descending chain enumerates exactly the old CSR bucket prefix in reverse.
fn build_predecessors(input: &[u8]) -> Result<Vec<u32>, GpuMatchError> {
    let valid_positions = input.len().saturating_sub(2);
    let positions = u32::try_from(valid_positions)
        .map_err(|_| GpuMatchError("GPU hash index exceeds u32::MAX positions".into()))?;
    let mut heads = vec![NO_PREDECESSOR; HASH_SIZE];
    // Nonempty short inputs still need a nonzero-sized storage binding.
    let mut predecessors = Vec::with_capacity(valid_positions.max(1));
    for (position, bytes) in (0..positions).zip(input.windows(3)) {
        let hash = hash3(bytes);
        predecessors.push(core::mem::replace(&mut heads[hash], position));
    }
    if predecessors.is_empty() {
        predecessors.push(NO_PREDECESSOR);
    }
    Ok(predecessors)
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
        let predecessors = build_predecessors(&input)?;
        let mut found_collision = false;
        for position in 0..input.len() - 2 {
            let bytes = &input[position..position + 3];
            let hash = hash3(bytes);
            let expected = (0..position)
                .rev()
                .filter(|&candidate| hash3(&input[candidate..candidate + 3]) == hash)
                .collect::<Vec<_>>();
            let mut actual = Vec::new();
            let mut next = predecessors[position];
            while next != NO_PREDECESSOR {
                let candidate = next as usize;
                assert!(candidate < position);
                found_collision |= &input[candidate..candidate + 3] != bytes;
                actual.push(candidate);
                next = predecessors[candidate];
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
                assert_eq!(table.as_slice().len(), input.len());
                for (position, &actual) in table.as_slice().iter().enumerate() {
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
            assert_eq!(build_predecessors(input)?, [NO_PREDECESSOR]);
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
        // Three unique cyclic triples; at budget one the nearest predecessor is at distance three.
        let input = (0..length).map(|i| b"abc"[i % 3]).collect::<Vec<_>>();
        let table = searcher.search(&input, 1)?;
        assert_eq!(table.as_slice().len(), length);
        for (position, &actual) in table.as_slice().iter().enumerate() {
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
