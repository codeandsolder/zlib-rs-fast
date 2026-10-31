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
@group(0) @binding(1) var<storage, read> bucket_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> bucket_positions: array<u32>;
@group(0) @binding(3) var<storage, read_write> results: array<vec2<u32>>;
@group(0) @binding(4) var<uniform> params: Params;

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

fn hash_at(position: u32) -> u32 {
    let value = load_byte(position)
        | (load_byte(position + 1u) << 8u)
        | (load_byte(position + 2u) << 16u);
    return (value * 2654435761u) >> 16u;
}

fn lower_bound(begin: u32, end: u32, value: u32) -> u32 {
    var low = begin;
    var high = end;
    loop {
        if low >= high { break; }
        let middle = low + ((high - low) >> 1u);
        if bucket_positions[middle] < value {
            low = middle + 1u;
        } else {
            high = middle;
        }
    }
    return low;
}

@compute @workgroup_size(128)
fn search(@builtin(global_invocation_id) gid: vec3<u32>) {
    let position = gid.x;
    if position >= params.input_len { return; }
    if position + 2u >= params.input_len {
        results[position] = vec2<u32>(0u, 0u);
        return;
    }

    let hash = hash_at(position);
    let bucket_begin = bucket_offsets[hash];
    let bucket_end = bucket_offsets[hash + 1u];

    var minimum_start = 0u;
    if position >= params.max_dist {
        // zlib's matcher uses cur_match > strstart - max_dist, not >=.
        minimum_start = position - params.max_dist + 1u;
    }

    let first = lower_bound(bucket_begin, bucket_end, minimum_start);
    var cursor = lower_bound(first, bucket_end, position);
    let max_len = min(params.max_match, params.input_len - position);

    var best_len = 0u;
    var best_start = 0u;
    var visited = 0u;

    // Buckets are sorted by input position. Search newest candidates first so a finite budget
    // spends work on the closest history, and tie-break deterministically toward the newest match.
    loop {
        if cursor <= first || visited >= params.search_budget { break; }
        cursor -= 1u;
        visited += 1u;
        let candidate = bucket_positions[cursor];

        var length = 0u;
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

struct HashIndex {
    offsets: Vec<u32>,
    positions: Vec<u32>,
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
    offsets: ReusableBuffer,
    positions: ReusableBuffer,
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
            offsets: ReusableBuffer::new(
                device,
                "match bucket-offset workspace",
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
            ),
            positions: ReusableBuffer::new(
                device,
                "match bucket-position workspace",
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
    input_words: Vec<u32>,
    index: HashIndex,
    params: [u32; 4],
    input_offset: u64,
    offsets_offset: u64,
    positions_offset: u64,
    result_offset: u64,
    params_offset: u64,
    download_offset: u64,
}

impl PreparedGpuInput {
    fn input_size(&self) -> u64 {
        (self.input_words.len() * size_of::<u32>()) as u64
    }

    fn offsets_size(&self) -> u64 {
        (self.index.offsets.len() * size_of::<u32>()) as u64
    }

    fn positions_data_size(&self) -> u64 {
        (self.index.positions.len() * size_of::<u32>()) as u64
    }

    fn positions_binding_size(&self) -> u64 {
        self.positions_data_size().max(size_of::<u32>() as u64)
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

/// Persistent Vulkan whole-input match-search engine.
///
/// Device selection, shader compilation, and compute-pipeline creation happen once in [`Self::new`].
/// Individual searches build a compact CPU CSR index, upload one input, dispatch the already-compiled
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
        Ok(tables
            .pop()
            .expect("one GPU match-search input produces one table"))
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
        let mut offsets_cursor = 0u64;
        let mut positions_cursor = 0u64;
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

            let index = build_hash_index(input)?;
            let input_words = pack_words(input);
            let mut item = PreparedGpuInput {
                input_index,
                input_len,
                input_words,
                index,
                params: [input_len, max_dist, MAX_MATCH, search_budget],
                input_offset: 0,
                offsets_offset: 0,
                positions_offset: 0,
                result_offset: 0,
                params_offset: 0,
                download_offset: 0,
            };

            for (name, size) in [
                ("input", item.input_size()),
                ("bucket offsets", item.offsets_size()),
                ("bucket positions", item.positions_binding_size()),
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
            item.offsets_offset =
                allocate_aligned(&mut offsets_cursor, item.offsets_size(), storage_alignment)?;
            item.positions_offset = allocate_aligned(
                &mut positions_cursor,
                item.positions_binding_size(),
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
                .map(|table| table.expect("empty inputs receive empty match tables"))
                .collect());
        }

        let mut workspace = self
            .workspace
            .lock()
            .map_err(|_| GpuMatchError("GPU match workspace lock was poisoned".into()))?;
        let max_buffer_size = limits.max_buffer_size;
        workspace
            .input
            .ensure_capacity(&self.device, input_cursor, max_buffer_size)?;
        workspace
            .offsets
            .ensure_capacity(&self.device, offsets_cursor, max_buffer_size)?;
        workspace
            .positions
            .ensure_capacity(&self.device, positions_cursor, max_buffer_size)?;
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
                &workspace.offsets.buffer,
                item.offsets_offset,
                cast_slice(&item.index.offsets),
            );
            if !item.index.positions.is_empty() {
                self.queue.write_buffer(
                    &workspace.positions.buffer,
                    item.positions_offset,
                    cast_slice(&item.index.positions),
                );
            }
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
                            buffer: &workspace.offsets.buffer,
                            offset: item.offsets_offset,
                            size: NonZeroU64::new(item.offsets_size()),
                        }),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: &workspace.positions.buffer,
                            offset: item.positions_offset,
                            size: NonZeroU64::new(item.positions_binding_size()),
                        }),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: &workspace.results.buffer,
                            offset: item.result_offset,
                            size: NonZeroU64::new(item.result_size()),
                        }),
                    },
                    BindGroupEntry {
                        binding: 4,
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
                pass.dispatch_workgroups(item.input_len.div_ceil(WORKGROUP_SIZE), 1, 1);
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
            .map(|table| table.expect("every GPU input receives a match table"))
            .collect())
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

fn build_hash_index(input: &[u8]) -> Result<HashIndex, GpuMatchError> {
    let valid_positions = input.len().saturating_sub(2);
    let _ = u32::try_from(valid_positions)
        .map_err(|_| GpuMatchError("GPU hash index exceeds u32::MAX positions".into()))?;
    let mut counts = vec![0u32; HASH_SIZE];
    for position in 0..valid_positions {
        counts[hash3(input, position) as usize] += 1;
    }

    let mut offsets = vec![0u32; HASH_SIZE + 1];
    for (index, count) in counts.into_iter().enumerate() {
        offsets[index + 1] = offsets[index]
            .checked_add(count)
            .ok_or_else(|| GpuMatchError("GPU hash index offset overflow".into()))?;
    }

    let mut cursors = offsets[..HASH_SIZE].to_vec();
    let mut positions = vec![0u32; valid_positions];
    for position in 0..valid_positions {
        let hash = hash3(input, position) as usize;
        let cursor = &mut cursors[hash];
        positions[*cursor as usize] = position as u32;
        *cursor += 1;
    }
    debug_assert!(positions_by_bucket_are_sorted(&offsets, &positions));
    Ok(HashIndex { offsets, positions })
}

fn positions_by_bucket_are_sorted(offsets: &[u32], positions: &[u32]) -> bool {
    offsets.windows(2).all(|bounds| {
        positions[bounds[0] as usize..bounds[1] as usize]
            .windows(2)
            .all(|pair| pair[0] < pair[1])
    })
}

fn hash3(input: &[u8], position: usize) -> u32 {
    let value = u32::from(input[position])
        | (u32::from(input[position + 1]) << 8)
        | (u32::from(input[position + 2]) << 16);
    value.wrapping_mul(2_654_435_761) >> 16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csr_buckets_preserve_input_order() {
        let input = b"abcabcabc/abcabcabc/xyzxyzxyz";
        let index = build_hash_index(input).unwrap();
        assert!(positions_by_bucket_are_sorted(
            &index.offsets,
            &index.positions
        ));
        assert_eq!(index.positions.len(), input.len() - 2);
    }
}
