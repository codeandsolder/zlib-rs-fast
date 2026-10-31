#![forbid(unsafe_code)]

use std::{fmt, sync::mpsc};

use bytemuck::cast_slice;
use wgpu::{
    util::DeviceExt, Adapter, Backends, BindGroupDescriptor, BindGroupEntry, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    Device, DeviceDescriptor, DeviceType, ExperimentalFeatures, Features, Instance,
    InstanceDescriptor, Limits, MemoryHints, PipelineCompilationOptions, Queue,
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

        Ok(Self {
            device,
            queue,
            pipeline,
            adapter_name,
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

        let layout = self.pipeline.get_bind_group_layout(0);
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("whole-input match batch command encoder"),
            });
        let mut downloads = Vec::with_capacity(inputs.len());

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
            let params = [input_len, max_dist, MAX_MATCH, search_budget];
            let result_size = u64::from(input_len) * 8;

            let input_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("match input"),
                    contents: cast_slice(&input_words),
                    usage: BufferUsages::STORAGE,
                });
            let offsets_buffer =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("match bucket offsets"),
                        contents: cast_slice(&index.offsets),
                        usage: BufferUsages::STORAGE,
                    });
            let positions_buffer =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("match bucket positions"),
                        contents: cast_slice(&index.positions),
                        usage: BufferUsages::STORAGE,
                    });
            let result_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("match results"),
                size: result_size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let params_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("match params"),
                    contents: cast_slice(&params),
                    usage: BufferUsages::UNIFORM,
                });

            let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                label: Some("whole-input match buffers"),
                layout: &layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: input_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: offsets_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: positions_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: result_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 4,
                        resource: params_buffer.as_entire_binding(),
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
                pass.dispatch_workgroups(input_len.div_ceil(WORKGROUP_SIZE), 1, 1);
            }

            let download = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("match result download"),
                size: result_size,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(&result_buffer, 0, &download, 0, result_size);
            downloads.push((input_index, input.len(), download));
        }

        if downloads.is_empty() {
            return Ok(tables
                .into_iter()
                .map(|table| table.expect("empty inputs receive empty match tables"))
                .collect());
        }

        self.queue.submit(Some(encoder.finish()));

        let (sender, receiver) = mpsc::channel();
        for (input_index, _, download) in &downloads {
            let sender = sender.clone();
            let input_index = *input_index;
            download
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    let _ = sender.send((input_index, result));
                });
        }
        drop(sender);

        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| GpuMatchError(format!("GPU poll failed: {error}")))?;

        for _ in 0..downloads.len() {
            let (input_index, result) = receiver
                .recv()
                .map_err(|_| GpuMatchError("GPU download callback was dropped".into()))?;
            result.map_err(|error| {
                GpuMatchError(format!(
                    "GPU result download for input {input_index} failed: {error}"
                ))
            })?;
        }

        for (input_index, input_len, download) in downloads {
            let mapped = download.slice(..).get_mapped_range();
            let bytes: &[u8] = &mapped;
            if bytes.len() != input_len * 8 {
                return Err(GpuMatchError(format!(
                    "GPU result buffer for input {input_index} has an unexpected length"
                )));
            }
            let entries = bytes
                .chunks_exact(8)
                .map(|bytes| GpuMatch {
                    length: u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                    start: u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
                })
                .collect();
            drop(mapped);
            download.unmap();
            tables[input_index] = Some(GpuMatchTable { entries });
        }

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
