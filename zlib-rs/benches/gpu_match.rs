use std::{env, hint::black_box, sync::mpsc, time::Instant};

use bytemuck::cast_slice;
use wgpu::{
    util::{DeviceExt, DownloadBuffer},
    Adapter, Backends, BindGroupDescriptor, BindGroupEntry, BufferUsages, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipelineDescriptor, DeviceDescriptor, DeviceType,
    ExperimentalFeatures, Features, Instance, InstanceDescriptor, Limits, MemoryHints,
    PipelineCompilationOptions, ShaderModuleDescriptor, Trace,
};

const HASH_SIZE: usize = 1 << 16;
const WORKGROUP_SIZE: u32 = 128;
const MAX_MATCH: u32 = 258;
const MAX_DIST: u32 = 32_768 - (258 + 3 + 1);
const DEFAULT_SEARCH_BUDGET: u32 = 4_096;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Match {
    len: u32,
    start: u32,
}

#[derive(Clone, Copy, Debug)]
enum Corpus {
    Structured,
    Repetitive,
    PseudoRandom,
}

struct HashIndex {
    offsets: Vec<u32>,
    positions: Vec<u32>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = env::args().collect::<Vec<_>>();
    let bytes = option_value(&args, "--bytes=").map_or(Ok(64 * 1024usize), str::parse)?;
    let search_budget =
        option_value(&args, "--budget=").map_or(Ok(DEFAULT_SEARCH_BUDGET), str::parse)?;
    assert!(search_budget != 0, "--budget must be nonzero");
    let corpus = match option_value(&args, "--corpus=").unwrap_or("structured") {
        "structured" => Corpus::Structured,
        "repetitive" => Corpus::Repetitive,
        "random" => Corpus::PseudoRandom,
        other => return Err(format!("unknown corpus {other:?}").into()),
    };
    let adapter_filter = option_value(&args, "--adapter=");

    if bytes < 3 {
        return Err("--bytes must be at least 3".into());
    }
    let _: u32 = bytes.try_into()?;

    let input = make_input(corpus, bytes);
    let index = timed("cpu CSR index", || build_hash_index(&input));
    let expected = timed("cpu match reference", || {
        match_reference(&input, &index, search_budget)
    });

    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::VULKAN,
        ..InstanceDescriptor::new_without_display_handle()
    });
    let adapter = select_adapter(&instance, adapter_filter)?;
    eprintln!("selected adapter: {:?}", adapter.get_info());
    let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
        required_features: Features::default(),
        required_limits: Limits::default(),
        memory_hints: MemoryHints::default(),
        experimental_features: ExperimentalFeatures::disabled(),
        trace: Trace::default(),
        label: Some("zlib-rs gpu match prototype"),
    }))?;

    let started = Instant::now();
    let actual = gpu_match_table(&device, &queue, &input, &index, search_budget)?;
    let gpu_elapsed = started.elapsed();
    assert_eq!(
        actual, expected,
        "GPU match table differs from CPU reference"
    );

    let matches = actual.iter().filter(|item| item.len >= 3).count();
    let matched_bytes: u64 = actual.iter().map(|item| u64::from(item.len)).sum();
    println!(
        "validated {} {:?} bytes at budget {}: {} matches, {} total matched bytes; GPU setup+dispatch+readback {:?}",
        input.len(), corpus, search_budget, matches, matched_bytes, gpu_elapsed
    );
    black_box(actual);
    Ok(())
}

fn option_value<'a>(args: &'a [String], prefix: &str) -> Option<&'a str> {
    args.iter().find_map(|arg| arg.strip_prefix(prefix))
}

fn select_adapter(
    instance: &Instance,
    filter: Option<&str>,
) -> Result<Adapter, Box<dyn std::error::Error>> {
    let adapters = pollster::block_on(instance.enumerate_adapters(Backends::VULKAN));
    for adapter in &adapters {
        eprintln!("available adapter: {:?}", adapter.get_info());
    }

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
        format!(
            "no Vulkan adapter matched {}",
            filter.unwrap_or("the default selection")
        )
        .into()
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

fn timed<T>(label: &str, run: impl FnOnce() -> T) -> T {
    let started = Instant::now();
    let result = run();
    eprintln!("{label}: {:?}", started.elapsed());
    result
}

fn gpu_match_table(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    input: &[u8],
    index: &HashIndex,
    search_budget: u32,
) -> Result<Vec<Match>, Box<dyn std::error::Error>> {
    let input_words = pack_words(input);
    let result_words = vec![[0u32; 2]; input.len()];
    let params = [input.len() as u32, MAX_DIST, MAX_MATCH, search_budget];

    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("match input"),
        contents: cast_slice(&input_words),
        usage: BufferUsages::STORAGE,
    });
    let offsets_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("match bucket offsets"),
        contents: cast_slice(&index.offsets),
        usage: BufferUsages::STORAGE,
    });
    let positions_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("match bucket positions"),
        contents: cast_slice(&index.positions),
        usage: BufferUsages::STORAGE,
    });
    let result_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("match results"),
        contents: cast_slice(&result_words),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("match params"),
        contents: cast_slice(&params),
        usage: BufferUsages::UNIFORM,
    });

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
    let layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
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

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("whole-input match command encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("whole-input match search"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((input.len() as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
    }
    queue.submit(Some(encoder.finish()));

    let (sender, receiver) = mpsc::sync_channel(1);
    DownloadBuffer::read_buffer(device, queue, &result_buffer.slice(..), move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::PollType::wait_indefinitely())?;
    let download = receiver
        .recv()
        .map_err(|_| "GPU download callback was dropped")??;
    let bytes: &[u8] = &download;
    if bytes.len() != input.len() * 8 {
        return Err("GPU result buffer has an unexpected length".into());
    }

    Ok(bytes
        .chunks_exact(8)
        .map(|bytes| Match {
            len: u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            start: u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        })
        .collect())
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

fn build_hash_index(input: &[u8]) -> HashIndex {
    let valid_positions = input.len().saturating_sub(2);
    let mut counts = vec![0u32; HASH_SIZE];
    for position in 0..valid_positions {
        counts[hash3(input, position) as usize] += 1;
    }

    let mut offsets = vec![0u32; HASH_SIZE + 1];
    for (index, count) in counts.into_iter().enumerate() {
        offsets[index + 1] = offsets[index] + count;
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
    HashIndex { offsets, positions }
}

fn positions_by_bucket_are_sorted(offsets: &[u32], positions: &[u32]) -> bool {
    offsets.windows(2).all(|bounds| {
        positions[bounds[0] as usize..bounds[1] as usize]
            .windows(2)
            .all(|pair| pair[0] < pair[1])
    })
}

fn match_reference(input: &[u8], index: &HashIndex, search_budget: u32) -> Vec<Match> {
    let mut results = vec![Match { len: 0, start: 0 }; input.len()];
    for position in 0..input.len().saturating_sub(2) {
        let hash = hash3(input, position) as usize;
        let bucket =
            &index.positions[index.offsets[hash] as usize..index.offsets[hash + 1] as usize];
        let minimum_start = if position >= MAX_DIST as usize {
            position - MAX_DIST as usize + 1
        } else {
            0
        };
        let first = bucket.partition_point(|&candidate| candidate < minimum_start as u32);
        let end = bucket.partition_point(|&candidate| candidate < position as u32);
        let max_len = (input.len() - position).min(MAX_MATCH as usize);
        let mut best = Match { len: 0, start: 0 };

        for &candidate in bucket[first..end].iter().rev().take(search_budget as usize) {
            let candidate = candidate as usize;
            let len = input[position..position + max_len]
                .iter()
                .zip(&input[candidate..candidate + max_len])
                .position(|(left, right)| left != right)
                .unwrap_or(max_len);
            if len >= 3
                && (len > best.len as usize
                    || (len == best.len as usize && candidate > best.start as usize))
            {
                best = Match {
                    len: len as u32,
                    start: candidate as u32,
                };
            }
        }
        results[position] = best;
    }
    results
}

fn hash3(input: &[u8], position: usize) -> u32 {
    let value = u32::from(input[position])
        | (u32::from(input[position + 1]) << 8)
        | (u32::from(input[position + 2]) << 16);
    value.wrapping_mul(2_654_435_761) >> 16
}

fn make_input(corpus: Corpus, len: usize) -> Vec<u8> {
    match corpus {
        Corpus::Repetitive => {
            const PATTERN: &[u8] = b"zlib-rs-gpu-match/";
            (0..len)
                .map(|index| PATTERN[index % PATTERN.len()])
                .collect()
        }
        Corpus::Structured => {
            let mut out = Vec::with_capacity(len);
            let mut row = 0u64;
            while out.len() < len {
                let line = format!(
                    "{{\"row\":{},\"sensor\":{},\"value\":{},\"status\":\"{}\"}}\n",
                    row,
                    row % 37,
                    (row.wrapping_mul(2_654_435_761) >> 11) % 100_000,
                    if row % 11 == 0 { "warn" } else { "ok" }
                );
                let take = (len - out.len()).min(line.len());
                out.extend_from_slice(&line.as_bytes()[..take]);
                row = row.wrapping_add(1);
            }
            out
        }
        Corpus::PseudoRandom => {
            let mut state = 0x1234_5678_9abc_def0u64;
            (0..len)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    state as u8
                })
                .collect()
        }
    }
}
