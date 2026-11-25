mod resident;

use super::{
    GpuBatch, GpuMatchError, GpuMatchTable, InputLayout, PreparedGpuInput, MAX_MATCH,
    WORKGROUP_SIZE,
};
use cudarc::{
    driver::{
        CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg,
        ValidAsZeroBits,
    },
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};
pub(in crate::deflate) use resident::DEFAULT_WINDOW;
use std::sync::{Arc, Mutex};

pub(super) struct CudaBackend {
    stream: Arc<CudaStream>,
    packed: CudaFunction,
    aligned: CudaFunction,
    expand: CudaFunction,
    workspace: Mutex<Workspace>,
    resident: Option<Box<resident::ResidentPool>>,
}

struct Workspace {
    input: CudaSlice<u32>,
    aligned_input: CudaSlice<u32>,
    predecessors: CudaSlice<u32>,
    results: CudaSlice<u32>,
}

fn failure(error: impl core::fmt::Debug) -> GpuMatchError {
    GpuMatchError(format!("CUDA match search: {error:?}"))
}

fn reserve<T: DeviceRepr + ValidAsZeroBits>(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<T>,
    len: usize,
) -> Result<(), GpuMatchError> {
    if buffer.len() < len {
        *buffer = stream
            .alloc_zeros(len.next_power_of_two())
            .map_err(failure)?;
    }
    Ok(())
}

impl CudaBackend {
    #[expect(
        unsafe_code,
        reason = "cudarc requires unsafe dynamic library availability probes"
    )]
    pub(super) fn new(ordinal: usize, use_resident: bool) -> Result<(Self, String), GpuMatchError> {
        // SAFETY: these probes only load and immediately close the named
        // CUDA driver/NVRTC libraries; no device pointers or kernels are used.
        if !unsafe {
            cudarc::driver::sys::is_culib_present() && cudarc::nvrtc::sys::is_culib_present()
        } {
            return Err(GpuMatchError(
                "CUDA driver or NVRTC runtime library not found".into(),
            ));
        }
        let ctx = CudaContext::new(ordinal).map_err(failure)?;
        // SAFETY: all device allocations in this private context are
        // exclusively owned by one workspace and never cross streams.
        // The dense matcher always uses its own default stream. Each
        // resident lane allocates, uses and releases its buffers on its
        // own nonblocking stream, under the lane lease. Stream ordering
        // covers allocation, reuse and free; no inter-stream buffer
        // dependencies exist. Avoid redundant events on every argument.
        unsafe { ctx.disable_event_tracking() };

        let adapter_name = ctx.name().map_err(failure)?;
        let stream = ctx.default_stream();
        let (major, minor) = ctx.compute_capability().map_err(failure)?;
        let mut source = format!(
            "#define MATCH_LIMIT {MAX_MATCH}u\n#define GROUP_SIZE {WORKGROUP_SIZE}u\n{}",
            include_str!("search.cu")
        );
        if use_resident {
            source.push_str(include_str!("resident.cu"));
            source.push_str(include_str!("resident_parallel.cu"));
        }
        let ptx = compile_ptx_with_opts(
            source,
            CompileOptions {
                options: vec![
                    "--std=c++17".into(),
                    format!("--gpu-architecture=compute_{major}{minor}"),
                ],
                ..CompileOptions::default()
            },
        )
        .map_err(failure)?;
        let module = ctx.load_module(ptx).map_err(failure)?;
        let packed = module.load_function("search_packed").map_err(failure)?;
        let aligned = module.load_function("search_aligned").map_err(failure)?;
        let expand = module.load_function("expand_aligned").map_err(failure)?;
        let workspace = Mutex::new(Workspace {
            input: stream.alloc_zeros(1).map_err(failure)?,
            aligned_input: stream.alloc_zeros(1).map_err(failure)?,
            predecessors: stream.alloc_zeros(1).map_err(failure)?,
            results: stream.alloc_zeros(1).map_err(failure)?,
        });
        let resident = use_resident
            .then(|| resident::ResidentPool::new(
                &ctx, &module,
                u32::try_from(ctx.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT).map_err(failure)?).map_err(failure)?,
            ).map(Box::new))
            .transpose()?;
        Ok((
            Self {
                stream,
                packed,
                aligned,
                expand,
                workspace,
                resident,
            },
            adapter_name,
        ))
    }

    pub(super) fn supports_resident(&self, config: crate::DeflateConfig) -> bool {
        self.resident.is_some() && resident::supports(config)
    }

    pub(super) fn compress_resident(
        &self,
        input: &[u8],
        limit: usize,
        config: crate::DeflateConfig,
        budget: u32,
        window: core::num::NonZeroUsize,
        distance: u32,
    ) -> Result<(Vec<u8>, crate::ReturnCode), GpuMatchError> {
        self.resident
            .as_ref()
            .ok_or_else(|| GpuMatchError("resident CUDA encoder not selected".into()))?
            .compress(input, limit, config, budget, window, distance)
    }

    pub(super) fn execute(&self, batch: &mut GpuBatch) -> Result<(), GpuMatchError> {
        let mut workspace = self.workspace.lock().map_err(failure)?;
        for dispatch in &batch.dispatches {
            batch.tables[dispatch.input.input_index] =
                self.search(&mut workspace, &dispatch.input)?;
        }
        Ok(())
    }

    #[expect(
        unsafe_code,
        reason = "CUDA kernel launch requires explicit host-side buffer and ABI proof"
    )]
    fn search(
        &self,
        workspace: &mut Workspace,
        input: &PreparedGpuInput,
    ) -> Result<GpuMatchTable, GpuMatchError> {
        let len = input.input_len as usize;
        reserve(&self.stream, &mut workspace.input, input.input_words.len())?;
        reserve(
            &self.stream,
            &mut workspace.predecessors,
            input.predecessors.len(),
        )?;
        reserve(&self.stream, &mut workspace.results, len)?;
        self.stream
            .memcpy_htod(&input.input_words, &mut workspace.input)
            .map_err(failure)?;
        self.stream
            .memcpy_htod(&input.predecessors, &mut workspace.predecessors)
            .map_err(failure)?;
        let [size, distance, cooperative, budget] = input.params;
        let geometry = LaunchConfig {
            grid_dim: (size.div_ceil(WORKGROUP_SIZE), 1, 1),
            block_dim: (WORKGROUP_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };
        let (function, input_buffer) = match input.input_layout {
            InputLayout::Packed => (&self.packed, &workspace.input),
            InputLayout::Aligned => {
                reserve(&self.stream, &mut workspace.aligned_input, len)?;
                let mut expand = self.stream.launch_builder(&self.expand);
                expand.arg(&workspace.input);
                expand.arg(&mut workspace.aligned_input);
                expand.arg(&size);
                // SAFETY: packed input has ceil(size/4) words; the expansion
                // writes one aligned word for each position, padding its tail.
                unsafe { expand.launch(geometry) }.map_err(failure)?;
                (&self.aligned, &workspace.aligned_input)
            }
        };
        let mut launch = self.stream.launch_builder(function);
        launch.arg(input_buffer);
        launch.arg(&workspace.predecessors);
        launch.arg(&mut workspace.results);
        launch.arg(&size);
        launch.arg(&distance);
        launch.arg(&cooperative);
        launch.arg(&budget);
        // SAFETY: the two functions have the same seven argument ABI.
        // PreparedGpuInput owns the correctly padded input and predecessor
        // arrays; buffers are at least their sizes. One lane writes each
        // result position below size; all reads are bounded by size and
        // the predecessor/window contract. The stream orders upload,
        // launch and download and the workspace lock excludes reuse.
        unsafe { launch.launch(geometry) }.map_err(failure)?;
        let entries = self
            .stream
            .clone_dtoh(&workspace.results.slice(..len))
            .map_err(failure)?;
        Ok(GpuMatchTable { entries })
    }
}
