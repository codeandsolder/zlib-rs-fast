use super::{
    GpuBatch, GpuMatchError, GpuMatchTable, InputLayout, PreparedGpuInput, MAX_MATCH,
    WORKGROUP_SIZE,
};
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};
use std::sync::{Arc, Mutex};

pub(super) struct CudaBackend {
    stream: Arc<CudaStream>,
    packed: CudaFunction,
    aligned: CudaFunction,
    workspace: Mutex<Workspace>,
}

struct Workspace {
    input: CudaSlice<u32>,
    predecessors: CudaSlice<u32>,
    results: CudaSlice<u32>,
}

fn failure(error: impl core::fmt::Debug) -> GpuMatchError {
    GpuMatchError(format!("CUDA match search: {error:?}"))
}

fn reserve(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<u32>,
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
    pub(super) fn new(ordinal: usize) -> Result<(Self, String), GpuMatchError> {
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
        let adapter_name = ctx.name().map_err(failure)?;
        let stream = ctx.default_stream();
        let (major, minor) = ctx.compute_capability().map_err(failure)?;
        let source = format!(
            "#define MATCH_LIMIT {MAX_MATCH}u\n#define GROUP_SIZE {WORKGROUP_SIZE}u\n{}",
            include_str!("search.cu")
        );
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
        let workspace = Mutex::new(Workspace {
            input: stream.alloc_zeros(1).map_err(failure)?,
            predecessors: stream.alloc_zeros(1).map_err(failure)?,
            results: stream.alloc_zeros(1).map_err(failure)?,
        });
        Ok((
            Self {
                stream,
                packed,
                aligned,
                workspace,
            },
            adapter_name,
        ))
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
        let function = match input.input_layout {
            InputLayout::Packed => &self.packed,
            InputLayout::Aligned => &self.aligned,
        };
        let [size, distance, cooperative, budget] = input.params;
        let mut launch = self.stream.launch_builder(function);
        launch.arg(&workspace.input);
        launch.arg(&workspace.predecessors);
        launch.arg(&mut workspace.results);
        launch.arg(&size);
        launch.arg(&distance);
        launch.arg(&cooperative);
        launch.arg(&budget);
        let geometry = LaunchConfig {
            grid_dim: (size.div_ceil(WORKGROUP_SIZE), 1, 1),
            block_dim: (WORKGROUP_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };
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
