# Benchmarking this fork

The upstream automated benchmark workflow was bound to the upstream project's
dedicated Linux x86, Apple Silicon and LoongArch hosts and result repository.
Those runners are not registered here. This fork uses the in-tree benchmark
harness on explicitly selected, controlled machines instead.

Run from the exact source checkout with the repository's pinned Rust toolchain
and the existing managed sccache wrapper:

```sh
export RUSTC_WRAPPER=/usr/local/libexec/sccache-client
export CARGO_INCREMENTAL=0
cargo bench --locked -p zlib-rs --bench single
cargo bench --locked -p zlib-rs --bench batch --features parallel
```

Select the same hardware, CPU/power policy, compiler, feature set and workload
for before/after comparisons. Record the commit, CPU model, Rust version,
compiler flags and unmodified benchmark output with the result. Use the
single-stream harness's `--full`, `--only=...` or `--sweep` options as
appropriate. Do not publish measurements to the upstream benchmark repository.
Routine correctness CI is independent of performance measurement.

## Isolated real-file profiles

Export the standalone profiler to a directory for this measurement run.
This also works with ephemeral Cargo build directories:

```sh
cargo build --locked --release -p zlib-rs --example profile_file \
  -Z unstable-options --artifact-dir ./profile-artifacts
```

Use the exported executable `./profile-artifacts/profile_file`.
The file profiler uses explicit positional arguments. Prepare an input
stream once, outside the profiled process, then use the same compressed
stream and reference bytes for every decoder comparison:

```sh
./profile-artifacts/profile_file prepare raw.bin 6 prepared.zlib
perf stat -e cycles,instructions,branches,branch-misses -- ./profile-artifacts/profile_file decode-slice prepared.zlib raw.bin 100
perf stat -e cycles,instructions,branches,branch-misses -- ./profile-artifacts/profile_file decode-vector prepared.zlib raw.bin 100
perf stat -e cycles,instructions,branches,branch-misses -- ./profile-artifacts/profile_file encode raw.bin 9 100
```

For near-optimal compression, build the same example with
`--features near-optimal` and select a positive pass count explicitly:

```sh
cargo build --locked --release -p zlib-rs --example profile_file --features near-optimal \
  -Z unstable-options --artifact-dir ./profile-artifacts
./profile-artifacts/profile_file prepare raw.bin near-optimal:3 prepared-optimal.zlib
perf stat -e cycles,instructions,branches,branch-misses -- ./profile-artifacts/profile_file encode raw.bin near-optimal:3 10
```

The near-optimal method does not run a stock level-9 encode in the profiled
process. Decode either prepared stream with the same decoder command.

`prepare` creates a new file and refuses to overwrite an existing one.
The decoder modes allocate the bounded destination before their repeated
calls, verify consumed input and produced output lengths, and check the
decoded bytes against the supplied reference. The slice mode also checks
untouched tail bytes. Both use direct independent jobs, without a result
vector allocation or batch scheduling. The printed elapsed time covers
the repeated codec calls; process-level perf counters additionally include
startup, file reading and final validation, so use enough iterations to
amortize that cost. Decode processes do not encode data. For encoding,
the final round-trip check occurs outside the reported elapsed time.
Run `./profile-artifacts/profile_file --self-test` to check the profiler's argument and output invariants. Remove the explicitly exported directory
after collecting measurements.
