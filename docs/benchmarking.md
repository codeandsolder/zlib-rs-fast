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
