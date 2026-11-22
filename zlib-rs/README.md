A pure-rust implementation of [zlib](https://www.zlib.net/manual.html).

This is a low-level implementation crate for use in libraries like `flate2` and `rustls`. For a more high-level interface, use [`flate2`](https://crates.io/crates/flate2).

For a [zlib](https://www.zlib.net/manual.html) -compatible rust api of this crate, see [`libz-rs-sys`](https://crates.io/crates/libz-rs-sys). For instructions on integrating zlib-rs into a C library, see [`libz-rs-sys-cdylib`](https://crates.io/crates/libz-rs-sys-cdylib).

## Example

```rust
use zlib_rs::ReturnCode;
use zlib_rs::{DeflateConfig, compress_bound, compress_slice};
use zlib_rs::{InflateConfig, decompress_slice};

let input = b"Hello World";

// --- compress ---
let mut compressed_buf = vec![0u8; compress_bound(input.len())];
let (compressed, rc) =
    compress_slice(&mut compressed_buf, input, DeflateConfig::default());
assert_eq!(rc, ReturnCode::Ok);

// --- decompress ---
let mut decompressed_buf = vec![0u8; input.len()];
let (decompressed, rc) =
    decompress_slice(&mut decompressed_buf, compressed, InflateConfig::default());
assert_eq!(rc, ReturnCode::Ok);

assert_eq!(decompressed, input);
```


## Whole-stream vector destinations

With the `std` feature, `BatchJob::with_vec` reserves a bounded output
capacity without initializing the unused bytes. A completed or failed
operation replaces the vector with exactly the initialized output prefix.
Use a single job directly when no batch scheduling is needed:

```rust
# #[cfg(feature = "std")]
# fn main() -> Result<(), std::collections::TryReserveError> {
use zlib_rs::{BatchJob, DeflateConfig, InflateConfig, ReturnCode, compress_bound};

let input = b"Hello World";
let mut encoded = Vec::new();
let result = BatchJob::with_vec(input, &mut encoded, compress_bound(input.len()))
    ?
    .compress(DeflateConfig::default());
assert_eq!(result.code, ReturnCode::Ok);

let mut decoded = Vec::new();
let result = BatchJob::with_vec(&encoded, &mut decoded, input.len())
    ?
    .decompress(InflateConfig::default());
assert_eq!(result.code, ReturnCode::Ok);
assert_eq!(result.read, encoded.len());
assert_eq!(decoded, input);
# Ok(())
# }
# #[cfg(not(feature = "std"))]
# fn main() {}
```

For multiple independent streams, pass the same jobs to
`compress_slices_batch` or `decompress_slices_batch`; the optional
`BatchCodec` executes jobs concurrently on a caller-owned pool.
