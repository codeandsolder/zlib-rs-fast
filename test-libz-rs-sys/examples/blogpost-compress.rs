//! Native C ABI compression comparison for studying optimized assembly.

#[path = "shared/native_deflate.rs"]
mod native_deflate;

use native_deflate::{compress_ng, compress_rs, NativeResult};

const OUTPUT_CAPACITY: usize = 1 << 28;

fn main() -> NativeResult<()> {
    let mut args = std::env::args().skip(1);
    let level: i32 = args
        .next()
        .ok_or("usage: blogpost-compress <level> <ng|rs> <input>")?
        .parse()?;
    let backend = args.next().ok_or("missing native compressor backend")?;
    let input = std::fs::read(args.next().ok_or("missing input path")?)?;
    if args.next().is_some() {
        return Err("unexpected extra command-line arguments".into());
    }

    let mut output = vec![0u8; OUTPUT_CAPACITY];
    let produced = match backend.as_str() {
        "ng" => compress_ng(&mut output, &input, level)?,
        "rs" => compress_rs(&mut output, &input, level)?,
        _ => return Err(format!("unknown native compressor backend: {backend}").into()),
    };
    println!("compressed {produced} bytes");
    Ok(())
}
