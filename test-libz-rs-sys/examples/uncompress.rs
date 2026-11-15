//! Native zlib-rs and zlib-ng one-shot decompression comparison.
//! Caller chooses the input and output paths explicitly.

#[path = "shared/native_uncompress.rs"]
mod native_uncompress;

use native_uncompress::{uncompress_ng, uncompress_rs, NativeResult};

fn main() -> NativeResult<()> {
    let mut args = std::env::args().skip(1);
    let backend = args
        .next()
        .ok_or("usage: uncompress <ng|rs> <input.zlib> <output>")?;
    let input_path = args.next().ok_or("missing compressed input path")?;
    let output_path = args.next().ok_or("missing decompressed output path")?;
    if args.next().is_some() {
        return Err("unexpected additional command-line arguments".into());
    }

    let input = std::fs::read(input_path)?;
    let mut output = vec![0u8; 1 << 28];
    let produced = match backend.as_str() {
        "ng" => uncompress_ng(&mut output, &input)?,
        "rs" => uncompress_rs(&mut output, &input)?,
        _ => return Err(format!("unknown native decompressor backend: {backend}").into()),
    };
    std::fs::write(output_path, &output[..produced])?;
    println!("decompressed {produced} bytes");
    Ok(())
}
