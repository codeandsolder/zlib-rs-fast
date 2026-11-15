//! Native C ABI compression comparison with explicit, caller-owned output files.

#[path = "shared/native_deflate.rs"]
mod native_deflate;

use native_deflate::{compress_ng, compress_rs, NativeResult};

const OUTPUT_CAPACITY: usize = 1 << 28;

fn main() -> NativeResult<()> {
    let mut args = std::env::args().skip(1);
    let command = args.next().ok_or(
        "usage: compress <ng|rs> <level> <input> <output> | compress compare <output-ng> <output-rs>",
    )?;

    if command == "compare" {
        let ng = std::fs::read(args.next().ok_or("missing ng compressed file")?)?;
        let rs = std::fs::read(args.next().ok_or("missing rs compressed file")?)?;
        if args.next().is_some() {
            return Err("unexpected extra arguments".into());
        }
        if ng != rs {
            let mismatch = ng.iter().zip(&rs).position(|(left, right)| left != right);
            return Err(format!(
                "compressed streams differ: ng length {}, rs length {}, first mismatch {mismatch:?}",
                ng.len(),
                rs.len()
            )
            .into());
        }
        println!("identical compressed streams: {} bytes", ng.len());
        return Ok(());
    }

    let level: i32 = args.next().ok_or("missing compression level")?.parse()?;
    let input = std::fs::read(args.next().ok_or("missing input path")?)?;
    let output_path = args.next().ok_or("missing output path")?;
    if args.next().is_some() {
        return Err("unexpected extra arguments".into());
    }

    let mut output = vec![0u8; OUTPUT_CAPACITY];
    let produced = match command.as_str() {
        "ng" => compress_ng(&mut output, &input, level)?,
        "rs" => compress_rs(&mut output, &input, level)?,
        _ => return Err(format!("unknown native compressor: {command}").into()),
    };
    std::fs::write(output_path, &output[..produced])?;
    println!("compressed {produced} bytes");
    Ok(())
}
