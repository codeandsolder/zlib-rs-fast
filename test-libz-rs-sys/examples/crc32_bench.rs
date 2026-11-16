//! Direct checksum backend comparison without duplicate file I/O and unchecked CLI paths.

use std::error::Error;

fn checksum(mode: &str, input: &[u8]) -> Result<u32, &'static str> {
    let chunk = match mode {
        "sse" | "crc32fast" => input.len().max(1),
        "sse-chunked" | "crc32fast-chunked" => 32,
        "adler32" => return Ok(zlib_rs::adler32::adler32(42, input)),
        _ => return Err("expected sse, crc32fast, sse-chunked, crc32fast-chunked or adler32"),
    };
    match mode {
        "sse" | "sse-chunked" => {
            let mut state = zlib_rs::crc32::Crc32Fold::new();
            for bytes in input.chunks(chunk) {
                state.fold(bytes);
            }
            Ok(state.finish())
        }
        "crc32fast" | "crc32fast-chunked" => {
            let mut state = crc32fast::Hasher::new_with_initial(0);
            for bytes in input.chunks(chunk) {
                state.update(bytes);
            }
            Ok(state.finalize())
        }
        _ => Err("invalid checksum backend"),
    }
}

pub fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let mode = args
        .next()
        .ok_or("usage: crc32_bench <sse|crc32fast|sse-chunked|crc32fast-chunked|adler32> <file>")?;
    let path = args.next().ok_or("missing checksum input file")?;
    if args.next().is_some() {
        return Err("unexpected extra arguments".into());
    }

    let input = std::fs::read(path)?;
    let value = checksum(&mode, &input)?;
    println!("{value:#x}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incremental_crc_backends_agree_on_empty_and_chunk_boundary_inputs(
    ) -> Result<(), &'static str> {
        for len in [0, 1, 2, 31, 32, 33, 63, 64, 65, 4096] {
            let bytes = (0u32..len)
                .map(|index| index.wrapping_mul(37).to_le_bytes()[0])
                .collect::<Vec<_>>();
            let reference = checksum("crc32fast", &bytes)?;
            for mode in ["sse", "sse-chunked", "crc32fast-chunked"] {
                assert_eq!(checksum(mode, &bytes)?, reference, "{mode} len={len}");
            }
            assert_eq!(
                checksum("adler32", &bytes)?,
                zlib_rs::adler32::adler32(42, &bytes)
            );
        }
        assert!(checksum("unsupported", b"").is_err());
        Ok(())
    }
}
