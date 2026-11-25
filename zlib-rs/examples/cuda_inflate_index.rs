#[cfg(all(feature = "cuda", feature = "__internal-api"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::{fs, time::Instant};
    let mut args = std::env::args().skip(1);
    let input = fs::read(args.next().ok_or("input path")?)?;
    let path = args.next().ok_or("output index path")?;
    let limit: u32 = args.next().ok_or("decoded bound")?.parse()?;
    for _ in 0..4 {
        let now = Instant::now();
        let index = zlib_rs::inflate::gpu_index::index_zlib(&input, limit, 1_000_000)
            .map_err(|code| format!("{code:?}"))?;
        eprintln!(
            "index_ms={:.3} segments={} decoded={} adler={:08x}",
            now.elapsed().as_secs_f64() * 1000.0,
            index.segments.len(),
            index.decoded_len,
            index.adler
        );
        let output = index
            .segments
            .iter()
            .map(|s| {
                format!(
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                    s.header_bit,
                    s.start_bit,
                    s.end_bit,
                    s.start,
                    s.end,
                    s.kind,
                    u32::from(s.ends_block)
                )
            })
            .collect::<String>();
        fs::write(&path, output)?;
    }
    Ok(())
}
#[cfg(not(all(feature = "cuda", feature = "__internal-api")))]
fn main() {}
