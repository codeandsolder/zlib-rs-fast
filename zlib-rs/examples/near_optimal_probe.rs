use std::{env, fs, path::Path};

use zlib_rs::deflate::compress_slice_near_optimal;
use zlib_rs::{compress_bound, compress_slice, DeflateConfig, ReturnCode};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: near_optimal_probe FILE [BYTES] [PASSES]")?;
    let requested = args.next().map(|value| value.parse()).transpose()?;
    let passes: usize = args
        .next()
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(3);

    let bytes = fs::read(Path::new(&path))?;
    let limit = requested.unwrap_or(bytes.len()).min(bytes.len());
    let input = &bytes[..limit];

    let mut stock_buf = vec![0u8; compress_bound(input.len())];
    let (stock, stock_code) = compress_slice(&mut stock_buf, input, DeflateConfig::new(9));
    assert_eq!(stock_code, ReturnCode::Ok);

    let mut optimal_buf = vec![0u8; compress_bound(input.len())];
    let start = std::time::Instant::now();
    let (optimal, optimal_code) = compress_slice_near_optimal(&mut optimal_buf, input, passes);
    let elapsed = start.elapsed();
    assert_eq!(optimal_code, ReturnCode::Ok);

    println!("input_bytes={}", input.len());
    println!("stock_l9_bytes={}", stock.len());
    println!("near_optimal_bytes={}", optimal.len());
    println!("delta_bytes={}", optimal.len() as i64 - stock.len() as i64);
    println!(
        "delta_pct={:+.5}%",
        (optimal.len() as f64 / stock.len() as f64 - 1.0) * 100.0
    );
    println!("optimization_passes={passes}");
    println!("near_optimal_ms={:.3}", elapsed.as_secs_f64() * 1000.0);
    Ok(())
}
