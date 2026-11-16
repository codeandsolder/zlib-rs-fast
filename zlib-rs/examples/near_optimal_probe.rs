use std::{env, fs, path::Path};

use zlib_rs::deflate::compress_slice_near_optimal;
use zlib_rs::{compress_bound, compress_slice, DeflateConfig, ReturnCode};

/// Produce the exact signed byte delta and a percentage rounded to five
/// decimal places. Integer arithmetic avoids losing precision above 2^53.
fn compression_delta(stock: usize, optimized: usize) -> Result<(i128, String), &'static str> {
    // 100 percent * 100,000 decimal units. Even a 64-bit usize times
    // 10 million fits comfortably inside u128.
    const PERCENT_SCALE: u128 = 10_000_000;
    if stock == 0 {
        return Err("cannot calculate percentage against an empty compressed stream");
    }
    let stock_wide = u64::try_from(stock).map_err(|_| "stock size exceeds 64-bit host range")?;
    let optimized_wide =
        u64::try_from(optimized).map_err(|_| "optimized size exceeds 64-bit host range")?;
    let delta = i128::from(optimized_wide) - i128::from(stock_wide);
    let original = u128::from(stock_wide);
    let scaled = (delta.unsigned_abs() * PERCENT_SCALE + original / 2) / original;
    let sign = if delta.is_negative() { "-" } else { "+" };
    let pct = format!("{sign}{}.{:05}", scaled / 100_000, scaled % 100_000);
    Ok((delta, pct))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: near_optimal_probe FILE [BYTES] [PASSES] [ENCODED_OUT]")?;
    let requested: Option<usize> = args.next().map(|value| value.parse()).transpose()?;
    let passes: usize = args
        .next()
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(10);
    let encoded_path = args.next();
    if args.next().is_some() {
        return Err("unexpected extra argument after ENCODED_OUT".into());
    }
    if passes == 0 {
        return Err("optimization passes must be positive".into());
    }

    let bytes = fs::read(Path::new(&path))?;
    let limit = requested.unwrap_or(bytes.len()).min(bytes.len());
    let input = &bytes[..limit];

    let mut stock_buf = vec![0u8; compress_bound(input.len())];
    let (stock, stock_code) = compress_slice(&mut stock_buf, input, DeflateConfig::new(9));
    if stock_code != ReturnCode::Ok {
        return Err(format!("stock level-9 compression failed: {stock_code:?}").into());
    }

    let mut optimal_buf = vec![0u8; compress_bound(input.len())];
    let start = std::time::Instant::now();
    let (optimal, optimal_code) = compress_slice_near_optimal(&mut optimal_buf, input, passes);
    let elapsed = start.elapsed();
    if optimal_code != ReturnCode::Ok {
        return Err(format!("near-optimal compression failed: {optimal_code:?}").into());
    }
    if let Some(path) = encoded_path {
        // An explicit caller-supplied output path is the only write destination.
        fs::write(path, &*optimal)?;
    }

    let (delta, delta_pct) = compression_delta(stock.len(), optimal.len())?;
    println!("input_bytes={}", input.len());
    println!("stock_l9_bytes={}", stock.len());
    println!("near_optimal_bytes={}", optimal.len());
    println!("delta_bytes={delta}");
    println!("delta_pct={delta_pct}%");
    println!("optimization_passes={passes}");
    println!("near_optimal_ms={:.3}", elapsed.as_secs_f64() * 1000.0);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_delta_handles_zero_baseline_sign_and_rounding() -> Result<(), &'static str> {
        assert!(compression_delta(0, 1).is_err());
        assert_eq!(compression_delta(100, 100)?, (0, "+0.00000".into()));
        assert_eq!(compression_delta(100, 95)?, (-5, "-5.00000".into()));
        assert_eq!(compression_delta(100, 105)?, (5, "+5.00000".into()));
        assert_eq!(compression_delta(3, 2)?, (-1, "-33.33333".into()));
        assert_eq!(compression_delta(3, 4)?, (1, "+33.33333".into()));
        #[cfg(target_pointer_width = "64")]
        {
            let large = usize::try_from(9_007_199_254_740_999_u64)
                .map_err(|_| "64-bit fixture does not fit usize")?;
            assert_eq!(compression_delta(large, large - 1)?.0, -1);
        }
        Ok(())
    }
}
