#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::{fs, num::NonZeroUsize, time::Instant};
    use zlib_rs::{
        compress_slices_batch_gpu_windowed, decompress_slice, BatchJob, DeflateConfig,
        GpuMatchSearcher, GpuTranscodeOptions, InflateConfig, ReturnCode,
    };
    let mut args = std::env::args().skip(1);
    let source = fs::read(args.next().ok_or("source path")?)?;
    let decoded = fs::read(args.next().ok_or("decoded path")?)?;
    let cap: usize = args.next().ok_or("output cap")?.parse()?;
    let budget: u32 = args.next().ok_or("budget")?.parse()?;
    let window =
        NonZeroUsize::new(args.next().ok_or("window")?.parse()?).ok_or("nonzero window")?;
    let config = DeflateConfig::new(9);
    let searcher = GpuMatchSearcher::new(Some("cuda:0"))?;
    let mut scratch = vec![0; decoded.len()];
    println!("round,mode,ms,decoded,encoded,code,processed,source_upload,descriptor_upload");
    for round in 0..4 {
        let t = Instant::now();
        let (raw, code) = decompress_slice(&mut scratch, &source, InflateConfig::default());
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(code, ReturnCode::Ok);
        assert_eq!(raw, decoded);
        println!(
            "{round},cpu-inflate,{ms:.3},{},{},{code:?},0,0,0",
            decoded.len(),
            source.len()
        );
        let mut raw_output = Vec::new();
        let run_raw = |output: &mut Vec<u8>| -> Result<_, Box<dyn std::error::Error>> {
            let t = Instant::now();
            let result = compress_slices_batch_gpu_windowed(
                &mut [BatchJob::with_vec(&decoded, output, cap)?],
                config,
                &searcher,
                budget,
                window,
            )?[0];
            println!(
                "{round},raw-gpu,{:.3},{},{},{:?},0,0,0",
                t.elapsed().as_secs_f64() * 1000.0,
                decoded.len(),
                result.written,
                result.code
            );
            Ok(result)
        };
        let run_transcode = || -> Result<_, Box<dyn std::error::Error>> {
            let t = Instant::now();
            let result = searcher
                .recompress_zlib(
                    &source,
                    GpuTranscodeOptions {
                        output_limit: cap,
                        decoded_limit: u32::try_from(decoded.len())?,
                        minimum_decoded_len: 0,
                        config,
                        search_budget: budget,
                        window,
                    },
                )?
                .ok_or("unexpected transcode routing")?;
            println!(
                "{round},transcode,{:.3},{},{},{:?},{},{},{}",
                t.elapsed().as_secs_f64() * 1000.0,
                result.decoded_len.unwrap_or(result.decoded_bytes_indexed),
                result.output.len(),
                result.code,
                result.decoded_bytes_processed,
                result.source_bytes_uploaded,
                result.descriptor_bytes_uploaded
            );
            Ok(result)
        };
        let (a, b) = if round % 2 == 0 {
            (run_raw(&mut raw_output)?, run_transcode()?)
        } else {
            let b = run_transcode()?;
            (run_raw(&mut raw_output)?, b)
        };
        if b.code == ReturnCode::Ok {
            assert_eq!(b.decoded_len, Some(decoded.len()));
        }
        assert_eq!(a.code, b.code);
        assert_eq!(raw_output, b.output);
        if b.code == ReturnCode::Ok {
            let (raw, code) = decompress_slice(&mut scratch, &b.output, InflateConfig::default());
            assert_eq!(code, ReturnCode::Ok);
            assert_eq!(raw, decoded);
        }
    }
    Ok(())
}
#[cfg(not(feature = "cuda"))]
fn main() {}
