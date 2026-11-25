#[cfg(all(feature = "cuda", feature = "parallel"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::{fs, num::NonZeroUsize, time::Instant};
    use zlib_rs::{
        decompress_slice, BatchCodec, DeflateConfig, GpuMatchSearcher, GpuTranscodeOptions,
        InflateConfig, ReturnCode,
    };
    let mut args = std::env::args().skip(1);
    let manifest = fs::read_to_string(args.next().ok_or("manifest")?)?;
    let workers: NonZeroUsize = args.next().ok_or("workers")?.parse()?;
    let window: NonZeroUsize = args.next().ok_or("window")?.parse()?;
    let budget: u32 = args.next().ok_or("budget")?.parse()?;
    let rounds: usize = args.next().ok_or("rounds")?.parse()?;
    let mut entries = Vec::new();
    for line in manifest.lines().filter(|line| !line.is_empty()) {
        let mut fields = line.split_whitespace();
        let source = fs::read(fields.next().ok_or("source")?)?;
        let raw = fs::read(fields.next().ok_or("decoded")?)?;
        let cap = fields.next().ok_or("cap")?.parse()?;
        let options = GpuTranscodeOptions {
            output_limit: cap,
            decoded_limit: u32::try_from(raw.len())?,
            minimum_decoded_len: 0,
            config: DeflateConfig::new(9),
            search_budget: budget,
            window,
        };
        entries.push((source, raw, options));
    }
    // Give the scheduler the long jobs first, as the document queue does.
    entries.sort_by_key(|(source, _, _)| std::cmp::Reverse(source.len()));
    let searcher = GpuMatchSearcher::new(Some("cuda:0"))?;
    let pool = BatchCodec::new(workers)?;
    let mut jobs = entries
        .iter()
        .map(|(source, _, options)| (source.as_slice(), *options))
        .collect::<Vec<_>>();
    let mut reference = Vec::new();
    // Independently validate every result, including exact bounded prefixes.
    for (source, raw, options) in &entries {
        let mut output = Vec::new();
        let result = zlib_rs::compress_slices_batch_gpu_windowed(
            &mut [zlib_rs::BatchJob::with_vec(
                raw,
                &mut output,
                options.output_limit,
            )?],
            options.config,
            &searcher,
            budget,
            window,
        )?[0];
        if result.code == ReturnCode::Ok {
            let mut scratch = vec![0; raw.len()];
            let (decoded, code) = decompress_slice(&mut scratch, &output, InflateConfig::default());
            assert_eq!(code, ReturnCode::Ok);
            assert_eq!(decoded, raw);
        }
        assert!(!source.is_empty());
        reference.push((output, result.code));
    }
    println!(
        "workers,round,ms,streams,decoded,accepted,output_bytes,source_upload,descriptor_upload"
    );
    for round in 0..rounds {
        let t = Instant::now();
        let results = pool.recompress_zlib_batch(&mut jobs, &searcher)?;
        let elapsed = t.elapsed().as_secs_f64() * 1000.0;
        let mut totals = [0usize; 5];
        for (result, (expected, code)) in results.into_iter().zip(&reference) {
            let result = result.ok_or("unexpected transcode routing")?;
            assert_eq!(result.code, *code);
            assert_eq!(&result.output, expected);
            totals[0] += result.decoded_bytes_processed;
            totals[1] += usize::from(result.code == ReturnCode::Ok);
            totals[2] += result.output.len();
            totals[3] += result.source_bytes_uploaded;
            totals[4] += result.descriptor_bytes_uploaded;
        }
        println!(
            "{workers},{round},{elapsed:.3},{},{},{},{},{},{}",
            entries.len(),
            totals[0],
            totals[1],
            totals[2],
            totals[3],
            totals[4]
        );
    }
    Ok(())
}
#[cfg(not(all(feature = "cuda", feature = "parallel")))]
fn main() {}
