//! Real-file profiling, with explicit preparation separated from decoding.

use std::{
    error::Error,
    fmt, fs,
    hint::black_box,
    num::NonZeroUsize,
    time::{Duration, Instant},
};

use zlib_rs::{compress_bound, compress_slice, BatchJob, DeflateConfig, InflateConfig, ReturnCode};

const USAGE: &str = "usage: profile_file prepare INPUT METHOD OUTPUT | encode INPUT METHOD ITERATIONS | decode-slice STREAM EXPECTED ITERATIONS | decode-vector STREAM EXPECTED ITERATIONS; METHOD is 0..=9 or near-optimal:PASSES (feature near-optimal)";

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const GUARD: u8 = 0xa5;
const GUARD_BYTES: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Destination {
    Slice,
    Vector,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Encoder {
    Level(i32),
    #[cfg(feature = "near-optimal")]
    NearOptimal(NonZeroUsize),
}

impl Encoder {
    fn parse(value: &str) -> Result<Self> {
        if let Some(passes) = value.strip_prefix("near-optimal:") {
            #[cfg(feature = "near-optimal")]
            return Ok(Self::NearOptimal(passes.parse()?));
            #[cfg(not(feature = "near-optimal"))]
            return Err(format!("near-optimal:{passes} requires the near-optimal feature").into());
        }
        let level = value.parse()?;
        if !(0..=9).contains(&level) {
            return Err("compression level must be in 0..=9".into());
        }
        Ok(Self::Level(level))
    }

    fn encode(self, input: &[u8], output: &mut [u8]) -> Result<usize> {
        let (written, code) = match self {
            Self::Level(level) => compress_slice(output, input, DeflateConfig::new(level)),
            #[cfg(feature = "near-optimal")]
            Self::NearOptimal(passes) => {
                zlib_rs::compress_slice_near_optimal(output, input, passes.get())
            }
        };
        if code != ReturnCode::Ok {
            return Err(format!("compression failed: {code:?}").into());
        }
        Ok(written.len())
    }
}

impl fmt::Display for Encoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Level(level) => write!(f, "L{level}"),
            #[cfg(feature = "near-optimal")]
            Self::NearOptimal(passes) => write!(f, "near-optimal:{passes}"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Operation<'a> {
    Prepare {
        input: &'a str,
        encoder: Encoder,
        output: &'a str,
    },
    Encode {
        input: &'a str,
        encoder: Encoder,
        iterations: NonZeroUsize,
    },
    Decode {
        destination: Destination,
        stream: &'a str,
        expected: &'a str,
        iterations: NonZeroUsize,
    },
}

impl<'a> Operation<'a> {
    fn parse(args: &'a [String]) -> Result<Self> {
        let [mode, first, second, third] = args else {
            return Err(USAGE.into());
        };
        match mode.as_str() {
            "prepare" => Ok(Self::Prepare {
                input: first,
                encoder: Encoder::parse(second)?,
                output: third,
            }),
            "encode" => Ok(Self::Encode {
                input: first,
                encoder: Encoder::parse(second)?,
                iterations: third.parse()?,
            }),
            "decode-slice" | "decode-vector" => Ok(Self::Decode {
                destination: if mode == "decode-slice" {
                    Destination::Slice
                } else {
                    Destination::Vector
                },
                stream: first,
                expected: second,
                iterations: third.parse()?,
            }),
            _ => Err(USAGE.into()),
        }
    }
}

fn verify_decode(job: &mut BatchJob<'_>, encoded_len: usize, expected_len: usize) -> Result<usize> {
    let result = job.decompress(InflateConfig::default());
    if result.code != ReturnCode::Ok || result.read != encoded_len || result.written != expected_len
    {
        return Err(format!("unexpected decode result: {result:?}; expected {encoded_len} input and {expected_len} output bytes").into());
    }
    Ok(result.written)
}

fn profile_decode_job(
    job: &mut BatchJob<'_>,
    expected_len: usize,
    iterations: NonZeroUsize,
) -> Result<Duration> {
    let input_len = job.input.len();
    let start = Instant::now();
    for _ in 0..iterations.get() {
        black_box(verify_decode(
            black_box(&mut *job),
            input_len,
            expected_len,
        )?);
    }
    Ok(start.elapsed())
}

// The caller creates the compressed stream in a distinct process before
// measuring decoder counters. The reference bytes only validate output.
fn decode_profile(
    stream: &[u8],
    expected: &[u8],
    destination: Destination,
    iterations: NonZeroUsize,
) -> Result<()> {
    let capacity = expected
        .len()
        .checked_add(GUARD_BYTES)
        .ok_or("output size overflow")?;
    match destination {
        Destination::Slice => {
            let mut output = vec![GUARD; capacity];
            let elapsed = {
                let mut job = BatchJob::new(stream, &mut output);
                profile_decode_job(&mut job, expected.len(), iterations)?
            };
            if output[..expected.len()] != *expected
                || !output[expected.len()..].iter().all(|&byte| byte == GUARD)
            {
                return Err("slice output or untouched tail differs from reference".into());
            }
            println!(
                "decode-slice: {} compressed bytes -> {} bytes, {iterations} iterations, {elapsed:?}",
                stream.len(),
                expected.len(),
            );
        }
        Destination::Vector => {
            let mut output = Vec::with_capacity(capacity);
            let elapsed = {
                let mut job = BatchJob::with_vec(stream, &mut output, capacity)?;
                profile_decode_job(&mut job, expected.len(), iterations)?
            };
            if output != expected {
                return Err("vector output differs from reference".into());
            }
            println!(
                "decode-vector: {} compressed bytes -> {} bytes, {iterations} iterations, {elapsed:?}",
                stream.len(),
                expected.len(),
            );
        }
    }
    Ok(())
}

fn encode_profile(input: &[u8], encoder: Encoder, iterations: NonZeroUsize) -> Result<()> {
    let mut output = vec![0; compress_bound(input.len())];
    let start = Instant::now();
    let mut written = 0;
    for _ in 0..iterations.get() {
        written = black_box(encoder.encode(black_box(input), black_box(&mut output))?);
    }
    let elapsed = start.elapsed();
    let mut decoded = vec![0; input.len()];
    let mut job = BatchJob::new(&output[..written], &mut decoded);
    verify_decode(&mut job, written, input.len())?;
    if decoded != input {
        return Err("encoded stream does not round-trip".into());
    }
    println!("encode {encoder}: {} bytes -> {written} compressed bytes, {iterations} iterations, {elapsed:?}", input.len());
    Ok(())
}

fn run(args: &[String]) -> Result<()> {
    match Operation::parse(args)? {
        Operation::Prepare {
            input,
            encoder,
            output,
        } => {
            use std::io::Write;
            let input = fs::read(input)?;
            let mut buffer = vec![0; compress_bound(input.len())];
            let count = encoder.encode(&input, &mut buffer)?;
            let mut file = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(output)?;
            file.write_all(&buffer[..count])?;
            println!("prepared {count} compressed bytes");
            Ok(())
        }
        Operation::Encode {
            input,
            encoder,
            iterations,
        } => encode_profile(&fs::read(input)?, encoder, iterations),
        Operation::Decode {
            destination,
            stream,
            expected,
            iterations,
        } => decode_profile(
            &fs::read(stream)?,
            &fs::read(expected)?,
            destination,
            iterations,
        ),
    }
}

fn self_test() -> Result<()> {
    let args = |values: &[&str]| values.iter().map(ToString::to_string).collect::<Vec<_>>();
    assert!(Operation::parse(&args(&["encode", "input", "10", "1"])).is_err());
    assert!(Operation::parse(&args(&["decode-vector", "stream", "input", "0"])).is_err());
    assert!(Operation::parse(&args(&["unknown", "a", "b", "c"])).is_err());
    assert!(Operation::parse(&args(&["prepare", "input", "6"])).is_err());
    let iters = NonZeroUsize::MIN.saturating_add(1);
    assert_eq!(
        Operation::parse(&args(&["encode", "input", "9", "2"]))?,
        Operation::Encode {
            input: "input",
            encoder: Encoder::Level(9),
            iterations: iters
        }
    );
    #[cfg(feature = "near-optimal")]
    {
        assert_eq!(
            Encoder::parse("near-optimal:2")?,
            Encoder::NearOptimal(iters)
        );
        assert!(Encoder::parse("near-optimal:0").is_err());
        assert!(Encoder::parse("near-optimal:").is_err());
        encode_profile(b"abcabcabcabcabc", Encoder::NearOptimal(iters), iters)?;
    }
    #[cfg(not(feature = "near-optimal"))]
    assert!(Encoder::parse("near-optimal:2").is_err());
    for input in [&b""[..], &b"overlapping match overlapping match"[..]] {
        let mut stream = vec![0; compress_bound(input.len())];
        let count = Encoder::Level(6).encode(input, &mut stream)?;
        for destination in [Destination::Slice, Destination::Vector] {
            decode_profile(&stream[..count], input, destination, iters)?;
            assert!(decode_profile(&stream[..count], b"wrong bytes", destination, iters).is_err());
            if !input.is_empty() {
                let mut wrong = input.to_vec();
                wrong[0] ^= 1;
                assert!(decode_profile(&stream[..count], &wrong, destination, iters).is_err());
            }
            assert!(decode_profile(&stream[..count - 1], input, destination, iters).is_err());
            let mut trailing = stream[..count].to_vec();
            trailing.push(0);
            assert!(decode_profile(&trailing, input, destination, iters).is_err());
        }
        encode_profile(input, Encoder::Level(6), iters)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args == ["--self-test"] {
        self_test()?;
        println!("file profiler invariants passed");
        return Ok(());
    }
    run(&args)
}
