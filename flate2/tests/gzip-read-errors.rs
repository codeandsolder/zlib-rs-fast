use std::cell::Cell;
use std::io::{self, BufReader, Read};

struct FailReads<'a> {
    data: &'a [u8],
    remaining: usize,
    kind: io::ErrorKind,
    reads: &'a Cell<usize>,
}

impl Read for FailReads<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.reads.set(self.reads.get() + 1);
        if self.remaining > 0 {
            self.remaining -= 1;
            return Err(io::Error::new(self.kind, "injected input error"));
        }
        self.data.read(buf)
    }
}

fn encoder(source: FailReads<'_>, buffered: bool) -> Box<dyn Read + '_> {
    if buffered {
        Box::new(flate2::bufread::GzEncoder::new(
            BufReader::new(source),
            flate2::Compression::default(),
        ))
    } else {
        Box::new(flate2::read::GzEncoder::new(
            source,
            flate2::Compression::default(),
        ))
    }
}

fn assert_roundtrip(compressed: &[u8], input: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    let mut decoded = Vec::new();
    flate2::read::GzDecoder::new(compressed).read_to_end(&mut decoded)?;
    assert_eq!(decoded, input);

    Ok(())
}

#[test]
fn gzip_header_survives_interrupted_input() -> Result<(), Box<dyn std::error::Error>> {
    let input = b"a signal should not corrupt the gzip header";
    for buffered in [false, true] {
        let reads = Cell::new(0);
        let source = FailReads {
            data: input,
            remaining: 1,
            kind: io::ErrorKind::Interrupted,
            reads: &reads,
        };
        let mut compressed = Vec::new();
        encoder(source, buffered).read_to_end(&mut compressed)?;
        assert_roundtrip(&compressed, input)?;
    }

    Ok(())
}

#[test]
fn gzip_reports_one_shot_input_errors() -> Result<(), Box<dyn std::error::Error>> {
    let input = b"a one-shot error must not disappear after the gzip header";
    for kind in [
        io::ErrorKind::WouldBlock,
        io::ErrorKind::PermissionDenied,
        io::ErrorKind::ConnectionReset,
        io::ErrorKind::UnexpectedEof,
        io::ErrorKind::Other,
    ] {
        for buffered in [false, true] {
            let reads = Cell::new(0);
            let source = FailReads {
                data: input,
                remaining: 1,
                kind,
                reads: &reads,
            };
            let mut encoder = encoder(source, buffered);
            let mut compressed = Vec::new();
            let err = encoder
                .read_to_end(&mut compressed)
                .err()
                .ok_or("expected an error")?;
            assert_eq!(err.kind(), kind);
            assert_eq!(err.to_string(), "injected input error");
            assert_eq!(compressed.len(), 10);

            encoder.read_to_end(&mut compressed)?;
            assert_roundtrip(&compressed, input)?;
        }
    }

    Ok(())
}

#[test]
fn gzip_reports_header_progress_before_input_error() -> Result<(), Box<dyn std::error::Error>> {
    let input = b"retrying a nonblocking source must preserve the header";
    for kind in [io::ErrorKind::Interrupted, io::ErrorKind::WouldBlock] {
        for remaining in [1, 2] {
            for buffered in [false, true] {
                let reads = Cell::new(0);
                let source = FailReads {
                    data: input,
                    remaining,
                    kind,
                    reads: &reads,
                };
                let mut encoder = encoder(source, buffered);
                let mut buffer = [0; 64];
                assert_eq!(encoder.read(&mut [])?, 0);
                assert_eq!(reads.get(), 0);
                let count = encoder.read(&mut buffer)?;
                assert_eq!(count, 10);
                assert_eq!(reads.get(), 0);
                let mut compressed = buffer[..count].to_vec();
                for attempted in 0..remaining {
                    assert_eq!(encoder.read(&mut [])?, 0);
                    assert_eq!(encoder.read(&mut [])?, 0);
                    assert_eq!(
                        reads.get(),
                        attempted,
                        "only earlier error-producing reads may be counted; header-only and empty reads must not access input"
                    );
                    let err = encoder.read(&mut buffer).err().ok_or("expected an error")?;
                    assert_eq!(err.kind(), kind);
                    assert_eq!(err.to_string(), "injected input error");
                    assert_eq!(
                        reads.get(),
                        attempted + 1,
                        "this nonempty read must obtain its error from exactly one new input attempt, not a cached error or an internal retry"
                    );
                }
                encoder.read_to_end(&mut compressed)?;
                assert_roundtrip(&compressed, input)?;
            }
        }
    }

    Ok(())
}

#[test]
fn gzip_reads_header_separately() -> Result<(), Box<dyn std::error::Error>> {
    for input in [
        &b""[..],
        &b"read the header separately from the compressed body"[..],
    ] {
        for buffered in [false, true] {
            for header_prefix in [0, 1, 7, 9, 10] {
                let reads = Cell::new(0);
                let source = FailReads {
                    data: input,
                    remaining: 0,
                    kind: io::ErrorKind::Other,
                    reads: &reads,
                };
                let mut encoder = encoder(source, buffered);
                let mut compressed = vec![0; header_prefix];
                encoder.read_exact(&mut compressed)?;
                assert_eq!(reads.get(), 0);
                let mut buffer = [0; 64];
                let count = encoder.read(&mut buffer)?;
                // The default gzip header is 10 bytes. Until read_exact has
                // consumed all of it, this read must return only its remaining
                // bytes without accessing input; afterward, it reads the body.
                if header_prefix < 10 {
                    assert_eq!(
                        count,
                        10 - header_prefix,
                        "a read with header bytes remaining must return exactly those bytes, without compressed body data"
                    );
                    assert_eq!(
                        reads.get(),
                        0,
                        "header bytes are already available; returning them must not read the input"
                    );
                } else {
                    assert!(
                        count > 0,
                        "finishing the header is not EOF; even empty input produces a compressed body"
                    );
                    assert!(
                        reads.get() > 0,
                        "the header is exhausted, so producing body data must access the input"
                    );
                }
                compressed.extend_from_slice(&buffer[..count]);
                encoder.read_to_end(&mut compressed)?;
                assert_roundtrip(&compressed, input)?;
            }
        }
    }

    Ok(())
}
