use flate2::read::GzDecoder;
use flate2::read::MultiGzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use std::fs::File;
use std::io::prelude::*;
use std::io::{self, BufReader};
use std::path::Path;

// test extraction of a gzipped file
#[test]
fn test_extract_success() -> Result<(), Box<dyn std::error::Error>> {
    let content = extract_file(Path::new("tests/good-file.gz"))?;
    let mut expected = Vec::new();
    File::open("tests/good-file.txt")?.read_to_end(&mut expected)?;
    assert_eq!(content, expected);

    Ok(())
}
//
// test partial extraction of a multistream gzipped file
#[test]
fn test_extract_success_partial_multi() -> Result<(), Box<dyn std::error::Error>> {
    let content = extract_file(Path::new("tests/multi.gz"))?;
    let mut expected = String::new();
    BufReader::new(File::open("tests/multi.txt")?).read_line(&mut expected)?;
    assert_eq!(content, expected.as_bytes());

    Ok(())
}

// test extraction fails on a corrupt file
#[test]
fn test_extract_failure() -> Result<(), Box<dyn std::error::Error>> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(b"corrupt me")?;
    let mut gzip = encoder.finish()?;
    // The gzip footer stores CRC32 followed by ISIZE; corrupt a CRC byte.
    let crc_offset = gzip.len() - 8;
    gzip[crc_offset] ^= 0xff;

    let mut output = Vec::new();
    let result = GzDecoder::new(&gzip[..]).read_to_end(&mut output);
    assert_eq!(
        result.err().ok_or("expected an error")?.kind(),
        io::ErrorKind::InvalidInput
    );

    Ok(())
}

//test complete extraction of a multistream gzipped file
#[test]
fn test_extract_success_multi() -> Result<(), Box<dyn std::error::Error>> {
    let content = extract_file_multi(Path::new("tests/multi.gz"))?;
    let mut expected = Vec::new();
    File::open("tests/multi.txt")?.read_to_end(&mut expected)?;
    assert_eq!(content, expected);

    Ok(())
}

// Tries to extract path into memory (assuming a .gz file).
fn extract_file(path_compressed: &Path) -> io::Result<Vec<u8>> {
    let mut v = Vec::new();
    let f = File::open(path_compressed)?;
    GzDecoder::new(f).read_to_end(&mut v)?;
    Ok(v)
}

// Tries to extract path into memory (decompressing all members in case
// of a multi member .gz file).
fn extract_file_multi(path_compressed: &Path) -> io::Result<Vec<u8>> {
    let mut v = Vec::new();
    let f = File::open(path_compressed)?;
    MultiGzDecoder::new(f).read_to_end(&mut v)?;
    Ok(v)
}

#[test]
fn empty_error_once() -> Result<(), Box<dyn std::error::Error>> {
    let data: &[u8] = &[];
    let cbjson = GzDecoder::new(data);
    let reader = BufReader::new(cbjson);
    let mut stream = reader.lines();
    assert!(stream.next().ok_or("missing item")?.is_err());
    assert!(stream.next().is_none());

    Ok(())
}
