use crate::stream::io::{Read, Write};
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use super::{read, write, GzBuilder, GzHeaderParser};
use crate::stream::{Compression, GzHeader};
use rand::{rngs::SmallRng, RngExt, SeedableRng};

#[test]
fn roundtrip() -> Result<(), Box<dyn std::error::Error>> {
    let mut e = write::GzEncoder::new(Vec::new(), Compression::default());
    e.write_all(b"foo bar baz")?;
    let inner = e.finish()?;
    let mut d = read::GzDecoder::new(&inner[..]);
    let mut s = String::new();
    d.read_to_string(&mut s)?;
    assert_eq!(s, "foo bar baz");

    Ok(())
}

#[test]
fn roundtrip_zero() -> Result<(), Box<dyn std::error::Error>> {
    let e = write::GzEncoder::new(Vec::new(), Compression::default());
    let inner = e.finish()?;
    let mut d = read::GzDecoder::new(&inner[..]);
    let mut s = String::new();
    d.read_to_string(&mut s)?;
    assert_eq!(s, "");

    Ok(())
}

#[test]
fn roundtrip_big() -> Result<(), Box<dyn std::error::Error>> {
    let mut real = Vec::new();
    let mut w = write::GzEncoder::new(Vec::new(), Compression::default());
    let v = crate::stream::random_bytes().take(1024).collect::<Vec<_>>();
    let mut rng = SmallRng::seed_from_u64(0x0063_6875_6e6b);
    for _ in 0..200 {
        let to_write = &v[..rng.random_range(0..v.len())];
        real.extend(to_write.iter().copied());
        w.write_all(to_write)?;
    }
    let result = w.finish()?;
    let mut r = read::GzDecoder::new(&result[..]);
    let mut v = Vec::new();
    r.read_to_end(&mut v)?;
    assert_eq!(v, real);

    Ok(())
}

#[test]
fn roundtrip_big2() -> Result<(), Box<dyn std::error::Error>> {
    let v = crate::stream::random_bytes()
        .take(1024 * 1024)
        .collect::<Vec<_>>();
    let mut r = read::GzDecoder::new(read::GzEncoder::new(&v[..], Compression::default()));
    let mut res = Vec::new();
    r.read_to_end(&mut res)?;
    assert_eq!(res, v);

    Ok(())
}

// A Rust implementation of CRC that closely matches the C code in RFC1952.
// Only use this to create CRCs for tests.
struct Rfc1952Crc {
    /* Table of CRCs of all 8-bit messages. */
    crc_table: [u32; 256],
}

impl Rfc1952Crc {
    fn new() -> Self {
        let mut crc = Self {
            crc_table: [0; 256],
        };
        /* Make the table for a fast CRC. */
        for (n, entry) in (0_u32..256).zip(&mut crc.crc_table) {
            let mut c = n;
            for _k in 0..8 {
                if c & 1 != 0 {
                    c = 0xedb8_8320 ^ (c >> 1);
                } else {
                    c >>= 1;
                }
            }
            *entry = c;
        }
        crc
    }

    /*
     Update a running crc with the bytes buf and return
     the updated crc. The crc should be initialized to zero. Pre- and
     post-conditioning (one's complement) is performed within this
     function so it shouldn't be done by the caller.
    */
    fn update_crc(&self, crc: u32, buf: &[u8]) -> u32 {
        let mut c = crc ^ 0xffff_ffff;

        for b in buf {
            c = self.crc_table[usize::from(c.to_le_bytes()[0] ^ *b)] ^ (c >> 8);
        }
        c ^ 0xffff_ffff
    }

    /* Return the CRC of the bytes buf. */
    fn crc(&self, buf: &[u8]) -> u32 {
        self.update_crc(0, buf)
    }
}

#[test]
fn roundtrip_header() -> Result<(), Box<dyn std::error::Error>> {
    let mut header = GzBuilder::new()
        .mtime(1234)
        .operating_system(57)
        .filename("filename")
        .comment("comment")
        .into_header(Compression::fast());

    // Add a CRC to the header
    header[3] ^= super::FHCRC;
    let rfc1952_crc = Rfc1952Crc::new();
    let crc32 = rfc1952_crc.crc(&header);
    let [a, b, ..] = crc32.to_le_bytes();
    let crc16 = u16::from_le_bytes([a, b]);
    header.extend(&crc16.to_le_bytes());

    let mut parser = GzHeaderParser::new();
    parser.parse(&mut header.as_slice())?;
    let actual = parser.header().ok_or("missing gzip header")?;
    assert_eq!(
        actual,
        &GzHeader {
            extra: None,
            filename: Some(b"filename".to_vec()),
            comment: Some(b"comment".to_vec()),
            operating_system: 57,
            mtime: 1234
        }
    );

    Ok(())
}

#[test]
fn gzip_encoder_matches_rfc1952() -> Result<(), Box<dyn std::error::Error>> {
    /// Extract CRC32 and ISIZE from gzip footer
    fn extract_zip_footer(compressed: &[u8]) -> (u32, u32) {
        assert!(compressed.len() >= 8, "Gzip output too short");
        let footer_start = compressed.len() - 8;

        let crc = u32::from_le_bytes([
            compressed[footer_start],
            compressed[footer_start + 1],
            compressed[footer_start + 2],
            compressed[footer_start + 3],
        ]);

        let size = u32::from_le_bytes([
            compressed[footer_start + 4],
            compressed[footer_start + 5],
            compressed[footer_start + 6],
            compressed[footer_start + 7],
        ]);

        (crc, size)
    }

    #[track_caller]
    fn verify_footer(
        compressed: &[u8],
        data: &[u8],
        expected_crc: u32,
        description: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let expected_size = u32::try_from(data.len())?;
        let (actual_crc, actual_size) = extract_zip_footer(compressed);
        assert_eq!(actual_crc, expected_crc, "CRC32 mismatch for {description}");
        assert_eq!(
            actual_size, expected_size,
            "Size mismatch for {description}"
        );
        Ok(())
    }

    #[track_caller]
    fn test_crc_for_write(
        data: &[u8],
        expected_crc: u32,
        description: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut encoder = write::GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(data)?;
        verify_footer(&encoder.finish()?, data, expected_crc, description)
    }

    #[track_caller]
    fn test_crc_for_read(
        data: &[u8],
        expected_crc: u32,
        description: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut encoder =
            read::GzEncoder::new(crate::stream::io::Cursor::new(data), Compression::default());
        let mut compressed = Vec::new();
        encoder.read_to_end(&mut compressed)?;
        verify_footer(&compressed, data, expected_crc, description)
    }

    #[track_caller]
    fn test_crc_for_data(data: &[u8], description: &str) -> Result<(), Box<dyn std::error::Error>> {
        let rfc1952_crc = Rfc1952Crc::new();
        let expected_crc = rfc1952_crc.crc(data);

        test_crc_for_write(data, expected_crc, description)?;
        test_crc_for_read(data, expected_crc, description)?;

        Ok(())
    }

    // Edge cases
    test_crc_for_data(&[], "empty data")?;
    test_crc_for_data(&[0x00], "single zero byte")?;
    test_crc_for_data(&[0xFF], "single 0xFF byte")?;

    // Simple text patterns
    test_crc_for_data(b"Hello World", "simple ASCII")?;
    test_crc_for_data(b"AAAAAAA", "repeated 'A'")?;
    test_crc_for_data(b"1234567890", "digits")?;

    // Binary patterns
    test_crc_for_data(&[0x00, 0x01, 0x02, 0x03, 0x04, 0x05], "sequential bytes")?;
    test_crc_for_data(&[0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55], "alternating pattern")?;
    test_crc_for_data(&[0x00; 10], "all zeros")?;
    test_crc_for_data(&[0xFF; 10], "all ones")?;

    // Large data
    let large_data = vec![0x42; 10240];
    test_crc_for_data(&large_data, "10 kiB data")?;

    // Test multi-write scenario to ensure CRC accumulation works correctly
    {
        let data = b"This is a test of multi-write CRC accumulation";
        let rfc1952_crc = Rfc1952Crc::new();
        let expected_crc = rfc1952_crc.crc(data);

        let mut encoder = write::GzEncoder::new(Vec::new(), Compression::default());
        // Write in chunks
        encoder.write_all(&data[..10])?;
        encoder.write_all(&data[10..20])?;
        encoder.write_all(&data[20..])?;
        let compressed = encoder.finish()?;

        let expected_size = u32::try_from(data.len())?;
        let (actual_crc, actual_size) = extract_zip_footer(&compressed);

        assert_eq!(
            expected_crc, actual_crc,
            "Multi-write CRC mismatch: expected {expected_crc:#08x}, got {actual_crc:#08x}"
        );
        assert_eq!(
            expected_size, actual_size,
            "Size mismatch for multi-write: expected {expected_size}, got {actual_size}"
        );
    }

    Ok(())
}

fn gzip_corrupted_crc() -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let test_data = b"The quick brown fox jumps over the lazy dog";

    let mut encoder = write::GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(test_data)?;
    let mut compressed = encoder.finish()?;

    // Corrupt the CRC32 in the footer
    let crc_offset = compressed.len() - 8;
    compressed[crc_offset] ^= 0xFF;

    Ok(compressed)
}

#[test]
fn read_decoder_detects_corrupted_crc() -> Result<(), Box<dyn std::error::Error>> {
    let compressed = gzip_corrupted_crc()?;
    let mut decoder = read::GzDecoder::new(&compressed[..]);
    let mut output = Vec::new();
    let error = decoder
        .read_to_end(&mut output)
        .err()
        .ok_or("expected an error")?;
    assert_eq!(error.kind(), crate::stream::io::ErrorKind::InvalidInput);

    Ok(())
}

#[test]
fn read_decoder_rejects_incomplete_deflate_stream() -> Result<(), Box<dyn std::error::Error>> {
    let mut compressed = gzip_corrupted_crc()?;
    compressed.truncate(11);
    let error = read::GzDecoder::new(&compressed[..])
        .read_to_end(&mut Vec::new())
        .err()
        .ok_or("expected an error")?;
    assert_eq!(error.kind(), crate::stream::io::ErrorKind::UnexpectedEof);
    assert_eq!(error.to_string(), "incomplete deflate stream");

    Ok(())
}

#[test]
fn write_decoder_detects_corrupted_crc() -> Result<(), Box<dyn std::error::Error>> {
    let compressed = gzip_corrupted_crc()?;
    let mut decoder = write::GzDecoder::new(Vec::new());
    decoder.write_all(&compressed)?;
    let error = decoder.finish().err().ok_or("expected an error")?;
    assert_eq!(error.kind(), crate::stream::io::ErrorKind::InvalidInput);

    Ok(())
}

#[test]
fn fields() -> Result<(), Box<dyn std::error::Error>> {
    let r = [0, 2, 4, 6];
    let e = GzBuilder::new()
        .filename("foo.rs")
        .comment("bar")
        .extra(vec![0, 1, 2, 3])
        .read(&r[..], Compression::default());
    let mut d = read::GzDecoder::new(e);
    assert_eq!(
        d.header().ok_or("missing gzip header")?.filename(),
        Some(&b"foo.rs"[..])
    );
    assert_eq!(
        d.header().ok_or("missing gzip header")?.comment(),
        Some(&b"bar"[..])
    );
    assert_eq!(
        d.header().ok_or("missing gzip header")?.extra(),
        Some(&b"\x00\x01\x02\x03"[..])
    );
    let mut res = Vec::new();
    d.read_to_end(&mut res)?;
    assert_eq!(res, vec![0, 2, 4, 6]);

    Ok(())
}

#[test]
#[should_panic(expected = "gzip extra field length cannot exceed u16::MAX")]
fn extra_too_long() {
    let _ = GzBuilder::new().extra(vec![0; usize::from(u16::MAX) + 1]);
}

#[test]
fn keep_reading_after_end() -> Result<(), Box<dyn std::error::Error>> {
    let mut e = write::GzEncoder::new(Vec::new(), Compression::default());
    e.write_all(b"foo bar baz")?;
    let inner = e.finish()?;
    let mut d = read::GzDecoder::new(&inner[..]);
    let mut s = String::new();
    d.read_to_string(&mut s)?;
    assert_eq!(s, "foo bar baz");
    d.read_to_string(&mut s)?;
    assert_eq!(s, "foo bar baz");

    Ok(())
}

#[test]
fn qc_reader() {
    fn test(v: Vec<u8>) -> Result<bool, Box<dyn std::error::Error>> {
        let r = read::GzEncoder::new(&v[..], Compression::default());
        let mut r = read::GzDecoder::new(r);
        let mut v2 = Vec::new();
        r.read_to_end(&mut v2)?;
        Ok(v.into_iter().eq(v2))
    }

    ::quickcheck::quickcheck(test as fn(_) -> _);
}

#[test]
fn flush_after_write() -> Result<(), Box<dyn std::error::Error>> {
    let mut f = write::GzEncoder::new(Vec::new(), Compression::default());
    write!(f, "Hello world")?;
    f.flush()?;

    Ok(())
}
