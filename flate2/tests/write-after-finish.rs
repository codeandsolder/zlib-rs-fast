use std::io::{Read, Write};

use flate2::{read, write, Compression};

#[test]
fn deflate_write_after_finish() -> Result<(), Box<dyn std::error::Error>> {
    for input in [&b""[..], &b"hello"[..]] {
        let mut encoder = write::DeflateEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(input)?;
        encoder.try_finish()?;
        let finished = encoder.get_ref().clone();

        assert_finished(&mut encoder);
        assert_eq!(encoder.get_ref(), &finished);
        assert_eq!(encoder.reset(Vec::new())?, finished);
        encoder.write_all(b"new stream")?;
        let compressed = encoder.finish()?;
        let mut decoded = Vec::new();
        read::DeflateDecoder::new(&compressed[..]).read_to_end(&mut decoded)?;
        assert_eq!(decoded, b"new stream");
    }

    Ok(())
}

#[test]
fn zlib_write_after_finish() -> Result<(), Box<dyn std::error::Error>> {
    for input in [&b""[..], &b"hello"[..]] {
        let mut encoder = write::ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(input)?;
        encoder.try_finish()?;
        let finished = encoder.get_ref().clone();

        assert_finished(&mut encoder);
        assert_eq!(encoder.get_ref(), &finished);
        assert_eq!(encoder.reset(Vec::new())?, finished);
        encoder.write_all(b"new stream")?;
        let compressed = encoder.finish()?;
        let mut decoded = Vec::new();
        read::ZlibDecoder::new(&compressed[..]).read_to_end(&mut decoded)?;
        assert_eq!(decoded, b"new stream");
    }

    Ok(())
}

fn assert_finished(writer: &mut impl Write) {
    assert!(matches!(writer.write(b"X"), Ok(0) | Err(_)));
    assert!(matches!(writer.write(b"X"), Ok(0) | Err(_)));
    assert!(writer.write_all(b"X").is_err());
}
