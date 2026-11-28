use crate::stream::bufread::GzDecoder;
use crate::stream::gz::write;
use crate::stream::io::{Read, Write};
use crate::stream::Compression;
use alloc::vec::Vec;

// GzDecoder consumes one gzip member and then returns 0 for subsequent reads, allowing any
// additional data to be consumed by the caller.
#[test]
fn decode_extra_data() -> Result<(), Box<dyn std::error::Error>> {
    let expected = "Hello World";

    let compressed = {
        let mut e = write::GzEncoder::new(Vec::new(), Compression::default());
        e.write_all(expected.as_ref())?;
        let mut b = e.finish()?;
        b.push(b'x');
        b
    };

    let mut output = Vec::new();
    let mut decoder = GzDecoder::new(compressed.as_slice());
    let decoded_bytes = decoder.read_to_end(&mut output)?;
    assert_eq!(decoded_bytes, output.len());
    let actual = core::str::from_utf8(&output)?;
    assert_eq!(
        actual, expected,
        "after decompression we obtain the original input"
    );

    output.clear();
    assert_eq!(
        decoder.read(&mut [])?,
        0,
        "subsequent read of decoder returns 0, but inner reader can return additional data"
    );
    let mut reader = decoder.into_inner();
    assert_eq!(
        reader.read_to_end(&mut output)?,
        1,
        "extra data is accessible in underlying buf-read"
    );
    assert_eq!(output, b"x");

    Ok(())
}

fn compress_data(data: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    use crate::stream::write::GzEncoder;
    use crate::stream::Compression;

    let mut e = GzEncoder::new(Vec::new(), Compression::default());
    e.write_all(data)?;
    Ok(e.finish()?)
}

#[test]
fn decode_with_reset() -> Result<(), Box<dyn std::error::Error>> {
    let data1 = b"Hello World";
    let data2 = b"Goodbye World";

    let compressed1 = compress_data(data1)?;
    let compressed2 = compress_data(data2)?;

    let mut output = Vec::new();
    let mut decoder = GzDecoder::new(compressed1.as_slice());
    decoder.read_to_end(&mut output)?;
    assert_eq!(output, data1);

    output.clear();
    decoder.reset(compressed2.as_slice());
    decoder.read_to_end(&mut output)?;
    assert_eq!(output, data2);

    Ok(())
}

#[test]
fn decode_with_reset_after_corruption() -> Result<(), Box<dyn std::error::Error>> {
    let valid_data = b"Hello World";
    let valid_compressed = compress_data(valid_data)?;

    // Create a corrupted payload (valid gzip header but corrupted body)
    let mut corrupted = valid_compressed.clone();
    assert!(corrupted.len() >= 14);
    corrupted[12] ^= 0xFF;
    corrupted[13] ^= 0xFF;

    // Try to decode corrupted data
    let mut decoder = GzDecoder::new(corrupted.as_slice());
    let mut output = Vec::new();
    let _ = decoder
        .read_to_end(&mut output)
        .err()
        .ok_or("expected an error")?;

    // Reset with valid payload and decode
    decoder.reset(valid_compressed.as_slice());
    output.clear();
    decoder.read_to_end(&mut output)?;
    assert_eq!(output, valid_data);

    Ok(())
}
