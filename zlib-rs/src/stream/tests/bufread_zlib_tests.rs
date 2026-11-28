use crate::stream::bufread::ZlibDecoder;
use crate::stream::io::{Read, Write};
use crate::stream::write;
use crate::stream::Compression;
use alloc::vec::Vec;

// ZlibDecoder consumes one zlib archive and then returns 0 for subsequent reads, allowing any
// additional data to be consumed by the caller.
#[test]
fn decode_extra_data() -> Result<(), Box<dyn std::error::Error>> {
    let expected = "Hello World";

    let compressed = {
        let mut e = write::ZlibEncoder::new(Vec::new(), Compression::default());
        e.write_all(expected.as_ref())?;
        let mut b = e.finish()?;
        b.push(b'x');
        b
    };

    let mut output = Vec::new();
    let mut decoder = ZlibDecoder::new(compressed.as_slice());
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
