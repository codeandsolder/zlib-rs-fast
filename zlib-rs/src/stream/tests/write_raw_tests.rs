use super::*;
use crate::stream::Compression;
use alloc::string::String;
use alloc::vec::Vec;

const STR: &str = "Hello World Hello World Hello World Hello World Hello World \
        Hello World Hello World Hello World Hello World Hello World \
        Hello World Hello World Hello World Hello World Hello World \
        Hello World Hello World Hello World Hello World Hello World \
        Hello World Hello World Hello World Hello World Hello World";

// DeflateDecoder consumes one zlib archive and then returns 0 for subsequent writes, allowing any
// additional data to be consumed by the caller.
#[test]
fn decode_extra_data() -> Result<(), Box<dyn std::error::Error>> {
    let compressed = {
        let mut e = DeflateEncoder::new(Vec::new(), Compression::default());
        e.write_all(STR.as_ref())?;
        let mut b = e.finish()?;
        b.push(b'x');
        b
    };

    let mut writer = Vec::new();
    let mut decoder = DeflateDecoder::new(writer);
    let mut consumed_bytes = 0;
    loop {
        let n = decoder.write(&compressed[consumed_bytes..])?;
        if n == 0 {
            break;
        }
        consumed_bytes += n;
    }
    writer = decoder.finish()?;
    let actual = String::from_utf8(writer)?;
    assert_eq!(actual, STR);
    assert_eq!(&compressed[consumed_bytes..], b"x");

    Ok(())
}
