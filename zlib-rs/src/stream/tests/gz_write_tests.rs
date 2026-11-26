use super::*;
use alloc::string::String;

const STR: &str = "Hello World Hello World Hello World Hello World Hello World \
                               Hello World Hello World Hello World Hello World Hello World \
                               Hello World Hello World Hello World Hello World Hello World \
                               Hello World Hello World Hello World Hello World Hello World \
                               Hello World Hello World Hello World Hello World Hello World";

#[test]
fn decode_writer_one_chunk() -> Result<(), Box<dyn std::error::Error>> {
    let mut e = GzEncoder::new(Vec::new(), Compression::default());
    e.write_all(STR.as_ref())?;
    let bytes = e.finish()?;

    let mut writer = Vec::new();
    let mut decoder = GzDecoder::new(writer);
    let n = decoder.write(&bytes[..])?;
    decoder.write_all(&bytes[n..])?;
    decoder.try_finish()?;
    writer = decoder.finish()?;
    let return_string = String::from_utf8(writer)?;
    assert_eq!(return_string, STR);

    Ok(())
}

#[test]
fn decode_writer_partial_header() -> Result<(), Box<dyn std::error::Error>> {
    let mut e = GzEncoder::new(Vec::new(), Compression::default());
    e.write_all(STR.as_ref())?;
    let bytes = e.finish()?;

    let mut writer = Vec::new();
    let mut decoder = GzDecoder::new(writer);
    assert_eq!(decoder.write(&bytes[..5])?, 5);
    let n = decoder.write(&bytes[5..])?;
    if n < bytes.len() - 5 {
        decoder.write_all(&bytes[n + 5..])?;
    }
    writer = decoder.finish()?;
    let return_string = String::from_utf8(writer)?;
    assert_eq!(return_string, STR);

    Ok(())
}

#[test]
fn decode_writer_partial_header_filename() -> Result<(), Box<dyn std::error::Error>> {
    let filename = "test.txt";
    let mut e = GzBuilder::new()
        .filename(filename)
        .read(STR.as_bytes(), Compression::default());
    let mut bytes = Vec::new();
    e.read_to_end(&mut bytes)?;

    let mut writer = Vec::new();
    let mut decoder = GzDecoder::new(writer);
    assert_eq!(decoder.write(&bytes[..12])?, 12);
    let n = decoder.write(&bytes[12..])?;
    if n < bytes.len() - 12 {
        decoder.write_all(&bytes[n + 12..])?;
    }
    assert_eq!(
        decoder
            .header()
            .ok_or("missing gzip header")?
            .filename()
            .ok_or("missing gzip filename")?,
        filename.as_bytes()
    );
    writer = decoder.finish()?;
    let return_string = String::from_utf8(writer)?;
    assert_eq!(return_string, STR);

    Ok(())
}

#[test]
fn decode_writer_partial_header_comment() -> Result<(), Box<dyn std::error::Error>> {
    let comment = "test comment";
    let mut e = GzBuilder::new()
        .comment(comment)
        .read(STR.as_bytes(), Compression::default());
    let mut bytes = Vec::new();
    e.read_to_end(&mut bytes)?;

    let mut writer = Vec::new();
    let mut decoder = GzDecoder::new(writer);
    assert_eq!(decoder.write(&bytes[..12])?, 12);
    let n = decoder.write(&bytes[12..])?;
    if n < bytes.len() - 12 {
        decoder.write_all(&bytes[n + 12..])?;
    }
    assert_eq!(
        decoder
            .header()
            .ok_or("missing gzip header")?
            .comment()
            .ok_or("missing gzip comment")?,
        comment.as_bytes()
    );
    writer = decoder.finish()?;
    let return_string = String::from_utf8(writer)?;
    assert_eq!(return_string, STR);

    Ok(())
}

#[test]
fn decode_writer_exact_header() -> Result<(), Box<dyn std::error::Error>> {
    let mut e = GzEncoder::new(Vec::new(), Compression::default());
    e.write_all(STR.as_ref())?;
    let bytes = e.finish()?;

    let mut writer = Vec::new();
    let mut decoder = GzDecoder::new(writer);
    assert_eq!(decoder.write(&bytes[..10])?, 10);
    decoder.write_all(&bytes[10..])?;
    writer = decoder.finish()?;
    let return_string = String::from_utf8(writer)?;
    assert_eq!(return_string, STR);

    Ok(())
}

#[test]
fn decode_writer_partial_crc() -> Result<(), Box<dyn std::error::Error>> {
    let mut e = GzEncoder::new(Vec::new(), Compression::default());
    e.write_all(STR.as_ref())?;
    let bytes = e.finish()?;

    let mut writer = Vec::new();
    let mut decoder = GzDecoder::new(writer);
    let l = bytes.len() - 5;
    let n = decoder.write(&bytes[..l])?;
    decoder.write_all(&bytes[n..])?;
    writer = decoder.finish()?;
    let return_string = String::from_utf8(writer)?;
    assert_eq!(return_string, STR);

    Ok(())
}

// Two or more gzip files concatenated form a multi-member gzip file. MultiGzDecoder will
// concatenate the decoded contents of all members.
#[test]
fn decode_multi_writer() -> Result<(), Box<dyn std::error::Error>> {
    let mut e = GzEncoder::new(Vec::new(), Compression::default());
    e.write_all(STR.as_ref())?;
    let bytes = e.finish()?.repeat(2);

    let mut writer = Vec::new();
    let mut decoder = MultiGzDecoder::new(writer);
    let mut count = 0;
    while count < bytes.len() {
        let n = decoder.write(&bytes[count..])?;
        assert!(n != 0);
        count += n;
    }
    writer = decoder.finish()?;
    let return_string = String::from_utf8(writer)?;
    let expected = STR.repeat(2);
    assert_eq!(return_string, expected);

    Ok(())
}

// GzDecoder consumes one gzip member and then returns 0 for subsequent writes, allowing any
// additional data to be consumed by the caller.
#[test]
fn decode_extra_data() -> Result<(), Box<dyn std::error::Error>> {
    let compressed = {
        let mut e = GzEncoder::new(Vec::new(), Compression::default());
        e.write_all(STR.as_ref())?;
        let mut b = e.finish()?;
        b.push(b'x');
        b
    };

    let mut writer = Vec::new();
    let mut decoder = GzDecoder::new(writer);
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
