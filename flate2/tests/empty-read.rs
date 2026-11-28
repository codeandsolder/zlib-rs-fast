use std::io::{Read, Write};

#[test]
fn deflate_decoder_empty_read() -> Result<(), Box<dyn std::error::Error>> {
    let original: &[u8] = b"Lorem ipsum dolor sit amet.";
    let mut compressor =
        flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
    compressor.write_all(original)?;
    let packed: Vec<u8> = compressor.finish()?;
    let mut inflater = flate2::read::DeflateDecoder::new(packed.as_slice());
    assert_eq!(inflater.read(&mut [])?, 0);
    let mut plain = Vec::new();
    inflater.read_to_end(&mut plain)?;
    assert_eq!(plain.as_slice(), original);

    Ok(())
}

#[test]
fn deflate_encoder_empty_read() -> Result<(), Box<dyn std::error::Error>> {
    let original: &[u8] = b"Lorem ipsum dolor sit amet.";
    let mut compressor =
        flate2::read::DeflateEncoder::new(original, flate2::Compression::default());
    assert_eq!(compressor.read(&mut [])?, 0);
    let mut packed = Vec::new();
    compressor.read_to_end(&mut packed)?;
    let mut inflater = flate2::read::DeflateDecoder::new(packed.as_slice());
    let mut plain = Vec::new();
    inflater.read_to_end(&mut plain)?;
    assert_eq!(plain.as_slice(), original);

    Ok(())
}

#[test]
fn gzip_decoder_empty_read() -> Result<(), Box<dyn std::error::Error>> {
    let original: &[u8] = b"Lorem ipsum dolor sit amet.";
    let mut compressor = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    compressor.write_all(original)?;
    let packed: Vec<u8> = compressor.finish()?;
    let mut inflater = flate2::read::GzDecoder::new(packed.as_slice());
    assert_eq!(inflater.read(&mut [])?, 0);
    let mut plain = Vec::new();
    inflater.read_to_end(&mut plain)?;
    assert_eq!(plain.as_slice(), original);

    Ok(())
}

fn gzip_encoder(original: &[u8], buffered: bool) -> Box<dyn Read + '_> {
    if buffered {
        Box::new(flate2::bufread::GzEncoder::new(
            original,
            flate2::Compression::default(),
        ))
    } else {
        Box::new(flate2::read::GzEncoder::new(
            original,
            flate2::Compression::default(),
        ))
    }
}

#[test]
fn gzip_encoder_empty_read() -> Result<(), Box<dyn std::error::Error>> {
    for original in [&b""[..], &b"Lorem ipsum dolor sit amet."[..]] {
        for buffered in [false, true] {
            let mut expected = Vec::new();
            gzip_encoder(original, buffered).read_to_end(&mut expected)?;
            for buffer_size in [1, 9, 10, 11, 64] {
                let mut compressor = gzip_encoder(original, buffered);
                let mut buffer = vec![0; buffer_size];
                let mut packed = Vec::new();
                loop {
                    assert_eq!(compressor.read(&mut [])?, 0);
                    assert_eq!(compressor.read(&mut [])?, 0);
                    let count = compressor.read(&mut buffer)?;
                    if count == 0 {
                        break;
                    }
                    packed.extend_from_slice(&buffer[..count]);
                }
                assert_eq!(compressor.read(&mut [])?, 0);
                assert_eq!(compressor.read(&mut buffer)?, 0);
                assert_eq!(packed, expected);
                let mut plain = Vec::new();
                flate2::read::GzDecoder::new(packed.as_slice()).read_to_end(&mut plain)?;
                assert_eq!(plain.as_slice(), original);
            }
        }
    }

    Ok(())
}

#[test]
fn gzip_encoder_empty_read_after_header() -> Result<(), Box<dyn std::error::Error>> {
    let original: &[u8] = b"An empty read after the header must not finish the stream.";
    for buffered in [false, true] {
        let mut compressor = gzip_encoder(original, buffered);
        let mut header = [0; 10];
        compressor.read_exact(&mut header)?;
        assert_eq!(compressor.read(&mut [])?, 0);
        let mut packed = header.to_vec();
        compressor.read_to_end(&mut packed)?;
        let mut plain = Vec::new();
        flate2::read::GzDecoder::new(packed.as_slice()).read_to_end(&mut plain)?;
        assert_eq!(plain.as_slice(), original);
    }

    Ok(())
}

#[test]
fn zlib_decoder_empty_read() -> Result<(), Box<dyn std::error::Error>> {
    let original: &[u8] = b"Lorem ipsum dolor sit amet.";
    let mut compressor =
        flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    compressor.write_all(original)?;
    let packed: Vec<u8> = compressor.finish()?;
    let mut inflater = flate2::read::ZlibDecoder::new(packed.as_slice());
    assert_eq!(inflater.read(&mut [])?, 0);
    let mut plain = Vec::new();
    inflater.read_to_end(&mut plain)?;
    assert_eq!(plain.as_slice(), original);

    Ok(())
}

#[test]
fn zlib_encoder_empty_read() -> Result<(), Box<dyn std::error::Error>> {
    let original: &[u8] = b"Lorem ipsum dolor sit amet.";
    let mut compressor = flate2::read::ZlibEncoder::new(original, flate2::Compression::default());
    assert_eq!(compressor.read(&mut [])?, 0);
    let mut packed = Vec::new();
    compressor.read_to_end(&mut packed)?;
    let mut inflater = flate2::read::ZlibDecoder::new(packed.as_slice());
    let mut plain = Vec::new();
    inflater.read_to_end(&mut plain)?;
    assert_eq!(plain.as_slice(), original);

    Ok(())
}
