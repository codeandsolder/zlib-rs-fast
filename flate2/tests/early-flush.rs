use std::io::{Read, Write};

use flate2::read::GzDecoder as ReadGzDecoder;
use flate2::write::{
    DeflateDecoder, DeflateEncoder, GzDecoder, GzEncoder, ZlibDecoder, ZlibEncoder,
};
use flate2::Compression;

#[test]
fn decoder_flush_drains_pending_output_after_a_short_chunk(
) -> Result<(), Box<dyn std::error::Error>> {
    let input: Vec<u8> = (0_u32..65_537)
        .map(|i| (i / 263 % 251).to_le_bytes()[0])
        .collect();
    let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&input)?;
    encoder.flush()?;

    for end in 0..=encoder.get_ref().len() {
        let mut decoder = DeflateDecoder::new(Vec::new());
        for chunk in encoder.get_ref()[..end].chunks(256) {
            decoder.write_all(chunk)?;
        }
        decoder.flush()?;
        let flushed = decoder.get_ref().len();
        decoder.flush()?;
        assert_eq!(decoder.get_ref().len(), flushed, "compressed prefix {end}");
        assert!(decoder.get_ref() == &input[..flushed]);
    }

    Ok(())
}

#[test]
fn flush_drains_all_input_without_finishing() -> Result<(), Box<dyn std::error::Error>> {
    macro_rules! check {
        ($encoder:ident, $decoder:ident) => {
            for len in [0, 32_763, 32_768, 32_769, 65_531, 65_536, 65_537, 131_072] {
                for level in [Compression::none(), Compression::default()] {
                    let input = vec![0; len];
                    let mut encoder = $encoder::new(Vec::new(), level);
                    let mut decoder = $decoder::new(Vec::new());
                    let mut consumed = 0;

                    for round in 1..=2 {
                        encoder.write_all(&input)?;
                        encoder.flush()?;
                        encoder.flush()?;
                        decoder.write_all(&encoder.get_ref()[consumed..])?;
                        consumed = encoder.get_ref().len();
                        decoder.flush()?;
                        assert_eq!(decoder.get_ref().len(), round * len);
                        assert!(decoder.get_ref().iter().all(|&byte| byte == 0));
                    }

                    let compressed = encoder.finish()?;
                    decoder.write_all(&compressed[consumed..])?;
                    assert_eq!(decoder.finish()?.len(), 2 * len);
                }
            }
        };
    }

    check!(DeflateEncoder, DeflateDecoder);
    check!(ZlibEncoder, ZlibDecoder);
    check!(GzEncoder, GzDecoder);

    Ok(())
}

#[test]
fn smoke() -> Result<(), Box<dyn std::error::Error>> {
    let mut w = GzEncoder::new(Vec::new(), flate2::Compression::default());
    w.flush()?;
    w.write_all(b"hello")?;

    let bytes = w.finish()?;

    let mut r = ReadGzDecoder::new(&bytes[..]);
    let mut s = String::new();
    r.read_to_string(&mut s)?;
    assert_eq!(s, "hello");

    Ok(())
}
