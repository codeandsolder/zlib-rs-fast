use zlib_rs::{deflate::DeflateConfig, inflate::InflateConfig, ReturnCode};

use crate::helpers::{compress_slice_ng, uncompress_slice_ng};

#[test]
#[cfg_attr(miri, ignore = "calls into C code")]
fn end_to_end() {
    ::quickcheck::quickcheck(
        (|data: Vec<u8>, config: DeflateConfig| test(&data, config)) as fn(_, _) -> _,
    );
}

#[must_use]
pub fn test(data: &[u8], config: DeflateConfig) -> bool {
    // Only proceed if this is a valid config.
    let mut stream = libz_rs_sys::z_stream::default();
    if zlib_rs::deflate::init(&mut stream, config) != ReturnCode::Ok {
        return true;
    }
    // SAFETY: `init` succeeded and the stream remains live until `deflateEnd`.
    let output_bound = unsafe { libz_rs_sys::deflateBound_z(&raw mut stream, data.len()) };
    // SAFETY: the initialized stream is being finalized exactly once.
    unsafe { libz_rs_sys::deflateEnd(&raw mut stream) };

    let mut buffer_rs = vec![0; output_bound];
    let (compressed_rs, code) = zlib_rs::deflate::compress_slice(&mut buffer_rs, data, config);
    assert_eq!(ReturnCode::Ok, code);

    let mut buffer_ng = vec![0; output_bound];
    let (compressed_ng, code) = compress_slice_ng(&mut buffer_ng, data, config);
    assert_eq!(ReturnCode::Ok, code);

    // Two encoders may produce different, equally valid DEFLATE representations.
    // Cross-decode both outputs: every encoder/decoder combination must round-trip.
    let inflate_config = InflateConfig {
        window_bits: match config.window_bits {
            8 => 9,
            n => n,
        },
    };
    let mut buffer_ng = vec![0; data.len()];
    let mut buffer_rs = vec![0; data.len()];
    for (encoder, compressed) in [("zlib-rs", &*compressed_rs), ("zlib-ng", &*compressed_ng)] {
        let (decoded, code) = uncompress_slice_ng(&mut buffer_ng, compressed, inflate_config);
        assert_eq!(ReturnCode::Ok, code, "zlib-ng decoding {encoder}");
        assert_eq!(decoded, data, "zlib-ng decoding {encoder}");

        let (decoded, code) =
            zlib_rs::inflate::decompress_slice(&mut buffer_rs, compressed, inflate_config);
        assert_eq!(ReturnCode::Ok, code, "zlib-rs decoding {encoder}");
        assert_eq!(decoded, data, "zlib-rs decoding {encoder}");
    }
    true
}

#[test]
#[cfg_attr(miri, ignore = "calls into C code")]
fn different_valid_gzip_encodings_are_accepted() {
    // Reproducer from the differential fuzz target: the two encoders choose
    // distinct, valid DEFLATE representations of the same gzip payload.
    const DATA: [u8; 51] = [
        0xff, 0xfc, 0xd2, 0xda, 0xff, 0xff, 0xff, 0xff, 0x01, 0x00, 0x8c, 0x06, 0x08, 0xff, 0x00,
        0xff, 0x7f, 0xff, 0xff, 0xff, 0x74, 0x00, 0x00, 0x0e, 0x00, 0x44, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0xfd, 0xea, 0x00, 0x03, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];
    let config = DeflateConfig {
        level: 3,
        window_bits: 31,
        mem_level: 1,
        ..DeflateConfig::default()
    };
    assert!(test(&DATA, config));
}

#[test]
#[cfg_attr(miri, ignore = "calls into C code")]
fn raw_deflate_variants_are_cross_decoded() {
    for window_bits in [-9, -15] {
        let config = DeflateConfig {
            level: 3,
            window_bits,
            mem_level: 1,
            ..DeflateConfig::default()
        };
        assert!(test(
            b"raw DEFLATE needs the same cross-decoder coverage",
            config
        ));
    }
}
