use crate::stream::io::Write;
use alloc::vec::Vec;

use crate::stream::write;
use crate::stream::{Compression, Decompress, FlushDecompress};

use crate::stream::{Compress, FlushCompress};

#[test]
fn issue51() -> Result<(), Box<dyn std::error::Error>> {
    let data = [
        0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0xb3, 0xc9, 0x28, 0xc9, 0xcd,
        0xb1, 0xe3, 0xe5, 0xb2, 0xc9, 0x48, 0x4d, 0x4c, 0xb1, 0xb3, 0x29, 0xc9, 0x2c, 0xc9, 0x49,
        0xb5, 0x33, 0x31, 0x30, 0x51, 0xf0, 0xcb, 0x2f, 0x51, 0x70, 0xcb, 0x2f, 0xcd, 0x4b, 0xb1,
        0xd1, 0x87, 0x08, 0xda, 0xe8, 0x83, 0x95, 0x00, 0x95, 0x26, 0xe5, 0xa7, 0x54, 0x2a, 0x24,
        0xa5, 0x27, 0xe7, 0xe7, 0xe4, 0x17, 0xd9, 0x2a, 0x95, 0x67, 0x64, 0x96, 0xa4, 0x2a, 0x81,
        0x8c, 0x48, 0x4e, 0xcd, 0x2b, 0x49, 0x2d, 0xb2, 0xb3, 0xc9, 0x30, 0x44, 0x37, 0x01, 0x28,
        0x62, 0xa3, 0x0f, 0x95, 0x06, 0xd9, 0x05, 0x54, 0x04, 0xe5, 0xe5, 0xa5, 0x67, 0xe6, 0x55,
        0xe8, 0x1b, 0xea, 0x99, 0xe9, 0x19, 0x21, 0xab, 0xd0, 0x07, 0xd9, 0x01, 0x32, 0x53, 0x1f,
        0xea, 0x3e, 0x00, 0x94, 0x85, 0xeb, 0xe4, 0xa8, 0x00, 0x00, 0x00,
    ];

    let mut plain = Vec::with_capacity(data.len() * 2);

    let mut d = Decompress::new(false);
    // decompressed whole deflate stream
    d.decompress_vec(&data[10..], &mut plain, FlushDecompress::Finish)?;

    // decompress data that has nothing to do with the deflate stream (this
    // used to panic)
    drop(d.decompress_vec(&[0], &mut plain, FlushDecompress::None));

    Ok(())
}

#[test]
fn reset() -> Result<(), Box<dyn std::error::Error>> {
    let string = b"hello world";
    let mut zlib = Vec::new();
    let mut deflate = Vec::new();

    let comp = Compression::default();
    write::ZlibEncoder::new(&mut zlib, comp).write_all(string)?;
    write::DeflateEncoder::new(&mut deflate, comp).write_all(string)?;

    let mut dst = [0; 1024];
    let mut inflater = Decompress::new(true);
    inflater.decompress(&zlib, &mut dst, FlushDecompress::Finish)?;
    assert_eq!(inflater.total_out(), string.len() as u64);
    assert!(dst.starts_with(string));

    inflater.reset(false);
    inflater.decompress(&deflate, &mut dst, FlushDecompress::Finish)?;
    assert_eq!(inflater.total_out(), string.len() as u64);
    assert!(dst.starts_with(string));

    Ok(())
}

#[test]
fn unfinished_compressor_can_be_dropped() -> Result<(), Box<dyn std::error::Error>> {
    let mut compressor = Compress::new(Compression::default(), false);
    compressor.compress(b"hello", &mut [0; 1024], FlushCompress::None)?;
    assert_eq!(compressor.total_in(), 5);

    Ok(())
}

#[test]
fn reset_preserves_gzip_configuration() -> Result<(), Box<dyn std::error::Error>> {
    let string = b"hello, hello!";

    let mut packed = Vec::with_capacity(1024);
    let mut compressor = Compress::new_gzip(Compression::default(), 9);
    compressor.compress_vec(string, &mut packed, FlushCompress::Finish)?;

    let mut inflater = Decompress::new_gzip(9);
    let mut plain = [0; 1024];
    inflater.decompress(&packed, &mut plain, FlushDecompress::Finish)?;
    assert_eq!(&plain[..usize::try_from(inflater.total_out())?], string);

    inflater.reset(true);
    let mut plain = [0; 1024];
    inflater.decompress(&packed, &mut plain, FlushDecompress::Finish)?;
    assert_eq!(&plain[..usize::try_from(inflater.total_out())?], string);

    Ok(())
}

#[test]
fn reset_preserves_window_bits() -> Result<(), Box<dyn std::error::Error>> {
    // A valid zlib header declaring a 32 KiB (15-bit) window: CMF 0x78, FLG 0x9c.
    let header_15_bit_window = [0x78, 0x9c];
    let mut plain = [0; 64];

    let mut inflater = Decompress::new_with_window_bits(true, 9);
    inflater
        .decompress(&header_15_bit_window, &mut plain, FlushDecompress::None)
        .err()
        .ok_or("expected an error")?;

    inflater.reset(true);
    inflater
        .decompress(&header_15_bit_window, &mut plain, FlushDecompress::None)
        .err()
        .ok_or("expected an error")?;

    Ok(())
}

#[test]
fn test_gzip_flate() -> Result<(), Box<dyn std::error::Error>> {
    let string = b"hello, hello!";

    let mut packed = Vec::with_capacity(1024);

    let mut compressor = Compress::new_gzip(Compression::default(), 9);

    compressor.compress_vec(string, &mut packed, FlushCompress::Finish)?;

    assert_eq!(compressor.total_in(), string.len() as u64);
    assert_eq!(compressor.total_out(), packed.len() as u64);

    let mut inflater = Decompress::new_gzip(9);

    let mut plain = [0; 1024];
    inflater.decompress(&packed, &mut plain, FlushDecompress::Finish)?;

    assert_eq!(&plain[..usize::try_from(inflater.total_out())?], string);

    Ok(())
}

#[test]
fn test_error_message() -> Result<(), Box<dyn std::error::Error>> {
    let mut inflater = Decompress::new(false);
    let mut plain = [0; 128];
    let garbage = b"xbvxzi";

    let err = inflater
        .decompress(garbage, &mut plain, FlushDecompress::Finish)
        .err()
        .ok_or("expected an error")?;

    assert_eq!(err.message(), Some("invalid stored block lengths"));

    Ok(())
}

fn compress_with_flush(flush: FlushCompress) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let incompressible = (0..=255).collect::<Vec<u8>>();
    let mut output = vec![0; 1024];

    // Feed in the incompressible data followed by the indicated flush type.
    let mut w = Compress::new(Compression::default(), false);
    w.compress(&incompressible, &mut output, flush)?;

    if flush != FlushCompress::None {
        // The first instance of incompressible input should have been written uncompressed.
        assert!(w.total_out() >= 261);
        assert_eq!(&output[0..5], &[0, 0, 1, 0xff, !1]);
        assert_eq!(&output[5..261], &incompressible);
    }

    // Feed in the same data again.
    let len = usize::try_from(w.total_out())?;
    w.compress(&incompressible, &mut output[len..], FlushCompress::Finish)?;

    if flush != FlushCompress::Full {
        // This time, the data should have been compressed (because it is an exact duplicate of
        // the earlier block).
        assert!(w.total_out() < 300);
    }

    // Assert that all input has been processed.
    assert_eq!(w.total_in(), 256 * 2);

    output.resize(usize::try_from(w.total_out())?, 0);
    Ok(output)
}

#[test]
fn test_partial_flush() -> Result<(), Box<dyn std::error::Error>> {
    let output = compress_with_flush(FlushCompress::Partial)?;

    // Check for partial flush marker.
    assert_eq!(output[261], 0x2);
    assert_eq!(output[262] & 0x7, 0x4);

    Ok(())
}

#[test]
fn test_sync_flush() -> Result<(), Box<dyn std::error::Error>> {
    let output = compress_with_flush(FlushCompress::Sync)?;

    // Check for sync flush marker.
    assert_eq!(&output[261..][..5], &[0, 0, 0, 0xff, 0xff]);

    Ok(())
}

#[test]
fn test_full_flush() -> Result<(), Box<dyn std::error::Error>> {
    let output = compress_with_flush(FlushCompress::Full)?;
    assert_eq!(output.len(), 527);

    // Check for sync flush marker.
    assert_eq!(&output[261..][..5], &[0, 0, 0, 0xff, 0xff]);

    // Check that the second instance of incompressible input was also written uncompressed.
    assert_eq!(&output[266..][..5], &[1, 0, 1, 0xff, !1]);

    Ok(())
}
