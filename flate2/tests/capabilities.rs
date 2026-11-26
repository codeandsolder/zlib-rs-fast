//! Validate that certain feature-gated functionality is still available.
use flate2::{Compress, Compression, Decompress, FlushCompress, FlushDecompress};

// Unsupported for `miniz_oxide`.
#[test]
fn compress_new_with_window_bits_is_present_and_works() -> Result<(), Box<dyn std::error::Error>> {
    let string = b"hello world";

    // Test with window_bits = 9 (minimum)
    let mut packed = Vec::with_capacity(1024);
    let mut compressor = Compress::new_with_window_bits(Compression::default(), true, 9);
    compressor.compress_vec(string, &mut packed, FlushCompress::Finish)?;
    assert_ne!(packed.len(), 0);

    let mut inflater = Decompress::new_with_window_bits(true, 9);
    let mut plain = [0; 1024];
    inflater.decompress(&packed, &mut plain, FlushDecompress::Finish)?;
    assert_eq!(&plain[..string.len()], string);

    // Test with window_bits = 15 (maximum)
    let mut packed = Vec::with_capacity(1024);
    let mut compressor = Compress::new_with_window_bits(Compression::default(), false, 15);
    compressor.compress_vec(string, &mut packed, FlushCompress::Finish)?;
    assert_ne!(packed.len(), 0);

    let mut inflater = Decompress::new_with_window_bits(false, 15);
    let mut plain = [0; 1024];
    inflater.decompress(&packed, &mut plain, FlushDecompress::Finish)?;
    assert_eq!(&plain[..string.len()], string);

    Ok(())
}

// Unsupported for `miniz_oxide`.
#[test]
fn decompress_new_gzip_window_bits_is_present_and_works() -> Result<(), Box<dyn std::error::Error>>
{
    let string = b"hello world";

    // Test with different window_bits values
    for window_bits in [9, 12, 15] {
        let mut packed = Vec::with_capacity(1024);
        let mut compressor = Compress::new_gzip(Compression::default(), window_bits);
        compressor.compress_vec(string, &mut packed, FlushCompress::Finish)?;

        let mut inflater = Decompress::new_gzip(window_bits);
        let mut plain = [0; 1024];
        inflater.decompress(&packed, &mut plain, FlushDecompress::Finish)?;
        assert_eq!(
            &plain[..string.len()],
            string,
            "Failed with window_bits={window_bits}"
        );
    }

    Ok(())
}

// Unsupported for `miniz_oxide`.
#[test]
#[should_panic(expected = "window_bits must be within 9 ..= 15")]
fn compress_new_with_window_bits_invalid_low() {
    let _ = Compress::new_with_window_bits(Compression::default(), true, 8);
}

// Unsupported for `miniz_oxide`.
#[test]
#[should_panic(expected = "window_bits must be within 9 ..= 15")]
fn compress_new_with_window_bits_invalid_high() {
    let _ = Compress::new_with_window_bits(Compression::default(), true, 16);
}

// Unsupported for `miniz_oxide`.
#[test]
#[should_panic(expected = "window_bits must be within 9 ..= 15")]
fn compress_new_gzip_invalid_low() {
    let _ = Compress::new_gzip(Compression::default(), 8);
}

// Unsupported for `miniz_oxide`.
#[test]
#[should_panic(expected = "window_bits must be within 9 ..= 15")]
fn compress_new_gzip_invalid_high() {
    let _ = Compress::new_gzip(Compression::default(), 16);
}

// Unsupported for `miniz_oxide`.
#[test]
fn set_dictionary_with_zlib_header() -> Result<(), Box<dyn std::error::Error>> {
    let string = b"hello, hello!";
    let dictionary = b"hello";

    let mut packed = Vec::with_capacity(1024);

    let mut compressor = Compress::new(Compression::default(), true);

    let dictionary_adler = compressor.set_dictionary(dictionary)?;

    compressor.compress_vec(string, &mut packed, FlushCompress::Finish)?;

    assert_eq!(compressor.total_in(), string.len() as u64);
    assert_eq!(compressor.total_out(), packed.len() as u64);

    let mut inflater = Decompress::new(true);
    let mut plain = [0; 1024];
    let decompress_error = inflater
        .decompress(&packed, &mut plain, FlushDecompress::Finish)
        .err()
        .ok_or("expected an error")?;

    let required_adler = decompress_error
        .needs_dictionary()
        .ok_or("missing dictionary checksum")?;

    assert_eq!(required_adler, dictionary_adler,
               "the Adler-32 checksum should match the value when the dictionary was set on the compressor");

    let actual_adler = inflater.set_dictionary(dictionary)?;

    assert_eq!(required_adler, actual_adler);

    // Decompress the rest of the input to the remainder of the output buffer
    let total_in = inflater.total_in();
    let total_out = inflater.total_out();

    let decompress_result = inflater.decompress(
        &packed[usize::try_from(total_in)?..],
        &mut plain[usize::try_from(total_out)?..],
        FlushDecompress::Finish,
    );
    assert!(decompress_result.is_ok());

    assert_eq!(&plain[..usize::try_from(inflater.total_out())?], string);

    Ok(())
}

// Unsupported for `miniz_oxide`.
#[test]
fn set_dictionary_raw() -> Result<(), Box<dyn std::error::Error>> {
    let string = b"hello, hello!";
    let dictionary = b"hello";

    let mut packed = Vec::with_capacity(1024);

    let mut compressor = Compress::new(Compression::default(), false);

    compressor.set_dictionary(dictionary)?;

    compressor.compress_vec(string, &mut packed, FlushCompress::Finish)?;

    assert_eq!(compressor.total_in(), string.len() as u64);
    assert_eq!(compressor.total_out(), packed.len() as u64);

    let mut inflater = Decompress::new(false);

    inflater.set_dictionary(dictionary)?;

    let mut plain = [0; 1024];
    let decompress_result = inflater.decompress(&packed, &mut plain, FlushDecompress::Finish);

    assert!(decompress_result.is_ok());

    assert_eq!(&plain[..usize::try_from(inflater.total_out())?], string);

    Ok(())
}

// Unsupported for `miniz_oxide`.
#[test]
fn compression_levels_are_effective() -> Result<(), Box<dyn std::error::Error>> {
    let input = b"hello hello hello hello hello hello hello hello";

    // Compress with no compression
    let mut encoded_none = Vec::new();
    Compress::new(Compression::none(), true).compress_vec(
        input,
        &mut encoded_none,
        FlushCompress::Finish,
    )?;

    // Compress with best compression
    let mut encoded_best = Vec::new();
    Compress::new(Compression::best(), true).compress_vec(
        input,
        &mut encoded_best,
        FlushCompress::Finish,
    )?;

    assert!(
        encoded_best.len() <= encoded_none.len(),
        "best compression produced larger output than no compression: best={}, none={}",
        encoded_best.len(),
        encoded_none.len(),
    );

    Ok(())
}

// Unsupported for `miniz_oxide`.
#[test]
fn set_level_is_effective() -> Result<(), Box<dyn std::error::Error>> {
    let input = b"hello hello hello hello hello hello hello hello";
    let no_compression = Compression::none();
    let best_compression = Compression::best();

    // Compress with no compression
    let mut encoded_none = Vec::new();
    let mut compress = Compress::new(best_compression, true);
    compress.set_level(no_compression)?;
    compress.compress_vec(input, &mut encoded_none, FlushCompress::Finish)?;

    // Compress with best compression
    let mut encoded_best = Vec::new();
    let mut compress = Compress::new(no_compression, true);
    compress.set_level(best_compression)?;
    compress.compress_vec(input, &mut encoded_best, FlushCompress::Finish)?;

    assert!(
        encoded_best.len() <= encoded_none.len(),
        "best compression produced larger output than no compression: best={}, none={}",
        encoded_best.len(),
        encoded_none.len(),
    );

    Ok(())
}
