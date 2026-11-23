#[cfg(test)]
mod tests {
    use std::io::{Read, Write};

    #[test]
    fn flate2_zlib_backend_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        let source = b"Zlib maintained fork through the public flate2 backend".repeat(40);
        let mut compressor =
            flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        compressor.write_all(&source)?;
        let compressed = compressor.finish()?;
        let mut decoder = flate2::read::ZlibDecoder::new(compressed.as_slice());
        let mut result = Vec::new();
        decoder.read_to_end(&mut result)?;
        assert_eq!(result, source);
        Ok(())
    }
}
