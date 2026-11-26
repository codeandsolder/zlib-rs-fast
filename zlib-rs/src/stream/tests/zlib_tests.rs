#[cfg(test)]
mod tests {
    use crate::stream::io;
    use crate::stream::io::{Read, Write};
    use alloc::string::ToString;
    use alloc::vec::Vec;

    use rand::{rngs::SmallRng, RngExt, SeedableRng};

    use crate::stream::Compression;
    use crate::stream::{read, write};

    #[test]
    fn roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        let mut real = Vec::new();
        let mut w = write::ZlibEncoder::new(Vec::new(), Compression::default());
        let v = crate::stream::random_bytes().take(1024).collect::<Vec<_>>();
        let mut rng = SmallRng::seed_from_u64(0x0063_6875_6e6b);
        for _ in 0..200 {
            let to_write = &v[..rng.random_range(0..v.len())];
            real.extend(to_write.iter().copied());
            w.write_all(to_write)?;
        }
        let result = w.finish()?;
        let mut r = read::ZlibDecoder::new(&result[..]);
        let mut ret = Vec::new();
        r.read_to_end(&mut ret)?;
        assert_eq!(ret, real);

        Ok(())
    }

    #[test]
    fn drop_writes() -> Result<(), Box<dyn std::error::Error>> {
        let mut data = Vec::new();
        write::ZlibEncoder::new(&mut data, Compression::default()).write_all(b"foo")?;
        let mut r = read::ZlibDecoder::new(&data[..]);
        let mut ret = Vec::new();
        r.read_to_end(&mut ret)?;
        assert_eq!(ret, b"foo");

        Ok(())
    }

    #[test]
    fn total_in() -> Result<(), Box<dyn std::error::Error>> {
        let mut real = Vec::new();
        let mut w = write::ZlibEncoder::new(Vec::new(), Compression::default());
        let v = crate::stream::random_bytes().take(1024).collect::<Vec<_>>();
        let mut rng = SmallRng::seed_from_u64(0x0063_6875_6e6b);
        for _ in 0..200 {
            let to_write = &v[..rng.random_range(0..v.len())];
            real.extend(to_write.iter().copied());
            w.write_all(to_write)?;
        }
        let mut result = w.finish()?;

        let result_len = result.len();

        for _ in 0..200 {
            result.extend(v.iter().copied());
        }

        let mut r = read::ZlibDecoder::new(&result[..]);
        let mut ret = Vec::new();
        r.read_to_end(&mut ret)?;
        assert_eq!(ret, real);
        assert_eq!(r.total_in(), result_len as u64);

        Ok(())
    }

    #[test]
    fn roundtrip2() -> Result<(), Box<dyn std::error::Error>> {
        let v = crate::stream::random_bytes()
            .take(1024 * 1024)
            .collect::<Vec<_>>();
        let mut r = read::ZlibDecoder::new(read::ZlibEncoder::new(&v[..], Compression::default()));
        let mut ret = Vec::new();
        r.read_to_end(&mut ret)?;
        assert_eq!(ret, v);

        Ok(())
    }

    #[test]
    fn roundtrip3() -> Result<(), Box<dyn std::error::Error>> {
        let v = crate::stream::random_bytes()
            .take(1024 * 1024)
            .collect::<Vec<_>>();
        let mut w =
            write::ZlibEncoder::new(write::ZlibDecoder::new(Vec::new()), Compression::default());
        w.write_all(&v)?;
        let w = w.finish()?.finish()?;
        assert_eq!(w, v);

        Ok(())
    }

    #[test]
    fn reset_decoder() -> Result<(), Box<dyn std::error::Error>> {
        let input = crate::stream::random_bytes()
            .take(1024 * 1024)
            .collect::<Vec<_>>();
        let mut writer = write::ZlibEncoder::new(Vec::new(), Compression::default());
        writer.write_all(&input)?;
        let data = writer.finish()?;

        {
            let (mut first, mut second, mut fresh) = (Vec::new(), Vec::new(), Vec::new());
            let mut reader = read::ZlibDecoder::new(&data[..]);
            reader.read_to_end(&mut first)?;
            reader.reset(&data);
            reader.read_to_end(&mut second)?;

            let mut reader = read::ZlibDecoder::new(&data[..]);
            reader.read_to_end(&mut fresh)?;
            assert!(first == second && second == fresh && fresh == input);
        }

        {
            let mut writer = write::ZlibDecoder::new(Vec::new());
            writer.write_all(&data)?;
            let first = writer.reset(Vec::new())?;
            writer.write_all(&data)?;
            let second = writer.finish()?;

            let mut writer = write::ZlibDecoder::new(Vec::new());
            writer.write_all(&data)?;
            let fresh = writer.finish()?;
            assert!(first == second && second == fresh && fresh == input);
        }

        Ok(())
    }

    #[test]
    fn bad_input() -> Result<(), Box<dyn std::error::Error>> {
        // regress tests: previously caused a panic on drop
        let mut out: Vec<u8> = Vec::new();
        let data: Vec<u8> = (0..255).cycle().take(1024).collect();
        let mut w = write::ZlibDecoder::new(&mut out);
        match w.write_all(&data[..]) {
            Ok(()) => return Err("expected invalid input".into()),
            Err(e) => assert_eq!(e.kind(), io::ErrorKind::InvalidInput),
        }

        Ok(())
    }

    #[test]
    fn rejects_incomplete_stream() -> Result<(), Box<dyn std::error::Error>> {
        let mut encoder = write::ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(b"hello")?;
        let compressed = encoder.finish()?;

        for end in compressed.len() - 4..compressed.len() {
            let mut decoder = read::ZlibDecoder::new(&compressed[..end]);
            let error = crate::stream::io::copy(&mut decoder, &mut crate::stream::io::sink())
                .err()
                .ok_or("expected an error")?;
            assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
            assert_eq!(error.to_string(), "incomplete deflate stream");
        }

        Ok(())
    }

    #[test]
    fn write_decoder_rejects_incomplete_stream() -> Result<(), Box<dyn std::error::Error>> {
        fn assert_incomplete(error: &io::Error) {
            assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
            assert_eq!(error.to_string(), "incomplete deflate stream");
        }

        let mut encoder = write::ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(b"hello")?;
        let compressed = encoder.finish()?;

        for end in 0..compressed.len() {
            let truncated = &compressed[..end];

            let mut decoder = write::ZlibDecoder::new(Vec::new());
            decoder.write_all(truncated)?;
            assert_incomplete(&decoder.finish().err().ok_or("expected an error")?);

            let mut decoder = write::ZlibDecoder::new(Vec::new());
            decoder.write_all(truncated)?;
            assert_incomplete(&decoder.try_finish().err().ok_or("expected an error")?);
            decoder.reset(Vec::new())?;
        }

        let mut decoder = write::ZlibDecoder::new(Vec::new());
        decoder.write_all(&compressed)?;
        decoder.try_finish()?;
        decoder.try_finish()?;
        assert_eq!(
            decoder.finish()?,
            b"hello",
            "finishing an already finished decoder should not change its output"
        );

        Ok(())
    }

    #[test]
    fn write_decoder_reset_drains_pending_output() -> Result<(), Box<dyn std::error::Error>> {
        let input = vec![b'a'; 32 * 1024 + 1];
        let mut encoder = write::ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&input)?;

        let mut decoder = write::ZlibDecoder::new(Vec::new());
        decoder.write_all(&encoder.finish()?)?;
        let output = decoder.reset(Vec::new())?;

        assert!(
            output == input,
            "reset should preserve all decoder output byte-for-byte"
        );

        Ok(())
    }

    #[test]
    fn qc_reader() {
        fn test(v: Vec<u8>) -> Result<bool, Box<dyn std::error::Error>> {
            let mut r =
                read::ZlibDecoder::new(read::ZlibEncoder::new(&v[..], Compression::default()));
            let mut v2 = Vec::new();
            r.read_to_end(&mut v2)?;
            Ok(v.into_iter().eq(v2))
        }

        ::quickcheck::quickcheck(test as fn(_) -> _);
    }

    #[test]
    fn qc_writer() {
        fn test(v: Vec<u8>) -> Result<bool, Box<dyn std::error::Error>> {
            let mut w = write::ZlibEncoder::new(
                write::ZlibDecoder::new(Vec::new()),
                Compression::default(),
            );
            w.write_all(&v)?;
            Ok(v.into_iter().eq(w.finish()?.finish()?))
        }

        ::quickcheck::quickcheck(test as fn(_) -> _);
    }
}
