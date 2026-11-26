#[cfg(test)]
mod tests {
    use crate::stream::io::{Read, Write};
    use alloc::string::ToString;
    use alloc::vec::Vec;

    use rand::{rngs::SmallRng, RngExt, SeedableRng};

    use crate::stream::Compression;
    use crate::stream::{read, write};

    #[test]
    fn roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        let mut real = Vec::new();
        let mut w = write::DeflateEncoder::new(Vec::new(), Compression::default());
        let v = crate::stream::random_bytes().take(1024).collect::<Vec<_>>();
        let mut rng = SmallRng::seed_from_u64(0x0063_6875_6e6b);
        for _ in 0..200 {
            let to_write = &v[..rng.random_range(0..v.len())];
            real.extend(to_write.iter().copied());
            w.write_all(to_write)?;
        }
        let result = w.finish()?;
        let mut r = read::DeflateDecoder::new(&result[..]);
        let mut ret = Vec::new();
        r.read_to_end(&mut ret)?;
        assert_eq!(ret, real);

        Ok(())
    }

    #[test]
    fn drop_writes() -> Result<(), Box<dyn std::error::Error>> {
        let mut data = Vec::new();
        write::DeflateEncoder::new(&mut data, Compression::default()).write_all(b"foo")?;
        let mut r = read::DeflateDecoder::new(&data[..]);
        let mut ret = Vec::new();
        r.read_to_end(&mut ret)?;
        assert_eq!(ret, b"foo");

        Ok(())
    }

    #[test]
    fn total_in() -> Result<(), Box<dyn std::error::Error>> {
        let mut real = Vec::new();
        let mut w = write::DeflateEncoder::new(Vec::new(), Compression::default());
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

        let mut r = read::DeflateDecoder::new(&result[..]);
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
        let mut r =
            read::DeflateDecoder::new(read::DeflateEncoder::new(&v[..], Compression::default()));
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
        let mut w = write::DeflateEncoder::new(
            write::DeflateDecoder::new(Vec::new()),
            Compression::default(),
        );
        w.write_all(&v)?;
        let w = w.finish()?.finish()?;
        assert_eq!(w, v);

        Ok(())
    }

    #[test]
    fn reset_writer() -> Result<(), Box<dyn std::error::Error>> {
        let v = crate::stream::random_bytes()
            .take(1024 * 1024)
            .collect::<Vec<_>>();
        let mut w = write::DeflateEncoder::new(Vec::new(), Compression::default());
        w.write_all(&v)?;
        let a = w.reset(Vec::new())?;
        w.write_all(&v)?;
        let b = w.finish()?;

        let mut w = write::DeflateEncoder::new(Vec::new(), Compression::default());
        w.write_all(&v)?;
        let c = w.finish()?;
        assert!(a == b && b == c);

        Ok(())
    }

    #[test]
    fn reset_reader() -> Result<(), Box<dyn std::error::Error>> {
        let v = crate::stream::random_bytes()
            .take(1024 * 1024)
            .collect::<Vec<_>>();
        let (mut a, mut b, mut c) = (Vec::new(), Vec::new(), Vec::new());
        let mut r = read::DeflateEncoder::new(&v[..], Compression::default());
        r.read_to_end(&mut a)?;
        r.reset(&v[..]);
        r.read_to_end(&mut b)?;

        let mut r = read::DeflateEncoder::new(&v[..], Compression::default());
        r.read_to_end(&mut c)?;
        assert!(a == b && b == c);

        Ok(())
    }

    #[test]
    fn reset_decoder() -> Result<(), Box<dyn std::error::Error>> {
        let v = crate::stream::random_bytes()
            .take(1024 * 1024)
            .collect::<Vec<_>>();
        let mut w = write::DeflateEncoder::new(Vec::new(), Compression::default());
        w.write_all(&v)?;
        let data = w.finish()?;

        {
            let (mut a, mut b, mut c) = (Vec::new(), Vec::new(), Vec::new());
            let mut r = read::DeflateDecoder::new(&data[..]);
            r.read_to_end(&mut a)?;
            r.reset(&data);
            r.read_to_end(&mut b)?;

            let mut r = read::DeflateDecoder::new(&data[..]);
            r.read_to_end(&mut c)?;
            assert!(a == b && b == c && c == v);
        }

        {
            let mut w = write::DeflateDecoder::new(Vec::new());
            w.write_all(&data)?;
            let a = w.reset(Vec::new())?;
            w.write_all(&data)?;
            let b = w.finish()?;

            let mut w = write::DeflateDecoder::new(Vec::new());
            w.write_all(&data)?;
            let c = w.finish()?;
            assert!(a == b && b == c && c == v);
        }

        Ok(())
    }

    #[test]
    fn zero_length_read_with_data() -> Result<(), Box<dyn std::error::Error>> {
        let m = vec![3u8; 128 * 1024 + 1];
        let mut c = read::DeflateEncoder::new(&m[..], Compression::default());

        let mut result = Vec::new();
        c.read_to_end(&mut result)?;

        let mut d = read::DeflateDecoder::new(&result[..]);
        assert_eq!(d.read(&mut [])?, 0);

        Ok(())
    }

    #[test]
    fn rejects_incomplete_stream() -> Result<(), Box<dyn std::error::Error>> {
        for input in [b"1".as_slice(), b"12", b"123", b"1234"] {
            let mut decoder = read::DeflateDecoder::new(input);
            let error = crate::stream::io::copy(&mut decoder, &mut crate::stream::io::sink())
                .err()
                .ok_or("expected an error")?;
            assert_eq!(error.kind(), crate::stream::io::ErrorKind::UnexpectedEof);
            assert_eq!(error.to_string(), "incomplete deflate stream");
        }

        Ok(())
    }

    #[test]
    fn write_decoder_rejects_incomplete_stream() -> Result<(), Box<dyn std::error::Error>> {
        fn assert_incomplete(error: &crate::stream::io::Error) {
            assert_eq!(error.kind(), crate::stream::io::ErrorKind::UnexpectedEof);
            assert_eq!(error.to_string(), "incomplete deflate stream");
        }

        let mut encoder = write::DeflateEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(b"hello")?;
        let compressed = encoder.finish()?;

        for end in 0..compressed.len() {
            let truncated = &compressed[..end];

            let mut decoder = write::DeflateDecoder::new(Vec::new());
            decoder.write_all(truncated)?;
            assert_incomplete(&decoder.finish().err().ok_or("expected an error")?);

            let mut decoder = write::DeflateDecoder::new(Vec::new());
            decoder.write_all(truncated)?;
            assert_incomplete(&decoder.try_finish().err().ok_or("expected an error")?);
            decoder.reset(Vec::new())?;
        }

        let mut decoder = write::DeflateDecoder::new(Vec::new());
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
        let mut encoder = write::DeflateEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&input)?;

        let mut decoder = write::DeflateDecoder::new(Vec::new());
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
            let mut r = read::DeflateDecoder::new(read::DeflateEncoder::new(
                &v[..],
                Compression::default(),
            ));
            let mut v2 = Vec::new();
            r.read_to_end(&mut v2)?;
            Ok(v.into_iter().eq(v2))
        }

        ::quickcheck::quickcheck(test as fn(_) -> _);
    }

    #[test]
    fn qc_writer() {
        fn test(v: Vec<u8>) -> Result<bool, Box<dyn std::error::Error>> {
            let mut w = write::DeflateEncoder::new(
                write::DeflateDecoder::new(Vec::new()),
                Compression::default(),
            );
            w.write_all(&v)?;
            Ok(v.into_iter().eq(w.finish()?.finish()?))
        }

        ::quickcheck::quickcheck(test as fn(_) -> _);
    }
}
