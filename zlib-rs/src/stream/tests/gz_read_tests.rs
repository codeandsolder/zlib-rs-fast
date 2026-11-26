use crate::stream::io::{Cursor, ErrorKind, Read, Result, Write};
use alloc::vec::Vec;

use super::GzDecoder;

//a cursor turning EOF into blocking errors
#[derive(Debug)]
pub struct BlockingCursor {
    pub cursor: Cursor<Vec<u8>>,
}

impl BlockingCursor {
    pub fn new() -> Self {
        Self {
            cursor: Cursor::new(Vec::new()),
        }
    }

    pub fn set_position(&mut self, pos: u64) {
        self.cursor.set_position(pos);
    }
}

impl Write for BlockingCursor {
    fn write(&mut self, buf: &[u8]) -> Result<usize> {
        self.cursor.write(buf)
    }
    fn flush(&mut self) -> Result<()> {
        self.cursor.flush()
    }
}

impl Read for BlockingCursor {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        //use the cursor, except it turns eof into blocking error
        let r = self.cursor.read(buf);
        match r {
            Err(ref err) => {
                if err.kind() == ErrorKind::UnexpectedEof {
                    return Err(ErrorKind::WouldBlock.into());
                }
            }
            Ok(0) => {
                //regular EOF turned into blocking error
                return Err(ErrorKind::WouldBlock.into());
            }
            Ok(_n) => {}
        }
        r
    }
}

#[test]
fn blocked_partial_header_read() -> core::result::Result<(), Box<dyn std::error::Error>> {
    // this is a reader which receives data afterwards
    let mut r = BlockingCursor::new();
    let data = vec![1, 2, 3];

    r.write_all(&data)?;
    r.set_position(0);

    // this is unused except for the buffering
    let mut decoder = GzDecoder::new(r);
    let mut out = [0; 7];
    let error = decoder.read(&mut out).err().ok_or("expected WouldBlock")?;
    assert_eq!(error.kind(), ErrorKind::WouldBlock);

    Ok(())
}

fn compress_data(data: &[u8]) -> core::result::Result<Vec<u8>, Box<dyn std::error::Error>> {
    use crate::stream::write::GzEncoder;
    use crate::stream::Compression;

    let mut e = GzEncoder::new(Vec::new(), Compression::default());
    e.write_all(data)?;
    Ok(e.finish()?)
}

#[test]
fn decode_with_reset() -> core::result::Result<(), Box<dyn std::error::Error>> {
    let data1 = b"Hello World";
    let data2 = b"Goodbye World";

    let compressed1 = compress_data(data1)?;
    let compressed2 = compress_data(data2)?;

    let mut output = Vec::new();
    let mut decoder = GzDecoder::new(compressed1.as_slice());
    decoder.read_to_end(&mut output)?;
    assert_eq!(output, data1);

    output.clear();
    decoder.reset(compressed2.as_slice());
    decoder.read_to_end(&mut output)?;
    assert_eq!(output, data2);

    Ok(())
}

#[test]
fn decode_with_reset_after_corruption() -> core::result::Result<(), Box<dyn std::error::Error>> {
    let valid_data = b"Hello World";
    let valid_compressed = compress_data(valid_data)?;

    // Create a corrupted payload (valid gzip header but corrupted body)
    let mut corrupted = valid_compressed.clone();
    assert!(corrupted.len() > 13);
    corrupted[12] ^= 0xFF;
    corrupted[13] ^= 0xFF;

    // Try to decode corrupted data
    let mut decoder = GzDecoder::new(Cursor::new(corrupted));
    let mut output = Vec::new();
    let _ = decoder
        .read_to_end(&mut output)
        .err()
        .ok_or("expected an error")?;

    // Reset with valid payload and decode
    decoder.reset(Cursor::new(valid_compressed));
    output.clear();
    decoder.read_to_end(&mut output)?;
    assert_eq!(output, valid_data);

    Ok(())
}
