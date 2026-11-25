//! Discover independent entropy decode segments without materializing output.
//! Reuse the ordinary inflater's validated tables and fast symbol decoder.
//! An index is not proof of a correct checksum: the device must check Adler
//! against the decoded bytes before publishing any recompressed result.
use super::{
    bitreader::{BitBuffer, BitReader},
    decode_distance_code, decode_huffman_code, inflate_fast_help_impl, FastDecodeOutput, Flags,
    InflateStateKind, Mode, State, INFLATE_FAST_MIN_HAVE, INFLATE_FAST_MIN_LEFT, MAX_BITS,
};
use crate::ReturnCode;

const MAX_SEGMENT_BYTES: usize = 128 * 1024;

/// A token-aligned entropy segment. Header and payload positions are bit
/// offsets in the entire zlib member, including the two-byte wrapper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InflateSegment {
    pub header_bit: u64,
    pub start_bit: u64,
    pub end_bit: u64,
    pub start: u32,
    pub end: u32,
    /// 0 for stored, 1/2 for fixed/dynamic Huffman.
    pub kind: u32,
    /// Huffman segment terminates with the actual end-of-block symbol.
    pub ends_block: bool,
}

/// A validated structural index of a single RFC 1950 member.
#[derive(Debug, PartialEq, Eq)]
pub struct InflateIndex {
    pub segments: Vec<InflateSegment>,
    pub decoded_len: u32,
    pub window: u32,
    pub adler: u32,
}

struct Counter {
    position: usize,
    limit: usize,
    window: usize,
}

impl FastDecodeOutput for Counter {
    #[inline]
    fn push(&mut self, _: u8) {
        self.position += 1;
    }

    #[inline]
    fn remaining(&self) -> usize {
        self.limit.saturating_sub(self.position)
    }

    #[inline]
    fn copy_back_reference(
        &mut self,
        _: &State<'_>,
        distance: usize,
        length: usize,
    ) -> Result<(), &'static str> {
        if distance > self.position || distance > self.window || distance == 0 {
            return Err("invalid distance too far back\0");
        }
        self.position += length;
        Ok(())
    }
}

fn offset(reader: &BitReader<'_>, total: usize) -> u64 {
    (total - reader.bytes_remaining()) as u64 * 8 - u64::from(reader.bits_in_buffer())
}

fn read_extra(reader: &mut BitReader<'_>, bits: u8) -> Result<usize, ReturnCode> {
    reader
        .need_bits(usize::from(bits))
        .map_err(|_| ReturnCode::DataError)?;
    let value = reader.bits_usize(usize::from(bits));
    reader.drop_bits(bits);
    Ok(value)
}

/// Read one complete token at the end of the fast path. Return true for EOB.
fn slow_token(
    state: &State<'_>,
    reader: &mut BitReader<'_>,
    counter: &mut Counter,
) -> Result<bool, ReturnCode> {
    let entry = decode_huffman_code::<true>(reader, state.len_table_ref(), state.len_table.bits)
        .map_err(|_| ReturnCode::DataError)?
        .entry;
    if entry.op == 0 {
        counter.position += 1;
    } else if entry.op & 32 != 0 {
        return Ok(true);
    } else if entry.op & 64 != 0 {
        return Err(ReturnCode::DataError);
    } else {
        let length = usize::from(entry.val) + read_extra(reader, entry.op & MAX_BITS)?;
        let d = decode_distance_code(reader, state.dist_table_ref(), state.dist_table.bits)
            .map_err(|_| ReturnCode::DataError)?;
        let distance = d.offset + read_extra(reader, d.extra)?;
        counter
            .copy_back_reference(state, distance, length)
            .map_err(|_| ReturnCode::DataError)?;
    }
    Ok(false)
}

fn read_huffman_segment(
    state: &mut State<'_>,
    reader: &mut BitReader<'_>,
    counter: &mut Counter,
    output_limit: usize,
) -> Result<bool, ReturnCode> {
    counter.limit = counter
        .position
        .saturating_add(MAX_SEGMENT_BYTES)
        .min(output_limit);
    state.mode = Mode::Len;
    if reader.bytes_remaining() >= INFLATE_FAST_MIN_HAVE
        && counter.remaining() >= INFLATE_FAST_MIN_LEFT
    {
        inflate_fast_help_impl::<true, _>(state, reader, counter);
        match state.mode {
            Mode::Type => return Ok(true),
            Mode::Len => {}
            _ => return Err(ReturnCode::DataError),
        }
    }
    // The fast decoder stops before consuming a token that could exceed its
    // reserved output room. Complete at most a token beyond the segment
    // target (up to 257 bytes), but never beyond the actual output limit.
    loop {
        let end = slow_token(state, reader, counter)?;
        if counter.position > output_limit {
            return Err(ReturnCode::BufError);
        }
        if end || counter.position >= counter.limit {
            return Ok(end);
        }
    }
}

/// Structurally index an RFC 1950 member, without output bytes or checksum work.
/// The caller supplies a decoded-byte bound and a descriptor bound to prevent
/// untrusted input from causing unbounded allocation. A dictionary member or
/// trailing bytes is rejected; the regular inflater remains available.
/// # Errors
/// Returns DataError for invalid/truncated members or distances, BufError when
/// the decoded size or descriptor limit is exceeded.
pub fn index_zlib(
    input: &[u8],
    output_limit: u32,
    segment_limit: usize,
) -> Result<InflateIndex, ReturnCode> {
    if input.len() < 6 || input.len() > u32::MAX as usize {
        return Err(ReturnCode::DataError);
    }
    let mut state = State::new(InflateStateKind::Regular);
    state.wrap = 1;
    state.wbits = 15;
    let mut reader = BitReader::from_buffer(BitBuffer::default(), input);
    if !matches!(state.decode_wrapper_head(&mut reader), Ok(Mode::Type)) {
        return Err(ReturnCode::DataError);
    }
    let mut counter = Counter {
        position: 0,
        limit: output_limit as usize,
        window: state.dmax,
    };
    let mut segments = Vec::new();
    loop {
        let header_bit = offset(&reader, input.len());
        let mode = state
            .decode_block_type(&mut reader)
            .map_err(|_| ReturnCode::DataError)?;
        let kind = match mode {
            Mode::Stored => {
                reader.return_unused_bytes();
                state
                    .decode_stored_header(&mut reader)
                    .map_err(|_| ReturnCode::DataError)?;
                let start_bit = offset(&reader, input.len());
                let start = counter.position;
                if state.length > reader.bytes_remaining() {
                    return Err(ReturnCode::DataError);
                }
                counter.position = start
                    .checked_add(state.length)
                    .filter(|&n| n <= output_limit as usize)
                    .ok_or(ReturnCode::BufError)?;
                reader.advance(state.length);
                if segments.len() == segment_limit {
                    return Err(ReturnCode::BufError);
                }
                segments.push(InflateSegment {
                    header_bit,
                    start_bit,
                    end_bit: offset(&reader, input.len()),
                    start: u32::try_from(start).map_err(|_| ReturnCode::BufError)?,
                    end: u32::try_from(counter.position).map_err(|_| ReturnCode::BufError)?,
                    kind: 0,
                    ends_block: true,
                });
                0
            }
            Mode::Len_ => 1,
            Mode::Table => {
                state
                    .decode_dynamic_table_header(&mut reader)
                    .map_err(|_| ReturnCode::DataError)?;
                state
                    .decode_code_length_alphabet(&mut reader)
                    .map_err(|_| ReturnCode::DataError)?;
                state
                    .decode_code_lengths(&mut reader)
                    .map_err(|_| ReturnCode::DataError)?;
                2
            }
            _ => return Err(ReturnCode::DataError),
        };
        if kind != 0 {
            loop {
                if segments.len() == segment_limit {
                    return Err(ReturnCode::BufError);
                }
                let start_bit = offset(&reader, input.len());
                let start = counter.position;
                let ends_block = read_huffman_segment(
                    &mut state,
                    &mut reader,
                    &mut counter,
                    output_limit as usize,
                )?;
                segments.push(InflateSegment {
                    header_bit,
                    start_bit,
                    end_bit: offset(&reader, input.len()),
                    start: u32::try_from(start).map_err(|_| ReturnCode::BufError)?,
                    end: u32::try_from(counter.position).map_err(|_| ReturnCode::BufError)?,
                    kind,
                    ends_block,
                });
                if ends_block {
                    break;
                }
            }
        }
        if state.flags.contains(Flags::IS_LAST_BLOCK) {
            break;
        }
    }
    let trailer = usize::try_from(offset(&reader, input.len()).div_ceil(8))
        .map_err(|_| ReturnCode::DataError)?;
    let bytes = input.get(trailer..).ok_or(ReturnCode::DataError)?;
    let adler = u32::from_be_bytes(bytes.try_into().map_err(|_| ReturnCode::DataError)?);
    Ok(InflateIndex {
        segments,
        decoded_len: u32::try_from(counter.position).map_err(|_| ReturnCode::BufError)?,
        window: u32::try_from(counter.window).map_err(|_| ReturnCode::DataError)?,
        adler,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{compress_bound, compress_slice, DeflateConfig};

    #[test]
    fn indexes_stored_fixed_dynamic_and_empty() -> Result<(), ReturnCode> {
        let data = b"abbbbbbbbbbb01234567890123456789".repeat(20_000);
        for input in [b"".as_slice(), b"a", &data] {
            for (level, strategy) in [
                (0, crate::Strategy::Default),
                (6, crate::Strategy::Fixed),
                (9, crate::Strategy::Default),
            ] {
                let mut buf = vec![0; compress_bound(input.len())];
                let (compressed, code) = compress_slice(
                    &mut buf,
                    input,
                    DeflateConfig {
                        level,
                        strategy,
                        ..DeflateConfig::default()
                    },
                );
                assert_eq!(code, ReturnCode::Ok);
                let index = index_zlib(
                    compressed,
                    u32::try_from(input.len()).map_err(|_| ReturnCode::BufError)?,
                    4096,
                )?;
                assert_eq!(index.decoded_len as usize, input.len());
                assert_eq!(index.adler, crate::adler32::adler32(1, input));
                assert_eq!(index.segments.first().map(|s| s.start), Some(0));
                assert_eq!(
                    index.segments.last().map(|s| s.end),
                    Some(index.decoded_len)
                );
                for pair in index.segments.windows(2) {
                    assert_eq!(pair[0].end, pair[1].start);
                    assert!(pair[0].end_bit <= pair[1].start_bit);
                }
                assert!(index_zlib(&compressed[..compressed.len() - 1], u32::MAX, 4096).is_err());
                assert!(index_zlib(compressed, u32::MAX, 0).is_err());
                if !input.is_empty() {
                    assert!(index_zlib(compressed, index.decoded_len - 1, 4096).is_err());
                }
            }
        }
        Ok(())
    }
}
