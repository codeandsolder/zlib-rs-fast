use core::simd::{cmp::SimdPartialEq, Simd};

// Keep the comparison mask derived from lane width, rather than assuming 32 lanes.
const LANES: usize = 32;
const ALL_LANES_MASK: u64 = u64::MAX >> (64 - LANES);
type Bytes = Simd<u8, LANES>;

/// Length of the common prefix of two equally-sized byte slices.
///
/// Full 32-byte chunks use portable SIMD. After equal complete chunks,
/// an end-aligned vector covers any remaining bytes without over-reading.
/// Shorter inputs use exact 16- and 8-byte comparisons and a scalar tail.
#[inline]
pub fn common_prefix_len(left: &[u8], right: &[u8]) -> usize {
    assert_eq!(
        left.len(),
        right.len(),
        "common-prefix inputs must have equal lengths"
    );

    let (left_chunks, left_tail) = left.as_chunks::<LANES>();
    let (right_chunks, right_tail) = right.as_chunks::<LANES>();
    if let Some(mismatch) = first_mismatch_chunks(left_chunks, right_chunks) {
        return mismatch;
    }

    if left_tail.is_empty() {
        return left.len();
    }
    // Any overlap with the last complete chunk is already equal.
    // Its first mismatch must therefore be inside the previously unread tail.
    if let (Some(a), Some(b)) = (left.last_chunk::<LANES>(), right.last_chunk::<LANES>()) {
        let equal = Bytes::from_array(*a)
            .simd_eq(Bytes::from_array(*b))
            .to_bitmask();
        return left.len() - LANES + equal.trailing_ones() as usize;
    }
    short_prefix(left_tail, right_tail)
}

/// Length of the prefix made entirely from `byte`.
#[inline]
pub fn repeated_prefix_len(byte: u8, src: &[u8]) -> usize {
    let repeated = Bytes::splat(byte);
    let (chunks, tail) = src.as_chunks::<LANES>();

    for (chunk_index, chunk) in chunks.iter().enumerate() {
        let equal = Bytes::from_array(*chunk).simd_eq(repeated).to_bitmask();
        if equal != ALL_LANES_MASK {
            return chunk_index * LANES + equal.trailing_ones() as usize;
        }
    }

    chunks.len() * LANES
        + tail
            .iter()
            .position(|value| *value != byte)
            .unwrap_or(tail.len())
}

#[inline]
fn first_mismatch_chunks(left: &[[u8; LANES]], right: &[[u8; LANES]]) -> Option<usize> {
    debug_assert_eq!(left.len(), right.len());

    for (chunk_index, (left, right)) in left.iter().zip(right).enumerate() {
        let equal = Bytes::from_array(*left)
            .simd_eq(Bytes::from_array(*right))
            .to_bitmask();
        if equal != ALL_LANES_MASK {
            return Some(chunk_index * LANES + equal.trailing_ones() as usize);
        }
    }

    None
}

/// Resolve short prefixes in descending fixed widths. Every load stays
/// within the exact logical slices, regardless of their alignment.
#[inline]
fn short_prefix(mut left: &[u8], mut right: &[u8]) -> usize {
    let original = left.len();
    if let (Some(a), Some(b)) = (left.first_chunk::<16>(), right.first_chunk::<16>()) {
        let mask = Simd::from_array(*a)
            .simd_eq(Simd::from_array(*b))
            .to_bitmask();
        if mask != 0xffff {
            return mask.trailing_ones() as usize;
        }
        left = &left[16..];
        right = &right[16..];
    }
    if let (Some(a), Some(b)) = (left.first_chunk::<8>(), right.first_chunk::<8>()) {
        let diff = u64::from_le_bytes(*a) ^ u64::from_le_bytes(*b);
        if diff != 0 {
            return original - left.len() + (diff.trailing_zeros() as usize >> 3);
        }
        left = &left[8..];
        right = &right[8..];
    }
    original - left.len()
        + left
            .iter()
            .zip(right)
            .position(|(a, b)| a != b)
            .unwrap_or(left.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_prefix_handles_every_length_and_mismatch_through_max_match() {
        const MAX: usize = crate::deflate::STD_MAX_MATCH;
        let left = [b'a'; MAX];
        let mut right = left;

        for len in 0..=MAX {
            assert_eq!(common_prefix_len(&left[..len], &right[..len]), len);
            for mismatch in 0..len {
                right[mismatch] = b'b';
                assert_eq!(common_prefix_len(&left[..len], &right[..len]), mismatch);
                right[mismatch] = b'a';
            }
        }
    }

    #[test]
    fn vector_endpoints_handle_independently_aligned_and_long_slices() {
        let left = [0x43u8; 1056];
        let mut right = left;
        for len in [
            0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 257, 258, 1024,
        ] {
            for offset in 0..32 {
                let a = &left[offset..offset + len];
                let other = 31 - offset;
                assert_eq!(common_prefix_len(a, &right[other..other + len]), len);
                for mismatch in [0, len / 2, len.saturating_sub(1)] {
                    if mismatch < len {
                        right[other + mismatch] ^= 0xff;
                        assert_eq!(common_prefix_len(a, &right[other..other + len]), mismatch);
                        right[other + mismatch] ^= 0xff;
                    }
                }
            }
        }
    }

    #[test]
    fn repeated_prefix_handles_every_length_and_mismatch_through_max_match() {
        const MAX: usize = crate::deflate::STD_MAX_MATCH;
        let mut input = [b'a'; MAX];

        for len in 0..=MAX {
            assert_eq!(repeated_prefix_len(b'a', &input[..len]), len);
            for mismatch in 0..len {
                input[mismatch] = b'b';
                assert_eq!(repeated_prefix_len(b'a', &input[..len]), mismatch);
                input[mismatch] = b'a';
            }
        }
    }
}
