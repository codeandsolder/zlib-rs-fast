use core::simd::{cmp::SimdPartialEq, Simd};

// Keep the comparison mask derived from lane width, rather than assuming 32 lanes.
const LANES: usize = 32;
const ALL_LANES_MASK: u64 = u64::MAX >> (64 - LANES);
type Bytes = Simd<u8, LANES>;

/// Length of the common prefix of two equally-sized byte slices.
///
/// Full 32-byte chunks use portable SIMD; the final partial chunk is compared scalar so callers
/// preserve their exact logical lookahead without padding or over-reading.
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

    left_chunks.len() * LANES
        + left_tail
            .iter()
            .zip(right_tail)
            .position(|(left, right)| left != right)
            .unwrap_or(left_tail.len())
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
