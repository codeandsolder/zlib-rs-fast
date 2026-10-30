use core::simd::{cmp::SimdPartialEq, Simd};

const LANES: usize = 32;
type Bytes = Simd<u8, LANES>;

/// Length of the common prefix of two equally-sized byte slices.
///
/// Full 32-byte chunks use portable SIMD; the final partial chunk is compared scalar so callers
/// preserve their exact logical lookahead without padding or over-reading.
#[inline]
pub(crate) fn common_prefix_len(left: &[u8], right: &[u8]) -> usize {
    assert_eq!(
        left.len(),
        right.len(),
        "common-prefix inputs must have equal lengths"
    );

    let vector_len = left.len() / LANES * LANES;
    if let Some(mismatch) = first_mismatch_chunks(&left[..vector_len], &right[..vector_len]) {
        return mismatch;
    }

    vector_len
        + left[vector_len..]
            .iter()
            .zip(&right[vector_len..])
            .position(|(left, right)| left != right)
            .unwrap_or(left.len() - vector_len)
}

/// Length of the prefix made entirely from `byte`.
#[inline]
pub(crate) fn repeated_prefix_len(byte: u8, src: &[u8]) -> usize {
    let repeated = Bytes::splat(byte);
    let vector_len = src.len() / LANES * LANES;

    for (chunk_index, chunk) in src[..vector_len].chunks_exact(LANES).enumerate() {
        let equal = Bytes::from_slice(chunk).simd_eq(repeated).to_bitmask();
        if equal != u32::MAX as u64 {
            return chunk_index * LANES + equal.trailing_ones() as usize;
        }
    }

    vector_len
        + src[vector_len..]
            .iter()
            .position(|value| *value != byte)
            .unwrap_or(src.len() - vector_len)
}

#[inline]
fn first_mismatch_chunks(left: &[u8], right: &[u8]) -> Option<usize> {
    debug_assert_eq!(left.len(), right.len());
    debug_assert_eq!(left.len() % LANES, 0);

    for (chunk_index, (left, right)) in left
        .chunks_exact(LANES)
        .zip(right.chunks_exact(LANES))
        .enumerate()
    {
        let equal = Bytes::from_slice(left)
            .simd_eq(Bytes::from_slice(right))
            .to_bitmask();
        if equal != u32::MAX as u64 {
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
