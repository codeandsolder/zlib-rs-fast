use core::simd::{cmp::SimdPartialEq, Simd};

const COMPARE_SIZE: usize = 256;
const LANES: usize = 32;
type Bytes = Simd<u8, LANES>;

pub fn compare256_slice(src0: &[u8], src1: &[u8]) -> usize {
    let src0 = src0
        .get(..COMPARE_SIZE)
        .expect("compare256 requires 256 bytes");
    let src1 = src1
        .get(..COMPARE_SIZE)
        .expect("compare256 requires 256 bytes");

    common_prefix_len(src0, src1)
}

/// Length of the common prefix of two equally-sized byte slices.
///
/// Full 32-byte chunks use portable SIMD; the final partial chunk is compared scalar so callers
/// can preserve their exact logical lookahead instead of padding or over-reading.
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

pub fn compare256_rle_slice(byte: u8, src: &[u8]) -> usize {
    let src = src
        .get(..COMPARE_SIZE)
        .expect("compare256_rle requires 256 bytes");
    let repeated = Bytes::splat(byte);

    for (chunk_index, chunk) in src.chunks_exact(LANES).enumerate() {
        let bytes = Bytes::from_slice(chunk);
        let equal = bytes.simd_eq(repeated).to_bitmask();
        if equal != u32::MAX as u64 {
            return chunk_index * LANES + equal.trailing_ones() as usize;
        }
    }

    COMPARE_SIZE
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
    fn compare_finds_every_possible_mismatch() {
        let left = [b'a'; COMPARE_SIZE];
        let mut right = left;

        for mismatch in 0..COMPARE_SIZE {
            right[mismatch] = b'b';
            assert_eq!(compare256_slice(&left, &right), mismatch);
            right[mismatch] = b'a';
        }
        assert_eq!(compare256_slice(&left, &right), COMPARE_SIZE);
    }

    #[test]
    fn rle_finds_every_possible_mismatch() {
        let mut input = [b'a'; COMPARE_SIZE];

        for mismatch in 0..COMPARE_SIZE {
            input[mismatch] = b'b';
            assert_eq!(compare256_rle_slice(b'a', &input), mismatch);
            input[mismatch] = b'a';
        }
        assert_eq!(compare256_rle_slice(b'a', &input), COMPARE_SIZE);
    }

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
    #[should_panic(expected = "compare256 requires 256 bytes")]
    fn compare_rejects_short_input() {
        compare256_slice(&[0; COMPARE_SIZE - 1], &[0; COMPARE_SIZE]);
    }

    #[test]
    #[should_panic(expected = "compare256_rle requires 256 bytes")]
    fn rle_rejects_short_input() {
        compare256_rle_slice(0, &[0; COMPARE_SIZE - 1]);
    }
}
