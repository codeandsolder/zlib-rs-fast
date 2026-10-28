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

    first_mismatch(src0, src1).unwrap_or(COMPARE_SIZE)
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
fn first_mismatch(left: &[u8], right: &[u8]) -> Option<usize> {
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
