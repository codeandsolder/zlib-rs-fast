use core::simd::{num::SimdUint, Simd};

use super::{BASE, NMAX};

const LANES: usize = 16;
type Bytes = Simd<u8, LANES>;
type Sums = Simd<u16, LANES>;

const WEIGHTS: Sums = Sums::from_array([16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]);

/// Updates an Adler-32 checksum using the algebraic form of the recurrence.
///
/// For a block `x[0..n]`, Adler's two running sums can be updated as:
///
/// `a' = a + sum(x)`
///
/// `b' = b + n * a + sum((n - i) * x[i])`
///
/// Keeping this as a pure portable-SIMD transform avoids architecture dispatch and leaves a clean
/// boundary for future batched execution.
pub(super) fn adler32(start_checksum: u32, data: &[u8]) -> u32 {
    let mut a = start_checksum & 0xffff;
    let mut b = start_checksum >> 16;

    // NMAX is zlib's maximum number of bytes that can be accumulated in u32 before reducing both
    // sums modulo BASE. It is exactly divisible by LANES, so every full block is entirely SIMD.
    debug_assert_eq!(NMAX as usize % LANES, 0);

    for block in data.chunks(NMAX as usize) {
        let mut vectors = block.chunks_exact(LANES);

        for chunk in &mut vectors {
            let values: Sums = Bytes::from_slice(chunk).cast();
            let byte_sum = values.reduce_sum() as u32;
            let weighted_sum = (values * WEIGHTS).reduce_sum() as u32;

            b += LANES as u32 * a + weighted_sum;
            a += byte_sum;
        }

        for &byte in vectors.remainder() {
            a += byte as u32;
            b += a;
        }

        a %= BASE;
        b %= BASE;
    }

    a | (b << 16)
}
