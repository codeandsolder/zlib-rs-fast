use core::simd::{num::SimdUint, Simd};

use super::{BASE, NMAX};

const LANES: usize = 16;
const GROUP_VECTORS: usize = 22;
const GROUP_BYTES: usize = LANES * GROUP_VECTORS;

type Bytes = Simd<u8, LANES>;
type Sums = Simd<u16, LANES>;

const LANE_INDEX: Simd<u32, LANES> =
    Simd::from_array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
const ZERO: Sums = Sums::splat(0);

/// Updates an Adler-32 checksum with a reduction-friendly portable-SIMD kernel.
///
/// The hot loop accumulates up to twenty-two 16-byte vectors before performing any horizontal reduction.
/// `lane_sums` holds the byte sum for each lane and `prefix_sums` accumulates those lane sums after
/// each vector. Their horizontal sums recover the Adler recurrence; one lane-index correction
/// supplies each byte's position within its vector.
///
/// All hot accumulators are `u16x16`, which maps naturally to 256-bit SIMD on AVX2-class machines.
/// A 352-byte group is the largest whole-vector group whose worst-case prefix accumulator still
/// fits `u16`; widening happens only for the three reductions at the group boundary.
pub(super) fn adler32(start_checksum: u32, data: &[u8]) -> u32 {
    let mut a = start_checksum & 0xffff;
    let mut b = start_checksum >> 16;

    for block in data.chunks(NMAX as usize) {
        let (groups, remainder) = block.as_chunks::<GROUP_BYTES>();
        for group in groups {
            (a, b) = adler_group(a, b, group);
        }

        let (complete_vectors, tail) = remainder.as_chunks::<LANES>();
        if !complete_vectors.is_empty() {
            (a, b) = adler_group(a, b, &remainder[..complete_vectors.len() * LANES]);
        }
        for &byte in tail {
            a += u32::from(byte);
            b += a;
        }

        a %= BASE;
        b %= BASE;
    }

    a | (b << 16)
}

#[inline]
fn adler_group(mut a: u32, b: u32, bytes: &[u8]) -> (u32, u32) {
    debug_assert!(
        !bytes.is_empty(),
        "a group must contain at least one SIMD vector"
    );
    debug_assert_eq!(bytes.len() % LANES, 0);
    debug_assert!(bytes.len() <= GROUP_BYTES);

    let mut lane_sums = ZERO;
    let mut prefix_sums = ZERO;
    for chunk in bytes.as_chunks::<LANES>().0 {
        let values: Sums = Bytes::from_array(*chunk).cast();
        lane_sums += values;
        prefix_sums += lane_sums;
    }

    let byte_sum = lane_sums.cast::<u32>().reduce_sum();
    let prefix_sum = prefix_sums.cast::<u32>().reduce_sum();
    let lane_correction = (lane_sums.cast::<u32>() * LANE_INDEX).reduce_sum();
    // At most 352 bytes per group: all lane sums and this weighted sum fit u32.
    let weighted_sum = 16 * prefix_sum - lane_correction;

    let next_b = u64::from(b) + bytes.len() as u64 * u64::from(a) + u64::from(weighted_sum);
    // NMAX=5552 bounds the worst-case initial (a,b)=(65535,65535),
    // all input bytes=255, to 4_294_773_495 (< u32::MAX). The groups
    // partition one NMAX block, so every intermediate sum is smaller.
    let next_b = u32::try_from(next_b)
        .unwrap_or_else(|_| unreachable!("Adler-32 block sum exceeded its proven NMAX bound"));
    a += byte_sum;
    (a, next_b)
}
