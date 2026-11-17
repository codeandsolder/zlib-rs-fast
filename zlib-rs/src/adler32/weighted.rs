//! Architecture-width portable SIMD Adler-32 without x86 vector intrinsics.
//!
//! Defer positional weighting until the block reduction: accumulating each
//! input byte's even/odd pair sums and even-byte sums requires only narrow
//! lane operations in the hot loop. The original weighted byte-dot product
//! is recovered exactly from those totals and the vector-prefix terms.

use core::simd::{num::SimdUint, Simd};

use super::{BASE, NMAX};

// For a byte pair (lo, hi), let p = lo + hi. Within a BYTES-wide
// input vector, the pair's weighted Adler contribution is
//     p * (BYTES - 1 - 2*i) + lo.
// Thus accumulate p and lo in u16 SIMD lanes for the entire block and
// apply the positional multiplication *once*, after the input loop.
//
// Every group processes 16 vectors. The largest per-lane prefix sum is
// 510 * (16*15/2) = 61_200, within u16.
// The cumulative per-lane pair sum must also fit u16, which permits at
// most floor(65_535 / 510) = 128 input vectors per reduction block.

macro_rules! checksum_body {
    ($pairs:literal, $bytes:literal, $initial:expr, $data:expr) => {{
        const GROUP: usize = 16;
        const GROUP_BYTES: usize = GROUP * $bytes;
        const MAX_U16_VECTORS: usize = u16::MAX as usize / 510;
        const MAX_NMAX_VECTORS: usize = NMAX as usize / $bytes;
        const MAX_VECTORS: usize = if MAX_NMAX_VECTORS < MAX_U16_VECTORS {
            MAX_NMAX_VECTORS
        } else {
            MAX_U16_VECTORS
        };
        const BLOCK_BYTES: usize = (MAX_VECTORS / GROUP) * GROUP_BYTES;

        let mut a = u64::from($initial & 0xffff);
        let mut b = u64::from($initial >> 16);
        let byte_mask = Simd::<u16, $pairs>::splat(0xff);
        let within_vector_weights = Simd::<u32, $pairs>::from_array(core::array::from_fn(|i| {
            u32::try_from($bytes - 1 - 2 * i)
                .unwrap_or_else(|_| unreachable!("Adler lane index exceeds u32"))
        }));
        let group_bytes_u32 =
            u32::try_from(GROUP_BYTES).unwrap_or_else(|_| unreachable!("Adler group exceeds u32"));
        let vector_bytes_u32 =
            u32::try_from($bytes).unwrap_or_else(|_| unreachable!("Adler vector exceeds u32"));

        for block in $data.chunks(BLOCK_BYTES) {
            let (groups, tail) = block.as_chunks::<GROUP_BYTES>();
            let mut pair_sums = Simd::<u16, $pairs>::splat(0);
            let mut low_sums = Simd::<u16, $pairs>::splat(0);
            let mut cross_vector_weighted = Simd::<u32, $pairs>::splat(0);

            for group in groups {
                let (vectors, remainder) = group.as_chunks::<$bytes>();
                debug_assert!(remainder.is_empty());
                let mut group_sum = Simd::<u16, $pairs>::splat(0);
                let mut group_prefix = Simd::<u16, $pairs>::splat(0);

                for source in vectors {
                    // This safe, endian-independent construction compiles to
                    // native unaligned SIMD loads on current LLVM.
                    let words = Simd::<u16, $pairs>::from_array(core::array::from_fn(|i| {
                        u16::from_le_bytes([source[2 * i], source[2 * i + 1]])
                    }));
                    let low = words & byte_mask;
                    let pair_sum = low + (words >> 8);
                    group_prefix += group_sum;
                    group_sum += pair_sum;
                    low_sums += low;
                }

                cross_vector_weighted += pair_sums.cast::<u32>() * Simd::splat(group_bytes_u32)
                    + group_prefix.cast::<u32>() * Simd::splat(vector_bytes_u32);
                pair_sums += group_sum;
            }

            let vector_count = groups.len() * GROUP_BYTES;
            let pairs = pair_sums.cast::<u32>();
            let intra_vector_weighted = pairs * within_vector_weights + low_sums.cast::<u32>();
            // NMAX plus the smaller u16-lane bound keep Adler's unreduced
            // checksum limbs within u32 for the complete reduction block.
            b += a * u64::try_from(vector_count)
                .unwrap_or_else(|_| unreachable!("Adler block exceeds u64"))
                + u64::from(cross_vector_weighted.reduce_sum())
                + u64::from(intra_vector_weighted.reduce_sum());
            a += u64::from(pairs.reduce_sum());
            for &byte in tail {
                a += u64::from(byte);
                b += a;
            }
            a %= u64::from(BASE);
            b %= u64::from(BASE);
        }

        u32::try_from(a | (b << 16))
            .unwrap_or_else(|_| unreachable!("reduced Adler checksum exceeds u32"))
    }};
}

/// # Safety
/// Requires AVX2 support from both the CPU and OS.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn checksum_avx2(start: u32, data: &[u8]) -> u32 {
    checksum_body!(16, 32, start, data)
}

/// # Safety
/// Requires AVX-512F and AVX-512BW support from both the CPU and OS.
#[target_feature(enable = "avx512f,avx512bw")]
pub(super) unsafe fn checksum_avx512(start: u32, data: &[u8]) -> u32 {
    checksum_body!(32, 64, start, data)
}
