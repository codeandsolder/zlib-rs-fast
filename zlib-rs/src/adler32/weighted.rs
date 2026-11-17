//! Adler-32 with architecture-selected portable SIMD widths.
//!
//! The arithmetic is architecture-neutral. Expanding its one macro body
//! inside target-feature thunks lets LLVM generate native AVX2/AVX-512
//! instructions without requiring inline(always) or CPU intrinsics.
use core::simd::{num::SimdUint, Simd};

use super::{BASE, NMAX};

// The u16 prefix accumulator fits 22 full vectors in the worst case:
// 255 * (22 * 23 / 2) = 64_515; 23 vectors would overflow.
const GROUP_VECTORS: usize = 22;

macro_rules! accumulate_group {
    ($group:expr, $lanes:expr, $remaining:expr, $weights:expr, $byte_lanes:ident, $weighted_lanes:ident) => {{
        let mut lane_sum = Simd::<u16, $lanes>::splat(0);
        let mut prefix_sum = Simd::<u16, $lanes>::splat(0);
        for input in $group {
            let bytes = Simd::<u8, $lanes>::from_array(*input).cast::<u16>();
            lane_sum += bytes;
            prefix_sum += lane_sum;
        }
        let lane_sum = lane_sum.cast::<u32>();
        let prefix_sum = prefix_sum.cast::<u32>();
        $byte_lanes += lane_sum;
        // Each lane contributes its intra-group positional weight, plus
        // one more weight for every byte following this group in the block.
        $weighted_lanes += (prefix_sum - lane_sum)
            * Simd::splat(
                u32::try_from($lanes)
                    .unwrap_or_else(|_| unreachable!("SIMD lane count exceeds u32")),
            )
            + lane_sum * ($weights + Simd::splat($remaining));
    }};
}

macro_rules! checksum_body {
    ($lanes:expr, $start:expr, $data:expr) => {{
        let mut a = $start & 0xffff;
        let mut b = $start >> 16;
        let reverse_weights = Simd::<u32, $lanes>::from_array(core::array::from_fn(|lane| {
            u32::try_from($lanes - lane)
                .unwrap_or_else(|_| unreachable!("SIMD lane index exceeds u32"))
        }));
        // NMAX (5552) is not divisible by all SIMD widths. Use the
        // largest whole-vector block within NMAX, avoiding per-block
        // scalar tails and preserving aligned vector boundaries.
        let block_bytes = (NMAX as usize / $lanes) * $lanes;
        for block in $data.chunks(block_bytes) {
            let (vectors, tail) = block.as_chunks::<$lanes>();
            let vector_bytes = vectors.len() * $lanes;
            let mut byte_lanes = Simd::<u32, $lanes>::splat(0);
            let mut weighted_lanes = Simd::<u32, $lanes>::splat(0);
            let (full_groups, remainder) = vectors.as_chunks::<GROUP_VECTORS>();
            let mut processed = 0usize;

            for group in full_groups {
                processed += GROUP_VECTORS;
                let remaining = u32::try_from(vector_bytes - processed * $lanes)
                    .unwrap_or_else(|_| unreachable!("Adler block exceeds u32"));
                accumulate_group!(
                    group,
                    $lanes,
                    remaining,
                    reverse_weights,
                    byte_lanes,
                    weighted_lanes
                );
            }
            if !remainder.is_empty() {
                processed += remainder.len();
                let remaining = u32::try_from(vector_bytes - processed * $lanes)
                    .unwrap_or_else(|_| unreachable!("Adler block exceeds u32"));
                accumulate_group!(
                    remainder,
                    $lanes,
                    remaining,
                    reverse_weights,
                    byte_lanes,
                    weighted_lanes
                );
            }

            let vector_sum = byte_lanes.reduce_sum();
            let weighted = weighted_lanes.reduce_sum();
            let vector_bytes = u64::try_from(vector_bytes)
                .unwrap_or_else(|_| unreachable!("Adler block exceeds u64"));
            let next_b = u64::from(b) + vector_bytes * u64::from(a) + u64::from(weighted);
            // NMAX=5552 bounds the whole block to u32 even when every
            // input byte and both incoming Adler limbs are maximal.
            b = u32::try_from(next_b).unwrap_or_else(|_| unreachable!("Adler NMAX bound exceeded"));
            a += vector_sum;
            for &byte in tail {
                a += u32::from(byte);
                b += a;
            }
            a %= BASE;
            b %= BASE;
        }
        a | (b << 16)
    }};
}

// Provide direct width-specific entrypoints for portability tests and for
// platforms whose native SIMD width is known at compile time.
#[cfg(any(
    test,
    all(target_arch = "x86_64", not(feature = "std")),
    target_arch = "aarch64",
    all(target_arch = "arm", target_feature = "neon"),
    all(target_arch = "wasm32", target_feature = "simd128")
))]
pub(super) fn checksum<const LANES: usize>(start: u32, data: &[u8]) -> u32 {
    checksum_body!(LANES, start, data)
}

/// # Safety
/// Caller must establish CPU and OS support for AVX2.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn checksum_avx2(start: u32, data: &[u8]) -> u32 {
    checksum_body!(32, start, data)
}

/// # Safety
/// Caller must establish CPU and OS support for AVX-512F and AVX-512BW.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[target_feature(enable = "avx512f,avx512bw")]
pub(super) unsafe fn checksum_avx512(start: u32, data: &[u8]) -> u32 {
    checksum_body!(64, start, data)
}
