//! AVX-512BW Adler-32 using native byte SAD and packed weighted dot products.
//!
//! Extends the production AVX2 algorithm to 64-byte input vectors, retaining
//! exact Adler-32 output with a short 5504-byte modular-reduction boundary.
//! Unaligned vector loads read only fully initialized input bytes.
use core::arch::x86_64::{
    __m512i, _mm512_add_epi32, _mm512_add_epi64, _mm512_loadu_epi8, _mm512_madd_epi16,
    _mm512_maddubs_epi16, _mm512_reduce_add_epi32, _mm512_reduce_add_epi64, _mm512_sad_epu8,
    _mm512_setzero_si512,
};

use super::{BASE, NMAX};
const VECTOR_BYTES: usize = 64;
const BLOCK: usize = (NMAX as usize / VECTOR_BYTES) * VECTOR_BYTES;

const fn weights() -> [u8; 64] {
    let mut weights = [0u8; 64];
    let mut i = 0;
    let mut weight = 64_u8;
    while i < VECTOR_BYTES {
        weights[i] = weight;
        weight -= 1;
        i += 1;
    }
    weights
}

const fn ones() -> [u8; 64] {
    let mut a = [0u8; 64];
    let mut i = 0;
    while i < 64 {
        a[i] = 1;
        i += 2;
    }
    a
}

// SAFETY: all bit patterns are valid for the opaque __m512i register type.
const W: __m512i = unsafe { core::mem::transmute(weights()) };
// SAFETY: all bit patterns are valid for the opaque __m512i register type.
const ONES: __m512i = unsafe { core::mem::transmute(ones()) };

/// # Safety
/// Caller must establish CPU/OS support for AVX-512F and AVX-512BW.
#[target_feature(enable = "avx512bw,avx512f")]
pub(super) unsafe fn checksum(start: u32, data: &[u8]) -> u32 {
    let mut a = start & 0xffff;
    let mut b = start >> 16;
    let zero = _mm512_setzero_si512();

    for block in data.chunks(BLOCK) {
        let old_a = a;
        let (vectors, tail) = block.as_chunks::<VECTOR_BYTES>();
        let mut sums = _mm512_setzero_si512();
        let mut prefixes = _mm512_setzero_si512();
        let mut dots = _mm512_setzero_si512();
        for bytes in vectors {
            // SAFETY: [u8;64] is initialized; the unaligned byte load
            // requires only 1-byte alignment and reads exactly 64 bytes.
            let value = unsafe { _mm512_loadu_epi8(bytes.as_ptr().cast::<i8>()) };
            prefixes = _mm512_add_epi64(prefixes, sums);
            sums = _mm512_add_epi64(sums, _mm512_sad_epu8(value, zero));
            // Largest signed-i16 pair: 255*(64+63)=32385 < 32767.
            let products = _mm512_maddubs_epi16(value, W);
            dots = _mm512_add_epi32(dots, _mm512_madd_epi16(products, ONES));
        }
        let sum = u32::try_from(_mm512_reduce_add_epi64(sums))
            .unwrap_or_else(|_| unreachable!("NMAX bounds Adler sum"));
        let prior_prefix = u64::try_from(_mm512_reduce_add_epi64(prefixes))
            .unwrap_or_else(|_| unreachable!("NMAX bounds Adler prefix"));
        let dot = u32::try_from(_mm512_reduce_add_epi32(dots))
            .unwrap_or_else(|_| unreachable!("NMAX bounds Adler weighted dot"));
        a += sum;
        b = u32::try_from(
            u64::from(b)
                + u64::from(old_a)
                    * u64::try_from(vectors.len() * VECTOR_BYTES)
                        .unwrap_or_else(|_| unreachable!("Adler vector count fits in u64"))
                + prior_prefix * 64
                + u64::from(dot),
        )
        .unwrap_or_else(|_| unreachable!("NMAX bounds Adler B"));
        for &byte in tail {
            a += u32::from(byte);
            b += a;
        }
        a %= BASE;
        b %= BASE;
    }
    a | (b << 16)
}
