//! AVX2 Adler-32 kernel with vector byte sums and weighted dot products.
//!
//! The public checksum dispatch guards this module with runtime AVX2 detection.
//! The vector operations are deliberately isolated from the portable-SIMD path.
use core::arch::x86_64::{
    __m256i, _mm256_add_epi32, _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_madd_epi16,
    _mm256_maddubs_epi16, _mm256_permutevar8x32_epi32, _mm256_sad_epu8, _mm256_slli_epi32,
    _mm256_zextsi128_si256, _mm_add_epi32, _mm_cvtsi128_si32, _mm_cvtsi32_si128, _mm_shuffle_epi32,
    _mm_unpackhi_epi64,
};

use crate::adler32::{BASE, NMAX};

#[inline]
fn scalar(adler: u32, bytes: &[u8]) -> u32 {
    let mut a = adler & 0xffff;
    let mut b = adler >> 16;
    for &byte in bytes {
        a += u32::from(byte);
        b += a;
    }
    (a % BASE) | ((b % BASE) << 16)
}

const fn __m256i_literal(bytes: [u8; 32]) -> __m256i {
    // SAFETY: any valid [u8; 32] represents a valid __m256i
    unsafe { core::mem::transmute(bytes) }
}

const DOT2V: __m256i = __m256i_literal([
    32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9,
    8, 7, 6, 5, 4, 3, 2, 1,
]);

const DOT3V: __m256i = __m256i_literal([
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
]);

const ZERO: __m256i = __m256i_literal([0u8; 32]);

/// 32 bit horizontal sum, adapted from Agner Fog's vector library.
#[target_feature(enable = "avx2")]
unsafe fn hsum256(x: __m256i) -> u32 {
    let sum1 = _mm_add_epi32(_mm256_extracti128_si256(x, 1), _mm256_castsi256_si128(x));
    let sum2 = _mm_add_epi32(sum1, _mm_unpackhi_epi64(sum1, sum1));
    let sum3 = _mm_add_epi32(sum2, _mm_shuffle_epi32(sum2, 1));
    _mm_cvtsi128_si32(sum3).cast_unsigned()
}

#[target_feature(enable = "avx2")]
unsafe fn partial_hsum256(x: __m256i) -> u32 {
    const PERM_VEC: __m256i = __m256i_literal([
        0, 0, 0, 0, //
        2, 0, 0, 0, //
        4, 0, 0, 0, //
        6, 0, 0, 0, //
        1, 0, 0, 0, //
        1, 0, 0, 0, //
        1, 0, 0, 0, //
        1, 0, 0, 0, //
    ]);

    let non_zero = _mm256_permutevar8x32_epi32(x, PERM_VEC);
    let non_zero_sse = _mm256_castsi256_si128(non_zero);
    let sum2 = _mm_add_epi32(non_zero_sse, _mm_unpackhi_epi64(non_zero_sse, non_zero_sse));
    let sum3 = _mm_add_epi32(sum2, _mm_shuffle_epi32(sum2, 1));
    _mm_cvtsi128_si32(sum3).cast_unsigned()
}

pub(super) fn adler32_avx2(adler: u32, src: &[u8]) -> u32 {
    assert!(std::is_x86_feature_detected!("avx2"));
    // SAFETY: the runtime feature check guarantees AVX2 for the guarded call.
    unsafe { adler32_avx2_help(adler, src) }
}

#[target_feature(enable = "avx2")]
unsafe fn adler32_avx2_help(adler: u32, src: &[u8]) -> u32 {
    if src.is_empty() {
        return adler;
    }

    // SAFETY: any 256 bits represent a valid __m256i; align_to returns
    // 32-byte-aligned fully initialized words and byte prefixes/suffixes.
    let (before, middle, after) = unsafe { src.align_to::<__m256i>() };

    let adler = scalar(adler, before);
    let mut adler1 = adler >> 16;
    let mut adler0 = adler & 0xffff;

    // Reduce every 173 vectors (5536 bytes) at most. NMAX=5552 bounds the
    // worst-case cumulative Adler sums below u32::MAX for every initial seed.
    for chunk in middle.chunks(NMAX as usize / 32) {
        // SAFETY: this function is reached only via the AVX2 feature-checked
        // entrypoint; the helper has the same AVX2 requirement.
        (adler0, adler1) = unsafe { helper_32_bytes(adler0, adler1, chunk) };
    }

    if !after.is_empty() {
        return scalar(adler0 | (adler1 << 16), after);
    }

    adler0 | (adler1 << 16)
}

#[target_feature(enable = "avx2")]
unsafe fn helper_32_bytes(adler0: u32, adler1: u32, src: &[__m256i]) -> (u32, u32) {
    let mut vs1 = _mm256_zextsi128_si256(_mm_cvtsi32_si128(adler0.cast_signed()));
    let mut vs2 = _mm256_zextsi128_si256(_mm_cvtsi32_si128(adler1.cast_signed()));

    let mut vs1_0 = vs1;
    let mut vs3 = ZERO;

    for vbuf in src.iter().copied() {
        let vs1_sad = _mm256_sad_epu8(vbuf, ZERO); // Sum of abs diff, resulting in 2 x int32's

        vs1 = _mm256_add_epi32(vs1, vs1_sad);
        vs3 = _mm256_add_epi32(vs3, vs1_0);
        // Largest two-byte dot product is 255 * (32 + 31) = 16065,
        // below i16::MAX, so saturating multiply-add cannot lose data.
        let v_short_sum2 = _mm256_maddubs_epi16(vbuf, DOT2V);
        let vsum2 = _mm256_madd_epi16(v_short_sum2, DOT3V);
        vs2 = _mm256_add_epi32(vsum2, vs2);
        vs1_0 = vs1;
    }

    /* Defer the multiplication with 32 to outside of the loop */
    vs3 = _mm256_slli_epi32(vs3, 5);
    vs2 = _mm256_add_epi32(vs2, vs3);

    // SAFETY: both reduction helpers have the same AVX2 requirement as
    // this function, reached only from the guarded AVX2 entrypoint.
    unsafe { (partial_hsum256(vs1) % BASE, hsum256(vs2) % BASE) }
}
