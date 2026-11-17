//! The adler32 checksum algorithm.

mod simd;

#[cfg(all(target_arch = "x86_64", feature = "std"))]
mod avx2;

// Measured on AVX2 hardware across 32 source alignments: below 1 KiB,
// portable SIMD avoids feature-check and scalar alignment-prefix overhead.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
const AVX2_MIN_INPUT: usize = 1024;

#[must_use]
pub fn adler32(start_checksum: u32, data: &[u8]) -> u32 {
    // AVX2 wins consistently at this size, but carries overhead on short
    // and misaligned incremental updates. Keep those on portable SIMD.
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    if data.len() >= AVX2_MIN_INPUT && std::is_x86_feature_detected!("avx2") {
        return avx2::adler32_avx2(start_checksum, data);
    }

    simd::adler32(start_checksum, data)
}

// The compressing parser benefits from preserving its portable-SIMD checksum
// path rather than taking the otherwise faster AVX2 checksum dispatch.
// Keep the choice local to the encoder, where the whole codec was measured.
#[inline]
pub(crate) fn adler32_for_deflate(start_checksum: u32, data: &[u8]) -> u32 {
    simd::adler32(start_checksum, data)
}

#[must_use]
pub fn adler32_combine(adler1: u32, adler2: u32, len2: u64) -> u32 {
    const BASE: u64 = self::BASE as u64;

    let rem = len2 % BASE;

    let adler1 = u64::from(adler1);
    let adler2 = u64::from(adler2);

    /* the derivation of this formula is left as an exercise for the reader */
    let mut sum1 = adler1 & 0xffff;
    let mut sum2 = rem * sum1;
    sum2 %= BASE;
    sum1 += (adler2 & 0xffff) + BASE - 1;
    sum2 += ((adler1 >> 16) & 0xffff) + ((adler2 >> 16) & 0xffff) + BASE - rem;

    if sum1 >= BASE {
        sum1 -= BASE;
    }
    if sum1 >= BASE {
        sum1 -= BASE;
    }
    if sum2 >= (BASE << 1) {
        sum2 -= BASE << 1;
    }
    if sum2 >= BASE {
        sum2 -= BASE;
    }

    // Both limbs have been reduced modulo BASE (< 2^16), so the packed
    // result always fits the Adler-32 return type.
    let packed = sum1 | (sum2 << 16);
    u32::try_from(packed)
        .unwrap_or_else(|_| unreachable!("two reduced Adler-32 limbs exceed 32 bits"))
}

// inefficient but correct, useful for testing
#[cfg(test)]
fn naive_adler32(start_checksum: u32, data: &[u8]) -> u32 {
    const MOD_ADLER: u32 = 65521; // Largest prime smaller than 2^16

    let mut a = start_checksum & 0xFFFF;
    let mut b = (start_checksum >> 16) & 0xFFFF;

    for &byte in data {
        a = (a + u32::from(byte)) % MOD_ADLER;
        b = (b + a) % MOD_ADLER;
    }

    (b << 16) | a
}

const BASE: u32 = 65521; /* largest prime smaller than 65536 */
const NMAX: u32 = 5552;

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn naive_is_fancy_small_inputs() {
        for i in 0..128 {
            let v = (0u8..i).collect::<Vec<_>>();
            assert_eq!(naive_adler32(1, &v), adler32(1, &v));
        }
    }

    quickcheck::quickcheck! {
        fn portable_simd_matches_naive(data: Vec<u8>, start: u32) -> bool {
            adler32(start, &data) == naive_adler32(start, &data)
        }
    }

    #[test]
    fn portable_simd_matches_naive_at_chunk_boundaries() {
        const LENGTHS: &[usize] = &[
            0,
            1,
            15,
            16,
            17,
            NMAX as usize - 1,
            NMAX as usize,
            NMAX as usize + 1,
            2 * NMAX as usize + 7,
        ];

        for &len in LENGTHS {
            let data: Vec<_> = (0..len)
                .map(|index| index.wrapping_mul(131).wrapping_add(17).to_le_bytes()[0])
                .collect();

            for start in [0, 1, 42, u32::MAX] {
                assert_eq!(
                    adler32(start, &data),
                    naive_adler32(start, &data),
                    "len={len}, start={start}"
                );
            }
        }
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    fn restored_avx2_matches_portable_at_lengths_offsets_and_seeds() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut state = 0x243f_6a88_85a3_08d3u64;
        let mut data = vec![0u8; 4 * 1024 * 1024 + 64];
        for byte in &mut data {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *byte = state.to_le_bytes()[4];
        }
        let lengths = [
            0,
            1,
            2,
            3,
            15,
            16,
            17,
            31,
            32,
            33,
            63,
            64,
            65,
            127,
            128,
            129,
            255,
            256,
            257,
            351,
            352,
            353,
            1023,
            1024,
            1025,
            NMAX as usize - 1,
            NMAX as usize,
            NMAX as usize + 1,
            2 * NMAX as usize,
            2 * NMAX as usize + 1,
            32 * 1024,
            64 * 1024,
            256 * 1024,
            4 * 1024 * 1024,
        ];
        for offset in 0..32 {
            for &len in &lengths {
                let slice = &data[offset..offset + len];
                for start in [0, 1, 42, u32::MAX, 0xffff_0001, 0x1234_abcd] {
                    let expected = simd::adler32(start, slice);
                    let actual = avx2::adler32_avx2(start, slice);
                    assert_eq!(
                        actual, expected,
                        "offset={offset} len={len} seed={start:#x}"
                    );
                    assert_eq!(
                        adler32(start, slice),
                        expected,
                        "dispatch offset={offset} len={len} seed={start:#x}"
                    );
                }
            }
        }
        for len in [
            127,
            128,
            129,
            NMAX as usize,
            NMAX as usize + 1,
            2 * NMAX as usize,
            4 * 1024 * 1024,
        ] {
            let all_ff = vec![u8::MAX; len];
            for start in [0, 1, 42, u32::MAX] {
                assert_eq!(
                    avx2::adler32_avx2(start, &all_ff),
                    naive_adler32(start, &all_ff),
                    "all-ff len={len} seed={start:#x}"
                );
            }
        }
    }

    #[test]
    fn test_adler32_combine() {
        fn test(data: &[u8]) -> bool {
            let Some(buf_len) = data.first().copied() else {
                return true;
            };

            let buf_size = Ord::max(buf_len, 1) as usize;

            let mut adler1 = 1;
            for chunk in data.chunks(buf_size) {
                adler1 = adler32(adler1, chunk);
            }

            let adler2 = adler32(1, data);

            assert_eq!(adler1, adler2);

            let combine1 = adler32_combine(adler1, adler2, data.len() as _);
            let combine2 = adler32_combine(adler1, adler1, data.len() as _);
            assert_eq!(combine1, combine2);

            true
        }

        let property: fn(Vec<u8>) -> bool = |data| test(&data);
        ::quickcheck::quickcheck(property);
    }

    #[test]
    fn largest_nmax_block_never_overflows_simd_adler() {
        for len in [
            NMAX as usize - 1,
            NMAX as usize,
            NMAX as usize + 1,
            2 * NMAX as usize,
        ] {
            let data = vec![u8::MAX; len];
            for initial in [1, u32::MAX, 0xffff_0001] {
                assert_eq!(
                    adler32(initial, &data),
                    naive_adler32(initial, &data),
                    "worst-case SIMD sum: len={len}, initial={initial:#x}"
                );
            }
        }
    }
}
