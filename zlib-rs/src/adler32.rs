//! The adler32 checksum algorithm.

mod simd;

// Compile SIMD-width variants where supported, and in tests.
#[cfg(any(
    test,
    all(
        target_arch = "x86_64",
        any(feature = "std", target_feature = "avx2", target_feature = "avx512bw")
    ),
    target_arch = "aarch64",
    all(target_arch = "arm", target_feature = "neon"),
    all(target_arch = "wasm32", target_feature = "simd128")
))]
mod weighted;

// Below 1 KiB, the existing short-update SIMD kernel is preferable.
const WIDE_SIMD_MIN_INPUT: usize = 1024;

#[must_use]
pub fn adler32(start_checksum: u32, data: &[u8]) -> u32 {
    if data.len() >= WIDE_SIMD_MIN_INPUT {
        #[cfg(all(target_arch = "x86_64", feature = "std"))]
        {
            // 64 u16 lanes under AVX-512BW, 32 under AVX2.
            if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw")
            {
                // SAFETY: runtime CPU and OS checks establish both features.
                return unsafe { weighted::checksum_avx512(start_checksum, data) };
            }
            if std::is_x86_feature_detected!("avx2") {
                // SAFETY: runtime CPU and OS feature detection establishes AVX2.
                return unsafe { weighted::checksum_avx2(start_checksum, data) };
            }
        }
        #[cfg(all(target_arch = "x86_64", not(feature = "std")))]
        {
            // Freestanding builds cannot perform runtime feature detection.
            // When the target itself requires AVX2 or AVX-512BW, use its
            // statically enabled native width.
            #[cfg(target_feature = "avx512bw")]
            return weighted::checksum::<64>(start_checksum, data);
            #[cfg(all(not(target_feature = "avx512bw"), target_feature = "avx2"))]
            return weighted::checksum::<32>(start_checksum, data);
        }
        #[cfg(any(
            target_arch = "aarch64",
            all(target_arch = "arm", target_feature = "neon"),
            all(target_arch = "wasm32", target_feature = "simd128")
        ))]
        {
            // 8 u16 lanes match a native 128-bit NEON or Wasm SIMD register.
            return weighted::checksum::<8>(start_checksum, data);
        }
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
    fn architecture_widths_match_portable_across_lengths_alignments_and_seeds() {
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
            175,
            176,
            177,
            255,
            256,
            257,
            351,
            352,
            353,
            703,
            704,
            705,
            1023,
            1024,
            1025,
            1407,
            1408,
            1409,
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
                if len > 256 * 1024 && offset != 0 && offset != 31 {
                    continue;
                }
                let slice = &data[offset..offset + len];
                for initial in [0, 1, 42, u32::MAX, 0xffff_0001, 0x1234_abcd] {
                    let expected = simd::adler32(initial, slice);
                    assert_eq!(
                        weighted::checksum::<8>(initial, slice),
                        expected,
                        "8 lanes offset={offset} len={len} initial={initial:#x}"
                    );
                    assert_eq!(
                        weighted::checksum::<16>(initial, slice),
                        expected,
                        "16 lanes offset={offset} len={len} initial={initial:#x}"
                    );
                    assert_eq!(
                        weighted::checksum::<32>(initial, slice),
                        expected,
                        "32 lanes offset={offset} len={len} initial={initial:#x}"
                    );
                    assert_eq!(
                        weighted::checksum::<64>(initial, slice),
                        expected,
                        "64 lanes offset={offset} len={len} initial={initial:#x}"
                    );
                    assert_eq!(
                        adler32(initial, slice),
                        expected,
                        "dispatch offset={offset} len={len} initial={initial:#x}"
                    );
                }
            }
        }
    }

    #[test]
    fn architecture_widths_preserve_worst_case_nmax_sums() {
        for len in [
            127,
            128,
            129,
            NMAX as usize - 1,
            NMAX as usize,
            NMAX as usize + 1,
            2 * NMAX as usize,
            4 * 1024 * 1024,
        ] {
            let all_ff = vec![u8::MAX; len];
            for initial in [0, 1, 42, u32::MAX] {
                let expected = naive_adler32(initial, &all_ff);
                for actual in [
                    weighted::checksum::<8>(initial, &all_ff),
                    weighted::checksum::<16>(initial, &all_ff),
                    weighted::checksum::<32>(initial, &all_ff),
                    weighted::checksum::<64>(initial, &all_ff),
                    adler32(initial, &all_ff),
                ] {
                    assert_eq!(actual, expected, "all-0xff len={len} initial={initial:#x}");
                }
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
