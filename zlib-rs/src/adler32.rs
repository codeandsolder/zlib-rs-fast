//! The adler32 checksum algorithm.

mod simd;

#[must_use]
pub fn adler32(start_checksum: u32, data: &[u8]) -> u32 {
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
