//! # Safety
//!
//! The functions in this module must only be executed on an ARM system with the CRC feature.
use core::arch::aarch64::{__crc32b, __crc32d, __crc32h, __crc32w};

use super::combine::{multmodp, x2nmodp};

// Constants empirically determined to maximize speed. These values are from
// measurements on a Cortex-A57. Your mileage may vary.
const Z_BATCH: usize = 3990;
const Z_BATCH_ZEROS: u32 = x2nmodp(Z_BATCH as u64, 6);
const _: () = assert!(Z_BATCH_ZEROS == 0xa10d_3d0c);
const Z_BATCH_MIN: usize = 800;

/// 3-way interleaved ARM CRC32, ported from stock zlib's ARMCRC32 path.
///
/// Runs three independent crc32x streams simultaneously to exploit the
/// ~3-cycle throughput of the instruction, then combines them with
/// polynomial multiplication.
///
/// Based on the ARMCRC32 code path in stock zlib.
#[target_feature(enable = "crc")]
pub fn crc32_acle_aarch64(crc: u32, buf: &[u8]) -> u32 {
    let mut crc = !crc;

    // SAFETY: [u8; 8] safely transmutes into u64.
    let (before, middle, after) = unsafe { buf.align_to::<u64>() };

    crc = remainder(crc, before);

    let mut words = middle;

    while words.len() >= 3 * Z_BATCH {
        let mut crc1: u32 = 0;
        let mut crc2: u32 = 0;
        for i in 0..Z_BATCH {
            crc = __crc32d(crc, words[i].to_le());
            crc1 = __crc32d(crc1, words[i + Z_BATCH].to_le());
            crc2 = __crc32d(crc2, words[i + 2 * Z_BATCH].to_le());
        }
        words = &words[3 * Z_BATCH..];
        crc = multmodp(Z_BATCH_ZEROS, crc) ^ crc1;
        crc = multmodp(Z_BATCH_ZEROS, crc) ^ crc2;
    }

    let last = words.len() / 3;
    if last >= Z_BATCH_MIN {
        let mut crc1: u32 = 0;
        let mut crc2: u32 = 0;
        for i in 0..last {
            crc = __crc32d(crc, words[i].to_le());
            crc1 = __crc32d(crc1, words[i + last].to_le());
            crc2 = __crc32d(crc2, words[i + 2 * last].to_le());
        }
        words = &words[3 * last..];
        let val = x2nmodp(last as u64, 6);
        crc = multmodp(val, crc) ^ crc1;
        crc = multmodp(val, crc) ^ crc2;
    }

    for &w in words {
        crc = __crc32d(crc, w.to_le());
    }

    crc = remainder(crc, after);

    !crc
}

#[inline]
#[target_feature(enable = "crc")]
fn remainder(mut c: u32, mut buf: &[u8]) -> u32 {
    if let [b0, b1, b2, b3, rest @ ..] = buf {
        c = __crc32w(c, u32::from_le_bytes([*b0, *b1, *b2, *b3]));
        buf = rest;
    }

    if let [b0, b1, rest @ ..] = buf {
        c = __crc32h(c, u16::from_le_bytes([*b0, *b1]));
        buf = rest;
    }

    if let [b0, rest @ ..] = buf {
        c = __crc32b(c, *b0);
        buf = rest;
    }

    debug_assert_eq!(buf, []);

    c
}

#[cfg(test)]
mod tests {
    use super::*;

    quickcheck::quickcheck! {
        fn crc32_acle_aarch64_is_crc32fast(v: Vec<u8>, start: u32) -> bool {
            if !crate::cpu_features::is_enabled_crc() {
                return true;
            }
            let mut h = crc32fast::Hasher::new_with_initial(start);
            h.update(&v);
            // SAFETY: the runtime feature check above verified CRC support.
            unsafe { crc32_acle_aarch64(start, &v) == h.finalize() }
        }
    }

    #[test]
    fn crc32_acle_handles_interleaved_batches_and_tails() {
        if !crate::cpu_features::is_enabled_crc() {
            return;
        }
        const CAPACITY: usize = 3 * Z_BATCH * 8 + 32;
        let input: Vec<u8> = (0..CAPACITY)
            .map(|index| index.to_le_bytes()[0].wrapping_mul(137))
            .collect();
        for offset in 0..8 {
            for length in [
                0,
                1,
                2,
                3,
                4,
                7,
                8,
                15,
                24,
                3 * Z_BATCH_MIN * 8 - 1,
                3 * Z_BATCH_MIN * 8,
                3 * Z_BATCH * 8 - 1,
                3 * Z_BATCH * 8,
                3 * Z_BATCH * 8 + 1,
                3 * Z_BATCH * 8 + 15,
            ] {
                for start in [0, 1, u32::MAX, 0xfedc_ba98] {
                    let bytes = &input[offset..offset + length];
                    let mut h = crc32fast::Hasher::new_with_initial(start);
                    h.update(bytes);
                    // SAFETY: CRC support was verified before entering the loop.
                    let actual = unsafe { crc32_acle_aarch64(start, bytes) };
                    assert_eq!(actual, h.finalize());
                }
            }
        }
    }
}
