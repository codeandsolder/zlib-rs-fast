use core::arch::loongarch64::{crc_w_b_w, crc_w_d_w, crc_w_h_w, crc_w_w_w};

use super::combine::{multmodp, x2nmodp};

// The constants here are taken from the AArch64 implementation.
// Playing around with them showed no gains from larger batches,
// but losses from smaller ones. No reason to deviate here.
const Z_BATCH: usize = 3990;
const Z_BATCH_ZEROS: u32 = x2nmodp(Z_BATCH as u64, 6);
const _: () = assert!(Z_BATCH_ZEROS == 0xa10d_3d0c);
const Z_BATCH_MIN: usize = 800;

/// 2-way interleaved CRC32, inspired by the AArch64 implementation.
///
/// Runs two independent crc32 streams simultaneously to exploit the
/// ~2-cycle throughput of the instruction, then combines them with
/// polynomial multiplication.
///
/// Throughput was measured around 1.995 on a Loongson 3B6000.
pub fn crc32_loongarch64(crc: u32, buf: &[u8]) -> u32 {
    let mut crc = (!crc).cast_signed();

    // SAFETY: [u8; 8] safely transmutes into i64.
    let (before, middle, after) = unsafe { buf.align_to::<i64>() };

    crc = remainder(crc, before);

    let mut words = middle;

    while words.len() >= 2 * Z_BATCH {
        let mut crc1: i32 = 0;
        for i in 0..Z_BATCH {
            crc = crc_w_d_w(words[i].to_le(), crc);
            crc1 = crc_w_d_w(words[i + Z_BATCH].to_le(), crc1);
        }
        words = &words[2 * Z_BATCH..];
        crc = multmodp(Z_BATCH_ZEROS, crc.cast_unsigned()).cast_signed() ^ crc1;
    }

    let last = words.len() / 2;
    if last >= Z_BATCH_MIN {
        let mut crc1: i32 = 0;
        for i in 0..last {
            crc = crc_w_d_w(words[i].to_le(), crc);
            crc1 = crc_w_d_w(words[i + last].to_le(), crc1);
        }
        words = &words[2 * last..];
        let val = x2nmodp(last as u64, 6);
        crc = multmodp(val, crc.cast_unsigned()).cast_signed() ^ crc1;
    }

    for &w in words {
        crc = crc_w_d_w(w.to_le(), crc);
    }

    crc = remainder(crc, after);

    (!crc).cast_unsigned()
}

#[inline]
fn remainder(mut c: i32, mut buf: &[u8]) -> i32 {
    if let [b0, b1, b2, b3, rest @ ..] = buf {
        c = crc_w_w_w(i32::from_le_bytes([*b0, *b1, *b2, *b3]), c);
        buf = rest;
    }

    if let [b0, b1, rest @ ..] = buf {
        c = crc_w_h_w(i16::from_le_bytes([*b0, *b1]), c);
        buf = rest;
    }

    if let [b0, rest @ ..] = buf {
        c = crc_w_b_w(b0.cast_signed(), c);
        buf = rest;
    }

    debug_assert_eq!(buf, []);

    c
}
