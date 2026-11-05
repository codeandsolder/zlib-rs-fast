use super::braid::CRC32_LSB_POLY;

pub const fn crc32_combine(crc1: u32, crc2: u32, len2: u64) -> u32 {
    crc32_combine_op(crc1, crc2, crc32_combine_gen(len2))
}

#[inline(always)]
pub const fn crc32_combine_gen(len2: u64) -> u32 {
    x2nmodp(len2, 3)
}

#[inline(always)]
pub const fn crc32_combine_op(crc1: u32, crc2: u32, op: u32) -> u32 {
    multmodp(op, crc1) ^ crc2
}

const X2N_TABLE: [u32; 32] = [
    0x4000_0000,
    0x2000_0000,
    0x0800_0000,
    0x0080_0000,
    0x0000_8000,
    0xedb8_8320,
    0xb1e6_b092,
    0xa06a_2517,
    0xed62_7dae,
    0x88d1_4467,
    0xd7bb_fe6a,
    0xec44_7f11,
    0x8e7e_a170,
    0x6427_800e,
    0x4d47_bae0,
    0x09fe_548f,
    0x8385_2d0f,
    0x3036_2f1a,
    0x7b5a_9cc3,
    0x31fe_c169,
    0x9fec_022a,
    0x6c8d_edc4,
    0x15d6_874d,
    0x5fde_7a4e,
    0xbad9_0e37,
    0x2e4e_5eef,
    0x4eab_a214,
    0xa8a4_72c0,
    0x429a_969e,
    0x148d_302a,
    0xc40b_a6d0,
    0xc4e2_2c3c,
];

// Return a(x) multiplied by b(x) modulo p(x), where p(x) is the CRC polynomial,
// reflected. For speed, this requires that a not be zero.
pub(crate) const fn multmodp(a: u32, mut b: u32) -> u32 {
    let mut m = 1 << 31;
    let mut p = 0;

    loop {
        if (a & m) != 0 {
            p ^= b;
            if (a & (m - 1)) == 0 {
                break;
            }
        }
        m >>= 1;
        b = if (b & 1) != 0 {
            (b >> 1) ^ CRC32_LSB_POLY as u32
        } else {
            b >> 1
        };
    }

    p
}

// Return x^(n * 2^k) modulo p(x).
pub(crate) const fn x2nmodp(mut n: u64, mut k: u32) -> u32 {
    let mut p: u32 = 1 << 31; /* x^0 == 1 */

    while n > 0 {
        if (n & 1) != 0 {
            p = multmodp(X2N_TABLE[k as usize & 31], p);
        }
        n >>= 1;
        k += 1;
    }

    p
}

#[cfg(test)]
mod test {
    use super::*;

    use crate::crc32::crc32;

    #[test]
    fn test_crc32_combine() {
        ::quickcheck::quickcheck(test as fn(_) -> _);

        fn test(data: Vec<u8>) -> bool {
            let Some(buf_len) = data.first().copied() else {
                return true;
            };

            let buf_size = Ord::max(buf_len, 1) as usize;

            let crc0 = 0;
            let mut crc1 = crc0;
            let mut crc2 = crc0;

            /* CRC32 */
            for chunk in data.chunks(buf_size) {
                let crc3 = crc32(crc0, chunk);
                let op = crc32_combine_gen(chunk.len() as _);
                let crc4 = crc32_combine_op(crc1, crc3, op);
                crc1 = crc32(crc1, chunk);

                assert_eq!(crc1, crc4);
            }

            crc2 = crc32(crc2, &data);

            assert_eq!(crc1, crc2);

            let combine1 = crc32_combine(crc1, crc2, data.len() as _);
            let combine2 = crc32_combine(crc1, crc1, data.len() as _);
            assert_eq!(combine1, combine2);

            // Fast CRC32 combine.
            let op = crc32_combine_gen(data.len() as _);
            let combine1 = crc32_combine_op(crc1, crc2, op);
            let combine2 = crc32_combine_op(crc2, crc1, op);
            assert_eq!(combine1, combine2);

            let combine1 = crc32_combine(crc1, crc2, data.len() as _);
            let combine2 = crc32_combine_op(crc2, crc1, op);
            assert_eq!(combine1, combine2);

            true
        }
    }
}
