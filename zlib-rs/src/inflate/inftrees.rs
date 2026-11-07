#![forbid(unsafe_code)]

use crate::{Code, ENOUGH_DISTS, ENOUGH_LENS};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodeType {
    Codes,
    Lens,
    Dists,
}

const MAX_BITS: usize = 15;

/// Length codes 257..285 base
const LBASE: [u16; 31] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258, 0, 0,
];
/// Length codes 257..285 extra
const LEXT: [u8; 31] = [
    16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20,
    21, 21, 21, 21, 16, 77, 202,
];
/// Distance codes 0..29 base
const DBASE: [u16; 32] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577, 0, 0,
];
/// Distance codes 0..29 extra
const DEXT: [u8; 32] = [
    16, 16, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26,
    27, 27, 28, 28, 29, 29, 64, 64,
];

#[repr(i32)]
#[derive(Debug, PartialEq, Eq)]
pub enum InflateTable {
    EnoughIsNotEnough = 1,
    Success { root: usize, used: usize } = 0,
    InvalidCode = -1,
}

pub fn inflate_table(
    codetype: CodeType,
    lens: &[u16],
    table: &mut [Code],
    bits: usize,
    work: &mut [u16],
) -> InflateTable {
    // The DEFLATE format bounds each alphabet. Enforce that contract once,
    // before counting lengths in u16 and storing symbol indices in u16 scratch.
    let max_symbols = match codetype {
        CodeType::Codes => 19,
        CodeType::Lens => 288, // includes the two fixed-tree-only symbols
        CodeType::Dists => 32,
    };
    if lens.len() > max_symbols {
        return InflateTable::InvalidCode;
    }
    if work.len() < lens.len() {
        return InflateTable::EnoughIsNotEnough;
    }

    // number of codes of each length
    let mut count = [0u16; MAX_BITS + 1];

    let (mut min, mut max) = (MAX_BITS, 0);
    for &len in lens {
        if usize::from(len) > MAX_BITS {
            return InflateTable::InvalidCode;
        }
        if len > 0 {
            count[usize::from(len)] += 1;
            max = Ord::max(max, usize::from(len));
            min = Ord::min(min, usize::from(len));
        }
    }

    if max == 0 {
        // no symbols to code at all
        let code = Code {
            op: 64,
            bits: 1,
            val: 0,
        };

        if table.len() < 2 {
            return InflateTable::EnoughIsNotEnough;
        }
        table[0] = code;
        table[1] = code;

        return InflateTable::Success { root: 1, used: 2 };
    }

    // NOTE: if max != 0, then min has been set to a value <= max.
    let root = bits.clamp(min, max);

    /* check for an over-subscribed or incomplete set of lengths */
    let mut left = 1u32;
    for &sym in &count[1..] {
        left = match (left << 1).checked_sub(u32::from(sym)) {
            None => return InflateTable::InvalidCode, // over-subscribed
            Some(v) => v,
        };
    }

    if left > 0 && (matches!(codetype, CodeType::Codes) || max != 1) {
        // incomplete set
        return InflateTable::InvalidCode;
    }

    /* generate offsets into symbol table for each length for sorting */

    // offsets in table for each length
    let mut offs = [0u16; MAX_BITS + 1];
    for len in 1..MAX_BITS {
        offs[len + 1] = offs[len] + count[len];
    }

    /* sort symbols by length, by symbol order within each length */
    for (sym, len) in lens.iter().copied().enumerate() {
        if len != 0 {
            let offset = &mut offs[usize::from(len)];
            let Ok(symbol) = u16::try_from(sym) else {
                return InflateTable::InvalidCode;
            };
            work[usize::from(*offset)] = symbol;
            *offset += 1;
        }
    }

    let (base, extra, match_) = match codetype {
        CodeType::Codes => (&[] as &[_], &[] as &[_], 20),
        CodeType::Lens => (&LBASE[..], &LEXT[..], 257),
        CodeType::Dists => (&DBASE[..], &DEXT[..], 0),
    };

    let mut used = 1 << root;

    /* check available table space */
    if matches!(codetype, CodeType::Lens) && used > ENOUGH_LENS {
        return InflateTable::EnoughIsNotEnough;
    }

    if matches!(codetype, CodeType::Dists) && used > ENOUGH_DISTS {
        return InflateTable::EnoughIsNotEnough;
    }
    if used > table.len() {
        return InflateTable::EnoughIsNotEnough;
    }

    let mut huff = 0; // starting code
    let mut reversed_huff = 0u32; // starting code, reversed
    let mut sym = 0;
    let mut len = min;
    let mut next = 0usize; // index into `table`
    let mut curr = root;
    let mut drop_ = 0;
    let mut low = usize::MAX; // trigger new subtable when len > root
    let mask = used - 1; /* mask for comparing low */

    // process all codes and make table entries
    'outer: loop {
        // The format guarantees a byte-sized bit count, so do not narrow
        // an invalid length when constructing a decode entry.
        let Ok(code_bits) = u8::try_from(len - drop_) else {
            return InflateTable::InvalidCode;
        };
        // create table entry
        let here = if work[sym] >= match_ {
            Code {
                bits: code_bits,
                op: extra[usize::from(work[sym] - match_)],
                val: base[usize::from(work[sym] - match_)],
            }
        } else if work[sym] + 1 < match_ {
            Code {
                bits: code_bits,
                op: 0,
                val: work[sym],
            }
        } else {
            Code {
                bits: code_bits,
                op: 0b0110_0000,
                val: 0,
            }
        };

        // replicate for those indices with low len bits equal to huff
        let incr = 1 << (len - drop_);
        let mut fill = 1 << curr;
        let min = fill;

        loop {
            fill -= incr;
            table[next + (huff >> drop_) + fill] = here;

            if fill == 0 {
                break;
            }
        }

        // backwards increment the len-bit code huff
        reversed_huff = reversed_huff.wrapping_add(0x8000_0000u32 >> (len - 1));
        let Ok(reversed) = usize::try_from(reversed_huff.reverse_bits()) else {
            return InflateTable::InvalidCode;
        };
        huff = reversed;

        // go to next symbol, update count, len
        sym += 1;
        count[len] -= 1;
        if count[len] == 0 {
            if len == max {
                break 'outer;
            }
            len = usize::from(lens[usize::from(work[sym])]);
        }

        // create new sub-table if needed
        if len > root && (huff & mask) != low {
            /* if first time, transition to sub-tables */
            if drop_ == 0 {
                drop_ = root;
            }

            /* increment past last table */
            next += min; /* here min is 1 << curr */

            /* determine length of next table */
            curr = len - drop_;
            let mut left = 1 << curr;
            while curr + drop_ < max {
                left -= i32::from(count[curr + drop_]);
                if left <= 0 {
                    break;
                }
                curr += 1;
                left <<= 1;
            }

            /* check for enough space */
            used += 1usize << curr;

            if matches!(codetype, CodeType::Lens) && used > ENOUGH_LENS {
                return InflateTable::EnoughIsNotEnough;
            }

            if matches!(codetype, CodeType::Dists) && used > ENOUGH_DISTS {
                return InflateTable::EnoughIsNotEnough;
            }
            if used > table.len() {
                return InflateTable::EnoughIsNotEnough;
            }

            /* point entry in root table to sub-table */
            low = huff & mask;
            let (Ok(op), Ok(bits), Ok(val)) =
                (u8::try_from(curr), u8::try_from(root), u16::try_from(next))
            else {
                return InflateTable::EnoughIsNotEnough;
            };
            table[low] = Code { op, bits, val };
        }
    }

    /* fill in remaining table entry if code is incomplete (guaranteed to have
    at most one remaining entry, since if the code is incomplete, the
    maximum code length that was allowed to get this far is one bit) */
    if huff != 0 {
        let Ok(bits) = u8::try_from(len - drop_) else {
            return InflateTable::InvalidCode;
        };
        let here = Code {
            op: 64,
            bits,
            val: 0,
        };

        table[next..][huff] = here;
    }

    /* set return parameters */
    InflateTable::Success { root, used }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn rejects_invalid_alphabets_without_indexing_outside_scratch() {
        for (kind, allowed) in [
            (CodeType::Codes, 19usize),
            (CodeType::Lens, 288),
            (CodeType::Dists, 32),
        ] {
            let mut table = [Code::default(); crate::ENOUGH_LENS];
            let mut work = [0u16; 288];
            let mut oversize = vec![0u16; allowed + 1];
            assert_eq!(
                inflate_table(kind, &oversize, &mut table, 7, &mut work),
                InflateTable::InvalidCode
            );
            oversize.truncate(allowed);
            if !oversize.is_empty() {
                oversize[0] = 16;
                assert_eq!(
                    inflate_table(kind, &oversize, &mut table, 7, &mut work),
                    InflateTable::InvalidCode
                );
            }
        }
    }

    #[test]
    fn rejects_insufficient_scratch_and_table_capacity() {
        let mut table = [Code::default(); 8];
        let mut short_work = [0u16; 1];
        let zero_codes = [0u16; 2];
        assert_eq!(
            inflate_table(CodeType::Codes, &zero_codes, &mut table, 7, &mut short_work),
            InflateTable::EnoughIsNotEnough
        );

        let mut work = [0u16; 2];
        assert_eq!(
            inflate_table(CodeType::Codes, &zero_codes, &mut table[..1], 7, &mut work),
            InflateTable::EnoughIsNotEnough
        );
        // Both a valid one-bit root and its corresponding two table entries
        // must fit, even if the caller supplies an empty or short output.
        let one_bit = [1u16, 1];
        assert_eq!(
            inflate_table(CodeType::Codes, &one_bit, &mut table[..1], 1, &mut work),
            InflateTable::EnoughIsNotEnough
        );
        assert_eq!(
            inflate_table(CodeType::Codes, &one_bit, &mut table[..2], 1, &mut work),
            InflateTable::Success { root: 1, used: 2 }
        );
    }

    #[test]
    fn not_enough_errors() {
        // we need to call inflate_table() directly in order to manifest
        // not-enough errors, since zlib insures that enough is always enough

        let table = [Code::default(); crate::ENOUGH_DISTS];

        let mut work = [0; 16];
        let lens: [u16; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15];

        let mut next = table;
        let bits = 15;
        let ret = inflate_table(CodeType::Dists, &lens[..16], &mut next, bits, &mut work);
        assert_eq!(ret, InflateTable::EnoughIsNotEnough);

        let mut next = table;
        let bits = 1;
        let ret = inflate_table(CodeType::Dists, &lens[..16], &mut next, bits, &mut work);
        assert_eq!(ret, InflateTable::EnoughIsNotEnough);
    }

    fn build_fixed_length_table(work: &mut [u16]) -> [Code; 512] {
        let mut lens = [0; 288];

        // literal/length table
        let mut sym = 0usize;
        while sym < 144 {
            lens[sym] = 8;
            sym += 1;
        }
        while sym < 256 {
            lens[sym] = 9;
            sym += 1;
        }
        while sym < 280 {
            lens[sym] = 7;
            sym += 1;
        }
        while sym < 288 {
            lens[sym] = 8;
            sym += 1;
        }

        let mut next = [Code::default(); 512];
        let bits = 9;
        inflate_table(CodeType::Lens, &lens[..288], &mut next, bits, work);

        core::array::from_fn(|i| {
            let mut code = next[i];

            code.op = if i & 0b0111_1111 == 99 { 64 } else { code.op };

            code
        })
    }

    #[test]
    fn generate_fixed_length_table() {
        let mut work = [0; 512];
        let generated = build_fixed_length_table(&mut work);

        assert_eq!(generated, crate::inflate::inffixed_tbl::LENFIX);
    }

    fn build_fixed_distance_table(work: &mut [u16]) -> [Code; 32] {
        let mut lens = [0; 288];

        let mut sym = 0;
        while sym < 32 {
            lens[sym] = 5;
            sym += 1;
        }

        let mut next = [Code::default(); 32];
        let bits = 5;
        inflate_table(CodeType::Dists, &lens[..32], &mut next, bits, work);

        next
    }

    #[test]
    fn generate_fixed_distance_table() {
        let mut work = [0; 512];
        let generated = build_fixed_distance_table(&mut work);

        assert_eq!(generated, crate::inflate::inffixed_tbl::DISTFIX);
    }
}
