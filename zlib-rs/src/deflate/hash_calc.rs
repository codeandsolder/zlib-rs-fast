#![forbid(unsafe_code)]

use crate::deflate::{State, HASH_SIZE, STD_MIN_MATCH};

// Hash-chain entries are 16-bit window positions even when their caller
// tracks the absolute position as usize. Retain the low word on rollover;
// rejecting an index over u16::MAX would change zlib's hash-history behavior.
// The source bytes are laid out in little-endian order irrespective of the
// target's native endianness, and this compiles to a truncation instruction.
#[inline]
pub(super) const fn hash_chain_position(position: usize) -> u16 {
    let bytes = position.to_le_bytes();
    u16::from_le_bytes([bytes[0], bytes[1]])
}

// Standard hashing deliberately uses all sixteen hash bits.
const _: () = assert!(HASH_SIZE == (1usize << u16::BITS));

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashCalcVariant {
    Standard,
    Roll,
}

pub struct StandardHashCalc;

impl StandardHashCalc {
    const HASH_CALC_OFFSET: usize = 0;

    const HASH_CALC_MASK: u32 = (1u32 << u16::BITS) - 1;

    const fn hash_calc(_: u32, val: u32) -> u32 {
        const HASH_SLIDE: u32 = 16;
        val.wrapping_mul(2_654_435_761) >> HASH_SLIDE
    }

    pub const fn update_hash(h: u32, val: u32) -> u32 {
        Self::hash_calc(h, val) & Self::HASH_CALC_MASK
    }

    /// Read a four-byte hash probe from initialized window bytes.
    ///
    /// Every caller has already established at least four bytes of lookahead.
    /// Array chunks express that precondition directly without an additional
    /// fallible slice-to-array conversion on the match-search hot path.
    #[inline]
    pub(crate) const fn first_four_bytes(bytes: &[u8]) -> u32 {
        u32::from_le_bytes(bytes.as_chunks::<4>().0[0])
    }

    #[inline]
    pub fn quick_insert_string(state: &mut State, string: usize) -> u16 {
        let slice = &state.window.filled()[string + Self::HASH_CALC_OFFSET..];
        let val = Self::first_four_bytes(slice);

        Self::quick_insert_value(state, string, val)
    }

    #[inline]
    pub const fn quick_insert_value(state: &mut State, string: usize, val: u32) -> u16 {
        let hm = Self::update_hash(0, val) as usize;

        let head = state.head.as_slice()[hm];
        let position = hash_chain_position(string);
        if head != position {
            let w_mask = state.w_mask();
            state.prev.as_mut_slice()[string & w_mask] = head;
            state.head.as_mut_slice()[hm] = position;
        }

        head
    }

    pub fn insert_string(state: &mut State, string: usize, count: usize) {
        let slice = &state.window.filled()[string + Self::HASH_CALC_OFFSET..];

        // it can happen that insufficient bytes are initialized
        // .take(count) generates worse assembly
        let slice = &slice[..Ord::min(slice.len(), count + 3)];

        let w_mask = state.w_mask();
        let head = state.head.as_mut_slice();
        let prev = state.prev.as_mut_slice();
        for (i, w) in slice.windows(4).enumerate() {
            let idx = hash_chain_position(string + i);
            let hash = Self::update_hash(0, Self::first_four_bytes(w)) as usize;
            let entry = &mut head[hash];
            if *entry != idx {
                prev[usize::from(idx) & w_mask] = *entry;
                *entry = idx;
            }
        }
    }
}

pub struct RollHashCalc;

impl RollHashCalc {
    const HASH_CALC_OFFSET: usize = STD_MIN_MATCH - 1;

    const HASH_CALC_MASK: u32 = (1 << 15) - 1;

    const fn hash_calc(h: u32, val: u32) -> u32 {
        const HASH_SLIDE: u32 = 5;
        (h << HASH_SLIDE) ^ val
    }

    pub const fn update_hash(h: u32, val: u32) -> u32 {
        Self::hash_calc(h, val) & Self::HASH_CALC_MASK
    }

    pub fn quick_insert_string(state: &mut State, string: usize) -> u16 {
        let val = u32::from(state.window.filled()[string + Self::HASH_CALC_OFFSET]);

        state.ins_h = Self::hash_calc(state.ins_h, val);
        state.ins_h &= Self::HASH_CALC_MASK;

        let hm = state.ins_h as usize;

        let head = state.head.as_slice()[hm];
        let position = hash_chain_position(string);
        if head != position {
            let w_mask = state.w_mask();
            state.prev.as_mut_slice()[string & w_mask] = head;
            state.head.as_mut_slice()[hm] = position;
        }

        head
    }

    pub fn insert_string(state: &mut State, string: usize, count: usize) {
        let slice = &state.window.filled()[string + Self::HASH_CALC_OFFSET..][..count];

        let w_mask = state.w_mask();
        let head = state.head.as_mut_slice();
        let prev = state.prev.as_mut_slice();
        let mut ins_h = state.ins_h;
        for (i, val) in slice.iter().copied().enumerate() {
            let idx = hash_chain_position(string + i);
            ins_h = Self::update_hash(ins_h, u32::from(val));
            let entry = &mut head[ins_h as usize];
            if *entry != idx {
                prev[usize::from(idx) & w_mask] = *entry;
                *entry = idx;
            }
        }
        state.ins_h = ins_h;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_chain_position_retains_low_word_across_window_boundary() {
        for (position, expected) in [
            (0, 0u16),
            (usize::from(u16::MAX), u16::MAX),
            (usize::from(u16::MAX) + 1, 0),
            (usize::from(u16::MAX) + 2, 1),
            (2 * usize::from(u16::MAX) + 1, u16::MAX),
            (usize::MAX, u16::MAX),
        ] {
            assert_eq!(hash_chain_position(position), expected);
        }
    }

    #[test]
    fn bulk_insert_matches_ordered_single_insertions() -> Result<(), crate::ReturnCode> {
        use crate::deflate::{set_dictionary, DeflateConfig, DeflateStream};
        // Include long collisions, alternating patterns and repeated runs.
        let dictionary: Vec<_> = (0..4096u32)
            .map(|index| {
                if index % 127 < 64 {
                    b'a'
                } else {
                    (index ^ (index >> 3)).to_le_bytes()[0]
                }
            })
            .collect();
        for level in [6, 9] {
            let config = DeflateConfig::new(level);
            let mut bulk = DeflateStream::new(config)?;
            let mut scalar = DeflateStream::new(config)?;
            for stream in [&mut bulk, &mut scalar] {
                assert_eq!(set_dictionary(stream, &dictionary), crate::ReturnCode::Ok);
            }
            // Compare individual and bulk updates over the same populated
            // table, including reinsertion of existing positions.
            for (start, count) in [
                (0, 0),
                (0, 1),
                (1, 7),
                (63, 128),
                (63, 128),
                (511, 1000),
                (4032, 60),
            ] {
                bulk.state.insert_string(start, count);
                for position in start..start + count {
                    scalar.state.quick_insert_string(position);
                }
                assert_eq!(bulk.state.head.as_slice(), scalar.state.head.as_slice());
                assert_eq!(bulk.state.prev.as_slice(), scalar.state.prev.as_slice());
                assert_eq!(bulk.state.ins_h, scalar.state.ins_h);
            }
        }
        Ok(())
    }

    #[test]
    fn deep_compression_presets_select_their_hash_history() -> Result<(), crate::ReturnCode> {
        use crate::deflate::{end, DeflateConfig, DeflateStream};

        for (level, rolling, max_chain, good_match) in [(8, false, 76, 18), (9, true, 4_096, 32)] {
            let mut stream = DeflateStream::new(DeflateConfig::new(level))?;
            assert_eq!(
                matches!(stream.state.hash_calc_variant, HashCalcVariant::Roll),
                rolling,
                "level {level} selected an incompatible hash representation",
            );
            assert_eq!(stream.state.max_chain_length, max_chain);
            assert_eq!(stream.state.good_match, good_match);
            assert_eq!(i32::from(stream.state.level), level);
            assert!(end(&mut stream).is_ok());
        }
        Ok(())
    }

    #[test]
    fn standard_hash_chain_rebinds_low_word_after_16_bit_rollover() -> Result<(), crate::ReturnCode>
    {
        use crate::deflate::{DeflateConfig, DeflateStream};

        let mut stream = DeflateStream::new(DeflateConfig::new(6))?;
        let state = &mut *stream.state;
        let val = 0x0403_0201u32;
        let hash = usize::try_from(StandardHashCalc::update_hash(0, val))
            .map_err(|_| crate::ReturnCode::StreamError)?;
        let mask = state.w_mask();
        let high = usize::from(u16::MAX);
        assert_eq!(StandardHashCalc::quick_insert_value(state, high, val), 0);
        assert_eq!(state.head.as_slice()[hash], u16::MAX);
        assert_eq!(
            StandardHashCalc::quick_insert_value(state, high + 1, val),
            u16::MAX
        );
        assert_eq!(state.head.as_slice()[hash], 0);
        assert_eq!(state.prev.as_slice()[(high + 1) & mask], u16::MAX);
        assert_eq!(
            StandardHashCalc::quick_insert_value(state, high + 2, val),
            0
        );
        assert_eq!(state.head.as_slice()[hash], 1);
        assert_eq!(state.prev.as_slice()[(high + 2) & mask], 0);
        Ok(())
    }

    #[test]
    fn roll_hash_calc() {
        assert_eq!(RollHashCalc::hash_calc(2565, 93), 82173);
        assert_eq!(RollHashCalc::hash_calc(16637, 10), 532_394);
        assert_eq!(RollHashCalc::hash_calc(8106, 100), 259_364);
        assert_eq!(RollHashCalc::hash_calc(29988, 101), 959_717);
        assert_eq!(RollHashCalc::hash_calc(9445, 98), 302_274);
        assert_eq!(RollHashCalc::hash_calc(7362, 117), 235_573);
        assert_eq!(RollHashCalc::hash_calc(6197, 103), 198_343);
        assert_eq!(RollHashCalc::hash_calc(1735, 32), 55488);
        assert_eq!(RollHashCalc::hash_calc(22720, 61), 727_101);
        assert_eq!(RollHashCalc::hash_calc(6205, 32), 198_528);
        assert_eq!(RollHashCalc::hash_calc(3826, 117), 122_421);
        assert_eq!(RollHashCalc::hash_calc(24117, 101), 771_781);
    }

    #[test]
    fn standard_hash_calc() {
        assert_eq!(StandardHashCalc::hash_calc(0, 807_411_760), 65468);
        assert_eq!(StandardHashCalc::hash_calc(0, 540_024_864), 42837);
        assert_eq!(StandardHashCalc::hash_calc(0, 538_980_384), 33760);
        assert_eq!(StandardHashCalc::hash_calc(0, 775_430_176), 8925);
        assert_eq!(StandardHashCalc::hash_calc(0, 941_629_472), 42053);
    }
}
