use core::simd::{num::SimdUint, Simd};

use super::{
    hash_calc::{HashCalcVariant, ROLL_HASH_SIZE},
    Pos,
};

const LANES: usize = 16;
type HashPositions = Simd<u16, LANES>;

/// Slide a stored 16-bit window position by a full-width byte displacement.
///
/// A displacement outside the range of `Pos` necessarily removes every
/// representable position; zero is the exact saturating result, not a
/// substitute narrowed distance.
#[inline]
pub(super) fn slide_position(position: Pos, displacement: usize) -> Pos {
    let Ok(displacement) = Pos::try_from(displacement) else {
        return 0;
    };
    position.saturating_sub(displacement)
}

pub fn slide_hash(state: &mut crate::deflate::State) {
    // Rolling insertion only addresses the lower 15-bit bucket range.
    // Initialization and history-layout changes clear the entire table,
    // so its unused upper half remains zero throughout rolling operation.
    let head = state.head.as_mut_slice();
    let active = match state.hash_calc_variant {
        HashCalcVariant::Standard => head,
        HashCalcVariant::Roll => {
            debug_assert!(head[ROLL_HASH_SIZE..].iter().all(|&pos| pos == 0));
            &mut head[..ROLL_HASH_SIZE]
        }
    };
    slide_hash_chain(active, state.w_size);
    slide_hash_chain(state.prev.as_mut_slice(), state.w_size);
}

fn slide_hash_chain(table: &mut [Pos], displacement: usize) {
    let Ok(displacement) = Pos::try_from(displacement) else {
        // No position can survive a slide larger than `Pos::MAX`.
        table.fill(0);
        return;
    };

    let offset = HashPositions::splat(displacement);
    let (chunks, tail) = table.as_chunks_mut::<LANES>();

    for chunk in chunks {
        HashPositions::from_array(*chunk)
            .saturating_sub(offset)
            .copy_to_slice(chunk);
    }

    for position in tail {
        *position = position.saturating_sub(displacement);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_slide(table: &mut [u16], window_size: u16) {
        for position in table {
            *position = position.saturating_sub(window_size);
        }
    }

    #[test]
    fn handles_saturation_vector_boundaries_and_tail() {
        const WINDOW_SIZE: u16 = 32_768;
        let pattern = [
            0,
            1,
            WINDOW_SIZE - 1,
            WINDOW_SIZE,
            WINDOW_SIZE + 1,
            u16::MAX,
        ];
        let mut input: Vec<_> = pattern.into_iter().cycle().take(LANES * 2 + 5).collect();
        let mut expected = input.clone();

        scalar_slide(&mut expected, WINDOW_SIZE);
        slide_hash_chain(&mut input, usize::from(WINDOW_SIZE));

        assert_eq!(input, expected);
    }

    #[test]
    fn full_width_displacements_preserve_saturating_position_semantics() {
        let positions = [0, 1, 255, 256, 32_767, 32_768, u16::MAX];
        for displacement in [
            0usize,
            1,
            256,
            32_767,
            32_768,
            usize::from(u16::MAX),
            usize::from(u16::MAX) + 1,
            usize::MAX,
        ] {
            let expected =
                positions.map(|position| usize::from(position).saturating_sub(displacement));
            let mut actual = positions;
            slide_hash_chain(&mut actual, displacement);
            assert_eq!(
                actual.map(usize::from),
                expected,
                "displacement={displacement}"
            );
            for (position, expected) in positions.into_iter().zip(expected) {
                assert_eq!(
                    usize::from(slide_position(position, displacement)),
                    expected,
                    "displacement={displacement}, position={position}"
                );
            }
        }
    }

    #[test]
    fn rolling_layout_switches_preserve_inactive_bucket_invariant() -> Result<(), crate::ReturnCode>
    {
        use crate::deflate::{
            params, reset, set_dictionary, DeflateConfig, DeflateStream, Strategy,
        };
        use crate::ReturnCode;
        for window_bits in 9..=crate::MAX_WBITS {
            let mut random = 0x38ef_120f_8c29_7d11u64;
            let dictionary: Vec<u8> = (0..1usize << window_bits)
                .map(|_| {
                    random ^= random << 13;
                    random ^= random >> 7;
                    random ^= random << 17;
                    random.to_le_bytes()[0]
                })
                .collect();
            let mut stream = DeflateStream::new(DeflateConfig {
                window_bits,
                ..DeflateConfig::new(6)
            })?;
            for level in [6, 9, 8, 9, 1, 9] {
                assert_eq!(reset(&mut stream), ReturnCode::Ok);
                assert_eq!(set_dictionary(&mut stream, &dictionary), ReturnCode::Ok);
                assert_eq!(
                    params(&mut stream, level, Strategy::Default),
                    ReturnCode::Ok
                );
                let state = &mut *stream.state;
                if level == 9 {
                    assert!(state.head.as_slice()[ROLL_HASH_SIZE..]
                        .iter()
                        .all(|&pos| pos == 0));
                }
                let displacement = state.w_size;
                let mut expected_head = state.head.as_slice().to_vec();
                let mut expected_prev = state.prev.as_slice().to_vec();
                for position in expected_head.iter_mut().chain(&mut expected_prev) {
                    *position = slide_position(*position, displacement);
                }
                slide_hash(state);
                assert_eq!(state.head.as_slice(), expected_head);
                assert_eq!(state.prev.as_slice(), expected_prev);
            }
        }
        Ok(())
    }

    #[test]
    fn empty_slice_is_a_noop() {
        slide_hash_chain(&mut [], 32_768);
    }

    quickcheck::quickcheck! {
        fn matches_scalar_reference(input: Vec<u16>, window_size: u16) -> bool {
            let mut input = input;
            let mut expected = input.clone();
            scalar_slide(&mut expected, window_size);
            slide_hash_chain(&mut input, usize::from(window_size));
            input == expected
        }
    }
}
