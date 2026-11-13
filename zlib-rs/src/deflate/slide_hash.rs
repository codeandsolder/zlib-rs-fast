use core::simd::{num::SimdUint, Simd};

use super::Pos;

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
    slide_hash_chain(state.head.as_mut_slice(), state.w_size);
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
