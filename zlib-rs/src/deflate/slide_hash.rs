use core::simd::{num::SimdUint, Simd};

const LANES: usize = 16;
type HashPositions = Simd<u16, LANES>;

pub fn slide_hash(state: &mut crate::deflate::State) {
    let window_size = state.w_size as u16;
    slide_hash_chain(state.head.as_mut_slice(), window_size);
    slide_hash_chain(state.prev.as_mut_slice(), window_size);
}

fn slide_hash_chain(table: &mut [u16], window_size: u16) {
    let offset = HashPositions::splat(window_size);
    let mut chunks = table.chunks_exact_mut(LANES);

    for chunk in &mut chunks {
        HashPositions::from_slice(chunk)
            .saturating_sub(offset)
            .copy_to_slice(chunk);
    }

    for position in chunks.into_remainder() {
        *position = position.saturating_sub(window_size);
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
        slide_hash_chain(&mut input, WINDOW_SIZE);

        assert_eq!(input, expected);
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
            slide_hash_chain(&mut input, window_size);
            input == expected
        }
    }
}
