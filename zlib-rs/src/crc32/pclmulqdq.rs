//! Polynomial folding over 128-bit CRC states, using native polynomial
//! arithmetic and portable SIMD. Target features select the available
//! hardware width; all loads, arithmetic and permutations are safe Rust.

use core::simd::{simd_swizzle, Simd, ToBytes};

type Polynomial = Simd<u8, 16>;
type Polynomial256 = Simd<u8, 32>;
type Polynomial512 = Simd<u8, 64>;

const ZERO: Polynomial = Polynomial::splat(0);
// A reflected polynomial power has an extra low zero bit in the
// carry-less multiplication representation.
const fn reflected_power(bits: u64) -> u64 {
    (super::combine::x2nmodp(bits, 0) as u64) << 1
}

// Advance one 128-bit word by the given number of words.
const fn folding_factors(words: u64) -> Polynomial {
    polynomial(
        reflected_power(128 * words - 32),
        reflected_power(128 * words + 32),
    )
}

const FOLD_1: Polynomial = folding_factors(1);
const FOLD_4: Polynomial = folding_factors(4);
const FOLD_8: Polynomial = folding_factors(8);
const FOLD_16: Polynomial = folding_factors(16);
const REDUCTION_FACTORS: Polynomial = polynomial(reflected_power(96), reflected_power(64));
// Reflected Barrett reciprocal and the reflected generator polynomial.
const BARRETT: Polynomial = polynomial(0x1_f701_1640, reflected_power(32));
// Paired Zen 5 measurements favor XMM below 1 KiB and wide folding from 1 KiB.
const WIDE_MIN_BYTES: usize = 1024;

const fn polynomial(low: u64, high: u64) -> Polynomial {
    let mut bytes = [0; 16];
    let (low_bytes, high_bytes) = bytes.split_at_mut(8);
    low_bytes.copy_from_slice(&low.to_le_bytes());
    high_bytes.copy_from_slice(&high.to_le_bytes());
    Polynomial::from_array(bytes)
}

const fn consecutive_indices<const N: usize>() -> [usize; N] {
    let mut indices = [0; N];
    let mut index = 0;
    while index < N {
        indices[index] = index;
        index += 1;
    }
    indices
}

#[inline]
fn pair(first: Polynomial, second: Polynomial) -> Polynomial256 {
    simd_swizzle!(first, second, consecutive_indices::<32>())
}

#[inline]
fn quad(
    first: Polynomial,
    second: Polynomial,
    third: Polynomial,
    fourth: Polynomial,
) -> Polynomial512 {
    simd_swizzle!(
        pair(first, second),
        pair(third, fourth),
        consecutive_indices::<64>()
    )
}

#[inline]
fn multiply<const LEFT: usize, const RIGHT: usize>(
    left: Polynomial,
    right: Polynomial,
) -> Polynomial {
    let left = Simd::<u64, 2>::from_le_bytes(left).to_array();
    let right = Simd::<u64, 2>::from_le_bytes(right).to_array();
    Polynomial::from_array(
        left[LEFT]
            .widening_carryless_mul(right[RIGHT])
            .to_le_bytes(),
    )
}

// Express each half in legal u64 SIMD lanes. Widening directly to u128
// lanes prevents LLVM from combining the independent products. Keeping
// both halves separate allows the backend to select full-width CLMUL.
#[inline]
fn products<const N: usize>(
    left: Simd<u64, N>,
    right: Simd<u64, N>,
) -> (Simd<u64, N>, Simd<u64, N>) {
    let left = left.to_array();
    let right = right.to_array();
    let low = Simd::from_array(core::array::from_fn(|i| left[i].carryless_mul(right[i])));
    let high = Simd::from_array(core::array::from_fn(|i| {
        (left[i].widening_carryless_mul(right[i]) >> 64) as u64
    }));
    (low, high)
}

#[inline]
fn fold128(value: Polynomial, factors: Polynomial) -> Polynomial {
    let value = Simd::<u64, 2>::from_le_bytes(value);
    let factors = simd_swizzle!(Simd::<u64, 2>::from_le_bytes(factors), [1, 0]);
    let (low, high) = products(value, factors);
    (simd_swizzle!(low, high, [0, 2]) ^ simd_swizzle!(low, high, [1, 3])).to_le_bytes()
}

#[inline]
fn fold256(value: Polynomial256, factors: Polynomial256) -> Polynomial256 {
    let value = Simd::<u64, 4>::from_le_bytes(value);
    let factors = simd_swizzle!(Simd::<u64, 4>::from_le_bytes(factors), [1, 0, 3, 2]);
    let (lo, hi) = products(value, factors);
    let (sl, sh) = products(
        simd_swizzle!(value, [1, 0, 3, 2]),
        simd_swizzle!(factors, [1, 0, 3, 2]),
    );
    (simd_swizzle!(lo, hi, [0, 4, 2, 6]) ^ simd_swizzle!(sl, sh, [0, 4, 2, 6])).to_le_bytes()
}

#[inline]
fn fold512(value: Polynomial512, factors: Polynomial512) -> Polynomial512 {
    let value = Simd::<u64, 8>::from_le_bytes(value);
    let factors = simd_swizzle!(
        Simd::<u64, 8>::from_le_bytes(factors),
        [1, 0, 3, 2, 5, 4, 7, 6]
    );
    let (lo, hi) = products(value, factors);
    let (sl, sh) = products(
        simd_swizzle!(value, [1, 0, 3, 2, 5, 4, 7, 6]),
        simd_swizzle!(factors, [1, 0, 3, 2, 5, 4, 7, 6]),
    );
    (simd_swizzle!(lo, hi, [0, 8, 2, 10, 4, 12, 6, 14])
        ^ simd_swizzle!(sl, sh, [0, 8, 2, 10, 4, 12, 6, 14]))
    .to_le_bytes()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Width {
    Narrow,
    NarrowAvx,
    Wide256,
    Wide512,
}

impl Width {
    fn supported(self) -> bool {
        #[cfg(feature = "std")]
        {
            match self {
                Self::Narrow => crate::cpu_features::is_enabled_pclmulqdq(),
                Self::NarrowAvx => std::is_x86_feature_detected!("avx"),
                Self::Wide256 => {
                    std::is_x86_feature_detected!("vpclmulqdq")
                        && std::is_x86_feature_detected!("avx2")
                }
                Self::Wide512 => {
                    std::is_x86_feature_detected!("vpclmulqdq")
                        && std::is_x86_feature_detected!("avx512f")
                }
            }
        }
        #[cfg(not(feature = "std"))]
        {
            match self {
                Self::Narrow => cfg!(target_feature = "pclmulqdq"),
                Self::NarrowAvx => cfg!(target_feature = "avx"),
                Self::Wide256 => cfg!(all(target_feature = "vpclmulqdq", target_feature = "avx2")),
                Self::Wide512 => cfg!(all(
                    target_feature = "vpclmulqdq",
                    target_feature = "avx512f"
                )),
            }
        }
    }

    fn detect() -> Self {
        if Self::Wide512.supported() {
            Self::Wide512
        } else if Self::Wide256.supported() {
            Self::Wide256
        } else if Self::NarrowAvx.supported() {
            Self::NarrowAvx
        } else {
            Self::Narrow
        }
    }
}

/// Four successive 128-bit folding states. All backends share this persistent
/// representation, allowing free changes of processing width at chunk boundaries.
#[derive(Debug, Clone, Copy)]
pub struct Accumulator {
    fold: [Polynomial; 4],
    width: Width,
}

impl Accumulator {
    pub fn new() -> Self {
        Self {
            fold: [polynomial(0x9db4_2487, 0), ZERO, ZERO, ZERO],
            width: Width::detect(),
        }
    }

    /// Fold one input fragment. A non-default initial CRC is supplied only with
    /// the first fragment of at least 64 bytes; the outer CRC API handles short
    /// seeded fragments through its scalar backend.
    #[target_feature(enable = "pclmulqdq")]
    pub fn fold(&mut self, mut input: &[u8], start: u32) {
        debug_assert!(start == 0 || input.len() >= 64);
        if start != 0 {
            let Some((first, remaining)) = input.split_first_chunk::<16>() else {
                unreachable!("non-default CRC seed requires at least 64 input bytes");
            };
            self.fold_word(Polynomial::from_array(*first) ^ polynomial(u64::from(start), 0));
            input = remaining;
        }

        let (blocks, remainder) = input.as_chunks::<64>();
        match self.width {
            Width::Wide512 if input.len() >= WIDE_MIN_BYTES => {
                // SAFETY: width is chosen once by runtime (or compile-time)
                // feature detection for VPCLMULQDQ and AVX-512F.
                unsafe { self.blocks512(blocks) };
            }
            Width::Wide256 if input.len() >= WIDE_MIN_BYTES => {
                // SAFETY: width is chosen once by runtime (or compile-time)
                // feature detection for VPCLMULQDQ and AVX2.
                unsafe { self.blocks256(blocks) };
            }
            Width::Narrow => self.blocks128(blocks),
            Width::NarrowAvx | Width::Wide256 | Width::Wide512 => {
                // SAFETY: every selected width here supports AVX.
                unsafe { self.blocks128_avx(blocks) };
            }
        }

        let (words, tail) = remainder.as_chunks::<16>();
        for word in words {
            self.fold_word(Polynomial::from_array(*word));
        }
        if !tail.is_empty() {
            self.partial(tail);
        }
    }

    #[inline]
    #[target_feature(enable = "pclmulqdq")]
    fn fold_word(&mut self, word: Polynomial) {
        let [a, b, c, d] = self.fold;
        self.fold = [b, c, d, fold128(a, FOLD_4) ^ word];
    }

    /// Shift the entire four-word polynomial reservoir by a nonempty partial
    /// word (1..15 bytes). Ordinary integer shifts express the cross-word
    /// operation without ISA-specific byte permutations or speculative loads.
    #[target_feature(enable = "pclmulqdq")]
    fn partial(&mut self, tail: &[u8]) {
        debug_assert!((1..16).contains(&tail.len()));
        let mut padded = [0; 16];
        padded[..tail.len()].copy_from_slice(tail);
        let right_shift = tail.len() * 8;
        let left_shift = 128 - right_shift;
        let [a, b, c, d] = self.fold.map(|value| u128::from_le_bytes(value.to_array()));
        let input = u128::from_le_bytes(padded);
        self.fold = [
            Polynomial::from_array(((a >> right_shift) | (b << left_shift)).to_le_bytes()),
            Polynomial::from_array(((b >> right_shift) | (c << left_shift)).to_le_bytes()),
            Polynomial::from_array(((c >> right_shift) | (d << left_shift)).to_le_bytes()),
            Polynomial::from_array(((d >> right_shift) | (input << left_shift)).to_le_bytes())
                ^ fold128(
                    Polynomial::from_array((a << left_shift).to_le_bytes()),
                    FOLD_4,
                ),
        ];
    }

    #[inline]
    #[target_feature(enable = "pclmulqdq")]
    fn blocks128(&mut self, blocks: &[[u8; 64]]) {
        let [mut a, mut b, mut c, mut d] = self.fold;
        for block in blocks {
            a = fold128(a, FOLD_4) ^ Polynomial::from_slice(block);
            b = fold128(b, FOLD_4) ^ Polynomial::from_slice(&block[16..]);
            c = fold128(c, FOLD_4) ^ Polynomial::from_slice(&block[32..]);
            d = fold128(d, FOLD_4) ^ Polynomial::from_slice(&block[48..]);
        }
        self.fold = [a, b, c, d];
    }

    // Reuse the narrow loop with nondestructive AVX encodings when available.
    #[target_feature(enable = "pclmulqdq,avx")]
    fn blocks128_avx(&mut self, blocks: &[[u8; 64]]) {
        self.blocks128(blocks);
    }

    // Four independent 256-bit accumulators hide the multiplication latency.
    #[target_feature(enable = "vpclmulqdq,avx2")]
    fn blocks256(&mut self, blocks: &[[u8; 64]]) {
        let [a, b, c, d] = self.fold;
        let mut a = pair(a, b);
        let mut b = pair(c, d);
        let k4 = pair(FOLD_4, FOLD_4);
        let k8 = pair(FOLD_8, FOLD_8);
        let (groups, tail) = blocks.as_chunks::<2>();
        if let Some(([first, second], remaining)) = groups.split_first() {
            a = fold256(a, k4) ^ Polynomial256::from_slice(first);
            b = fold256(b, k4) ^ Polynomial256::from_slice(&first[32..]);
            let mut c = Polynomial256::from_slice(second);
            let mut d = Polynomial256::from_slice(&second[32..]);
            for [first, second] in remaining {
                a = fold256(a, k8) ^ Polynomial256::from_slice(first);
                b = fold256(b, k8) ^ Polynomial256::from_slice(&first[32..]);
                c = fold256(c, k8) ^ Polynomial256::from_slice(second);
                d = fold256(d, k8) ^ Polynomial256::from_slice(&second[32..]);
            }
            a = fold256(a, k4) ^ c;
            b = fold256(b, k4) ^ d;
        }
        for block in tail {
            a = fold256(a, k4) ^ Polynomial256::from_slice(block);
            b = fold256(b, k4) ^ Polynomial256::from_slice(&block[32..]);
        }
        self.fold = [
            a.extract::<0, 16>(),
            a.extract::<16, 16>(),
            b.extract::<0, 16>(),
            b.extract::<16, 16>(),
        ];
    }

    // Widen to sixteen independent 128-bit lanes for sustained throughput,
    // then reduce to the common four-word persistent representation.
    #[target_feature(enable = "vpclmulqdq,avx512f")]
    fn blocks512(&mut self, blocks: &[[u8; 64]]) {
        let [first, second, third, fourth] = self.fold;
        let mut state = quad(first, second, third, fourth);
        let factors4 = quad(FOLD_4, FOLD_4, FOLD_4, FOLD_4);
        let factors16 = quad(FOLD_16, FOLD_16, FOLD_16, FOLD_16);
        let (groups, tail) = blocks.as_chunks::<4>();
        if let Some((first, remaining)) = groups.split_first() {
            let [word0, word1, word2, word3] = first.map(Polynomial512::from_array);
            let mut a = fold512(state, factors4) ^ word0;
            let mut b = word1;
            let mut c = word2;
            let mut d = word3;
            for block in remaining {
                let [word0, word1, word2, word3] = block.map(Polynomial512::from_array);
                a = fold512(a, factors16) ^ word0;
                b = fold512(b, factors16) ^ word1;
                c = fold512(c, factors16) ^ word2;
                d = fold512(d, factors16) ^ word3;
            }
            b ^= fold512(a, factors4);
            c ^= fold512(b, factors4);
            d ^= fold512(c, factors4);
            state = d;
        }
        for block in tail {
            state = fold512(state, factors4) ^ Polynomial512::from_array(*block);
        }
        self.fold = [
            state.extract::<0, 16>(),
            state.extract::<16, 16>(),
            state.extract::<32, 16>(),
            state.extract::<48, 16>(),
        ];
    }

    /// Reduce the four-word reservoir to the final reflected CRC-32 using
    /// the same polynomial folds and Barrett reduction as the original kernel.
    #[target_feature(enable = "pclmulqdq")]
    pub fn finish(self) -> u32 {
        let [a, b, c, d] = self.fold;
        let b = b ^ fold128(a, FOLD_1);
        let c = c ^ fold128(b, FOLD_1);
        let d = d ^ fold128(c, FOLD_1);

        let factors = REDUCTION_FACTORS;
        let upper = simd_swizzle!(
            d,
            ZERO,
            [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        );
        let reduced = multiply::<0, 0>(d, factors) ^ upper;
        let shifted = simd_swizzle!(
            reduced,
            ZERO,
            [16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        );
        let reduced = (reduced ^ multiply::<0, 1>(shifted, factors))
            & polynomial(0xffff_ffff_0000_0000, u64::MAX);

        let barrett = BARRETT;
        let quotient = (multiply::<0, 0>(reduced, barrett) ^ reduced) & polynomial(u64::MAX, 0);
        let result = reduced ^ quotient ^ multiply::<0, 1>(quotient, barrett);
        !Simd::<u32, 4>::from_le_bytes(result)[2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LENGTHS: [usize; 28] = [
        0, 1, 2, 7, 15, 16, 17, 31, 32, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513,
        1023, 1024, 1025, 1087, 1088, 4095, 4096,
    ];

    #[test]
    fn supported_widths_match_reference_at_boundaries_and_offsets() {
        if !Width::Narrow.supported() {
            return;
        }
        let input: Vec<u8> = (0usize..8192)
            .map(|index| index.to_le_bytes()[0].wrapping_mul(137) ^ (index >> 8).to_le_bytes()[0])
            .collect();
        for width in [
            Width::Narrow,
            Width::NarrowAvx,
            Width::Wide256,
            Width::Wide512,
        ] {
            if !width.supported() {
                continue;
            }
            for offset in 0..32 {
                for length in LENGTHS {
                    for start in [0, 1, 0x1234_5678, u32::MAX] {
                        if start != 0 && length < 64 {
                            continue;
                        }
                        let bytes = &input[offset..offset + length];
                        let mut reference = crc32fast::Hasher::new_with_initial(start);
                        reference.update(bytes);
                        let mut accumulator = Accumulator {
                            width,
                            ..Accumulator::new()
                        };
                        // SAFETY: the baseline and requested wide features were verified.
                        let actual = unsafe {
                            accumulator.fold(bytes, start);
                            accumulator.finish()
                        };
                        assert_eq!(
                            actual,
                            reference.finalize(),
                            "{width:?}, {offset}, {length}, {start}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn supported_widths_preserve_fragmented_crc_state() {
        if !Width::Narrow.supported() {
            return;
        }
        let input: Vec<u8> = (0usize..131_071)
            .map(|index| index.to_le_bytes()[0].wrapping_mul(173) ^ (index >> 8).to_le_bytes()[0])
            .collect();
        let start = 0x1234_5678;
        let mut reference = crc32fast::Hasher::new_with_initial(start);
        reference.update(&input);
        let expected = reference.finalize();
        for width in [
            Width::Narrow,
            Width::NarrowAvx,
            Width::Wide256,
            Width::Wide512,
        ] {
            if !width.supported() {
                continue;
            }
            for chunk_size in [1, 7, 15, 16, 17, 63, 64, 65, 255, 256, 511, 1024, 4096] {
                let mut accumulator = Accumulator {
                    width,
                    ..Accumulator::new()
                };
                // SAFETY: the width and narrow CPU feature were checked above.
                let actual = unsafe {
                    accumulator.fold(&input[..64], start);
                    for chunk in input[64..].chunks(chunk_size) {
                        accumulator.fold(chunk, 0);
                    }
                    accumulator.finish()
                };
                assert_eq!(actual, expected, "{width:?}, chunk={chunk_size}");
            }
        }
    }
}
