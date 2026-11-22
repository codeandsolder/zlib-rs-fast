//! Polynomial folding over 128-bit CRC states. Only carry-less multiplication
//! needs an ISA-specific operation; byte loads, permutations and XOR use
//! portable SIMD. LLVM 23 lowers the native widening_carryless_mul API to
//! individual XMM products with scalar register transfers, even for independent
//! products, so it cannot yet replace this small multiplication boundary.

use core::{
    arch::x86_64::{_mm256_clmulepi64_epi128, _mm512_clmulepi64_epi128, _mm_clmulepi64_si128},
    simd::{simd_swizzle, Simd, ToBytes},
};

type Polynomial = Simd<u8, 16>;
type Polynomial256 = Simd<u8, 32>;
type Polynomial512 = Simd<u8, 64>;

const ZERO: Polynomial = Polynomial::splat(0);
// Folding factors for two, four and sixteen consecutive 128-bit chunks,
// respectively, in the reflected IEEE CRC-32 polynomial representation.
const FOLD_2: Polynomial = polynomial(0xccaa_009e, 0x1_7519_97d0);
const FOLD_4: Polynomial = polynomial(0x1_c6e4_1596, 0x1_5444_2bd4);
const FOLD_16: Polynomial = polynomial(0x1_322d_1430, 0x1_1542_778a);
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
#[target_feature(enable = "pclmulqdq")]
fn multiply<const SELECT: i32>(left: Polynomial, right: Polynomial) -> Polynomial {
    _mm_clmulepi64_si128::<SELECT>(left.into(), right.into()).into()
}

#[inline]
#[target_feature(enable = "pclmulqdq")]
fn fold128(value: Polynomial, factors: Polynomial) -> Polynomial {
    multiply::<0x01>(value, factors) ^ multiply::<0x10>(value, factors)
}

#[inline]
#[target_feature(enable = "vpclmulqdq,avx2")]
fn fold256(value: Polynomial256, factors: Polynomial256) -> Polynomial256 {
    let high_low: Polynomial256 =
        _mm256_clmulepi64_epi128::<0x01>(value.into(), factors.into()).into();
    let low_high: Polynomial256 =
        _mm256_clmulepi64_epi128::<0x10>(value.into(), factors.into()).into();
    high_low ^ low_high
}

#[inline]
#[target_feature(enable = "vpclmulqdq,avx512f")]
fn fold512(value: Polynomial512, factors: Polynomial512) -> Polynomial512 {
    let high_low: Polynomial512 =
        _mm512_clmulepi64_epi128::<0x01>(value.into(), factors.into()).into();
    let low_high: Polynomial512 =
        _mm512_clmulepi64_epi128::<0x10>(value.into(), factors.into()).into();
    high_low ^ low_high
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Width {
    Narrow,
    Wide256,
    Wide512,
}

impl Width {
    fn supported(self) -> bool {
        #[cfg(feature = "std")]
        {
            match self {
                Self::Narrow => crate::cpu_features::is_enabled_pclmulqdq(),
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
            _ => self.blocks128(blocks),
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

    // Two independent 256-bit accumulators preserve the four-word spacing.
    #[target_feature(enable = "vpclmulqdq,avx2")]
    fn blocks256(&mut self, blocks: &[[u8; 64]]) {
        let [a, b, c, d] = self.fold;
        let mut first = pair(a, b);
        let mut second = pair(c, d);
        let factors = pair(FOLD_4, FOLD_4);
        for block in blocks {
            first = fold256(first, factors) ^ Polynomial256::from_slice(block);
            second = fold256(second, factors) ^ Polynomial256::from_slice(&block[32..]);
        }
        self.fold = [
            first.extract::<0, 16>(),
            first.extract::<16, 16>(),
            second.extract::<0, 16>(),
            second.extract::<16, 16>(),
        ];
    }

    // Widen to sixteen independent 128-bit lanes for sustained throughput,
    // then reduce to the common four-word persistent representation.
    #[target_feature(enable = "vpclmulqdq,avx512f")]
    fn blocks512(&mut self, blocks: &[[u8; 64]]) {
        let [a, b, c, d] = self.fold;
        let mut state = quad(a, b, c, d);
        let factors4 = quad(FOLD_4, FOLD_4, FOLD_4, FOLD_4);
        let factors16 = quad(FOLD_16, FOLD_16, FOLD_16, FOLD_16);
        let (groups, tail) = blocks.as_chunks::<4>();
        if let Some((first, remaining)) = groups.split_first() {
            let [w, x, y, z] = first.map(Polynomial512::from_array);
            let mut a = fold512(state, factors4) ^ w;
            let mut b = x;
            let mut c = y;
            let mut d = z;
            for block in remaining {
                let [w, x, y, z] = block.map(Polynomial512::from_array);
                a = fold512(a, factors16) ^ w;
                b = fold512(b, factors16) ^ x;
                c = fold512(c, factors16) ^ y;
                d = fold512(d, factors16) ^ z;
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
        let b = b ^ fold128(a, FOLD_2);
        let c = c ^ fold128(b, FOLD_2);
        let d = d ^ fold128(c, FOLD_2);

        let factors = polynomial(0xccaa_009e, 0x1_63cd_6124);
        let upper = simd_swizzle!(
            d,
            ZERO,
            [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        );
        let reduced = multiply::<0>(d, factors) ^ upper;
        let shifted = simd_swizzle!(
            reduced,
            ZERO,
            [16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        );
        let reduced = (reduced ^ multiply::<0x10>(shifted, factors))
            & polynomial(0xffff_ffff_0000_0000, u64::MAX);

        let barrett = polynomial(0x1_f701_1640, 0x1_db71_0640);
        let quotient = (multiply::<0>(reduced, barrett) ^ reduced) & polynomial(u64::MAX, 0);
        let result = reduced ^ quotient ^ multiply::<0x10>(quotient, barrett);
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
        for width in [Width::Narrow, Width::Wide256, Width::Wide512] {
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
        for width in [Width::Narrow, Width::Wide256, Width::Wide512] {
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
