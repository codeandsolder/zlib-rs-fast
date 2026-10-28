//! The crc32 checksum algorithm.

use crate::CRC32_INITIAL_VALUE;

#[cfg(target_arch = "aarch64")]
pub(crate) mod acle;
mod braid;
mod combine;
#[cfg(target_arch = "loongarch64")]
mod loongarch;
#[cfg(target_arch = "x86_64")]
mod pclmulqdq;
#[cfg(target_arch = "x86_64")]
#[cfg(feature = "vpclmulqdq")]
mod vpclmulqdq;

pub use combine::{crc32_combine, crc32_combine_gen, crc32_combine_op};

pub fn crc32(start: u32, buf: &[u8]) -> u32 {
    /* For lens < 64, crc32_braid method is faster. The CRC32 instruction for
     * these short lengths might also prove to be effective */
    if buf.len() < 64 {
        return crc32_braid(start, buf);
    }

    let mut crc_state = Crc32Fold::new_with_initial(start);
    crc_state.fold(buf);
    crc_state.finish()
}

fn crc32_braid(start: u32, buf: &[u8]) -> u32 {
    braid::crc32_braid::<5>(start, buf)
}

pub fn get_crc_table() -> &'static [u32; 256] {
    braid::get_crc_table()
}

#[derive(Debug, Clone, Copy)]
enum Crc32Backend {
    #[cfg(target_arch = "x86_64")]
    Pclmul {
        accumulator: pclmulqdq::Accumulator,
        pending_initial: Option<u32>,
    },
    #[cfg(target_arch = "aarch64")]
    ArmCrc(u32),
    #[cfg(target_arch = "loongarch64")]
    LoongArchCrc(u32),
    Scalar(u32),
}
#[cfg(feature = "__internal-test")]
#[derive(Debug, Clone, Copy)]
pub struct Crc32Fold {
    backend: Crc32Backend,
}
#[cfg(not(feature = "__internal-test"))]
#[derive(Debug, Clone, Copy)]
pub(crate) struct Crc32Fold {
    backend: Crc32Backend,
}
impl Default for Crc32Fold {
    fn default() -> Self {
        Self::new()
    }
}
impl Crc32Fold {
    pub fn new() -> Self {
        Self::new_with_initial(CRC32_INITIAL_VALUE)
    }
    pub fn new_with_initial(initial: u32) -> Self {
        #[cfg(target_arch = "x86_64")]
        let backend = if crate::cpu_features::is_enabled_pclmulqdq() {
            Crc32Backend::Pclmul {
                accumulator: pclmulqdq::Accumulator::new(),
                pending_initial: Some(initial),
            }
        } else {
            Crc32Backend::Scalar(initial)
        };
        #[cfg(target_arch = "aarch64")]
        let backend = if crate::cpu_features::is_enabled_crc() {
            Crc32Backend::ArmCrc(initial)
        } else {
            Crc32Backend::Scalar(initial)
        };
        #[cfg(target_arch = "loongarch64")]
        let backend = Crc32Backend::LoongArchCrc(initial);
        #[cfg(not(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "loongarch64"
        )))]
        let backend = Crc32Backend::Scalar(initial);
        Self { backend }
    }
    pub fn fold(&mut self, src: &[u8]) {
        #[cfg(target_arch = "x86_64")]
        match &mut self.backend {
            Crc32Backend::Pclmul {
                accumulator,
                pending_initial,
            } => {
                let initial = pending_initial.take().unwrap_or(CRC32_INITIAL_VALUE);
                if initial != CRC32_INITIAL_VALUE && src.len() < 64 {
                    self.backend = Crc32Backend::Scalar(braid::crc32_braid::<5>(initial, src));
                } else {
                    // SAFETY: selected only after runtime PCLMULQDQ/SSE4.1 detection.
                    unsafe { accumulator.fold(src, initial) };
                }
                return;
            }
            Crc32Backend::Scalar(value) => {
                *value = braid::crc32_braid::<5>(*value, src);
                return;
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        match &mut self.backend {
            #[cfg(target_arch = "aarch64")]
            Crc32Backend::ArmCrc(value) => {
                // SAFETY: this backend is selected only after runtime CRC feature detection.
                *value = unsafe { self::acle::crc32_acle_aarch64(*value, src) };
            }
            #[cfg(target_arch = "loongarch64")]
            Crc32Backend::LoongArchCrc(value) => {
                *value = self::loongarch::crc32_loongarch64(*value, src);
            }
            Crc32Backend::Scalar(value) => *value = braid::crc32_braid::<5>(*value, src),
        }
    }
    pub fn fold_copy(&mut self, dst: &mut [u8], src: &[u8]) {
        assert!(dst.len() >= src.len(), "destination is shorter than source");
        #[cfg(target_arch = "x86_64")]
        if let Crc32Backend::Pclmul {
            accumulator,
            pending_initial,
        } = &mut self.backend
        {
            let initial = pending_initial.take().unwrap_or(CRC32_INITIAL_VALUE);
            if initial == CRC32_INITIAL_VALUE {
                // SAFETY: selected only after runtime PCLMULQDQ/SSE4.1 detection.
                unsafe { accumulator.fold_copy(&mut dst[..src.len()], src) };
                return;
            }
            if src.len() >= 64 {
                // The fused copy path cannot inject a non-default initial CRC.
                unsafe { accumulator.fold(src, initial) };
                dst[..src.len()].copy_from_slice(src);
                return;
            }
            self.backend = Crc32Backend::Scalar(braid::crc32_braid::<5>(initial, src));
            dst[..src.len()].copy_from_slice(src);
            return;
        }
        self.fold(src);
        dst[..src.len()].copy_from_slice(src);
    }
    pub fn finish(self) -> u32 {
        #[cfg(target_arch = "x86_64")]
        return match self.backend {
            Crc32Backend::Pclmul {
                accumulator,
                pending_initial,
            } => match pending_initial {
                Some(initial) => initial,
                None => {
                    // SAFETY: selected only after runtime PCLMULQDQ/SSE4.1 detection.
                    unsafe { accumulator.finish() }
                }
            },
            Crc32Backend::Scalar(value) => value,
        };
        #[cfg(not(target_arch = "x86_64"))]
        match self.backend {
            #[cfg(target_arch = "aarch64")]
            Crc32Backend::ArmCrc(value) => value,
            #[cfg(target_arch = "loongarch64")]
            Crc32Backend::LoongArchCrc(value) => value,
            Crc32Backend::Scalar(value) => value,
        }
    }
}

#[cfg(test)]
mod test {
    use braid::crc32_braid;

    use super::*;

    const INPUT: [u8; 1024] = {
        let mut array = [0; 1024];
        let mut i = 0;
        while i < array.len() {
            array[i] = i as u8;
            i += 1;
        }

        array
    };

    #[test]
    fn test_crc32_fold() {
        // input large enough to trigger the SIMD
        let mut h = crc32fast::Hasher::new_with_initial(CRC32_INITIAL_VALUE);
        h.update(&INPUT);
        assert_eq!(crc32(CRC32_INITIAL_VALUE, &INPUT), h.finalize());
    }

    #[test]
    fn test_crc32_fold_align() {
        // SIMD algorithm is sensitive to alignment;
        for i in 0..16 {
            for start in [CRC32_INITIAL_VALUE, 42] {
                let mut h = crc32fast::Hasher::new_with_initial(start);
                h.update(&INPUT[i..]);
                assert_eq!(
                    crc32(start, &INPUT[i..]),
                    h.finalize(),
                    "offset = {i}, start = {start}"
                );
            }
        }
    }

    quickcheck::quickcheck! {
        fn crc_fold_is_crc32fast(v: Vec<u8>, start: u32) -> bool {
            let mut h = crc32fast::Hasher::new_with_initial(start);
            h.update(&v);

            let a = crc32(start, &v) ;
            let b = h.finalize();

            a == b
        }
    }

    #[test]
    fn chunked() {
        const INPUT: &[&[u8]] = &[
            &[116],
            &[111, 107, 105, 111, 44, 32, 97, 115],
            &[121, 110, 99, 45, 115, 116, 100, 44],
            &[32, 97, 110, 100, 32, 115, 109, 111],
            &[108, 46, 32, 89, 111, 117, 226, 128],
            &[153, 118, 101, 32, 112, 114, 111, 98],
            &[97, 98, 108, 121, 32, 117, 115, 101],
            &[100, 32, 116, 104, 101, 109, 32, 97],
            &[116, 32, 115, 111, 109, 101, 32, 112],
            &[111, 105, 110, 116, 44, 32, 101, 105],
            &[116, 104, 101, 114, 32, 100, 105, 114],
            &[101, 99, 116, 108, 121, 32, 111, 114],
            &[0],
        ];

        const START: u32 = 2380683574;

        let mut in_chunks = START;
        for chunk in INPUT {
            in_chunks = crc32(in_chunks, chunk);
        }

        let flattened: Vec<_> = INPUT.iter().copied().flatten().copied().collect();
        let flat = crc32(START, &flattened);

        assert_eq!(in_chunks, flat);
    }

    #[test]
    fn nasty_alignment() {
        const START: u32 = 2380683574;

        const FLAT: &[u8] = &[
            116, 111, 107, 105, 111, 44, 32, 97, 115, 121, 110, 99, 45, 115, 116, 100, 44, 32, 97,
            110, 100, 32, 115, 109, 111, 108, 46, 32, 89, 111, 117, 226, 128, 153, 118, 101, 32,
            112, 114, 111, 98, 97, 98, 108, 121, 32, 117, 115, 101, 100, 32, 116, 104, 101, 109,
            32, 97, 116, 32, 115, 111, 109, 101, 32, 112, 111, 105, 110, 116, 44, 32, 101, 105,
            116, 104, 101, 114, 32, 100, 105, 114, 101, 99, 116, 108, 121, 32, 111, 114, 0,
        ];

        let mut i = 0;
        let mut flat = FLAT.to_vec();
        while flat[i..].as_ptr() as usize % 16 != 15 {
            flat.insert(0, 0);
            i += 1;
        }

        let flat = &flat[i..];

        assert_eq!(crc32_braid::<5>(START, flat), crc32(START, flat));
        assert_eq!(crc32(2380683574, flat), 1175758345);
    }

    #[test]
    fn fold_handles_short_custom_initial_without_backend_assumptions() {
        let input = &INPUT[..31];
        let start = 1;
        let mut fold = Crc32Fold::new_with_initial(start);
        fold.fold(input);
        assert_eq!(fold.finish(), crc32_braid::<5>(start, input));
    }

    #[test]
    fn fold_copy_handles_custom_initial_for_short_and_long_inputs() {
        for input in [&INPUT[..31], &INPUT[..257]] {
            let start = 42;
            let mut dst = vec![0xaa; input.len()];
            let mut fold = Crc32Fold::new_with_initial(start);
            fold.fold_copy(&mut dst, input);
            assert_eq!(dst, input);
            assert_eq!(fold.finish(), crc32_braid::<5>(start, input));
        }
    }

    #[test]
    fn fold_is_chunking_independent() {
        for chunk_size in [1, 7, 16, 31, 63, 64, 65, 127, 256] {
            let start = 0x1234_5678;
            let mut fold = Crc32Fold::new_with_initial(start);
            for chunk in INPUT.chunks(chunk_size) {
                fold.fold(chunk);
            }
            assert_eq!(
                fold.finish(),
                crc32_braid::<5>(start, &INPUT),
                "chunk size {chunk_size}"
            );
        }
    }
}
