//! Public opt-in high-effort API must not rely on the internal feature.
#![cfg(feature = "near-optimal")]

use zlib_rs::{
    compress_bound, compress_slice_near_optimal, decompress_slice, InflateConfig, ReturnCode,
};

#[test]
fn near_optimal_public_round_trip_and_buffer_limits() {
    let mut original = Vec::new();
    for i in 0..5000u32 {
        original.extend_from_slice(format!("{i:09} 0.12345678 0.98765432 l\n").as_bytes());
    }
    for passes in [1, 2, 3] {
        let mut destination = vec![0u8; compress_bound(original.len())];
        let (compressed, code) = compress_slice_near_optimal(&mut destination, &original, passes);
        assert_eq!(code, ReturnCode::Ok);
        assert!(compressed.len() < original.len());
        let mut output = vec![0u8; original.len()];
        let (decoded, code) = decompress_slice(&mut output, compressed, InflateConfig::default());
        assert_eq!(code, ReturnCode::Ok);
        assert_eq!(decoded, original.as_slice());
    }
    let mut small = vec![0u8; 2];
    let (compressed, code) = compress_slice_near_optimal(&mut small, &original, 3);
    assert_eq!(code, ReturnCode::BufError);
    assert!(compressed.is_empty());
}

#[test]
fn near_optimal_public_empty_input_round_trip() {
    let mut destination = vec![0u8; compress_bound(0)];
    let (compressed, code) = compress_slice_near_optimal(&mut destination, &[], 3);
    assert_eq!(code, ReturnCode::Ok);
    let mut output = [0u8; 1];
    let (decoded, code) = decompress_slice(&mut output, compressed, InflateConfig::default());
    assert_eq!(code, ReturnCode::Ok);
    assert!(decoded.is_empty());
}
