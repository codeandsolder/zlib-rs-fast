use std::num::TryFromIntError;
use std::time::Duration;

/// Convert any usize width into a display-rate approximation without
/// truncating intermediate values into a smaller integer type.
pub fn approximately_f64(value: usize) -> f64 {
    value
        .to_be_bytes()
        .into_iter()
        .fold(0.0, |sum, octet| sum.mul_add(256.0, f64::from(octet)))
}

/// Bound sample count using exact nanosecond arithmetic. A nonzero first
/// measurement is rounded up to reach the target rather than rounded down.
pub fn iterations_for(
    one: Duration,
    sample_target: Duration,
    max_iters: usize,
) -> Result<usize, TryFromIntError> {
    if one.is_zero() {
        return Ok(max_iters);
    }
    let maximum = u128::try_from(max_iters)?;
    usize::try_from(
        sample_target
            .as_nanos()
            .div_ceil(one.as_nanos())
            .clamp(1, maximum),
    )
}

/// Match low-word, wrapping seed arithmetic on both 32- and 64-bit hosts.
pub fn repeating_phase(seed: u64) -> usize {
    seed.to_be_bytes()
        .into_iter()
        .fold(0usize, |word, byte| {
            word.wrapping_mul(256).wrapping_add(usize::from(byte))
        })
        .wrapping_mul(7)
}
