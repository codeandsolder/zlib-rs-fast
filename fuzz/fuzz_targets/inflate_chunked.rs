#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|input: (&[u8], u8, u8)| {
    // Small chunk sizes are much more useful to state-machine fuzzing than
    // overwhelmingly huge random usize values that behave like one-shot I/O.
    test_libz_rs_sys::chunked_inflate::cross_decode_chunked(
        input.0,
        usize::from(input.1) + 1,
        usize::from(input.2) + 1,
    );
});
