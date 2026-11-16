use zlib_rs::{deflate::DeflateConfig, ReturnCode};

use crate::helpers::compress_slice_ng;

fn cve_test(input: &[u8]) {
    // Keep the historical 128 KiB output capacity, but avoid placing two
    // large C ABI destination buffers on the test thread's stack.
    const OUTPUT_CAPACITY: usize = 1 << 17;
    let mut output_ng = vec![0; OUTPUT_CAPACITY];
    // flush type 4 = Finish is the default
    let config = DeflateConfig {
        window_bits: 15,
        mem_level: 1,
        ..DeflateConfig::default()
    };
    let (output_ng, err) = compress_slice_ng(&mut output_ng, input, config);
    assert_eq!(err, Ok(ReturnCode::Ok));

    let mut output_rs = vec![0; OUTPUT_CAPACITY];
    let (output_rs, err) = zlib_rs::deflate::compress_slice(&mut output_rs, input, config);
    assert_eq!(err, ReturnCode::Ok);

    assert_eq!(output_ng, output_rs);

    let mut output = vec![0; input.len()];
    let config = zlib_rs::inflate::InflateConfig { window_bits: 15 };
    let (output, err) = zlib_rs::inflate::decompress_slice(&mut output, output_rs, config);
    assert_eq!(err, ReturnCode::Ok);

    assert_eq!(input, output);
}

#[test]
#[cfg_attr(miri, ignore)]
fn zlib_ng_cve_2018_25032_default() {
    const DEFAULT: &str = include_str!("test-data/zlib-ng/CVE-2018-25032/default.txt");
    cve_test(DEFAULT.as_bytes());
}

#[test]
#[cfg_attr(miri, ignore)]
fn zlib_ng_cve_2018_25032_fixed() {
    const FIXED: &str = include_str!("test-data/zlib-ng/CVE-2018-25032/fixed.txt");
    cve_test(FIXED.as_bytes());
}

#[test]
#[cfg_attr(miri, ignore)]
fn zlib_ng_gh_382() {
    const DEFNEG: &[u8] = include_bytes!("test-data/zlib-ng/GH-382/defneg3.dat");
    cve_test(DEFNEG);
}
