//! Exact C-ABI one-shot decompression for native comparison tools.

use std::error::Error;
use std::ffi::c_ulong;

pub type NativeResult<T> = Result<T, Box<dyn Error>>;

macro_rules! native_one_shot {
    ($api:ident, $one_shot:ident) => {
        pub fn $one_shot(output: &mut [u8], input: &[u8]) -> NativeResult<usize> {
            use $api as ffi;

            let mut output_len = c_ulong::try_from(output.len())?;
            let input_len = c_ulong::try_from(input.len())?;
            // SAFETY: input is readable for exactly input_len bytes, output is writable
            // for exactly output_len bytes, and output_len is a live mutable C scalar.
            let code = unsafe {
                ffi::uncompress(
                    output.as_mut_ptr(),
                    &raw mut output_len,
                    input.as_ptr(),
                    input_len,
                )
            };
            if code != ffi::Z_OK {
                return Err(format!("native uncompress failed with status {code}").into());
            }
            let produced = usize::try_from(output_len)?;
            if produced > output.len() {
                return Err("native uncompress returned an output length beyond capacity".into());
            }
            Ok(produced)
        }
    };
}

native_one_shot!(libz_sys, uncompress_ng);
native_one_shot!(libz_rs_sys, uncompress_rs);
