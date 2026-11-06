#![no_main]
use core::ffi::{c_int, c_uchar, c_uint, c_void};
use core::mem::MaybeUninit;

use libfuzzer_sys::{fuzz_target, Corpus};

fuzz_target!(|input: &[u8]| -> Corpus { entry(input) });

fn entry(input: &[u8]) -> Corpus {
    if input.is_empty() {
        return Corpus::Reject;
    }

    differential_inflate_back::<1>(input);
    differential_inflate_back::<512>(input);

    Corpus::Keep
}

fn differential_inflate_back<const CHUNK: usize>(input: &[u8]) {
    // Per the documentation, only window_bits 15 is supported.
    let window_bits = 15;

    let ng_out = run_inflate_back_ng::<CHUNK>(input, window_bits);
    let rs_out = run_inflate_back_rs::<CHUNK>(input, window_bits);

    // Error-code parity is not a valid property for malformed streams:
    // zlib-ng built without INFLATE_STRICT may read uninitialized back-history
    // and report an eventual input-shortage (Z_BUF_ERROR), while zlib-rs
    // safely rejects the invalid distance (Z_DATA_ERROR).
    //
    // Always compare successful bytes exactly, and treat disagreement about
    // whether a stream is valid as a real differential failure.
    match (ng_out, rs_out) {
        (Ok(expected), Ok(actual)) => {
            assert_eq!(
                actual, expected,
                "inflateBack output mismatch for window_bits = {window_bits}, CHUNK = {CHUNK}",
            );
        }
        (Err(_), Err(_)) => {}
        (reference, ours) => {
            panic!(
                "inflateBack success/failure mismatch for window_bits = {window_bits}, CHUNK = {CHUNK}: reference={reference:?}, ours={ours:?}",
            );
        }
    }
}

/// Shared input context for the inflateBack `in` callback.
struct InputCtx<'a> {
    data: &'a [u8],
    pos: usize,
}

/// Shared output context for the inflateBack `out` callback.
struct OutputCtx {
    buf: Vec<u8>,
}

/// `in` callback: supplies more compressed data to inflateBack.
unsafe extern "C" fn pull_cb<const CHUNK: usize>(
    desc: *mut c_void,
    buf: *mut *const c_uchar,
) -> c_uint {
    let Some(ctx) = desc.cast::<InputCtx>().as_mut() else {
        return 0;
    };

    if ctx.pos >= ctx.data.len() {
        // No more data
        *buf = core::ptr::null();
        return 0;
    }

    // Feed one byte at a time (stress the state machine a bit)
    let remaining = ctx.data.len() - ctx.pos;
    let chunk = Ord::min(CHUNK, remaining);

    let ptr = ctx.data[ctx.pos..].as_ptr();
    *buf = ptr;

    ctx.pos += chunk;
    chunk as c_uint
}

/// `out` callback: collects decompressed bytes from inflateBack.
///
/// C signature:
///   int out_func(void *desc, unsigned char *buf, unsigned len);
unsafe extern "C" fn push_cb(desc: *mut c_void, buf: *mut c_uchar, len: c_uint) -> c_int {
    let Some(ctx) = desc.cast::<OutputCtx>().as_mut() else {
        // Signal error; inflateBack will return Z_BUF_ERROR.
        return 1;
    };

    let slice = core::slice::from_raw_parts(buf as *const u8, len as usize);
    ctx.buf.extend_from_slice(slice);

    // 0 means "ok, continue"
    0
}

fn run_inflate_back_ng<const CHUNK: usize>(
    input: &[u8],
    window_bits: c_int,
) -> Result<Vec<u8>, c_int> {
    let mut strm = MaybeUninit::zeroed();
    let mut window = vec![0xAA; 1 << window_bits];

    let mut in_ctx = InputCtx {
        data: input,
        pos: 0,
    };
    let mut out_ctx = OutputCtx { buf: Vec::new() };

    let in_desc: *mut c_void = &mut in_ctx as *mut _ as *mut c_void;
    let out_desc: *mut c_void = &mut out_ctx as *mut _ as *mut c_void;

    unsafe {
        let ret = libz_ng_sys::inflateBackInit_(
            strm.as_mut_ptr(),
            window_bits,
            window.as_mut_ptr(),
            libz_ng_sys::zlibVersion(),
            core::mem::size_of::<libz_ng_sys::z_stream>() as c_int,
        );

        if ret != libz_ng_sys::Z_OK {
            return Err(ret);
        }

        let ret = libz_ng_sys::inflateBack(
            strm.as_mut_ptr(),
            pull_cb::<CHUNK>,
            in_desc,
            push_cb,
            out_desc,
        );

        let _ = libz_ng_sys::inflateBackEnd(strm.as_mut_ptr());

        match ret {
            libz_ng_sys::Z_STREAM_END => Ok(out_ctx.buf),
            _ => Err(ret),
        }
    }
}

fn run_inflate_back_rs<const CHUNK: usize>(
    input: &[u8],
    window_bits: c_int,
) -> Result<Vec<u8>, c_int> {
    let mut strm = MaybeUninit::zeroed();
    let mut window = vec![0xAA; 1 << window_bits];

    let mut in_ctx = InputCtx {
        data: input,
        pos: 0,
    };
    let mut out_ctx = OutputCtx { buf: Vec::new() };

    let in_desc: *mut c_void = &mut in_ctx as *mut _ as *mut c_void;
    let out_desc: *mut c_void = &mut out_ctx as *mut _ as *mut c_void;

    unsafe {
        let ret = libz_rs_sys::inflateBackInit_(
            strm.as_mut_ptr(),
            window_bits,
            window.as_mut_ptr(),
            libz_rs_sys::zlibVersion(),
            core::mem::size_of::<libz_rs_sys::z_stream>() as c_int,
        );

        if ret != libz_rs_sys::Z_OK {
            return Err(ret);
        }

        let ret = libz_rs_sys::inflateBack(
            strm.as_mut_ptr(),
            Some(pull_cb::<CHUNK>),
            in_desc,
            Some(push_cb),
            out_desc,
        );

        let _ = libz_rs_sys::inflateBackEnd(strm.as_mut_ptr());

        match ret {
            libz_rs_sys::Z_STREAM_END => Ok(out_ctx.buf),
            _ => Err(ret),
        }
    }
}
