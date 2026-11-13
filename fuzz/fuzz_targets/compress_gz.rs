#![cfg_attr(not(miri), no_main)]

use std::{
    ffi::{c_int, c_uint, c_ulong, CString},
    mem::{size_of, MaybeUninit},
};

use libfuzzer_sys::{
    arbitrary::{self, Arbitrary},
    fuzz_target,
};

use zlib_rs::{deflate::DeflateConfig, DeflateFlush, ReturnCode};

#[derive(Debug, Arbitrary, Clone)]
struct GzHeaderData {
    text: i32,
    time: c_ulong,
    os: i32,
    extra: Vec<u8>,
    name: CString,
    comment: CString,
    hcrc: i32,
}

impl GzHeaderData {
    fn as_gz_header(&mut self) -> libz_rs_sys::gz_header {
        libz_rs_sys::gz_header {
            text: self.text,
            time: self.time,
            xflags: 0,
            os: self.os,
            extra: self.extra.as_mut_ptr(),
            extra_len: self.extra.len().try_into().unwrap(),
            extra_max: 0,                              // doesn't matter for writing.
            name: self.name.as_ptr() as *mut u8, // hack: UB if written to, but we shouldn't write during deflate.
            name_max: 0,                         // doesn't matter for writing.
            comment: self.comment.as_ptr() as *mut u8, // hack: UB if written to, but we shouldn't write during deflate.
            comm_max: 0,                               // doesn't matter for writing.
            hcrc: self.hcrc,
            done: 0, // doesn't matter for writing.
        }
    }
}

#[derive(Debug, Arbitrary)]
struct Input {
    source: Vec<u8>,
    config: DeflateConfig,
    flush: DeflateFlush,
    header: GzHeaderData,
}

fuzz_target!(|input: Input| {
    let reference = compress_gz_ng(&input);
    let candidate = compress_gz_rs(&input);

    match (reference, candidate) {
        (None, None) => {}
        // Preserve the original byte-parity fast path: only independently
        // decode streams when their otherwise-valid encodings actually differ.
        (Some(reference), Some(candidate)) if reference == candidate => {}
        (Some(reference), Some(candidate)) => {
            // DEFLATE permits different, equally valid block boundaries and
            // encodings for the same input. Gzip metadata and the integrity
            // trailer must agree exactly; independently inflate *both*
            // complete streams to verify the actual compression contract.
            assert_eq!(
                gzip_header(&reference),
                gzip_header(&candidate),
                "gzip metadata differs"
            );
            assert_eq!(
                reference.get(reference.len().saturating_sub(8)..),
                candidate.get(candidate.len().saturating_sub(8)..),
                "gzip integrity trailer differs"
            );
            assert!(
                reference.len() >= 18 && candidate.len() >= 18,
                "incomplete gzip output"
            );
            assert!(
                decompress_with_zlib_ng(&reference, &input.source),
                "reference gzip output is not a complete stream"
            );
            assert!(
                decompress_with_zlib_ng(&candidate, &input.source),
                "Rust gzip output does not recover the input"
            );
        }
        (reference, candidate) => {
            panic!(
                "gzip initialization/status disagreement: reference={reference:?}, Rust={candidate:?}"
            );
        }
    }
});

/// Extract the complete gzip header, including optional fields and FHCRC.
/// An incomplete/malformed header must never compare equal by accident.
fn gzip_header(gzip: &[u8]) -> Option<&[u8]> {
    if gzip.len() < 18 || gzip.get(..3)? != [0x1f, 0x8b, 8] {
        return None;
    }
    let flags = gzip[3];
    if flags & 0xe0 != 0 {
        return None;
    }
    let mut end = 10usize;
    if flags & 0x04 != 0 {
        let length = u16::from_le_bytes([*gzip.get(end)?, *gzip.get(end + 1)?]);
        end = end.checked_add(2 + usize::from(length))?;
        gzip.get(..end)?;
    }
    for flag in [0x08, 0x10] {
        if flags & flag != 0 {
            let zero = gzip.get(end..)?.iter().position(|&byte| byte == 0)?;
            end = end.checked_add(zero + 1)?;
        }
    }
    if flags & 0x02 != 0 {
        end = end.checked_add(2)?;
    }
    // At least the eight-byte trailer must remain beyond the header.
    (gzip.len().checked_sub(end)? >= 8).then(|| &gzip[..end])
}

/// Reference-library validation of the entire stream, including gzip's
/// CRC32 and uncompressed size; excess input is not a successful result.
fn decompress_with_zlib_ng(gzip: &[u8], expected: &[u8]) -> bool {
    use libz_ng_sys::*;

    let mut stream = MaybeUninit::<z_stream>::zeroed();
    // SAFETY: the z_stream buffer has the ABI-prescribed layout and size.
    let init = unsafe {
        inflateInit2_(
            stream.as_mut_ptr(),
            31, // gzip wrapper, 32 KiB LZ77 window
            zlibVersion(),
            size_of::<z_stream>() as c_int,
        )
    };
    if init != Z_OK {
        return false;
    }
    // SAFETY: successful inflateInit2_ initialized the stream.
    let stream = unsafe { stream.assume_init_mut() };
    let mut decoded = [0u8; 32 * 1024];
    let mut input_offset = 0usize;
    let mut output_offset = 0usize;
    let valid = loop {
        if stream.avail_in == 0 && input_offset < gzip.len() {
            let pending = &gzip[input_offset..];
            let amount = pending.len().min(decoded.len());
            stream.next_in = pending.as_ptr().cast_mut();
            stream.avail_in = amount as c_uint;
            input_offset += amount;
        }
        let before_avail_in = stream.avail_in;
        stream.next_out = decoded.as_mut_ptr();
        stream.avail_out = decoded.len() as c_uint;

        // SAFETY: pending compressed bytes and the bounded writable scratch
        // buffer remain alive for the entire inflate call.
        let result = unsafe { inflate(stream, Z_NO_FLUSH) };
        let produced = decoded.len() - stream.avail_out as usize;
        let Some(end) = output_offset.checked_add(produced) else {
            break false;
        };
        if expected.get(output_offset..end) != Some(&decoded[..produced]) {
            break false;
        }
        output_offset = end;
        if result == Z_STREAM_END {
            // Require the complete original input, valid CRC and ISIZE (zlib-ng
            // checks them), and no silently ignored trailing compressed bytes.
            break output_offset == expected.len()
                && stream.avail_in == 0
                && input_offset == gzip.len();
        }
        if result != Z_OK || (before_avail_in == stream.avail_in && produced == 0) {
            break false;
        }
    };
    // SAFETY: successful inflateInit2_ owns the current inflate state.
    let ended = unsafe { inflateEnd(stream) };
    valid && ended == Z_OK
}

fn compress_gz_rs(input: &Input) -> Option<Vec<u8>> {
    use libz_rs_sys::*;

    let Input {
        ref source,
        config,
        flush,
        ref header,
    } = input;

    let mut header = header.clone();

    // Initialize stream.
    let mut stream = MaybeUninit::zeroed();
    let err = unsafe {
        deflateInit2_(
            stream.as_mut_ptr(),
            config.level,
            config.method as i32,
            config.window_bits,
            config.mem_level,
            config.strategy as i32,
            zlibVersion(),
            size_of::<z_stream>() as c_int,
        )
    };

    if err != ReturnCode::Ok as i32 {
        return None;
    }

    let streamp = unsafe { stream.assume_init_mut() };

    // Create header.
    let mut header = header.as_gz_header();
    let err = unsafe { deflateSetHeader(streamp, &mut header as gz_headerp) };
    if err != ReturnCode::Ok as i32 {
        // Deallocate, so that we don't trigger ASAN leak detector.
        let err = unsafe { deflateEnd(streamp) };
        assert_eq!(err, ReturnCode::Ok as i32);
        return None;
    }

    let bound = unsafe { deflateBound(streamp, source.len() as _) };
    let buf_size = match flush {
        DeflateFlush::NoFlush | DeflateFlush::Finish => bound,
        // Other flush options might require more than `bound` bytes, so add a safety margin. We
        // _could_ catch Z_BUF_ERROR to allocate more memory, but that's not interesting right now.
        // (In fact, it's more interesting if NoFlush/Finish triggers UB write if we're
        // miscalculating)
        DeflateFlush::PartialFlush
        | DeflateFlush::SyncFlush
        | DeflateFlush::FullFlush
        | DeflateFlush::Block => bound * 2,
    };

    // We deliberately use uninitialized memory as the input buffer here, to simulate how these
    // functions will likely be used from C, and to catch us ever reading uninitialized memory.
    let mut dest: Vec<u8> = Vec::with_capacity(buf_size as usize);

    let max = c_uint::MAX as usize;

    let mut left = dest.capacity();
    let mut source_len = source.len();

    streamp.next_in = source.as_ptr().cast_mut().cast();
    streamp.next_out = dest.as_mut_ptr().cast();

    loop {
        if streamp.avail_out == 0 {
            streamp.avail_out = Ord::min(left, max) as _;
            left -= streamp.avail_out as usize;
        }

        if streamp.avail_in == 0 {
            streamp.avail_in = Ord::min(source_len, max) as _;
            source_len -= streamp.avail_in as usize;
        }

        let flush = if source_len > 0 {
            *flush
        } else {
            DeflateFlush::Finish
        };

        let err = unsafe { deflate(streamp, flush as i32) };
        if ReturnCode::from(err) != ReturnCode::Ok {
            break;
        }
    }

    unsafe { dest.set_len(streamp.total_out as usize) }

    let err = unsafe { deflateEnd(streamp) };
    assert_eq!(ReturnCode::from(err), ReturnCode::Ok);

    Some(dest)
}

fn compress_gz_ng(input: &Input) -> Option<Vec<u8>> {
    use libz_ng_sys::*;

    let Input {
        ref source,
        config,
        flush,
        ref header,
    } = input;

    let mut header = header.clone();

    // Initialize stream.
    let mut stream = MaybeUninit::zeroed();
    let err = unsafe {
        deflateInit2_(
            stream.as_mut_ptr(),
            config.level,
            config.method as i32,
            config.window_bits,
            config.mem_level,
            config.strategy as i32,
            zlibVersion(),
            size_of::<z_stream>() as c_int,
        )
    };

    if err != ReturnCode::Ok as i32 {
        return None;
    }

    let streamp = unsafe { stream.assume_init_mut() };

    // Create header. The layout is the same between zlib-rs and zlib-ng.
    let mut header =
        unsafe { core::mem::transmute::<libz_rs_sys::gz_header, gz_header>(header.as_gz_header()) };

    let err = unsafe { deflateSetHeader(streamp, &mut header as gz_headerp) };
    if err != ReturnCode::Ok as i32 {
        // Deallocate, so that we don't trigger ASAN leak detector.
        let err = unsafe { deflateEnd(streamp) };
        assert_eq!(err, ReturnCode::Ok as i32);
        return None;
    }

    let bound = unsafe { deflateBound(streamp, source.len() as _) };
    let buf_size = match flush {
        DeflateFlush::NoFlush | DeflateFlush::Finish => bound,
        // Other flush options might require more than `bound` bytes, so add a safety margin. We
        // _could_ catch Z_BUF_ERROR to allocate more memory, but that's not interesting right now.
        // (In fact, it's more interesting if NoFlush/Finish triggers UB write if we're
        // miscalculating)
        DeflateFlush::PartialFlush
        | DeflateFlush::SyncFlush
        | DeflateFlush::FullFlush
        | DeflateFlush::Block => bound * 2,
    };

    // We deliberately use uninitialized memory as the input buffer here, to simulate how these
    // functions will likely be used from C, and to catch us ever reading uninitialized memory.
    let mut dest: Vec<u8> = Vec::with_capacity(buf_size as usize);

    let max = c_uint::MAX as usize;

    let mut left = dest.capacity();
    let mut source_len = source.len();

    streamp.next_in = source.as_ptr().cast_mut().cast();
    streamp.next_out = dest.as_mut_ptr().cast();

    loop {
        if streamp.avail_out == 0 {
            streamp.avail_out = Ord::min(left, max) as _;
            left -= streamp.avail_out as usize;
        }

        if streamp.avail_in == 0 {
            streamp.avail_in = Ord::min(source_len, max) as _;
            source_len -= streamp.avail_in as usize;
        }

        let flush = if source_len > 0 {
            *flush
        } else {
            DeflateFlush::Finish
        };

        let err = unsafe { deflate(streamp, flush as i32) };
        if ReturnCode::from(err) != ReturnCode::Ok {
            break;
        }
    }

    unsafe { dest.set_len(streamp.total_out as _) }

    let err = unsafe { deflateEnd(streamp) };
    assert_eq!(ReturnCode::from(err), ReturnCode::Ok);

    Some(dest)
}
