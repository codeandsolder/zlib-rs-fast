use core::ffi::{c_char, c_int, c_uint, c_ulong, CStr};
use std::{ffi::CString, mem::MaybeUninit};

// we use the libz_sys but configure zlib-ng in zlib compat mode
use libz_sys as libz_ng_sys;

use libz_rs_sys::{
    deflate, deflateEnd, deflateInit2_, inflate, inflateEnd, inflateInit2_, Z_DEFLATED, Z_FILTERED,
    Z_NO_FLUSH, Z_OK,
};
use zlib_rs::{
    c_api::Z_BEST_COMPRESSION,
    deflate::{DeflateConfig, Method, Strategy},
    inflate::InflateConfig,
    DeflateFlush, ReturnCode,
};

use super::inflate::{test_c_stream_size, test_c_uint_len, test_c_ulong_len};

use crate::assert_eq_rs_ng;

const VERSION: *const c_char = libz_rs_sys::zlibVersion();

const PAPER_100K: &[u8] = include_bytes!("test-data/paper-100k.pdf");
const FIREWORKS: &[u8] = include_bytes!("test-data/fireworks.jpg");
const LCET10: &str = include_str!("test-data/lcet10.txt");

pub mod quick {
    use super::*;

    #[rustfmt::skip]
    const BI_VALID_INPUT: [u8; 554] = [
        0x8d, 0xff, 0xff, 0xff, 0xa2, 0x00, 0x00, 0xff, 0x00, 0x15, 0x1b, 0x1b, 0xa2, 0xa2, 0xaf, 0xa2,
        0xa2, 0x00, 0x00, 0x00, 0x02, 0x00, 0x1b, 0x3f, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x0b,
        0x00, 0xab, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x2b, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x01, 0x1e, 0x00, 0x00, 0x01, 0x40, 0x00, 0x00, 0x00, 0x07, 0x01, 0x18, 0x00, 0x22, 0x00,
        0x00, 0x00, 0xfd, 0x39, 0xff, 0x00, 0x00, 0x00, 0x1b, 0xfd, 0x3b, 0x00, 0x68, 0x00, 0x00, 0x01,
        0xff, 0xff, 0xff, 0x57, 0xf8, 0x1e, 0x00, 0x00, 0xf2, 0xf2, 0xf2, 0xf2, 0xfa, 0xff, 0xff, 0xff,
        0xff, 0x7e, 0x00, 0x00, 0x4a, 0x00, 0xc5, 0x00, 0x41, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00,
        0x00, 0x02, 0x01, 0x01, 0x00, 0xa2, 0x08, 0x00, 0x00, 0x00, 0x00, 0x27, 0x4a, 0x4a, 0x4a, 0x32,
        0x00, 0xf9, 0xff, 0x00, 0x02, 0x9a, 0xff, 0x00, 0x00, 0x3f, 0x50, 0x00, 0x03, 0x00, 0x00, 0x00,
        0x3d, 0x00, 0x08, 0x2f, 0x20, 0x00, 0x23, 0x00, 0x00, 0x00, 0x00, 0x23, 0x00, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0x7a, 0x7a, 0x9e, 0xff, 0xff, 0x00, 0x1b, 0x1b, 0x04, 0x00, 0x1b, 0x1b,
        0x1b, 0x1b, 0x00, 0x00, 0x00, 0xaf, 0xad, 0xaf, 0x00, 0x00, 0xa8, 0x00, 0x00, 0x00, 0x2e, 0xff,
        0xff, 0x2e, 0xc1, 0x00, 0x10, 0x00, 0x00, 0x00, 0x06, 0x70, 0x00, 0x00, 0x00, 0xda, 0x67, 0x01,
        0x47, 0x00, 0x00, 0x00, 0x0c, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0x00, 0x01, 0x00, 0x3f,
        0x54, 0x00, 0x00, 0x00, 0x1b, 0x00, 0x00, 0x00, 0x5c, 0x00, 0x00, 0x34, 0x3e, 0xc5, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x7a, 0x00, 0x00, 0x00, 0x0a, 0x01, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x7a, 0x7a, 0x7a, 0x7a, 0x7a, 0x00, 0x00, 0x00, 0x40, 0x1b, 0x1b, 0x88, 0x1b, 0x1b,
        0x1b, 0x1b, 0x1b, 0x1b, 0x1b, 0x1f, 0x1b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00,
        0x00, 0x04, 0x00, 0x00, 0x50, 0x3e, 0x7a, 0x7a, 0x00, 0x00, 0x40, 0x00, 0x40, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x08, 0x87, 0x00, 0x00, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x01, 0x00, 0xff, 0x3d, 0x00, 0x11, 0x4d, 0x00, 0x00, 0x01, 0xd4, 0xd4, 0xd4, 0xd4, 0x2d, 0xd4,
        0xd4, 0xff, 0xff, 0xff, 0xfa, 0x01, 0xd4, 0x00, 0xd4, 0x00, 0x00, 0xd4, 0xd4, 0xd4, 0xd4, 0xd4,
        0xd4, 0x1e, 0x1e, 0x1e, 0x1e, 0x00, 0x00, 0xfe, 0xf9, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x00,
        0x16, 0xd4, 0xd4, 0xd4, 0xd4, 0xd4, 0xd4, 0xd4, 0xd4, 0xd4, 0x00, 0x00, 0x80, 0x20, 0x00, 0x00,
        0xff, 0x2b, 0x2b, 0x2b, 0x2b, 0x35, 0xd4, 0xd4, 0x47, 0x3f, 0xd4, 0xd4, 0xd6, 0xd4, 0xd4, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x32, 0x4a, 0x4a, 0x4a, 0x4a, 0x71, 0x00, 0x1b, 0x1b, 0x1b, 0x1b, 0x1b,
        0x1f, 0x1b, 0x1b, 0x1b, 0x57, 0x57, 0x57, 0x57, 0x00, 0x00, 0x1b, 0x08, 0x2b, 0x16, 0xc3, 0x00,
        0x00, 0x00, 0x29, 0x30, 0x03, 0xff, 0x03, 0x03, 0x03, 0x03, 0x07, 0x00, 0x00, 0x01, 0x0b, 0xff,
        0xff, 0xf5, 0xf5, 0xf5, 0x00, 0x00, 0xfe, 0xfa, 0x0f, 0x0f, 0x08, 0x00, 0xff, 0x00, 0x53, 0x3f,
        0x00, 0x04, 0x5d, 0xa8, 0x2e, 0xff, 0xff, 0x00, 0x2f, 0x2f, 0x05, 0xff, 0xff, 0xff, 0x2f, 0x2f,
        0x2f, 0x0a, 0x0a, 0x0a, 0x0a, 0x30, 0xff, 0xff, 0xff, 0xf0, 0x0a, 0x0a, 0x0a, 0x00, 0xff, 0x3f,
        0x4f, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x71, 0x00, 0x2e, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x71, 0x71, 0x00, 0x71, 0x71, 0x71, 0xf5, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0xff,
        0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0xdb, 0x3f, 0x00, 0xfa, 0x71, 0x71, 0x71, 0x00, 0x00,
        0x00, 0x01, 0x00, 0x00, 0x00, 0x71, 0x71, 0x71, 0x71, 0x71
    ];

    #[test]
    fn bi_valid() {
        let mut raw = MaybeUninit::<libz_rs_sys::z_stream>::zeroed();
        // SAFETY: writable C stream with valid gzip settings, version and
        // checked structure size. Initial mem_level=1 is deliberate.
        let init = unsafe {
            deflateInit2_(
                raw.as_mut_ptr(),
                1,
                Z_DEFLATED,
                31,
                1,
                Z_FILTERED,
                VERSION,
                test_c_stream_size::<libz_rs_sys::z_stream>(),
            )
        };
        assert_eq!(init, libz_rs_sys::Z_OK);
        // SAFETY: the successful C initialization populates the stream.
        let stream = unsafe { raw.assume_init_mut() };

        let mut compressed = [0xa5u8; 1_236];
        stream.next_in = BI_VALID_INPUT.as_ptr().cast_mut();
        stream.avail_in = test_c_uint_len(BI_VALID_INPUT.len());
        stream.next_out = compressed.as_mut_ptr();
        stream.avail_out = 31;

        // SAFETY: the live source is readable; only the first 31 bytes
        // of the independent compressed allocation are writable.
        assert_eq!(
            // SAFETY: the live input and exactly 31 writable output bytes
            // are disjoint and the C encoder was initialized successfully.
            unsafe { deflate(stream, DeflateFlush::Finish as c_int) },
            libz_rs_sys::Z_OK
        );
        let written = usize::try_from(stream.total_out).unwrap_or(usize::MAX);
        assert!(written <= 31);
        assert!(compressed[written..].iter().all(|&b| b == 0xa5));

        // Preserve stream.avail_in: an earlier Z_OK does *not* imply that
        // all offered source bytes were consumed. Continue from the real
        // next_in and total_out cursors until StreamEnd.
        loop {
            let written = usize::try_from(stream.total_out).unwrap_or(usize::MAX);
            assert!(written < compressed.len(), "compressed output overflow");
            let offer = (compressed.len() - written).min(498);
            stream.next_out = compressed[written..].as_mut_ptr();
            stream.avail_out = test_c_uint_len(offer);
            let before = (stream.total_in, stream.total_out);
            // SAFETY: input and output refer to live, disjoint allocations;
            // next_in and avail_in retain the actual unread source suffix.
            let status = unsafe { deflate(stream, DeflateFlush::Finish as c_int) };
            if status == libz_rs_sys::Z_STREAM_END {
                break;
            }
            assert_eq!(status, libz_rs_sys::Z_OK);
            assert_ne!(
                (stream.total_in, stream.total_out),
                before,
                "encoder stalled before finishing"
            );
        }
        assert_eq!(stream.avail_in, 0);
        assert_eq!(usize::try_from(stream.total_in), Ok(BI_VALID_INPUT.len()));
        let produced = usize::try_from(stream.total_out).unwrap_or(usize::MAX);
        assert!(produced <= compressed.len());
        assert!(compressed[produced..].iter().all(|&b| b == 0xa5));
        // SAFETY: the finished encoder is closed once with all borrowed
        // allocations still valid.
        assert_eq!(unsafe { deflateEnd(stream) }, libz_rs_sys::Z_OK);

        let mut plaintext = [0xa5u8; BI_VALID_INPUT.len() + 16];
        let (decoded, status) = zlib_rs::decompress_slice(
            &mut plaintext,
            &compressed[..produced],
            InflateConfig { window_bits: 31 },
        );
        assert_eq!(status, ReturnCode::Ok);
        assert_eq!(decoded, BI_VALID_INPUT);
        assert_eq!(&plaintext[BI_VALID_INPUT.len()..], &[0xa5u8; 16]);
    }

    #[rustfmt::skip]
    const BLOCK_OPEN_INPUT: [u8; 495] = [
        0x1d, 0x1d, 0x00, 0x00, 0x00, 0x4a, 0x4a, 0x4a, 0xaf, 0xaf, 0xaf, 0xaf, 0x4a, 0x4a, 0x4a, 0x4a,
        0x3f, 0x3e, 0xaf, 0xff, 0xff, 0xff, 0x11, 0xff, 0xff, 0xff, 0xff, 0xdf, 0x00, 0x00, 0x00, 0x01,
        0x3f, 0x7d, 0x00, 0x50, 0x00, 0x00, 0xc8, 0x01, 0x2b, 0x60, 0xc8, 0x00, 0x24, 0x06, 0xff, 0xff,
        0x4a, 0x4e, 0x4a, 0x7d, 0xc8, 0x01, 0xf1, 0x2b, 0x28, 0xb2, 0xb2, 0x60, 0x25, 0xc8, 0x06, 0x00,
        0x00, 0x00, 0x31, 0x00, 0x01, 0xb2, 0xb2, 0xb2, 0xff, 0xff, 0xfd, 0xb2, 0xb2, 0x40, 0xff, 0x7d,
        0x3b, 0x34, 0x3e, 0xff, 0xff, 0x4a, 0x4a, 0x01, 0xf1, 0xff, 0x02, 0xff, 0x3f, 0xff, 0x02, 0xff,
        0xff, 0xff, 0xbf, 0x0a, 0xff, 0x00, 0x01, 0x3f, 0xb3, 0xff, 0x26, 0x00, 0x00, 0x13, 0x00, 0xc8,
        0x3e, 0x3e, 0x3e, 0x4a, 0x76, 0x4a, 0x4a, 0x2e, 0x7d, 0x3e, 0x3e, 0x3e, 0x3e, 0x1d, 0x1d, 0x1d,
        0xfe, 0xea, 0xef, 0x80, 0x01, 0x00, 0x00, 0x40, 0x00, 0x00, 0xba, 0x00, 0x06, 0xfa, 0xb9, 0x11,
        0xbf, 0x98, 0xee, 0x45, 0x7e, 0x04, 0x00, 0xff, 0xff, 0xff, 0x67, 0xc3, 0xc3, 0xc3, 0xc3, 0x00,
        0x1d, 0x1d, 0xe1, 0xe3, 0x00, 0xc3, 0x1d, 0x98, 0x1d, 0x1d, 0x1d, 0x1d, 0x1d, 0x00, 0x00, 0x00,
        0x02, 0x00, 0x00, 0x00, 0xe8, 0x00, 0x00, 0x1d, 0x1d, 0x1d, 0xfa, 0x1e, 0x12, 0xff, 0xff, 0xff,
        0x00, 0x01, 0xa7, 0xff, 0xff, 0xff, 0x1d, 0x1d, 0x1d, 0x63, 0xff, 0xff, 0xff, 0x1f, 0x00, 0x00,
        0x10, 0x40, 0x00, 0x00, 0xad, 0xff, 0xff, 0x3f, 0x51, 0x00, 0xf8, 0xff, 0xff, 0x8a, 0x01, 0x05,
        0x00, 0x00, 0x03, 0x00, 0x00, 0xff, 0x00, 0x00, 0x00, 0x05, 0x40, 0x1f, 0x08, 0x0a, 0x00, 0xff,
        0xff, 0x01, 0x00, 0x12, 0x00, 0x00, 0x01, 0x00, 0x3f, 0x40, 0x1d, 0x1d, 0x1d, 0x1d, 0x1d, 0x1d,
        0x21, 0x00, 0x1d, 0x00, 0x00, 0x00, 0xe4, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0xe6, 0xe6, 0x34,
        0xe6, 0xe6, 0xe6, 0xe6, 0xff, 0x2b, 0xee, 0x1d, 0x1d, 0x1d, 0x93, 0x1d, 0x1d, 0x1d, 0xee, 0x2b,
        0xee, 0x01, 0x81, 0x1d, 0x00, 0x00, 0x58, 0x00, 0x00, 0x01, 0x14, 0x00, 0x1b, 0x00, 0x00, 0x2c,
        0x00, 0x00, 0x00, 0xdb, 0x00, 0x45, 0x7e, 0x00, 0x00, 0x00, 0xfb, 0xbd, 0x00, 0x06, 0x21, 0xd3,
        0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x49, 0x49, 0xc9, 0x49, 0x3d, 0x00, 0x34, 0x01, 0x00,
        0x00, 0x6a, 0x2b, 0x00, 0x00, 0x50, 0x40, 0xf0, 0xf0, 0xf0, 0xf0, 0xa3, 0xa3, 0xa3, 0xa3, 0xf0,
        0xf0, 0x06, 0xfa, 0xa9, 0x01, 0x10, 0xbf, 0x98, 0x9d, 0x2b, 0xee, 0x2d, 0x21, 0x01, 0xdb, 0x00,
        0x45, 0x10, 0x00, 0x00, 0x7e, 0x00, 0x00, 0xe7, 0x00, 0xff, 0xff, 0x00, 0xf6, 0x00, 0x00, 0x00,
        0xf9, 0x00, 0x00, 0x00, 0x11, 0x00, 0x00, 0x00, 0xe2, 0x00, 0x00, 0x00, 0x2d, 0x00, 0x00, 0x00,
        0x2f, 0x00, 0x3f, 0x54, 0x1d, 0x1d, 0x1d, 0x4c, 0x4c, 0x4c, 0x4c, 0x2a, 0x4c, 0x4c, 0x10, 0xff,
        0xff, 0x1a, 0x00, 0x00, 0x01, 0xff, 0x00, 0xff, 0xf9, 0x00, 0x3f, 0x53, 0xcc, 0xcc, 0xcc, 0xcc,
        0x6e, 0x00, 0x00, 0x01, 0xf8, 0xff, 0xff, 0xff, 0x49, 0x04, 0x2c, 0x01, 0x00, 0x1d, 0x00, 0x07,
        0x01, 0xff, 0x00, 0x00, 0x00, 0xf8, 0xff, 0x09, 0x00, 0x27, 0x00, 0x08, 0x21, 0x1c, 0x00, 0x00,
        0x00, 0x00, 0x1d, 0x05, 0x00, 0x00, 0x00, 0x2c, 0x53, 0x3f, 0x00, 0x01, 0x00, 0x00, 0xe6, 0xff,
        0xff, 0xff, 0x6a, 0x2b, 0xee, 0xe6, 0x6a, 0x2b, 0xee, 0x2b, 0xee, 0xee, 0x2b, 0xee, 0x00
    ];

    /// The original block-open cases differ only by compression level.
    /// Keep their raw-DEFLATE ABI setup, 38-byte finish windows, and the
    /// exact historical 495-byte fixture in one bounded regression oracle.
    fn compress_block_open(level: c_int) -> Vec<u8> {
        const MAX_WBITS: c_int = 15;
        const FINISH_WINDOW: usize = 38;
        let mut raw = MaybeUninit::<libz_rs_sys::z_stream>::zeroed();

        // SAFETY: writable C stream storage and matching ABI metadata;
        // level is a valid tested compression level, window -15 selects
        // raw DEFLATE, with the original minimal mem_level=1 configuration.
        let init = unsafe {
            deflateInit2_(
                raw.as_mut_ptr(),
                level,
                Z_DEFLATED,
                -MAX_WBITS,
                1,
                Z_FILTERED,
                VERSION,
                test_c_stream_size::<libz_rs_sys::z_stream>(),
            )
        };
        assert_eq!(init, libz_rs_sys::Z_OK);
        // SAFETY: successful initialization wrote the entire C stream.
        let stream = unsafe { raw.assume_init_mut() };

        let mut compressed = [0xa5u8; 1_116];
        stream.next_in = BLOCK_OPEN_INPUT.as_ptr().cast_mut();
        stream.avail_in = test_c_uint_len(BLOCK_OPEN_INPUT.len());

        loop {
            let written = usize::try_from(stream.total_out).unwrap_or(usize::MAX);
            assert!(written < compressed.len(), "compressed buffer exhausted");
            let free = (compressed.len() - written).min(FINISH_WINDOW);
            stream.next_out = compressed[written..].as_mut_ptr();
            stream.avail_out = test_c_uint_len(free);
            let before = (stream.total_in, stream.total_out);
            // SAFETY: the live input is the unread suffix of BLOCK_OPEN_INPUT
            // and the output is the disjoint unwritten portion of compressed.
            let ret = unsafe { deflate(stream, DeflateFlush::Finish as c_int) };
            if ret == libz_rs_sys::Z_STREAM_END {
                break;
            }
            assert_eq!(ret, libz_rs_sys::Z_OK, "finish must make progress");
            assert_ne!(
                (stream.total_in, stream.total_out),
                before,
                "finish stalled without consuming input or producing output"
            );
        }

        let compressed_size = usize::try_from(stream.total_out).unwrap_or(usize::MAX);
        assert!(compressed_size <= compressed.len());
        assert_eq!(stream.avail_in, 0);
        assert_eq!(usize::try_from(stream.total_in), Ok(BLOCK_OPEN_INPUT.len()));
        assert!(
            compressed[compressed_size..].iter().all(|&b| b == 0xa5),
            "raw compressor overwrote unoffered output"
        );
        // SAFETY: the successfully initialized and finished encoder is
        // ended once; all caller-owned input/output buffers remain live.
        assert_eq!(unsafe { deflateEnd(stream) }, libz_rs_sys::Z_OK);

        compressed[..compressed_size].to_vec()
    }

    fn assert_block_open_decodes(compressed: &[u8]) {
        const MAX_WBITS: c_int = 15;
        let mut decoded_raw = MaybeUninit::<libz_rs_sys::z_stream>::zeroed();
        // SAFETY: separate writable stream and matching checked C ABI
        // metadata; -15 selects the same raw wrapper as the encoder.
        let init = unsafe {
            inflateInit2_(
                decoded_raw.as_mut_ptr(),
                -MAX_WBITS,
                VERSION,
                test_c_stream_size::<libz_rs_sys::z_stream>(),
            )
        };
        assert_eq!(init, libz_rs_sys::Z_OK);
        // SAFETY: inflateInit2_ successfully initialized this C stream.
        let decoder = unsafe { decoded_raw.assume_init_mut() };
        decoder.next_in = compressed.as_ptr().cast_mut();
        decoder.avail_in = test_c_uint_len(compressed.len());
        let mut plaintext = [0xa5u8; BLOCK_OPEN_INPUT.len() + 16];
        decoder.next_out = plaintext.as_mut_ptr();
        decoder.avail_out = test_c_uint_len(plaintext.len());
        // SAFETY: encoded data and plaintext are distinct, live, properly
        // sized C buffers; decoder state was initialized successfully.
        assert_eq!(
            // SAFETY: independent live compressed input and writable
            // plaintext arrays, with bounded C byte counts and decoder state.
            unsafe { inflate(decoder, Z_NO_FLUSH) },
            libz_rs_sys::Z_STREAM_END
        );
        assert_eq!(decoder.avail_in, 0);
        assert_eq!(usize::try_from(decoder.total_in), Ok(compressed.len()));
        assert_eq!(
            usize::try_from(decoder.total_out),
            Ok(BLOCK_OPEN_INPUT.len())
        );
        assert_eq!(&plaintext[..BLOCK_OPEN_INPUT.len()], &BLOCK_OPEN_INPUT);
        assert_eq!(&plaintext[BLOCK_OPEN_INPUT.len()..], &[0xa5u8; 16]);
        // SAFETY: the initialized decoder has reached stream end and is
        // closed once before its borrowed allocations leave scope.
        assert_eq!(unsafe { inflateEnd(decoder) }, libz_rs_sys::Z_OK);
    }
    /// Complete C-ABI round trip for the original narrow-window regression.
    fn block_open_round_trip(level: c_int) {
        let compressed = compress_block_open(level);
        assert_block_open_decodes(&compressed);
    }

    #[test]
    fn block_open_quick() {
        block_open_round_trip(1);
    }

    #[test]
    fn block_open_fast() {
        block_open_round_trip(2);
    }

    #[test]
    fn block_open_slow() {
        block_open_round_trip(9);
    }
}

#[test]
fn test_deflate_tune() {
    let input = b"Hello, World!\n";
    let mut raw = MaybeUninit::<libz_rs_sys::z_stream>::zeroed();

    // SAFETY: the uninitialized decoder must reject scalar tuning without
    // reading any stream fields beyond its valid zeroed C storage.
    assert_eq!(
        // SAFETY: zeroed C stream state is rejected without dereference.
        unsafe { libz_rs_sys::deflateTune(raw.as_mut_ptr(), 3, 5, 18, 6) },
        libz_rs_sys::Z_STREAM_ERROR
    );

    // SAFETY: this writable stream is initialized using the matching ABI
    // version and stream size with an ordinary zlib wrapper.
    assert_eq!(
        // SAFETY: correct ABI version/size and live writable stream.
        unsafe {
            libz_rs_sys::deflateInit_(
                raw.as_mut_ptr(),
                libz_rs_sys::Z_BEST_COMPRESSION,
                VERSION,
                test_c_stream_size::<libz_rs_sys::z_stream>(),
            )
        },
        libz_rs_sys::Z_OK
    );
    // SAFETY: successful initialization populated the C stream.
    let stream = unsafe { raw.assume_init_mut() };

    // SAFETY: deflateTune reads a live initialized encoder and scalar
    // settings. No output/input spans are required for this call.
    assert_eq!(
        // SAFETY: fully initialized C stream and scalar settings.
        unsafe { libz_rs_sys::deflateTune(stream, 3, 5, 18, 6) },
        libz_rs_sys::Z_OK
    );

    let mut compressed = vec![0xa5u8; 128];
    let mut finishing = false;
    loop {
        let consumed = usize::try_from(stream.total_in).unwrap_or(usize::MAX);
        let produced = usize::try_from(stream.total_out).unwrap_or(usize::MAX);
        assert!(
            consumed <= input.len(),
            "tuned encoder consumed too much input"
        );
        assert!(
            produced <= compressed.len(),
            "tuned encoder wrote past output"
        );
        if produced == compressed.len() {
            let expanded = compressed.len().checked_mul(2).unwrap_or(0);
            assert!(expanded > compressed.len(), "tuned output size overflow");
            compressed.resize(expanded, 0xa5);
        }

        // Always supply exactly one writable byte, but repoint that byte
        // to the actual unwritten suffix rather than repeatedly offering
        // an already-consumed output location.
        stream.next_out = compressed[produced..].as_mut_ptr();
        stream.avail_out = 1;
        stream.next_in = input[consumed..].as_ptr().cast_mut();
        stream.avail_in = test_c_uint_len(input.len() - consumed);

        finishing |= consumed == input.len();
        let previous = (stream.total_in, stream.total_out);
        let flush = if finishing {
            DeflateFlush::Finish as c_int
        } else {
            Z_NO_FLUSH
        };
        // SAFETY: both current input and one-byte output spans are live,
        // and the initialized encoder does not alias their allocations.
        let status = unsafe { libz_rs_sys::deflate(stream, flush) };
        if status == libz_rs_sys::Z_STREAM_END {
            break;
        }
        assert_eq!(status, libz_rs_sys::Z_OK, "tuned compressor status");
        assert_ne!(
            (stream.total_in, stream.total_out),
            previous,
            "one-byte streaming encoder made no progress"
        );
    }

    assert_eq!(usize::try_from(stream.total_in), Ok(input.len()));
    let written = usize::try_from(stream.total_out).unwrap_or(usize::MAX);
    assert!(written <= compressed.len());
    assert!(compressed[written..].iter().all(|&b| b == 0xa5));
    // SAFETY: initialized encoder has reached Z_STREAM_END and is closed
    // exactly once with the caller-owned output allocation still live.
    assert_eq!(
        // SAFETY: successful initialization paired with one final close.
        unsafe { libz_rs_sys::deflateEnd(stream) },
        libz_rs_sys::Z_OK
    );
    compressed.truncate(written);

    let mut restored = [0xa5u8; 64];
    let (decoded, status) =
        zlib_rs::decompress_slice(&mut restored, &compressed, InflateConfig::default());
    assert_eq!(status, ReturnCode::Ok);
    assert_eq!(decoded, input);
    assert!(restored[input.len()..].iter().all(|&b| b == 0xa5));
}

#[test]
fn deflate_tune_wide_parameters_round_trip_with_zlib_ng() -> Result<(), &'static str> {
    let input: Vec<u8> = (0..8_192usize)
        .map(|i| {
            let offset = i / 157;
            let position = i % 157;
            ((position * 19 + offset % 7 * 13) ^ (position / 5)).to_le_bytes()[0]
        })
        .collect();
    let input_abi_len = c_uint::try_from(input.len()).map_err(|_| "test input exceeds uInt")?;
    let output_abi_len =
        c_uint::try_from(input.len() * 2).map_err(|_| "test output exceeds uInt")?;

    // Bounds above 16 bits must not wrap to zero: the actual DEFLATE match length
    // and reachable hash-chain length are much smaller than these tuning values.
    for params in [
        (8, 16, 128, 129),
        (65_536, 65_536, 65_536, 65_536),
        (65_537, 65_537, 65_537, 65_537),
        (-1, -1, -1, -1),
    ] {
        let compressed = assert_eq_rs_ng!({
            let mut raw = MaybeUninit::<z_stream>::zeroed();
            // SAFETY: valid zlib ABI version and stream size for initialization.
            assert_eq!(
                unsafe {
                    deflateInit_(
                        raw.as_mut_ptr(),
                        9,
                        zlibVersion(),
                        test_c_stream_size::<z_stream>(),
                    )
                },
                Z_OK,
            );
            // SAFETY: successful initialization establishes a live stream.
            let stream = unsafe { raw.assume_init_mut() };
            // SAFETY: deflateTune consumes only scalar parameters and a valid initialized stream.
            let status = unsafe { deflateTune(stream, params.0, params.1, params.2, params.3) };
            let mut output = vec![0u8; input.len() * 2];
            stream.next_in = input.as_ptr().cast_mut();
            stream.avail_in = input_abi_len;
            stream.next_out = output.as_mut_ptr();
            stream.avail_out = output_abi_len;
            // SAFETY: input and output buffers are disjoint, valid, and allocated for the call.
            let code = unsafe { deflate(stream, Z_FINISH) };
            let consumed = stream.avail_in;
            let written = output.len() - stream.avail_out as usize;
            output.truncate(written);
            // SAFETY: stream was initialized once, and is finalized exactly once.
            let end = unsafe { deflateEnd(stream) };
            // Tuned zlib-rs and zlib-ng can choose different, valid DEFLATE matches.
            // Compare ABI outcomes and validate each emitted stream independently.
            let mut recovered = vec![0u8; input.len()];
            let (decoded, rc) =
                zlib_rs::decompress_slice(&mut recovered, &output, InflateConfig::default());
            (
                status,
                code,
                consumed,
                end,
                rc == ReturnCode::Ok && decoded == input,
            )
        });
        assert_eq!(compressed.0, Z_OK, "tune params {params:?}");
        assert_eq!(
            compressed.1,
            libz_rs_sys::Z_STREAM_END,
            "tune params {params:?}"
        );
        assert_eq!(compressed.2, 0, "tune params {params:?}");
        assert_eq!(compressed.3, Z_OK, "tune params {params:?}");
        assert!(
            compressed.4,
            "compression did not round-trip for {params:?}"
        );
    }
    Ok(())
}

#[test]
fn deflate_tune_hash_strategy_after_history_matches_zlib_ng_contract() -> Result<(), &'static str> {
    let input: Vec<u8> = (0..45_000usize)
        .map(|i| ((i % 317 * 7 + i / 997 * 13) ^ (i % 53)).to_le_bytes()[0])
        .collect();
    let output_cap = input.len() * 2;
    let output_abi_len = c_uint::try_from(output_cap).map_err(|_| "test output exceeds uInt")?;

    for (level, chain) in [(7, 4_096), (7, 128), (9, 128), (9, 4_096)] {
        for first_len in [0usize, 12_288] {
            let first_abi_len =
                c_uint::try_from(first_len).map_err(|_| "test split exceeds uInt")?;
            let remaining_abi_len =
                c_uint::try_from(input.len() - first_len).map_err(|_| "test input exceeds uInt")?;

            let outcome = assert_eq_rs_ng!({
                let mut raw = MaybeUninit::<z_stream>::zeroed();
                // SAFETY: use a correctly sized z_stream and the valid zlib version.
                assert_eq!(
                    unsafe {
                        deflateInit_(
                            raw.as_mut_ptr(),
                            level,
                            zlibVersion(),
                            test_c_stream_size::<z_stream>(),
                        )
                    },
                    Z_OK
                );
                // SAFETY: successful initialization owns a live z_stream.
                let stream = unsafe { raw.assume_init_mut() };
                let mut output = vec![0u8; output_cap];
                stream.next_out = output.as_mut_ptr();
                stream.avail_out = output_abi_len;
                let mut first_status = Z_OK;

                if first_len != 0 {
                    stream.next_in = input.as_ptr().cast_mut();
                    stream.avail_in = first_abi_len;
                    // SAFETY: current input and output spans remain live and disjoint.
                    first_status = unsafe { deflate(stream, Z_NO_FLUSH) };
                }
                let first_left = stream.avail_in;
                // SAFETY: stream remains initialized and deflateTune reads only scalar parameters.
                let tune_status = unsafe { deflateTune(stream, 8, 16, 128, chain) };
                stream.next_in = input[first_len..].as_ptr().cast_mut();
                stream.avail_in = remaining_abi_len;
                // SAFETY: both buffers remain live and disjoint until this call completes.
                let finish_status = unsafe { deflate(stream, Z_FINISH) };
                let finish_left = stream.avail_in;
                let produced = output.len() - stream.avail_out as usize;
                output.truncate(produced);
                // SAFETY: initialized stream is finalized exactly once.
                let end_status = unsafe { deflateEnd(stream) };

                let mut decoded_buf = vec![0u8; input.len()];
                let (decoded, decode_status) =
                    zlib_rs::decompress_slice(&mut decoded_buf, &output, InflateConfig::default());
                (
                    first_status,
                    first_left,
                    tune_status,
                    finish_status,
                    finish_left,
                    end_status,
                    decode_status == ReturnCode::Ok && decoded == input,
                )
            });

            assert_eq!(
                outcome.0, Z_OK,
                "level {level}, chain {chain}, split {first_len}"
            );
            assert_eq!(outcome.1, 0, "first input not consumed");
            assert_eq!(outcome.2, Z_OK, "tune rejected");
            assert_eq!(outcome.3, libz_rs_sys::Z_STREAM_END, "finish unsuccessful");
            assert_eq!(outcome.4, 0, "final input not consumed");
            assert_eq!(outcome.5, Z_OK, "stream cleanup failed");
            assert!(outcome.6, "invalid compressed stream");
        }
    }
    Ok(())
}

#[test]
fn deflate_params_preserves_hashed_history_across_levels() -> Result<(), &'static str> {
    let input: Vec<u8> = (0..45_000usize)
        .map(|i| ((i % 509 * 19 + i / 1_019 * 11) ^ (i % 41)).to_le_bytes()[0])
        .collect();
    let first_len = 12_288;
    let input_first_len =
        c_uint::try_from(first_len).map_err(|_| "test split too large for zlib uInt")?;
    let input_rest_len = c_uint::try_from(input.len() - first_len)
        .map_err(|_| "test tail too large for zlib uInt")?;
    let output_cap = input.len() * 2;
    let output_abi_len =
        c_uint::try_from(output_cap).map_err(|_| "test output too large for zlib uInt")?;

    for (old_level, new_level) in [(7, 9), (9, 7), (6, 9)] {
        let outcome = assert_eq_rs_ng!({
            let mut raw = MaybeUninit::<z_stream>::zeroed();
            // SAFETY: the ABI stream version and size correspond to the initialized library.
            assert_eq!(
                unsafe {
                    deflateInit_(
                        raw.as_mut_ptr(),
                        old_level,
                        zlibVersion(),
                        test_c_stream_size::<z_stream>(),
                    )
                },
                Z_OK
            );
            // SAFETY: initialization succeeded and the stream is live.
            let stream = unsafe { raw.assume_init_mut() };
            let mut output = vec![0u8; output_cap];
            stream.next_out = output.as_mut_ptr();
            stream.avail_out = output_abi_len;
            stream.next_in = input.as_ptr().cast_mut();
            stream.avail_in = input_first_len;
            // SAFETY: the caller's input and output spans remain valid and disjoint.
            let first_status = unsafe { deflate(stream, Z_NO_FLUSH) };
            let first_left = stream.avail_in;

            // SAFETY: the stream is still initialized and has a writable output span.
            let params_status =
                unsafe { deflateParams(stream, new_level, libz_rs_sys::Z_DEFAULT_STRATEGY) };
            stream.next_in = input[first_len..].as_ptr().cast_mut();
            stream.avail_in = input_rest_len;
            // SAFETY: the new input span remains live and disjoint from output.
            let finish_status = unsafe { deflate(stream, Z_FINISH) };
            let finish_left = stream.avail_in;
            let produced = output.len() - stream.avail_out as usize;
            output.truncate(produced);
            // SAFETY: the initialized stream is finalized exactly once.
            let end_status = unsafe { deflateEnd(stream) };

            let mut decoded_buf = vec![0u8; input.len()];
            let (decoded, decoded_status) =
                zlib_rs::decompress_slice(&mut decoded_buf, &output, InflateConfig::default());
            (
                first_status,
                first_left,
                params_status,
                finish_status,
                finish_left,
                end_status,
                decoded_status == ReturnCode::Ok && decoded == input,
            )
        });
        assert_eq!(
            outcome.0, Z_OK,
            "first deflate failed for {old_level} -> {new_level}"
        );
        assert_eq!(outcome.1, 0, "first input not consumed");
        assert_eq!(
            outcome.2, Z_OK,
            "deflateParams failed for {old_level} -> {new_level}"
        );
        assert_eq!(
            outcome.3,
            libz_rs_sys::Z_STREAM_END,
            "finish failed for {old_level} -> {new_level}"
        );
        assert_eq!(outcome.4, 0, "tail input not consumed");
        assert_eq!(outcome.5, Z_OK, "end failed");
        assert!(
            outcome.6,
            "stream did not round-trip for {old_level} -> {new_level}"
        );
    }
    Ok(())
}

#[test]
fn deflate_medium_fizzle_bug() {
    const EXPECTED: &[u8] = &[
        120, 156, 99, 96, 128, 3, 73, 6, 26, 3, 71, 218, 2, 28, 182, 214, 17, 225, 50, 85, 100, 14,
        0, 132, 7, 24, 220,
    ];

    const INPUT: &str = "\0\0\0\0\0\0\0\0\0\0\0\u{19}\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0~\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0%\0\0\0\0\0\0\0\0\0\0\0\0";

    let mut output = [0; EXPECTED.len()];

    let config = DeflateConfig::new(6);
    let (output, err) = zlib_rs::deflate::compress_slice(&mut output, INPUT.as_bytes(), config);
    assert_eq!(err, ReturnCode::Ok);

    assert_eq!(output, EXPECTED);
}

fn assert_deflate_bound_correct((config, source_len): (DeflateConfig, c_ulong)) {
    assert_eq_rs_ng!(unsafe {
        let mut strm = MaybeUninit::zeroed();

        // first validate the config
        let err = deflateInit2_(
            strm.as_mut_ptr(),
            config.level,
            config.method as i32,
            config.window_bits,
            config.mem_level,
            config.strategy as i32,
            VERSION,
            test_c_stream_size::<libz_rs_sys::z_stream>(),
        );

        assert_eq!(ReturnCode::from(err), ReturnCode::Ok);

        let bound = deflateBound(strm.as_mut_ptr(), source_len);

        let ret = { deflateEnd(strm.as_mut_ptr()) };
        assert_eq!(ret, Z_OK);

        bound
    });
}

#[test]
#[cfg_attr(miri, ignore = "slow")]
fn deflate_bound_correct() {
    ::quickcheck::quickcheck(test as fn(_) -> _);

    fn test(input: (DeflateConfig, c_ulong)) -> bool {
        assert_deflate_bound_correct(input);

        true
    }
}

#[test]
fn deflate_bound_correct_basic() {
    let config = DeflateConfig::default();

    assert_deflate_bound_correct((config, 42));
    assert_deflate_bound_correct((config, 123456));
    assert_deflate_bound_correct((config, 1 << 24));
}

#[test]
fn deflate_bound_correct_windows() {
    // on windows, c_ulong is just 32 bits wide. That leads to rounding that is different to what
    // we'd get when using usize in rust

    let config = DeflateConfig {
        level: 9,
        method: Method::Deflated,
        window_bits: -13,
        mem_level: 5,
        strategy: Strategy::Filtered,
    };

    // this value is dangerously close to u32::MAX, and the calculation will run into overflow
    //    u32::MAX = 4294967296
    let source_len = 4294967233;

    assert_deflate_bound_correct((config, source_len));

    let config = DeflateConfig {
        level: 0,
        method: Method::Deflated,
        window_bits: 15,
        mem_level: 5,
        strategy: Strategy::HuffmanOnly,
    };
    // this value is dangerously close to u32::MAX, and the calculation will run into overflow
    //    u32::MAX = 4294967296
    let source_len = 4294967289;

    assert_deflate_bound_correct((config, source_len));
}

fn deflate_bound_gzip_header_help(
    (config, source_len, extra, name, comment): (DeflateConfig, c_ulong, CString, CString, CString),
) -> bool {
    let extra_len = extra.as_bytes().len();

    let extra = extra.as_ptr().cast_mut().cast::<u8>();
    let name = name.as_ptr().cast_mut().cast::<u8>();
    let comment = comment.as_ptr().cast_mut().cast::<u8>();

    assert_eq_rs_ng!(unsafe {
        let mut strm = MaybeUninit::zeroed();

        // first validate the config
        let err = deflateInit2_(
            strm.as_mut_ptr(),
            config.level,
            config.method as i32,
            config.window_bits,
            config.mem_level,
            config.strategy as i32,
            zlibVersion(),
            test_c_stream_size::<z_stream>(),
        );

        if err != 0 {
            return true;
        }

        let mut header = gz_header {
            text: 0,
            time: 0,
            xflags: 0,
            os: 0,
            extra_len: extra_len as _,
            extra,
            extra_max: 0,
            name,
            name_max: 0,
            comment,
            comm_max: 0,
            hcrc: 1,
            done: 0,
        };

        // this may fail if the config is not set up for gzip
        let _ = deflateSetHeader(strm.as_mut_ptr(), &raw mut header);

        let bound = deflateBound(strm.as_mut_ptr(), source_len);

        deflateEnd(strm.as_mut_ptr());

        bound
    });

    true
}

#[test]
fn gz_header_text_and_hcrc_check() {
    // when constructing the gz header flags, zlib-rs performed a check
    // for `text > 0` that should have been `text != 0`, similarly for hcrc.
    let config = DeflateConfig {
        level: 9,
        method: Method::Deflated,
        window_bits: 31,
        mem_level: 3,
        strategy: Strategy::Fixed,
    };

    assert_eq_rs_ng!(unsafe {
        let mut strm = MaybeUninit::zeroed();

        // first validate the config
        let err = deflateInit2_(
            strm.as_mut_ptr(),
            config.level,
            config.method as i32,
            config.window_bits,
            config.mem_level,
            config.strategy as i32,
            zlibVersion(),
            test_c_stream_size::<z_stream>(),
        );

        assert_eq!(err, 0);

        let strm = strm.assume_init_mut();
        let empty_c_string = [0u8];

        let mut header = gz_header {
            text: -42,
            time: 0,
            os: 0,
            extra: core::ptr::null_mut(),
            extra_len: 0,
            name: empty_c_string.as_ptr().cast_mut(),
            comment: empty_c_string.as_ptr().cast_mut(),
            hcrc: -42,
            //
            xflags: 0,
            extra_max: 0,
            name_max: 0,
            comm_max: 0,
            done: 0,
        };

        // this may fail if the config is not set up for gzip
        let _ = deflateSetHeader(strm, &raw mut header);

        let input = "\0\u{2}\0\0\0\0\0\0\0\u{10}";
        let mut output = [0u8; 32];

        strm.avail_in = test_c_uint_len(input.len());
        strm.avail_out = test_c_uint_len(output.len());

        strm.next_in = input.as_ptr().cast_mut();
        strm.next_out = output.as_mut_ptr();

        let err = deflate(strm, DeflateFlush::NoFlush as i32);
        assert_eq!(err, 0);

        deflateEnd(strm);

        output
    });
}

#[test]
#[cfg_attr(miri, ignore = "slow")]
fn deflate_bound_gzip_header() {
    ::quickcheck::quickcheck(deflate_bound_gzip_header_help as fn(_) -> _);
}

#[test]
fn deflate_bound_gzip_header_basic() {
    deflate_bound_gzip_header_help((
        DeflateConfig::default(),
        1234,
        CString::from_vec_with_nul(b"extra\0".to_vec()).unwrap(),
        CString::from_vec_with_nul(b"name\0".to_vec()).unwrap(),
        CString::from_vec_with_nul(b"comment\0".to_vec()).unwrap(),
    ));
}

#[test]
#[cfg_attr(
    target_endian = "big",
    ignore = "we don't support DFLTCC, which changes the bounds in zlib-ng"
)]
fn test_compress_bound_windows() {
    let source_len = (u32::MAX - 6) as core::ffi::c_ulong;

    assert_eq_rs_ng!(unsafe { compressBound(source_len as _) }, {
        compressBound(source_len as _)
    });

    assert_eq!(
        libz_rs_sys::compressBound(source_len as _),
        libz_rs_sys::compressBound_z(source_len as _) as c_ulong,
    );
}

#[test]
#[cfg_attr(
    target_endian = "big",
    ignore = "we don't support DFLTCC, which changes the bounds in zlib-ng"
)]
fn test_compress_bound() {
    ::quickcheck::quickcheck(test as fn(_) -> _);

    fn test(source_len: core::ffi::c_ulong) -> bool {
        assert_eq_rs_ng!(unsafe { compressBound(source_len as _) }, {
            compressBound(source_len as _)
        });

        assert_eq!(
            libz_rs_sys::compressBound(source_len as _),
            libz_rs_sys::compressBound_z(source_len as _) as c_ulong,
        );

        true
    }
}

#[test]
#[cfg_attr(
    target_endian = "big",
    ignore = "architecture-specific deflate bounds diverge from zlib-ng"
)]
fn test_compress_bound_extreme_c_abi_matches_zlib_ng() {
    let samples = [
        0,
        9,
        c_ulong::from(u32::MAX - 6),
        c_ulong::MAX / 9,
        c_ulong::MAX - 64,
        c_ulong::MAX,
    ];
    for len in samples {
        // SAFETY: zlib-ng's compressBound takes a by-value integer and accesses no buffers.
        assert_eq_rs_ng!(unsafe { compressBound(len) }, compressBound(len));
        // The C ABI deliberately narrows by wrapping, not by saturating. On Windows x64,
        // c_ulong is 32-bit even though compressBound_z takes a 64-bit usize.
        assert_eq!(
            libz_rs_sys::compressBound(len),
            libz_rs_sys::compressBound_z(len as usize) as c_ulong,
        );
    }
}

#[test]
fn deflate_total_out_counter_wraps_as_c_unsigned_long() {
    assert_eq_rs_ng!({
        let mut raw = MaybeUninit::<z_stream>::zeroed();
        // SAFETY: initialize the zlib stream with the matching ABI struct size and version.
        assert_eq!(
            unsafe {
                deflateInit_(
                    raw.as_mut_ptr(),
                    -1,
                    zlibVersion(),
                    test_c_stream_size::<z_stream>(),
                )
            },
            Z_OK,
        );
        // SAFETY: deflateInit_ succeeded and initialized the stream.
        let stream = unsafe { raw.assume_init_mut() };
        let mut out = [0u8; 128];
        stream.total_out = c_ulong::MAX - 1;
        stream.next_in = core::ptr::null_mut();
        stream.avail_in = 0;
        stream.next_out = out.as_mut_ptr();
        stream.avail_out = 128; // Fixed-size output array.

        // SAFETY: a successful initialized stream and writable output are valid for deflate.
        assert_eq!(unsafe { deflate(stream, Z_FINISH) }, Z_STREAM_END);
        let written = out.len() - stream.avail_out as usize;
        assert!(written >= 8);
        let observed = (stream.total_out, stream.avail_out, out[..written].to_vec());
        // SAFETY: close the initialized stream exactly once.
        assert_eq!(unsafe { deflateEnd(stream) }, Z_OK);
        observed
    });
}

#[test]
fn test_compress_param() {
    const INPUT: &[u8] =
        b"Scheduling and executing async tasks is a job handled by an async runtime, such as\0";
    let config = DeflateConfig::new(2);

    let compressed = assert_eq_rs_ng!({
        let mut uninitialized = MaybeUninit::<z_stream>::zeroed();
        // SAFETY: live writable C stream storage, valid zlib settings,
        // compatible library version and checked ABI structure size.
        let init = unsafe {
            deflateInit2_(
                uninitialized.as_mut_ptr(),
                config.level,
                config.method as c_int,
                config.window_bits,
                config.mem_level,
                config.strategy as c_int,
                zlibVersion(),
                test_c_stream_size::<z_stream>(),
            )
        };
        assert_eq!(init, Z_OK);
        // SAFETY: successful initialization wrote all C stream fields.
        let stream = unsafe { uninitialized.assume_init_mut() };

        let mut output = [0xa5u8; 1_024];
        stream.next_out = output.as_mut_ptr();
        stream.avail_out = test_c_uint_len(output.len());
        let halfway = INPUT.len() / 2;
        stream.next_in = INPUT.as_ptr().cast_mut();
        stream.avail_in = test_c_uint_len(halfway);

        // Complete the initial low-compression segment before changing
        // the level/strategy. Z_OK alone never promises full consumption.
        while stream.avail_in != 0 {
            let before = (stream.total_in, stream.total_out);
            // SAFETY: live readable input and disjoint writable output
            // with C lengths bounded by their actual allocations.
            assert_eq!(unsafe { deflate(stream, Z_NO_FLUSH) }, Z_OK);
            assert_ne!(before, (stream.total_in, stream.total_out));
            assert_ne!(stream.avail_out, 0, "compressed output exhausted");
        }
        assert_eq!(usize::try_from(stream.total_in), Ok(halfway));
        assert_eq!(stream.next_in, INPUT[halfway..].as_ptr().cast_mut());

        // SAFETY: initialized stream; all prior input was consumed and
        // output remains writable, so the level switch can flush safely.
        assert_eq!(
            // SAFETY: all prior input was consumed; both live C pointers
            // refer to valid disjoint buffers with checked capacities.
            unsafe { deflateParams(stream, 8, Strategy::Rle as c_int) },
            Z_OK
        );
        assert_eq!(usize::try_from(stream.total_in), Ok(halfway));

        stream.next_in = INPUT[halfway..].as_ptr().cast_mut();
        stream.avail_in = test_c_uint_len(INPUT.len() - halfway);
        loop {
            let before = (stream.total_in, stream.total_out);
            // SAFETY: the remaining second-half input is readable and
            // the already-advertised output buffer stays live.
            let status = unsafe { deflate(stream, DeflateFlush::Finish as c_int) };
            if status == Z_STREAM_END {
                break;
            }
            assert_eq!(status, Z_OK, "unexpected finish status");
            assert_ne!(before, (stream.total_in, stream.total_out));
            assert_ne!(stream.avail_out, 0, "compressed output exhausted");
        }

        assert_eq!(stream.avail_in, 0);
        assert_eq!(usize::try_from(stream.total_in), Ok(INPUT.len()));
        let produced = usize::try_from(stream.total_out).unwrap_or(usize::MAX);
        assert!(produced <= output.len());
        assert!(output[produced..].iter().all(|&b| b == 0xa5));
        // SAFETY: initialized finished encoder is ended exactly once.
        assert_eq!(unsafe { deflateEnd(stream) }, Z_OK);
        output[..produced].to_vec()
    });

    let mut decoded = vec![0xa5u8; INPUT.len() + 16];
    let (plain, status) =
        zlib_rs::decompress_slice(&mut decoded, &compressed, InflateConfig::default());
    assert_eq!(status, ReturnCode::Ok);
    assert_eq!(plain, INPUT);
    assert!(decoded[INPUT.len()..].iter().all(|&b| b == 0xa5));
}

#[test]
fn test_dict_deflate() {
    const DICTIONARY: &[u8] = b"hello";
    const HELLO: &[u8] = b"hello, hello!\0";
    let config = DeflateConfig::best_compression();

    let (dict_id, compressed) = assert_eq_rs_ng!({
        let mut raw = MaybeUninit::<z_stream>::zeroed();
        // SAFETY: live writable C stream, matching library ABI version,
        // valid zlib wrapper and checked stream structure size.
        let init = unsafe {
            deflateInit2_(
                raw.as_mut_ptr(),
                config.level,
                config.method as c_int,
                config.window_bits,
                config.mem_level,
                config.strategy as c_int,
                zlibVersion(),
                test_c_stream_size::<z_stream>(),
            )
        };
        assert_eq!(init, Z_OK);
        // SAFETY: successful initialization populated the entire C stream.
        let stream = unsafe { raw.assume_init_mut() };
        // SAFETY: the preset dictionary is a live readable span of the
        // exact supplied C length, and stream is initialized.
        assert_eq!(
            // SAFETY: dictionary is a live, readable span, and the encoder
            // is initialized and accepts a preset dictionary now.
            unsafe {
                deflateSetDictionary(
                    stream,
                    DICTIONARY.as_ptr(),
                    test_c_uint_len(DICTIONARY.len()),
                )
            },
            Z_OK
        );
        let dict_id = stream.adler;
        let before_plain = stream.total_in;

        let mut encoded = [0xa5u8; 64];
        stream.next_out = encoded.as_mut_ptr();
        stream.avail_out = test_c_uint_len(encoded.len());
        stream.next_in = HELLO.as_ptr().cast_mut();
        stream.avail_in = test_c_uint_len(HELLO.len());
        // SAFETY: both caller-owned input and writable output spans are
        // live and disjoint, with checked C byte counts.
        assert_eq!(
            // SAFETY: source and writable encoder output are live
            // disjoint spans with their real C byte counts.
            unsafe { deflate(stream, DeflateFlush::Finish as c_int) },
            Z_STREAM_END
        );
        assert_eq!(stream.avail_in, 0);
        assert_eq!(
            stream.total_in.wrapping_sub(before_plain),
            test_c_ulong_len(HELLO.len())
        );
        let used = usize::try_from(stream.total_out).unwrap_or(usize::MAX);
        assert!(used <= encoded.len());
        assert!(encoded[used..].iter().all(|&b| b == 0xa5));
        // SAFETY: the successfully initialized finished encoder is ended once.
        assert_eq!(unsafe { deflateEnd(stream) }, Z_OK);

        let mut decode_raw = MaybeUninit::<z_stream>::zeroed();
        // SAFETY: independent writable C stream with a compatible zlib
        // decoder version and checked structure size.
        let init = unsafe {
            inflateInit_(
                decode_raw.as_mut_ptr(),
                zlibVersion(),
                test_c_stream_size::<z_stream>(),
            )
        };
        assert_eq!(init, Z_OK);
        // SAFETY: successful inflateInit_ populated decoder state.
        let decoder = unsafe { decode_raw.assume_init_mut() };
        decoder.next_in = encoded.as_mut_ptr();
        decoder.avail_in = test_c_uint_len(used);
        let mut plaintext = [0xa5u8; 64];
        decoder.next_out = plaintext.as_mut_ptr();
        decoder.avail_out = test_c_uint_len(plaintext.len());

        // SAFETY: decoder's compressed input and writable output are
        // separate live spans with exactly accounted C lengths.
        assert_eq!(unsafe { inflate(decoder, Z_NO_FLUSH) }, Z_NEED_DICT);
        assert_eq!(decoder.adler, dict_id, "decoder requested wrong dictionary");
        // The zlib header (2 bytes) and dictionary Adler-32 (4 bytes)
        // must be consumed before Z_NEED_DICT. The reference decoder
        // reports total_in=0 here, so inspect the actual C I/O cursor.
        assert_eq!(usize::try_from(decoder.avail_in), Ok(used - 6));
        assert_eq!(decoder.next_in, encoded[6..].as_ptr().cast_mut());
        assert_eq!(decoder.total_out, 0);
        assert!(plaintext.iter().all(|&b| b == 0xa5));
        // SAFETY: dictionary is the exact live span the encoder used,
        // supplied only in the Z_NEED_DICT state.
        assert_eq!(
            // SAFETY: exact dictionary is live and supplied only after
            // the initialized decoder requested it.
            unsafe {
                inflateSetDictionary(
                    decoder,
                    DICTIONARY.as_ptr(),
                    test_c_uint_len(DICTIONARY.len()),
                )
            },
            Z_OK
        );
        // SAFETY: decoder and its input/output allocations remain live;
        // it resumes from the compressed cursor after the zlib header.
        assert_eq!(unsafe { inflate(decoder, Z_NO_FLUSH) }, Z_STREAM_END);
        assert_eq!(decoder.avail_in, 0);
        // inflateSetDictionary can reset the stream's total_in counter.
        // Inspect the actual cursor rather than assuming it remains
        // cumulative across dictionary installation.
        assert_eq!(decoder.next_in, encoded[used..].as_ptr().cast_mut());
        assert_eq!(usize::try_from(decoder.total_out), Ok(HELLO.len()));
        assert_eq!(&plaintext[..HELLO.len()], HELLO);
        assert!(plaintext[HELLO.len()..].iter().all(|&b| b == 0xa5));
        // SAFETY: decoder initialized successfully, ended exactly once.
        assert_eq!(unsafe { inflateEnd(decoder) }, Z_OK);

        (dict_id, encoded[..used].to_vec())
    });

    let expected_adler = zlib_rs::adler32::adler32(1, DICTIONARY);
    assert_eq!(dict_id, c_ulong::from(expected_adler));
    assert_ne!(compressed, [] as [u8; 0]);
}

#[test]
fn deflate_prime_only_inserts_requested_bits() {
    fn compress_after_prime(bits: i32, value: i32) -> Vec<u8> {
        use libz_rs_sys::*;
        let mut raw = MaybeUninit::<z_stream>::zeroed();
        assert_eq!(
            // SAFETY: writable C stream storage, matching checked ABI
            // version/size and supported raw compressor configuration.
            unsafe {
                deflateInit2_(
                    raw.as_mut_ptr(),
                    6,
                    Z_DEFLATED,
                    -15,
                    8,
                    Z_DEFAULT_STRATEGY,
                    zlibVersion(),
                    test_c_stream_size::<z_stream>(),
                )
            },
            Z_OK
        );
        // SAFETY: successful initialization created a live stream.
        let stream = unsafe { raw.assume_init_mut() };
        assert_eq!(
            // SAFETY: the initialized raw stream accepts the requested
            // 0..=31-bit prefix, after which Finish flushes the pending bits.
            unsafe { deflatePrime(stream, bits, value) },
            Z_OK
        );
        let mut output = [0xa5u8; 128];
        stream.next_out = output.as_mut_ptr();
        stream.avail_out = test_c_uint_len(output.len());
        assert_eq!(
            // SAFETY: only the allocated output is writable, with zero
            // compressed input and a live initialized encoder.
            unsafe { deflate(stream, Z_FINISH) },
            Z_STREAM_END
        );
        let len = usize::try_from(stream.total_out).unwrap_or(usize::MAX);
        assert!(len <= output.len());
        assert_eq!(usize::try_from(stream.avail_out), Ok(output.len() - len));
        assert!(output[len..].iter().all(|&byte| byte == 0xa5));
        assert_eq!(
            // SAFETY: this encoder is closed once after StreamEnd.
            unsafe { deflateEnd(stream) },
            Z_OK
        );
        output[..len].to_vec()
    }

    for (bits, high_value) in [
        (0, -1),
        (0, i32::MAX),
        (1, -1),
        (1, 0x7fff),
        (3, -1),
        (5, 0x7fff),
        (7, i32::MIN),
        (9, -1),
        (15, 0x7fff_ffff),
        (17, i32::MIN),
        (31, -1),
    ] {
        let low_bits = high_value.cast_unsigned() & ((1u32 << bits) - 1);
        let low_value = low_bits.cast_signed();
        let full = compress_after_prime(bits, high_value);
        let masked = compress_after_prime(bits, low_value);
        assert_eq!(full, masked, "bits={bits}, high_value={high_value}");
    }
}

#[test]
fn deflate_prime_accepts_32_bits_like_zlib_ng() {
    assert_eq_rs_ng!(unsafe {
        let mut raw = MaybeUninit::<z_stream>::zeroed();
        let init = deflateInit2_(
            raw.as_mut_ptr(),
            6,
            Z_DEFLATED,
            -15,
            8,
            Z_DEFAULT_STRATEGY,
            zlibVersion(),
            test_c_stream_size::<z_stream>(),
        );
        assert_eq!(init, Z_OK);
        let stream = raw.assume_init_mut();

        let prime = deflatePrime(stream, 32, 0x1234_5678);
        let end = deflateEnd(stream);

        (prime, end)
    });
}

/// Prime 32 gzip bits as two low-first 16-bit chunks.
/// Explicit byte extraction preserves the sign bit of CRC32 values.
fn prime_gzip_word(stream: &mut libz_rs_sys::z_stream, value: u32) -> c_int {
    let bytes = value.to_le_bytes();
    let low = i32::from(u16::from_le_bytes([bytes[0], bytes[1]]));
    let high = i32::from(u16::from_le_bytes([bytes[2], bytes[3]]));
    // SAFETY: live initialized raw deflater accepts 16 primed bits.
    let status = unsafe { libz_rs_sys::deflatePrime(stream, 16, low) };
    if status != libz_rs_sys::Z_OK {
        return status;
    }
    // SAFETY: first prime succeeded; the same raw deflater remains live.
    unsafe { libz_rs_sys::deflatePrime(stream, 16, high) }
}

/// Prime the gzip header with the original split 3+5-bit flags.
fn prime_gzip_header(stream: &mut libz_rs_sys::z_stream) {
    for (bits, value) in [(16, 0x8b1f), (8, 8), (3, 0), (5, 0)] {
        assert_eq!(
            // SAFETY: the initialized raw deflater accepts up to 16 bits.
            unsafe { libz_rs_sys::deflatePrime(stream, bits, value) },
            libz_rs_sys::Z_OK
        );
    }
    assert_eq!(prime_gzip_word(stream, 0), libz_rs_sys::Z_OK);
    for (bits, value) in [(8, 0), (8, 255)] {
        assert_eq!(
            // SAFETY: two valid eight-bit gzip header fields.
            unsafe { libz_rs_sys::deflatePrime(stream, bits, value) },
            libz_rs_sys::Z_OK
        );
    }
}

#[test]
fn test_deflate_prime() {
    const HELLO: &[u8] = b"hello, hello!\0";
    let config = DeflateConfig {
        level: -1,
        method: Method::Deflated,
        window_bits: -15,
        mem_level: 8,
        strategy: Strategy::Default,
    };
    let mut encoder_raw = MaybeUninit::<libz_rs_sys::z_stream>::zeroed();
    // SAFETY: writable C stream, matching library ABI and valid raw options.
    let init = unsafe {
        libz_rs_sys::deflateInit2_(
            encoder_raw.as_mut_ptr(),
            config.level,
            config.method as c_int,
            config.window_bits,
            config.mem_level,
            config.strategy as c_int,
            VERSION,
            test_c_stream_size::<libz_rs_sys::z_stream>(),
        )
    };
    assert_eq!(init, libz_rs_sys::Z_OK);
    // SAFETY: successful initialization populated the C encoder stream.
    let encoder = unsafe { encoder_raw.assume_init_mut() };
    prime_gzip_header(encoder);

    encoder.next_in = HELLO.as_ptr().cast_mut();
    encoder.avail_in = test_c_uint_len(HELLO.len());
    let mut compressed = [0xaau8; 64];
    encoder.next_out = compressed.as_mut_ptr();
    encoder.avail_out = test_c_uint_len(compressed.len());
    assert_eq!(
        // SAFETY: live disjoint input/output spans with checked C capacities.
        unsafe { libz_rs_sys::deflate(encoder, DeflateFlush::Finish as c_int) },
        libz_rs_sys::Z_STREAM_END
    );
    assert_eq!(encoder.avail_in, 0);
    assert_eq!(usize::try_from(encoder.total_in), Ok(HELLO.len()));
    let produced = usize::try_from(encoder.total_out).unwrap_or(usize::MAX);
    assert!(produced <= compressed.len());
    assert!(compressed[produced..].iter().all(|&b| b == 0xaa));
    assert_eq!(&compressed[..10], &[0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 255]);

    // CRC and ISIZE are deliberately primed after the raw encoder finished,
    // so they are not flushed to the compressed stream. The gzip decoder
    // should subsequently report that its eight-byte trailer is missing.
    // SAFETY: HELLO is live and its C byte length is checked.
    let crc = unsafe { libz_rs_sys::crc32(0, HELLO.as_ptr(), test_c_uint_len(HELLO.len())) };
    let crc_bits = u32::try_from(crc).unwrap_or_default();
    assert_eq!(c_ulong::from(crc_bits), crc);
    assert_eq!(prime_gzip_word(encoder, crc_bits), libz_rs_sys::Z_OK);
    let len_bits = u32::try_from(HELLO.len()).unwrap_or_default();
    assert_eq!(usize::try_from(len_bits), Ok(HELLO.len()));
    assert_eq!(prime_gzip_word(encoder, len_bits), libz_rs_sys::Z_OK);
    assert_eq!(usize::try_from(encoder.total_out), Ok(produced));
    // SAFETY: the finished encoder is closed once.
    assert_eq!(
        // SAFETY: the initialized encoder is closed exactly once.
        unsafe { libz_rs_sys::deflateEnd(encoder) },
        libz_rs_sys::Z_OK
    );

    let mut decoder_raw = MaybeUninit::<libz_rs_sys::z_stream>::zeroed();
    // SAFETY: distinct writable stream, valid gzip inflate window and ABI.
    let init = unsafe {
        libz_rs_sys::inflateInit2_(
            decoder_raw.as_mut_ptr(),
            15 + 32,
            VERSION,
            test_c_stream_size::<libz_rs_sys::z_stream>(),
        )
    };
    assert_eq!(init, libz_rs_sys::Z_OK);
    // SAFETY: successful initialization populated the decoder.
    let decoder = unsafe { decoder_raw.assume_init_mut() };
    decoder.next_in = compressed.as_mut_ptr();
    decoder.avail_in = test_c_uint_len(produced);
    let mut restored = [0xa5u8; 32];
    decoder.next_out = restored.as_mut_ptr();
    decoder.avail_out = test_c_uint_len(restored.len());

    // SAFETY: live incomplete gzip member and disjoint writable buffer.
    let status = unsafe { libz_rs_sys::inflate(decoder, DeflateFlush::Finish as c_int) };
    assert_eq!(ReturnCode::from(status), ReturnCode::BufError);
    assert_eq!(decoder.avail_in, 0);
    assert_eq!(usize::try_from(decoder.total_in), Ok(produced));
    assert_eq!(usize::try_from(decoder.total_out), Ok(HELLO.len()));
    assert_eq!(&restored[..HELLO.len()], HELLO);
    assert!(restored[HELLO.len()..].iter().all(|&b| b == 0xa5));
    // SAFETY: initialized decoder closed exactly once, even on BufError.
    assert_eq!(
        // SAFETY: the initialized decoder is closed exactly once.
        unsafe { libz_rs_sys::inflateEnd(decoder) },
        libz_rs_sys::Z_OK
    );
}

#[test]
fn dictionary_just_over_window_is_truncated_like_zlib_ng() {
    let dictionary = vec![0x5a; (1 << 15) + 1];

    for initial_total_in in [0, c_ulong::MAX - 1] {
        let result = assert_eq_rs_ng!(unsafe {
            let mut strm = MaybeUninit::<z_stream>::zeroed();
            let init = deflateInit2_(
                strm.as_mut_ptr(),
                6,
                Z_DEFLATED,
                15,
                8,
                Z_DEFAULT_STRATEGY,
                zlibVersion(),
                test_c_stream_size::<z_stream>(),
            );
            let strm = strm.assume_init_mut();
            strm.total_in = initial_total_in;
            let set = deflateSetDictionary(strm, dictionary.as_ptr(), dictionary.len() as _);
            let total_in = strm.total_in;
            let adler = strm.adler;
            let mut dictionary_len = 0;
            let get = deflateGetDictionary(strm, core::ptr::null_mut(), &raw mut dictionary_len);
            let end = deflateEnd(strm);

            (init, set, total_in, adler, get, dictionary_len, end)
        });
        assert_eq!(result.0, Z_OK);
        assert_eq!(result.1, Z_OK);
        assert_eq!(
            result.2,
            initial_total_in.wrapping_add((1 << 15) as c_ulong),
            "only the retained dictionary tail contributes to total_in"
        );
    }
}

#[test]
fn small_window() {
    let deflate_config = DeflateConfig {
        level: Z_BEST_COMPRESSION,
        window_bits: -9,
        ..DeflateConfig::default()
    };
    let plain: [u8; 128] = std::array::from_fn(|i| u8::try_from(i).unwrap_or(u8::MAX));
    let previous_dictionary = vec![b'a'; (1 << 9) - plain.len() / 2];

    let compressed = assert_eq_rs_ng!({
        let mut encoder_raw = MaybeUninit::<z_stream>::zeroed();
        // SAFETY: writable uninitialized stream storage, matching ABI size
        // and version, and supported raw 9-bit DEFLATE window.
        let init = unsafe {
            deflateInit2_(
                encoder_raw.as_mut_ptr(),
                deflate_config.level,
                deflate_config.method as c_int,
                deflate_config.window_bits,
                deflate_config.mem_level,
                deflate_config.strategy as c_int,
                zlibVersion(),
                test_c_stream_size::<z_stream>(),
            )
        };
        assert_eq!(init, Z_OK);
        // SAFETY: successful initialization populated the full C stream.
        let encoder = unsafe { encoder_raw.assume_init_mut() };

        // SAFETY: both consecutive dictionaries are live, readable
        // binary arrays. Replacing history before input is intentional.
        assert_eq!(
            // SAFETY: the first dictionary is readable for its checked C length.
            unsafe {
                deflateSetDictionary(
                    encoder,
                    previous_dictionary.as_ptr(),
                    test_c_uint_len(previous_dictionary.len()),
                )
            },
            Z_OK
        );
        assert_eq!(
            // SAFETY: the second dictionary is readable and must replace history.
            unsafe { deflateSetDictionary(encoder, plain.as_ptr(), test_c_uint_len(plain.len())) },
            Z_OK
        );

        // Dictionary registration contributes bytes to total_in; the
        // subsequent payload's consumption must be measured as a delta.
        let dictionary_total_in = encoder.total_in;
        let mut compressed_output = [0xa5u8; 32];
        encoder.next_in = plain.as_ptr().cast_mut();
        encoder.avail_in = test_c_uint_len(plain.len());
        encoder.next_out = compressed_output.as_mut_ptr();
        encoder.avail_out = test_c_uint_len(compressed_output.len());
        // SAFETY: input and writable 32-byte output are live, disjoint
        // buffers and the initialized raw encoder owns its state.
        assert_eq!(
            // SAFETY: the live input and disjoint, writable 32-byte
            // output span have checked exact C byte lengths.
            unsafe { deflate(encoder, DeflateFlush::Finish as c_int) },
            Z_STREAM_END
        );
        assert_eq!(encoder.avail_in, 0);
        assert_eq!(
            encoder.total_in.wrapping_sub(dictionary_total_in),
            test_c_ulong_len(plain.len()),
            "encoder did not consume exactly the payload after its dictionaries"
        );
        let produced = usize::try_from(encoder.total_out).unwrap_or(usize::MAX);
        assert!(produced <= compressed_output.len());
        assert!(compressed_output[produced..]
            .iter()
            .all(|&byte| byte == 0xa5));
        // SAFETY: this finished encoder is ended once.
        assert_eq!(unsafe { deflateEnd(encoder) }, Z_OK);

        let mut decoder_raw = MaybeUninit::<z_stream>::zeroed();
        // SAFETY: another independent writable C stream for matching
        // raw -9-bit inflate, initialized with checked ABI metadata.
        let init = unsafe {
            inflateInit2_(
                decoder_raw.as_mut_ptr(),
                deflate_config.window_bits,
                zlibVersion(),
                test_c_stream_size::<z_stream>(),
            )
        };
        assert_eq!(init, Z_OK);
        // SAFETY: successful inflateInit2_ populated this decoder stream.
        let decoder = unsafe { decoder_raw.assume_init_mut() };

        // SAFETY: the decoder dictionaries are live and readable, matching
        // exactly the sequence installed on the encoder above.
        assert_eq!(
            // SAFETY: the first history array has a checked C capacity.
            unsafe {
                inflateSetDictionary(
                    decoder,
                    previous_dictionary.as_ptr(),
                    test_c_uint_len(previous_dictionary.len()),
                )
            },
            Z_OK
        );
        assert_eq!(
            // SAFETY: the second history array is live for this C call.
            unsafe { inflateSetDictionary(decoder, plain.as_ptr(), test_c_uint_len(plain.len())) },
            Z_OK
        );

        decoder.next_in = compressed_output.as_mut_ptr();
        decoder.avail_in = test_c_uint_len(produced);
        let mut restored = [0xa5u8; 128 + 16];
        decoder.next_out = restored.as_mut_ptr();
        decoder.avail_out = test_c_uint_len(restored.len());
        // SAFETY: only the produced compressed prefix is offered, and the
        // writable plaintext array is independent of it and C state.
        assert_eq!(unsafe { inflate(decoder, Z_NO_FLUSH) }, Z_STREAM_END);
        assert_eq!(decoder.avail_in, 0);
        assert_eq!(usize::try_from(decoder.total_in), Ok(produced));
        assert_eq!(usize::try_from(decoder.total_out), Ok(plain.len()));
        assert_eq!(&restored[..plain.len()], &plain);
        assert_eq!(&restored[plain.len()..], &[0xa5u8; 16]);
        // SAFETY: decoder initialization is paired with one successful close.
        assert_eq!(unsafe { inflateEnd(decoder) }, Z_OK);

        compressed_output[..produced].to_vec()
    });
    assert!(
        !compressed.is_empty(),
        "small-window encoder emitted no data"
    );
}

#[test]
fn test_deflate_pending() {
    const HELLO: &str = "hello, hello!\0";

    let config = DeflateConfig::default();

    let mut strm = MaybeUninit::zeroed();

    unsafe {
        // first validate the config
        let err = libz_rs_sys::deflateInit2_(
            strm.as_mut_ptr(),
            config.level,
            config.method as i32,
            config.window_bits,
            config.mem_level,
            config.strategy as i32,
            VERSION,
            test_c_stream_size::<libz_rs_sys::z_stream>(),
        );
        assert_eq!(err, 0);

        let strm = strm.assume_init_mut();

        let mut compr = [0; 32];
        strm.next_in = HELLO.as_ptr().cast_mut();
        strm.next_out = compr.as_mut_ptr();

        let it_in = HELLO.as_bytes().chunks(1);
        let mut it_out = compr.chunks(1);

        for (_, _) in it_in.zip(&mut it_out) {
            strm.avail_in = 1;
            strm.avail_out = 1;
            let err = deflate(strm, DeflateFlush::NoFlush as i32);
            assert_eq!(err, 0);
        }

        let mut ped = 0;
        let mut bits = 0;

        let err = libz_rs_sys::deflatePending(strm, std::ptr::null_mut(), std::ptr::null_mut());
        assert_eq!(err, 0);

        let err = libz_rs_sys::deflatePending(strm, &raw mut ped, std::ptr::null_mut());
        assert_eq!(err, 0);

        let err = libz_rs_sys::deflatePending(strm, std::ptr::null_mut(), &raw mut bits);
        assert_eq!(err, 0);

        let err = libz_rs_sys::deflatePending(strm, &raw mut ped, &raw mut bits);
        assert_eq!(err, 0);

        assert!(bits >= 0);
        assert!(bits <= 7);

        /* Finish the stream, still forcing small buffers: */
        for _ in it_out {
            strm.avail_out = 1;
            let err = deflate(strm, libz_rs_sys::Z_FINISH);
            if err == libz_rs_sys::Z_STREAM_END {
                break;
            };
            assert_eq!(err, 0);
        }

        let err = deflateEnd(strm);
        assert_eq!(err, 0);
    }
}

#[test]
fn test_deflate_used() {
    const HELLO: &str = "hello, hello!\0";

    let config = DeflateConfig::default();

    let mut strm = MaybeUninit::zeroed();

    unsafe {
        // a NULL stream is a stream error
        assert_eq!(
            libz_rs_sys::deflateUsed(core::ptr::null_mut(), &mut 0),
            ReturnCode::StreamError as i32,
        );

        let err = libz_rs_sys::deflateInit2_(
            strm.as_mut_ptr(),
            config.level,
            config.method as i32,
            config.window_bits,
            config.mem_level,
            config.strategy as i32,
            VERSION,
            test_c_stream_size::<libz_rs_sys::z_stream>(),
        );
        assert_eq!(err, 0);

        let strm = strm.assume_init_mut();

        // A NULL `bits` pointer is tolerated (and just ignored), still Z_OK
        let err = libz_rs_sys::deflateUsed(strm, std::ptr::null_mut());
        assert_eq!(err, 0);

        // Before any flush to a byte boundary, the value is 0
        let mut bits = -1;
        let err = libz_rs_sys::deflateUsed(strm, &raw mut bits);
        assert_eq!(err, 0);
        assert_eq!(bits, 0);

        let mut compr = [0; 32];
        strm.next_in = HELLO.as_ptr().cast_mut();
        strm.next_out = compr.as_mut_ptr();
        strm.avail_in = test_c_uint_len(HELLO.len());
        strm.avail_out = test_c_uint_len(compr.len());

        let err = deflate(strm, libz_rs_sys::Z_FINISH);
        assert_eq!(err, libz_rs_sys::Z_STREAM_END);

        // after flushing the final block to a byte boundary, the number of bits
        // used in the last byte is in 1..=8.
        let mut bits = -1;
        let err = libz_rs_sys::deflateUsed(strm, &raw mut bits);
        assert_eq!(err, 0);

        // 7 bits is what stock zlib reports.
        assert_eq!(bits, 7);

        let err = deflateEnd(strm);
        assert_eq!(err, 0);
    }
}

/// The stored (level 0) implementation has some custom logic for `bits_used`.
/// Stored-block direct copies must retain exactly the current input call's tail as history.
///
/// A first small input call primes the window. The next large call crosses multiple
/// 65535-byte stored blocks and exercises the fast path that copies caller input directly
/// into output, then restores the recent history from that call's input buffer.
#[test]
fn stored_direct_copy_history_across_input_calls_matches_zlib_ng() {
    let input: Vec<u8> = (0..145_000)
        .map(|i: usize| (i * 73 + i / 251).to_le_bytes()[0])
        .collect();

    for split in [1_024, 32_000, 32_769] {
        let compressed = assert_eq_rs_ng!({
            let mut raw = MaybeUninit::<z_stream>::zeroed();
            // SAFETY: the zlib stream receives the correct ABI version and size.
            assert_eq!(
                // SAFETY: writable stream and checked ABI version/size;
                // level 0 is a valid stored-block compressor setting.
                unsafe {
                    deflateInit_(
                        raw.as_mut_ptr(),
                        0,
                        zlibVersion(),
                        test_c_stream_size::<z_stream>(),
                    )
                },
                Z_OK,
            );
            // SAFETY: initialization succeeded.
            let strm = unsafe { raw.assume_init_mut() };
            let mut output = vec![0xa5u8; input.len() + 2_048];
            strm.next_out = output.as_mut_ptr();
            strm.avail_out = test_c_uint_len(output.len());

            strm.next_in = input.as_ptr().cast_mut();
            strm.avail_in = test_c_uint_len(split);
            // SAFETY: current input and output spans are disjoint and remain allocated.
            assert_eq!(unsafe { deflate(strm, Z_NO_FLUSH) }, Z_OK);
            assert_eq!(strm.avail_in, 0);

            strm.next_in = input[split..].as_ptr().cast_mut();
            strm.avail_in = test_c_uint_len(input.len() - split);
            // SAFETY: updated input and remaining output spans remain disjoint and allocated.
            assert_eq!(unsafe { deflate(strm, Z_FINISH) }, Z_STREAM_END);
            assert_eq!(strm.avail_in, 0);
            assert_eq!(usize::try_from(strm.total_in), Ok(input.len()));
            let written = usize::try_from(strm.total_out).unwrap_or(usize::MAX);
            assert!(written <= output.len());
            assert_eq!(usize::try_from(strm.avail_out), Ok(output.len() - written));
            assert!(output[written..].iter().all(|&b| b == 0xa5));
            output.truncate(written);
            // SAFETY: terminate a successfully initialized stream exactly once.
            assert_eq!(unsafe { deflateEnd(strm) }, Z_OK);
            output
        });
        let mut decoded = vec![0xa5u8; input.len() + 16];
        let (result, code) =
            zlib_rs::decompress_slice(&mut decoded, &compressed, InflateConfig::default());
        assert_eq!(code, ReturnCode::Ok, "split {split}");
        assert_eq!(result, input, "split {split}");
        assert!(decoded[input.len()..].iter().all(|&b| b == 0xa5));
    }
}

#[test]
fn test_deflate_used_stored() {
    let config = DeflateConfig {
        level: 0,
        ..DeflateConfig::default()
    };

    let mut strm = MaybeUninit::zeroed();

    unsafe {
        let err = libz_rs_sys::deflateInit2_(
            strm.as_mut_ptr(),
            config.level,
            config.method as i32,
            config.window_bits,
            config.mem_level,
            config.strategy as i32,
            VERSION,
            test_c_stream_size::<libz_rs_sys::z_stream>(),
        );
        assert_eq!(err, 0);

        let strm = strm.assume_init_mut();

        let input = b"a stored block carries raw bytes\0";
        let mut compr = [0; 128];
        strm.next_in = input.as_ptr().cast_mut();
        strm.next_out = compr.as_mut_ptr();
        strm.avail_in = test_c_uint_len(input.len());
        strm.avail_out = test_c_uint_len(compr.len());

        let err = deflate(strm, libz_rs_sys::Z_FINISH);
        assert_eq!(err, libz_rs_sys::Z_STREAM_END);

        let mut bits = -1;
        let err = libz_rs_sys::deflateUsed(strm, &raw mut bits);
        assert_eq!(err, 0);

        // Always a full byte.
        assert_eq!(bits, 8);

        let err = deflateEnd(strm);
        assert_eq!(err, 0);
    }
}

#[test]
fn test_deflate_used_copy() {
    let config = DeflateConfig::default();

    let mut source = MaybeUninit::zeroed();

    unsafe {
        let err = libz_rs_sys::deflateInit2_(
            source.as_mut_ptr(),
            config.level,
            config.method as i32,
            config.window_bits,
            config.mem_level,
            config.strategy as i32,
            VERSION,
            test_c_stream_size::<libz_rs_sys::z_stream>(),
        );
        assert_eq!(err, 0);

        let source = source.assume_init_mut();

        let input = b"just some string...\0";
        let mut compr = [0; 128];
        source.next_in = input.as_ptr().cast_mut();
        source.next_out = compr.as_mut_ptr();
        source.avail_in = test_c_uint_len(input.len());
        source.avail_out = test_c_uint_len(compr.len());

        let err = deflate(source, libz_rs_sys::Z_FINISH);
        assert_eq!(err, libz_rs_sys::Z_STREAM_END);

        let mut src_bits = -1;
        assert_eq!(libz_rs_sys::deflateUsed(source, &raw mut src_bits), Z_OK);
        assert!((1..=8).contains(&src_bits));

        let mut dest = MaybeUninit::zeroed();
        let err = libz_rs_sys::deflateCopy(dest.as_mut_ptr(), source);
        assert_eq!(err, Z_OK);
        let dest = dest.assume_init_mut();

        let mut dst_bits = -1;
        assert_eq!(libz_rs_sys::deflateUsed(dest, &raw mut dst_bits), Z_OK);
        assert_eq!(dst_bits, src_bits);

        assert_eq!(deflateEnd(source), 0);
        assert_eq!(deflateEnd(dest), 0);
    }
}

/// A full-flush marker must let inflateSync recover from a damaged
/// earlier block without silently returning unrelated or truncated bytes.
#[test]
fn test_flush() {
    const HELLO: &[u8] = b"hello, hello!\0";
    const FIRST_PART: usize = 3;

    let config = DeflateConfig::default();
    let mut raw = MaybeUninit::<libz_rs_sys::z_stream>::zeroed();
    // SAFETY: writable C stream, valid deflate configuration and
    // matching ABI version/structure size.
    let init = unsafe {
        libz_rs_sys::deflateInit2_(
            raw.as_mut_ptr(),
            config.level,
            config.method as c_int,
            config.window_bits,
            config.mem_level,
            config.strategy as c_int,
            VERSION,
            test_c_stream_size::<libz_rs_sys::z_stream>(),
        )
    };
    assert_eq!(init, Z_OK);
    // SAFETY: the successful initializer populated the stream.
    let encoder = unsafe { raw.assume_init_mut() };

    let mut compressed = [0xa5u8; 32];
    encoder.next_in = HELLO.as_ptr().cast_mut();
    encoder.avail_in = test_c_uint_len(FIRST_PART);
    encoder.next_out = compressed.as_mut_ptr();
    encoder.avail_out = test_c_uint_len(compressed.len());

    // SAFETY: all C buffers are live, disjoint and correctly sized.
    assert_eq!(
        // SAFETY: initialized encoder, live disjoint input and output
        // spans with checked byte lengths and valid FullFlush setting.
        unsafe { libz_rs_sys::deflate(encoder, DeflateFlush::FullFlush as c_int) },
        Z_OK
    );
    assert_eq!(encoder.avail_in, 0);
    assert_eq!(usize::try_from(encoder.total_in), Ok(FIRST_PART));
    assert_eq!(encoder.next_in, HELLO[FIRST_PART..].as_ptr().cast_mut());

    // Damage the first compressed block, preserving the full-flush
    // marker that follows it. Require that the byte was actually emitted.
    let first_written = usize::try_from(encoder.total_out).unwrap_or(usize::MAX);
    assert!((4..=compressed.len()).contains(&first_written));
    compressed[3] = compressed[3].wrapping_add(1);

    encoder.avail_in = test_c_uint_len(HELLO.len() - FIRST_PART);
    // SAFETY: the remaining payload starts at the C next_in cursor and
    // the existing C output cursor still points into compressed.
    assert_eq!(
        // SAFETY: remaining input and output cursors are still
        // live and in bounds after the first completed flush call.
        unsafe { libz_rs_sys::deflate(encoder, DeflateFlush::Finish as c_int) },
        libz_rs_sys::Z_STREAM_END
    );
    assert_eq!(encoder.avail_in, 0);
    assert_eq!(usize::try_from(encoder.total_in), Ok(HELLO.len()));
    let produced = usize::try_from(encoder.total_out).unwrap_or(usize::MAX);
    assert!(produced <= compressed.len());
    assert!(compressed[produced..].iter().all(|&b| b == 0xa5));
    // SAFETY: initialized, finished encoder closed once.
    assert_eq!(unsafe { libz_rs_sys::deflateEnd(encoder) }, Z_OK);

    test_sync(&compressed[..produced], &HELLO[FIRST_PART..]);
}

/// Decode from the zlib header, skip a corrupted block at the next
/// full-flush marker, and check the exact recovered plaintext.
fn test_sync(compressed: &[u8], expected_suffix: &[u8]) {
    let mut raw = MaybeUninit::<libz_rs_sys::z_stream>::zeroed();
    // SAFETY: live writable decoder storage and matching ABI/version.
    let init = unsafe {
        libz_rs_sys::inflateInit2_(
            raw.as_mut_ptr(),
            InflateConfig::default().window_bits,
            VERSION,
            test_c_stream_size::<libz_rs_sys::z_stream>(),
        )
    };
    assert_eq!(init, Z_OK);
    // SAFETY: successful initialization populated the C decoder stream.
    let decoder = unsafe { raw.assume_init_mut() };

    assert!(compressed.len() >= 2, "missing zlib header");
    decoder.next_in = compressed.as_ptr().cast_mut();
    decoder.avail_in = 2;
    let mut restored = [0xa5u8; 32];
    decoder.next_out = restored.as_mut_ptr();
    decoder.avail_out = test_c_uint_len(restored.len());

    // SAFETY: live two-byte header and independent writable output.
    assert_eq!(unsafe { libz_rs_sys::inflate(decoder, Z_NO_FLUSH) }, Z_OK);
    assert_eq!(decoder.avail_in, 0);
    assert_eq!(decoder.next_in, compressed[2..].as_ptr().cast_mut());
    decoder.avail_in = test_c_uint_len(compressed.len() - 2);
    // SAFETY: the unread compressed suffix is valid; inflateSync scans
    // for the next stored full-flush marker, skipping damaged data.
    assert_eq!(unsafe { libz_rs_sys::inflateSync(decoder) }, Z_OK);
    // SAFETY: the C input cursor now points into the remaining
    // compressed input, and output is still live and writable.
    assert_eq!(
        // SAFETY: initialized decoder, valid remaining compressed input
        // following the sync marker, and disjoint writable plaintext.
        unsafe { libz_rs_sys::inflate(decoder, DeflateFlush::Finish as c_int) },
        libz_rs_sys::Z_STREAM_END
    );
    assert_eq!(decoder.avail_in, 0);
    let written = usize::try_from(decoder.total_out).unwrap_or(usize::MAX);
    assert_eq!(written, expected_suffix.len());
    assert_eq!(&restored[..written], expected_suffix);
    assert!(restored[written..].iter().all(|&b| b == 0xa5));
    // SAFETY: the initialized decoder is closed once.
    assert_eq!(unsafe { libz_rs_sys::inflateEnd(decoder) }, Z_OK);
}

#[test]
fn test_deflate_hash_head_0() {
    use libz_rs_sys::{
        deflate, deflateEnd, deflateInit2_, deflateParams, Z_DEFLATED, Z_FILTERED, Z_FINISH,
        Z_FULL_FLUSH, Z_HUFFMAN_ONLY, Z_OK, Z_STREAM_END, Z_SYNC_FLUSH,
    };

    let mut original = [0x30u8; 9_698];
    original[8_193] = 0;
    original[8_194] = 0;
    original[8_195] = 0;
    original[8_199] = 0x8a;
    let mut compressed = vec![0xa5u8; 21_572];

    let mut raw = MaybeUninit::<libz_rs_sys::z_stream>::zeroed();
    // SAFETY: initialized storage and matching C ABI metadata, with
    // supported raw-DEFLATE, Huffman-only settings.
    let init = unsafe {
        deflateInit2_(
            raw.as_mut_ptr(),
            1,
            Z_DEFLATED,
            -15,
            4,
            Z_HUFFMAN_ONLY,
            VERSION,
            test_c_stream_size::<libz_rs_sys::z_stream>(),
        )
    };
    assert_eq!(init, Z_OK);
    // SAFETY: successful initialization populated the C stream.
    let stream = unsafe { raw.assume_init_mut() };
    stream.next_in = original.as_mut_ptr();
    stream.next_out = compressed.as_mut_ptr();

    stream.avail_in = 0;
    stream.avail_out = 1_348;
    // SAFETY: compressor state is live and a writable output suffix
    // with 1348 bytes is actually allocated.
    assert_eq!(unsafe { deflateParams(stream, 3, Z_FILTERED) }, Z_OK);

    stream.avail_in = 6_728;
    stream.avail_out = 2_696;
    // SAFETY: both source and destination are live arrays; their remaining
    // spans are sufficient for these explicitly bounded byte counts.
    assert_eq!(unsafe { deflate(stream, Z_SYNC_FLUSH) }, Z_OK);

    stream.avail_in = 15;
    stream.avail_out = 1_348;
    // SAFETY: the prior call left valid byte cursors inside both arrays.
    assert_eq!(unsafe { deflateParams(stream, 9, Z_FILTERED) }, Z_OK);

    stream.avail_in = 1_453;
    stream.avail_out = 1_348;
    // SAFETY: the previous compressor input/output cursors stay live.
    assert_eq!(unsafe { deflate(stream, Z_FULL_FLUSH) }, Z_OK);

    // C cursors must stay inside the actual arrays. Never derive a fresh
    // avail count from unchecked pointer subtraction or assume all the
    // previously offered bytes were consumed by a successful call.
    let input_offset = (stream.next_in as usize)
        .checked_sub(original.as_ptr() as usize)
        .unwrap_or(usize::MAX);
    let output_offset = (stream.next_out as usize)
        .checked_sub(compressed.as_ptr() as usize)
        .unwrap_or(usize::MAX);
    assert!(input_offset <= original.len());
    assert!(output_offset <= compressed.len());
    stream.avail_in = test_c_uint_len(original.len() - input_offset);
    stream.avail_out = test_c_uint_len(compressed.len() - output_offset);
    // SAFETY: after the cursor checks, each C field describes exactly
    // the valid unread input and writable unfilled output suffix.
    assert_eq!(unsafe { deflate(stream, Z_FINISH) }, Z_STREAM_END);
    assert_eq!(stream.avail_in, 0);
    assert_eq!(usize::try_from(stream.total_in), Ok(original.len()));
    let produced = usize::try_from(stream.total_out).unwrap_or(usize::MAX);
    assert!(produced <= compressed.len());
    assert!(compressed[produced..].iter().all(|&byte| byte == 0xa5));
    // SAFETY: this initialized and finished encoder is closed exactly once.
    assert_eq!(unsafe { deflateEnd(stream) }, Z_OK);

    let mut inflate_raw = MaybeUninit::<libz_rs_sys::z_stream>::zeroed();
    // SAFETY: a separate writable C stream is initialized in raw mode
    // with the exact size and valid version for this implementation.
    assert_eq!(
        // SAFETY: the allocated C stream is writable and this raw inflate
        // mode uses the correct ABI version and checked structure size.
        unsafe {
            inflateInit2_(
                inflate_raw.as_mut_ptr(),
                -15,
                VERSION,
                test_c_stream_size::<libz_rs_sys::z_stream>(),
            )
        },
        Z_OK
    );
    // SAFETY: initialization successfully populated the decoder C stream.
    let decoder = unsafe { inflate_raw.assume_init_mut() };
    decoder.next_in = compressed.as_mut_ptr();
    decoder.avail_in = test_c_uint_len(produced);
    let mut restored = vec![0xa5u8; original.len() + 16];
    decoder.next_out = restored.as_mut_ptr();
    decoder.avail_out = test_c_uint_len(restored.len());
    // SAFETY: decoder input is a live compressed prefix and its writable
    // output buffer is distinct and offered with its exact C byte length.
    assert_eq!(unsafe { inflate(decoder, Z_NO_FLUSH) }, Z_STREAM_END);
    assert_eq!(decoder.avail_in, 0);
    assert_eq!(usize::try_from(decoder.total_in), Ok(produced));
    assert_eq!(usize::try_from(decoder.total_out), Ok(original.len()));
    assert_eq!(&restored[..original.len()], &original);
    assert!(restored[original.len()..].iter().all(|&byte| byte == 0xa5));
    // SAFETY: the initialized decoder is closed exactly once.
    assert_eq!(unsafe { inflateEnd(decoder) }, Z_OK);
}

#[test]
fn test_deflate_copy_small() {
    test_deflate_copy_help(b"hello, hello!\0");
}

#[test]
#[cfg_attr(miri, ignore = "too slow")]
fn test_deflate_copy_large() {
    test_deflate_copy_help(LCET10.as_bytes());
}

/// A copied encoder must yield the same complete compressed bytes as the
/// original encoder and both streams must decode to the original plaintext.
/// The copy is made after half the data and a sync flush.
fn test_deflate_copy_help(input: &[u8]) {
    let config = DeflateConfig::default();
    let compressed = assert_eq_rs_ng!({
        // This function is instantiated independently for the two C ABIs
        // by the differential test macro.
        fn feed_until(stream: &mut z_stream, input: &[u8], position: &mut usize, stop: usize) {
            assert!(stop <= input.len());
            while *position < stop {
                stream.next_in = input[*position..stop].as_ptr().cast_mut();
                stream.avail_in = test_c_uint_len(stop - *position);
                let before = stream.total_in;
                // SAFETY: this is a live initialized C deflater. Its input is
                // the readable suffix and the caller keeps writable output
                // allocated and correctly advertised for the whole call.
                let status = unsafe { deflate(stream, Z_NO_FLUSH) };
                assert_eq!(status, Z_OK);
                assert_ne!(stream.avail_out, 0, "output exhausted during input");
                let consumed = usize::try_from(stream.total_in - before).unwrap_or(usize::MAX);
                assert!(
                    (1..=stop - *position).contains(&consumed),
                    "input did not advance within the offered span"
                );
                *position += consumed;
            }
            assert_eq!(stream.avail_in, 0);
        }

        fn flush_sync(stream: &mut z_stream) {
            stream.next_in = core::ptr::null_mut();
            stream.avail_in = 0;
            // SAFETY: an initialized deflater has live state and the
            // remaining output suffix is writable and within its allocation.
            assert_eq!(unsafe { deflate(stream, Z_SYNC_FLUSH) }, Z_OK);
            assert_ne!(stream.avail_out, 0, "sync flush exhausted output");
        }

        fn finish(stream: &mut z_stream) -> usize {
            loop {
                // SAFETY: no input remains, the encoder and its output
                // allocation stay alive and the output suffix is writable.
                let status = unsafe { deflate(stream, Z_FINISH) };
                if status == Z_STREAM_END {
                    break;
                }
                assert_eq!(status, Z_OK, "unexpected finish status");
                assert_ne!(stream.avail_out, 0, "finish exhausted output");
            }
            usize::try_from(stream.total_out).unwrap_or(usize::MAX)
        }

        let capacity = input
            .len()
            .checked_mul(2)
            .and_then(|n| n.checked_add(64))
            .unwrap_or(0);
        assert!(capacity >= input.len());
        let half = input.len() / 2;

        let mut first_raw = MaybeUninit::<z_stream>::zeroed();
        // SAFETY: writable stream storage and checked ABI metadata are
        // passed with a valid standard compressor configuration.
        assert_eq!(
            unsafe {
                deflateInit2_(
                    first_raw.as_mut_ptr(),
                    config.level,
                    config.method as c_int,
                    config.window_bits,
                    config.mem_level,
                    config.strategy as c_int,
                    zlibVersion(),
                    test_c_stream_size::<z_stream>(),
                )
            },
            Z_OK
        );
        // SAFETY: successful initialization populated the whole C stream.
        let first = unsafe { first_raw.assume_init_mut() };
        let mut full_output = vec![0xa5u8; capacity];
        first.next_out = full_output.as_mut_ptr();
        first.avail_out = test_c_uint_len(full_output.len());
        let mut position = 0;
        feed_until(first, input, &mut position, half);
        flush_sync(first);
        feed_until(first, input, &mut position, input.len());
        flush_sync(first);
        let full_size = finish(first);
        assert_eq!(usize::try_from(first.total_in), Ok(input.len()));
        assert!(full_size <= full_output.len());
        assert!(full_output[full_size..].iter().all(|&byte| byte == 0xa5));
        // SAFETY: the fully finished stream is ended exactly once.
        assert_eq!(unsafe { deflateEnd(first) }, Z_OK);
        full_output.truncate(full_size);

        let mut source_raw = MaybeUninit::<z_stream>::zeroed();
        // SAFETY: a separate writable C stream and same valid configuration.
        assert_eq!(
            unsafe {
                deflateInit2_(
                    source_raw.as_mut_ptr(),
                    config.level,
                    config.method as c_int,
                    config.window_bits,
                    config.mem_level,
                    config.strategy as c_int,
                    zlibVersion(),
                    test_c_stream_size::<z_stream>(),
                )
            },
            Z_OK
        );
        // SAFETY: successful initialization populated the source stream.
        let source = unsafe { source_raw.assume_init_mut() };
        let mut copy_output = vec![0xa5u8; capacity];
        source.next_out = copy_output.as_mut_ptr();
        source.avail_out = test_c_uint_len(copy_output.len());
        position = 0;
        feed_until(source, input, &mut position, half);
        flush_sync(source);

        let mut copy_raw = MaybeUninit::<z_stream>::uninit();
        // SAFETY: source is initialized; destination is distinct writable
        // uninitialized stream storage. The output pointer refers to the
        // shared, still-live copied stream's output allocation.
        assert_eq!(unsafe { deflateCopy(copy_raw.as_mut_ptr(), source) }, Z_OK);
        // SAFETY: the source stream is no longer needed after deflateCopy;
        // zlib reports DataError when ending this unfinished stream.
        assert_eq!(unsafe { deflateEnd(source) }, Z_DATA_ERROR);
        // SAFETY: a successful deflateCopy initialized the entire copy.
        let copy = unsafe { copy_raw.assume_init_mut() };

        feed_until(copy, input, &mut position, input.len());
        flush_sync(copy);
        let copy_size = finish(copy);
        assert_eq!(usize::try_from(copy.total_in), Ok(input.len()));
        assert!(copy_size <= copy_output.len());
        assert!(copy_output[copy_size..].iter().all(|&byte| byte == 0xa5));
        // SAFETY: only the source was ended; the copied stream still owns
        // its state and is independently ended once after StreamEnd.
        assert_eq!(unsafe { deflateEnd(copy) }, Z_OK);
        copy_output.truncate(copy_size);
        assert_eq!(full_output, copy_output, "copied encoder output diverged");
        full_output
    });

    // Identical outputs alone are not an independent correctness oracle:
    // verify exact round-trip bytes, including the end of the input.
    let mut decoded = vec![0xa5u8; input.len().checked_add(32).unwrap_or(0)];
    assert!(decoded.len() >= input.len());
    let (plain, status) =
        zlib_rs::decompress_slice(&mut decoded, &compressed, InflateConfig::default());
    assert_eq!(status, ReturnCode::Ok);
    assert_eq!(plain, input);
    assert!(decoded[input.len()..].iter().all(|&byte| byte == 0xa5));
}

#[test]
fn version_error() {
    use libz_rs_sys::{
        deflateInit_, z_stream, zlibVersion, Z_DEFAULT_STRATEGY, Z_OK, Z_VERSION_ERROR,
    };

    let mut raw = MaybeUninit::<z_stream>::zeroed();
    // SAFETY: writable C stream storage, matching version and checked
    // structure size; initialization is required before later teardown.
    let init = unsafe {
        deflateInit_(
            raw.as_mut_ptr(),
            1,
            zlibVersion(),
            test_c_stream_size::<z_stream>(),
        )
    };
    assert_eq!(init, Z_OK);
    // SAFETY: successful deflateInit_ initialized every C stream field.
    let stream = unsafe { raw.assume_init_mut() };
    // SAFETY: the initialized stream is closed once before negative cases.
    assert_eq!(unsafe { deflateEnd(stream) }, Z_OK);

    // The following negative cases use independently valid options, except
    // for the one contract field intentionally being tested.
    // SAFETY: the live writable stream and valid library version are
    // supplied; the wrong stream-size argument must be rejected.
    assert_eq!(
        // SAFETY: writable stream storage and valid live version string;
        // deliberately wrong stream size must be rejected before use.
        unsafe { deflateInit_(stream, 1, zlibVersion(), 1) },
        Z_VERSION_ERROR
    );
    // SAFETY: the live stream and checked ABI size are valid, but the
    // missing version pointer must be rejected before dereference.
    assert_eq!(
        // SAFETY: stream is writable and size valid; the intentionally
        // null version must be rejected without dereferencing it.
        unsafe {
            deflateInit_(
                stream,
                1,
                core::ptr::null(),
                test_c_stream_size::<z_stream>(),
            )
        },
        Z_VERSION_ERROR
    );

    // SAFETY: c"!" contains a live NUL-terminated incompatible version
    // string; every other argument uses a legitimate encoder configuration.
    assert_eq!(
        // SAFETY: incompatible version points to a live NUL-terminated
        // string; all other parameters and storage are valid.
        unsafe { deflateInit_(stream, 1, c"!".as_ptr(), test_c_stream_size::<z_stream>()) },
        Z_VERSION_ERROR
    );
    // SAFETY: the invalid version is the *only* invalid deflateInit2
    // parameter: method, window size, memory level, strategy and C ABI
    // stream size all describe a normal zlib encoder.
    assert_eq!(
        // SAFETY: only the live version string is incompatible; the
        // writable stream and all compressor settings have valid values.
        unsafe {
            deflateInit2_(
                stream,
                1,
                Z_DEFLATED,
                15,
                8,
                Z_DEFAULT_STRATEGY,
                c"!".as_ptr(),
                test_c_stream_size::<z_stream>(),
            )
        },
        Z_VERSION_ERROR
    );

    // Rejected version/size checks must not prevent reinitializing the
    // same caller-owned stream with valid parameters.
    // SAFETY: the stream storage remains live and its C ABI options are
    // all valid again.
    assert_eq!(
        // SAFETY: all invalid attempts have returned; storage remains
        // live and matching library metadata permits fresh initialization.
        unsafe { deflateInit_(stream, 1, zlibVersion(), test_c_stream_size::<z_stream>()) },
        Z_OK
    );
    // SAFETY: this successfully reinitialized stream is ended once.
    assert_eq!(unsafe { deflateEnd(stream) }, Z_OK);
}

#[test]
fn gzip_header_survives_deflate_resets_like_zlib_ng() {
    for reset_keep in [false, true] {
        assert_eq_rs_ng!(unsafe {
            let mut raw = MaybeUninit::<z_stream>::zeroed();
            let init = deflateInit2_(
                raw.as_mut_ptr(),
                6,
                Z_DEFLATED,
                31,
                8,
                Z_DEFAULT_STRATEGY,
                zlibVersion(),
                test_c_stream_size::<z_stream>(),
            );
            assert_eq!(init, Z_OK);
            let stream = raw.assume_init_mut();

            let mut name = *b"survives-reset\0";
            let mut header = gz_header {
                text: 1,
                time: 0x1234_5678,
                xflags: 0,
                os: 3,
                extra: core::ptr::null_mut(),
                extra_len: 0,
                extra_max: 0,
                name: name.as_mut_ptr(),
                name_max: 0,
                comment: core::ptr::null_mut(),
                comm_max: 0,
                hcrc: 1,
                done: 0,
            };
            assert_eq!(deflateSetHeader(stream, &raw mut header), Z_OK);

            let reset = if reset_keep {
                deflateResetKeep(stream)
            } else {
                deflateReset(stream)
            };
            assert_eq!(reset, Z_OK);

            let input = b"payload after reset";
            let mut output = [0u8; 256];
            stream.next_in = input.as_ptr().cast_mut();
            stream.avail_in = test_c_uint_len(input.len());
            stream.next_out = output.as_mut_ptr();
            stream.avail_out = test_c_uint_len(output.len());
            assert_eq!(deflate(stream, Z_FINISH), Z_STREAM_END);

            let len = stream.total_out as usize;
            let result = output[..len].to_vec();
            assert_eq!(deflateEnd(stream), Z_OK);
            result
        });
    }
}

#[test]
fn gzip_with_header() {
    let extra = b"some extra stuff\0";
    let name = b"nomen est omen\0";
    let comment = b"such comment\0";
    let original = b"Hello World\n";

    let config = DeflateConfig {
        window_bits: 31,
        ..Default::default()
    };

    let compressed = assert_eq_rs_ng!({
        let mut raw = MaybeUninit::<z_stream>::zeroed();
        // SAFETY: writable C stream, matching ABI metadata, and valid gzip
        // encoder options remain live across initialization.
        let status = unsafe {
            deflateInit2_(
                raw.as_mut_ptr(),
                config.level,
                config.method as i32,
                config.window_bits,
                config.mem_level,
                config.strategy as i32,
                zlibVersion(),
                test_c_stream_size::<z_stream>(),
            )
        };
        assert_eq!(status, Z_OK);
        // SAFETY: successful deflateInit2_ populated the entire C stream.
        let stream = unsafe { raw.assume_init_mut() };

        let mut header = gz_header {
            text: 0,
            time: 0,
            xflags: 0,
            os: 0,
            extra: extra.as_ptr().cast_mut(),
            extra_len: test_c_uint_len(extra.len()),
            extra_max: 0,
            name: name.as_ptr().cast_mut(),
            name_max: 0,
            comment: comment.as_ptr().cast_mut(),
            comm_max: 0,
            hcrc: 1,
            done: 0,
        };
        // SAFETY: the header and all its live, immutable metadata input
        // buffers remain accessible until deflateEnd.
        assert_eq!(unsafe { deflateSetHeader(stream, &raw mut header) }, Z_OK);

        let mut compressed_storage = [0xa5u8; 256];
        stream.next_in = original.as_ptr().cast_mut();
        stream.avail_in = test_c_uint_len(original.len());
        stream.next_out = compressed_storage.as_mut_ptr();
        stream.avail_out = test_c_uint_len(compressed_storage.len());
        // SAFETY: read-only original and independent writable output spans
        // have exact C byte counts and do not alias decoder state.
        assert_eq!(
            unsafe { deflate(stream, DeflateFlush::Finish as _) },
            ReturnCode::StreamEnd as c_int
        );
        assert_eq!(stream.avail_in, 0);
        assert_eq!(usize::try_from(stream.total_in), Ok(original.len()));
        let produced = usize::try_from(stream.total_out).unwrap_or(usize::MAX);
        assert!(produced <= compressed_storage.len());
        assert_eq!(produced, 81, "unexpected gzip member size");
        assert_eq!(
            &compressed_storage[produced..],
            &[0xa5u8; 256][produced..],
            "compressor wrote past produced byte count"
        );
        // SAFETY: the initialized encoder is ended exactly once, while the
        // borrowed gzip header metadata are still alive.
        assert_eq!(unsafe { deflateEnd(stream) }, Z_OK);
        compressed_storage[..produced].to_vec()
    });

    assert_eq_rs_ng!({
        let mut raw = MaybeUninit::<z_stream>::zeroed();
        // SAFETY: writable uninitialized C stream with valid gzip window
        // option, matching ABI version and checked structure size.
        let status = unsafe {
            inflateInit2_(
                raw.as_mut_ptr(),
                config.window_bits,
                zlibVersion(),
                test_c_stream_size::<z_stream>(),
            )
        };
        assert_eq!(status, Z_OK);
        // SAFETY: successful initialization populated the complete C stream.
        let stream = unsafe { raw.assume_init_mut() };

        let mut decoded = [0xa5u8; 32];
        let mut extra_buf = [0xa5u8; 64];
        let mut name_buf = [0xa5u8; 64];
        let mut comment_buf = [0xa5u8; 64];

        let mut header = gz_header {
            text: 0,
            time: 0,
            xflags: 0,
            os: 0,
            extra: extra_buf.as_mut_ptr(),
            extra_len: 0,
            extra_max: test_c_uint_len(extra_buf.len()),
            name: name_buf.as_mut_ptr(),
            name_max: test_c_uint_len(name_buf.len()),
            comment: comment_buf.as_mut_ptr(),
            comm_max: test_c_uint_len(comment_buf.len()),
            hcrc: 0,
            done: 0,
        };
        // SAFETY: all metadata output buffers are live writable spans with
        // matching C capacities and are disjoint from decoder state.
        assert_eq!(unsafe { inflateGetHeader(stream, &raw mut header) }, Z_OK);

        stream.next_in = compressed.as_ptr().cast_mut();
        stream.avail_in = test_c_uint_len(compressed.len());
        stream.next_out = decoded.as_mut_ptr();
        stream.avail_out = test_c_uint_len(decoded.len());

        // SAFETY: the complete compressed member and writable output are
        // separate live allocations described by actual C widths.
        assert_eq!(unsafe { inflate(stream, Z_NO_FLUSH) }, Z_STREAM_END);
        assert_eq!(stream.avail_in, 0);
        assert_eq!(usize::try_from(stream.total_in), Ok(compressed.len()));
        assert_eq!(usize::try_from(stream.total_out), Ok(original.len()));
        assert_eq!(&decoded[..original.len()], original);
        assert_eq!(&decoded[original.len()..], &[0xa5u8; 32][original.len()..]);
        assert_eq!(header.done, 1);
        assert_eq!(header.extra_len, test_c_uint_len(extra.len()));
        assert_eq!(&extra_buf[..extra.len()], extra);
        assert_eq!(&name_buf[..name.len()], name);
        assert_eq!(&comment_buf[..comment.len()], comment);

        // SAFETY: the initialized decoder is closed exactly once after
        // verifying all metadata pointers and output counters.
        assert_eq!(unsafe { inflateEnd(stream) }, Z_OK);
    });
}

mod fuzz_based_tests {
    use super::{FIREWORKS, LCET10, PAPER_100K};
    use crate::helpers::compress_slice_ng;
    use libz_rs_sys::gz_header;
    use zlib_rs::{
        deflate::{compress_slice, DeflateConfig, Method, Strategy},
        ReturnCode,
    };

    fn fuzz_based_test(input: &[u8], config: DeflateConfig, expected: &[u8]) {
        let mut output_rs = vec![0; 1 << 19];
        let (output_rs, err) = compress_slice(&mut output_rs, input, config);
        assert_eq!(err, ReturnCode::Ok);

        if !cfg!(miri) {
            let mut output_ng = vec![0; 1 << 19];
            let (output_ng, err) = compress_slice_ng(&mut output_ng, input, config);
            assert_eq!(err, ReturnCode::Ok);

            assert_eq!(output_rs, output_ng);
        }

        if !expected.is_empty() {
            assert_eq!(output_rs, expected);
        }
    }

    #[cfg(not(miri))]
    quickcheck::quickcheck! {
        fn rs_is_ng(bytes: Vec<u8>) -> bool {
            fuzz_based_test(&bytes, DeflateConfig::default(), &[]);

            true
        }
    }

    #[test]
    #[cfg_attr(miri, ignore = "too slow")]
    fn compress_lcet10() {
        fuzz_based_test(LCET10.as_bytes(), DeflateConfig::default(), &[])
    }

    #[test]
    #[cfg_attr(miri, ignore = "too slow")]
    #[cfg_attr(
        target_family = "wasm",
        ignore = "zlib-ng compresses differently on wasm"
    )]
    fn compress_paper_100k() {
        let mut config = DeflateConfig::default();

        for n in 0..=9 {
            config.level = n;
            fuzz_based_test(PAPER_100K, config, &[])
        }
    }

    #[test]
    #[cfg_attr(miri, ignore = "too slow")]
    fn compress_fireworks() {
        let mut config = DeflateConfig::default();

        for n in 0..=9 {
            config.level = n;
            fuzz_based_test(FIREWORKS, config, &[])
        }
    }

    #[test]
    fn simple_rle() {
        fuzz_based_test(
            b"\0\0\0\0\x06",
            DeflateConfig {
                level: -1,
                method: Method::Deflated,
                window_bits: 11,
                mem_level: 4,
                strategy: Strategy::Rle,
            },
            &[56, 17, 99, 0, 2, 54, 0, 0, 11, 0, 7],
        )
    }

    #[test]
    fn fill_window_out_of_bounds() {
        const INPUT: &[u8] = &[
            0x71, 0x71, 0x71, 0x71, 0x71, 0x6a, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71,
            0x71, 0x71, 0x71, 0x71, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x1d, 0x1d, 0x1d, 0x1d, 0x63,
            0x63, 0x63, 0x63, 0x63, 0x1d, 0x1d, 0x1d, 0x1d, 0x1d, 0x1d, 0x1d, 0x1d, 0x1d, 0x1d,
            0x1d, 0x27, 0x0, 0x0, 0x0, 0x1d, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x71, 0x71, 0x71,
            0x71, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x31, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x1d, 0x1d, 0x0, 0x0, 0x0, 0x0, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50,
            0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x48, 0x50,
            0x50, 0x50, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x2c, 0x0, 0x0, 0x0, 0x0, 0x4a,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71,
            0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x70, 0x71, 0x71, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x71, 0x71, 0x71, 0x71, 0x6a, 0x0, 0x0, 0x0, 0x0,
            0x71, 0x0, 0x71, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x31, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71,
            0x71, 0x71, 0x0, 0x4a, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x71, 0x71,
            0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71,
            0x70, 0x71, 0x71, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x71, 0x71, 0x71, 0x71,
            0x6a, 0x0, 0x0, 0x0, 0x0, 0x71, 0x0, 0x71, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x31, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x1d, 0x1d, 0x0, 0x0, 0x0, 0x0,
            0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50,
            0x50, 0x50, 0x50, 0x50, 0x48, 0x50, 0x0, 0x0, 0x71, 0x71, 0x71, 0x71, 0x3b, 0x3f, 0x71,
            0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x50, 0x50, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x2c, 0x0, 0x0, 0x0, 0x0, 0x4a, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x71,
            0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71,
            0x71, 0x70, 0x71, 0x71, 0x0, 0x0, 0x0, 0x6, 0x0, 0x0, 0x0, 0x0, 0x0, 0x71, 0x71, 0x71,
            0x71, 0x71, 0x71, 0x70, 0x71, 0x71, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x71, 0x71, 0x71, 0x71, 0x3b, 0x3f, 0x71, 0x71, 0x71,
            0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71,
            0x71, 0x3b, 0x3f, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x20, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x71, 0x75, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71,
            0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x10, 0x0, 0x71, 0x71,
            0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x3b, 0x71, 0x71, 0x71, 0x71, 0x71,
            0x71, 0x76, 0x71, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x71, 0x71, 0x71, 0x71, 0x71,
            0x71, 0x71, 0x71, 0x10, 0x0, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71,
            0x71, 0x3b, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x76, 0x71, 0x34, 0x34, 0x34, 0x34,
            0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34,
            0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x0, 0x0, 0x0, 0x0, 0x0, 0x34, 0x34, 0x34, 0x34,
            0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34,
            0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34,
            0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34,
            0x34, 0x34, 0x30, 0x34, 0x34, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x71, 0x0, 0x0, 0x0, 0x0, 0x6,
        ];

        fuzz_based_test(
            INPUT,
            DeflateConfig {
                level: -1,
                method: Method::Deflated,
                window_bits: 9,
                mem_level: 1,
                strategy: Strategy::HuffmanOnly,
            },
            &[
                0x18, 0x19, 0x4, 0xc1, 0x21, 0x1, 0xc4, 0x0, 0x10, 0x3, 0xb0, 0x18, 0x29, 0x1e,
                0x7e, 0x17, 0x83, 0xf5, 0x70, 0x6c, 0xac, 0xfe, 0xc9, 0x27, 0xdb, 0xb6, 0x6f, 0xdb,
                0xb6, 0x6d, 0xdb, 0x80, 0x24, 0xb9, 0xbb, 0xbb, 0x24, 0x49, 0x92, 0x24, 0xf, 0x2,
                0xd8, 0x36, 0x0, 0xf0, 0x3, 0x0, 0x0, 0x24, 0xd0, 0xb6, 0x6d, 0xdb, 0xb6, 0x6d,
                0xdb, 0xbe, 0x6d, 0xf9, 0x13, 0x4, 0xc7, 0x4, 0x0, 0x80, 0x30, 0x0, 0xc3, 0x22,
                0x68, 0xf, 0x36, 0x90, 0xc2, 0xb5, 0xfa, 0x7f, 0x48, 0x80, 0x81, 0xb, 0x40, 0x55,
                0x55, 0x55, 0xd5, 0x16, 0x80, 0xaa, 0x7, 0x9, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                0xe, 0x7c, 0x82, 0xe0, 0x98, 0x0, 0x0, 0x0, 0x4, 0x60, 0x10, 0xf9, 0x8c, 0xe2,
                0xe5, 0xfa, 0x3f, 0x2, 0x54, 0x55, 0x55, 0x65, 0x0, 0xa8, 0xaa, 0xaa, 0xaa, 0xba,
                0x2, 0x50, 0xb5, 0x90, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x78, 0x82, 0xe0, 0xd0,
                0x8a, 0x41, 0x0, 0x0, 0xa2, 0x58, 0x54, 0xb7, 0x60, 0x83, 0x9a, 0x6a, 0x4, 0x96,
                0x87, 0xba, 0x51, 0xf8, 0xfb, 0x9b, 0x26, 0xfc, 0x0, 0x1c, 0x7, 0x6c, 0xdb, 0xb6,
                0x6d, 0xdb, 0xb6, 0x6d, 0xf7, 0xa8, 0x3a, 0xaf, 0xaa, 0x6a, 0x3, 0xf8, 0xc2, 0x3,
                0x40, 0x55, 0x55, 0x55, 0xd5, 0x5b, 0xf8, 0x80, 0xaa, 0x7a, 0xb, 0x0, 0x7f, 0x82,
                0xe0, 0x98, 0x0, 0x40, 0x18, 0x0, 0x82, 0xd8, 0x49, 0x40, 0x2, 0x22, 0x7e, 0xeb,
                0x80, 0xa6, 0xc, 0xa0, 0x9f, 0xa4, 0x2a, 0x38, 0xf, 0x0, 0x0, 0xe7, 0x1, 0xdc,
                0x55, 0x95, 0x17, 0x0, 0x0, 0xae, 0x0, 0x38, 0xc0, 0x67, 0xdb, 0x36, 0x80, 0x2b,
                0x0, 0xe, 0xf0, 0xd9, 0xf6, 0x13, 0x4, 0xc7, 0x4, 0x0, 0x0, 0x30, 0xc, 0x83, 0x22,
                0x69, 0x7, 0xc6, 0xea, 0xff, 0x19, 0x0, 0x0, 0x80, 0xaa, 0x0, 0x0, 0x0, 0x0, 0x0,
                0x0, 0x8e, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0x6a,
                0xf5, 0x63, 0x60, 0x60, 0x3, 0x0, 0xee, 0x8a, 0x88, 0x67,
            ],
        )
    }

    #[test]
    fn read_buf_window_uninitialized() {
        // copies more in `read_buf_window` than is initialized at that point
        const INPUT: &str = include_str!("test-data/read_buf_window_uninitialized.txt");

        fuzz_based_test(
            INPUT.as_bytes(),
            DeflateConfig {
                level: 0,
                method: Method::Deflated,
                window_bits: 10,
                mem_level: 6,
                strategy: Strategy::Default,
            },
            &[],
        )
    }

    #[test]
    fn gzip_no_header() {
        let config = DeflateConfig {
            level: 9,
            method: Method::Deflated,
            window_bits: 31, // gzip
            ..Default::default()
        };

        let input = b"Hello World!";
        let os = gz_header::OS_CODE;

        fuzz_based_test(
            input,
            config,
            &[
                31, 139, 8, 0, 0, 0, 0, 0, 2, os, 243, 72, 205, 201, 201, 87, 8, 207, 47, 202, 73,
                81, 4, 0, 163, 28, 41, 28, 12, 0, 0, 0,
            ],
        )
    }

    #[test]
    #[rustfmt::skip]
    fn gzip_stored_block_checksum() {
        fuzz_based_test(
            &[
                27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 9, 0,
            ],
            DeflateConfig {
                level: 0,
                method: Method::Deflated,
                window_bits: 26,
                mem_level: 6,
                strategy: Strategy::Default,
            },
            &[
                31, 139, 8, 0, 0, 0, 0, 0, 4, gz_header::OS_CODE, 1, 18, 0, 237, 255, 27, 27, 27, 27, 27, 27, 27,
                27, 27, 27, 27, 27, 27, 27, 27, 27, 9, 0, 60, 101, 156, 55, 18, 0, 0, 0,
            ],
        )
    }

    #[test]
    fn hash_calc_difference() {
        // exposed an issue in the crc32 acle hash calc where the incorrect instruction was used.
        // a different hash function changes which match is found first in the hash map, and
        // depending on the algorithm that can give different results. Note that in this case, the
        // output of aarch64 and x86_64 do line up with the current aarch64 hashing approach.

        // historical note: previously we used the crc32 hardware instructions, but later versions
        // of zlib-ng stopped doing that (the dispatch cost was too high), and we followed suit.
        let output = &[
            24, 149, 99, 96, 102, 24, 104, 160, 7, 38, 57, 96, 92, 117, 6, 14, 6, 38, 60, 202, 65,
            14, 86, 99, 208, 3, 3, 6, 67, 6, 38, 22, 58, 56, 17, 10, 40, 119, 41, 84, 175, 26, 0,
            172, 56, 3, 232,
        ];

        fuzz_based_test(
            &[
                0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 0, 0, 0, 0, 0, 8, 0,
                0, 0, 0, 0, 0, 0, 0, 39, 0, 8, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 38, 0, 46, 46, 46, 46, 46, 46, 0,
                49, 0, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                46, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 39, 0, 8, 0, 2, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 38, 0, 46,
                46, 46, 46, 46, 46, 0, 49, 0, 2, 4, 0, 0, 8, 0, 0, 0, 0, 0, 0, 38,
            ],
            DeflateConfig {
                level: -1,
                method: Method::Deflated,
                window_bits: 8,
                mem_level: 2,
                strategy: Strategy::Default,
            },
            output,
        )
    }

    #[test]
    fn longest_match_difference() {
        let output = &[
            24, 87, 83, 81, 97, 100, 96, 96, 228, 98, 0, 3, 123, 6, 6, 77, 21, 16, 67, 5, 36, 10,
            102, 73, 139, 67, 164, 24, 194, 64, 32, 144, 75, 55, 16, 5, 248, 65, 65, 52, 152, 116,
            99, 100, 96, 96, 16, 98, 96, 151, 241, 243, 243, 11, 12, 52, 128, 41, 2, 153, 206, 6,
            50, 21, 106, 20, 20, 56, 66, 40, 5, 48, 169, 98, 13, 166, 4, 24, 98, 25, 20, 192, 138,
            173, 37, 24, 184, 32, 64, 65, 26, 68, 50, 112, 128, 57, 26, 32, 83, 224, 134, 73, 162,
            154, 8, 7, 14, 40, 60, 78, 12, 121, 38, 12, 17, 6, 6, 176, 15, 144, 248, 0, 125, 74,
            22, 82,
        ];

        fuzz_based_test(
            &[
                36, 36, 1, 0, 0, 1, 10, 0, 0, 0, 0, 0, 0, 63, 0, 0, 41, 36, 0, 0, 0, 0, 36, 36, 1,
                0, 0, 36, 0, 0, 0, 0, 27, 23, 0, 0, 0, 0, 0, 0, 0, 86, 86, 86, 86, 86, 81, 10, 45,
                81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 78, 78, 78, 78, 78, 78, 78,
                78, 91, 78, 78, 78, 78, 78, 70, 1, 0, 0, 0, 18, 0, 7, 28, 78, 78, 78, 81, 81, 48,
                81, 81, 81, 81, 81, 81, 81, 81, 10, 0, 0, 0, 6, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 65, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 36, 59, 0, 0, 0, 0,
                0, 16, 0, 93, 0, 32, 0, 6, 0, 0, 0, 59, 24, 0, 10, 10, 10, 10, 10, 10, 10, 32, 27,
                10, 10, 10, 10, 0, 8, 10, 10, 10, 10, 10, 40, 0, 36, 0, 0, 6, 0, 0, 0, 36, 0, 0, 0,
                0, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 78, 78, 78, 78,
                78, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            DeflateConfig {
                level: 4,
                method: Method::Deflated,
                window_bits: 8,
                mem_level: 6,
                strategy: Strategy::Default,
            },
            output,
        )
    }

    // a change introduced in [commit].
    // make sure we match the new behavior, and throw an error if the test suite runs with older
    // versions
    //
    // [commit]: https://github.com/zlib-ng/zlib-ng/commit/322753f36e833343ae030e499564691da15eef32
    #[test]
    fn deflate_medium_bypass() {
        let mut input = [0u8; 268];
        input[0] = 0x16;
        input[263] = 0x5;

        fuzz_based_test(
            &input,
            DeflateConfig::default(),
            &[
                120, 156, 19, 99, 24, 5, 12, 12, 12, 172, 32, 2, 0, 24, 45, 0, 28,
            ],
        )
    }

    // Found an arithmetic bug in `fill_window` when using small window_bits.
    #[test]
    fn fill_window_bug() {
        let input = &[
            10, 1, 16, 0, 0, 0, 0, 0, 0, 0, 0, 4, 58, 5, 0, 0, 0, 8, 0, 8, 0, 0, 0, 5, 0, 0, 0, 0,
            0, 0, 0, 5, 4, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0,
            8, 0, 0, 0, 1, 0, 0, 0, 0, 90, 90, 77, 77, 109, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77,
            77, 77, 0, 0, 0, 0, 0, 0, 4, 58, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77,
            77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77,
            77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 52, 5, 0, 0, 0, 0, 0, 0, 77, 77, 77, 77, 77,
            77, 77, 90, 113, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 0, 0, 0, 0, 0,
            0, 8, 0, 0, 0, 0, 0, 0, 35, 0, 0, 0, 1, 24, 2, 0, 0, 56, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0,
            0, 8, 0, 0, 0, 1, 0, 0, 0, 0, 56, 0, 0, 0, 93, 0, 0, 0, 0, 0, 97, 97, 97, 97, 97, 12,
            12, 12, 12, 12, 12, 12, 12, 12, 12, 0, 0, 0, 0, 0, 0, 55, 55, 55, 55, 55, 55, 55, 55,
            55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 12, 12, 12, 12, 12, 12,
            12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
            12, 12, 12, 12, 12, 12, 12, 12, 12, 44, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
            12, 12, 12, 12, 12, 12, 12, 12, 12, 1, 0, 110, 110, 110, 110, 110, 1, 0, 110, 110, 110,
            110, 110, 110, 30, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110,
            110, 110, 110, 110, 0, 91, 110, 110, 1, 0, 0, 0, 0, 44, 110, 110, 0, 0, 110, 110, 110,
            110, 110, 110, 110, 110, 110, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            91, 0, 0, 0, 0, 0, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77,
            77, 77, 77, 77, 77, 3, 0, 0, 0, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77,
            77, 77, 77, 77, 43, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 90, 113, 90, 90, 90, 90,
            90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 39, 0, 0, 0, 0, 0, 8, 0, 8, 0, 12, 12, 12, 65,
            12, 1, 0, 110, 110, 110, 110, 110, 1, 0, 110, 110, 110, 110, 110, 110, 110, 110, 110,
            110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 0, 91, 110, 110,
            1, 0, 0, 0, 0, 0, 0, 44, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 0, 0,
            0, 16, 10, 0, 2, 0, 74, 0, 0, 0, 56, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 91, 0, 0, 0, 0, 0, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77,
            77, 77, 77, 77, 77, 77, 77, 77, 3, 0, 0, 0, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77,
            77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 90, 113, 90,
            90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 0, 0, 0, 0, 0, 0, 8, 0, 8, 0, 0, 5,
            0, 0, 0, 0, 0, 0, 0, 0, 5, 4, 0, 0, 0, 0, 1, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46,
            46, 46, 55, 55, 55, 55, 55, 55, 91, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55,
            55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 12, 12, 0, 0, 0, 0, 0,
            60, 0, 0, 0, 8, 0, 0, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 44, 12, 12, 12, 12, 12,
            12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 1, 0, 110, 110, 110,
            110, 110, 1, 0, 110, 110, 110, 110, 110, 110, 30, 110, 110, 110, 110, 110, 110, 110,
            110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 0, 91, 110, 110, 1, 0, 0, 0, 0, 44,
            110, 110, 0, 0, 110, 110, 110, 110, 110, 110, 110, 110, 110, 0, 0, 0, 16, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 91, 0, 0, 0, 0, 0, 77, 77, 77, 77, 77, 77, 77, 77, 77,
            77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 3, 0, 0, 0, 77, 77, 77, 77, 77, 77,
            77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 43, 77, 77, 77, 77, 77, 77, 77, 77, 77,
            77, 90, 113, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 39, 0, 0, 0, 0, 0,
            8, 0, 8, 0, 12, 12, 12, 65, 12, 1, 0, 110, 110, 110, 110, 110, 1, 0, 110, 110, 110,
            110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110,
            110, 110, 110, 0, 91, 110, 110, 1, 0, 0, 0, 0, 0, 0, 44, 110, 110, 110, 110, 110, 110,
            110, 110, 110, 110, 110, 0, 0, 0, 16, 10, 0, 2, 0, 74, 0, 0, 0, 56, 0, 0, 0, 9, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 91, 0, 0, 0, 0, 0, 77, 77, 77, 77, 77, 77,
            77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 3, 0, 0, 0, 77, 77, 77,
            77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77,
            77, 77, 77, 77, 90, 113, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 0, 0,
            0, 0, 0, 0, 8, 0, 8, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 5, 4, 0, 77, 77, 77, 3, 0, 0, 0,
            77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 43, 77, 77, 77,
            77, 77, 77, 77, 77, 77, 77, 90, 113, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90,
            90, 90, 0, 0, 0, 0, 0, 0, 8, 0, 8, 0, 12, 12, 12, 65, 12, 1, 0, 110, 110, 110, 110,
            110, 1, 0, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110,
            110, 110, 110, 110, 110, 110, 110, 110, 0, 91, 110, 110, 1, 0, 0, 0, 0, 0, 0, 44, 110,
            110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 0, 0, 0, 16, 10, 0, 2, 0, 74, 0, 0,
            0, 56, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 74, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 74, 0, 0, 0, 0, 0, 0, 35, 0, 0, 0, 1, 24, 2, 0, 0, 56, 0, 0, 0,
        ];

        let config = DeflateConfig {
            level: 9,
            window_bits: 8,
            ..DeflateConfig::default()
        };

        fuzz_based_test(input, config, &[]);
    }
}

#[test]
fn issue_169() {
    const INPUT: &str = include_str!("test-data/issue-169.js");
    let buf = INPUT;

    crate::assert_eq_rs_ng!(unsafe {
        let mut out = [0u8; 4096];
        let mut ret;

        let mut strm = MaybeUninit::zeroed();

        // first validate the config
        ret = deflateInit2_(
            strm.as_mut_ptr(),
            Z_BEST_SPEED,
            Z_DEFLATED,
            -15,
            8,
            Z_DEFAULT_STRATEGY,
            VERSION,
            test_c_stream_size::<libz_rs_sys::z_stream>(),
        );
        assert_eq!(ret, Z_OK);

        let strm = strm.assume_init_mut();

        strm.avail_in = 2048;
        strm.avail_out = 1053;
        strm.next_in = buf.as_ptr().cast_mut();
        strm.next_out = out.as_mut_ptr();
        ret = deflate(strm, Z_NO_FLUSH);
        assert_eq!(ret, Z_OK);

        assert_eq!(strm.avail_in, 0);
        assert_eq!(strm.avail_out, 1053);
        assert_eq!(strm.total_in, 2048);
        assert_eq!(strm.total_out, 0);

        strm.avail_in = 67;
        ret = deflate(strm, Z_FINISH);
        assert_eq!(ret, Z_OK);

        // libz-sys behavior is sligtly different on CI windows
        #[cfg(not(windows))]
        {
            assert_eq!(strm.avail_in, 67);
            assert_eq!(strm.avail_out, 0);
            assert_eq!(strm.total_in, 2048);
            assert_eq!(strm.total_out, 1053);
        }

        strm.avail_out = (out.len() as core::ffi::c_ulong - strm.total_out) as uInt;
        ret = deflate(strm, Z_FINISH);

        // libz-sys behavior is sligtly different on CI windows
        #[cfg(not(windows))]
        {
            assert_eq!(strm.avail_in, 0);
            assert_eq!(strm.avail_out, 2854);
            assert_eq!(strm.total_in, 2115);
            assert_eq!(strm.total_out, 1242);
        }

        assert_eq!(ret, Z_STREAM_END);
        assert_eq!(strm.avail_in, 0);

        deflateEnd(strm);

        out
    });
}

// Exact original regression header bytes; this fixture must remain unchanged.
const PENDING_HEADER_NAME: [u8; 510] = [
    0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b,
    0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x97, 0x97, 0x97, 0x97, 0x97, 0x97, 0x97, 0x97,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x8b, 0x8b, 0x8b, 0x8b,
    0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b,
    0x8b, 0x8b, 0x97, 0x97, 0x97, 0x97, 0x97, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x8b,
    0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b,
    0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x97, 0x97, 0x97, 0x97, 0x97, 0x97, 0x97, 0x97,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x97, 0x97, 0x97,
    0x97, 0x97, 0x97, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b,
    0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x97, 0x97, 0x97, 0x97,
    0x97, 0x97, 0x97, 0x97, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0x97, 0x97, 0x97, 0x97, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x99, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xbf, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0x97, 0x97, 0x97, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x97, 0x97, 0x97, 0x97, 0x97, 0x97,
    0x97, 0x97, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x97,
    0x97, 0x97, 0x97, 0x97, 0x97, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b,
    0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x8b, 0x97, 0x97,
    0x97, 0x97, 0x97, 0x97, 0x97, 0x97, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
];

#[test]
fn flood_pending_buffer() {
    // this test configures a very small mem_level, resulting in a small `Pending` buffer. In
    // combination with large chunks of gzip header data, this exposed a logic error in the
    // `Pending` implementation, that resulted in out-of-bounds writes.

    let name = CString::new(PENDING_HEADER_NAME).unwrap_or_default();
    assert_eq!(name.as_bytes(), &PENDING_HEADER_NAME);
    let comment = CString::default();

    let mut extra = [
        139, 139, 139, 139, 139, 139, 139, 139, 139, 139, 139, 139, 151, 151, 151, 151, 151, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 139, 0, 139, 139, 139, 139, 139, 139, 139, 139, 139, 139, 139,
        139, 139, 139, 139, 139, 223, 223, 223, 223, 223, 139, 139, 139, 255, 255, 255, 255, 255,
        193, 193, 193, 193, 193, 193, 193, 193, 193, 193, 193, 193, 193, 193, 193, 193, 193, 193,
        193, 193, 193, 193, 193, 193, 193, 193, 193, 193, 193, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 151, 151, 151, 151, 151, 151, 151, 39,
    ]
    .to_vec();

    let compressed = assert_eq_rs_ng!({
        let config = DeflateConfig {
            level: Z_DEFAULT_COMPRESSION,
            method: Method::Deflated,
            window_bits: 31,
            mem_level: 1, // Deliberately tiny pending buffer
            strategy: Strategy::HuffmanOnly,
        };
        let mut raw = MaybeUninit::<z_stream>::zeroed();
        // SAFETY: valid gzip configuration with matching ABI version and
        // checked size, writing into live uninitialized C stream storage.
        let init = unsafe {
            deflateInit2_(
                raw.as_mut_ptr(),
                config.level,
                config.method as c_int,
                config.window_bits,
                config.mem_level,
                config.strategy as c_int,
                zlibVersion(),
                test_c_stream_size::<z_stream>(),
            )
        };
        assert_eq!(init, Z_OK);
        // SAFETY: initialization populated the C stream.
        let stream = unsafe { raw.assume_init_mut() };

        let mut header = gz_header {
            text: 0,
            time: 0,
            xflags: 0,
            os: -1_953_789_045,
            extra: extra.as_mut_ptr(),
            extra_len: test_c_uint_len(extra.len()),
            extra_max: 0,
            name: name.as_ptr().cast_mut().cast(),
            name_max: 0,
            comment: comment.as_ptr().cast_mut().cast(),
            comm_max: 0,
            hcrc: 222,
            done: 0,
        };
        // SAFETY: the gzip metadata buffers are live for the entire encoder
        // lifetime, and this header is writable and disjoint from state.
        assert_eq!(unsafe { deflateSetHeader(stream, &raw mut header) }, Z_OK);

        // Include an actual payload: it tests that a large pending header
        // does not corrupt bytes that are emitted after the header.
        let payload = b"header spanning multiple pending-buffer flushes";
        stream.next_in = payload.as_ptr().cast_mut();
        stream.avail_in = test_c_uint_len(payload.len());

        // deflateBound is only an initial allocation hint here; a tiny
        // pending buffer can need repeated output-buffer refills.
        // SAFETY: the initialized stream is live and the requested bound is
        // a valid checked C unsigned long.
        let bound = unsafe { deflateBound(stream, test_c_ulong_len(payload.len())) };
        let initial = usize::try_from(bound).unwrap_or(0);
        assert!(initial > 0, "invalid initial gzip output bound");
        let mut encoded = vec![0xa5u8; initial];

        loop {
            let used = usize::try_from(stream.total_out).unwrap_or(usize::MAX);
            assert!(used <= encoded.len(), "encoder exceeded output storage");
            if used == encoded.len() {
                let expanded = encoded.len().checked_mul(2).unwrap_or(0);
                assert!(expanded > encoded.len(), "output size overflow");
                encoded.resize(expanded, 0xa5);
            }
            stream.next_out = encoded[used..].as_mut_ptr();
            stream.avail_out = test_c_uint_len(encoded.len() - used);
            let in_before = stream.total_in;
            let out_before = stream.total_out;
            // SAFETY: the remaining input pointer and each writable suffix
            // are live, disjoint, and represented with exact C byte counts.
            let status = unsafe { deflate(stream, Z_FINISH) };
            if status == Z_STREAM_END {
                break;
            }
            assert_eq!(status, Z_OK, "unexpected pending-buffer status");
            assert!(
                stream.total_in != in_before || stream.total_out != out_before,
                "pending-buffer encoder stalled"
            );
        }

        assert_eq!(stream.avail_in, 0);
        assert_eq!(usize::try_from(stream.total_in), Ok(payload.len()));
        let used = usize::try_from(stream.total_out).unwrap_or(usize::MAX);
        assert!(used <= encoded.len());
        assert!(encoded[used..].iter().all(|&byte| byte == 0xa5));
        // SAFETY: the finished encoder is ended once, before dropping the
        // borrowed metadata buffers and compressed output.
        assert_eq!(unsafe { deflateEnd(stream) }, Z_OK);
        encoded.truncate(used);
        encoded
    });

    let payload = b"header spanning multiple pending-buffer flushes";
    let mut restored = vec![0xa5u8; payload.len() + 16];
    let (decoded, status) = zlib_rs::decompress_slice(
        &mut restored,
        &compressed,
        InflateConfig { window_bits: 31 },
    );
    assert_eq!(status, ReturnCode::Ok);
    assert_eq!(decoded, payload);
    assert!(restored[payload.len()..].iter().all(|&byte| byte == 0xa5));
}

#[test]
fn copy_uninitialized_window_section() {
    // caused a panic because an attempt was made to copy uninitialized bytes within the window.
    // see also https://github.com/trifectatechfoundation/zlib-rs/issues/218
    const INPUT: [u8; 2067] = [
        0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 0, 1, 1, 0, 0, 0, 9, 0, 0, 0, 39, 39, 39, 39, 39,
        45, 45, 54, 0, 54, 38, 39, 39, 0, 0, 0, 0, 0, 78, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 98, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 97, 97, 39,
        97, 97, 97, 15, 0, 0, 0, 0, 0, 5, 0, 0, 113, 0, 0, 0, 0, 25, 26, 26, 0, 0, 0, 0, 5, 0, 0,
        40, 0, 0, 0, 3, 0, 0, 0, 32, 59, 0, 0, 0, 0, 0, 0, 31, 0, 0, 0, 2, 0, 0, 39, 39, 39, 39,
        39, 39, 39, 61, 64, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 39, 39, 39, 97,
        113, 24, 113, 113, 113, 113, 113, 121, 113, 97, 97, 97, 97, 39, 97, 97, 97, 113, 113, 113,
        113, 113, 113, 113, 113, 121, 113, 44, 44, 44, 0, 0, 0, 10, 0, 0, 1, 4, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 25, 26, 65, 0, 0, 0, 0, 5, 0, 0, 40, 0, 0, 0, 3, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 40, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 39, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 58, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 108, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 121, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 39, 121, 31, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 121, 0, 0, 0, 0, 123, 121, 0, 121, 121, 121, 0, 0, 0, 0, 0, 0, 110, 1, 8, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 2, 16, 122, 0, 0, 0, 0, 8, 8, 0, 31, 0, 2, 16, 9,
        0, 0, 0, 0, 0, 8, 0, 31, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 43, 0, 0, 0, 0, 0, 0, 0,
        58, 0, 0, 0, 0, 31, 39, 97, 97, 97, 97, 39, 97, 97, 97, 15, 0, 0, 0, 0, 0, 5, 0, 0, 113, 0,
        1, 39, 26, 26, 25, 48, 0, 0, 0, 0, 5, 0, 0, 40, 0, 0, 0, 3, 0, 0, 0, 32, 59, 0, 0, 0, 0, 0,
        91, 31, 0, 0, 0, 2, 0, 0, 39, 39, 39, 39, 39, 39, 39, 61, 64, 61, 61, 61, 61, 61, 61, 61,
        61, 61, 61, 61, 61, 41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 97, 39, 97, 97, 97, 15, 0,
        0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 121, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 121,
        31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 121, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 121, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 121, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        121, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
        61, 39, 39, 39, 97, 113, 24, 113, 113, 113, 113, 113, 121, 113, 97, 97, 97, 97, 39, 97, 97,
        97, 113, 113, 113, 113, 113, 113, 113, 113, 121, 113, 44, 44, 44, 0, 0, 0, 10, 0, 0, 1, 4,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 26, 65, 0, 0, 0, 0, 5, 0, 0, 40, 0, 0, 0, 3, 0, 0, 0, 32,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 39, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        58, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 108, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 121, 31, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 39, 121, 31, 2, 0, 0, 0, 0, 39, 49, 39, 49, 39, 39, 39, 0,
        0, 0, 0, 0, 48, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        7, 0, 46, 6, 0, 0, 0, 0, 0, 0, 31, 0, 0, 0, 0, 0, 64, 0, 0, 0, 4, 0, 0, 9, 0, 0, 0, 0, 3,
        0, 2, 61, 0, 0, 0, 0, 0, 1, 0, 0, 52, 56, 53, 53, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0,
        40, 0, 10, 4, 0, 0, 0, 0, 0, 0, 46, 121, 121, 121, 121, 121, 121, 121, 121, 0, 40, 0, 0, 0,
        0, 16, 0, 0, 0, 0,
    ];

    let config = DeflateConfig {
        level: 0,
        method: Method::Deflated,
        window_bits: 25,
        mem_level: 3,
        strategy: Strategy::Default,
    };

    let compressed = assert_eq_rs_ng!({
        // This local helper is compiled separately against the reference
        // zlib-ng and fork C ABI stream types by assert_eq_rs_ng!.
        fn prepare_output(stream: &mut z_stream, output: &mut Vec<u8>) {
            let used = usize::try_from(stream.total_out).unwrap_or(usize::MAX);
            assert!(used <= output.len(), "encoder exceeded output allocation");
            if used == output.len() {
                let expanded = output.len().checked_mul(2).unwrap_or(0);
                assert!(expanded > output.len(), "compressed output size overflow");
                output.resize(expanded, 0xa5);
            }
            // Previous output bytes must survive Vec growth/reallocation.
            // Repoint to the unwritten suffix, never the buffer beginning.
            stream.next_out = output[used..].as_mut_ptr();
            stream.avail_out = test_c_uint_len(output.len() - used);
        }

        let mut header = gz_header {
            text: 825307441,
            time: 14641,
            xflags: 0,
            os: 0,
            extra: core::ptr::null_mut(),
            extra_len: 0,
            extra_max: 0,
            name: core::ptr::null_mut(),
            name_max: 0,
            comment: core::ptr::null_mut(),
            comm_max: 0,
            hcrc: 0,
            done: 0,
        };

        let mut raw = MaybeUninit::<z_stream>::zeroed();
        // SAFETY: writable C stream storage, gzip-compatible settings,
        // matching version and checked C structure size.
        let status = unsafe {
            deflateInit2_(
                raw.as_mut_ptr(),
                config.level,
                config.method as i32,
                config.window_bits,
                config.mem_level,
                config.strategy as i32,
                zlibVersion(),
                test_c_stream_size::<z_stream>(),
            )
        };
        assert_eq!(status, Z_OK);
        // SAFETY: deflateInit2_ fully initialized the C stream on success.
        let stream = unsafe { raw.assume_init_mut() };
        // SAFETY: the initialized encoder borrows writable header fields
        // for the duration of compression; no metadata pointers are live.
        assert_eq!(unsafe { deflateSetHeader(stream, &raw mut header) }, Z_OK);

        // The bound applies to a single finish operation; PartialFlush may
        // produce additional blocks, so the output can need more space.
        // SAFETY: deflateBound only reads the initialized encoder state.
        let bound = unsafe { deflateBound(stream, test_c_ulong_len(INPUT.len())) };
        let initial_capacity = usize::try_from(bound).unwrap_or(0);
        assert!(initial_capacity > 0, "invalid compressed output bound");
        let mut encoded = vec![0xa5u8; initial_capacity];
        const CHUNK: usize = 47;

        // Offer each source span exactly once. A successful deflate call is
        // not guaranteed to have consumed the entire offered input chunk.
        for part in INPUT.chunks(CHUNK) {
            stream.next_in = part.as_ptr().cast_mut();
            stream.avail_in = test_c_uint_len(part.len());
            while stream.avail_in != 0 {
                prepare_output(stream, &mut encoded);
                let input_before = stream.total_in;
                let output_before = stream.total_out;
                // SAFETY: part is a live readable input span and the output
                // pointer describes only the live unwritten output suffix.
                let ret = unsafe { deflate(stream, DeflateFlush::PartialFlush as c_int) };
                let code = ReturnCode::from(ret);
                assert!(
                    matches!(code, ReturnCode::Ok | ReturnCode::BufError),
                    "unexpected PartialFlush status: {code:?}"
                );
                assert!(
                    stream.total_in != input_before || stream.total_out != output_before,
                    "PartialFlush stalled with unconsumed input"
                );
            }
        }
        assert_eq!(usize::try_from(stream.total_in), Ok(INPUT.len()));

        // Finish while retaining all previously produced output and
        // requiring progress before retrying with any remaining capacity.
        loop {
            prepare_output(stream, &mut encoded);
            let output_before = stream.total_out;
            // SAFETY: input has been consumed; the encoder has live state
            // and a writable, correctly sized output suffix.
            let ret = unsafe { deflate(stream, DeflateFlush::Finish as c_int) };
            let code = ReturnCode::from(ret);
            if code == ReturnCode::StreamEnd {
                break;
            }
            assert!(
                matches!(code, ReturnCode::Ok | ReturnCode::BufError),
                "unexpected finish status: {code:?}"
            );
            assert!(
                stream.total_out > output_before,
                "finish stalled without producing any compressed bytes"
            );
        }

        let produced = usize::try_from(stream.total_out).unwrap_or(usize::MAX);
        assert!(
            produced <= encoded.len(),
            "encoder exceeded allocated output"
        );
        assert_eq!(usize::try_from(stream.total_in), Ok(INPUT.len()));
        assert_eq!(
            &encoded[produced..],
            &vec![0xa5u8; encoded.len() - produced]
        );
        // SAFETY: the initialized stream is closed exactly once and all
        // compressed bytes have already been copied into our Vec.
        assert_eq!(unsafe { deflateEnd(stream) }, Z_OK);
        encoded.truncate(produced);
        encoded
    });

    // Differential compressed-byte equality alone is insufficient:
    // two backends could both produce the same truncated/corrupt member
    // if the harness skipped input or reset next_out after reallocation.
    let mut decoded = vec![0xa5u8; INPUT.len() + 32];
    let (plain, status) = zlib_rs::inflate::decompress_slice(
        &mut decoded,
        &compressed,
        InflateConfig { window_bits: 31 },
    );
    assert_eq!(status, ReturnCode::Ok);
    assert_eq!(plain, INPUT);
    assert_eq!(
        &decoded[INPUT.len()..],
        &vec![0xa5u8; 32],
        "decoded output exceeded the expected payload"
    );
}

/// Return the exact unwritten suffix of an initialized output buffer,
/// growing it when necessary. The raw pointer is valid until the caller
/// next reallocates the vector, so the C call must use it immediately.
fn checked_test_output_suffix(output: &mut Vec<u8>, already_written: c_ulong) -> (*mut u8, c_uint) {
    let written = usize::try_from(already_written).unwrap_or(usize::MAX);
    assert!(written <= output.len(), "encoder output exceeds allocation");
    if written == output.len() {
        let new_len = output.len().checked_mul(2).unwrap_or(0);
        assert!(new_len > output.len(), "output growth overflow");
        output.resize(new_len, 0xa5);
    }
    (
        output[written..].as_mut_ptr(),
        test_c_uint_len(output.len() - written),
    )
}

/// Require a complete gzip member to decode to all original bytes without
/// overwriting even the first byte past the expected plaintext.
fn assert_gzip_decodes_exact(compressed: &[u8], expected: &[u8]) {
    let capacity = expected.len().checked_add(16).unwrap_or(0);
    assert!(capacity >= expected.len(), "decoded scratch size overflow");
    let mut restored = vec![0xa5u8; capacity];
    let (plain, status) =
        zlib_rs::decompress_slice(&mut restored, compressed, InflateConfig { window_bits: 31 });
    assert_eq!(status, ReturnCode::Ok);
    assert_eq!(plain, expected);
    assert!(restored[expected.len()..].iter().all(|&b| b == 0xa5));
}

// The original issue #219 trigger requires level 0, a gzip wrapper,
// mem_level 3 and the default strategy before switching to level 4.
const CRC32_ISSUE_219_CONFIG: DeflateConfig = DeflateConfig {
    level: 0,
    method: Method::Deflated,
    window_bits: 31,
    mem_level: 3,
    strategy: Strategy::Default,
};

// Exact 309-byte reproducer for upstream issue #219. The byte sequence
// exercises a history/hash transition immediately after changing deflate
// compression level; do not synthesize or shorten this fixture.
const CRC32_ISSUE_219_INPUT: [u8; 309] = [
    96, 48, 113, 0, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 71, 0, 0, 0, 0, 0, 0, 7, 0, 0,
    0, 64, 0, 0, 0, 38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86,
    86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86,
    86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86,
    86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86,
    86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 0, 0, 0, 0,
    121, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 121, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 8, 0, 0, 0, 0, 0, 1, 0, 0, 0, 121, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0,
    121, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 121,
    31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0,
];

#[test]
fn crc32_hash_calc_uninitialized_memory() {
    // The original bug involved reading uninitialized hash-history bytes
    // after a two-byte initial feed and deflateParams(level=4).
    const CHUNK: usize = 2;
    let source = &CRC32_ISSUE_219_INPUT;
    let config = CRC32_ISSUE_219_CONFIG;

    let compressed = assert_eq_rs_ng!({
        // This helper is instantiated against the relevant C stream type
        // for each backend. A Vec reallocation invalidates old next_out.
        fn writable_suffix(stream: &mut z_stream, output: &mut Vec<u8>) {
            let (ptr, available) = checked_test_output_suffix(output, stream.total_out);
            stream.next_out = ptr;
            stream.avail_out = available;
        }

        let mut uninitialized = MaybeUninit::<z_stream>::zeroed();
        // SAFETY: writable C storage with a valid gzip configuration,
        // matching version and a checked C structure size.
        let init = unsafe {
            deflateInit2_(
                uninitialized.as_mut_ptr(),
                config.level,
                config.method as c_int,
                config.window_bits,
                config.mem_level,
                config.strategy as c_int,
                zlibVersion(),
                test_c_stream_size::<z_stream>(),
            )
        };
        assert_eq!(init, Z_OK);
        // SAFETY: successful initialization wrote all stream fields.
        let stream = unsafe { uninitialized.assume_init_mut() };

        // SAFETY: the initialized encoder is live and the byte count
        // fits c_ulong; bound is only an initial allocation estimate.
        let bound = unsafe { deflateBound(stream, test_c_ulong_len(source.len())) };
        let bound = usize::try_from(bound).unwrap_or(0);
        assert!(bound > 0, "invalid compression bound");
        // Force the historical buffer-relocation path even when
        // deflateBound would fit the whole output in one allocation.
        let initial_capacity = bound.min(32);
        let mut output = vec![0xa5u8; initial_capacity];

        // Historical reproducer: offer precisely the first two input
        // bytes at compression level 0 before switching to level 4.
        stream.next_in = source.as_ptr().cast_mut();
        stream.avail_in = test_c_uint_len(CHUNK);
        writable_suffix(stream, &mut output);
        // SAFETY: the first two bytes are readable and the output is
        // a live, disjoint writable suffix with exact C capacity.
        assert_eq!(unsafe { deflate(stream, Z_NO_FLUSH) }, Z_OK);
        assert!(usize::try_from(stream.total_in).unwrap_or(usize::MAX) <= CHUNK);

        // Switching levels may flush pending output and report BufError
        // when the output buffer fills. Retry while retaining the real
        // source cursor, never resetting next_out to the Vec beginning.
        loop {
            writable_suffix(stream, &mut output);
            let before = (stream.total_in, stream.total_out);
            // SAFETY: the stream and output allocation remain live and
            // pointer/availability fields describe valid remaining spans.
            let status = unsafe { deflateParams(stream, 4, config.strategy as c_int) };
            if status == Z_OK {
                break;
            }
            assert_eq!(status, Z_BUF_ERROR, "unexpected parameter-change error");
            assert!(
                (stream.total_in, stream.total_out) != before || stream.avail_out == 0,
                "deflateParams stalled with writable space available"
            );
        }

        // Do not assume a Z_OK call consumes all offered input. Base
        // each new 2-byte offer on the actual C total_in cursor.
        loop {
            let consumed = usize::try_from(stream.total_in).unwrap_or(usize::MAX);
            assert!(consumed <= source.len(), "compressor exceeded input");
            if consumed == source.len() {
                break;
            }
            let end = (consumed + CHUNK).min(source.len());
            stream.next_in = source[consumed..end].as_ptr().cast_mut();
            stream.avail_in = test_c_uint_len(end - consumed);
            while stream.avail_in != 0 {
                writable_suffix(stream, &mut output);
                let before = (stream.total_in, stream.total_out);
                // SAFETY: the current input and output spans are live,
                // separate allocations and accurately C-length-bounded.
                let status = unsafe { deflate(stream, Z_NO_FLUSH) };
                assert!(
                    status == Z_OK || status == Z_BUF_ERROR,
                    "unexpected chunk status {status}"
                );
                assert!(
                    (stream.total_in, stream.total_out) != before || stream.avail_out == 0,
                    "compressor stalled without consuming or producing data"
                );
            }
        }
        assert_eq!(usize::try_from(stream.total_in), Ok(source.len()));

        loop {
            writable_suffix(stream, &mut output);
            let before = stream.total_out;
            // SAFETY: the source is entirely consumed, encoder initialized,
            // and output points to a live, checked writable suffix.
            let status = unsafe { deflate(stream, DeflateFlush::Finish as c_int) };
            if status == Z_STREAM_END {
                break;
            }
            assert_eq!(status, Z_OK, "unexpected final deflate status");
            assert!(
                stream.total_out > before || stream.avail_out == 0,
                "deflate finish stalled"
            );
        }

        assert_eq!(usize::try_from(stream.total_in), Ok(source.len()));
        let produced = usize::try_from(stream.total_out).unwrap_or(usize::MAX);
        assert!(produced <= output.len());
        assert!(
            output.len() > initial_capacity,
            "issue #219 fixture must exercise output-buffer growth"
        );
        assert!(output[produced..].iter().all(|&b| b == 0xa5));
        // SAFETY: this finished initialized encoder is ended exactly once.
        assert_eq!(unsafe { deflateEnd(stream) }, Z_OK);
        output.truncate(produced);
        output
    });

    // Differential bytes alone cannot catch a shared harness bug.
    assert_gzip_decodes_exact(&compressed, source);
}

#[test]
fn test_deflate_get_dict() {
    // deflateGetDictionary returns a binary span and its explicit length,
    // never a NUL-terminated C string. Include a trailing NUL in the input
    // specifically to detect accidental CStr-style truncation.
    let plain = *b"hello, hello!\0";
    let (dictionary, compressed) = assert_eq_rs_ng!({
        let mut raw = MaybeUninit::<z_stream>::zeroed();
        // SAFETY: live writable C stream storage, matching library version
        // and checked ABI structure size.
        let init = unsafe {
            deflateInit_(
                raw.as_mut_ptr(),
                Z_BEST_COMPRESSION,
                zlibVersion(),
                test_c_stream_size::<z_stream>(),
            )
        };
        assert_eq!(init, Z_OK);
        // SAFETY: successful initialization populated the entire stream.
        let stream = unsafe { raw.assume_init_mut() };

        let mut encoded = [0xa5u8; 1_024];
        stream.next_out = encoded.as_mut_ptr();
        stream.avail_out = test_c_uint_len(encoded.len());
        stream.next_in = plain.as_ptr().cast_mut();
        stream.avail_in = test_c_uint_len(plain.len());

        // SAFETY: live, disjoint source and writable output buffers
        // are advertised with exact C byte counts.
        assert_eq!(unsafe { deflate(stream, Z_FINISH) }, Z_STREAM_END);
        assert_eq!(stream.avail_in, 0);
        assert_eq!(usize::try_from(stream.total_in), Ok(plain.len()));
        let produced = usize::try_from(stream.total_out).unwrap_or(usize::MAX);
        assert!(produced <= encoded.len());
        assert!(encoded[produced..].iter().all(|&b| b == 0xa5));

        let mut dictionary_storage = [0xa5u8; 256];
        let mut dictionary_len = test_c_uint_len(dictionary_storage.len());
        // SAFETY: live initialized stream and writable destination,
        // whose actual 256-byte capacity fits in the checked C length.
        assert_eq!(
            // SAFETY: the dictionary bytes are binary; the C API returns
            // the exact number of initialized bytes through dictionary_len.
            unsafe {
                deflateGetDictionary(
                    stream,
                    dictionary_storage.as_mut_ptr(),
                    &raw mut dictionary_len,
                )
            },
            Z_OK
        );
        let dictionary_len = usize::try_from(dictionary_len).unwrap_or(usize::MAX);
        assert!(dictionary_len <= dictionary_storage.len());
        assert!(dictionary_storage[dictionary_len..]
            .iter()
            .all(|&b| b == 0xa5));
        let dictionary = dictionary_storage[..dictionary_len].to_vec();

        // SAFETY: encoder initialized successfully and is closed once.
        assert_eq!(unsafe { deflateEnd(stream) }, Z_OK);
        (dictionary, encoded[..produced].to_vec())
    });

    assert_eq!(dictionary, plain, "incorrect binary dictionary history");
    let mut restored = [0xa5u8; 64];
    let (decoded, status) =
        zlib_rs::decompress_slice(&mut restored, &compressed, InflateConfig::default());
    assert_eq!(status, ReturnCode::Ok);
    assert_eq!(decoded, plain);
    assert!(restored[plain.len()..].iter().all(|&b| b == 0xa5));
}

#[derive(Debug, Clone)]
struct GzHeaderData {
    text: i32,
    time: core::ffi::c_ulong,
    os: i32,
    extra: Vec<u8>,
    name: CString,
    comment: CString,
    hcrc: i32,
}

trait GzHeaderFromData {
    fn from_data(data: &mut GzHeaderData) -> Self;
}

macro_rules! impl_gz_header_from_data {
    ($ty:path) => {
        impl GzHeaderFromData for $ty {
            fn from_data(data: &mut GzHeaderData) -> Self {
                Self {
                    text: data.text,
                    time: data.time,
                    xflags: 0,
                    os: data.os,
                    extra: data.extra.as_mut_ptr(),
                    extra_len: test_c_uint_len(data.extra.len()),
                    extra_max: 0,
                    name: data.name.as_ptr().cast_mut().cast(),
                    name_max: 0,
                    comment: data.comment.as_ptr().cast_mut().cast(),
                    comm_max: 0,
                    hcrc: data.hcrc,
                    done: 0,
                }
            }
        }
    };
}
impl_gz_header_from_data!(libz_sys::gz_header);
impl_gz_header_from_data!(libz_rs_sys::gz_header);

impl GzHeaderData {
    fn as_gz_header<T: GzHeaderFromData>(&mut self) -> T {
        T::from_data(self)
    }
}

#[derive(Debug)]
struct Input {
    source: String,
    config: DeflateConfig,
    flush: DeflateFlush,
    header: GzHeaderData,
}

/// Exact large, highly repetitive gzip Block-flush fixture. The
/// escaped source is kept verbatim; it deliberately exercises many
/// transitions across narrow pending buffers.
fn gzip_block_config_fixture() -> Input {
    // Preserve the original low C-unsigned-long bits, regardless of
    // whether the target C ABI uses four or eight bytes for c_ulong.
    let original_time = 17_940_362_863_843_014_904u64.to_le_bytes();
    let mut time_bytes = [0u8; core::mem::size_of::<c_ulong>()];
    let width = time_bytes.len();
    time_bytes.copy_from_slice(&original_time[..width]);
    let time = c_ulong::from_le_bytes(time_bytes);

    let comment = CString::new([0xffu8; 10]).unwrap_or_default();
    assert_eq!(comment.as_bytes(), &[0xffu8; 10]);
    Input {
	    source: "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0~\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0_\0\0\0\0%\0\0\0\0\0\0\0\0\0\u{13}\0\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}i\\\\\\\\\\\\\\\\\\\\\\\\\\b\\\\\\\\\\\\\\\\\\\\\\\\\0\0\0\0\0\0\\\\\0\0\0\0\0\0\0\0\0\0\0\0\0\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{1a}\u{1b}\u{1b}&\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\0\u{3}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\\\\\\\0\0\0\0\0\0\0_\0\0\0\0%\0\0\0\0\0\0\0\0\0\u{13}\0\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}i\\\\\\\\\\\\\\\\\\\\\\\\\\b\\\\\\\\\\\\\\\\\\\\\\\\\0\0\0\0\0\0\\\\\0\0\0\0\0\0\0\0\0\0\0\0\0\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{1a}\u{1b}\u{1b}&\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\0\u{3}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\\\\\\\\\\\\L\\\\L\\\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\0\u{3}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\u{7f}\\\\\\\\\\\\L\\\\L\\*".to_owned(),
	    config: DeflateConfig {
	        level: -1,
	        method: Method::Deflated,
	        window_bits: 31,
	        mem_level: 1,
	        strategy: Strategy::Fixed,
	    },
	    flush: DeflateFlush::Block,
	    header: GzHeaderData {
	        text: -120_653_576,
	        time, // preserves original low c_ulong bits.
	        os: -1,
	        extra: vec![
	            0,
	        ],
	        name: CString::default(),
	        comment,
	        hcrc: 1_548_508_252,
	    },
    }
}

#[test]
fn test_gzip_deflate_config() {
    let input = gzip_block_config_fixture();

    let compressed = assert_eq_rs_ng!({
        let Input {
            ref source,
            config,
            flush,
            ref header,
        } = input;
        let mut metadata = header.clone();
        let mut raw = MaybeUninit::<z_stream>::zeroed();

        // SAFETY: writable uninitialized C stream, valid gzip parameters,
        // matching ABI version and checked stream size.
        let init = unsafe {
            deflateInit2_(
                raw.as_mut_ptr(),
                config.level,
                config.method as c_int,
                config.window_bits,
                config.mem_level,
                config.strategy as c_int,
                zlibVersion(),
                test_c_stream_size::<z_stream>(),
            )
        };
        assert_eq!(init, Z_OK);
        // SAFETY: successful initialization populated the entire stream.
        let stream = unsafe { raw.assume_init_mut() };

        let mut c_header: gz_header = metadata.as_gz_header();
        // SAFETY: metadata and all nested vectors/CStrings stay allocated
        // and unchanged for the entire C encoder lifetime.
        let header_status = unsafe { deflateSetHeader(stream, &raw mut c_header) };
        assert_eq!(header_status, Z_OK, "gzip metadata unexpectedly rejected");

        // SAFETY: deflateBound reads only this initialized stream and a
        // checked input length. It is an allocation hint, not a proof that
        // alternate flush sequences cannot exceed the bound.
        let bound = unsafe { deflateBound(stream, test_c_ulong_len(source.len())) };
        let initial = usize::try_from(bound).unwrap_or(0);
        assert!(initial > 0, "invalid gzip initial capacity");
        let mut output = vec![0xa5u8; initial];

        // Offer the entire input once; do not treat merely *offered* bytes
        // as consumed. The decoder advances next_in and avail_in itself.
        stream.next_in = source.as_ptr().cast_mut();
        stream.avail_in = test_c_uint_len(source.len());
        loop {
            // Repoint after any Vec relocation, preserving all previously
            // produced bytes and advertising only the valid writable suffix.
            let (next_out, avail_out) = checked_test_output_suffix(&mut output, stream.total_out);
            stream.next_out = next_out;
            stream.avail_out = avail_out;

            let before_in = stream.total_in;
            let before_out = stream.total_out;
            let mode = if stream.avail_in == 0 {
                DeflateFlush::Finish
            } else {
                flush
            };
            // SAFETY: input refers to source's live readable suffix, and
            // output to the disjoint writable unwritten suffix of output.
            let status = unsafe { deflate(stream, mode as c_int) };
            if status == Z_STREAM_END {
                break;
            }
            assert_eq!(status, Z_OK, "gzip block/finish status");
            assert!(
                stream.total_in != before_in || stream.total_out != before_out,
                "encoder stalled during gzip header/block flush"
            );
        }

        assert_eq!(stream.avail_in, 0);
        assert_eq!(usize::try_from(stream.total_in), Ok(source.len()));
        let used = usize::try_from(stream.total_out).unwrap_or(usize::MAX);
        assert!(used <= output.len());
        assert!(output[used..].iter().all(|&b| b == 0xa5));
        // SAFETY: the finished stream is ended once while the gzip metadata
        // and all original buffers remain live.
        assert_eq!(unsafe { deflateEnd(stream) }, Z_OK);
        output.truncate(used);
        output
    });

    assert_gzip_decodes_exact(&compressed, input.source.as_bytes());
}

#[test]
fn test_issue_455() {
    // Compressing this with default settings results in a 33-byte deflate
    // stream but we only supply a 32-byte buffer. This should fail.
    let data = &[
        0, 0, 0, 0, 252, 0, 0, 62, 255, 255, 255, 42, 255, 255, 247, 255, 247, 255, 255, 255, 156,
        70, 255, 255,
    ];

    assert_eq!(
        assert_eq_rs_ng!(unsafe {
            let mut output = [0u8; 32];
            let mut length = 32;
            compress(
                output.as_mut_ptr(),
                &raw mut length,
                data.as_ptr(),
                data.len() as _,
            )
        }),
        ReturnCode::BufError as _
    );
}

/// See https://github.com/trifectatechfoundation/zlib-rs/issues/459.
mod deflate_reset_deterministic {
    use super::{test_c_stream_size, test_c_uint_len};
    use core::mem::MaybeUninit;
    use libz_rs_sys::{
        deflate, deflateEnd, deflateInit2_, deflateReset, z_stream, Z_DEFAULT_STRATEGY, Z_DEFLATED,
        Z_FINISH, Z_OK, Z_STREAM_END,
    };

    const DATA_A: &[u8] = b"\0AAAA\0AAAAAAAA";
    const DATA_B: &[u8] = &[1u8; DATA_A.len() + 1];

    /// Compress one input with a live, initialized C deflater.
    ///
    /// # Safety
    ///
    /// The caller must have initialized zs using deflateInit2_ or reset
    /// it using deflateReset; the stream may not have been ended.
    unsafe fn compress_data(src: &[u8], zs: &mut z_stream) -> Vec<u8> {
        const BUFFER_SIZE: usize = 1024;
        let mut buffer = [0xa5u8; BUFFER_SIZE];

        zs.next_in = src.as_ptr().cast_mut();
        zs.avail_in = test_c_uint_len(src.len());
        zs.next_out = buffer.as_mut_ptr();
        zs.avail_out = test_c_uint_len(buffer.len());

        // SAFETY: the caller initialized zs; source and writable output
        // are live and disjoint, with exact C byte capacities.
        let status = unsafe { deflate(zs, Z_FINISH) };
        assert_eq!(status, Z_STREAM_END, "deflate failed: {status}");
        assert_eq!(zs.avail_in, 0);
        assert_eq!(usize::try_from(zs.total_in), Ok(src.len()));
        let written = usize::try_from(zs.total_out).unwrap_or(usize::MAX);
        assert!(written <= buffer.len());
        assert_eq!(usize::try_from(zs.avail_out), Ok(buffer.len() - written));
        assert!(buffer[written..].iter().all(|&b| b == 0xa5));
        buffer[..written].to_vec()
    }

    #[test]
    fn reset_deterministic() {
        // SAFETY: each caller-owned z_stream is initialized using the
        // matching C ABI before compress_data/deflateReset/deflateEnd;
        // all source and destination allocations outlive their C calls,
        // and each initialized stream is ended exactly once.
        unsafe {
            let compression_level: i32 = 6;
            let window_bits: i32 = 15;

            // ---- Compress a with newly created z_stream. ----
            let mut a_stream = MaybeUninit::<z_stream>::zeroed().assume_init();

            let err = deflateInit2_(
                &raw mut a_stream,
                compression_level,
                Z_DEFLATED,
                window_bits,
                8,
                Z_DEFAULT_STRATEGY,
                libz_rs_sys::zlibVersion(),
                test_c_stream_size::<z_stream>(),
            );
            assert_eq!(err, Z_OK);

            let a_compressed = compress_data(DATA_A, &mut a_stream);

            let err = deflateEnd(&raw mut a_stream);
            assert_eq!(err, Z_OK);

            // ---- Compress b with newly created z_stream. ----
            let mut b_stream = MaybeUninit::<z_stream>::zeroed().assume_init();

            let err = deflateInit2_(
                &raw mut b_stream,
                compression_level,
                Z_DEFLATED,
                window_bits,
                8,
                Z_DEFAULT_STRATEGY,
                libz_rs_sys::zlibVersion(),
                test_c_stream_size::<z_stream>(),
            );
            assert_eq!(err, Z_OK);

            let _b_compressed = compress_data(DATA_B, &mut b_stream);

            // ---- Reset the stream. ----
            let err = deflateReset(&raw mut b_stream);
            assert_eq!(err, Z_OK);

            let a_compressed2 = compress_data(DATA_A, &mut b_stream);

            let err = deflateEnd(&raw mut b_stream);
            assert_eq!(err, Z_OK);

            assert_eq!(a_compressed, a_compressed2);
            for encoded in [&a_compressed, &a_compressed2] {
                let mut decoded = [0xa5u8; DATA_A.len() + 16];
                let (plain, status) = zlib_rs::decompress_slice(
                    &mut decoded,
                    encoded,
                    zlib_rs::inflate::InflateConfig::default(),
                );
                assert_eq!(status, zlib_rs::ReturnCode::Ok);
                assert_eq!(plain, DATA_A);
                assert!(decoded[DATA_A.len()..].iter().all(|&b| b == 0xa5));
            }
        }
    }
}

mod stable_api {
    use core::mem::MaybeUninit;

    use zlib_rs::{
        inflate::{decompress_slice, InflateConfig},
        DeflateFlush, ReturnCode, Status,
    };

    #[test]
    #[should_panic = "StreamError"]
    fn invalid_config() {
        let _ = zlib_rs::Deflate::new(9000, true, 15);
    }

    fn with_window_bits(window_bits: i32) {
        let input = "Hello World!";

        let mut deflate =
            zlib_rs::Deflate::new(9, window_bits >= 0, window_bits.unsigned_abs() as u8);
        let mut compressed = [0u8; 64];
        let ret = deflate.compress(input.as_bytes(), &mut compressed, DeflateFlush::Finish);
        assert_eq!(ret, Ok(Status::StreamEnd));

        assert_eq!(deflate.total_in() as usize, input.len());

        let compressed = &compressed[..deflate.total_out() as usize];
        let mut decompressed = [0u8; 64];
        let (decompressed, ret) =
            decompress_slice(&mut decompressed, compressed, InflateConfig { window_bits });
        assert_eq!(ret, ReturnCode::Ok);

        assert_eq!(decompressed, input.as_bytes());
    }

    #[test]
    fn raw() {
        with_window_bits(-15);
    }

    #[test]
    fn zlib_header() {
        with_window_bits(15);
    }

    #[test]
    fn gz_header() {
        with_window_bits(16 + 15);
    }

    #[test]
    fn reset_reuse_stream() {
        let input1 = "Hello World!";
        let input2 = "Goodbye World!";

        let mut deflate = zlib_rs::Deflate::new(9, true, 15);

        let mut compressed1 = [0u8; 64];
        let ret = deflate.compress(input1.as_bytes(), &mut compressed1, DeflateFlush::Finish);
        assert_eq!(ret, Ok(Status::StreamEnd));
        assert_eq!(deflate.total_in() as usize, input1.len());

        let compressed1 = &compressed1[..deflate.total_out() as usize];

        deflate.reset();

        let mut compressed2 = [0u8; 64];
        let ret = deflate.compress(input2.as_bytes(), &mut compressed2, DeflateFlush::Finish);
        assert_eq!(ret, Ok(Status::StreamEnd));
        // After reset, total_in should reflect only the second input
        assert_eq!(deflate.total_in() as usize, input2.len());

        let compressed2 = &compressed2[..deflate.total_out() as usize];

        let mut decompressed1 = [0u8; 64];
        let (decompressed1, ret) = decompress_slice(
            &mut decompressed1,
            compressed1,
            InflateConfig { window_bits: 15 },
        );
        assert_eq!(ret, ReturnCode::Ok);
        assert_eq!(decompressed1, input1.as_bytes());

        let mut decompressed2 = [0u8; 64];
        let (decompressed2, ret) = decompress_slice(
            &mut decompressed2,
            compressed2,
            InflateConfig { window_bits: 15 },
        );
        assert_eq!(ret, ReturnCode::Ok);
        assert_eq!(decompressed2, input2.as_bytes());
    }

    #[test]
    fn set_level_no_flush() {
        let mut d = zlib_rs::Deflate::new(1, true, 15);

        let input = vec![b'a'; 128];

        {
            let mut output = vec![MaybeUninit::<u8>::uninit(); 512];
            d.compress_uninit(&input, &mut output, DeflateFlush::NoFlush)
                .unwrap();
        }

        // Due to NoFlush there is data to write, but no buffer to write it into.
        assert_eq!(d.set_level(9), Err(zlib_rs::DeflateError::StreamError));
    }

    /// Test that the Deflate state does not retain the dictionary.
    #[test]
    fn set_dictionary_drop_dictionary() {
        let mut d = zlib_rs::Deflate::new(1, true, 15);

        {
            let dict = vec![b'a'; 128];
            d.set_dictionary(&dict).unwrap();
        }

        let input = vec![b'a'; 128];
        let mut output = vec![MaybeUninit::<u8>::uninit(); 512];
        d.compress_uninit(&input, &mut output, DeflateFlush::NoFlush)
            .unwrap();
    }
}
