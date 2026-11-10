use core::ffi::{c_char, c_int, c_uchar, c_uint, c_ulong, c_void, CStr};
use core::mem::MaybeUninit;

use zlib_rs::c_api::{
    Z_DATA_ERROR, Z_ERRNO, Z_NEED_DICT, Z_OK, Z_STREAM_END, Z_STREAM_ERROR, Z_VERSION_ERROR,
};
use zlib_rs::deflate::{compress_slice, DeflateConfig};
use zlib_rs::inflate::{decompress_slice, set_mode_dict, InflateConfig, INFLATE_STATE_SIZE};
use zlib_rs::{InflateFlush, ReturnCode, MAX_WBITS};

use crate::assert_eq_rs_ng;

const VERSION: *const c_char = libz_rs_sys::zlibVersion();
const STREAM_SIZE: c_int = core::mem::size_of::<libz_rs_sys::z_stream>() as c_int;

#[derive(Clone, Copy)]
struct MemItem {
    ptr: *mut c_void,
    size: usize,
}

struct MemZone {
    items: Vec<MemItem>,
    total: usize,
    highwater: usize,
    limit: usize,
    not_lifo: usize,
    rogue: usize,
}

unsafe extern "C" fn mem_alloc(mem: *mut c_void, count: u32, size: u32) -> *mut c_void {
    let count = count as usize;
    let size = size as usize;
    // C allocator inputs are untrusted. Overflow must report allocation
    // failure rather than unwind across the extern "C" callback boundary.
    let Some(len) = count.checked_mul(size) else {
        return std::ptr::null_mut();
    };

    // induced allocation failure
    if mem.is_null() {
        return std::ptr::null_mut();
    }

    // SAFETY: opaque is the live MemZone allocated by mem_setup.
    // The synchronous C allocator callback borrows it exclusively;
    // mem_done alone reclaims the original Box.
    let zone = unsafe { &mut *mem.cast::<MemZone>() };

    // Check the accounting even without a configured memory limit: the
    // callback must never allocate a smaller buffer than it advertises.
    let Some(new_total) = zone.total.checked_add(len) else {
        return std::ptr::null_mut();
    };
    if zone.limit != 0 && new_total > zone.limit {
        return std::ptr::null_mut();
    }

    // Keep the accounting vector fallible within the extern-C callback:
    // a failed metadata reservation must be an allocation failure too.
    if zone.items.try_reserve(1).is_err() {
        return core::ptr::null_mut();
    }

    extern "C" {
        fn malloc(size: usize) -> *mut c_void;
    }

    // SAFETY: libc malloc accepts any usize request and returns either a
    // fresh allocation of at least len bytes or null on failure.
    let ptr = unsafe { malloc(len) };
    if ptr.is_null() {
        return std::ptr::null_mut();
    }

    // SAFETY: non-null malloc returned writable storage of at least len
    // bytes; this initializes exactly the allocated range.
    unsafe { std::ptr::write_bytes(ptr, 0xa5, len) };

    // create a new item for the list
    let item = MemItem { ptr, size: len };

    // update the statistics
    zone.total = new_total;
    zone.highwater = zone.highwater.max(new_total);

    // insert item
    zone.items.push(item);

    // return the allocated memory
    ptr
}

unsafe extern "C" fn mem_free(mem: *mut c_void, ptr: *mut c_void) {
    extern "C" {
        fn free(p: *mut c_void);
    }

    if ptr.is_null() {
        // C free(NULL) is an ordinary no-op, not an unrecognized allocation.
        return;
    }

    if mem.is_null() {
        // SAFETY: the fallback callback requires an allocation originating
        // from the matching libc allocator when no accounting zone is used.
        unsafe { free(ptr) };
        return;
    }

    // SAFETY: this synchronous callback receives the live owned MemZone.
    // Borrowing it does not transfer its Box ownership away from mem_done.
    let zone = unsafe { &mut *mem.cast::<MemZone>() };

    let last = zone.items.pop();
    let mut found = None;

    if let Some(last) = last {
        if last.ptr == ptr {
            // this is it
            found = Some(last);
        } else {
            zone.items.push(last);

            let index = zone.items.iter().position(|item| item.ptr == ptr);

            if let Some(index) = index {
                let last = zone.items.remove(index);
                found = Some(last);
                zone.not_lifo += 1;
            }
        }
    }

    if let Some(item) = found {
        zone.total -= item.size;
        // SAFETY: only tracked allocations were returned by our libc
        // malloc callback; this pointer has been removed from the list
        // before it can be released again.
        unsafe { free(item.ptr) };
    } else {
        // Do not call libc free with a pointer we cannot establish came
        // from malloc. In particular, a repeated free must not double-free.
        zone.rogue += 1;
    }
}

fn mem_setup() -> libz_rs_sys::z_stream {
    let zone = MemZone {
        items: Vec::new(),
        total: 0,
        highwater: 0,
        limit: 0,
        not_lifo: 0,
        rogue: 0,
    };
    let zone = Box::new(zone);

    let stream = libz_rs_sys::z_stream {
        next_in: std::ptr::null_mut(),
        avail_in: 0,
        total_in: 0,
        next_out: std::ptr::null_mut(),
        avail_out: 0,
        total_out: 0,
        msg: std::ptr::null_mut(),
        state: std::ptr::null_mut(),
        zalloc: Some(mem_alloc),
        zfree: Some(mem_free),
        opaque: Box::into_raw(zone).cast::<c_void>(),
        data_type: 0,
        adler: 0,
        reserved: 0,
    };

    stream
}

fn mem_limit(stream: &mut libz_rs_sys::z_stream, limit: usize) {
    assert!(!stream.opaque.is_null());

    // SAFETY: stream.opaque holds the owned, live MemZone. The mutable
    // stream reference excludes concurrent use of the allocation callback.
    let zone = unsafe { &mut *stream.opaque.cast::<MemZone>() };
    zone.limit = limit;
}

fn mem_highwater(stream: &libz_rs_sys::z_stream) -> usize {
    assert!(!stream.opaque.is_null());
    // SAFETY: stream.opaque owns the initialized MemZone for the whole
    // lifetime of this stream until mem_done.
    unsafe { (*(stream.opaque.cast_const().cast::<MemZone>())).highwater }
}

fn mem_done(stream: &mut libz_rs_sys::z_stream) {
    assert!(!stream.opaque.is_null());

    extern "C" {
        fn free(p: *mut c_void);
    }

    // zone not wrapped in ManuallyDrop so it is free'd
    // SAFETY: this is the only transfer of ownership back from the raw
    // pointer created by Box::into_raw in mem_setup.
    let mut zone = unsafe { Box::from_raw(stream.opaque.cast::<MemZone>()) };

    let count = zone.items.len();

    for item in zone.items.drain(..) {
        // SAFETY: every tracked item came from libc malloc in mem_alloc
        // and has not yet been released by the matching free callback.
        unsafe { free(item.ptr) };
    }

    assert_eq!(
        (count, zone.total),
        (0, 0),
        "{} bytes in {count} blocks not freed",
        zone.total
    );
    assert_eq!(zone.not_lifo, 0, "{} frees not LIFO", zone.not_lifo);
    assert_eq!(zone.rogue, 0, "{} frees not recognized", zone.rogue);

    stream.opaque = std::ptr::null_mut();
    stream.zalloc = None;
    stream.zfree = None;
}

#[test]
fn mem_allocator_rejects_unknown_and_duplicate_free_without_touching_memory() {
    let mut stream = mem_setup();
    let opaque = stream.opaque;
    // SAFETY: mem_alloc receives an owned, live MemZone and small valid counts.
    let ptr = unsafe { mem_alloc(opaque, 1, 16) };
    assert!(!ptr.is_null());

    // The first free releases the malloc allocation, and the second is
    // deliberately given only the dangling pointer value, not read.
    // A rogue stack pointer must not be handed to libc free either.
    // SAFETY: the callback only compares unknown pointers to its tracked
    // allocations. It never dereferences or frees unmatched pointers.
    unsafe {
        mem_free(opaque, ptr);
        mem_free(opaque, ptr);
        let mut stack_byte = 0u8;
        mem_free(opaque, (&raw mut stack_byte).cast::<c_void>());
        mem_free(opaque, core::ptr::null_mut());
    }

    // SAFETY: the stream is still the exclusive owner of the accounting
    // zone; callbacks have returned, so this read cannot race them.
    let zone = unsafe { &mut *opaque.cast::<MemZone>() };
    assert_eq!(zone.total, 0);
    assert_eq!(zone.rogue, 2);
    assert!(zone.items.is_empty());
    zone.rogue = 0; // the test intentionally injected exactly two rogue frees
    mem_done(&mut stream);
}

#[test]
fn mem_allocator_rejects_overflow_without_unwinding_through_ffi() {
    let mut stream = mem_setup();

    // A u32 count * size exceeds usize on 32-bit targets. On 64-bit
    // targets the limit prevents an impractically large allocation.
    mem_limit(&mut stream, 1);
    assert!(unsafe { mem_alloc(stream.opaque, u32::MAX, u32::MAX) }.is_null());

    // The running total can overflow even when the new request fits
    // usize. The deliberately forged accounting is restored before cleanup.
    // SAFETY: opaque owns a live, exclusively accessed MemZone, and no
    // allocator callback runs concurrently with this test.
    {
        // SAFETY: opaque is our live allocation, and no allocator callback
        // runs while the mutable reference is held.
        let zone = unsafe { &mut *stream.opaque.cast::<MemZone>() };
        zone.limit = 0;
        zone.total = usize::MAX;
    }
    // SAFETY: opaque remains a live MemZone; no memory was allocated and
    // the deliberately overflowed accounting prevents any access to malloc.
    assert!(unsafe { mem_alloc(stream.opaque, 1, 1) }.is_null());
    // SAFETY: the callback completed and the same owned zone is still live.
    unsafe { (*stream.opaque.cast::<MemZone>()).total = 0 };

    mem_done(&mut stream);
}

/// Stateful callback fixture for the actual C ABI, exercising both the
/// single-span fast path and repeated short callback-fed input chunks.
struct BackRoundTrip {
    input: Vec<u8>,
    next: usize,
    chunk_size: usize,
    output: Vec<u8>,
    output_calls: usize,
}

/// Convert a caller's desired chunk into a C input callback byte count.
///
/// Returning both representations from one checked conversion ensures that
/// advancing the input cursor and reporting the C byte count cannot diverge
/// when the host usize can represent more than c_uint.
fn callback_chunk_size(requested: usize, remaining: usize) -> (usize, c_uint) {
    let max_count = usize::try_from(c_uint::MAX).unwrap_or(usize::MAX);
    let len = requested.min(remaining).min(max_count);
    match c_uint::try_from(len) {
        Ok(count) => (len, count),
        Err(_) => (0, 0),
    }
}

#[test]
fn back_round_trip_callback_count_never_truncates() {
    let (len, count) = callback_chunk_size(usize::MAX, usize::MAX);
    assert_eq!(len, usize::try_from(count).unwrap_or(usize::MAX));
    assert_eq!(count, c_uint::MAX);

    // 64-bit hosts can express one more byte than a C unsigned int.
    if let Some(larger_than_c_uint) = usize::try_from(c_uint::MAX)
        .ok()
        .and_then(|max| max.checked_add(1))
    {
        let (len, count) = callback_chunk_size(larger_than_c_uint, usize::MAX);
        assert_eq!(count, c_uint::MAX);
        assert_eq!(len, usize::try_from(c_uint::MAX).unwrap_or(usize::MAX));
    }

    assert_eq!(callback_chunk_size(7, 3), (3, 3));
    assert_eq!(callback_chunk_size(0, usize::MAX), (0, 0));
}

unsafe extern "C" fn read_back_round_trip(
    descriptor: *mut c_void,
    next: *mut *const c_uchar,
) -> c_uint {
    // SAFETY: descriptor is a uniquely borrowed BackRoundTrip fixture during
    // each synchronous inflateBack callback.
    let context = unsafe { &mut *descriptor.cast::<BackRoundTrip>() };
    let (len, count) = callback_chunk_size(context.chunk_size, context.input.len() - context.next);
    if count == 0 {
        // SAFETY: next is a valid output pointer supplied by inflateBack.
        unsafe { *next = core::ptr::null() };
        return 0;
    }
    // SAFETY: len is bounded by the remaining input bytes, whose Vec
    // remains unchanged for the duration of this inflateBack invocation.
    unsafe { *next = context.input.as_ptr().add(context.next) };
    context.next += len;
    count
}

unsafe extern "C" fn write_back_round_trip(
    descriptor: *mut c_void,
    bytes: *mut c_uchar,
    count: c_uint,
) -> c_int {
    // SAFETY: descriptor points to the live fixture; the output pointer covers
    // exactly count initialized bytes until this synchronous callback returns.
    let context = unsafe { &mut *descriptor.cast::<BackRoundTrip>() };
    let bytes = unsafe { core::slice::from_raw_parts(bytes, count as usize) };
    context.output.extend_from_slice(bytes);
    context.output_calls += 1;
    0
}

#[test]
fn inflate_back_round_trip_crosses_multiple_windows_and_input_chunks() {
    use libz_rs_sys::*;

    // Repeated medium-range patterns generate both dynamic Huffman symbols
    // and back-references, including those spanning callback output windows.
    let seed: Vec<u8> = (0..2048usize)
        .map(|i| (i.wrapping_mul(73) ^ (i >> 3) ^ (i >> 8)) as u8)
        .collect();
    let original: Vec<u8> = (0..100_123usize)
        .map(|i| seed[(i + i / 8192) % seed.len()])
        .collect();
    let mut compressed_storage = vec![0; zlib_rs::deflate::compress_bound(original.len())];
    let (compressed, code) = compress_slice(
        &mut compressed_storage,
        &original,
        DeflateConfig {
            window_bits: -15,
            ..DeflateConfig::default()
        },
    );
    assert_eq!(code, ReturnCode::Ok);
    let compressed = compressed.to_vec();

    for chunk_size in [7, 31, 4096, usize::MAX] {
        let mut fixture = BackRoundTrip {
            input: compressed.clone(),
            next: 0,
            chunk_size,
            output: Vec::with_capacity(original.len()),
            output_calls: 0,
        };
        let mut stream = mem_setup();
        let mut window = vec![0xa5; 32768];
        assert_eq!(
            unsafe {
                inflateBackInit_(
                    &raw mut stream,
                    15,
                    window.as_mut_ptr(),
                    VERSION,
                    STREAM_SIZE,
                )
            },
            Z_OK
        );
        let (count, avail_in) = callback_chunk_size(chunk_size, fixture.input.len());
        stream.next_in = fixture.input.as_mut_ptr();
        stream.avail_in = avail_in;
        fixture.next = count;
        let descriptor = (&raw mut fixture).cast::<c_void>();
        let result = unsafe {
            inflateBack(
                &raw mut stream,
                Some(read_back_round_trip),
                descriptor,
                Some(write_back_round_trip),
                descriptor,
            )
        };
        assert_eq!(result, Z_STREAM_END, "input chunk {chunk_size}");
        assert_eq!(fixture.output, original, "input chunk {chunk_size}");
        assert_eq!(fixture.output_calls, 4, "input chunk {chunk_size}");
        assert_eq!(unsafe { inflateBackEnd(&raw mut stream) }, Z_OK);
        mem_done(&mut stream);
    }
}

#[test]
fn inflate_back_init_allocates_state_only() {
    use libz_rs_sys::*;

    let mut regular = mem_setup();
    assert_eq!(
        unsafe { inflateInit_(&raw mut regular, VERSION, STREAM_SIZE) },
        Z_OK
    );
    let regular_highwater = mem_highwater(&regular);
    assert_eq!(unsafe { inflateEnd(&raw mut regular) }, Z_OK);
    mem_done(&mut regular);

    let mut back = mem_setup();
    let mut window = vec![0u8; 1 << 15];
    assert_eq!(
        unsafe { inflateBackInit_(&raw mut back, 15, window.as_mut_ptr(), VERSION, STREAM_SIZE) },
        Z_OK
    );
    let back_highwater = mem_highwater(&back);
    assert_eq!(unsafe { inflateBackEnd(&raw mut back) }, Z_OK);
    mem_done(&mut back);

    assert!(
        regular_highwater >= back_highwater + (1 << 15),
        "regular inflate used {regular_highwater} bytes vs {back_highwater} for inflateBack"
    );
}

#[test]
fn gzip_header_check() {
    use libz_rs_sys::*;

    let input: &[u8] = &[
        0x1f, 0x8b, 0x08, 0x1f, 0x44, 0x0a, 0x45, 0x65, 0x00, 0x03, 0x0e, 0x00, 0x54, 0x47, 0x0a,
        0x00, 0x45, 0x58, 0x54, 0x52, 0x41, 0x20, 0x44, 0x41, 0x54, 0x41, 0x74, 0x65, 0x73, 0x74,
        0x2e, 0x74, 0x78, 0x74, 0x00, 0x41, 0x20, 0x63, 0x6f, 0x6d, 0x6d, 0x65, 0x6e, 0x74, 0x00,
        0x3b, 0x92,
    ];

    let len = 1;
    let mut stream = mem_setup();

    // A gzip-header parsing test must fail if decoder initialization fails,
    // rather than silently skipping all assertions after the error.
    // SAFETY: mem_setup supplies the correct allocator callbacks and
    // live writable C stream storage for the gzip-wrapper decoder.
    let init_err = unsafe { inflateInit2_(&raw mut stream, 47, VERSION, STREAM_SIZE) };
    assert_eq!(init_err, Z_OK, "gzip header decoder initialization failed");

    let mut out = vec![0u8; len];
    let mut extra: [u8; 14] = [0; 14];
    let mut name: [u8; 9] = [0; 9];
    let mut comment: [u8; 10] = [0; 10];

    // Set header
    // See: https://www.zlib.net/manual.html
    let mut header = gz_header {
        text: 0,
        time: 0,
        xflags: 0,
        os: 0,
        extra: extra.as_mut_ptr(),
        extra_len: 0,
        extra_max: test_c_uint_len(extra.len()),
        name: name.as_mut_ptr(),
        name_max: test_c_uint_len(name.len()),
        comment: comment.as_mut_ptr(),
        comm_max: test_c_uint_len(comment.len()),
        hcrc: 0,
        done: 0,
    };

    // SAFETY: initialized stream and valid writable gzip header whose
    // metadata pointers refer to the live caller-owned arrays.
    let ret = unsafe { inflateGetHeader(&raw mut stream, &raw mut header) };
    assert_eq!(ReturnCode::from(ret), ReturnCode::Ok);

    stream.avail_in = test_c_uint_len(input.len());
    stream.next_in = input.as_ptr().cast_mut();

    stream.avail_out = test_c_uint_len(out.len());
    stream.next_out = out.as_mut_ptr();

    // SAFETY: all input bytes and one writable output byte remain live,
    // with accurate advertised C byte counts.
    let ret = unsafe { inflate(&raw mut stream, InflateFlush::NoFlush as _) };

    assert_eq!(ret, Z_OK);
    assert_eq!(usize::try_from(stream.total_in), Ok(input.len()));
    assert_eq!(stream.avail_in, 0, "gzip header input was not consumed");
    assert_eq!(stream.total_out, 0, "header-only gzip emitted payload");
    assert_eq!(out, [0u8; 1], "gzip header unexpectedly wrote output");

    assert_eq!(header.text, 1);
    assert_eq!(header.time, 1_699_023_428);
    assert_eq!(header.os, 3);
    assert_eq!(header.hcrc, 1);
    assert_eq!(header.done, 1);

    // Compare the exact bytes, including the NUL terminators. No UTF-8
    // decoding or panic branch is needed to test these C header fields.
    assert_eq!(&comment, b"A comment\0");
    assert_eq!(&name, b"test.txt\0");

    // Check header extra
    let extra_bytes = [84, 71, 10, 0, 69, 88, 84, 82, 65, 32, 68, 65, 84, 65];
    assert_eq!(extra_bytes, extra);

    // SAFETY: the decoder remains initialized; test raw-mode reset after
    // successfully processing the complete gzip metadata.
    let ret = unsafe { inflateReset2(&raw mut stream, -8) };
    assert_eq!(ret, Z_OK);

    // SAFETY: the initialized stream is closed exactly once.
    let ret = unsafe { inflateEnd(&raw mut stream) };
    assert_eq!(ret, Z_OK);

    mem_done(&mut stream);
}

fn inf(input: &[u8], _what: &str, step: usize, win: i32, len: usize, err: c_int) {
    use libz_rs_sys::*;

    let mut err = Some(err);

    let mut stream = mem_setup();

    // SAFETY: mem_setup provided writable ABI stream storage and valid
    // allocation callbacks; invalid win values are tested intentionally.
    let init_err = unsafe { inflateInit2_(&raw mut stream, win, VERSION, STREAM_SIZE) };
    if init_err != Z_OK {
        mem_done(&mut stream);
        // The single intentionally invalid initialization uses window_bits=1.
        // Matching the final expected status is not enough: otherwise a
        // failed setup could accidentally pass a test targeting inflate().
        assert_eq!(win, 1, "unexpected init failure in {_what}");
        assert_eq!(init_err, Z_STREAM_ERROR);
        return;
    }

    let mut out = vec![0u8; len];

    let mut extra: [u8; 1024] = [0; 1024];
    let mut name: [u8; 64] = [0; 64];
    let mut comment: [u8; 64] = [0; 64];

    // Set header
    // See: https://www.zlib.net/manual.html
    let mut header = gz_header {
        text: 0,
        time: 0,
        xflags: 0,
        os: 0,
        extra: extra.as_mut_ptr(),
        extra_len: 0,
        extra_max: test_c_uint_len(extra.len()),
        name: name.as_mut_ptr(),
        name_max: test_c_uint_len(name.len()),
        comment: comment.as_mut_ptr(),
        comm_max: test_c_uint_len(comment.len()),
        hcrc: 0,
        done: 0,
    };

    if win == 47 {
        // SAFETY: header points to live writable fields with capacities
        // equal to the actual allocated test arrays.
        let err = unsafe { inflateGetHeader(&raw mut stream, &raw mut header) };
        assert_eq!(ReturnCode::from(err), ReturnCode::Ok);
    }

    let mut have = input.len();
    let step = if step == 0 || step > have { have } else { step };

    stream.avail_in = test_c_uint_len(step);
    have -= step;
    stream.next_in = input.as_ptr().cast_mut();

    loop {
        // These deliberately tiny buffers are scratch output for testing
        // status transitions, not a cumulative decoded-output allocation.
        stream.avail_out = test_c_uint_len(out.len());
        stream.next_out = out.as_mut_ptr();
        let available_before = stream.avail_in;
        let total_out_before = stream.total_out;

        // SAFETY: the offered input is a live prefix of the remaining input
        // slice; the scratch buffer is live and disjoint, including len == 0.
        let ret = unsafe { inflate(&raw mut stream, InflateFlush::NoFlush as _) };
        assert!(
            stream.avail_in <= available_before,
            "inflate reported more input than the helper provided"
        );

        if let Some(expected) = err {
            assert_eq!(ret, expected, "unexpected first result in {_what}");
        }

        if !matches!(ret, Z_OK | Z_BUF_ERROR | Z_NEED_DICT) {
            break;
        }

        if matches!(ret, Z_NEED_DICT) {
            // SAFETY: the one-byte candidate dictionary resides inside
            // the live input fixture. Rejection is required by this test.
            let ret = unsafe { inflateSetDictionary(&raw mut stream, input.as_ptr(), 1) };
            assert_eq!(ret, Z_DATA_ERROR);

            // SAFETY: this intentional test-only state mutation restores
            // the dictionary-request state before trying a zero-byte span.
            unsafe { set_mode_dict(&mut stream) }
            // SAFETY: zero-length dictionary requires no dereference of out.
            let ret = unsafe { inflateSetDictionary(&raw mut stream, out.as_ptr(), 0) };
            assert_eq!(ret, Z_OK);

            // SAFETY: the initialized stream and caller-owned spans remain
            // live through this expected no-progress dictionary probe.
            let ret = unsafe { inflate(&raw mut stream, InflateFlush::NoFlush as _) };
            assert_eq!(ret, Z_BUF_ERROR);
        }

        let inflate_result = ret;
        let mut copy = z_stream::default();
        // SAFETY: source is an initialized live decoder, and copy is
        // writable C ABI storage with a separate destination address.
        let ret = unsafe { inflateCopy(&raw mut copy, &raw const stream) };
        assert_eq!(ret, Z_OK);
        assert_ne!(copy.state, stream.state, "inflateCopy shared state storage");

        // SAFETY: successful inflateCopy initialized an independent state
        // which is closed here exactly once.
        let ret = unsafe { inflateEnd(&raw mut copy) };
        assert_eq!(ret, Z_OK);

        // This intentionally tiny scratch buffer can cause Z_BUF_ERROR
        // without consuming input or producing output. Repeating that same
        // call forever cannot change the result, so end the coverage probe.
        if inflate_result == Z_BUF_ERROR
            && stream.avail_in == available_before
            && stream.total_out == total_out_before
        {
            break;
        }
        assert!(
            inflate_result == Z_NEED_DICT
                || stream.avail_in < available_before
                || stream.total_out > total_out_before,
            "inflate coverage probe stalled in {_what}"
        );

        // Only the initial return code is a fixed fixture expectation.
        err = None;

        // Existing next_in advanced by the number of consumed bytes;
        // re-advertise up to step contiguous bytes including any unconsumed
        // suffix, bounded by the complete remaining fixture.
        let still_offered = usize::try_from(stream.avail_in).unwrap_or(usize::MAX);
        assert!(still_offered <= step);
        assert!(have <= input.len().saturating_sub(still_offered));
        have += still_offered;
        stream.avail_in = test_c_uint_len(step.min(have));
        have -= usize::try_from(stream.avail_in).unwrap_or(usize::MAX);

        if stream.avail_in == 0 {
            break;
        }
    }

    // SAFETY: the coverage probe has an initialized stream; reset before
    // closing to exercise the alternate window configuration.
    let ret = unsafe { inflateReset2(&raw mut stream, -8) };
    assert_eq!(ret, Z_OK);

    // SAFETY: this stream was initialized and is ended exactly once.
    let ret = unsafe { inflateEnd(&raw mut stream) };
    assert_eq!(ret, Z_OK);

    mem_done(&mut stream);
}

#[test]
fn support() {
    use libz_rs_sys::*;

    let mut stream = mem_setup();

    let ret = unsafe {
        inflateInit_(
            &raw mut stream,
            zlibVersion(),
            test_c_stream_size::<z_stream>(),
        )
    };
    assert_eq!(ret, Z_OK);

    let ret = unsafe { inflateSetDictionary(&raw mut stream, std::ptr::null(), 0) };
    assert_eq!(ret, Z_STREAM_ERROR);

    let ret = unsafe { inflateEnd(&raw mut stream) };
    assert_eq!(ret, Z_OK);

    mem_done(&mut stream);
}

#[test]
fn raw_inflate_dictionary_rejects_null_nonzero_length_without_state_change() {
    use libz_rs_sys::*;

    let mut raw = MaybeUninit::<z_stream>::zeroed();
    // SAFETY: this is the correct ABI version and stream size; -15 enables raw DEFLATE.
    assert_eq!(
        unsafe {
            inflateInit2_(
                raw.as_mut_ptr(),
                -15,
                zlibVersion(),
                test_c_stream_size::<z_stream>(),
            )
        },
        Z_OK,
    );
    // SAFETY: initialization succeeded.
    let stream = unsafe { raw.assume_init_mut() };
    let mut length_before = 0;
    // SAFETY: the initialized stream and length output are valid; NULL dictionary means query.
    assert_eq!(
        unsafe { inflateGetDictionary(stream, core::ptr::null_mut(), &raw mut length_before) },
        Z_OK,
    );
    assert_eq!(length_before, 0);
    // SAFETY: NULL is a documented possible pointer, and the wrapper must reject
    // a nonzero length before attempting to materialize or dereference any slice.
    assert_eq!(
        unsafe { inflateSetDictionary(stream, core::ptr::null(), 1) },
        Z_STREAM_ERROR,
    );
    let mut length_after = 0;
    // SAFETY: initialized stream and length output remain valid.
    assert_eq!(
        unsafe { inflateGetDictionary(stream, core::ptr::null_mut(), &raw mut length_after) },
        Z_OK,
    );
    assert_eq!(length_after, length_before);
    // SAFETY: zero-length input does not require a backing dictionary buffer.
    assert_eq!(
        unsafe { inflateSetDictionary(stream, core::ptr::null(), 0) },
        Z_OK,
    );
    let valid = b"raw dictionary history";
    // SAFETY: valid is readable and remains live for the duration of the call.
    assert_eq!(
        unsafe { inflateSetDictionary(stream, valid.as_ptr(), test_c_uint_len(valid.len()),) },
        Z_OK,
    );
    let mut recovered = [0u8; 64];
    // SAFETY: a valid output span is provided; recovered is sufficiently large.
    assert_eq!(
        unsafe { inflateGetDictionary(stream, recovered.as_mut_ptr(), &raw mut length_after) },
        Z_OK,
    );
    assert_eq!(usize::try_from(length_after).ok(), Some(valid.len()));
    assert_eq!(&recovered[..valid.len()], valid);

    // Rejecting a NULL nonzero dictionary must leave existing window bytes
    // untouched too, not just an initially empty dictionary's length.
    // SAFETY: NULL with nonzero length is deliberately rejected before any
    // read; the stream already contains a valid dictionary.
    assert_eq!(
        unsafe { inflateSetDictionary(stream, core::ptr::null(), 1) },
        Z_STREAM_ERROR
    );
    recovered.fill(0xa5);
    // SAFETY: the destination spans the complete live 64-byte buffer and
    // the stream remains initialized with its original valid history.
    assert_eq!(
        unsafe { inflateGetDictionary(stream, recovered.as_mut_ptr(), &raw mut length_after) },
        Z_OK
    );
    assert_eq!(usize::try_from(length_after), Ok(valid.len()));
    assert_eq!(&recovered[..valid.len()], valid);

    // SAFETY: end the initialized stream exactly once.
    assert_eq!(unsafe { inflateEnd(stream) }, Z_OK);
}

#[test]
fn force_window_allocation() {
    inf(&[0x63, 0x00], "force window allocation", 0, -15, 1, Z_OK);
}

#[test]
fn force_window_replacement() {
    inf(
        &[0x63, 0x18, 0x05],
        "force window replacement",
        0,
        -8,
        259,
        Z_OK,
    );
}

#[test]
fn force_split_window_update() {
    inf(
        &[0x63, 0x18, 0x68, 0x30, 0xd0, 0x0, 0x0],
        "force split window update",
        4,
        -8,
        259,
        Z_OK,
    );
}

#[test]
fn use_fixed_blocks() {
    inf(&[0x03, 0x00], "use fixed blocks", 0, -15, 1, Z_STREAM_END);
}

#[test]
fn bad_window_size() {
    inf(&[], "bad window size", 0, 1, 0, Z_STREAM_ERROR);
}

#[test]
fn cover_wrap() {
    use libz_rs_sys::*;

    assert_eq!(unsafe { inflate(std::ptr::null_mut(), 0) }, Z_STREAM_ERROR);
    assert_eq!(unsafe { inflateEnd(std::ptr::null_mut()) }, Z_STREAM_ERROR);

    let mut strm = mem_setup();
    strm.avail_in = 0;
    strm.next_in = std::ptr::null_mut();
    // SAFETY: stream has an allocator and correct ABI version/size; the
    // -8 window size is a valid raw DEFLATE initialization.
    let mut ret = unsafe { inflateInit2_(&raw mut strm, -8, VERSION, STREAM_SIZE) };
    assert_eq!(ret, Z_OK, "cover_wrap failed to initialize the stream");
    let mut input = [0x63u8, 0x00];
    strm.avail_in = test_c_uint_len(input.len());
    strm.next_in = input.as_mut_ptr();

    // Keep decoder output separate from the variable holding its C return
    // codes. Writing decoded bytes through the latter pointer can silently
    // corrupt the test's assertions and is not a real output allocation.
    let mut scratch = [0u8; 1];
    strm.avail_out = test_c_uint_len(scratch.len());
    strm.next_out = scratch.as_mut_ptr();

    let dict = [0u8; 257];
    // SAFETY: initialized stream and a live 257-byte dictionary fixture.
    ret =
        unsafe { inflateSetDictionary(&raw mut strm, dict.as_ptr(), test_c_uint_len(dict.len())) };
    assert_eq!(ret, Z_OK);

    // ------------------------------

    let size = 2 * INFLATE_STATE_SIZE + 256;
    mem_limit(&mut strm, size);

    // SAFETY: the initialized stream remains live; the bit pattern is
    // supplied as a value, not a pointer.
    ret = unsafe { inflatePrime(&raw mut strm, 16, 0) };
    assert_eq!(ret, Z_OK);

    let mut input = [0x80u8, 0x00];
    strm.avail_in = test_c_uint_len(input.len());
    strm.next_in = input.as_mut_ptr();
    // SAFETY: next_in points to the currently live, bounded input fixture.
    ret = unsafe { inflateSync(&raw mut strm) };
    assert_eq!(ret, Z_DATA_ERROR);

    // SAFETY: the input buffer and one-byte output scratch are disjoint
    // live allocations, with capacities described in the z_stream fields.
    ret = unsafe { inflate(&raw mut strm, InflateFlush::NoFlush as _) };
    assert_eq!(ret, Z_STREAM_ERROR);

    let mut input = [0u8, 0, 0xFF, 0xFF];
    strm.avail_in = test_c_uint_len(input.len());
    strm.next_in = input.as_mut_ptr();
    // SAFETY: next_in points to the currently live, bounded input fixture.
    ret = unsafe { inflateSync(&raw mut strm) };
    assert_eq!(ret, Z_OK);
    // SAFETY: query a live initialized stream; the result is incidental
    // to this allocation-failure regression.
    let _ = unsafe { inflateSyncPoint(&raw mut strm) };

    let mut copy = z_stream::default();
    // SAFETY: source has valid live decoder state, destination is writable
    // C stream storage. The active allocator limit forces Z_MEM_ERROR.
    ret = unsafe { inflateCopy(&raw mut copy, &raw const strm) };
    assert_eq!(ret, Z_MEM_ERROR);

    mem_limit(&mut strm, 0);
    // SAFETY: no memory limit remains and the decoder state is live.
    ret = unsafe { inflateUndermine(&raw mut strm, 1) };
    assert_eq!(ret, Z_OK);

    // SAFETY: read-only query on a live initialized decoder state.
    let _ = unsafe { inflateMark(&raw const strm) };
    // SAFETY: the initialized stream is closed exactly once.
    ret = unsafe { inflateEnd(&raw mut strm) };
    assert_eq!(ret, Z_OK);
    mem_done(&mut strm);
}

#[test]
fn undermine_initializes_invalid_back_history_in_fragmented_and_fast_modes(
) -> Result<(), &'static str> {
    use libz_rs_sys::{inflate, inflateEnd, inflateInit2_, inflateUndermine};
    // Final fixed-Huffman block: length-3 match at distance 1, then EOB.
    // There are no preceding literals, so all three history bytes are missing.
    // In normal mode
    // it is Z_DATA_ERROR. inflateUndermine requests zlib's alternative:
    // explicitly synthesize zero history instead of reading uninitialized
    // window storage or panicking.
    const MALFORMED: &[u8] = &[0x03, 0x02, 0x00];

    for subvert in [1, 2, -1] {
        let mut observed = Vec::new();
        for &(output_chunk, trailing) in &[(1usize, 0usize), (512, 32)] {
            let mut source = MALFORMED.to_vec();
            source.extend(core::iter::repeat_n(0u8, trailing));
            let mut stream = mem_setup();
            assert_eq!(
                unsafe { inflateInit2_(&raw mut stream, -15, VERSION, STREAM_SIZE) },
                Z_OK,
            );
            assert_eq!(unsafe { inflateUndermine(&raw mut stream, subvert) }, Z_OK);
            stream.next_in = source.as_mut_ptr();
            stream.avail_in =
                u32::try_from(source.len()).map_err(|_| "test input exceeds C width")?;

            let mut result = Vec::new();
            let mut done = false;
            for _ in 0..256 {
                let mut output = [0xa5u8; 512];
                stream.next_out = output.as_mut_ptr();
                stream.avail_out =
                    u32::try_from(output_chunk).map_err(|_| "test output exceeds C width")?;
                let before_in = stream.avail_in;
                let code = unsafe { inflate(&raw mut stream, InflateFlush::NoFlush as i32) };
                let written = output_chunk - stream.avail_out as usize;
                result.extend_from_slice(&output[..written]);
                if code == Z_STREAM_END {
                    done = true;
                    break;
                }
                assert_eq!(code, Z_OK, "unexpected result for chunk {output_chunk}");
                assert!(
                    written != 0 || stream.avail_in != before_in,
                    "inflate must make progress for chunk {output_chunk}"
                );
            }
            assert!(
                done,
                "invalid-history recovery must finish for {output_chunk}"
            );
            assert!(!result.is_empty());
            assert!(
                result.iter().all(|&b| b == 0),
                "only synthesized zero history is allowed"
            );
            assert_eq!(unsafe { inflateEnd(&raw mut stream) }, Z_OK);
            mem_done(&mut stream);
            observed.push(result);
        }
        assert_eq!(
            observed[0], observed[1],
            "decoder paths must agree for {subvert}"
        );
    }
    Ok(())
}

#[test]
fn undermine_zero_restores_strict_history_validation() -> Result<(), &'static str> {
    use libz_rs_sys::{inflate, inflateEnd, inflateInit2_, inflateUndermine};

    let mut input = [0x03u8, 0x02, 0x00];
    let mut output = [0xa5u8; 32];
    let mut stream = mem_setup();
    assert_eq!(
        unsafe { inflateInit2_(&raw mut stream, -15, VERSION, STREAM_SIZE) },
        Z_OK
    );
    assert_eq!(unsafe { inflateUndermine(&raw mut stream, 1) }, Z_OK);
    assert_eq!(unsafe { inflateUndermine(&raw mut stream, 0) }, Z_OK);
    stream.next_in = input.as_mut_ptr();
    stream.avail_in = u32::try_from(input.len()).map_err(|_| "test input exceeds C width")?;
    stream.next_out = output.as_mut_ptr();
    stream.avail_out = u32::try_from(output.len()).map_err(|_| "test output exceeds C width")?;
    assert_eq!(
        unsafe { inflate(&raw mut stream, InflateFlush::NoFlush as i32) },
        Z_DATA_ERROR,
    );
    assert_eq!(unsafe { inflateEnd(&raw mut stream) }, Z_OK);
    mem_done(&mut stream);
    Ok(())
}

#[test]
fn inflate_mark_is_independent_of_unbound_io_buffers() {
    // inflateMark examines the initialized decode state. It does not require
    // the stream's caller-owned input/output buffers to remain attached between calls.
    let observed = assert_eq_rs_ng!({
        let mut raw = MaybeUninit::<z_stream>::zeroed();
        // SAFETY: pass the correct ABI version and stream size to initialize the stream.
        assert_eq!(
            unsafe {
                inflateInit_(
                    raw.as_mut_ptr(),
                    zlibVersion(),
                    core::mem::size_of::<z_stream>() as c_int,
                )
            },
            Z_OK
        );
        // SAFETY: inflateInit_ initialized the stream successfully.
        // SAFETY: successful inflateInit2_ initialized every field in the C ABI stream.
        let stream = unsafe { raw.assume_init_mut() };
        // SAFETY: the initialized inflate state is live; inflateMark only observes it.
        let before = unsafe { inflateMark(stream) };
        let compressed = [0x78u8, 0x9c, 0x03, 0, 0, 0, 0, 1];
        let mut output = [0u8; 8];
        stream.next_in = compressed.as_ptr().cast_mut();
        stream.avail_in = c_uint::try_from(compressed.len()).unwrap_or(c_uint::MAX);
        stream.next_out = output.as_mut_ptr();
        stream.avail_out = c_uint::try_from(output.len()).unwrap_or(c_uint::MAX);

        // SAFETY: the input and output buffers remain valid and disjoint.
        let decode_code = unsafe { inflate(stream, Z_NO_FLUSH) };
        // SAFETY: the initialized inflate state remains live; inflateMark is read-only.
        let with_io = unsafe { inflateMark(stream) };

        // Marking a live decoder after the caller buffers have been detached
        // must not change its answer, just like zlib-ng.
        stream.next_in = core::ptr::null_mut();
        stream.avail_in = 0;
        stream.next_out = core::ptr::null_mut();
        stream.avail_out = 0;
        // SAFETY: the initialized decoder state remains live; marking does not dereference I/O.
        let without_io = unsafe { inflateMark(stream) };
        // SAFETY: terminate the successfully initialized stream exactly once.
        let end_code = unsafe { inflateEnd(stream) };
        (before, decode_code, with_io, without_io, end_code)
    });
    assert_eq!(observed.0, -65_536);
    assert_eq!(observed.1, Z_STREAM_END);
    assert_eq!(observed.2, observed.3);
    assert_eq!(observed.4, Z_OK);
}

#[test]
fn inflate_mark_reports_unfinished_stored_block_without_io_buffers() {
    // Valid zlib header followed by a final stored block with 20 payload bytes.
    // Stop immediately after the block header: inflateMark must report the
    // remaining copy length even after pointers to caller buffers are detached.
    let observed = assert_eq_rs_ng!({
        let mut raw = MaybeUninit::<z_stream>::zeroed();
        // SAFETY: correct version and ABI stream size for initialization.
        assert_eq!(
            unsafe {
                inflateInit_(
                    raw.as_mut_ptr(),
                    zlibVersion(),
                    core::mem::size_of::<z_stream>() as c_int,
                )
            },
            Z_OK
        );
        // SAFETY: initialization succeeded.
        // SAFETY: successful inflateInit2_ initialized every field in the C ABI stream.
        let stream = unsafe { raw.assume_init_mut() };
        let compressed_prefix = [0x78u8, 0x01, 0x01, 20, 0, 235, 255];
        let mut output = [0u8; 32];
        stream.next_in = compressed_prefix.as_ptr().cast_mut();
        stream.avail_in = c_uint::try_from(compressed_prefix.len()).unwrap_or(c_uint::MAX);
        stream.next_out = output.as_mut_ptr();
        stream.avail_out = c_uint::try_from(output.len()).unwrap_or(c_uint::MAX);
        // SAFETY: both caller buffers are live, valid, and disjoint.
        let status = unsafe { inflate(stream, Z_NO_FLUSH) };
        // SAFETY: the initialized inflate state remains live; inflateMark is read-only.
        let with_io = unsafe { inflateMark(stream) };
        stream.next_in = core::ptr::null_mut();
        stream.avail_in = 0;
        stream.next_out = core::ptr::null_mut();
        stream.avail_out = 0;
        // SAFETY: marking only inspects the initialized stream state.
        let without_io = unsafe { inflateMark(stream) };
        // SAFETY: finalizes the initialized stream once.
        let end_status = unsafe { inflateEnd(stream) };
        (status, with_io, without_io, end_status)
    });
    assert_eq!(observed.0, Z_OK);
    assert_eq!(observed.1, -65_536 + 20);
    assert_eq!(observed.1, observed.2);
    assert_eq!(observed.3, Z_OK);
}

#[test]
fn bad_gzip_method() {
    inf(
        &[0x1f, 0x8b, 0x00, 0x00],
        "bad gzip method",
        0,
        31,
        0,
        Z_DATA_ERROR,
    )
}
#[test]
fn bad_gzip_flags() {
    inf(
        &[0x1f, 0x8b, 0x08, 0x80],
        "bad gzip flags",
        0,
        31,
        0,
        Z_DATA_ERROR,
    )
}
#[test]
fn bad_zlib_method() {
    inf(&[0x77, 0x85], "bad zlib method", 0, 15, 0, Z_DATA_ERROR)
}
#[test]
fn set_window_size_from_header() {
    inf(&[0x08, 0x99], "set window size from header", 0, 0, 0, Z_OK)
}
#[test]
fn bad_zlib_window_size() {
    inf(&[0x78, 0x9c], "bad zlib window size", 0, 8, 0, Z_DATA_ERROR)
}
#[test]
fn check_adler32() {
    inf(
        &[0x78, 0x9c, 0x63, 0x0, 0x0, 0x0, 0x1, 0x0, 0x1],
        "check adler32",
        0,
        15,
        1,
        Z_STREAM_END,
    )
}
#[test]
fn bad_header_crc() {
    inf(
        &[
            0x1f, 0x8b, 0x8, 0x1e, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x00,
        ],
        "bad header crc",
        0,
        47,
        1,
        Z_DATA_ERROR,
    )
}

#[test]
fn check_gzip_length() {
    inf(
        &[
            0x1f, 0x8b, 0x8, 0x2, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x1d, 0x26, 0x3, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
        ],
        "check gzip length",
        0,
        47,
        0,
        Z_STREAM_END,
    )
}

#[test]
fn bad_zlib_header_check() {
    inf(
        &[0x78, 0x90],
        "bad zlib header check",
        0,
        47,
        0,
        Z_DATA_ERROR,
    )
}

#[test]
fn good_zlib_header_check() {
    inf(
        &[
            0x1f, 0x8b, 0x08, 0x1f, 0x44, 0x0a, 0x45, 0x65, 0x00, 0x03, 0x0e, 0x00, 0x54, 0x47,
            0x0a, 0x00, 0x45, 0x58, 0x54, 0x52, 0x41, 0x20, 0x44, 0x41, 0x54, 0x41, 0x74, 0x65,
            0x73, 0x74, 0x2e, 0x74, 0x78, 0x74, 0x00, 0x41, 0x20, 0x63, 0x6f, 0x6d, 0x6d, 0x65,
            0x6e, 0x74, 0x00, 0x3b, 0x92,
        ],
        "good zlib header check",
        0,
        47,
        0,
        Z_OK,
    )
}

#[test]
fn need_dictionary() {
    inf(
        &[0x08, 0xb8, 0x0, 0x0, 0x0, 0x1],
        "need dictionary",
        0,
        8,
        0,
        Z_NEED_DICT,
    )
}

#[test]
fn compute_adler32() {
    inf(&[0x78, 0x9c, 0x63, 0x0], "compute adler32", 0, 15, 1, Z_OK)
}

/* do a raw inflate of data in hexadecimal with both inflate and inflateBack */
fn try_inflate(input: &[u8], expected_err: c_int) -> c_int {
    use libz_rs_sys::*;

    // Inflate in Z_TREES mode to exercise malformed-block handling. Keep
    // all emitted output instead of restarting the write pointer on each call.
    let capacity = input.len().checked_mul(8);
    assert!(capacity.is_some(), "test output capacity overflow");
    let mut output = vec![0u8; capacity.unwrap_or_default().max(1)];
    let mut stream = mem_setup();
    stream.next_in = core::ptr::null_mut();
    stream.avail_in = 0;

    // SAFETY: mem_setup provided live C stream storage, allocator
    // callbacks and matching ABI version/size.
    let mut result = unsafe {
        inflateInit2_(
            &raw mut stream,
            if expected_err < 0 { 47 } else { -15 },
            VERSION,
            STREAM_SIZE,
        )
    };
    assert_eq!(result, Z_OK);
    stream.next_in = input.as_ptr().cast_mut();
    stream.avail_in = test_c_uint_len(input.len());
    stream.next_out = output.as_mut_ptr();
    stream.avail_out = test_c_uint_len(output.len());
    let mut stalled_calls = 0;

    loop {
        let before_in = stream.avail_in;
        let before_out = stream.total_out;
        let before_type = stream.data_type;
        // SAFETY: next_in refers to the live input fixture and next_out
        // refers into a disjoint, writable suffix of the output allocation.
        result = unsafe { inflate(&raw mut stream, InflateFlush::Trees as _) };
        assert!(!matches!(result, Z_STREAM_ERROR | Z_MEM_ERROR));
        assert!(
            stream.avail_in <= before_in,
            "inflate increased input availability"
        );
        assert!(
            usize::try_from(stream.total_out).unwrap_or(usize::MAX) <= output.len(),
            "inflate output exceeds allocated buffer"
        );
        if matches!(result, Z_DATA_ERROR | Z_NEED_DICT | Z_STREAM_END) {
            break;
        }
        assert!(
            matches!(result, Z_OK | Z_BUF_ERROR),
            "unexpected Z_TREES status {result}"
        );

        if stream.avail_out == 0 {
            let written = usize::try_from(stream.total_out).unwrap_or(usize::MAX);
            assert_eq!(written, output.len(), "output position mismatch");
            let old_len = output.len();
            let new_len = old_len.checked_mul(2).unwrap_or(old_len);
            assert!(new_len > old_len, "output capacity overflow");
            output.resize(new_len, 0);
            stream.next_out = output[written..].as_mut_ptr();
            stream.avail_out = test_c_uint_len(new_len - written);
        }
        if stream.avail_in == 0 {
            break;
        }
        // A Z_TREES boundary need not consume input or produce output,
        // but infinitely repeating an identical state is not progress.
        if stream.avail_in < before_in
            || stream.total_out > before_out
            || stream.data_type != before_type
        {
            stalled_calls = 0;
        } else {
            stalled_calls += 1;
            assert!(stalled_calls < 3, "Z_TREES decoder stalled");
        }
    }
    if expected_err != Z_OK {
        assert_eq!(result, Z_DATA_ERROR);
    }
    // SAFETY: the successfully initialized stream is ended exactly once.
    assert_eq!(unsafe { inflateEnd(&raw mut stream) }, Z_OK);
    mem_done(&mut stream);

    if expected_err >= 0 {
        result = try_inflate_back(input, expected_err);
    }
    result
}

fn try_inflate_back(input: &[u8], expected_err: c_int) -> c_int {
    use libz_rs_sys::*;

    let mut window = vec![0u8; 32768];
    let mut stream = mem_setup();

    // SAFETY: the custom allocator and 32 KiB window are live, and the
    // supplied version and stream structure match the C ABI.
    let result = unsafe {
        inflateBackInit_(
            &raw mut stream,
            15,
            window.as_mut_ptr(),
            VERSION,
            STREAM_SIZE,
        )
    };
    assert_eq!(result, Z_OK);
    stream.avail_in = test_c_uint_len(input.len());
    stream.next_in = input.as_ptr().cast_mut();

    // SAFETY: input and the window remain live and disjoint during the
    // callback-driven decode; the null pull context requests no more
    // input beyond the initial span, and push discards output bytes.
    let result = unsafe {
        inflateBack(
            &raw mut stream,
            Some(pull),
            core::ptr::null_mut(),
            Some(push),
            core::ptr::null_mut(),
        )
    };
    assert_ne!(result, Z_STREAM_ERROR);
    if expected_err != Z_OK && result != Z_BUF_ERROR {
        assert_eq!(result, Z_DATA_ERROR);
    }

    // SAFETY: matching teardown for the initialized inflateBack state.
    assert_eq!(unsafe { inflateBackEnd(&raw mut stream) }, Z_OK);
    mem_done(&mut stream);
    result
}

#[test]
fn invalid_stored_block_length() {
    try_inflate(&[0, 0, 0, 0, 0], Z_STREAM_END);
}

#[test]
fn fixed() {
    try_inflate(&[3, 0], Z_OK);
}

#[test]
fn invalid_block_type() {
    try_inflate(&[6], Z_STREAM_END);
}
#[test]
fn stored() {
    try_inflate(&[0x1, 0x1, 0x0, 0xfe, 0xff, 0x0], Z_OK);
}

#[test]
fn too_many_length_or_distance_symbols() {
    try_inflate(&[0xfc, 0x0, 0x0], Z_STREAM_END);
}

#[test]
fn invalid_code_lengths_set() {
    try_inflate(&[0x4, 0x0, 0xfe, 0xff], Z_STREAM_END);
}
#[test]
fn invalid_bit_length_repeat_1() {
    try_inflate(&[0x4, 0x0, 0x24, 0x49, 0x0], Z_STREAM_END);
}
#[test]
fn invalid_bit_length_repeat_2() {
    try_inflate(&[0x4, 0x0, 0x24, 0xe9, 0xff, 0xff], Z_STREAM_END);
}
#[test]
fn invalid_code_missing_end_of_block() {
    try_inflate(&[0x4, 0x0, 0x24, 0xe9, 0xff, 0x6d], Z_STREAM_END);
}
#[test]
fn invalid_literal_lengths_set() {
    try_inflate(
        &[
            0x4, 0x80, 0x49, 0x92, 0x24, 0x49, 0x92, 0x24, 0x71, 0xff, 0xff, 0x93, 0x11, 0x0,
        ],
        Z_STREAM_END,
    );
}
#[test]
fn invalid_distances_set() {
    try_inflate(
        &[
            0x4, 0x80, 0x49, 0x92, 0x24, 0x49, 0x92, 0x24, 0xf, 0xb4, 0xff, 0xff, 0xc3, 0x84,
        ],
        Z_STREAM_END,
    );
}
#[test]
fn invalid_literal_length_code() {
    try_inflate(
        &[
            0x4, 0xc0, 0x81, 0x8, 0x0, 0x0, 0x0, 0x0, 0x20, 0x7f, 0xeb, 0xb, 0x0, 0x0,
        ],
        Z_STREAM_END,
    );
}
#[test]
fn invalid_distance_code() {
    try_inflate(&[0x2, 0x7e, 0xff, 0xff], Z_STREAM_END);
}
#[test]
fn invalid_distance_too_far_back() {
    try_inflate(
        &[
            0xc, 0xc0, 0x81, 0x0, 0x0, 0x0, 0x0, 0x0, 0x90, 0xff, 0x6b, 0x4, 0x0,
        ],
        Z_STREAM_END,
    );
}

#[test]
fn incorrect_data_check() {
    try_inflate(
        &[
            0x1f, 0x8b, 0x8, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x3, 0x0, 0x0, 0x0, 0x0, 0x1,
        ],
        Z_ERRNO,
    );
}
#[test]
fn incorrect_length_check() {
    try_inflate(
        &[
            0x1f, 0x8b, 0x8, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x3, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x1,
        ],
        Z_ERRNO,
    );
}

#[test]
fn zlib_ng_pull_17() {
    try_inflate(
        &[
            0x5, 0xc0, 0x21, 0xd, 0x0, 0x0, 0x0, 0x80, 0xb0, 0xfe, 0x6d, 0x2f, 0x91, 0x6c,
        ],
        Z_OK,
    );
}
#[test]
fn long_code() {
    try_inflate(
        &[
            0x5, 0xe0, 0x81, 0x91, 0x24, 0xcb, 0xb2, 0x2c, 0x49, 0xe2, 0xf, 0x2e, 0x8b, 0x9a, 0x47,
            0x56, 0x9f, 0xfb, 0xfe, 0xec, 0xd2, 0xff, 0x1f,
        ],
        Z_OK,
    );
}
#[test]
fn length_extra() {
    try_inflate(
        &[
            0xed, 0xc0, 0x1, 0x1, 0x0, 0x0, 0x0, 0x40, 0x20, 0xff, 0x57, 0x1b, 0x42, 0x2c, 0x4f,
        ],
        Z_OK,
    );
}
#[test]
fn long_distance_and_extra() {
    try_inflate(
        &[
            0xed, 0xcf, 0xc1, 0xb1, 0x2c, 0x47, 0x10, 0xc4, 0x30, 0xfa, 0x6f, 0x35, 0x1d, 0x1,
            0x82, 0x59, 0x3d, 0xfb, 0xbe, 0x2e, 0x2a, 0xfc, 0xf, 0xc,
        ],
        Z_OK,
    );
}
#[test]
fn window_end() {
    try_inflate(
        &[
            0xed, 0xc0, 0x81, 0x0, 0x0, 0x0, 0x0, 0x80, 0xa0, 0xfd, 0xa9, 0x17, 0xa9, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x6,
        ],
        Z_OK,
    );
}

#[test]
fn back_too_far() {
    try_inflate_back(
        &[75, 39, 108, 46, 8, 59, 81, 81, 0, 81, 81, 0, 0, 0, 10, 10],
        Z_DATA_ERROR,
    );
}

#[test]
fn inflate_fast_type_return() {
    inf(
        &[0x2, 0x8, 0x20, 0x80, 0x0, 0x3, 0x0],
        "inflate_fast TYPE return",
        0,
        -15,
        258,
        Z_STREAM_END,
    );
}
#[test]
fn window_wrap() {
    inf(
        &[0x63, 0x18, 0x5, 0x40, 0xc, 0x0],
        "window wrap",
        3,
        -8,
        300,
        Z_OK,
    );
}

#[test]
fn fast_length_extra_bits() {
    inf(
        &[
            0xe5, 0xe0, 0x81, 0xad, 0x6d, 0xcb, 0xb2, 0x2c, 0xc9, 0x01, 0x1e, 0x59, 0x63, 0xae,
            0x7d, 0xee, 0xfb, 0x4d, 0xfd, 0xb5, 0x35, 0x41, 0x68, 0xff, 0x7f, 0x0f, 0x0, 0x0, 0x0,
        ],
        "fast length extra bits",
        0,
        -8,
        258,
        Z_DATA_ERROR,
    );
}
#[test]
fn fast_distance_extra_bits() {
    inf(
        &[
            0x25, 0xfd, 0x81, 0xb5, 0x6d, 0x59, 0xb6, 0x6a, 0x49, 0xea, 0xaf, 0x35, 0x6, 0x34,
            0xeb, 0x8c, 0xb9, 0xf6, 0xb9, 0x1e, 0xef, 0x67, 0x49, 0x50, 0xfe, 0xff, 0xff, 0x3f,
            0x0, 0x0,
        ],
        "fast distance extra bits",
        0,
        -8,
        258,
        Z_DATA_ERROR,
    )
}
#[test]
fn fast_invalid_distance_code() {
    inf(
        &[0x3, 0x7e, 0x0, 0x0, 0x0, 0x0, 0x0],
        "fast invalid distance code",
        0,
        -8,
        258,
        Z_DATA_ERROR,
    );
}
#[test]
fn fast_invalid_literal_length_code() {
    inf(
        &[0x1b, 0x7, 0x0, 0x0, 0x0, 0x0, 0x0],
        "fast invalid literal/length code",
        0,
        -8,
        258,
        Z_DATA_ERROR,
    );
}
#[test]
fn fast_2nd_level_codes_and_too_far_back() {
    inf(
        &[
            0xd, 0xc7, 0x1, 0xae, 0xeb, 0x38, 0xc, 0x4, 0x41, 0xa0, 0x87, 0x72, 0xde, 0xdf, 0xfb,
            0x1f, 0xb8, 0x36, 0xb1, 0x38, 0x5d, 0xff, 0xff, 0x0,
        ],
        "fast 2nd level codes and too far back",
        0,
        -8,
        258,
        Z_DATA_ERROR,
    );
}
#[test]
fn very_common_case() {
    inf(
        &[0x63, 0x18, 0x5, 0x8c, 0x10, 0x8, 0x0, 0x0, 0x0, 0x0],
        "very common case",
        0,
        -8,
        259,
        Z_OK,
    );
}
#[test]
fn contiguous_and_wrap_around_window() {
    inf(
        &[
            0x63, 0x60, 0x60, 0x18, 0xc9, 0x0, 0x8, 0x18, 0x18, 0x18, 0x26, 0xc0, 0x28, 0x0, 0x29,
            0x0, 0x0, 0x0,
        ],
        "contiguous and wrap around window",
        6,
        -8,
        259,
        Z_OK,
    );
}
#[test]
fn copy_direct_from_output() {
    inf(
        &[0x63, 0x0, 0x3, 0x0, 0x0, 0x0, 0x0, 0x0],
        "copy direct from output",
        0,
        -8,
        259,
        Z_STREAM_END,
    );
}

#[test]
fn cover_cve_2022_37434() {
    inf(
        &[
            0x1f, 0x8b, 0x08, 0x04, 0x61, 0x62, 0x63, 0x64, 0x61, 0x62, 0x52, 0x51, 0x1f, 0x8b,
            0x08, 0x04, 0x61, 0x62, 0x63, 0x64, 0x61, 0x62, 0x52, 0x51, 0x1f, 0x8b, 0x08, 0x04,
            0x61, 0x62, 0x63, 0x64, 0x61, 0x62, 0x52, 0x51, 0x1f, 0x8b, 0x08, 0x04, 0x61, 0x62,
            0x63, 0x64, 0x61, 0x62, 0x52, 0x51,
        ],
        "wtf",
        13,
        47,
        12,
        Z_OK,
    );
}

fn uncompress_help(input: &[u8]) -> Vec<u8> {
    use libz_rs_sys::*;

    let mut decoded = vec![0u8; 1 << 16];
    let mut decoded_len = test_c_ulong_len(decoded.len());

    // SAFETY: the compressed input and writable output allocations are
    // disjoint and live through this synchronous C call, and their actual
    // capacities are represented without narrowing in c_ulong.
    let status = unsafe {
        uncompress(
            decoded.as_mut_ptr(),
            &raw mut decoded_len,
            input.as_ptr(),
            test_c_ulong_len(input.len()),
        )
    };
    assert_eq!(
        status,
        Z_OK,
        "uncompress failed: {:?}",
        ReturnCode::from(status)
    );
    let size = usize::try_from(decoded_len).unwrap_or(usize::MAX);
    assert!(size <= decoded.len(), "uncompress exceeded output capacity");
    decoded.truncate(size);
    decoded
}

#[test]
fn hello_world_uncompressed() {
    // "Hello World!" compressed at level 0
    let deflated = [
        0x78, 0x01, 0x01, 0x0d, 0x00, 0xf2, 0xff, 0x48, 0x65, 0x6c, 0x6c, 0x6f, 0x20, 0x57, 0x6f,
        0x72, 0x6c, 0x64, 0x21, 0x0a, 0x20, 0x91, 0x04, 0x48,
    ];

    let output = uncompress_help(&deflated);
    assert_eq!(output, b"Hello World!\n");
}

#[test]
fn copy_direct_from_next_in_to_next_out() {
    assert_eq_rs_ng!({
        // Stored block: precisely two decoded bytes, with no trailing
        // compressed input ignored by the uncompress2 API.
        let input = [120, 1, 1, 2, 0, 253, 255, 6, 10, 0, 24, 0, 17];
        let expected = [6u8, 10];
        let mut output = vec![0u8; 1 << 16];
        let mut output_len = test_c_ulong_len(output.len());
        let mut input_len = test_c_ulong_len(input.len());

        // SAFETY: the source is a live immutable compressed fixture and
        // destination is a distinct live writable allocation. Each C length
        // pointer refers to independently owned writable local storage.
        let status = unsafe {
            uncompress2(
                output.as_mut_ptr(),
                &raw mut output_len,
                input.as_ptr(),
                &raw mut input_len,
            )
        };
        assert_eq!(status, Z_OK, "stored block uncompress failed");
        assert_eq!(usize::try_from(input_len), Ok(input.len()));
        assert_eq!(usize::try_from(output_len), Ok(expected.len()));
        assert_eq!(&output[..expected.len()], &expected);
        output.truncate(expected.len());
        output
    });
}

/// Unit tests for the _z variants of (un)compress(2).
#[test]
fn compress_uncompress_z() {
    use libz_rs_sys::{compress, compress_z, uncompress, uncompress_z};

    let source = b"there is no horizon";

    let mut compr_z = vec![0u8; 128];
    let mut compr_z_len = compr_z.len();

    let mut compr = vec![0u8; 128];
    let mut compr_len = test_c_ulong_len(compr.len());

    // SAFETY: the live source and distinct compressor output spans
    // are valid for the declared checked C buffer lengths.
    unsafe {
        assert_eq!(
            compress_z(
                compr_z.as_mut_ptr(),
                &raw mut compr_z_len,
                source.as_ptr(),
                source.len(),
            ),
            Z_OK,
        );
        assert_eq!(
            compress(
                compr.as_mut_ptr(),
                &raw mut compr_len,
                source.as_ptr(),
                test_c_ulong_len(source.len()),
            ),
            Z_OK,
        );
    }

    assert_eq!(usize::try_from(compr_len), Ok(compr_z_len));
    assert!(compr_z_len <= compr_z.len() && compr_z_len <= compr.len());
    assert_eq!(&compr_z[..compr_z_len], &compr[..compr_z_len]);

    let mut compr2_z = vec![0u8; 128];
    let mut compr2_z_len = compr2_z.len();

    let mut compr2 = vec![0u8; 128];
    let mut compr2_len = test_c_ulong_len(compr2.len());

    // SAFETY: both level-9 encoders read the same live input and write
    // to separate output buffers with correctly described capacities.
    unsafe {
        assert_eq!(
            libz_rs_sys::compress2_z(
                compr2_z.as_mut_ptr(),
                &raw mut compr2_z_len,
                source.as_ptr(),
                source.len(),
                9,
            ),
            Z_OK,
        );
        assert_eq!(
            libz_rs_sys::compress2(
                compr2.as_mut_ptr(),
                &raw mut compr2_len,
                source.as_ptr(),
                test_c_ulong_len(source.len()),
                9,
            ),
            Z_OK,
        );
    }

    assert_eq!(usize::try_from(compr2_len), Ok(compr2_z_len));
    assert!(compr2_z_len <= compr2_z.len() && compr2_z_len <= compr2.len());
    assert_eq!(&compr2_z[..compr2_z_len], &compr2[..compr2_z_len]);

    assert!(compr_z_len <= compr_z.len());
    assert!(compr2_z_len <= compr2_z.len());

    let mut dest_z = vec![0u8; source.len()];
    let mut dest_z_len = dest_z.len();

    let mut dest = vec![0u8; source.len()];
    let mut dest_len = test_c_ulong_len(dest.len());

    // SAFETY: all three decompressors have live, disjoint input/output
    // spans and the reported compressed lengths were bounded above.
    unsafe {
        assert_eq!(
            uncompress_z(
                dest_z.as_mut_ptr(),
                &raw mut dest_z_len,
                compr_z.as_ptr(),
                compr_z_len,
            ),
            Z_OK,
        );
        assert_eq!(
            uncompress(
                dest.as_mut_ptr(),
                &raw mut dest_len,
                compr.as_ptr(),
                compr_len
            ),
            Z_OK,
        );

        let mut dest2 = vec![0u8; source.len()];
        let mut dest2_len = dest2.len();
        let mut source2_len = compr_z_len;
        assert_eq!(
            libz_rs_sys::uncompress2_z(
                dest2.as_mut_ptr(),
                &raw mut dest2_len,
                compr_z.as_ptr(),
                &raw mut source2_len,
            ),
            Z_OK,
        );

        assert_eq!(source2_len, compr_z_len);
        assert_eq!(dest2_len, source.len());
        assert_eq!(&dest2[..dest2_len], source);
    }

    assert_eq!(usize::try_from(dest_len), Ok(dest_z_len));
    assert_eq!(dest_z_len, source.len());
    assert_eq!(&dest_z[..dest_z_len], source);
    assert_eq!(&dest[..dest_z_len], source);
}

#[test]
fn hello_world_fixed() {
    // "Hello World!" compressed with `minideflate -k -f -9`
    let deflated = [
        0x78, 0x01, 0xf3, 0x48, 0xcd, 0xc9, 0xc9, 0x57, 0x08, 0xcf, 0x2f, 0xca, 0x49, 0x51, 0xe4,
        0x02, 0x00, 0x20, 0x91, 0x04, 0x48,
    ];
    let output = uncompress_help(&deflated);
    assert_eq!(output, b"Hello World!\n");
}

#[test]
fn hello_world_dynamic() {
    // smallest input that the fuzzer found that triggers a dynamic block
    let input = "\n\0\0\0\0l\0\nl\0l\0l\u{1}llll\n";
    let deflated = [
        120, 156, 5, 193, 177, 1, 0, 0, 8, 195, 160, 184, 246, 86, 254, 159, 133, 85, 105, 146,
        131, 61, 24, 141, 3, 128,
    ];

    let output = uncompress_help(&deflated);
    assert_eq!(output, input.as_bytes());
}

#[test]
fn inflate_adler() {
    use libz_rs_sys::*;

    // The compressed fixture contains the 44 text bytes, not the NUL
    // terminator in the old expected string. A prefix-only comparison
    // previously hid that disagreement.
    const ORIGINAL: &[u8] = b"The quick brown fox jumped over the lazy dog";

    const COMPRESSED: [u8; 52] = [
        0x78, 0x9c, 0x0b, 0xc9, 0x48, 0x55, 0x28, 0x2c, 0xcd, 0x4c, 0xce, 0x56, 0x48, 0x2a, 0xca,
        0x2f, 0xcf, 0x53, 0x48, 0xcb, 0xaf, 0x50, 0xc8, 0x2a, 0xcd, 0x2d, 0x48, 0x4d, 0x51, 0xc8,
        0x2f, 0x4b, 0x2d, 0x52, 0x28, 0xc9, 0x48, 0x55, 0xc8, 0x49, 0xac, 0xaa, 0x54, 0x48, 0xc9,
        0x4f, 0x07, 0x00, 0x6b, 0x93, 0x10, 0x30,
    ];

    let mut stream = z_stream {
        next_in: std::ptr::null_mut(),
        avail_in: 0,
        total_in: 0,
        next_out: std::ptr::null_mut(),
        avail_out: 0,
        total_out: 0,
        msg: std::ptr::null_mut(),
        state: std::ptr::null_mut(),
        zalloc: None,
        zfree: None,
        opaque: std::ptr::null_mut(),
        data_type: 0,
        adler: 0,
        reserved: 0,
    };

    // SAFETY: the caller owns writable, ABI-compatible stream storage;
    // the version/size match this C API and the selected window bits
    // enable zlib/gzip wrapper detection.
    let err = unsafe { inflateInit2_(&raw mut stream, 32 + MAX_WBITS, VERSION, STREAM_SIZE) };
    assert_eq!(err, Z_OK);

    let mut uncompressed = [0u8; 1024];

    stream.next_in = COMPRESSED.as_ptr().cast_mut();
    stream.avail_in = test_c_uint_len(COMPRESSED.len());
    stream.next_out = uncompressed.as_mut_ptr();
    stream.avail_out = test_c_uint_len(uncompressed.len());

    // SAFETY: compressed input and writable output are live and disjoint,
    // with correct C availability counts.
    let err = unsafe { inflate(&raw mut stream, Z_NO_FLUSH) };
    assert_eq!(err, Z_STREAM_END);
    assert_eq!(
        stream.avail_in, 0,
        "trailing compressed bytes were not read"
    );
    assert_eq!(usize::try_from(stream.total_in), Ok(COMPRESSED.len()));
    assert_eq!(usize::try_from(stream.total_out), Ok(ORIGINAL.len()));
    assert_eq!(&uncompressed[..ORIGINAL.len()], ORIGINAL);
    assert_eq!(stream.adler, 0x6b93_1030);

    // SAFETY: the initialized decoder is closed exactly once.
    let err = unsafe { inflateEnd(&raw mut stream) };
    assert_eq!(err, Z_OK);
}

#[test]
fn inflate_get_header_non_gzip_stream() {
    use libz_rs_sys::*;

    let mut stream = mem_setup();

    let win = 15; // i.e. zlib compression (and not gzip)
    let init_err = unsafe { inflateInit2_(&raw mut stream, win, VERSION, STREAM_SIZE) };
    assert_eq!(init_err, Z_OK, "non-gzip decoder initialization failed");

    let mut header = gz_header::default();

    assert_eq!(
        unsafe { inflateGetHeader(&raw mut stream, &raw mut header) },
        ReturnCode::StreamError as i32
    );

    let ret = unsafe { inflateEnd(&raw mut stream) };
    assert_eq!(ret, Z_OK);

    mem_done(&mut stream);
}

#[test]
fn inflate_window_bits_0_is_15() {
    let input = b"Hello World!\n";

    let mut compressed = [0; 64];
    let (compressed, err) = compress_slice(&mut compressed, input, DeflateConfig::new(6));
    assert_eq!(err, ReturnCode::Ok);

    let config = InflateConfig { window_bits: 15 };
    let mut output_15 = [0; 64];
    let (output_15, err) = decompress_slice(&mut output_15, compressed, config);
    assert_eq!(err, ReturnCode::Ok);

    let config = InflateConfig { window_bits: 0 };
    let mut output_0 = [0; 64];
    let (output_0, err) = decompress_slice(&mut output_0, compressed, config);
    assert_eq!(err, ReturnCode::Ok);

    // for window size 0, the default of 15 is picked
    // NOTE: the window size does not actually influence
    // the output for an input this small.
    assert_eq!(output_15, output_0);

    assert_eq!(output_15, input);
}

#[test]
fn inflate_total_in_counter_wraps_as_c_unsigned_long() {
    assert_eq_rs_ng!({
        let mut raw = MaybeUninit::<z_stream>::zeroed();
        // SAFETY: initialize the zlib stream with the matching ABI struct size and version.
        assert_eq!(
            unsafe {
                inflateInit_(
                    raw.as_mut_ptr(),
                    zlibVersion(),
                    core::mem::size_of::<z_stream>() as c_int,
                )
            },
            Z_OK,
        );
        // SAFETY: inflateInit_ succeeded and initialized the stream.
        // SAFETY: successful inflateInit2_ initialized every field in the C ABI stream.
        let stream = unsafe { raw.assume_init_mut() };
        let mut compressed = [0x78u8, 0x9c, 0x03, 0, 0, 0, 0, 1];
        let mut out = [0u8; 1];
        stream.total_in = c_ulong::MAX - 1;
        stream.next_in = compressed.as_mut_ptr();
        stream.avail_in = 8; // Fixed-size RFC1950 empty stream.
        stream.next_out = out.as_mut_ptr();
        stream.avail_out = 1; // Fixed-size scratch output.
                              // SAFETY: all buffers remain allocated, and the stream is initialized.
        assert_eq!(unsafe { inflate(stream, Z_FINISH) }, Z_STREAM_END);
        let observed = (stream.total_in, stream.total_out, stream.avail_in);
        // SAFETY: close a successfully initialized stream exactly once.
        assert_eq!(unsafe { inflateEnd(stream) }, Z_OK);
        observed
    });
}

#[test]
fn inflate_sync_counter_wraps_as_c_unsigned_long() {
    assert_eq_rs_ng!({
        let mut raw = MaybeUninit::<z_stream>::zeroed();
        // SAFETY: initialize the zlib stream with the matching ABI struct size and version.
        assert_eq!(
            unsafe {
                inflateInit_(
                    raw.as_mut_ptr(),
                    zlibVersion(),
                    core::mem::size_of::<z_stream>() as c_int,
                )
            },
            Z_OK,
        );
        // SAFETY: inflateInit_ succeeded and initialized the stream.
        // SAFETY: successful inflateInit2_ initialized every field in the C ABI stream.
        let stream = unsafe { raw.assume_init_mut() };
        let mut marker = [0u8, 0, 0xff, 0xff];
        stream.total_in = c_ulong::MAX - 1;
        stream.next_in = marker.as_mut_ptr();
        stream.avail_in = 4; // Four-byte recovery marker.
                             // SAFETY: the marker is a valid readable input buffer for the initialized stream.
        assert_eq!(unsafe { inflateSync(stream) }, Z_OK);
        let observed = (stream.total_in, stream.avail_in);
        // SAFETY: close a successfully initialized stream exactly once.
        assert_eq!(unsafe { inflateEnd(stream) }, Z_OK);
        observed
    });
}

#[test]
fn inflate_sync_preserves_output_total_after_recovery() {
    assert_eq_rs_ng!({
        // A non-final stored DEFLATE block yields "ABC", followed by a
        // reserved block type (BTYPE=3) that requires recovery.
        // The 00 00 ff ff marker precedes a final stored block "DEF".
        let mut before_sync = [0x00u8, 0x03, 0x00, 0xfc, 0xff, b'A', b'B', b'C', 0x06];
        let mut after_sync = [
            0x00u8, 0x00, 0xff, 0xff, 0x01, 0x03, 0x00, 0xfc, 0xff, b'D', b'E', b'F',
        ];
        let mut output = [0xa5u8; 16];
        let mut raw = MaybeUninit::<z_stream>::zeroed();

        // SAFETY: the stream is writable and the version/size match this ABI.
        let init_status = unsafe {
            inflateInit2_(
                raw.as_mut_ptr(),
                -15,
                zlibVersion(),
                c_int::try_from(core::mem::size_of::<z_stream>()).unwrap_or(c_int::MAX),
            )
        };
        assert_eq!(init_status, Z_OK);
        // SAFETY: successful inflateInit2_ initialized every field in the C ABI stream.
        let stream = unsafe { raw.assume_init_mut() };
        stream.next_in = before_sync.as_mut_ptr();
        stream.avail_in = c_uint::try_from(before_sync.len()).unwrap_or(c_uint::MAX);
        stream.next_out = output.as_mut_ptr();
        stream.avail_out = c_uint::try_from(output.len()).unwrap_or(c_uint::MAX);

        // SAFETY: the provided input and output spans stay live and disjoint.
        let first_status = unsafe { inflate(stream, Z_NO_FLUSH) };
        assert_eq!(first_status, Z_DATA_ERROR);
        assert_eq!(stream.total_out, 3);
        assert_eq!(&output[..3], b"ABC");

        stream.next_in = after_sync.as_mut_ptr();
        stream.avail_in = c_uint::try_from(after_sync.len()).unwrap_or(c_uint::MAX);
        // SAFETY: the recovery search sees the complete live input span.
        let sync_status = unsafe { inflateSync(stream) };
        assert_eq!(sync_status, Z_OK);
        let total_at_sync = stream.total_out;

        // SAFETY: the remaining DEFLATE span and output allocation stay valid.
        let final_status = unsafe { inflate(stream, Z_FINISH) };
        let total_after = stream.total_out;
        let input_remaining = stream.avail_in;
        // SAFETY: successfully initialized stream is freed once.
        let end_status = unsafe { inflateEnd(stream) };
        assert_eq!(end_status, Z_OK);
        assert_eq!(&output[..6], b"ABCDEF");

        (
            sync_status,
            total_at_sync,
            final_status,
            total_after,
            input_remaining,
        )
    });
}

#[test]
fn uncompress_edge_cases() {
    let config = InflateConfig { window_bits: 15 };

    let (result, err) = decompress_slice(&mut [], &[], config);
    assert_eq!(err, ReturnCode::DataError);
    assert!(result.is_empty());

    let mut output = [0; 1];
    let (result, err) = decompress_slice(&mut output, &[], config);
    assert_eq!(err, ReturnCode::DataError);
    assert!(result.is_empty());

    let input = b"Hello World!\n";

    let mut compressed = [0; 64];
    let (compressed, err) = compress_slice(&mut compressed, input, DeflateConfig::new(6));
    assert_eq!(err, ReturnCode::Ok);

    let (result, err) = decompress_slice(&mut [], compressed, config);
    assert_eq!(err, ReturnCode::DataError);
    assert!(result.is_empty());
}

/// Convert a test's real buffer extent to the C ABI byte-count type.
/// Passing a wrapped length would silently make the test exercise less data.
fn test_c_uint_len(len: usize) -> c_uint {
    let count = c_uint::try_from(len);
    assert!(count.is_ok(), "C ABI test span exceeds unsigned-int limit");
    count.unwrap_or_default()
}

/// An API taking C unsigned long must not silently compress or decompress a
/// truncated caller-supplied input/output capacity.
fn test_c_ulong_len(len: usize) -> c_ulong {
    let result = c_ulong::try_from(len);
    assert!(
        result.is_ok(),
        "C unsigned long is too narrow for test buffer"
    );
    result.unwrap_or_default()
}

/// Check the actual C stream structure size rather than passing a cast that
/// could wrap. Both the fork and the system zlib use this in differential tests.
fn test_c_stream_size<T>() -> c_int {
    let size = c_int::try_from(core::mem::size_of::<T>());
    assert!(size.is_ok(), "C ABI stream structure exceeds int limit");
    size.unwrap_or_default()
}

#[test]
fn gzip_header_fields_insufficient_space() {
    let chunk_size = 16;

    let input = b"Hello World\n";

    let extra =
        "Scheduling and executing async tasks is a job handled by an async runtime, such as\0";
    let name =
        "tokio, async-std, and smol. You've probably used them at some point, either directly or\0";
    let comment =
        "indirectly. They, along with many frameworks that require async, do their best to hide\0";

    let config = DeflateConfig {
        window_bits: 31,
        ..Default::default()
    };

    let mut stream = MaybeUninit::<libz_rs_sys::z_stream>::zeroed();

    // SAFETY: writable C stream storage, valid gzip settings and
    // the matching ABI version and checked C structure size.
    let err = unsafe {
        libz_rs_sys::deflateInit2_(
            stream.as_mut_ptr(),
            config.level,
            config.method as i32,
            config.window_bits,
            config.mem_level,
            config.strategy as i32,
            libz_rs_sys::zlibVersion(),
            test_c_stream_size::<libz_rs_sys::z_stream>(),
        )
    };
    assert_eq!(err, 0);

    // SAFETY: successful deflateInit2_ initialized the entire C stream.
    let stream = unsafe { stream.assume_init_mut() };

    let mut header = libz_rs_sys::gz_header {
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

    // SAFETY: header and its metadata buffers are live and readable
    // until compression is finished.
    let err = unsafe { libz_rs_sys::deflateSetHeader(stream, &raw mut header) };
    assert_eq!(err, 0);

    let mut output_rs = [0u8; 512];
    stream.next_out = output_rs.as_mut_ptr();
    stream.avail_out = test_c_uint_len(output_rs.len());

    for chunk in input.chunks(chunk_size) {
        stream.next_in = chunk.as_ptr().cast_mut();
        stream.avail_in = test_c_uint_len(chunk.len());

        // SAFETY: the chunk is a live input span, and output_rs remains a
        // writable destination until the stream is ended.
        let err = unsafe { libz_rs_sys::deflate(stream, InflateFlush::NoFlush as _) };

        assert_eq!(err, ReturnCode::Ok as i32, "{:?}", stream.msg);
        assert_eq!(stream.avail_in, 0, "deflate dropped an input chunk");
    }

    // SAFETY: input was fully consumed and next_out refers to the
    // remaining writable suffix of the compressed output buffer.
    let err = unsafe { libz_rs_sys::deflate(stream, InflateFlush::Finish as _) };
    assert_eq!(ReturnCode::from(err), ReturnCode::StreamEnd);

    assert_eq!(
        usize::try_from(stream.total_in),
        Ok(input.len()),
        "deflate did not consume the complete payload"
    );
    let compressed_len = usize::try_from(stream.total_out).unwrap_or(usize::MAX);
    assert!(
        compressed_len <= output_rs.len(),
        "deflate output exceeds its buffer"
    );
    let output_rs = &mut output_rs[..compressed_len];

    // SAFETY: deflateInit2_ succeeded and this is the only stream end.
    let err = unsafe { libz_rs_sys::deflateEnd(stream) };
    assert_eq!(ReturnCode::from(err), ReturnCode::Ok);

    assert_eq_rs_ng!({
        let mut stream = MaybeUninit::<z_stream>::zeroed();

        // SAFETY: writable C stream storage and valid gzip configuration,
        // version and checked ABI structure size.
        let err = unsafe {
            inflateInit2_(
                stream.as_mut_ptr(),
                config.window_bits,
                libz_rs_sys::zlibVersion(),
                test_c_stream_size::<z_stream>(),
            )
        };
        assert_eq!(err, 0);

        // SAFETY: successful inflateInit2_ initialized every C stream field.
        let stream = unsafe { stream.assume_init_mut() };

        let mut output = [0u8; 64];
        stream.next_out = output.as_mut_ptr();
        stream.avail_out = test_c_uint_len(output.len());

        let mut extra_buf = [0u8; 64];
        assert!(extra_buf.len() < extra.len());
        let mut name_buf = [0u8; 1];
        assert!(name_buf.len() < name.len());
        let mut comment_buf = [0u8; 64];
        assert!(comment_buf.len() < comment.len());

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

        // SAFETY: the header's live writable buffers are described by
        // their checked actual C capacities.
        let err = unsafe { inflateGetHeader(stream, &raw mut header) };
        assert_eq!(err, 0);

        let mut completed = false;
        for chunk in output_rs.chunks_mut(chunk_size) {
            stream.next_in = chunk.as_mut_ptr();
            stream.avail_in = test_c_uint_len(chunk.len());

            // SAFETY: the initialized inflate stream reads the supplied
            // compressed bytes and writes into the live output allocation.
            let err = unsafe { inflate(stream, InflateFlush::NoFlush as _) };
            assert!(
                matches!(err, Z_OK | Z_STREAM_END),
                "gzip inflate failed while parsing a chunk: {err}"
            );
            assert_eq!(stream.avail_in, 0, "inflate left unconsumed gzip bytes");
            if err == Z_STREAM_END {
                completed = true;
                break;
            }
        }

        assert!(completed, "gzip input ended without reaching Z_STREAM_END");
        assert_eq!(usize::try_from(stream.total_in), Ok(output_rs.len()));
        assert_eq!(usize::try_from(stream.total_out), Ok(input.len()));
        assert_eq!(&output[..input.len()], input);
        assert_eq!(header.done, 1, "gzip header parsing never completed");
        assert_eq!(header.extra_len, test_c_uint_len(extra.len()));

        // SAFETY: the initialized inflate stream is closed exactly once.
        let err = unsafe { inflateEnd(stream) };
        assert_eq!(err, ReturnCode::Ok as i32);

        assert!(!header.extra.is_null());
        assert_eq!(extra_buf.as_slice(), &extra.as_bytes()[..extra_buf.len()]);
        assert!(!header.name.is_null());
        assert_eq!(name_buf.as_slice(), &name.as_bytes()[..name_buf.len()]);
        assert!(!header.comment.is_null());
        assert_eq!(
            comment_buf.as_slice(),
            &comment.trim_end_matches('\0').as_bytes()[..comment_buf.len()]
        );

        (extra_buf, name_buf, comment_buf)
    });
}

#[test]
fn gzip_chunked_1_byte() {
    gzip_chunked(1);
}

#[test]
fn gzip_chunked_2_bytes() {
    gzip_chunked(2);
}

#[test]
fn gzip_chunked_15_bytes() {
    gzip_chunked(15);
}

#[test]
fn gzip_chunked_16_bytes() {
    gzip_chunked(16);
}

#[test]
fn gzip_chunked_17_bytes() {
    gzip_chunked(17);
}

#[test]
fn gzip_chunked_32_bytes() {
    gzip_chunked(32);
}

#[test]
fn gzip_chunked_512_bytes() {
    gzip_chunked(512);
}

fn gzip_chunked(chunk_size: usize) {
    let input = b"Hello World\n";

    let extra =
        "Scheduling and executing async tasks is a job handled by an async runtime, such as\0";
    let name =
        "tokio, async-std, and smol. You’ve probably used them at some point, either directly or\0";
    let comment =
        "indirectly. They, along with many frameworks that require async, do their best to hide\0";

    let config = DeflateConfig {
        window_bits: 31,
        ..Default::default()
    };

    let mut stream = MaybeUninit::<libz_rs_sys::z_stream>::zeroed();

    // SAFETY: writable C stream storage, valid gzip settings and
    // the matching ABI version and checked C structure size.
    let err = unsafe {
        libz_rs_sys::deflateInit2_(
            stream.as_mut_ptr(),
            config.level,
            config.method as i32,
            config.window_bits,
            config.mem_level,
            config.strategy as i32,
            libz_rs_sys::zlibVersion(),
            test_c_stream_size::<libz_rs_sys::z_stream>(),
        )
    };
    assert_eq!(err, 0);

    // SAFETY: successful deflateInit2_ initialized the entire C stream.
    let stream = unsafe { stream.assume_init_mut() };

    let mut header = libz_rs_sys::gz_header {
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

    // SAFETY: header and its metadata buffers are live and readable
    // until compression is finished.
    let err = unsafe { libz_rs_sys::deflateSetHeader(stream, &raw mut header) };
    assert_eq!(err, 0);

    let mut output_rs = [0u8; 512];
    stream.next_out = output_rs.as_mut_ptr();
    stream.avail_out = test_c_uint_len(output_rs.len());

    for chunk in input.chunks(chunk_size) {
        stream.next_in = chunk.as_ptr().cast_mut();
        stream.avail_in = test_c_uint_len(chunk.len());

        // SAFETY: the chunk is a live input span, and output_rs remains a
        // writable destination until the stream is ended.
        let err = unsafe { libz_rs_sys::deflate(stream, InflateFlush::NoFlush as _) };

        assert_eq!(err, ReturnCode::Ok as i32, "{:?}", stream.msg);
        assert_eq!(stream.avail_in, 0, "deflate dropped an input chunk");
    }

    // SAFETY: input was fully consumed and next_out refers to the
    // remaining writable suffix of the compressed output buffer.
    let err = unsafe { libz_rs_sys::deflate(stream, InflateFlush::Finish as _) };
    assert_eq!(ReturnCode::from(err), ReturnCode::StreamEnd);

    assert_eq!(
        usize::try_from(stream.total_in),
        Ok(input.len()),
        "deflate did not consume the complete payload"
    );
    let compressed_len = usize::try_from(stream.total_out).unwrap_or(usize::MAX);
    assert!(
        compressed_len <= output_rs.len(),
        "deflate output exceeds its buffer"
    );
    let output_rs = &mut output_rs[..compressed_len];

    // SAFETY: deflateInit2_ succeeded and this is the only stream end.
    let err = unsafe { libz_rs_sys::deflateEnd(stream) };
    assert_eq!(ReturnCode::from(err), ReturnCode::Ok);

    {
        use libz_rs_sys::{
            gz_header, inflate, inflateEnd, inflateGetHeader, inflateInit2_, z_stream,
        };

        let mut stream = MaybeUninit::<z_stream>::zeroed();

        // SAFETY: writable C stream storage and valid gzip configuration,
        // version and checked ABI structure size.
        let err = unsafe {
            inflateInit2_(
                stream.as_mut_ptr(),
                config.window_bits,
                libz_rs_sys::zlibVersion(),
                test_c_stream_size::<z_stream>(),
            )
        };
        assert_eq!(err, 0);

        // SAFETY: successful inflateInit2_ initialized every C stream field.
        let stream = unsafe { stream.assume_init_mut() };

        let mut output = [0u8; 64];
        stream.next_out = output.as_mut_ptr();
        stream.avail_out = test_c_uint_len(output.len());

        let mut extra_buf = [0u8; 64];
        let mut name_buf = [0u8; 64];
        let mut comment_buf = [0u8; 256];

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

        // SAFETY: the header's live writable buffers are described by
        // their checked actual C capacities.
        let err = unsafe { inflateGetHeader(stream, &raw mut header) };
        assert_eq!(err, 0);

        let mut completed = false;
        for chunk in output_rs.chunks_mut(chunk_size) {
            stream.next_in = chunk.as_mut_ptr();
            stream.avail_in = test_c_uint_len(chunk.len());

            // SAFETY: the initialized inflate stream reads the supplied
            // compressed bytes and writes into the live output allocation.
            let err = unsafe { inflate(stream, InflateFlush::NoFlush as _) };
            assert!(
                matches!(err, Z_OK | Z_STREAM_END),
                "gzip inflate failed while parsing a chunk: {err}"
            );
            assert_eq!(stream.avail_in, 0, "inflate left unconsumed gzip bytes");
            if err == Z_STREAM_END {
                completed = true;
                break;
            }
        }

        assert!(completed, "gzip input ended without reaching Z_STREAM_END");
        assert_eq!(usize::try_from(stream.total_in), Ok(output_rs.len()));
        assert_eq!(usize::try_from(stream.total_out), Ok(input.len()));
        assert_eq!(&output[..input.len()], input);
        assert_eq!(header.done, 1, "gzip header parsing never completed");
        assert_eq!(header.extra_len, test_c_uint_len(extra.len()));

        // SAFETY: the initialized inflate stream is closed exactly once.
        let err = unsafe { inflateEnd(stream) };
        assert_eq!(err, ReturnCode::Ok as i32);

        assert!(!header.extra.is_null());
        assert_eq!(extra_buf.as_slice(), &extra.as_bytes()[..extra_buf.len()]);
        assert!(!header.name.is_null());
        assert_eq!(name_buf.as_slice(), &name.as_bytes()[..name_buf.len()]);
        assert!(!header.comment.is_null());
        // Scan only within the allocated header buffer, never beyond its end.
        assert_eq!(
            CStr::from_bytes_until_nul(&comment_buf)
                .ok()
                .map(CStr::to_bytes),
            Some(comment.trim_end_matches('\0').as_bytes())
        );
    }
}

#[test]
fn chunked_output_rs() {
    use libz_rs_sys::*;

    let input = [99u8, 96, 192, 11, 24, 25, 0];

    let mut stream = MaybeUninit::<z_stream>::zeroed();

    // SAFETY: the zeroed stream pointer is writable and version/size
    // match the C ABI; -15 requests raw DEFLATE decoding.
    let err = unsafe { inflateInit2_(stream.as_mut_ptr(), -15, VERSION, STREAM_SIZE) };
    assert_eq!(ReturnCode::from(err), ReturnCode::Ok);

    // SAFETY: a successful initialization wrote the full C stream.
    let stream = unsafe { stream.assume_init_mut() };

    stream.next_in = input.as_ptr().cast_mut();
    stream.avail_in = test_c_uint_len(input.len());

    let mut output = [0; 33];

    stream.next_out = output.as_mut_ptr();
    stream.avail_out = 32;

    // SAFETY: the 7-byte compressed input and first 32 bytes of output
    // remain live and non-overlapping during the decode call.
    let err = unsafe { inflate(stream, InflateFlush::NoFlush as _) };
    assert_eq!(ReturnCode::from(err), ReturnCode::Ok);

    assert_eq!(stream.avail_out, 0);
    assert_eq!(stream.total_out, 32);

    stream.avail_out = 1;

    // SAFETY: the stream has advanced its output pointer to the final
    // writable byte, and avail_out is limited to that remaining byte.
    let err = unsafe { inflate(stream, InflateFlush::Finish as _) };
    assert_eq!(ReturnCode::from(err), ReturnCode::StreamEnd);

    // SAFETY: the successful inflateInit2_ call initialized this stream
    // and it has not yet been closed.
    assert_eq!(unsafe { inflateEnd(stream) }, Z_OK);

    assert_eq!(stream.total_out, 33);
}

#[test]
fn version_error() {
    use libz_rs_sys::*;

    let mut stream = core::mem::MaybeUninit::zeroed();

    // SAFETY: a writable ABI stream pointer is supplied for each init
    // attempt. Negative cases intentionally provide invalid version data,
    // which the API must reject before it accesses any stream state.
    let ret = unsafe {
        inflateInit_(
            stream.as_mut_ptr(),
            zlibVersion(),
            test_c_stream_size::<z_stream>(),
        )
    };
    assert_eq!(ret, Z_OK);

    // SAFETY: the first initialization succeeded, and the stream is live.
    let ret = unsafe { inflateEnd(stream.as_mut_ptr()) };
    assert_eq!(ret, Z_OK);

    // invalid stream size
    // SAFETY: the version is valid and the deliberately incorrect
    // structure-size argument must be rejected before initialization.
    let ret = unsafe { inflateInit_(stream.as_mut_ptr(), zlibVersion(), 1) };
    assert_eq!(ret, Z_VERSION_ERROR);

    // version is null
    // SAFETY: a writable ABI stream pointer is supplied for each init
    // attempt. Negative cases intentionally provide invalid version data,
    // which the API must reject before it accesses any stream state.
    let ret = unsafe {
        inflateInit_(
            stream.as_mut_ptr(),
            core::ptr::null(),
            test_c_stream_size::<z_stream>(),
        )
    };
    assert_eq!(ret, Z_VERSION_ERROR);

    // version is invalid
    // SAFETY: a writable ABI stream pointer is supplied for each init
    // attempt. Negative cases intentionally provide invalid version data,
    // which the API must reject before it accesses any stream state.
    let ret = unsafe {
        inflateInit_(
            stream.as_mut_ptr(),
            c"!".as_ptr(),
            test_c_stream_size::<z_stream>(),
        )
    };
    assert_eq!(ret, Z_VERSION_ERROR);

    // version is invalid, inflateInit2_
    // SAFETY: the version buffer is NUL-terminated and lives throughout
    // the call; the invalid version is intentionally rejected by the API.
    let ret = unsafe {
        inflateInit2_(
            stream.as_mut_ptr(),
            1,
            c"!".as_ptr(),
            test_c_stream_size::<z_stream>(),
        )
    };
    assert_eq!(ret, Z_VERSION_ERROR);
}

#[test]
fn issue_109() {
    let input = &include_bytes!("test-data/issue-109.gz")[10..][..32758];

    let window_bits = -15;

    assert_eq_rs_ng!(unsafe {
        let mut output: Vec<u8> = Vec::with_capacity(1 << 15);

        let mut buf = [0; 8192];

        let mut stream = MaybeUninit::<z_stream>::zeroed();

        let err = {
            inflateInit2_(
                stream.as_mut_ptr(),
                window_bits,
                zlibVersion(),
                core::mem::size_of::<z_stream>() as c_int,
            )
        };
        assert_eq!(ReturnCode::from(err), ReturnCode::Ok);

        let stream = { stream.assume_init_mut() };

        stream.next_in = input.as_ptr().cast_mut();
        let input_size = c_uint::try_from(input.len());
        assert!(
            input_size.is_ok(),
            "inflate fixture exceeds c_uint input limit"
        );
        stream.avail_in = input_size.unwrap_or_default();

        while stream.avail_in != 0 {
            let available_before = stream.avail_in;
            stream.next_out = buf.as_mut_ptr();
            stream.avail_out = c_uint::try_from(buf.len()).unwrap_or_default();

            let err = { inflate(stream, InflateFlush::NoFlush as _) };
            let remaining = stream.avail_out as usize;
            assert!(
                remaining <= buf.len(),
                "inflate reported invalid output capacity"
            );
            let produced = buf.len() - remaining;
            output.extend_from_slice(&buf[..produced]);

            assert!(
                matches!(
                    ReturnCode::from(err),
                    ReturnCode::Ok | ReturnCode::BufError | ReturnCode::StreamEnd
                ),
                "unexpected inflate status: {err}"
            );
            if err == Z_STREAM_END {
                break;
            }
            // A truncated test stream may end without Z_STREAM_END, but
            // a decoder that consumes no bytes and produces nothing is stuck.
            assert!(
                stream.avail_in < available_before || produced != 0,
                "inflate made no progress with {} input bytes remaining",
                stream.avail_in
            );
        }

        let err = inflateEnd(stream);
        assert_eq!(ReturnCode::from(err), ReturnCode::Ok);

        output
    });
}

#[test]
fn window_match_bug() {
    // Regression for a match-copy error that wrote incorrect bytes at the
    // current output cursor. The expected full output was independently
    // decoded using reference zlib with raw DEFLATE window_bits=-10.
    let input = include_bytes!("test-data/window-match-bug.zraw");
    let output = assert_eq_rs_ng!({
        let mut decoded = Vec::with_capacity(1 << 15);
        let mut scratch = [0u8; 402];
        let mut raw = MaybeUninit::<z_stream>::zeroed();

        // SAFETY: writable C stream storage, raw-DEFLATE window mode,
        // matching ABI version and checked stream structure size.
        let err = unsafe {
            inflateInit2_(
                raw.as_mut_ptr(),
                -10,
                zlibVersion(),
                test_c_stream_size::<z_stream>(),
            )
        };
        assert_eq!(ReturnCode::from(err), ReturnCode::Ok);
        // SAFETY: successful inflateInit2_ initialized every stream field.
        let stream = unsafe { raw.assume_init_mut() };
        stream.next_in = input.as_ptr().cast_mut();
        stream.avail_in = test_c_uint_len(input.len());

        loop {
            stream.next_out = scratch.as_mut_ptr();
            stream.avail_out = test_c_uint_len(scratch.len());
            let remaining_before = stream.avail_in;

            // SAFETY: input and independent writable scratch remain live
            // throughout each synchronous decoder invocation.
            let err = unsafe { inflate(stream, InflateFlush::Finish as _) };
            let remaining_out = usize::try_from(stream.avail_out).unwrap_or(usize::MAX);
            assert!(remaining_out <= scratch.len(), "invalid output capacity");
            let produced = scratch.len() - remaining_out;
            decoded.extend_from_slice(&scratch[..produced]);

            match ReturnCode::from(err) {
                ReturnCode::BufError => {
                    assert_eq!(remaining_out, 0, "buffer error without full output");
                    assert!(
                        produced > 0 || stream.avail_in < remaining_before,
                        "decoder stalled with input remaining"
                    );
                }
                ReturnCode::StreamEnd => break,
                other => panic!("unexpected raw DEFLATE decoder status: {other:?}"),
            }
        }

        assert_eq!(stream.avail_in, 0, "trailing raw DEFLATE bytes ignored");
        assert_eq!(usize::try_from(stream.total_in), Ok(input.len()));
        assert_eq!(usize::try_from(stream.total_out), Ok(decoded.len()));
        // SAFETY: successfully initialized decoder is closed exactly once.
        assert_eq!(unsafe { inflateEnd(stream) }, Z_OK);
        decoded
    });

    assert_eq!(
        output.len(),
        1284,
        "window-match regression output length changed"
    );
    assert_eq!(
        zlib_rs::crc32::crc32(0, &output),
        0x336c_0bf1,
        "window-match regression output differs from reference zlib"
    );
}

#[test]
fn op_len_edge_case() {
    let window_bits = -9;

    let input = &include_bytes!("test-data/op-len-edge-case.zraw");

    assert_eq_rs_ng!(unsafe {
        let mut output: Vec<u8> = Vec::with_capacity(1 << 15);

        let mut buf = [0; 266];

        let mut stream = MaybeUninit::<z_stream>::zeroed();

        let err = {
            inflateInit2_(
                stream.as_mut_ptr(),
                window_bits,
                zlibVersion(),
                core::mem::size_of::<z_stream>() as c_int,
            )
        };
        assert_eq!(ReturnCode::from(err), ReturnCode::Ok);

        let stream = { stream.assume_init_mut() };

        stream.next_in = input.as_ptr().cast_mut();
        let input_size = c_uint::try_from(input.len());
        assert!(
            input_size.is_ok(),
            "inflate fixture exceeds c_uint input limit"
        );
        stream.avail_in = input_size.unwrap_or_default();

        while stream.avail_in != 0 {
            let available_before = stream.avail_in;
            stream.next_out = buf.as_mut_ptr();
            stream.avail_out = c_uint::try_from(buf.len()).unwrap_or_default();

            let err = { inflate(stream, InflateFlush::NoFlush as _) };
            let remaining = stream.avail_out as usize;
            assert!(
                remaining <= buf.len(),
                "inflate reported invalid output capacity"
            );
            let produced = buf.len() - remaining;
            output.extend_from_slice(&buf[..produced]);

            assert!(
                matches!(
                    ReturnCode::from(err),
                    ReturnCode::Ok | ReturnCode::BufError | ReturnCode::StreamEnd
                ),
                "unexpected inflate status: {err}"
            );
            if err == Z_STREAM_END {
                break;
            }
            // A truncated test stream may end without Z_STREAM_END, but
            // a decoder that consumes no bytes and produces nothing is stuck.
            assert!(
                stream.avail_in < available_before || produced != 0,
                "inflate made no progress with {} input bytes remaining",
                stream.avail_in
            );
        }

        let err = inflateEnd(stream);
        assert_eq!(ReturnCode::from(err), ReturnCode::Ok);

        output
    });
}

// Fills the provided buffer with pseudorandom bytes based on the given seed
// Duplicates bytes by `step` in a row
fn prng_bytes(seed: u64, bytes: &mut [u8], step: usize) {
    const M: u64 = 2u64.pow(32);
    const A: u64 = 1664525;
    const C: u64 = 1013904223;
    let mut state = seed;
    for chunk in bytes.chunks_mut(4 * step) {
        state = (A * state + C) % M;
        let rand_bytes = state.to_le_bytes();
        for (i, byte) in chunk.iter_mut().enumerate() {
            *byte = rand_bytes[i / step];
        }
    }
}

#[test]
#[cfg_attr(miri, ignore = "slow")]
fn test_inflate_flush_block() {
    let window_bits = -15; // Raw
    const CHUNK: usize = 16384;

    // Create a compressed vector of random data that's bigger then the zlib block size
    let mut data = vec![0u8; 160000];
    prng_bytes(314159, &mut data, 4);
    let config = DeflateConfig {
        window_bits,
        ..DeflateConfig::default()
    };
    let mut output = vec![0u8; 80000];
    // Compress the data
    let (compressed_data, return_code) = compress_slice(&mut output, &data, config);
    assert_eq!(return_code, ReturnCode::Ok);

    // Capture not just the Z_BLOCK state transitions but the bytes produced
    // at each transition. Matching counters alone could hide bad output.
    let mut zlib_trace = Vec::new();
    let mut zlib_decoded = Vec::with_capacity(data.len());
    {
        let mut stream = MaybeUninit::<libz_sys::z_stream>::zeroed();

        // SAFETY: stream storage, version, and structure size match the
        // installed reference zlib C ABI.
        let ret = unsafe {
            libz_sys::inflateInit2_(
                stream.as_mut_ptr(),
                window_bits,
                libz_sys::zlibVersion(),
                test_c_stream_size::<libz_sys::z_stream>(),
            )
        };
        assert_eq!(ReturnCode::from(ret), ReturnCode::Ok);

        // SAFETY: successful inflateInit2_ initialized every stream field.
        let stream = unsafe { stream.assume_init_mut() };
        let mut scratch = vec![0u8; CHUNK * 2];
        stream.next_in = compressed_data.as_ptr().cast_mut();
        stream.avail_in = test_c_uint_len(compressed_data.len());
        loop {
            stream.next_out = scratch.as_mut_ptr();
            stream.avail_out = test_c_uint_len(scratch.len());
            let input_before = stream.avail_in;
            let output_before = stream.total_out;
            let data_type_before = stream.data_type;

            // SAFETY: the compressed input and the disjoint writable scratch
            // buffer stay live for the synchronous reference decoder call.
            let ret = unsafe { libz_sys::inflate(stream, InflateFlush::Block as i32) };
            assert_eq!(ReturnCode::from(ret), ReturnCode::Ok);
            let remaining = usize::try_from(stream.avail_out).unwrap_or(usize::MAX);
            assert!(
                remaining <= scratch.len(),
                "reference output capacity invalid"
            );
            let produced = scratch.len() - remaining;
            zlib_decoded.extend_from_slice(&scratch[..produced]);
            zlib_trace.push((
                stream.avail_in,
                stream.avail_out,
                stream.data_type,
                scratch[..produced].to_vec(),
            ));

            if stream.avail_in == 0 {
                break;
            }
            assert!(
                stream.avail_in < input_before
                    || stream.total_out > output_before
                    || stream.data_type != data_type_before,
                "reference Z_BLOCK decoder stalled"
            );
        }

        assert_eq!(usize::try_from(stream.total_in), Ok(compressed_data.len()));
        assert_eq!(usize::try_from(stream.total_out), Ok(data.len()));
        // SAFETY: reference stream initialization succeeded and is ended once.
        assert_eq!(unsafe { libz_sys::inflateEnd(stream) }, Z_OK);
    }
    assert_eq!(zlib_decoded, data, "reference Z_BLOCK decode lost data");

    // Compare *every* reference transition and output chunk with zlib-rs;
    // also require the same number of calls (no unverified extra/missing
    // transitions).
    {
        use libz_rs_sys::*;

        let mut stream = MaybeUninit::<z_stream>::zeroed();

        // SAFETY: writable stream storage and the matching zlib-rs ABI.
        let ret = unsafe {
            inflateInit2_(
                stream.as_mut_ptr(),
                window_bits,
                zlibVersion(),
                test_c_stream_size::<z_stream>(),
            )
        };
        assert_eq!(ReturnCode::from(ret), ReturnCode::Ok);

        // SAFETY: the successful initialization wrote the complete C stream.
        let stream = unsafe { stream.assume_init_mut() };
        let mut scratch = vec![0u8; CHUNK * 2];
        let mut decoded = Vec::with_capacity(data.len());
        let mut reference_index = 0;
        stream.next_in = compressed_data.as_ptr().cast_mut();
        stream.avail_in = test_c_uint_len(compressed_data.len());
        loop {
            stream.next_out = scratch.as_mut_ptr();
            stream.avail_out = test_c_uint_len(scratch.len());
            let input_before = stream.avail_in;
            let output_before = stream.total_out;
            let data_type_before = stream.data_type;

            // SAFETY: compressed_data and scratch are live non-overlapping
            // byte spans with accurate C ABI availability counts.
            let ret = unsafe { inflate(stream, InflateFlush::Block as i32) };
            assert_eq!(ReturnCode::from(ret), ReturnCode::Ok);
            let remaining = usize::try_from(stream.avail_out).unwrap_or(usize::MAX);
            assert!(
                remaining <= scratch.len(),
                "zlib-rs output capacity invalid"
            );
            let produced = scratch.len() - remaining;
            assert!(
                reference_index < zlib_trace.len(),
                "zlib-rs produced an extra Z_BLOCK transition"
            );
            let (expected_in, expected_out, expected_type, expected_chunk) =
                &zlib_trace[reference_index];
            reference_index += 1;
            assert_eq!(
                (stream.avail_in, stream.avail_out, stream.data_type),
                (*expected_in, *expected_out, *expected_type),
                "Z_BLOCK state transition differs from reference"
            );
            assert_eq!(
                &scratch[..produced],
                expected_chunk,
                "Z_BLOCK output differs"
            );
            decoded.extend_from_slice(&scratch[..produced]);

            if stream.avail_in == 0 {
                break;
            }
            assert!(
                stream.avail_in < input_before
                    || stream.total_out > output_before
                    || stream.data_type != data_type_before,
                "zlib-rs Z_BLOCK decoder stalled"
            );
        }

        assert_eq!(
            reference_index,
            zlib_trace.len(),
            "zlib-rs skipped Z_BLOCK transitions"
        );
        assert_eq!(usize::try_from(stream.total_in), Ok(compressed_data.len()));
        assert_eq!(usize::try_from(stream.total_out), Ok(data.len()));
        assert_eq!(decoded, data, "zlib-rs Z_BLOCK decode lost data");
        // SAFETY: initialized zlib-rs stream is ended exactly once.
        assert_eq!(unsafe { inflateEnd(stream) }, Z_OK);
    }
}

#[test]
fn issue_172() {
    const BUF: &[u8] = &[
        31, 139, 8, 0, 0, 0, 0, 0, 0, 3, 75, 173, 40, 72, 77, 46, 73, 77, 81, 200, 47, 45, 41, 40,
        45, 1, 0, 176, 1, 57, 179, 15, 0, 0, 0,
    ];

    // Every chunk size from a single compressed byte through the full
    // fixture must decode exactly the same output.
    assert_eq_rs_ng!({
        let mut completed_sizes = Vec::new();
        for chunk_size in 1..=BUF.len() {
            let mut output = [0u8; 32];
            let mut raw = MaybeUninit::<z_stream>::zeroed();

            // SAFETY: writable stream storage and a matching C ABI version
            // and structure size are supplied for initialization.
            let ret = unsafe {
                inflateInit2_(
                    raw.as_mut_ptr(),
                    31,
                    zlibVersion(),
                    test_c_stream_size::<z_stream>(),
                )
            };
            assert_eq!(ret, Z_OK);
            // SAFETY: successful initialization populated the entire stream.
            let stream = unsafe { raw.assume_init_mut() };
            stream.next_out = output.as_mut_ptr();
            stream.avail_out = test_c_uint_len(output.len());

            let mut offered = 0;
            loop {
                // Only supply a new chunk after the previous one was fully
                // consumed. Never replace a still-live next_in/avail_in span.
                if stream.avail_in == 0 && offered < BUF.len() {
                    let end = offered.saturating_add(chunk_size).min(BUF.len());
                    let pending = &BUF[offered..end];
                    stream.next_in = pending.as_ptr().cast_mut();
                    stream.avail_in = test_c_uint_len(pending.len());
                    offered = end;
                }

                let input_before = stream.avail_in;
                let output_before = stream.total_out;
                // SAFETY: next_in refers to an allocated portion of BUF;
                // next_out refers into the disjoint, writable output buffer.
                let ret = unsafe { inflate(stream, Z_NO_FLUSH) };
                assert!(
                    matches!(ret, Z_OK | Z_STREAM_END),
                    "chunk size {chunk_size}: unexpected inflate status {ret}"
                );
                if ret == Z_STREAM_END {
                    break;
                }
                assert!(
                    stream.avail_in < input_before || stream.total_out > output_before,
                    "chunk size {chunk_size}: decoder made no progress"
                );
                assert!(
                    stream.avail_in != 0 || offered != BUF.len(),
                    "chunk size {chunk_size}: compressed input ended without Z_STREAM_END"
                );
            }

            assert_eq!(offered, BUF.len(), "did not supply the full gzip fixture");
            assert_eq!(stream.avail_in, 0, "gzip trailer left unconsumed");
            assert_eq!(usize::try_from(stream.total_in), Ok(BUF.len()));
            assert_eq!(
                usize::try_from(stream.total_out),
                Ok(b"expected output".len())
            );
            assert_eq!(&output[..b"expected output".len()], b"expected output");

            // SAFETY: the successfully initialized stream is ended once.
            assert_eq!(unsafe { inflateEnd(stream) }, Z_OK);
            completed_sizes.push(chunk_size);
        }
        completed_sizes
    });
}
#[test]
fn header_configured_but_no_gzip() {
    // This is raw DEFLATE rather than a gzip member. Registering a gzip
    // header receiver must not trick automatic wrapper detection into
    // accepting the input or writing unexpected output bytes.
    let input: &[u8] = &[0x63, 0x18, 0x68, 0x30, 0xd0, 0x0, 0x0];

    assert_eq_rs_ng!({
        let mut raw = MaybeUninit::<z_stream>::zeroed();
        // SAFETY: writable C stream storage, valid auto-wrapper mode,
        // correct version and checked structure size.
        let status = unsafe {
            inflateInit2_(
                raw.as_mut_ptr(),
                47,
                zlibVersion(),
                test_c_stream_size::<z_stream>(),
            )
        };
        assert_eq!(status, Z_OK);
        // SAFETY: successful inflateInit2_ initialized the whole C stream.
        let stream = unsafe { raw.assume_init_mut() };

        let mut header = gz_header {
            text: 0,
            time: 0,
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
            done: 123,
        };

        // SAFETY: the live C stream and writable header struct remain
        // valid throughout header registration and the following decode.
        let status = unsafe { inflateGetHeader(stream, &raw mut header) };
        assert_eq!(status, Z_OK);

        stream.avail_in = test_c_uint_len(input.len());
        stream.next_in = input.as_ptr().cast_mut();

        let mut output = [0xa5u8; 32];
        stream.avail_out = test_c_uint_len(output.len());
        stream.next_out = output.as_mut_ptr();

        // SAFETY: the bounded input bytes and writable output array are
        // disjoint and live for this synchronous inflate call.
        let status = unsafe { inflate(stream, InflateFlush::NoFlush as _) };
        assert_eq!(status, Z_DATA_ERROR);
        assert_eq!(stream.total_out, 0);
        assert_eq!(output, [0xa5u8; 32]);
        assert!(usize::try_from(stream.total_in).unwrap_or(usize::MAX) <= input.len());

        // The requested header was not a gzip header; its done flag must
        // report that, instead of silently retaining the caller's 123.
        assert_eq!(header.done, -1);
        // SAFETY: the successfully initialized C stream is ended once.
        assert_eq!(unsafe { inflateEnd(stream) }, Z_OK);
        (status, header.done)
    });
}

#[test]
fn issue_232() {
    use libz_rs_sys::*;

    const INPUT: &[u8] = &[
        0x18, 0x57, 0x0a, 0xa8, 0xa8, 0xa8, 0xa8, 0x7e, 0x18, 0x57, 0xa8, 0xa8, 0xa8, 0xa8, 0xa8,
        0xa8, 0x83, 0x83, 0xa8, 0x83, 0x83, 0x83, 0x83, 0x84, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83,
        0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83,
        0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x01, 0x04, 0xa8, 0xa8, 0x18, 0x57, 0x18, 0x0a,
        0x57, 0x4f, 0xa8, 0xa8, 0xa8, 0xa8, 0x11, 0x04, 0xa8, 0xa8, 0xe1, 0xe1, 0xe1, 0xe1, 0xe0,
        0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1,
        0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1,
        0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1,
        0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xc2, 0x83,
        0x83, 0x84, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83,
        0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83,
        0x01, 0x04, 0xa8, 0xa8, 0x18, 0x57, 0x18, 0x0a, 0x57, 0x4f, 0xa8, 0xa8, 0xa8, 0xa8, 0x11,
        0x04, 0xe8, 0xa8, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1,
        0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1,
        0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1,
        0xe1, 0xe1, 0xe1, 0xe1, 0x3a, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1,
        0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xc2, 0x83, 0x83, 0x84, 0x83, 0x83, 0x83, 0x83, 0x83,
        0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83,
        0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83,
        0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83,
        0x83, 0x83, 0x83, 0x01, 0x04, 0xa8, 0xa8, 0xa8, 0xa8, 0xa8, 0xa8, 0x11, 0x04, 0xa8, 0xa8,
        0xa8, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x01, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83,
        0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x01, 0x04, 0xa8, 0xa8, 0xa8,
        0xa8, 0xa8, 0xa8, 0x11, 0x04, 0xa8, 0xa8, 0xa8, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x01,
        0x04, 0xa8, 0xa8, 0xa8, 0xa8, 0xa8, 0xa8, 0x11, 0x04, 0xa8, 0xa8, 0x04, 0xa8, 0xa8, 0xa8,
        0xa8, 0xa8, 0xa8, 0x11, 0x04, 0xa8, 0xa8, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83,
        0x83, 0x83, 0x01, 0x04, 0xa8, 0xa8, 0x18, 0x57, 0x18, 0x0a, 0x57, 0x4f, 0xa8, 0xa8, 0xa8,
        0xa8, 0x11, 0x04, 0xe8, 0xa8, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1,
        0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1,
        0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1,
        0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0x3a, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1,
        0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xc2, 0x83, 0x83, 0x84, 0x83, 0x83, 0x83,
        0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83,
        0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x01, 0x04, 0xa8, 0xa8, 0x18,
        0x57, 0x18, 0x0a, 0x57, 0x4f, 0xa8, 0xa8, 0xa8, 0xa8, 0x11, 0x04, 0xa8, 0xa8, 0xe1, 0xe1,
        0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1,
        0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1,
        0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1,
        0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1,
        0xe1, 0xc2, 0x83, 0x83, 0x84, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83,
        0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83,
        0x83, 0x83, 0x83, 0x01, 0x04, 0xa8, 0xa8, 0x18, 0x57, 0x18, 0x0a, 0x57, 0x4f, 0xa8, 0xa8,
        0x83, 0x83, 0x01, 0x04, 0xa8, 0xa8, 0x18, 0x57, 0x18, 0x0a, 0x57, 0x4f, 0xa8, 0xa8, 0xa8,
        0xa8, 0x11, 0x04, 0xa8, 0xa8, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1,
        0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1,
        0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1,
        0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1,
        0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xc2, 0x83, 0x83, 0x84, 0x83, 0x83, 0x83,
        0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83,
        0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x01, 0x04, 0xa8, 0xa8, 0x18,
        0x57, 0x18, 0x0a, 0x57, 0x4f, 0xa8, 0xa8, 0xa8, 0xa8, 0x11, 0x04, 0xe8, 0xa8, 0xe1, 0xe1,
        0xe1, 0xe1, 0xa8, 0xa8, 0x11, 0x04, 0xe8, 0xa8, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1,
        0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1,
        0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xc2, 0x83, 0x83,
        0x84, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83,
        0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x01,
        0x04, 0xa8, 0xa8, 0x18, 0x57, 0x18, 0x0a, 0x57, 0x4f, 0xa8, 0xa8, 0x83, 0x83, 0x01, 0x04,
        0xa8, 0xa8, 0x18, 0x57, 0x18, 0x0a, 0x57, 0x4f, 0xa8, 0xa8, 0xa8, 0xa8, 0x11, 0x04, 0xa8,
        0xa8, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1,
        0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1,
        0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1,
        0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1,
        0xe1, 0xe1, 0xe1, 0xe1, 0xc2, 0x83, 0x83, 0x84, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83,
        0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83, 0x83,
        0x83, 0x83, 0x83, 0x83, 0x83, 0x7d, 0xfe, 0x04, 0xa8, 0xa8, 0x18, 0x57, 0x18, 0x0a, 0x57,
        0x4f, 0xa8, 0xa8, 0xa8, 0xa8, 0x11, 0x04, 0x68, 0xa8, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1,
        0xa8, 0xa8, 0x11, 0x04, 0xe8, 0xa8, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1,
        0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1, 0xe1,
        0xe1, 0xe1, 0xe1, 0xa8, 0xa8, 0xa8, 0xa8, 0x83, 0x61, 0xa8, 0xa8, 0x7e, 0x18, 0x57, 0xa8,
        0xa8, 0xa8, 0xa8, 0xa8, 0xa8, 0x83, 0x83, 0x83, 0x01, 0x04, 0xa8, 0x83, 0xa8, 0xa8, 0xa8,
        0xa8, 0x0e,
    ];

    let mut stream = z_stream::default();

    // SAFETY: initialized writable z_stream and valid ABI version/size.
    let err = unsafe {
        inflateInit2_(
            &raw mut stream,
            15 + 32,
            zlibVersion(),
            test_c_stream_size::<z_stream>(),
        )
    };
    assert_eq!(ReturnCode::from(err), ReturnCode::Ok);

    let mut output = vec![0; INPUT.len()];
    stream.next_out = output.as_mut_ptr();
    stream.avail_out = test_c_uint_len(output.len());
    stream.next_in = INPUT.as_ptr();
    stream.avail_in = test_c_uint_len(INPUT.len());

    // This fuzz reproducer need not contain a complete gzip stream. It
    // exercises the decoder until the data is consumed or a stream ends.
    loop {
        let input_before = stream.avail_in;
        let output_before = stream.total_out;
        // SAFETY: both C byte spans are live, disjoint, and bounded by
        // their stated avail_in/avail_out counts.
        let err = unsafe { inflate(&raw mut stream, InflateFlush::Finish as _) };
        assert!(
            matches!(
                ReturnCode::from(err),
                ReturnCode::StreamEnd | ReturnCode::BufError | ReturnCode::Ok
            ),
            "unexpected inflate result: {err}"
        );
        let produced = usize::try_from(stream.total_out).unwrap_or(usize::MAX);
        assert!(
            produced <= output.len(),
            "inflate output exceeds the vector"
        );
        if err == Z_STREAM_END {
            break;
        }

        // A full output buffer must grow, but the C pointer must advance
        // to the *unwritten suffix* after Vec::resize, not its beginning.
        if stream.avail_out == 0 {
            let old_len = output.len();
            let new_len = old_len.checked_mul(2).unwrap_or(old_len);
            assert!(new_len > old_len, "inflate output buffer size overflow");
            output.resize(new_len, 0);
            stream.next_out = output[produced..].as_mut_ptr();
            stream.avail_out = test_c_uint_len(new_len - produced);
            continue;
        }
        if stream.avail_in == 0 {
            break; // Truncated fixture may not produce Z_STREAM_END.
        }
        assert!(
            stream.avail_in < input_before || stream.total_out > output_before,
            "inflate stalled with input and output space remaining"
        );
    }

    // The reproducer contains no complete gzip trailer. zlib nevertheless
    // produces this 35,001-byte prefix, independently checked against zlib.
    // Verify its contents as well as its size: previously, resetting
    // next_out to the Vec start on growth could silently overwrite bytes.
    let decoded_len = usize::try_from(stream.total_out).unwrap_or(usize::MAX);
    assert_eq!(decoded_len, 35_001);
    assert_eq!(
        zlib_rs::crc32::crc32(0, &output[..decoded_len]),
        0x0cf8_f315,
        "issue_232 decoded output was overwritten or corrupted"
    );

    // SAFETY: the initialized inflate stream is ended exactly once.
    assert_eq!(unsafe { inflateEnd(&raw mut stream) }, Z_OK);
}

#[test]
#[cfg_attr(miri, ignore = "slow")]
fn blow_up_the_stack_1() {
    // requires a sequence of states that would blow up the stack if inflate is not stack safe.

    const INPUT: &[u8] = include_bytes!("test-data/blow_up_the_stack_1.gz");

    let mut output_ng = vec![0; INPUT.len() * 128];
    let mut output_rs = vec![0; INPUT.len() * 128];

    let config = InflateConfig::default();

    let (_, err) = crate::helpers::uncompress_slice_ng(&mut output_ng, INPUT, config);
    assert_eq!(err, ReturnCode::DataError);

    let (_, err) = decompress_slice(&mut output_rs, INPUT, config);
    assert_eq!(err, ReturnCode::DataError);
}

#[test]
#[cfg_attr(miri, ignore = "slow")]
fn blow_up_the_stack_2() {
    // requires a sequence of states that would blow up the stack if inflate is not stack safe.

    const INPUT: &[u8] = include_bytes!("test-data/blow_up_the_stack_2.gz");

    let mut output_ng = vec![0; INPUT.len() * 128];
    let mut output_rs = vec![0; INPUT.len() * 128];

    let config = InflateConfig::default();

    let (_, err) = crate::helpers::uncompress_slice_ng(&mut output_ng, INPUT, config);
    assert_eq!(err, ReturnCode::DataError);

    let (_, err) = decompress_slice(&mut output_rs, INPUT, config);
    assert_eq!(err, ReturnCode::DataError);
}

#[test]
fn codes_used() {
    // -1 is returned on NULL
    assert_eq_rs_ng!(unsafe { inflateCodesUsed(core::ptr::null_mut()) });
    assert_eq!(
        unsafe { libz_rs_sys::inflateCodesUsed(core::ptr::null_mut()) },
        c_ulong::MAX
    );

    let inputs = &[
        include_bytes!("test-data/compression-corpus/The fastest WASM zlib.md.gzip-0.gz")
            .as_slice(),
        include_bytes!("test-data/compression-corpus/The fastest WASM zlib.md.gzip-9.gz")
            .as_slice(),
        include_bytes!("test-data/compression-corpus/The fastest WASM zlib.md.gzip-filtered-9.gz")
            .as_slice(),
        include_bytes!("test-data/compression-corpus/The fastest WASM zlib.md.gzip-fixed-9.gz")
            .as_slice(),
        include_bytes!("test-data/compression-corpus/The fastest WASM zlib.md.gzip-huffman-9.gz")
            .as_slice(),
        include_bytes!("test-data/compression-corpus/The fastest WASM zlib.md.gzip-rle-9.gz")
            .as_slice(),
    ];

    let mut reference_output: Option<Vec<u8>> = None;
    for input in inputs {
        let (decoded, _) = assert_eq_rs_ng!({
            let mut output = vec![0u8; 1 << 15];
            let mut code_counts = Vec::new();
            let mut raw = MaybeUninit::<z_stream>::zeroed();

            // SAFETY: initialize an ABI-compatible, writable z_stream.
            let err = unsafe {
                inflateInit2_(
                    raw.as_mut_ptr(),
                    16 + 15,
                    zlibVersion(),
                    test_c_stream_size::<z_stream>(),
                )
            };
            assert_eq!(ReturnCode::from(err), ReturnCode::Ok);
            // SAFETY: inflateInit2_ initialized the entire stream on success.
            let stream = unsafe { raw.assume_init_mut() };
            // SAFETY: the initialized C stream contains a live inflate state.
            code_counts.push(unsafe { inflateCodesUsed(stream) });

            stream.next_out = output.as_mut_ptr();
            stream.avail_out = test_c_uint_len(output.len());

            let mut completed = false;
            for chunk in input.chunks(16) {
                stream.next_in = chunk.as_ptr().cast_mut();
                stream.avail_in = test_c_uint_len(chunk.len());

                // Do not discard an unconsumed suffix when another chunk
                // arrives. Zlib may consume only part of the supplied input.
                while stream.avail_in != 0 {
                    if stream.avail_out == 0 {
                        let used = usize::try_from(stream.total_out).unwrap_or(usize::MAX);
                        assert!(used <= output.len(), "inflate output count exceeds buffer");
                        let old_len = output.len();
                        let new_len = old_len.checked_mul(2).unwrap_or(old_len);
                        assert!(new_len > old_len, "inflate output length overflow");
                        output.resize(new_len, 0);
                        stream.next_out = output[used..].as_mut_ptr();
                        stream.avail_out = test_c_uint_len(new_len - used);
                    }

                    let input_before = stream.avail_in;
                    let output_before = stream.total_out;
                    // SAFETY: C input points into the current live chunk, and
                    // output points into the live, disjoint output vector.
                    let err = unsafe { inflate(stream, InflateFlush::NoFlush as _) };
                    // SAFETY: the successful initialization remains active.
                    code_counts.push(unsafe { inflateCodesUsed(stream) });
                    assert!(
                        matches!(err, Z_OK | Z_STREAM_END),
                        "invalid gzip fixture or decoder status: {err}"
                    );
                    assert!(
                        stream.avail_in < input_before || stream.total_out > output_before,
                        "inflate made no progress with unconsumed compressed input"
                    );
                    if err == Z_STREAM_END {
                        completed = true;
                        break;
                    }
                }
                if completed {
                    break;
                }
            }

            assert!(completed, "gzip fixture ended without Z_STREAM_END");
            assert_eq!(
                usize::try_from(stream.total_in),
                Ok(input.len()),
                "gzip fixture was not fully consumed"
            );
            let decoded_len = usize::try_from(stream.total_out).unwrap_or(usize::MAX);
            assert!(decoded_len <= output.len(), "inflate output exceeds buffer");
            output.truncate(decoded_len);

            // SAFETY: this initialized inflate stream is closed exactly once.
            assert_eq!(unsafe { inflateEnd(stream) }, Z_OK);
            (output, code_counts)
        });

        assert!(
            !decoded.is_empty(),
            "gzip fixture unexpectedly decoded empty"
        );
        if let Some(ref baseline) = reference_output {
            assert_eq!(&decoded, baseline, "gzip compression variants disagree");
        } else {
            reference_output = Some(decoded);
        }
    }
}

#[test]
fn test_dict_inflate() {
    // Maximum dictionary size, according to inflateGetDictionary() description.
    const MAX_DICTIONARY_SIZE: usize = 32768;

    let mut dictionary = *b"hello\0";
    let mut hello = *b"hello\0";

    let (compr, dict_id, compressed_len) = assert_eq_rs_ng!({
        let mut compr = [0u8; 1024];

        let mut strm = MaybeUninit::zeroed();

        let ret = unsafe {
            deflateInit_(
                strm.as_mut_ptr(),
                Z_BEST_COMPRESSION,
                zlibVersion(),
                test_c_stream_size::<z_stream>(),
            )
        };

        assert_eq!(ReturnCode::from(ret), ReturnCode::Ok);
        // SAFETY: deflateInit_ successfully initialized the complete stream.
        let c_stream = unsafe { strm.assume_init_mut() };

        // SAFETY: the initialized encoder borrows the live dictionary only
        // during the synchronous deflateSetDictionary call.
        let err = unsafe {
            deflateSetDictionary(
                c_stream,
                dictionary.as_ptr(),
                test_c_uint_len(dictionary.len()),
            )
        };
        assert_eq!(ReturnCode::from(err), ReturnCode::Ok);

        let dict_id = c_stream.adler;
        c_stream.avail_out = test_c_uint_len(compr.len());
        c_stream.next_out = compr.as_mut_ptr();

        c_stream.avail_in = test_c_uint_len(hello.len());
        c_stream.next_in = hello.as_mut_ptr();

        // SAFETY: both input and output buffers remain live, writable as
        // required, and disjoint for the entire encoder call.
        let err = unsafe { deflate(c_stream, Z_FINISH) };
        assert_eq!(ReturnCode::from(err), ReturnCode::StreamEnd);
        // deflateSetDictionary contributes the preset dictionary bytes to
        // total_in before the actual payload is presented to deflate().
        assert_eq!(
            usize::try_from(c_stream.total_in),
            Ok(dictionary.len() + hello.len())
        );
        let compressed_len = usize::try_from(c_stream.total_out).unwrap_or(usize::MAX);
        assert!(
            compressed_len <= compr.len(),
            "deflate wrote outside its buffer"
        );

        // SAFETY: this initialized deflate stream is ended exactly once.
        let err = unsafe { deflateEnd(c_stream) };
        assert_eq!(ReturnCode::from(err), ReturnCode::Ok);

        (compr, dict_id, compressed_len)
    });

    assert_eq_rs_ng!({
        let mut compr = compr;
        let mut uncompr = *b"garbage garbage garbage\0";

        let mut stream = MaybeUninit::<z_stream>::zeroed();

        // SAFETY: the destination is a writable C stream, and the supplied
        // version and stream size match this ABI.
        let err = unsafe {
            inflateInit_(
                stream.as_mut_ptr(),
                zlibVersion(),
                test_c_stream_size::<z_stream>(),
            )
        };
        assert_eq!(ReturnCode::from(err), ReturnCode::Ok);
        // SAFETY: inflateInit_ initialized every C stream field.
        let d_stream = unsafe { stream.assume_init_mut() };

        d_stream.adler = 0;
        // The encoder may use only a prefix of the 1024-byte scratch array.
        // Never offer the uninitialized-as-compressed trailing padding.
        d_stream.avail_in = test_c_uint_len(compressed_len);
        d_stream.next_in = compr.as_mut_ptr();

        d_stream.avail_out = test_c_uint_len(uncompr.len());
        d_stream.next_out = uncompr.as_mut_ptr();

        let mut check_dictionary = [0u8; MAX_DICTIONARY_SIZE];
        let mut supplied_dictionary = false;

        loop {
            let previous_input = d_stream.avail_in;
            let previous_output = d_stream.total_out;
            // SAFETY: the stream is initialized and its input/output spans
            // refer to separate, live allocations with the stated capacities.
            let mut err = unsafe { inflate(d_stream, Z_NO_FLUSH) };

            if err == Z_STREAM_END {
                break;
            }

            if err == Z_NEED_DICT {
                assert!(
                    !supplied_dictionary,
                    "inflate requested the dictionary twice"
                );
                assert_eq!(d_stream.adler, dict_id, "incorrect dictionary identifier");
                // SAFETY: the borrowed dictionary bytes remain live until this
                // synchronous inflateSetDictionary call returns.
                err = unsafe {
                    inflateSetDictionary(
                        d_stream,
                        dictionary.as_mut_ptr(),
                        test_c_uint_len(dictionary.len()),
                    )
                };
                supplied_dictionary = true;
            } else {
                assert!(
                    d_stream.avail_in < previous_input || d_stream.total_out > previous_output,
                    "inflate stalled without consuming input or producing output"
                );
            }
            assert_eq!(ReturnCode::from(err), ReturnCode::Ok);
        }
        assert!(
            supplied_dictionary,
            "decode never requested its preset dictionary"
        );
        // Preset-dictionary setup may reset the decoder's total_in
        // counter. avail_in directly verifies that every supplied byte
        // from the actual compressed span was consumed.
        assert_eq!(d_stream.avail_in, 0, "compressed payload left unread");
        assert_eq!(usize::try_from(d_stream.total_out), Ok(hello.len()));

        let mut check_dictionary_len = 0;
        // SAFETY: a null dictionary pointer requests only its current size.
        let err = unsafe {
            inflateGetDictionary(
                d_stream,
                core::ptr::null_mut(),
                &raw mut check_dictionary_len,
            )
        };
        assert_eq!(ReturnCode::from(err), ReturnCode::Ok);
        assert!(usize::try_from(check_dictionary_len).unwrap_or(0) >= dictionary.len());
        assert!(
            usize::try_from(check_dictionary_len).unwrap_or(usize::MAX) <= check_dictionary.len()
        );

        // SAFETY: the returned size was checked against the writable
        // dictionary buffer and the stream is still initialized.
        let err = unsafe {
            inflateGetDictionary(
                d_stream,
                check_dictionary.as_mut_ptr(),
                &raw mut check_dictionary_len,
            )
        };
        assert_eq!(ReturnCode::from(err), ReturnCode::Ok);
        assert_eq!(&check_dictionary[..dictionary.len()], dictionary);

        // SAFETY: the initialized inflate stream is ended exactly once.
        let err = unsafe { inflateEnd(d_stream) };
        assert_eq!(ReturnCode::from(err), ReturnCode::Ok);

        assert_eq!(uncompr[..hello.len()], hello);
    });
}

mod stable_api {
    use zlib_rs::{
        deflate::{compress_slice, DeflateConfig},
        InflateError, ReturnCode, Status,
    };

    fn with_window_bits(n: i32) {
        let config = DeflateConfig {
            window_bits: n,
            ..DeflateConfig::default()
        };

        let mut compressed_buffer = [0u8; 64];
        let input = b"Hello World!";
        let (compressed, ret) = compress_slice(&mut compressed_buffer, input, config);
        assert_eq!(ret, ReturnCode::Ok);

        let mut decompressed = [0u8; 64];
        let window_bits = u8::try_from(n.unsigned_abs()).unwrap_or(0);
        assert_ne!(window_bits, 0, "invalid test window bits");
        let mut decoder = zlib_rs::Inflate::new(n >= 0, window_bits);
        assert_eq!(
            decoder.decompress(compressed, &mut decompressed, zlib_rs::InflateFlush::Finish),
            Ok(Status::StreamEnd),
            "decoder did not complete the compressed member"
        );
        assert_eq!(usize::try_from(decoder.total_in()), Ok(compressed.len()));
        assert_eq!(usize::try_from(decoder.total_out()), Ok(input.len()));
        assert_eq!(&decompressed[..input.len()], input);
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
    #[should_panic = "StreamError"]
    fn invalid_config() {
        let _ = zlib_rs::Inflate::new(true, 123);
    }

    #[test]
    fn invalid_data() {
        let mut inflate = zlib_rs::Inflate::new(true, 15);

        // Clearly invalid input.
        let compressed = [0xAA; 64];
        let mut decompressed = [0u8; 64];

        let ret = inflate.decompress(
            &compressed,
            &mut decompressed,
            zlib_rs::InflateFlush::Finish,
        );

        assert_eq!(ret, Err(InflateError::DataError));
    }

    #[test]
    fn need_dict() {
        let mut inflate = zlib_rs::Inflate::new(true, 15);

        let compressed = [0x08, 0xb8, 0x0, 0x0, 0x0, 0x1];
        let mut decompressed = [0u8; 64];

        let ret = inflate.decompress(
            &compressed,
            &mut decompressed,
            zlib_rs::InflateFlush::Finish,
        );

        assert_eq!(ret, Err(InflateError::NeedDict { dict_id: 1 }));
    }

    #[test]
    fn reset_reuse() {
        let first = b"Hello World!";
        let second = b"Goodbye World!";

        let mut first_compressed = [0u8; 64];
        let (first_stream, status) = compress_slice(
            &mut first_compressed,
            first,
            DeflateConfig {
                window_bits: 15,
                ..DeflateConfig::default()
            },
        );
        assert_eq!(status, ReturnCode::Ok);

        let mut second_compressed = [0u8; 64];
        let (second_stream, status) = compress_slice(
            &mut second_compressed,
            second,
            DeflateConfig {
                window_bits: -15,
                ..DeflateConfig::default()
            },
        );
        assert_eq!(status, ReturnCode::Ok);

        // First complete a wrapped zlib stream.
        let mut decoder = zlib_rs::Inflate::new(true, 15);
        let mut decoded = [0xa5u8; 64];
        assert_eq!(
            decoder.decompress(first_stream, &mut decoded, zlib_rs::InflateFlush::Finish),
            Ok(Status::StreamEnd)
        );
        assert_eq!(usize::try_from(decoder.total_in()), Ok(first_stream.len()));
        assert_eq!(usize::try_from(decoder.total_out()), Ok(first.len()));
        assert_eq!(&decoded[..first.len()], first);

        // Reuse the very same decoder for a raw DEFLATE stream. Reset
        // must clear counters and old wrapper state before the new call.
        decoder.reset(false);
        assert_eq!(decoder.total_in(), 0);
        assert_eq!(decoder.total_out(), 0);
        decoded.fill(0xa5);
        assert_eq!(
            decoder.decompress(second_stream, &mut decoded, zlib_rs::InflateFlush::Finish),
            Ok(Status::StreamEnd)
        );
        assert_eq!(usize::try_from(decoder.total_in()), Ok(second_stream.len()));
        assert_eq!(usize::try_from(decoder.total_out()), Ok(second.len()));
        assert_eq!(&decoded[..second.len()], second);

        // Switching the same decoder back to wrapped mode catches stale
        // raw-state history which a one-way reset cannot reveal.
        decoder.reset(true);
        assert_eq!(decoder.total_in(), 0);
        assert_eq!(decoder.total_out(), 0);
        decoded.fill(0xa5);
        assert_eq!(
            decoder.decompress(first_stream, &mut decoded, zlib_rs::InflateFlush::Finish),
            Ok(Status::StreamEnd)
        );
        assert_eq!(usize::try_from(decoder.total_in()), Ok(first_stream.len()));
        assert_eq!(usize::try_from(decoder.total_out()), Ok(first.len()));
        assert_eq!(&decoded[..first.len()], first);
    }

    #[test]
    #[cfg_attr(
        not(target_pointer_width = "64"),
        ignore = "test only works on 64-bit systems"
    )]
    #[cfg_attr(miri, ignore = "slow")]
    fn buffer_overflows_c_uint() {
        // "Hello World!" compressed at level 0
        let deflated = [
            0x78u8, 0x01, 0x01, 0x0d, 0x00, 0xf2, 0xff, 0x48, 0x65, 0x6c, 0x6c, 0x6f, 0x20, 0x57,
            0x6f, 0x72, 0x6c, 0x64, 0x21, 0x0a, 0x20, 0x91, 0x04, 0x48,
        ];

        let mut v = vec![0u8; u32::MAX as usize + deflated.len() / 2];
        v[..deflated.len()].copy_from_slice(&deflated);

        assert!(((v.len() as u32) as usize) < deflated.len());

        let mut decompressed = [0u8; 64];
        let mut inflate = zlib_rs::Inflate::new(true, 15);
        let status = inflate
            .decompress(&v, &mut decompressed, zlib_rs::InflateFlush::Finish)
            .unwrap();

        assert_eq!(status, zlib_rs::Status::StreamEnd);
        assert_eq!(b"Hello World!", &decompressed[..12]);
    }
}

extern "C" fn push(desc: *mut c_void, _buf: *mut c_uchar, _len: c_uint) -> c_int {
    (!desc.is_null()) as _
}

// Callback input is per-invocation, not shared between parallel test runs.
struct BackPullSource<'a> {
    bytes: &'a [u8],
    position: usize,
}

unsafe extern "C" fn pull(desc: *mut c_void, buf: *mut *const c_uchar) -> c_uint {
    if desc.is_null() || buf.is_null() {
        return 0; // Initial input was supplied via next_in; no more data.
    }

    // SAFETY: the live input descriptor is a unique, writable BackPullSource,
    // independent of the z_stream and its internal state.
    let source = unsafe { &mut *desc.cast::<BackPullSource<'_>>() };
    let Some(remaining) = source.bytes.get(source.position..) else {
        return 0;
    };
    let Ok(length) = c_uint::try_from(remaining.len()) else {
        return 0; // Never panic across the extern "C" boundary.
    };
    if length == 0 {
        return 0;
    }
    // SAFETY: buf is a writable callback output pointer; remaining stays
    // alive, immutable, and readable for the rest of this decode call.
    unsafe { *buf = remaining.as_ptr() };
    source.position = source.bytes.len();
    length
}

/// Cover `inflateBack` edge cases.
#[test]
fn cover_back() {
    let mut window = [0u8; 32768];

    let ret = assert_eq_rs_ng!(unsafe {
        inflateBackInit_(
            core::ptr::null_mut(),
            0,
            window.as_mut_ptr(),
            core::ptr::null(),
            0,
        )
    });
    assert_eq!(ret, Z_VERSION_ERROR);

    let ret = assert_eq_rs_ng!(unsafe {
        inflateBackInit_(
            core::ptr::null_mut(),
            0,
            window.as_mut_ptr(),
            VERSION,
            STREAM_SIZE,
        )
    });
    assert_eq!(ret, Z_STREAM_ERROR);

    let ret = unsafe {
        libz_rs_sys::inflateBack(
            core::ptr::null_mut(),
            None,
            core::ptr::null_mut(),
            None,
            core::ptr::null_mut(),
        )
    };
    assert_eq!(ret, Z_STREAM_ERROR);

    let ret = assert_eq_rs_ng!(unsafe { inflateBackEnd(core::ptr::null_mut(),) });
    assert_eq!(ret, Z_STREAM_ERROR);

    use libz_rs_sys::*;

    let mut stream = mem_setup();
    // SAFETY: the custom allocator, C stream and full 32 KiB output window
    // remain live throughout the initialized back-inflate lifetime.
    assert_eq!(
        unsafe {
            inflateBackInit_(
                &raw mut stream,
                15,
                window.as_mut_ptr(),
                VERSION,
                STREAM_SIZE,
            )
        },
        Z_OK
    );

    // Exercise a real non-null callback context. The empty DEFLATE block
    // is supplied by pull rather than through the initial next_in buffer.
    stream.avail_in = 0;
    stream.next_in = core::ptr::null_mut();
    let mut pulled = BackPullSource {
        bytes: b"\x03\0",
        position: 0,
    };
    // SAFETY: the independent input descriptor owns the live compressed
    // span and the decoder output window remains writable.
    let ret = unsafe {
        inflateBack(
            &raw mut stream,
            Some(pull),
            (&raw mut pulled).cast(),
            Some(push),
            core::ptr::null_mut(),
        )
    };
    assert_eq!(ret, Z_STREAM_END);
    assert_eq!(pulled.position, pulled.bytes.len());

    // A non-null output callback descriptor requests a deliberate failure.
    let mut input = *b"\x63\0\0";
    stream.avail_in = test_c_uint_len(input.len());
    stream.next_in = input.as_mut_ptr();
    let mut output_error_context = 0u8;
    // SAFETY: input remains live, and the push callback only checks
    // descriptor nullness without touching decoder state or its window.
    let ret = unsafe {
        inflateBack(
            &raw mut stream,
            Some(pull),
            core::ptr::null_mut(),
            Some(push),
            (&raw mut output_error_context).cast(),
        )
    };
    assert_eq!(ret, Z_BUF_ERROR);

    // Invalid callback configurations must produce StreamError without
    // reading from or mutating the initialized decoder. This replaces the
    // old UB-only test that altered Mode::Sync from inside a callback:
    // inflateBack resets its state to Mode::Type at the start of every call.
    // SAFETY: the initialized decoder remains live; the absent input
    // callback is rejected before any callback-dependent buffer access.
    let ret = unsafe {
        inflateBack(
            &raw mut stream,
            None,
            core::ptr::null_mut(),
            Some(push),
            core::ptr::null_mut(),
        )
    };
    assert_eq!(ret, Z_STREAM_ERROR);

    // SAFETY: likewise, a missing output callback is rejected without
    // touching the output window or any initial input data.
    let ret = unsafe {
        inflateBack(
            &raw mut stream,
            Some(pull),
            core::ptr::null_mut(),
            None,
            core::ptr::null_mut(),
        )
    };
    assert_eq!(ret, Z_STREAM_ERROR);

    // SAFETY: this initialized back-inflate stream is closed once.
    assert_eq!(unsafe { inflateBackEnd(&raw mut stream) }, Z_OK);
    mem_done(&mut stream);

    // Re-initialize the same caller-owned window with fresh allocator
    // storage and verify the previous teardown left no stale state.
    let mut stream = mem_setup();
    // SAFETY: this newly allocated stream owns valid C ABI callbacks and
    // uses the still-live independent window.
    assert_eq!(
        unsafe {
            inflateBackInit_(
                &raw mut stream,
                15,
                window.as_mut_ptr(),
                VERSION,
                STREAM_SIZE,
            )
        },
        Z_OK
    );
    // SAFETY: matching closure of the second initialized stream.
    assert_eq!(unsafe { inflateBackEnd(&raw mut stream) }, Z_OK);
    mem_done(&mut stream);
}

#[test]
fn done_state_returns_stream_end() {
    let input = [
        31u8, 139, 8, 0, 0, 0, 0, 0, 0, 3, 203, 72, 205, 201, 201, 87, 40, 207, 47, 202, 73, 81,
        200, 0, 179, 33, 36, 68, 4, 89, 28, 137, 13, 0, 181, 147, 9, 162, 53, 0, 0, 0,
    ];
    const PLAIN: &[u8] = b"hello world hello hello world world hello world hello";

    assert_eq_rs_ng!({
        let mut output = [0u8; 64];
        let mut raw = MaybeUninit::<z_stream>::zeroed();

        // SAFETY: writable uninitialized C stream storage with the
        // correct gzip window mode, ABI version and checked size.
        let ret = unsafe {
            inflateInit2_(
                raw.as_mut_ptr(),
                15 + 16,
                zlibVersion(),
                test_c_stream_size::<z_stream>(),
            )
        };
        assert_eq!(ret, Z_OK);
        // SAFETY: successful initialization populated every C stream field.
        let stream = unsafe { raw.assume_init_mut() };

        stream.avail_in = test_c_uint_len(input.len());
        stream.next_in = input.as_ptr().cast_mut();
        stream.avail_out = test_c_uint_len(output.len());
        stream.next_out = output.as_mut_ptr();

        // SAFETY: decoder reads the complete live compressed fixture and
        // writes into the disjoint writable output array.
        assert_eq!(unsafe { inflate(stream, Z_NO_FLUSH) }, Z_STREAM_END);
        assert_eq!(stream.avail_in, 0);
        assert_eq!(usize::try_from(stream.total_in), Ok(input.len()));
        assert_eq!(usize::try_from(stream.total_out), Ok(PLAIN.len()));
        assert_eq!(&output[..PLAIN.len()], PLAIN);

        let consumed = stream.total_in;
        let produced = stream.total_out;
        // Even new valid input offered to the completed stream must remain
        // unconsumed, and no additional output bytes may be produced.
        let sentinel_input = *b"extra";
        let mut sentinel_output = [0xa5u8; 8];
        stream.avail_in = test_c_uint_len(sentinel_input.len());
        stream.next_in = sentinel_input.as_ptr().cast_mut();
        stream.avail_out = test_c_uint_len(sentinel_output.len());
        stream.next_out = sentinel_output.as_mut_ptr();

        // SAFETY: the complete decoder retains its live internal state;
        // both new caller-owned spans remain valid and disjoint.
        assert_eq!(unsafe { inflate(stream, Z_FINISH) }, Z_STREAM_END);
        assert_eq!(stream.total_in, consumed);
        assert_eq!(stream.total_out, produced);
        assert_eq!(stream.avail_in, test_c_uint_len(sentinel_input.len()));
        assert_eq!(stream.avail_out, test_c_uint_len(sentinel_output.len()));
        assert_eq!(sentinel_output, [0xa5u8; 8]);

        // SAFETY: the successfully initialized decoder is ended once.
        assert_eq!(unsafe { inflateEnd(stream) }, Z_OK);
    });
}

#[test]
fn inflate_copy_does_not_require_live_io_pointers() {
    assert_eq_rs_ng!({
        let mut raw_source = MaybeUninit::<z_stream>::zeroed();
        // SAFETY: writable stream storage and matching C ABI metadata.
        let ret = unsafe {
            inflateInit_(
                raw_source.as_mut_ptr(),
                zlibVersion(),
                test_c_stream_size::<z_stream>(),
            )
        };
        assert_eq!(ret, Z_OK);
        // SAFETY: inflateInit_ successfully initialized the whole stream.
        let source = unsafe { raw_source.assume_init_mut() };

        // inflateCopy must copy stream metadata without dereferencing I/O
        // pointers: this intentionally has a NULL input with avail_in > 0.
        source.next_in = core::ptr::null_mut();
        source.avail_in = 7;
        source.next_out = core::ptr::null_mut();
        source.avail_out = 0;

        let mut raw_copy = MaybeUninit::<z_stream>::zeroed();
        // SAFETY: source owns live inflate state; copying the state does not
        // require dereferencing its intentionally absent I/O buffers.
        let ret = unsafe { inflateCopy(raw_copy.as_mut_ptr(), source) };
        assert_eq!(ret, Z_OK);
        // SAFETY: successful inflateCopy initialized the entire destination.
        let copy = unsafe { raw_copy.assume_init_mut() };

        assert!(copy.next_in.is_null());
        assert!(copy.next_out.is_null());
        assert_eq!(copy.avail_in, source.avail_in);
        assert_eq!(copy.avail_out, source.avail_out);
        assert_eq!(copy.total_in, source.total_in);
        assert_eq!(copy.total_out, source.total_out);
        assert_ne!(
            copy.state, source.state,
            "inflateCopy must own distinct state"
        );

        // SAFETY: each initialized decoder owns distinct state; close both
        // exactly once without ever trying to decode through NULL I/O spans.
        assert_eq!(unsafe { inflateEnd(copy) }, Z_OK);
        // SAFETY: the live original decoder has not yet been closed.
        assert_eq!(unsafe { inflateEnd(source) }, Z_OK);
    });
}

#[test]
fn inflate_copy_after_half_input() {
    let input = include_bytes!("test-data/compression-corpus/The fastest WASM zlib.md.gzip-9.gz");

    assert_eq_rs_ng!({
        // Preserve the prefix already written when the output Vec grows.
        // A small initial buffer exercises reallocation on this fixture.
        fn grow_output(stream: &mut z_stream, output: &mut Vec<u8>) {
            assert_eq!(stream.avail_out, 0, "grow only after exhausting output");
            let used = usize::try_from(stream.total_out).unwrap_or(usize::MAX);
            assert_eq!(used, output.len(), "output pointer and count disagree");
            let old_len = output.len();
            let new_len = old_len.checked_mul(2).unwrap_or(old_len);
            assert!(new_len > old_len, "inflate output size overflow");
            output.resize(new_len, 0);
            stream.next_out = output[used..].as_mut_ptr();
            stream.avail_out = test_c_uint_len(new_len - used);
        }

        let mut raw = MaybeUninit::<z_stream>::zeroed();
        // SAFETY: writable C stream storage and matching version/structure size.
        let ret = unsafe {
            inflateInit2_(
                raw.as_mut_ptr(),
                16 + 15,
                zlibVersion(),
                test_c_stream_size::<z_stream>(),
            )
        };
        assert_eq!(ret, Z_OK);
        // SAFETY: successful inflateInit2_ initialized the entire C stream.
        let stream = unsafe { raw.assume_init_mut() };

        let mut out = vec![0u8; 1024];
        let half = input.len() / 2;
        assert!(half > 0 && half < input.len());
        stream.next_in = input.as_ptr().cast_mut();
        stream.avail_in = test_c_uint_len(half);
        stream.next_out = out.as_mut_ptr();
        stream.avail_out = test_c_uint_len(out.len());

        // Decode the entire first offered half, without assuming every call
        // consumes all input or has enough available output space.
        loop {
            let input_before = stream.avail_in;
            let output_before = stream.total_out;
            // SAFETY: input is a live slice of the fixture and output points
            // to the disjoint, live writable suffix of out.
            let ret = unsafe { inflate(stream, InflateFlush::NoFlush as _) };
            assert!(
                matches!(ret, Z_OK | Z_BUF_ERROR),
                "first half unexpectedly ended or failed: {ret}"
            );
            if stream.avail_out == 0 {
                grow_output(stream, &mut out);
            }
            if stream.avail_in == 0 {
                break;
            }
            assert!(
                stream.avail_in < input_before || stream.total_out > output_before,
                "inflate stalled before consuming the first half"
            );
        }
        assert_eq!(usize::try_from(stream.total_in), Ok(half));

        let prefix_len = usize::try_from(stream.total_out).unwrap_or(usize::MAX);
        assert!(prefix_len <= out.len());

        // Make an independent decoder copy only after the source init and
        // the state-copy call have both returned success.
        let mut raw_copy = MaybeUninit::<z_stream>::zeroed();
        // SAFETY: inflateCopy receives a fully initialized source stream and
        // writable destination; source internal state is valid and live.
        let ret = unsafe { inflateCopy(raw_copy.as_mut_ptr(), stream) };
        assert_eq!(ret, Z_OK);
        // SAFETY: successful inflateCopy initialized the copy's full ABI state.
        let copy = unsafe { raw_copy.assume_init_mut() };
        assert_eq!(copy.total_in, stream.total_in);
        assert_eq!(copy.total_out, stream.total_out);

        // Copy the already-decoded prefix and redirect the copied stream's
        // output pointer to its own allocation, not the original allocation.
        let mut out_copy = out.clone();
        copy.next_out = out_copy[prefix_len..].as_mut_ptr();
        copy.avail_out = test_c_uint_len(out_copy.len() - prefix_len);

        let remainder = &input[half..];
        stream.next_in = remainder.as_ptr().cast_mut();
        stream.avail_in = test_c_uint_len(remainder.len());
        copy.next_in = stream.next_in;
        copy.avail_in = stream.avail_in;

        loop {
            let input_before = stream.avail_in;
            let output_before = stream.total_out;

            // SAFETY: each independent initialized decoder reads the same
            // live compressed suffix, but writes into its own output Vec.
            let ret1 = unsafe { inflate(stream, InflateFlush::NoFlush as _) };
            // SAFETY: the cloned decoder owns its own internal state and its
            // next_out pointer refers to its distinct output allocation.
            let ret2 = unsafe { inflate(copy, InflateFlush::NoFlush as _) };

            assert_eq!(ret1, ret2, "copied decoder returned a different status");
            assert!(
                matches!(ret1, Z_OK | Z_BUF_ERROR | Z_STREAM_END),
                "unexpected inflate status after state copy: {ret1}"
            );
            assert_eq!(stream.total_in, copy.total_in);
            assert_eq!(stream.total_out, copy.total_out);
            assert_eq!(stream.avail_in, copy.avail_in);
            assert_eq!(stream.avail_out, copy.avail_out);

            if ret1 == Z_STREAM_END {
                break;
            }

            if stream.avail_out == 0 {
                grow_output(stream, &mut out);
                grow_output(copy, &mut out_copy);
            } else {
                assert!(
                    stream.avail_in < input_before || stream.total_out > output_before,
                    "copied inflate streams stalled before reaching Z_STREAM_END"
                );
            }
        }

        assert_eq!(stream.avail_in, 0, "gzip trailer left unconsumed");
        assert_eq!(copy.avail_in, 0);
        assert_eq!(usize::try_from(stream.total_in), Ok(input.len()));
        assert_eq!(stream.total_out, copy.total_out);

        let total = usize::try_from(stream.total_out).unwrap_or(usize::MAX);
        assert!(total <= out.len() && total <= out_copy.len());
        assert_eq!(&out[..total], &out_copy[..total]);
        // Independently measured with the reference zlib gzip decoder.
        assert_eq!(
            total, 5929,
            "copied streams did not decode the entire fixture"
        );
        assert_eq!(zlib_rs::crc32::crc32(0, &out[..total]), 0xefc2_9986);

        // SAFETY: source and copied streams are distinct, successfully
        // initialized states and each is ended exactly once.
        assert_eq!(unsafe { inflateEnd(stream) }, Z_OK);
        // SAFETY: the copied stream has not yet been ended.
        assert_eq!(unsafe { inflateEnd(copy) }, Z_OK);
        out[..total].to_vec()
    });
}

#[test]
fn inflate_validate_toggles_checksum_update() {
    let input = include_bytes!("test-data/compression-corpus/The fastest WASM zlib.md.gzip-9.gz");

    // The three passes must decode exactly the same complete gzip stream:
    // checksum enabled, disabled, then explicitly toggled off and on again.
    assert_eq_rs_ng!({
        let mut raw = MaybeUninit::<z_stream>::zeroed();
        // SAFETY: writable z_stream storage with matching ABI version/size.
        let ret = unsafe {
            inflateInit2_(
                raw.as_mut_ptr(),
                16 + 15,
                zlibVersion(),
                test_c_stream_size::<z_stream>(),
            )
        };
        assert_eq!(ret, Z_OK);
        // SAFETY: successful initialization populated every stream field.
        let stream = unsafe { raw.assume_init_mut() };

        let mut checks = Vec::new();
        let mut reference_output: Option<Vec<u8>> = None;
        for pass in 0..3 {
            if pass != 0 {
                // SAFETY: inflateReset reuses the existing initialized stream
                // after the preceding run completed successfully.
                assert_eq!(unsafe { inflateReset(stream) }, Z_OK);
            }
            let validate = if pass == 1 { 0 } else { 1 };
            if pass == 2 {
                // This third run exercises the *transition* back to enabled,
                // not just starting with validation already enabled.
                // SAFETY: the initialized decoder has no pending input.
                assert_eq!(unsafe { inflateValidate(stream, 0) }, Z_OK);
            }
            // SAFETY: control call on a live initialized inflate stream.
            assert_eq!(unsafe { inflateValidate(stream, validate) }, Z_OK);

            let mut out = vec![0u8; 1024];
            stream.next_in = input.as_ptr().cast_mut();
            stream.avail_in = test_c_uint_len(input.len());
            stream.next_out = out.as_mut_ptr();
            stream.avail_out = test_c_uint_len(out.len());

            loop {
                let before_in = stream.avail_in;
                let before_out = stream.total_out;
                // SAFETY: the compressed input is live and immutable, and
                // next_out points into a disjoint writable output allocation.
                let ret = unsafe { inflate(stream, InflateFlush::NoFlush as _) };
                assert!(
                    matches!(ret, Z_OK | Z_BUF_ERROR | Z_STREAM_END),
                    "unexpected inflate return on validation pass {pass}: {ret}"
                );
                if ret == Z_STREAM_END {
                    break;
                }

                if stream.avail_out == 0 {
                    let produced = usize::try_from(stream.total_out).unwrap_or(usize::MAX);
                    assert_eq!(produced, out.len(), "invalid output position");
                    let old_len = out.len();
                    let new_len = old_len.checked_mul(2).unwrap_or(old_len);
                    assert!(new_len > old_len, "output capacity overflow");
                    out.resize(new_len, 0);
                    stream.next_out = out[produced..].as_mut_ptr();
                    stream.avail_out = test_c_uint_len(new_len - produced);
                } else {
                    assert!(
                        stream.avail_in < before_in || stream.total_out > before_out,
                        "decoder stalled during validation pass {pass}"
                    );
                }
                assert!(
                    stream.avail_in != 0 || stream.avail_out != 0,
                    "decoder exhausted input without reaching stream end on pass {pass}"
                );
            }

            assert_eq!(stream.avail_in, 0, "gzip trailer not fully consumed");
            assert_eq!(usize::try_from(stream.total_in), Ok(input.len()));
            let produced = usize::try_from(stream.total_out).unwrap_or(usize::MAX);
            assert!(produced <= out.len(), "inflate wrote beyond output");
            out.truncate(produced);

            // Fixed external reference generated with the system zlib gzip
            // decoder. Even with checksum validation disabled, output bytes
            // must remain exactly correct.
            assert_eq!(out.len(), 5929);
            assert_eq!(zlib_rs::crc32::crc32(0, &out), 0xefc2_9986);
            if let Some(previous) = &reference_output {
                assert_eq!(&out, previous, "validation toggle altered plaintext");
            } else {
                reference_output = Some(out);
            }
            checks.push(stream.adler);
        }

        assert_eq!(checks.len(), 3);
        assert_eq!(checks[0], 0xefc2_9986, "validated gzip CRC mismatch");
        assert_ne!(checks[0], checks[1], "disabling validation had no effect");
        assert_eq!(checks[0], checks[2], "re-enabling validation changed CRC");
        // SAFETY: all three complete runs used the same live stream, closed
        // exactly once after the last pass.
        assert_eq!(unsafe { inflateEnd(stream) }, Z_OK);
    });
}
