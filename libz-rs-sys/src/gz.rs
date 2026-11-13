use zlib_rs::c_api::*;

#[cfg(feature = "export-symbols")]
use crate::prefix;
use crate::{
    deflate, deflateEnd, deflateInit2_, deflateReset, inflate, inflateEnd, inflateInit2_,
    inflateReset, z_off64_t, z_off_t, zlibVersion,
};
use core::cmp::Ordering;
use core::ffi::{c_char, c_int, c_uint, c_void, CStr};
use core::mem::MaybeUninit;
use core::ptr::{self, NonNull};
use libc::size_t; // FIXME: Switch to core::ffi::c_size_t when it's stable.
use libc::{O_APPEND, O_CREAT, O_EXCL, O_RDONLY, O_TRUNC, O_WRONLY, SEEK_CUR, SEEK_END, SEEK_SET};
use zlib_rs::deflate::Strategy;
use zlib_rs::MAX_WBITS;

/// Opaque gzip file handle compatible with the zlib C API.
///
/// In the zlib C API, this structure exposes just enough of the internal state
/// of an open [`gzFile`] to support the `gzgetc` C macro. Since Rust code won't be
/// using that C macro, we define [`gzFile_s`] as an empty structure.
// For ABI compatibility with zlib and zlib-ng, the first fields in [`GzState`] match
// what would be in the C version of [`gzFile_s`]. But we don't want new users to rely
// on this internal implementation, so the Rust [`gzFile_s`] is intentionally opaque.
#[repr(C)]
pub struct gzFile_s {
    _private: [u8; 0],
}

/// File handle for an open gzip file.
#[expect(
    non_camel_case_types,
    reason = "gzFile is the public zlib C handle type"
)]
pub type gzFile = *mut gzFile_s;

struct GzCString {
    ptr: NonNull<c_char>,
    len_with_nul: usize,
}

impl GzCString {
    fn clone_from_c_str(src: &CStr) -> Option<Self> {
        let bytes = src.to_bytes_with_nul();
        let ptr = ALLOCATOR.allocate_slice_raw::<c_char>(bytes.len())?;

        // Safety: `ptr` is a fresh allocation for exactly `bytes.len()` c_char values.
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr().cast::<c_char>(), ptr.as_ptr(), bytes.len())
        };

        Some(Self {
            ptr,
            len_with_nul: bytes.len(),
        })
    }

    fn concat(parts: &[&[u8]]) -> Option<Self> {
        let len_with_nul = parts
            .iter()
            .try_fold(1usize, |len, part| len.checked_add(part.len()))?;
        let ptr = ALLOCATOR.allocate_slice_raw::<c_char>(len_with_nul)?;
        let mut offset = 0usize;

        for part in parts {
            // Safety: `offset + part.len()` is bounded by `len_with_nul - 1`, and
            // the fresh destination allocation cannot overlap any source slice.
            unsafe {
                ptr::copy_nonoverlapping(
                    part.as_ptr(),
                    ptr.as_ptr().cast::<u8>().add(offset),
                    part.len(),
                )
            };
            offset += part.len();
        }
        // Safety: `offset == len_with_nul - 1`, so the terminator is in-bounds.
        unsafe { *ptr.as_ptr().cast::<u8>().add(offset) = 0 };

        Some(Self { ptr, len_with_nul })
    }

    const fn as_ptr(&self) -> *const c_char {
        self.ptr.as_ptr()
    }

    fn as_c_str(&self) -> &CStr {
        // Safety: construction always writes one trailing NUL after bytes that came from
        // either a C string or internal byte slices, and the allocation stays owned by self.
        unsafe { CStr::from_ptr(self.as_ptr()) }
    }
}

impl Drop for GzCString {
    fn drop(&mut self) {
        // Safety: `ptr` was allocated by `ALLOCATOR` for exactly `len_with_nul` c_char values.
        unsafe { ALLOCATOR.deallocate(self.ptr.as_ptr(), self.len_with_nul) };
    }
}

struct GzBuffer {
    ptr: NonNull<u8>,
    capacity: usize,
    // Cached C ABI length, checked against the allocator's usize at creation.
    // Neither field can be mutated after allocation.
    usable: usize,
    usable_abi: c_uint,
}

impl GzBuffer {
    // Every owned byte buffer is both a Rust allocation/slice and a zlib C
    // availability span. On 32-bit targets, isize::MAX is the tighter bound.
    fn checked_abi_len(len: usize) -> Option<c_uint> {
        if len > isize::MAX.unsigned_abs() {
            return None;
        }
        c_uint::try_from(len).ok()
    }

    fn allocate(capacity: usize, usable: usize) -> Option<Self> {
        // Also guard private callers: never store lengths that can wrap when
        // published through the C ABI, or create an invalid Rust slice.
        if capacity == 0 || usable > capacity {
            return None;
        }
        Self::checked_abi_len(capacity)?;
        let usable_abi = Self::checked_abi_len(usable)?;
        let ptr = ALLOCATOR.allocate_slice_raw::<u8>(capacity)?;
        Some(Self {
            ptr,
            capacity,
            usable,
            usable_abi,
        })
    }

    const fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    // Validate an ABI cursor using integer addresses, never offset_from on
    // potentially unrelated pointers. All returned slices are subsequently
    // derived from the allocation owner, retaining its pointer provenance.
    fn checked_range_within(
        &self,
        cursor: *const u8,
        len: usize,
        limit: usize,
    ) -> Option<core::ops::Range<usize>> {
        let start = (cursor as usize).checked_sub(self.as_ptr() as usize)?;
        if start > limit || len > limit - start {
            return None;
        }
        Some(start..start + len)
    }

    fn checked_range(&self, cursor: *const u8, len: usize) -> Option<core::ops::Range<usize>> {
        self.checked_range_within(cursor, len, self.usable)
    }

    // Write input has additional owned capacity for formatted output.
    fn checked_allocated_range(
        &self,
        cursor: *const u8,
        len: usize,
    ) -> Option<core::ops::Range<usize>> {
        self.checked_range_within(cursor, len, self.capacity)
    }

    const fn usable(&self) -> usize {
        self.usable
    }

    fn usable_uninit_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        // SAFETY: this buffer uniquely owns an allocation of at least its
        // usable length. MaybeUninit<u8> allows uninitialized bytes.
        unsafe {
            core::slice::from_raw_parts_mut(
                self.ptr.as_ptr().cast::<MaybeUninit<u8>>(),
                self.usable(),
            )
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PushBackError {
    Full,
    InvalidCursor,
}

struct GzWriteInput {
    next: *const u8,
    remaining: usize,
}

impl GzWriteInput {
    /// Bind a caller-owned readable span for the duration of one gzip write operation.
    ///
    /// # Safety
    /// `next` must be readable for `len` bytes when `len != 0`.\n    /// The entire span must also be no longer than `isize::MAX` bytes.
    unsafe fn from_raw(next: *const c_void, len: usize) -> Self {
        debug_assert!(len == 0 || !next.is_null());
        debug_assert!(len <= isize::MAX.unsigned_abs());
        Self {
            next: next.cast::<u8>(),
            remaining: len,
        }
    }

    const fn from_slice(input: &[u8]) -> Self {
        Self {
            next: input.as_ptr(),
            remaining: input.len(),
        }
    }

    const fn len(&self) -> usize {
        self.remaining
    }

    const fn is_empty(&self) -> bool {
        self.remaining == 0
    }

    const fn as_ptr(&self) -> *const u8 {
        self.next
    }

    fn advance(&mut self, len: usize) {
        assert!(len <= self.remaining);
        if len != 0 {
            // Safety: construction guarantees the full remaining span is readable,
            // and `len` is bounded by that remaining span.
            self.next = unsafe { self.next.add(len) };
            self.remaining -= len;
        }
    }

    fn copy_to(&mut self, output: &mut [MaybeUninit<u8>]) -> usize {
        let len = Ord::min(self.remaining, output.len());
        if len != 0 {
            // Safety: the input cursor guarantees `len` readable bytes and `output`
            // provides `len` writable bytes. `copy` deliberately tolerates overlap.
            unsafe { ptr::copy(self.next, output.as_mut_ptr().cast::<u8>(), len) };
            self.advance(len);
        }
        len
    }
}

impl Drop for GzBuffer {
    fn drop(&mut self) {
        // Safety: `ptr` was allocated by `ALLOCATOR` for exactly `capacity` bytes,
        // and this `GzBuffer` uniquely owns that allocation.
        unsafe { ALLOCATOR.deallocate(self.ptr.as_ptr(), self.capacity) };
    }
}

// The internals of a gzip file handle (the thing gzFile actually points to, with the
// public gzFile_s part at the front for ABI compatibility).
#[repr(C)]
struct GzState {
    // Public interface:
    // These first three fields must match the structure gzFile_s in the C version
    // of zlib. In the C library, a macro called gzgetc() reads and writes these
    // fields directly.
    have: c_uint,       // number of bytes available at next
    next: *const Bytef, // next byte of uncompressed data
    pos: i64,           // current offset in uncompressed data stream

    // End of public interface:
    // All fields after this point are opaque to C code using this library,
    // so they can be rearranged without breaking compatibility.

    // Fields used for both reading and writing
    mode: GzMode,
    fd: c_int, // file descriptor
    source: Option<Source>,
    want: usize,              // requested buffer size, default is GZBUFSIZE
    input: Option<GzBuffer>,  // input storage (double-sized when writing)
    output: Option<GzBuffer>, // output storage (double-sized when reading)
    direct: bool,             // true in pass-through mode, false if processing gzip data

    // Fields used just for reading
    how: How,
    start: i64,
    eof: bool,  // whether we have reached the end of the input file
    past: bool, // whether a read past the end has been requested

    // Fields used just for writing
    level: i8,
    strategy: Strategy,
    reset: bool, // whether a reset is pending after a Z_FINISH

    // Fields used for seek requests
    skip: i64,  // amount to skip (already rewound if backwards)
    seek: bool, // whether a seek request is pending

    // Error information
    err: c_int,             // last error (0 if no error)
    msg: Option<GzCString>, // error message from last error
    // A nonblocking read that made no progress may be retried when more data arrives.
    read_would_block: bool,

    // zlib inflate or deflate stream
    stream: z_stream,
}

impl GzState {
    fn new() -> Self {
        let mut stream = z_stream::default();
        stream.zalloc = Some(ALLOCATOR.zalloc);
        stream.zfree = Some(ALLOCATOR.zfree);
        stream.opaque = ALLOCATOR.opaque;

        Self {
            have: 0,
            next: ptr::null(),
            pos: 0,
            mode: GzMode::None,
            fd: -1,
            source: None,
            want: GZBUFSIZE,
            input: None,
            output: None,
            direct: false,
            how: How::Look,
            start: 0,
            eof: false,
            past: false,
            level: -1, // The C API default compression level.
            strategy: Strategy::Default,
            reset: false,
            skip: 0,
            seek: false,
            err: Z_OK,
            msg: None,
            read_would_block: false,
            stream,
        }
    }

    // The only recoverable read error is a genuine nonblocking stall.
    // Other I/O errors remain sticky until an explicit gzclearerr().
    fn prepare_read(&mut self) -> bool {
        if self.mode != GzMode::Read {
            return false;
        }
        if self.read_would_block {
            gz_error(self, None);
        }
        self.err == Z_OK || self.err == Z_BUF_ERROR
    }

    fn configure(&mut self, mode: &[u8]) -> Result<(bool, bool), ()> {
        let mut exclusive = false;
        let mut cloexec = false;

        for &ch in mode {
            if ch.is_ascii_digit() {
                self.level = i8::try_from(ch - b'0').map_err(|_| ())?;
            } else {
                match ch {
                    b'r' => self.mode = GzMode::Read,
                    b'w' => self.mode = GzMode::Write,
                    b'a' => self.mode = GzMode::Append,
                    b'+' => {
                        // Read+Write mode isn't supported
                        return Err(());
                    }
                    b'b' => {} // binary mode is the default
                    b'e' => cloexec = true,
                    b'x' => exclusive = true,
                    b'f' => self.strategy = Strategy::Filtered,
                    b'h' => self.strategy = Strategy::HuffmanOnly,
                    b'R' => self.strategy = Strategy::Rle,
                    b'F' => self.strategy = Strategy::Fixed,
                    b'T' => self.direct = true,
                    _ => {} // for compatibility with zlib-ng, ignore unexpected characters in the mode
                }
            }
        }

        Ok((exclusive, cloexec))
    }

    // Every gzip allocation must fit both the Rust allocator and the
    // z_stream avail_in/avail_out ABI counters, including the doubled buffer.
    fn doubled_buffer_capacity(want: usize) -> Option<usize> {
        let doubled = want.checked_mul(2)?;
        GzBuffer::checked_abi_len(doubled)?;
        Some(doubled)
    }

    // Get the number of bytes allocated for the input buffer.
    fn in_capacity(&self) -> Option<usize> {
        let doubled = Self::doubled_buffer_capacity(self.want)?;
        Some(if self.mode == GzMode::Write {
            doubled
        } else {
            self.want
        })
    }

    // Get the number of bytes allocated for the output buffer.
    fn out_capacity(&self) -> Option<usize> {
        let doubled = Self::doubled_buffer_capacity(self.want)?;
        Some(if self.mode == GzMode::Read {
            doubled
        } else {
            self.want
        })
    }

    fn input_ptr(&self) -> *mut u8 {
        self.input
            .as_ref()
            .expect("gzip input buffer must be initialized")
            .as_ptr()
    }

    fn output_ptr(&self) -> *mut u8 {
        self.output
            .as_ref()
            .expect("gzip output buffer must be initialized")
            .as_ptr()
    }

    fn allocate_input_buffer(&self) -> Option<GzBuffer> {
        let capacity = self.in_capacity()?;
        let usable = match self.mode {
            GzMode::Write => self.want,
            _ => capacity,
        };
        GzBuffer::allocate(capacity, usable)
    }

    fn allocate_output_buffer(&self) -> Option<GzBuffer> {
        let capacity = self.out_capacity()?;
        GzBuffer::allocate(capacity, capacity)
    }

    fn in_size(&self) -> usize {
        self.input.as_ref().map_or(0, GzBuffer::usable)
    }

    fn out_avail(&self) -> c_uint {
        self.output.as_ref().map_or(0, |buf| buf.usable_abi)
    }
    // The uncompressed gzip position is a signed 64-bit C offset. Check
    // *before* publishing or consuming bytes, not after a buffer mutation.
    fn checked_position_after(&self, bytes: usize) -> Option<i64> {
        self.pos.checked_add(i64::try_from(bytes).ok()?)
    }

    fn position_room(&self) -> usize {
        let available = i128::from(i64::MAX) - i128::from(self.pos);
        usize::try_from(available).unwrap_or(usize::MAX)
    }

    fn fail_position_overflow(&mut self) {
        gz_error(
            self,
            Some((
                Z_STREAM_ERROR,
                "uncompressed position exceeds signed 64-bit range",
            )),
        );
    }

    fn ensure_position_room(&mut self, bytes: usize) -> Result<(), ()> {
        let pending = if self.seek { i128::from(self.skip) } else { 0 };
        let Ok(bytes) = i128::try_from(bytes) else {
            self.fail_position_overflow();
            return Err(());
        };
        let target = i128::from(self.pos) + pending + bytes;
        if i64::try_from(target).is_err() {
            self.fail_position_overflow();
            return Err(());
        }
        Ok(())
    }

    fn advance_position(&mut self, bytes: usize) -> Result<(), ()> {
        let Some(pos) = self.checked_position_after(bytes) else {
            self.fail_position_overflow();
            return Err(());
        };
        self.pos = pos;
        Ok(())
    }

    /// Validate the publicly visible read cursor without dereferencing it.
    /// An inline C gzgetc macro may have advanced this pointer and length.
    fn read_output_range(&self) -> Option<core::ops::Range<usize>> {
        if self.have == 0 {
            return Some(0..0);
        }
        let output = self.output.as_ref()?;
        output.checked_range(self.next, self.have as usize)
    }

    /// Expose initialized read output through a pointer derived from its owner.
    fn buffered_output(&self) -> Option<&[u8]> {
        let range = self.read_output_range()?;
        if range.is_empty() {
            return Some(&[]);
        }
        let output = self.output.as_ref()?;
        // SAFETY: checked_range established that these initialized bytes lie
        // in the live output allocation; the pointer comes from its owner.
        Some(unsafe { core::slice::from_raw_parts(output.as_ptr().add(range.start), range.len()) })
    }

    fn consume_buffered_output(&mut self, len: usize) -> bool {
        let Some(range) = self.read_output_range() else {
            return false;
        };
        let Ok(consumed) = c_uint::try_from(len) else {
            return false;
        };
        if len > range.len() {
            return false;
        }
        if len != 0 {
            let Some(output) = self.output.as_ref() else {
                return false;
            };
            // SAFETY: checked_range and len establish an in-allocation
            // cursor (possibly one-past). Preserve the owner's provenance.
            self.next = unsafe { output.as_ptr().add(range.start + len) };
            self.have -= consumed;
        }
        true
    }

    fn invalid_read_cursor(&mut self) {
        gz_error(
            self,
            Some((Z_STREAM_ERROR, "invalid gzip read buffer cursor")),
        );
    }

    /// Publish an initialized range in the owned read output buffer through the ABI prefix.
    ///
    /// Callers must only use ranges they have just initialized through a typed output slice.
    fn publish_buffered_output(&mut self, offset: usize, len: usize) {
        debug_assert_eq!(self.mode, GzMode::Read);
        let output = self
            .output
            .as_ref()
            .expect("read output must be allocated before publishing bytes");
        assert!(offset <= output.usable());
        assert!(len <= output.usable() - offset);
        let have = c_uint::try_from(len).expect("gzip buffered output length must fit in uInt");
        // Safety: `offset` is in-bounds (or one-past when `len == 0`) for this allocation.
        self.next = unsafe { output.as_ptr().add(offset) };
        self.have = have;
    }

    fn push_back_buffered_byte(&mut self, byte: u8) -> Result<(), PushBackError> {
        let Some(output) = self.output.as_ref() else {
            return Err(PushBackError::InvalidCursor);
        };
        let size = output.usable();
        let have = self.have as usize;
        if have > size {
            return Err(PushBackError::InvalidCursor);
        }
        if have == size {
            return Err(PushBackError::Full);
        }
        let range = if have == 0 {
            0..0
        } else {
            output
                .checked_range(self.next, have)
                .ok_or(PushBackError::InvalidCursor)?
        };

        // Push into the final byte if the buffer is currently empty. For an
        // already-buffered prefix, make room if its cursor is at offset zero.
        let new_offset = if have == 0 {
            size - 1
        } else if range.start == 0 {
            let moved_offset = size - have;
            let Some(output) = self.output.as_mut() else {
                return Err(PushBackError::InvalidCursor);
            };
            output
                .usable_uninit_mut()
                .copy_within(0..have, moved_offset);
            moved_offset - 1
        } else {
            range.start - 1
        };
        let Some(output) = self.output.as_mut() else {
            return Err(PushBackError::InvalidCursor);
        };
        output.usable_uninit_mut()[new_offset].write(byte);
        self.publish_buffered_output(new_offset, have + 1);
        Ok(())
    }

    fn read_input_range(&self) -> Option<core::ops::Range<usize>> {
        let len = self.stream.avail_in as usize;
        if len == 0 {
            return Some(0..0);
        }
        self.input.as_ref()?.checked_range(self.stream.next_in, len)
    }

    /// Read initialized bytes through a cursor checked against owned input.
    fn buffered_read_input(&self) -> Option<&[u8]> {
        let range = self.read_input_range()?;
        if range.is_empty() {
            return Some(&[]);
        }
        let input = self.input.as_ref()?;
        // SAFETY: the checked range is inside the initialized owned input,
        // and this pointer is derived from its owner, not from the ABI cursor.
        Some(unsafe { core::slice::from_raw_parts(input.as_ptr().add(range.start), range.len()) })
    }

    /// Publish an initialized input range to the internal inflate stream.
    fn publish_read_input(&mut self, offset: usize, len: usize) -> bool {
        let Some(input) = self.input.as_ref() else {
            return false;
        };
        if offset > input.usable() || len > input.usable() - offset {
            return false;
        }
        let Ok(avail) = c_uint::try_from(len) else {
            return false;
        };
        // SAFETY: checked bounds establish a valid owned input cursor,
        // including the one-past address for an empty tail.
        self.stream.next_in = unsafe { input.as_ptr().add(offset) };
        self.stream.avail_in = avail;
        true
    }

    fn clear_read_input(&mut self) {
        debug_assert_eq!(self.mode, GzMode::Read);
        self.stream.avail_in = 0;
        self.stream.next_in = self
            .input
            .as_ref()
            .map_or(ptr::null(), |input| input.as_ptr().cast_const());
    }

    /// Move the remaining initialized read input to the start of owned storage.
    fn compact_read_input(&mut self) -> Option<usize> {
        let range = self.read_input_range()?;
        let len = range.len();
        if len != 0 && range.start != 0 {
            let input = self.input.as_mut()?;
            input.usable_uninit_mut().copy_within(range, 0);
        }
        self.publish_read_input(0, len).then_some(len)
    }

    /// Return the logical end offset of write input currently owned by this state.
    ///
    /// The large-write zero-copy path temporarily violates this invariant, but it does
    /// not call this helper while caller memory is bound to `next_in`.
    fn owned_write_input_end(&self) -> Option<usize> {
        let input = self.input.as_ref()?;
        input
            .checked_allocated_range(self.stream.next_in, self.stream.avail_in as usize)
            .map(|range| range.end)
    }

    // Validate both write output cursors and the deflate-owned suffix.
    // Slices are derived from the owned allocation, never from ABI cursors.
    fn pending_write_output(&self) -> Option<&[u8]> {
        let output = self.output.as_ref()?;
        let start = output.checked_range(self.next, 0)?.start;
        let remaining =
            output.checked_range(self.stream.next_out, self.stream.avail_out as usize)?;
        if start > remaining.start || remaining.end != output.usable() {
            return None;
        }
        // SAFETY: deflate initializes the prefix; both cursors were validated.
        Some(unsafe {
            core::slice::from_raw_parts(output.as_ptr().add(start), remaining.start - start)
        })
    }

    fn invalid_write_cursor(&mut self) {
        gz_error(
            self,
            Some((Z_STREAM_ERROR, "invalid gzip write buffer cursor")),
        );
    }
}

/// Owns the custom allocation containing one fully initialized [`GzState`].
///
/// A live gzip C handle is an escaped `GzStateOwner`; the matching close function
/// reclaims that ownership after validating that it is the correct close operation.
struct GzStateOwner {
    ptr: NonNull<GzState>,
}

impl GzStateOwner {
    fn allocate() -> Option<Self> {
        let ptr = ALLOCATOR.allocate_raw::<GzState>()?;
        // Safety: `ptr` has the size and alignment required for `GzState`, and this
        // owner has exclusive access to the fresh allocation.
        unsafe { ptr.as_ptr().write(GzState::new()) };
        Some(Self { ptr })
    }

    const fn as_mut(&mut self) -> &mut GzState {
        // Safety: the owner uniquely owns a live, fully initialized `GzState`.
        unsafe { self.ptr.as_mut() }
    }

    const fn into_raw(self) -> NonNull<GzState> {
        let ptr = self.ptr;
        core::mem::forget(self);
        ptr
    }

    /// Reclaim an allocation previously escaped through [`Self::into_raw`].
    ///
    /// # Safety
    /// `ptr` must identify a live `GzState` allocation produced by `into_raw`, and
    /// ownership of that allocation must be transferred exclusively to the returned owner.
    const unsafe fn reclaim(ptr: NonNull<GzState>) -> Self {
        Self { ptr }
    }
}

impl Drop for GzStateOwner {
    fn drop(&mut self) {
        // Safety: the owner uniquely owns one fully initialized `GzState` in an
        // allocation created by `ALLOCATOR`. Run field drop glue before returning storage.
        unsafe {
            ptr::drop_in_place(self.ptr.as_ptr());
            ALLOCATOR.deallocate(self.ptr.as_ptr(), 1);
        }
    }
}

// Gzip operating modes
// NOTE: These values match what zlib-ng uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GzMode {
    None = 0,
    Read = 7247,
    Write = 31153,
    Append = 1,
}

// gzip read strategies
// NOTE: These values match what zlib-ng uses.
#[derive(Debug, PartialEq, Eq)]
enum How {
    Look = 0, // look for a gzip header
    Copy = 1, // copy input directly
    Gzip = 2, // decompress a gzip stream
}

const GZBUFSIZE: usize = 128 * 1024;

#[cfg(feature = "rust-allocator")]
use zlib_rs::allocate::RUST as ALLOCATOR;

#[cfg(not(feature = "rust-allocator"))]
#[cfg(feature = "c-allocator")]
use zlib_rs::allocate::C as ALLOCATOR;

#[cfg(not(feature = "rust-allocator"))]
#[cfg(not(feature = "c-allocator"))]
compile_error!("Either rust-allocator or c-allocator feature is required");

// The different ways to specify the source for gzopen_bound
enum Source {
    Path(GzCString),
    Fd(c_int),
}

impl Source {
    /// Clone a caller-owned path so all state after the FFI boundary has explicit ownership.
    fn clone_path(path: &CStr) -> Option<Self> {
        GzCString::clone_from_c_str(path).map(Self::Path)
    }
}

/// Open a gzip file for reading or writing.
///
/// # Returns
///
/// * If successful, an opaque handle that the caller can later free with [`gzfree`]
/// * On error, a null pointer
///
/// # Safety
///
/// The caller must ensure that `path` and `mode` point to valid C strings. If the
/// return value is non-NULL, caller must delete it using only [`gzclose`].
///
/// [`gzfree`]: crate::z_stream
#[cfg_attr(feature = "export-symbols", export_name = prefix!(gzopen64))]
#[must_use]
pub unsafe extern "C" fn gzopen64(path: *const c_char, mode: *const c_char) -> gzFile {
    if path.is_null() {
        return ptr::null_mut();
    }
    // Safety: guaranteed by this public function's caller contract.
    let path = unsafe { CStr::from_ptr(path) };
    let Some(source) = Source::clone_path(path) else {
        return ptr::null_mut();
    };
    if mode.is_null() {
        return ptr::null_mut();
    }
    // Safety: guaranteed by this public function's caller contract.
    let mode = unsafe { CStr::from_ptr(mode) };
    gzopen_bound(source, mode).map_or(ptr::null_mut(), |owner| {
        owner.into_raw().as_ptr().cast::<gzFile_s>()
    })
}

/// Open a gzip file for reading or writing.
///
/// # Returns
///
/// * If successful, an opaque handle that the caller can later free with [`gzfree`]
/// * On error, a null pointer
///
/// # Safety
///
/// The caller must ensure that `path` and `mode` point to valid C strings. If the
/// return value is non-NULL, caller must delete it using only [`gzclose`].
///
/// [`gzfree`]: crate::z_stream
#[cfg_attr(feature = "export-symbols", export_name = prefix!(gzopen))]
#[must_use]
pub unsafe extern "C" fn gzopen(path: *const c_char, mode: *const c_char) -> gzFile {
    if path.is_null() {
        return ptr::null_mut();
    }
    // Safety: guaranteed by this public function's caller contract.
    let path = unsafe { CStr::from_ptr(path) };
    let Some(source) = Source::clone_path(path) else {
        return ptr::null_mut();
    };
    if mode.is_null() {
        return ptr::null_mut();
    }
    // Safety: guaranteed by this public function's caller contract.
    let mode = unsafe { CStr::from_ptr(mode) };
    gzopen_bound(source, mode).map_or(ptr::null_mut(), |owner| {
        owner.into_raw().as_ptr().cast::<gzFile_s>()
    })
}

/// Given an open file descriptor, prepare to read or write a gzip file.
/// NOTE: This is similar to [`gzopen`], but for cases where the caller already
/// has the file open.
///
/// # Returns
///
/// * If successful, an opaque handle that the caller can later free with [`gzfree`]
/// * On error, a null pointer
///
/// The file descriptor is not used until the next `gz{read, write, seek, close}` operation,
/// so `gzdopen` will not detect if `fd` is invalid (unless `fd == -1` which returns a null
/// pointer).
///
/// # Safety
///
/// The caller must ensure that `mode` points to a valid C string. If the
/// return value is non-NULL, caller must delete it using only [`gzclose`].
///
/// [`gzfree`]: crate::z_stream
#[cfg_attr(feature = "export-symbols", export_name = prefix!(gzdopen))]
#[must_use]
pub unsafe extern "C" fn gzdopen(fd: c_int, mode: *const c_char) -> gzFile {
    if mode.is_null() {
        return ptr::null_mut();
    }
    // Safety: guaranteed by this public function's caller contract.
    let mode = unsafe { CStr::from_ptr(mode) };
    gzopen_bound(Source::Fd(fd), mode).map_or(ptr::null_mut(), |owner| {
        owner.into_raw().as_ptr().cast::<gzFile_s>()
    })
}

/// Safe construction core shared by the raw gzip-open FFI shims.
fn gzopen_bound(source: Source, mode: &CStr) -> Option<GzStateOwner> {
    let mut state_owner = GzStateOwner::allocate()?;
    let state = state_owner.as_mut();

    let Ok((exclusive, cloexec)) = state.configure(mode.to_bytes()) else {
        return None;
    };

    // Must specify read, write, or append
    if state.mode == GzMode::None {
        return None;
    }

    // Can't force transparent read
    if state.mode == GzMode::Read {
        if state.direct {
            return None;
        }
        state.direct = true; // Assume an empty file for now. Later, we'll check for a gzip header.
    }

    // Open the file unless the caller passed a file descriptor.
    match source {
        Source::Fd(fd) => {
            state.fd = fd;
            state.source = Some(Source::Fd(fd));
        }
        Source::Path(path) => {
            // The caller path was cloned at the FFI boundary, so it remains valid both for
            // opening the file and for later byte-exact error reporting.
            // FIXME: support Windows wide characters for compatibility with zlib-ng
            let path_ptr = path.as_ptr();
            let mut oflag = 0;

            #[cfg(target_os = "linux")]
            {
                oflag |= libc::O_LARGEFILE;
            }
            #[cfg(target_os = "windows")]
            {
                oflag |= libc::O_BINARY;
            }
            if cloexec {
                #[cfg(target_os = "linux")]
                {
                    oflag |= libc::O_CLOEXEC;
                }
            }

            if state.mode == GzMode::Read {
                oflag |= O_RDONLY;
            } else {
                oflag |= O_WRONLY | O_CREAT;
                if exclusive {
                    oflag |= O_EXCL;
                }
                if state.mode == GzMode::Write {
                    oflag |= O_TRUNC;
                } else {
                    oflag |= O_APPEND;
                }
            }
            // FIXME: support _wopen for WIN32
            // Safety: `path_ptr` points into the owned `path` C string.
            state.fd = unsafe { libc::open(path_ptr, oflag, 0o666) };
            state.source = Some(Source::Path(path));
        }
    }

    if state.fd == -1 {
        return None;
    }

    // NOTE: the `lseek64` return value is ignored, an invalid file descriptor will only be
    // detected by the next `gz{read, write, seek, close}`.
    if state.mode == GzMode::Append {
        let _ = lseek64(state.fd, 0, SEEK_END); // so gzoffset() is correct
        state.mode = GzMode::Write; // simplify later checks
    }

    if state.mode == GzMode::Read {
        // Save the current position for rewinding
        state.start = lseek64(state.fd, 0, SEEK_CUR) as _;
        if state.start == -1 {
            state.start = 0;
        }
    }

    // Initialize stream
    gz_reset(state);

    Some(state_owner)
}

// Format a fake file path corresponding to an fd, for use in error messages.
fn fd_path(buf: &mut [u8; 27], fd: c_int) -> &CStr {
    // This is equivalent to `format!("<fd:{}>\0", fd)`, but without the dependency on std.

    use core::fmt::Write;

    // The array size is chosen so that any file descriptor value will fit. We need space for 6
    // characters, plus space for the largest decimal value for the `c_int` type. On some systems
    // the c_int type can actually be 64 bits. The `i64::MIN` value has 20 digits, and the minus
    // sign, for a total of 6 + 20 + 1 = 27.
    struct Writer<'a> {
        buf: &'a mut [u8; 27],
        len: usize,
    }

    impl Write for Writer<'_> {
        fn write_str(&mut self, s: &str) -> core::fmt::Result {
            let Some(dst) = self.buf.get_mut(self.len..self.len + s.len()) else {
                return Err(core::fmt::Error);
            };

            dst.copy_from_slice(s.as_bytes());
            self.len += s.len();

            Ok(())
        }
    }

    let mut w = Writer { buf, len: 0 };

    write!(w, "<fd:{fd}>\0").unwrap();

    // SAFETY: The stack formatter wrote a trailing NUL within the local byte buffer; the pointer remains valid for this borrowed C string.
    unsafe { CStr::from_ptr(w.buf[..w.len].as_ptr().cast()) }
}

// Reset the internal state of an open gzip stream according to
// its mode (read or write)
fn gz_reset(state: &mut GzState) {
    state.have = 0; // no output data available
    if state.mode == GzMode::Read {
        state.eof = false; // not at end of file
        state.past = false; // have not read past end yet
        state.how = How::Look; // look for gzip header
    } else {
        state.reset = false; // no deflateReset pending
    }
    state.seek = false; // no seek request pending
    gz_error(state, None); // clear error status
    state.pos = 0; // no uncompressed data yet
    if state.mode == GzMode::Read {
        state.clear_read_input();
    } else {
        state.stream.avail_in = 0; // no input data yet
    }
}

// Set the error message for a gzip stream, replacing any previously set message.
//
// # Arguments
//
// * `state` - An initialized stream state.
// * `err_msg` - `None` or `Some(err, msg)`. In the latter case, this function makes
//   an owned C-string copy, so `msg` need not remain in scope after the call.
fn gz_error(state: &mut GzState, err_msg: Option<(c_int, &str)>) {
    state.msg = None;
    state.read_would_block = false;

    match err_msg {
        None => {
            state.err = Z_OK;
        }
        Some((err, msg)) => {
            // On error, set state.have to 0 so that the `gzgetc()` C macro fails.
            if err != Z_OK && err != Z_BUF_ERROR {
                state.have = 0;
            }

            state.err = err;

            // For an out-of-memory error, don't try to allocate another error string.
            // `gzerror` supplies the literal OOM message instead.
            if err == Z_MEM_ERROR {
                return;
            }

            let sep = b": ";
            let mut fd_buf = [0u8; 27];
            state.msg = match state.source.as_ref() {
                Some(Source::Path(path)) => {
                    GzCString::concat(&[path.as_c_str().to_bytes(), sep, msg.as_bytes()])
                }
                Some(Source::Fd(fd)) => {
                    GzCString::concat(&[fd_path(&mut fd_buf, *fd).to_bytes(), sep, msg.as_bytes()])
                }
                None => GzCString::concat(&[msg.as_bytes()]),
            };

            if state.msg.is_none() {
                state.err = Z_MEM_ERROR;
            }
        }
    }
}

// Release any lazily allocated gzip I/O buffers while keeping the `GzState` alive.
fn free_buffers(state: &mut GzState) {
    state.input = None;
    state.output = None;
}

/// Close an open gzip file and free the internal data structures referenced by the file handle.
///
/// # Returns
///
/// * [`Z_ERRNO`] if closing the file failed
/// * [`Z_OK`] otherwise
///
/// # Safety
///
/// `file` must be one of the following:
/// - A file handle must have been obtained from a function in this library, such as [`gzopen`].
/// - A null pointer.
///
/// This function may be called at most once for any file handle.
///
/// `file` must not be used after this call returns, as the memory it references may have
/// been deallocated.
#[cfg_attr(feature = "export-symbols", export_name = prefix!(gzclose))]
pub unsafe extern "C" fn gzclose(file: gzFile) -> c_int {
    // SAFETY: The caller supplies either null or a live, correctly aligned gzip handle; as_ref returns None for null and only borrows it.
    let Some(state) = (unsafe { file.cast::<GzState>().as_ref() }) else {
        return Z_STREAM_ERROR;
    };

    match state.mode {
        // SAFETY: The live handle was inspected above and has read mode; gzclose_r consumes that read handle.
        GzMode::Read => unsafe { gzclose_r(file) },
        // SAFETY: The live handle was inspected above and is not in read mode; gzclose_w handles the remaining states and consumes it.
        GzMode::Write | GzMode::Append | GzMode::None => unsafe { gzclose_w(file) },
    }
}

/// Close a gzip file that was opened for reading.
///
/// # Returns
///
/// * Z_OK if `state` has no outstanding error and the file is closed successfully.
/// * A Z_ error code if the `state` is null or the file close operation fails.
///
/// # Safety
///
/// `file` must be one of the following:
/// - A file handle must have been obtained from a function in this library, such as [`gzopen`].
/// - A null pointer.
///
/// `file` must not be used after this call returns, as the memory it references may have
/// been deallocated.
#[cfg_attr(feature = "export-symbols", export_name = prefix!(gzclose_r))]
pub unsafe extern "C" fn gzclose_r(file: gzFile) -> c_int {
    let Some(state_ptr) = NonNull::new(file.cast::<GzState>()) else {
        return Z_STREAM_ERROR;
    };
    // Safety: the public contract requires a live gzip handle when `file` is non-null.
    let state = unsafe { state_ptr.as_ref() };

    // A mode mismatch must not consume the handle; the matching close operation may still use it.
    if state.mode != GzMode::Read {
        return Z_STREAM_ERROR;
    }

    // Safety: mode validation above establishes that this close operation consumes the live handle.
    let mut state_owner = unsafe { GzStateOwner::reclaim(state_ptr) };
    let state = state_owner.as_mut();

    // Process any buffered input.
    if state.input.is_some() {
        // Safety: state.stream was properly initialized as a z_stream during gzip open.
        unsafe { inflateEnd(&raw mut state.stream) };
    }

    let err = match state.err {
        Z_BUF_ERROR => Z_BUF_ERROR,
        _ => Z_OK,
    };

    // SAFETY: The state owns the integer file descriptor; close accepts any descriptor value and reports an invalid one with -1.
    let ret = match unsafe { libc::close(state.fd) } {
        0 => err,
        _ => Z_ERRNO,
    };

    // `state_owner` drops the complete state and returns its custom allocation here.
    ret
}

/// Close a gzip file that was opened for writing.
///
/// # Returns
///
/// * Z_OK if `state` has no outstanding error and the file is closed successfully.
/// * A Z_ error code if the `state` is null or the file close operation fails.
///
/// # Safety
///
/// `file` must be one of the following:
/// - A file handle must have been obtained from a function in this library, such as [`gzopen`].
/// - A null pointer.
///
/// `file` must not be used after this call returns, as the memory it references may have
/// been deallocated.
#[cfg_attr(feature = "export-symbols", export_name = prefix!(gzclose_w))]
pub unsafe extern "C" fn gzclose_w(file: gzFile) -> c_int {
    let mut ret = Z_OK;

    let Some(state_ptr) = NonNull::new(file.cast::<GzState>()) else {
        return Z_STREAM_ERROR;
    };
    // Safety: the public contract requires a live gzip handle when `file` is non-null.
    let state = unsafe { state_ptr.as_ref() };

    // A mode mismatch must not consume the handle; the matching close operation may still use it.
    if state.mode != GzMode::Write {
        return Z_STREAM_ERROR;
    }

    // Safety: mode validation above establishes that this close operation consumes the live handle.
    let mut state_owner = unsafe { GzStateOwner::reclaim(state_ptr) };
    let state = state_owner.as_mut();

    // Check for a pending seek request
    if state.seek {
        state.seek = false;
        if gz_zero_pending(state).is_err() {
            ret = state.err;
        }
    }

    // Compress (if not in direct mode) and output any data left in the input buffer.
    if gz_comp(state, Z_FINISH).is_err() {
        ret = state.err;
    }
    if state.input.is_some() && !state.direct {
        // Safety: state.stream was properly initialized as a z_stream during gzip open.
        unsafe { deflateEnd(&raw mut state.stream) };
    }
    // SAFETY: The state owns this descriptor until close; libc::close returns an error for invalid descriptors rather than dereferencing them.
    if unsafe { libc::close(state.fd) } == -1 {
        ret = Z_ERRNO;
    }

    // `state_owner` drops the complete state and returns its custom allocation here.
    ret
}

/// Configure the buffer capacity of an open gzip file.
///
/// Set the internal buffer size used by this library's functions for `file` to
/// `size`.  The default buffer size is 128 KB.  This function must be called
/// after [`gzopen`] or [`gzdopen`], but before any other calls that read or write
/// the file (including [`gzdirect`]).  The buffer memory allocation is always
/// deferred to the first read or write.  Three times `size` in buffer space is
/// allocated.
///
/// # Returns
///
/// * `0` on success.
/// * `-1` on failure.
///
/// # Arguments
///
/// * `file` - file handle.
/// * `size` - requested buffer size in bytes.
///
/// # Safety
///
/// `file` must be one of the following:
/// - A file handle must have been obtained from a function in this library, such as [`gzopen`].
/// - A null pointer.
#[cfg_attr(feature = "export-symbols", export_name = prefix!(gzbuffer))]
pub unsafe extern "C" fn gzbuffer(file: gzFile, size: c_uint) -> c_int {
    // SAFETY: The gzip handle contract gives exclusive access to a live, aligned state, or null; as_mut handles null.
    let Some(state) = (unsafe { file.cast::<GzState>().as_mut() }) else {
        return -1;
    };
    if state.mode != GzMode::Read && state.mode != GzMode::Write {
        return -1;
    }

    // Make sure we haven't already allocated memory.
    if state.input.is_some() {
        return -1;
    }

    // Check the doubled buffer against both usize arithmetic and the C
    // stream's uInt-sized avail_in/avail_out fields. On 64-bit machines,
    // checking usize alone would accept sizes that later truncate in z_stream.
    let Ok(size) = usize::try_from(size) else {
        return -1;
    };
    if GzState::doubled_buffer_capacity(size).is_none() {
        return -1;
    }

    // Use a minimum buffer size of 8 to work with flush semantics elsewhere in the implementation.
    state.want = Ord::max(size, 8);

    0
}

/// Retrieve the zlib error code and a human-readable string description of
/// the most recent error on a gzip file stream.
///
/// # Arguments
///
/// * `file` - A gzip file handle, or null
/// * `errnum` - A pointer to a C integer in which the zlib error code should be
///   written, or null if the caller does not need the numeric error code.
///
/// # Returns
///
/// * A pointer to a null-terminated C string describing the error, if `file` is non-null
///   and has an error
/// * A pointer to an empty (zero-length), null-terminated C string, if `file` is non-null
///   but has no error
/// * Null otherwise
///
/// # Safety
///
/// `file` must be one of the following:
/// - A file handle obtained from [`gzopen`] or [`gzdopen`].
/// - A null pointer.
///
/// If this function returns a non-null string, the caller must not modifiy or
/// deallocate the string.
///
/// If `errnum` is non-null, it must point to an address at which a [`c_int`] may be written.
#[cfg_attr(feature = "export-symbols", export_name = prefix!(gzerror))]
pub unsafe extern "C" fn gzerror(file: gzFile, errnum: *mut c_int) -> *const c_char {
    // Get internal structure and check integrity
    // SAFETY: The gzip handle contract gives a valid, aligned live state for non-null file; as_ref does not take ownership.
    let Some(state) = (unsafe { file.cast::<GzState>().as_ref() }) else {
        return ptr::null();
    };
    if state.mode != GzMode::Read && state.mode != GzMode::Write {
        return ptr::null();
    }

    // Return error information
    if !errnum.is_null() {
        // Safety:
        // * `errnum` is non-null
        // * The caller is responsible for ensuring that `errnum` points to writable
        //   memory with proper alignment.
        unsafe { *errnum = state.err };
    }
    if state.err == Z_MEM_ERROR {
        b"out of memory\0".as_ptr().cast::<c_char>()
    } else if let Some(msg) = state.msg.as_ref() {
        msg.as_ptr()
    } else {
        b"\0".as_ptr().cast::<c_char>()
    }
}

/// Clear the error and end-of-file state for `file`.
///
/// # Arguments
///
/// * `file` - A gzip file handle, or null
///
/// # Safety
///
/// `file`, if non-null, must be an open file handle obtained from [`gzopen`] or [`gzdopen`].
#[cfg_attr(feature = "export-symbols", export_name = prefix!(gzclearerr))]
pub unsafe extern "C" fn gzclearerr(file: gzFile) {
    // Get internal structure and check integrity
    // SAFETY: The gzip handle contract gives exclusive access to a valid state for non-null file; as_mut handles null.
    let Some(state) = (unsafe { file.cast::<GzState>().as_mut() }) else {
        return;
    };
    if state.mode != GzMode::Read && state.mode != GzMode::Write {
        return;
    }

    // Clear error and EOF
    if state.mode == GzMode::Read {
        state.eof = false;
        state.past = false;
    }

    gz_error(state, None);
}

/// Check whether a read operation has tried to read beyond the end of `file`.
///
/// # Returns
///
/// * 1 if the end-of-file indicator is set. Note that this indicator is set only
///   if a read tries to go past the end of the input. If the last read request
///   attempted to read exactly the number of bytes remaining in the file, the
///   end-of-file indicator will not be set.
/// * 0 the end-of-file indicator is not set or `file` is null
///
/// # Arguments
///
/// * `file` - A gzip file handle, or null
///
/// # Safety
///
/// `file`, if non-null, must be an open file handle obtained from [`gzopen`] or [`gzdopen`].
#[cfg_attr(feature = "export-symbols", export_name = prefix!(gzeof))]
pub unsafe extern "C" fn gzeof(file: gzFile) -> c_int {
    // Get internal structure and check integrity
    // SAFETY: The gzip handle contract gives a valid, aligned live state for non-null file; as_ref handles null.
    let Some(state) = (unsafe { file.cast::<GzState>().as_ref() }) else {
        return 0;
    };
    if state.mode != GzMode::Read {
        return 0;
    }

    // Per the semantics described in the function comments above, the return value
    // is based on state.past, rather than state.eof.
    state.past as _
}

/// Check whether `file` is in direct mode (reading or writing literal bytes without compression).
///
/// NOTE: If `gzdirect` is called immediately after [`gzopen`] or [`gzdopen`], it may allocate
/// buffers internally to read the file header and determine whether the content is a gzip file.
/// If [`gzbuffer`] is used, it should be called before `gzdirect`.
///
/// # Returns
///
/// 0 if `file` is null.
///
/// If `file` is being read,
/// * 1 if the contents are being read directly, without decompression.
/// * 0 if the contents are being decompressed when read.
///
/// If `file` is being written,
/// * 1 if transparent mode was requested upon open (with the `"wT"` mode flag for [`gzopen`]).
/// * 0 otherwise.
///
/// # Arguments
///
/// * `file` - A gzip file handle, or null
///
/// # Safety
///
/// `file`, if non-null, must be an open file handle obtained from [`gzopen`] or [`gzdopen`].
#[cfg_attr(feature = "export-symbols", export_name = prefix!(gzdirect))]
pub unsafe extern "C" fn gzdirect(file: gzFile) -> c_int {
    // SAFETY: The gzip handle contract provides a live and uniquely borrowed GzState for non-null file.
    let Some(state) = (unsafe { file.cast::<GzState>().as_mut() }) else {
        return 0;
    };

    // In write mode, the direct flag was set in `gzopen_bound`. In read mode, the
    // direct status is determined lazily on the first read operation. If the first
    // read hasn't happened yet, we can look at the header now to determine if the
    // file is in gzip format.
    if state.mode == GzMode::Read && state.read_would_block {
        let _ = state.prepare_read();
    }
    if state.mode == GzMode::Read && state.how == How::Look && state.have == 0 {
        let _ = gz_look(state);
    }

    state.direct as _
}

/// Read and decompress up to `len` uncompressed bytes from `file` into `buf`.  If
/// the input file is not in gzip format, `gzread` copies up to `len` bytes into
/// the buffer directly from the file.
///
/// After reaching the end of a gzip stream in the input, `gzread` will continue
/// to read, looking for another gzip stream.  Any number of gzip streams may be
/// concatenated in the input file, and will all be decompressed by `gzread()`.
/// If something other than a gzip stream is encountered after a gzip stream,
/// `gzread` ignores that remaining trailing garbage (and no error is returned).
///
/// `gzread` can be used to read a gzip file that is being concurrently written.
/// Upon reaching the end of the input, `gzread` will return with the available
/// data.  If the error code returned by [`gzerror`] is `Z_OK` or `Z_BUF_ERROR`,
/// then [`gzclearerr`] can be used to clear the end of file indicator in order
/// to permit `gzread` to be tried again.  `Z_OK` indicates that a gzip stream
/// was completed on the last `gzread`.  `Z_BUF_ERROR` indicates that the input
/// file ended in the middle of a gzip stream.  Note that `gzread` does not return
/// `-1` in the event of an incomplete gzip stream.  This error is deferred until
/// [`gzclose`], which will return `Z_BUF_ERROR` if the last gzread ended in the
/// middle of a gzip stream.  Alternatively, `gzerror` can be used before `gzclose`
/// to detect this case.
///
/// For nonblocking file descriptors, a read that makes no progress and
/// encounters `EAGAIN` or `EWOULDBLOCK` reports the I/O error for that
/// call. A subsequent read automatically retries when input becomes available,
/// without requiring `gzclearerr`. Other I/O failures remain sticky.
///
/// /// If the unsigned value `len` is too large to fit in the signed return type
/// `c_int`, then nothing is read, `-1` is returned, and the error state is set to
/// `Z_STREAM_ERROR`.
///
/// # Returns
///
/// * The number of uncompressed bytes read from the file into `buf`, which may
///   be smaller than `len` if there is insufficient data in the file.
/// * `-1` on error.
///
/// # Arguments
///
/// * `file` - A gzip file handle, or null.
/// * `buf` - Buffer where the read data should be stored. The caller retains ownership of this buffer.
/// * `len` - Number of bytes to attempt to read into `buf`.
///
/// # Safety
///
/// - `file`, if non-null, must be an open file handle obtained from [`gzopen`] or [`gzdopen`].
/// - The caller must ensure that `buf` points to at least `len` writable bytes.
#[cfg_attr(feature = "export-symbols", export_name = prefix!(gzread))]
pub unsafe extern "C" fn gzread(file: gzFile, buf: *mut c_void, len: c_uint) -> c_int {
    // SAFETY: The caller guarantees an exclusive live gzip handle; as_mut rejects null without dereferencing it.
    let Some(state) = (unsafe { file.cast::<GzState>().as_mut() }) else {
        return -1;
    };

    // Check that we're reading and that there's no (serious) error.
    if !state.prepare_read() {
        return -1;
    }

    // Check that the requested number of bytes can be represented by the return type.
    if c_int::try_from(len).is_err() {
        const MSG: &str = "request does not fit in an int";
        gz_error(state, Some((Z_STREAM_ERROR, MSG)));
        return -1;
    }

    // Bind the caller-owned output once at the FFI boundary. `MaybeUninit` accurately
    // represents that the incoming writable bytes need not have been initialized yet.
    let got = if len == 0 {
        gz_read(state, &mut [])
    } else {
        // Safety: the caller guarantees a writable span of `len` bytes when `len != 0`.
        let output =
            unsafe { core::slice::from_raw_parts_mut(buf.cast::<MaybeUninit<u8>>(), len as usize) };
        gz_read(state, output)
    };

    // Check for an error
    if got == 0 && state.err != Z_OK && state.err != Z_BUF_ERROR {
        -1
    } else {
        c_int::try_from(got).unwrap_or(-1)
    }
}

// A Rust slice or raw cursor must not span more than isize::MAX bytes,
// even when a size_t product itself is representable. Keep the check here,
// before either FFI entry point forms a caller-memory view.
fn checked_gz_transfer_len(state: &mut GzState, size: size_t, nitems: size_t) -> Option<usize> {
    let Some(len) = size.checked_mul(nitems) else {
        gz_error(
            state,
            Some((Z_STREAM_ERROR, "request does not fit in a size_t")),
        );
        return None;
    };
    if len > isize::MAX.unsigned_abs() {
        gz_error(
            state,
            Some((
                Z_STREAM_ERROR,
                "request exceeds the maximum addressable byte span",
            )),
        );
        return None;
    }
    Some(len)
}

/// Read and decompress up to `nitems` items of size `size` from `file` into `buf`,
/// otherwise operating as [`gzread`] does. This duplicates the interface of
/// C stdio's `fread()`, with `size_t` request and return types.
///
/// `gzfread` returns the number of full items read of size `size`, or zero if
/// the end of the file was reached and a full item could not be read, or if
/// there was an error.  [`gzerror`] must be consulted if zero is returned in
/// order to determine if there was an error.  If the multiplication of `size` and
/// `nitems` overflows, i.e. the product does not fit in a `size_t`, then nothing
/// is read, zero is returned, and the error state is set to `Z_STREAM_ERROR`.\n/// An otherwise representable product exceeding `isize::MAX` bytes is also rejected.
///
/// In the event that the end of file is reached and only a partial item is
/// available at the end, i.e. the remaining uncompressed data length is not a
/// multiple of `size`, then the final partial item is nevertheless read into `buf`
/// and the end-of-file flag is set.  The length of the partial item read is not
/// provided, but could be inferred from the result of [`gztell`].  This behavior
/// is the same as the behavior of `fread` implementations in common libraries,
/// but it prevents the direct use of `gzfread` to read a concurrently written
/// file, resetting and retrying on end-of-file, when `size` is not 1.
///
/// # Returns
///
/// - The number of complete object of size `size` read into `buf`.
/// - `0` on error or end-of-file.
///
/// # Safety
///
/// - `file`, if non-null, must be an open file handle obtained from [`gzopen`] or [`gzdopen`].
/// - The caller must ensure that `buf` points to at least `size * nitems` writable bytes.
#[cfg_attr(feature = "export-symbols", export_name = prefix!(gzfread))]
pub unsafe extern "C" fn gzfread(
    buf: *mut c_void,
    size: size_t,
    nitems: size_t,
    file: gzFile,
) -> size_t {
    if size == 0 || buf.is_null() {
        return 0;
    }

    // SAFETY: The enclosing gzip entry point receives an exclusive live handle or null, so as_mut is valid for non-null file.
    let Some(state) = (unsafe { file.cast::<GzState>().as_mut() }) else {
        return 0;
    };

    // Check that we're reading and that there's no (serious) error.
    if !state.prepare_read() {
        return 0;
    }

    // Validate before constructing a Rust caller-output slice.
    let Some(len) = checked_gz_transfer_len(state, size, nitems) else {
        return 0;
    };

    if len == 0 {
        len
    } else {
        // Safety: the caller guarantees a writable span of `len = size * nitems` bytes.
        let output = unsafe { core::slice::from_raw_parts_mut(buf.cast::<MaybeUninit<u8>>(), len) };
        gz_read(state, output) / size
    }
}

// Attempt to read enough bytes from the input to fill the supplied `output`.
//
// The destination is a typed uninitialized span bound once at the public FFI boundary.
// Internal reads never construct or advance a raw caller-output pointer.
fn gz_read(state: &mut GzState, output: &mut [MaybeUninit<u8>]) -> usize {
    if output.is_empty() {
        return 0;
    }

    // Process a skip request.
    if state.seek {
        state.seek = false;
        if gz_skip(state, state.skip).is_err() {
            return 0;
        }
    }

    let mut written = 0usize;
    while written < output.len() {
        if state.have == 0 && state.eof && state.stream.avail_in == 0 {
            state.past = true;
            break;
        }
        let remaining = output.len() - written;
        // Even a valid writable caller buffer cannot advance a signed C
        // stream position beyond i64::MAX. Return bytes already copied.
        // The fatal error clears the exposed have count so the C gzgetc macro
        // cannot advance the position without consulting the error state.
        let room = state.position_room();
        if room == 0 {
            state.fail_position_overflow();
            return written;
        }
        let mut n = remaining.min(c_uint::MAX as usize).min(room);

        if state.have != 0 {
            let Some(available) = state.buffered_output() else {
                state.invalid_read_cursor();
                return written;
            };
            n = Ord::min(n, available.len());
            output[written..written + n].write_copy_of_slice(&available[..n]);
            if !state.consume_buffered_output(n) {
                state.invalid_read_cursor();
                return written;
            }
        } else if state.how == How::Look || n < state.in_size() * 2 {
            if gz_fetch(state).is_err() {
                // The caller already owns any bytes copied in earlier passes.
                // A later refill failure must not discard that initialized prefix.
                return written;
            }
            continue;
        } else if state.how == How::Copy {
            let target = &mut output[written..written + n];
            let (bytes_read, eof) = match load_from_fd(state.fd, target) {
                Ok(result) => result,
                Err(error) => {
                    gz_read_io_error(state, error);
                    return written;
                }
            };
            if eof {
                state.eof = true;
            }
            n = bytes_read;
        } else {
            debug_assert_eq!(state.how, How::Gzip);
            n = match gz_decomp_into(state, &mut output[written..written + n]) {
                Ok(produced) => produced,
                Err(()) => return written,
            };
        }

        if state.advance_position(n).is_err() {
            return written;
        }
        written += n;
    }

    written
}

// Skip len uncompressed bytes of output.
//
// # Returns
//
// - `Ok` on success.
// - `Err` on error.
fn gz_skip(state: &mut GzState, len: i64) -> Result<(), ()> {
    // Public seek operations schedule only forward skips. Reject invalid internal
    // requests before inspecting or consuming output or touching the file.
    let Ok(mut remaining) = u64::try_from(len) else {
        gz_error(state, Some((Z_STREAM_ERROR, "negative gzip skip distance")));
        return Err(());
    };
    // Skip over the requested bytes or reach EOF, whichever comes first.
    while remaining != 0 {
        if state.have != 0 {
            // A C buffered byte count and a signed 64-bit seek are compared in
            // one unsigned width, avoiding len-to-usize truncation on 32-bit hosts.
            let amount = u64::from(state.have).min(remaining);
            let Ok(n) = usize::try_from(amount) else {
                state.invalid_read_cursor();
                return Err(());
            };
            // amount <= the original nonnegative i64 skip, so it fits i64.
            let Ok(take) = i64::try_from(amount) else {
                state.fail_position_overflow();
                return Err(());
            };
            let Some(next) = state.pos.checked_add(take) else {
                state.fail_position_overflow();
                return Err(());
            };
            if !state.consume_buffered_output(n) {
                state.invalid_read_cursor();
                return Err(());
            }
            state.pos = next;
            remaining -= amount;
        } else if state.eof && state.stream.avail_in == 0 {
            // Output buffer empty -- return if we're at the end of the input.
            break;
        } else {
            // Need more data to skip -- load up output buffer.
            // Get more output, looking for header if required.
            if gz_fetch(state).is_err() {
                return Err(());
            }
        }
    }
    Ok(())
}

// Given a gzip file opened for reading, check for a gzip header, and set
// `state.direct` accordingly.
//
// # Returns
//
// * `Ok` on success
// * `Err` on failure
//
// Lazily allocate the owned gzip read input/output and initialize inflate.
// No input is consumed: header detection remains the responsibility of gz_look.
fn gz_init_read(state: &mut GzState) -> Result<(), ()> {
    // Allocate buffers if needed.
    if state.input.is_none() {
        let Some(input) = state.allocate_input_buffer() else {
            gz_error(state, Some((Z_MEM_ERROR, "out of memory")));
            return Err(());
        };
        state.input = Some(input);

        if state.output.is_none() {
            let Some(output) = state.allocate_output_buffer() else {
                free_buffers(state);
                gz_error(state, Some((Z_MEM_ERROR, "out of memory")));
                return Err(());
            };
            state.output = Some(output);
        }

        // Allocate memory for inflate.
        state.clear_read_input();
        // Safety: `gzopen_bound` initialized `state.stream`'s `zalloc`, `zfree`, and
        // `opaque` fields as needed by `inflateInit2`.
        if unsafe {
            inflateInit2_(
                &raw mut state.stream,
                MAX_WBITS + 16,
                zlibVersion(),
                core::mem::size_of::<z_stream>() as i32,
            )
        } != Z_OK
        {
            free_buffers(state);
            gz_error(state, Some((Z_MEM_ERROR, "out of memory")));
            return Err(());
        }
    }

    Ok(())
}

fn gz_look(state: &mut GzState) -> Result<(), ()> {
    // Initialize storage independently of header detection. The caller
    // asking for a pushback byte must not cause a read from its descriptor.
    gz_init_read(state)?;

    // Get at least the magic bytes in the input buffer.
    if state.stream.avail_in < 2 {
        // `gz_avail` attempts to read as much data as available from the underlying file
        // into the input buffer. This will hopefully give us enough bytes to check for a
        // gzip file header.
        if gz_avail(state)? == 0 {
            return Ok(());
        }
    }

    // Look for gzip magic bytes.
    // Note: If we are reading a partially written gzip file, and all that is available to read is
    // the first byte of the gzip magic number, we cannot tell whether what follows will be the
    // rest of the gzip magic. For simplicity, we assume that the writer of a gzip file will
    // write the header (or at least the magic number at the start of the header) atomically,
    // so if our initial read found a single byte it is a sufficient indication that the file
    // is not in gzip format.
    let Some(input) = state.buffered_read_input() else {
        state.invalid_read_cursor();
        return Err(());
    };
    if input.len() > 1 && input[0] == 31 && input[1] == 139 {
        // Safety: We initialized `state.stream` with `inflateInit2` above.
        unsafe { inflateReset(&raw mut state.stream) };
        state.how = How::Gzip;
        state.direct = false;
        return Ok(());
    }

    // No gzip header. If we were decoding gzip before, the remaining bytes
    // are trailing garbage that can be ignored.
    if !state.direct {
        state.clear_read_input();
        state.eof = true;
        state.have = 0;
        return Ok(());
    }

    // The file is not in gzip format, so enable direct mode and move the buffered
    // owned input into owned output storage.
    let Some(mut output) = state.output.take() else {
        state.invalid_read_cursor();
        return Err(());
    };
    let produced = {
        let Some(input) = state.buffered_read_input() else {
            state.output = Some(output);
            state.invalid_read_cursor();
            return Err(());
        };
        output.usable_uninit_mut()[..input.len()].write_copy_of_slice(input);
        input.len()
    };
    state.output = Some(output);
    state.publish_buffered_output(0, produced);
    state.clear_read_input();
    state.how = How::Copy;
    state.direct = true;

    Ok(())
}

// Load data into the input buffer and set the eof flag if the last of the data has been
// loaded.
//
// # Returns
//
// * `Ok(n)` on success, where `n` is the number of bytes available (`state.stream.avail_in`)
// * `Err` on error
//
fn gz_avail(state: &mut GzState) -> Result<usize, ()> {
    if state.err != Z_OK && state.err != Z_BUF_ERROR {
        return Err(());
    }
    if !state.eof {
        let Some(already_buffered) = state.compact_read_input() else {
            state.invalid_read_cursor();
            return Err(());
        };
        let load_result = {
            let Some(input) = state.input.as_mut() else {
                state.invalid_read_cursor();
                return Err(());
            };
            let storage = input.usable_uninit_mut();
            load_from_fd(state.fd, &mut storage[already_buffered..])
        };
        let (got, eof) = match load_result {
            Ok(result) => result,
            Err(error) => {
                gz_read_io_error(state, error);
                return Err(());
            }
        };
        if eof {
            state.eof = true;
        }
        if !state.publish_read_input(0, already_buffered + got) {
            state.invalid_read_cursor();
            return Err(());
        }
    }
    Ok(state.stream.avail_in as usize)
}

// Read into an uninitialized byte span from an underlying file descriptor.
// Returns `(bytes_read, reached_eof)`. A failed system read is reported separately so
// the gzip state can attach its source-specific error message at the caller.
// Consume the whole byte span even when the OS performs short writes.
// Both direct gzip passthrough and compressed output use this same loop.
// Keep the syscall injectable so partial and zero-progress writes are testable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IoAttemptError {
    Interrupted,
    WouldBlock,
    Failed,
}

// errno is thread-local and must be inspected immediately after a failed syscall.
// The accessor differs by C runtime. Unsupported targets still report genuine
// I/O failure; the platforms below also retry EINTR.
fn last_io_attempt_error() -> IoAttemptError {
    #[cfg(any(
        target_os = "linux",
        target_os = "dragonfly",
        target_os = "fuchsia",
        target_os = "emscripten",
        target_os = "hurd",
        target_os = "redox",
        target_os = "wasi"
    ))]
    // SAFETY: the C runtime supplies a live pointer to this thread's errno.
    let code = unsafe { *libc::__errno_location() };

    #[cfg(any(target_os = "android", target_os = "netbsd", target_os = "openbsd"))]
    // SAFETY: the C runtime supplies a live pointer to this thread's errno.
    let code = unsafe { *libc::__errno() };

    #[cfg(any(target_vendor = "apple", target_os = "freebsd"))]
    // SAFETY: the C runtime supplies a live pointer to this thread's errno.
    let code = unsafe { *libc::__error() };

    #[cfg(windows)]
    let code = {
        unsafe extern "C" {
            fn _errno() -> *mut c_int;
        }
        // SAFETY: the CRT supplies the current thread's errno storage.
        unsafe { *_errno() }
    };

    #[cfg(not(any(
        target_os = "linux",
        target_os = "dragonfly",
        target_os = "fuchsia",
        target_os = "emscripten",
        target_os = "hurd",
        target_os = "redox",
        target_os = "wasi",
        target_os = "android",
        target_os = "netbsd",
        target_os = "openbsd",
        target_vendor = "apple",
        target_os = "freebsd",
        windows
    )))]
    let code = 0;

    if code == libc::EINTR {
        IoAttemptError::Interrupted
    } else if code == libc::EAGAIN || code == libc::EWOULDBLOCK {
        IoAttemptError::WouldBlock
    } else {
        IoAttemptError::Failed
    }
}

fn gz_read_io_error(state: &mut GzState, error: IoAttemptError) {
    gz_error(state, Some((Z_ERRNO, "read error")));
    // If recording the error itself ran out of memory, do not later clear
    // the resulting Z_MEM_ERROR as though it were a transient I/O stall.
    state.read_would_block = error == IoAttemptError::WouldBlock && state.err == Z_ERRNO;
}

fn write_all_with(
    mut remaining: &[u8],
    mut write_some: impl FnMut(&[u8]) -> Result<usize, IoAttemptError>,
) -> Result<(), ()> {
    while !remaining.is_empty() {
        match write_some(remaining) {
            Err(IoAttemptError::Interrupted) => {}
            Err(IoAttemptError::Failed | IoAttemptError::WouldBlock) | Ok(0) => {
                return Err(());
            }
            Ok(written) if written > remaining.len() => return Err(()),
            Ok(written) => remaining = &remaining[written..],
        }
    }
    Ok(())
}

fn write_all_to_fd(fd: c_int, input: &[u8]) -> Result<(), ()> {
    write_all_with(input, |remaining| {
        // On some platforms write(2) cannot accept lengths exceeding SSIZE_MAX.
        let len = remaining.len().min(isize::MAX as usize);
        // SAFETY: the initialized input slice stays live across this syscall
        // and the request cannot exceed its size.
        let written = unsafe { libc::write(fd, remaining.as_ptr().cast::<c_void>(), len) };
        usize::try_from(written).map_err(|_| last_io_attempt_error())
    })
}

fn read_with(
    output: &mut [MaybeUninit<u8>],
    mut read_some: impl FnMut(&mut [MaybeUninit<u8>]) -> Result<usize, IoAttemptError>,
) -> Result<(usize, bool), IoAttemptError> {
    let mut have = 0usize;
    while have < output.len() {
        let remaining = &mut output[have..];
        match read_some(remaining) {
            Err(IoAttemptError::Interrupted) => {}
            // A short successful read already consumed these bytes from the
            // underlying file. Publish that initialized prefix now; only a
            // subsequent operation with no progress reports the I/O error.
            Err(IoAttemptError::Failed | IoAttemptError::WouldBlock) if have != 0 => {
                return Ok((have, false));
            }
            Err(error) => return Err(error),
            Ok(0) => return Ok((have, true)),
            Ok(n) if n > remaining.len() => return Err(IoAttemptError::Failed),
            Ok(n) => have += n,
        }
    }
    Ok((have, false))
}

fn load_from_fd(
    fd: c_int,
    output: &mut [MaybeUninit<u8>],
) -> Result<(usize, bool), IoAttemptError> {
    read_with(output, |remaining| {
        let len = remaining.len().min(isize::MAX as usize);
        // SAFETY: the output is writable uninitialized storage for len
        // bytes, and stays live across this syscall.
        let got = unsafe { libc::read(fd, remaining.as_mut_ptr().cast::<c_void>(), len) };
        usize::try_from(got).map_err(|_| last_io_attempt_error())
    })
}

// Fetch data and put it in the output buffer, decompressing if needed. If the header has
// not been read yet, parse it to determine whether the file is in gzip format.
//
// # Returns
//
// * `Ok` on success.
// * `Err` on error. Check [`gzerror`] for more information on the error condition.
//
// `state.have` must be zero on entry.
fn gz_fetch(state: &mut GzState) -> Result<(), ()> {
    debug_assert_eq!(state.have, 0);
    loop {
        match &state.how {
            How::Look => {
                gz_look(state)?;
                if state.how == How::Look {
                    return Ok(());
                }
            }
            How::Copy => {
                let load_result = {
                    let output = state
                        .output
                        .as_mut()
                        .expect("gzip read output must be allocated in copy mode");
                    load_from_fd(state.fd, output.usable_uninit_mut())
                };
                let (bytes_read, eof) = match load_result {
                    Ok(result) => result,
                    Err(error) => {
                        gz_read_io_error(state, error);
                        return Err(());
                    }
                };
                if eof {
                    state.eof = true;
                }
                state.publish_buffered_output(0, bytes_read);
                return Ok(());
            }
            How::Gzip => {
                // Temporarily detach the output owner so the decompressor can borrow the
                // rest of `state` without aliasing storage still reachable through it.
                let mut output = state
                    .output
                    .take()
                    .expect("gzip read output must be allocated in gzip mode");
                let result = gz_decomp_into(state, output.usable_uninit_mut());
                state.output = Some(output);
                let produced = result?;
                state.publish_buffered_output(0, produced);
            }
        }

        if state.have != 0 || (state.eof && state.stream.avail_in == 0) {
            break;
        }
    }

    Ok(())
}

// Decompress into an explicit uninitialized output span. The caller decides whether
// that span is the owned gzip output buffer or the caller's large-read destination.
fn gz_decomp_into(state: &mut GzState, output: &mut [MaybeUninit<u8>]) -> Result<usize, ()> {
    let bound_avail_out = output.len() as c_uint;
    state.stream.avail_out = bound_avail_out;
    state.stream.next_out = output.as_mut_ptr().cast::<u8>();

    let result = (|| -> Result<(), ()> {
        loop {
            if state.stream.avail_in == 0 && gz_avail(state).is_err() {
                return Err(());
            }
            if state.stream.avail_in == 0 {
                gz_error(state, Some((Z_BUF_ERROR, "unexpected end of file")));
                break;
            }

            // SAFETY: The gzip stream was initialized for decompression and its input/output pointers refer to the live owned buffers during this call.
            match unsafe { inflate(&raw mut state.stream, Z_NO_FLUSH) } {
                Z_STREAM_ERROR | Z_NEED_DICT => {
                    const MSG: &str = "internal error: inflate stream corrupt";
                    gz_error(state, Some((Z_STREAM_ERROR, MSG)));
                    return Err(());
                }
                Z_MEM_ERROR => {
                    gz_error(state, Some((Z_MEM_ERROR, "out of memory")));
                    return Err(());
                }
                Z_DATA_ERROR => {
                    gz_error(state, Some((Z_DATA_ERROR, "compressed data error")));
                    return Err(());
                }
                Z_STREAM_END => {
                    state.how = How::Look;
                    break;
                }
                _ => {}
            }

            if state.stream.avail_out == 0 {
                break;
            }
        }
        Ok(())
    })();

    let produced = (bound_avail_out - state.stream.avail_out) as usize;
    // `next_out` is a borrowed pointer into either caller memory or detached owned output.
    // Never let that borrow escape this operation through persistent z_stream state.
    state.stream.next_out = ptr::null_mut();
    state.stream.avail_out = 0;

    // A later I/O or inflate error cannot revoke bytes that the inflater
    // already initialized in the destination. Return that valid prefix now,
    // leaving the error status in GzState for the next public operation.
    if produced != 0 {
        Ok(produced)
    } else {
        result.map(|()| produced)
    }
}

/// Compress and write the len uncompressed bytes at buf to file.
///
/// # Returns
///
/// - The number of uncompressed bytes written, on success.
/// - Or 0 in case of error.
///
/// # Safety
///
/// - `file`, if non-null, must be an open file handle obtained from [`gzopen`] or [`gzdopen`].
/// - `buf` must point to at least `len` bytes of readable memory.
#[cfg_attr(feature = "export-symbols", export_name = prefix!(gzwrite))]
pub unsafe extern "C" fn gzwrite(file: gzFile, buf: *const c_void, len: c_uint) -> c_int {
    // SAFETY: The writer receives an exclusive valid gzip handle or null, and as_mut only dereferences non-null handles.
    let Some(state) = (unsafe { file.cast::<GzState>().as_mut() }) else {
        return 0;
    };

    // Check that we're writing and that there's no error.
    if state.mode != GzMode::Write || state.err != Z_OK {
        return 0;
    }

    // Check that the requested number of bytes can be represented by the return type.
    if c_int::try_from(len).is_err() {
        const MSG: &str = "requested length does not fit in int";
        gz_error(state, Some((Z_DATA_ERROR, MSG)));
        return 0;
    }

    // Also check that the requested number of bytes can be represented by usize,
    // which gz_write uses internally.
    let Ok(len) = usize::try_from(len) else {
        const MSG: &str = "requested length does not fit in usize";
        gz_error(state, Some((Z_DATA_ERROR, MSG)));
        return 0;
    };

    // Safety: the public contract guarantees a readable caller span of `len` bytes.
    let mut input = unsafe { GzWriteInput::from_raw(buf, len) };
    c_int::try_from(gz_write(state, &mut input)).unwrap_or(0)
}

/// Compress and write `nitems` items of size `size` from `buf` to `file`, duplicating
/// the interface of C stdio's `fwrite`, with `size_t` request and return types.
///
/// # Returns
///
/// - The number of full items written of size `size` on success.
/// - Zero on error.
///
/// Note: If the multiplication of `size` and `nitems` overflows, i.e. the product does
/// not fit in a `size_t`, then nothing is written, zero is returned, and the error state
/// is set to `Z_STREAM_ERROR`. The same applies to products exceeding\n/// `isize::MAX` bytes.
///
/// # Safety
///
/// - `file`, if non-null, must be an open file handle obtained from [`gzopen`] or [`gzdopen`].
/// - The caller must ensure that `buf` points to at least `size * nitems` readable bytes.
#[cfg_attr(feature = "export-symbols", export_name = prefix!(gzfwrite))]
pub unsafe extern "C" fn gzfwrite(
    buf: *const c_void,
    size: size_t,
    nitems: size_t,
    file: gzFile,
) -> size_t {
    if size == 0 || buf.is_null() {
        return 0;
    }

    // SAFETY: The enclosing gzip entry point provides an exclusive live handle or null, checked by as_mut.
    let Some(state) = (unsafe { file.cast::<GzState>().as_mut() }) else {
        return 0;
    };

    // Check that we're writing and that there's no error.
    if state.mode != GzMode::Write || state.err != Z_OK {
        return 0;
    }

    // Validate before binding the caller-owned raw input cursor.
    let Some(len) = checked_gz_transfer_len(state, size, nitems) else {
        return 0;
    };

    if len == 0 {
        len
    } else {
        // Safety: the public contract guarantees a readable caller span of
        // `len = size * nitems` bytes.
        let mut input = unsafe { GzWriteInput::from_raw(buf, len) };
        gz_write(state, &mut input) / size
    }
}

// Internal implementation shared by `gzwrite` and `gzfwrite` after their raw
// caller input has been bound to a checked cursor at the FFI boundary.
fn gz_write(state: &mut GzState, input: &mut GzWriteInput) -> usize {
    if input.is_empty() {
        return 0;
    }

    // Include a pending lazy seek: writing must not exceed the signed C
    // offset range even after a successful virtual seek near i64::MAX.
    if state.ensure_position_room(input.len()).is_err() {
        return 0;
    }

    if state.input.is_none() && gz_init(state).is_err() {
        return 0;
    }

    if state.seek {
        state.seek = false;
        if gz_zero_pending(state).is_err() {
            return 0;
        }
    }

    let put = input.len();

    if input.len() < state.in_size() {
        loop {
            if state.stream.avail_in == 0 {
                state.stream.next_in = state.input_ptr();
            }
            let Some(have) = state.owned_write_input_end() else {
                state.invalid_write_cursor();
                return 0;
            };
            let capacity = state.in_size().saturating_sub(have);
            let copy = {
                let owned = state
                    .input
                    .as_mut()
                    .expect("write input must remain allocated")
                    .usable_uninit_mut();
                input.copy_to(&mut owned[have..have + Ord::min(capacity, input.len())])
            };
            if state.advance_position(copy).is_err() {
                return 0;
            }
            state.stream.avail_in += copy as c_uint;
            if !input.is_empty() && gz_comp(state, Z_NO_FLUSH).is_err() {
                return 0;
            }
            if input.is_empty() {
                break;
            }
        }
    } else {
        if state.stream.avail_in != 0 && gz_comp(state, Z_NO_FLUSH).is_err() {
            return 0;
        }

        let save_next_in = state.stream.next_in;
        let result = loop {
            let n = Ord::min(input.len(), c_uint::MAX as usize);
            state.stream.next_in = input.as_ptr();
            state.stream.avail_in = n as c_uint;
            if state.advance_position(n).is_err() {
                break Err(());
            }
            if gz_comp(state, Z_NO_FLUSH).is_err() {
                break Err(());
            }
            debug_assert_eq!(state.stream.avail_in, 0);
            input.advance(n);
            if input.is_empty() {
                break Ok(());
            }
        };
        // `next_in` borrowed caller memory above. Restore owned-state identity on
        // every exit, including I/O and compression failures.
        state.stream.next_in = save_next_in;
        if result.is_err() {
            state.stream.avail_in = 0;
            return 0;
        }
    }

    put
}

// Compress `len` null bytes to output.
//
// # Returns
//
// - `Ok` on success.
// - `Err` on error.
// Pending write seeks are signed 64-bit positions, even on 32-bit targets.
// Consume them in representable chunks rather than truncating an i64 offset to usize.
fn gz_zero_pending(state: &mut GzState) -> Result<(), ()> {
    let Ok(mut remaining) = u64::try_from(state.skip) else {
        state.fail_position_overflow();
        return Err(());
    };
    if remaining == 0 {
        return gz_zero(state, 0);
    }

    while remaining != 0 {
        let count = usize::try_from(remaining).unwrap_or(usize::MAX);
        gz_zero(state, count)?;
        let Ok(consumed) = u64::try_from(count) else {
            state.fail_position_overflow();
            return Err(());
        };
        remaining -= consumed;
    }
    Ok(())
}

fn gz_zero(state: &mut GzState, mut len: usize) -> Result<(), ()> {
    state.ensure_position_room(len)?;

    // A pending write seek may be resolved by gzflush, gzsetparams, gzputc
    // or gzclose before the first write. Every nonempty zero fill needs the
    // owned compression input buffer, regardless of which caller got here.
    if len != 0 && state.input.is_none() {
        gz_init(state)?;
    }

    // Consume whatever is left in the input buffer.
    if state.stream.avail_in != 0 && gz_comp(state, Z_NO_FLUSH).is_err() {
        return Err(());
    }

    // Compress `len` zeros.
    let mut first = true;
    while len != 0 {
        let n = Ord::min(state.in_size(), len);
        if first {
            // gz_zero initialized the owned input buffer before entering this
            // loop, and n is bounded by its usable compression prefix.
            // SAFETY: gz_zero initialized the owned writable input above, and
            // n is bounded by its usable prefix, so the fill stays inside it.
            unsafe { state.input_ptr().write_bytes(0u8, n) };
            first = false;
        }
        state.stream.avail_in = n as _;
        state.stream.next_in = state.input_ptr();
        state.advance_position(n)?;
        if gz_comp(state, Z_NO_FLUSH).is_err() {
            return Err(());
        }
        len -= n;
    }

    Ok(())
}

// Initialize the lazily owned buffers and zlib stream for writing a gzip file.
//
// # Returns
//
// - `Ok` on success.
// - `Err` on error.
fn gz_init(state: &mut GzState) -> Result<(), ()> {
    // Allocate input buffer.
    // The buffer is twice as big as state.want, while `in_size()` exposes only half
    // of it to ordinary compression. The reason for this is to
    // ensure that we always have state.want bytes available for exclusive use
    // by gzprintf.
    let Some(input) = state.allocate_input_buffer() else {
        gz_error(state, Some((Z_MEM_ERROR, "out of memory")));
        return Err(());
    };
    state.input = Some(input);
    // Note: zlib-ng fills the input buffer with zeroes here, but it's unneeded.

    // Only need output buffer and deflate state if compressing.
    if !state.direct {
        // Allocate output buffer.
        let Some(output) = state.allocate_output_buffer() else {
            free_buffers(state);
            gz_error(state, Some((Z_MEM_ERROR, "out of memory")));
            return Err(());
        };
        state.output = Some(output);

        // Allocate deflate memory, set up for gzip compression.
        state.stream.zalloc = Some(ALLOCATOR.zalloc);
        state.stream.zfree = Some(ALLOCATOR.zfree);
        state.stream.opaque = ALLOCATOR.opaque;
        const DEF_MEM_LEVEL: c_int = 8;
        // SAFETY: The embedded z_stream has its allocator and opaque state set, and the initialization call receives a valid writable stream structure.
        if unsafe {
            deflateInit2_(
                &raw mut state.stream,
                state.level as _,
                Z_DEFLATED,
                MAX_WBITS + 16,
                DEF_MEM_LEVEL,
                state.strategy as _,
                zlibVersion(),
                core::mem::size_of::<z_stream>() as _,
            )
        } != Z_OK
        {
            free_buffers(state);
            gz_error(state, Some((Z_MEM_ERROR, "out of memory")));
            return Err(());
        }
        state.stream.next_in = ptr::null_mut();
    }

    // Buffer presence itself is the initialization state; no separate size sentinel is needed.

    // Initialize write buffer if compressing.
    if !state.direct {
        state.stream.avail_out = state.out_avail();
        state.stream.next_out = state.output_ptr();
        state.next = state.stream.next_out;
    }

    Ok(())
}

// Compress whatever is at avail_in and next_in (unless in direct mode) and write
// to the output file.
//
// # Returns
//
// - `Ok` on success.
// - `Err` on error.
fn gz_comp(state: &mut GzState, flush: c_int) -> Result<(), ()> {
    // Allocate memory if this is the first time through
    if state.input.is_none() && gz_init(state).is_err() {
        return Err(());
    }

    // Write directly if requested.
    if state.direct {
        let consumed = state.stream.avail_in as usize;
        if consumed != 0 {
            // SAFETY: the current gzip write input cursor is bound to
            // consumed initialized bytes, either owned or caller-borrowed
            // for this operation. No slice is formed for NULL + zero length.
            let input = unsafe { core::slice::from_raw_parts(state.stream.next_in, consumed) };
            if write_all_to_fd(state.fd, input).is_err() {
                gz_error(state, Some((Z_ERRNO, "write error")));
                return Err(());
            }
        }
        if consumed != 0 {
            // SAFETY: a nonempty direct-write input span was validated above.
            state.stream.next_in = unsafe { state.stream.next_in.add(consumed) };
        }
        state.stream.avail_in = 0;
        return Ok(());
    }

    // Check for a pending reset.
    if state.reset {
        // Don't start a new gzip stream unless there is data to write.
        if state.stream.avail_in == 0 {
            return Ok(());
        }
        // Safety: `state.reset` is set only in `gz_comp`, which first initializes
        // `state.stream` using `deflateInit2_`.
        let _ = unsafe { deflateReset(&raw mut state.stream) };
        state.reset = false;
    }

    // Run deflate on the provided input until it produces no more output.
    let mut ret = Z_OK;
    loop {
        // Do not pass malformed owned output cursors to the deflater.
        if state.pending_write_output().is_none() {
            state.invalid_write_cursor();
            return Err(());
        }
        // Write out current buffer contents if full, or if flushing, but if
        // doing Z_FINISH then don't write until we get to Z_STREAM_END.
        if state.stream.avail_out == 0
            || (flush != Z_NO_FLUSH && (flush != Z_FINISH || ret == Z_STREAM_END))
        {
            let Some(bytes) = state.pending_write_output() else {
                state.invalid_write_cursor();
                return Err(());
            };
            if !bytes.is_empty() && write_all_to_fd(state.fd, bytes).is_err() {
                gz_error(state, Some((Z_ERRNO, "write error")));
                return Err(());
            }
            if state.stream.avail_out == 0 {
                state.stream.avail_out = state.out_avail();
                state.stream.next_out = state.output_ptr();
            }
            state.next = state.stream.next_out;
        }

        // Compress.
        let mut have = state.stream.avail_out;
        // SAFETY: The deflate stream is initialized; the input and output cursors point within owned live gzip buffers at this call.
        ret = unsafe { deflate(&raw mut state.stream, flush) };
        if ret == Z_STREAM_ERROR {
            const MSG: &str = "internal error: deflate stream corrupt";
            gz_error(state, Some((Z_STREAM_ERROR, MSG)));
            return Err(());
        }
        have -= state.stream.avail_out;

        if have == 0 {
            break;
        }
    }

    // If that completed a deflate stream, allow another to start.
    if flush == Z_FINISH {
        state.reset = true;
    }

    Ok(())
}

/// Flush pending compressed output from a gzip file.
///
/// Flush all pending output buffered in `file`. The parameter `flush` is interpreted
/// the same way as in the [`deflate`] function. The return value is the zlib error
/// number (see [`gzerror`]). `gzflush` is permitted only when writing.
///
/// # Returns
///
/// - `Z_OK` on success.
/// - a `Z_` error code on error.
///
/// # Safety
///
/// - `file`, if non-null, must be an open file handle obtained from [`gzopen`] or [`gzdopen`].
#[cfg_attr(feature = "export-symbols", export_name = prefix!(gzflush))]
pub unsafe extern "C" fn gzflush(file: gzFile, flush: c_int) -> c_int {
    // SAFETY: The flush call receives an exclusive live gzip state when file is non-null; as_mut checks null.
    let Some(state) = (unsafe { file.cast::<GzState>().as_mut() }) else {
        return Z_STREAM_ERROR;
    };

    // Check that we're writing and that there's no error.
    if state.mode != GzMode::Write || state.err != Z_OK {
        return Z_STREAM_ERROR;
    }

    // Check flush parameter.
    if !(0..=Z_FINISH).contains(&flush) {
        return Z_STREAM_ERROR;
    }

    // Check for seek request.
    if state.seek {
        state.seek = false;
        if gz_zero_pending(state).is_err() {
            return state.err;
        }
    }

    // Compress remaining data with requested flush.
    let _ = gz_comp(state, flush);
    state.err
}

/// Get the uncompressed stream position for the next operation.
///
/// Return the starting position for the next [`gzread`] or [`gzwrite`] on `file`.
/// This position represents a number of bytes in the uncompressed data stream,
/// and is zero when starting, even if appending or reading a gzip stream from
/// the middle of a file using [`gzdopen`].
///
/// # Returns
///
/// * The number of bytes prior to the current read or write position in the
///   uncompressed data stream, on success.
/// * -1 on error.
///
/// # Safety
///
/// - `file`, if non-null, must be an open file handle obtained from [`gzopen`] or [`gzdopen`].
#[cfg_attr(feature = "export-symbols", export_name = prefix!(gztell64))]
pub unsafe extern "C" fn gztell64(file: gzFile) -> z_off64_t {
    // SAFETY: The tell operation borrows a live aligned gzip state for non-null file; as_ref checks null.
    let Some(state) = (unsafe { file.cast::<GzState>().as_ref() }) else {
        return -1;
    };

    // Check integrity.
    if state.mode != GzMode::Read && state.mode != GzMode::Write {
        // Unreachable if `file` was initialized with `gzopen` or `gzdopen`.
        return -1;
    }

    // The logical position includes any lazy skip, but must always fit the
    // signed C offset type even for an internally inconsistent stream.
    if state.seek {
        state.pos.checked_add(state.skip).unwrap_or(-1)
    } else {
        state.pos
    }
}

/// Get the uncompressed stream position for the next operation.
///
/// Return the starting position for the next [`gzread`] or [`gzwrite`] on `file`.
/// This position represents a number of bytes in the uncompressed data stream,
/// and is zero when starting, even if appending or reading a gzip stream from
/// the middle of a file using [`gzdopen`].
///
/// # Returns
///
/// * The number of bytes prior to the current read or write position in the
///   uncompressed data stream, on success.
/// * -1 on error.
///
/// # Safety
///
/// - `file`, if non-null, must be an open file handle obtained from [`gzopen`] or [`gzdopen`].
#[cfg_attr(feature = "export-symbols", export_name = prefix!(gztell))]
pub unsafe extern "C" fn gztell(file: gzFile) -> z_off_t {
    // SAFETY: The public gztell wrapper inherits the same live-handle precondition as gztell64 and forwards it unchanged.
    z_off_t::try_from(unsafe { gztell64(file) }).unwrap_or(-1)
}

/// Get the compressed byte offset of an open gzip file.
///
/// Return the current compressed (actual) read or write offset of `file`.  This
/// offset includes the count of bytes that precede the gzip stream, for example
/// when appending or when using [`gzdopen`] for reading. When reading, the
/// offset does not include as yet unused buffered input. This information can
//  be used for a progress indicator.
///
/// # Returns
///
/// * The number of bytes prior to the current read or write position in the
///   compressed data stream, on success.
/// * -1 on error.
///
/// # Safety
///
/// - `file`, if non-null, must be an open file handle obtained from [`gzopen`] or [`gzdopen`].
#[cfg_attr(feature = "export-symbols", export_name = prefix!(gzoffset64))]
pub unsafe extern "C" fn gzoffset64(file: gzFile) -> z_off64_t {
    // SAFETY: The offset getter shares a valid aligned live gzip state for non-null file; as_ref checks null.
    let Some(state) = (unsafe { file.cast::<GzState>().as_ref() }) else {
        return -1;
    };

    // Check integrity.
    if state.mode != GzMode::Read && state.mode != GzMode::Write {
        // Unreachable if `file` was initialized with `gzopen` or `gzdopen`.
        return -1;
    }

    // Compute and return effective offset in file.
    let offset = lseek64(state.fd, 0, SEEK_CUR) as z_off64_t;
    if offset == -1 {
        return -1;
    }

    // When reading, don't count buffered input.
    match state.mode {
        GzMode::Read => offset - state.stream.avail_in as z_off64_t,
        GzMode::None | GzMode::Write | GzMode::Append => offset,
    }
}

/// Get the compressed byte offset of an open gzip file.
///
/// Return the current compressed (actual) read or write offset of `file`.  This
/// offset includes the count of bytes that precede the gzip stream, for example
/// when appending or when using [`gzdopen`] for reading. When reading, the
/// offset does not include as yet unused buffered input. This information can
//  be used for a progress indicator.
///
/// # Returns
///
/// * The number of bytes prior to the current read or write position in the
///   compressed data stream, on success.
/// * -1 on error.
///
/// # Safety
///
/// - `file`, if non-null, must be an open file handle obtained from [`gzopen`] or [`gzdopen`].
#[cfg_attr(feature = "export-symbols", export_name = prefix!(gzoffset))]
pub unsafe extern "C" fn gzoffset(file: gzFile) -> z_off_t {
    // SAFETY: The public gzoffset wrapper inherits the same handle precondition as gzoffset64 and forwards it unchanged.
    z_off_t::try_from(unsafe { gzoffset64(file) }).unwrap_or(-1)
}

/// Compress and write `c`, converted to an unsigned 8-bit char, into `file`.
///
/// # Returns
///
///  - The value that was written, on success.
///  - `-1` on error.
///
/// # Safety
///
/// - `file`, if non-null, must be an open file handle obtained from [`gzopen`] or [`gzdopen`].
#[cfg_attr(feature = "export-symbols", export_name = prefix!(gzputc))]
pub unsafe extern "C" fn gzputc(file: gzFile, c: c_int) -> c_int {
    // SAFETY: The caller supplies a uniquely borrowed live gzip handle or null; as_mut checks null.
    let Some(state) = (unsafe { file.cast::<GzState>().as_mut() }) else {
        return -1;
    };

    // Check that we're writing and that there's no error.
    if state.mode != GzMode::Write || state.err != Z_OK {
        return -1;
    }
    if state.ensure_position_room(1).is_err() {
        return -1;
    }

    // Check for seek request.
    if state.seek {
        state.seek = false;
        if gz_zero_pending(state).is_err() {
            return -1;
        }
    }

    // Try writing to the owned input buffer for speed when it has been initialized.
    if state.input.is_some() {
        if state.stream.avail_in == 0 {
            state.stream.next_in = state.input_ptr();
        }
        // Safety: `state.stream.next_in` points into the buffer starting at `state.input`.
        // (This is an invariant maintained throughout this module, except for a specific
        // block within `gz_write` that does not call any function that might call `gzputc`.)
        let Some(have) = state.owned_write_input_end() else {
            state.invalid_write_cursor();
            return -1;
        };
        if have < state.in_size() {
            state
                .input
                .as_mut()
                .expect("write input must remain allocated")
                .usable_uninit_mut()[have]
                .write(c.to_le_bytes()[0]);
            if state.advance_position(1).is_err() {
                return -1;
            }
            state.stream.avail_in += 1;
            return c & 0xff;
        }
    }

    // No room in buffer or not initialized, use gz_write.
    let buf = [c.to_le_bytes()[0]];
    // Safety: We have confirmed that `state` is valid, and `buf` contains 1 readable byte of data.
    let mut input = GzWriteInput::from_slice(&buf);
    match gz_write(state, &mut input) {
        1 => c & 0xff,
        _ => -1,
    }
}

/// Compress and write the given null-terminated string `s` to file, excluding
/// the terminating null character.
///
/// # Returns
///
/// - the number of characters written, on success.
/// - `-1` in case of error.
///
/// # Safety
///
/// - `file`, if non-null, must be an open file handle obtained from [`gzopen`] or [`gzdopen`].
/// - `s` must point to a null-terminated C string.
#[cfg_attr(feature = "export-symbols", export_name = prefix!(gzputs))]
pub unsafe extern "C" fn gzputs(file: gzFile, s: *const c_char) -> c_int {
    // SAFETY: The caller supplies a uniquely borrowed live gzip handle or null; as_mut checks null.
    let Some(state) = (unsafe { file.cast::<GzState>().as_mut() }) else {
        return -1;
    };

    if s.is_null() {
        return -1;
    }

    // Check that we're writing and that there's no error.
    if state.mode != GzMode::Write || state.err != Z_OK {
        return -1;
    }

    // Write string.
    // SAFETY: The gzputs C API requires s to point to a valid NUL-terminated string; strlen reads through that terminator.
    let len = unsafe { libc::strlen(s) };
    let Ok(len_abi) = c_int::try_from(len) else {
        const MSG: &str = "string length does not fit in int";
        gz_error(state, Some((Z_STREAM_ERROR, MSG)));
        return -1;
    };
    // Safety: `s` was validated above as a C string containing at least `len` bytes.
    let mut input = unsafe { GzWriteInput::from_raw(s.cast::<c_void>(), len) };
    let put = gz_write(state, &mut input);
    match put.cmp(&len) {
        Ordering::Less => -1,
        Ordering::Equal | Ordering::Greater => len_abi,
    }
}

/// Read one decompressed byte from `file`.
///
/// Note: The C header file `zlib.h` provides a macro wrapper for `gzgetc` that implements
/// the fast path inline and calls this function for the slow path.
///
/// # Returns
///
/// - The byte read, on success.
/// - `-1` on error.
///
/// # Safety
///
/// - `file`, if non-null, must be an open file handle obtained from [`gzopen`] or [`gzdopen`].
#[cfg_attr(feature = "export-symbols", export_name = prefix!(gzgetc))]
pub unsafe extern "C" fn gzgetc(file: gzFile) -> c_int {
    // SAFETY: The caller supplies an exclusive live gzip handle or null; as_mut checks null.
    let Some(state) = (unsafe { file.cast::<GzState>().as_mut() }) else {
        return -1;
    };

    // Check that we're reading and that there's no (serious) error.
    if !state.prepare_read() {
        return -1;
    }

    // Try output buffer (no need to check for skip request).
    if state.have != 0 {
        let Some(&ret) = state.buffered_output().and_then(|bytes| bytes.first()) else {
            state.invalid_read_cursor();
            return -1;
        };
        if state.advance_position(1).is_err() {
            return -1;
        }
        if !state.consume_buffered_output(1) {
            state.invalid_read_cursor();
            return -1;
        }
        return c_int::from(ret);
    }

    // Nothing there -- try the typed read core.
    let mut c = [MaybeUninit::<u8>::uninit()];
    match gz_read(state, &mut c) {
        // SAFETY: gz_read returned exactly one byte, so it initialized the first MaybeUninit entry before we read it.
        1 => c_int::from(unsafe { c[0].assume_init() }),
        _ => -1,
    }
}

/// Backward-compatibility alias for [`gzgetc`].
///
/// # Returns
///
/// - The byte read, on success.
/// - `-1` on error.
///
/// # Safety
///
/// - `file`, if non-null, must be an open file handle obtained from [`gzopen`] or [`gzdopen`].
#[cfg_attr(feature = "export-symbols", export_name = prefix!(gzgetc_))]
pub unsafe extern "C" fn gzgetc_(file: gzFile) -> c_int {
    // Safety: The caller has ensured that `file` is null or a valid file handle.
    unsafe { gzgetc(file) }
}

/// Push `c` back onto the stream for file to be read as the first character on
/// the next read.  At least one character of push-back is always allowed.
///
/// `gzungetc` will fail if `c` is `-1`, and may fail if a character has been pushed
/// but not read yet. If `gzungetc` is used immediately after [`gzopen`] or [`gzdopen`],
/// at least the output buffer size of pushed characters is allowed.  (See [`gzbuffer`].)
///
/// The pushed character will be discarded if the stream is repositioned with
/// [`gzseek`] or [`gzrewind`].
///
/// # Returns
///
/// - The character pushed, on success.
/// - `-1` on failure.
///
/// # Safety
///
/// - `file`, if non-null, must be an open file handle obtained from [`gzopen`] or [`gzdopen`].
#[cfg_attr(feature = "export-symbols", export_name = prefix!(gzungetc))]
pub unsafe extern "C" fn gzungetc(c: c_int, file: gzFile) -> c_int {
    // SAFETY: The caller supplies an exclusive live gzip handle or null; as_mut checks null.
    let Some(state) = (unsafe { file.cast::<GzState>().as_mut() }) else {
        return -1;
    };

    if c < 0 {
        return -1;
    }
    if !state.prepare_read() {
        return -1;
    }

    if state.how == How::Look && state.have == 0 {
        // Pushback requires writable owned output, not compressed-file input.
        // In particular, a pipe may be empty or nonblocking at this point.
        if gz_init_read(state).is_err() {
            return -1;
        }
    }

    if state.seek {
        state.seek = false;
        if gz_skip(state, state.skip).is_err() {
            return -1;
        }
    }

    let Some(new_pos) = state.pos.checked_sub(1) else {
        state.fail_position_overflow();
        return -1;
    };
    // C's unsigned-char conversion keeps the low eight bits of a nonnegative
    // int. Express that explicitly without a truncating signed-to-byte cast.
    match state.push_back_buffered_byte(c.to_le_bytes()[0]) {
        Ok(()) => {}
        Err(PushBackError::Full) => {
            gz_error(
                state,
                Some((Z_DATA_ERROR, "out of room to push characters")),
            );
            return -1;
        }
        Err(PushBackError::InvalidCursor) => {
            state.invalid_read_cursor();
            return -1;
        }
    }

    state.pos = new_pos;
    state.past = false;
    c
}

/// Read a NUL-terminated line of decompressed text.
///
/// Read decompressed bytes from `file` into `buf`, until `len-1` characters are
/// read, or until a newline character is read and transferred to `buf`, or an
/// end-of-file condition is encountered.  If any characters are read or if `len`
/// is one, the string is terminated with a null character.  If no characters
/// are read due to an end-of-file or `len` is less than one, then the buffer is
/// left untouched.
///
/// Note: This function generally only makes sense for files where the decompressed
/// content is text. If there are any null bytes, this function will copy them into
/// `buf` just like any other character, resulting in early truncation of the
/// returned C string. To read gzip files whose decompressed content is binary,
/// please see [`gzread`].
///
/// # Returns
///
/// - `buf`, which now is a null-terminated string, on success.
/// - `null` if no characters were obtained (except when `len` is one),
///   including on an initial read error. If an error occurs after some
///   characters have been copied, a NUL-terminated partial line is returned;
///   consult [`gzerror`] to detect the read failure.
///
/// # Safety
///
/// - `file`, if non-null, must be an open file handle obtained from [`gzopen`] or [`gzdopen`].
/// - `buf` must be null or a pointer to at least `len` writable bytes.
#[cfg_attr(feature = "export-symbols", export_name = prefix!(gzgets))]
pub unsafe extern "C" fn gzgets(file: gzFile, buf: *mut c_char, len: c_int) -> *mut c_char {
    if buf.is_null() || len < 1 {
        return ptr::null_mut();
    }

    // SAFETY: The enclosing gzip operation holds the live handle exclusively, or receives null, which as_mut rejects.
    let Some(state) = (unsafe { file.cast::<GzState>().as_mut() }) else {
        return ptr::null_mut();
    };
    if !state.prepare_read() {
        return ptr::null_mut();
    }

    if state.seek {
        state.seek = false;
        if gz_skip(state, state.skip).is_err() {
            return ptr::null_mut();
        }
    }

    // Safety: `buf` is non-null and the public contract guarantees `len` writable bytes.
    let output =
        unsafe { core::slice::from_raw_parts_mut(buf.cast::<MaybeUninit<u8>>(), len as usize) };
    let payload_capacity = output.len() - 1;
    if payload_capacity == 0 {
        output[0].write(0);
        return buf;
    }

    let mut written = 0usize;
    while written < payload_capacity {
        if state.have == 0 && gz_fetch(state).is_err() {
            // Preserve a partial line if a later fetch fails. As in zlib,
            // gzgets still returns NULL when no bytes were read; callers can
            // inspect gzerror for the underlying failure in either case.
            break;
        }
        if state.have == 0 {
            state.past = true;
            break;
        }

        let (through_eol, saw_newline) = {
            let Some(available) = state.buffered_output() else {
                state.invalid_read_cursor();
                return ptr::null_mut();
            };
            let limit = Ord::min(payload_capacity - written, available.len());
            let chunk = &available[..limit];
            let eol = chunk.iter().position(|&byte| byte == b'\n');
            let through_eol = eol.map_or(limit, |index| index + 1);
            (through_eol, eol.is_some())
        };

        if state.advance_position(through_eol).is_err() {
            return ptr::null_mut();
        }
        let Some(available) = state.buffered_output() else {
            state.invalid_read_cursor();
            return ptr::null_mut();
        };
        output[written..written + through_eol].write_copy_of_slice(&available[..through_eol]);
        if !state.consume_buffered_output(through_eol) {
            state.invalid_read_cursor();
            return ptr::null_mut();
        }
        written += through_eol;

        if saw_newline {
            break;
        }
    }

    if written == 0 {
        ptr::null_mut()
    } else {
        output[written].write(0);
        buf
    }
}

/// Change the compression level and strategy of an open gzip file.
///
/// Dynamically update the compression level and strategy for `file`. See the
/// description of [`deflateInit2_`] for the meaning of these parameters. Previously
/// provided data is flushed before applying the parameter changes.
///
/// Note: If `level` is not valid, this function will silently fail with a return
/// value of `Z_OK`, matching the semantics of the C zlib version. However, if
/// `strategy` is not valid, this function will return an error.
///
/// # Returns
///
/// - [`Z_OK`] on success.
/// - [`Z_STREAM_ERROR`] if the file was not opened for writing.
/// - [`Z_ERRNO`] if there is an error writing the flushed data.
/// - [`Z_MEM_ERROR`] if there is a memory allocation error.
///
/// # Safety
///
/// - `file`, if non-null, must be an open file handle obtained from [`gzopen`] or [`gzdopen`].
#[cfg_attr(feature = "export-symbols", export_name = prefix!(gzsetparams))]
pub unsafe extern "C" fn gzsetparams(file: gzFile, level: c_int, strategy: c_int) -> c_int {
    let Ok(strategy) = Strategy::try_from(strategy) else {
        return Z_STREAM_ERROR;
    };
    // SAFETY: The enclosing gzip operation holds the live handle exclusively, or receives null, which as_mut rejects.
    let Some(state) = (unsafe { file.cast::<GzState>().as_mut() }) else {
        return Z_STREAM_ERROR;
    };

    // Check that we're writing and that there's no error.
    if state.mode != GzMode::Write || state.err != Z_OK || state.direct {
        return Z_STREAM_ERROR;
    }

    // zlib-compatible behavior: ignore an invalid level without mutating the
    // stream. Valid levels are the default sentinel -1 and levels 0 through 9.
    let Ok(level_storage) = i8::try_from(level) else {
        return Z_OK;
    };
    if level_storage != -1 && !(0..=9).contains(&level_storage) {
        return Z_OK;
    }

    // If no change is requested, then do nothing.
    if level == c_int::from(state.level) && strategy == state.strategy {
        return Z_OK;
    }

    // Check for seek request.
    if state.seek {
        state.seek = false;
        if gz_zero_pending(state).is_err() {
            return state.err;
        }
    }

    // Change compression parameters for subsequent input.
    if state.input.is_some() {
        // Flush previous input with previous parameters before changing.
        if state.stream.avail_in != 0 && gz_comp(state, Z_BLOCK).is_err() {
            return state.err;
        }
        // Safety: Because `state` is in write mode and owns an input buffer, `state.stream`
        // was initialized using `deflateInit2` in `gz_init`.
        // SAFETY: gz_init initialized this live stream; the state holds its
        // compression input/output buffers throughout the parameter change.
        let ret = unsafe { super::deflateParams(&raw mut state.stream, level, strategy as c_int) };
        if ret != Z_OK {
            return ret;
        }
    }
    state.level = level_storage;
    state.strategy = strategy;
    Z_OK
}

/// Seek to an uncompressed byte offset of a gzip stream.
///
/// Set the starting position to `offset` relative to `whence` for the next [`gzread`]
/// or [`gzwrite`] on `file`. The `offset` represents a number of bytes in the
/// uncompressed data stream. The `whence` parameter is defined as in `lseek(2)`,
/// but only `SEEK_CUR` (relative to current position) and `SEEK_SET` (absolute from
/// start of the uncompressed data stream) are supported.
///
/// If `file` is open for reading, this function is emulated but can extremely
/// slow (because it operates on the decompressed data stream).  If `file` is open
/// for writing, only forward seeks are supported; `gzseek` then compresses a sequence
/// of zeroes up to the new starting position. If a negative `offset` is specified in
/// write mode, `gzseek` returns -1.
///
/// # Returns
///
/// - The resulting offset location as measured in bytes from the beginning of the uncompressed
///   stream, on success.
/// - `-1` on error.
///
/// # Safety
///
/// - `file`, if non-null, must be an open file handle obtained from [`gzopen`] or [`gzdopen`].
#[cfg_attr(feature = "export-symbols", export_name = prefix!(gzseek64))]
pub unsafe extern "C" fn gzseek64(file: gzFile, offset: z_off64_t, whence: c_int) -> z_off64_t {
    // SAFETY: The seek call receives an exclusive valid gzip handle or null; as_mut checks null.
    let Some(state) = (unsafe { file.cast::<GzState>().as_mut() }) else {
        return -1;
    };
    if state.mode != GzMode::Read && state.mode != GzMode::Write {
        // Unreachable if `file` was initialized with `gzopen` or `gzdopen`.
        return -1;
    }

    // A stalled read is retryable; all other outstanding errors remain sticky.
    if state.mode == GzMode::Read && !state.prepare_read() {
        return -1;
    }
    if state.err != Z_OK && state.err != Z_BUF_ERROR {
        return -1;
    }

    // Can only seek from start or relative to current position.
    if whence != SEEK_SET && whence != SEEK_CUR {
        return -1;
    }

    // Resolve the requested absolute logical position before changing any
    // pending seek state. i128 covers all combinations of i64 position, skip
    // and offset without risking a panic in this C ABI entry point.
    let current = i128::from(state.pos)
        + if state.seek {
            i128::from(state.skip)
        } else {
            0
        };
    let target = if whence == SEEK_SET {
        i128::from(offset)
    } else {
        current + i128::from(offset)
    };
    let Ok(target) = i64::try_from(target) else {
        return -1;
    };
    if target < 0 {
        return -1;
    }

    // Direct reads can seek in the underlying file without reconstructing
    // decompressor state. Account for the bytes still exposed via have.
    if state.mode == GzMode::Read && state.how == How::Copy {
        let disk_delta = i128::from(target) - i128::from(state.pos) - i128::from(state.have);
        let Ok(disk_delta) = z_off64_t::try_from(disk_delta) else {
            return -1;
        };
        if lseek64(state.fd, disk_delta, SEEK_CUR) == -1 {
            return -1;
        }
        state.have = 0;
        state.eof = false;
        state.past = false;
        state.seek = false;
        state.skip = 0;
        gz_error(state, None);
        state.stream.avail_in = 0;
        state.pos = target;
        return target;
    }

    // A write cannot undo already-committed uncompressed bytes, even if an
    // earlier seek is still pending.
    if state.mode == GzMode::Write {
        let Some(ahead) = target.checked_sub(state.pos) else {
            return -1;
        };
        if ahead < 0 {
            return -1;
        }
        state.seek = ahead != 0;
        state.skip = ahead;
        return target;
    }

    // For compressed reads, a backward seek needs a rewind and subsequent
    // forward decoding. Rewind first, and do not discard a pending seek if
    // the underlying descriptor refuses the operation.
    if target < state.pos && gzrewind_help(state) == -1 {
        return -1;
    }
    let Some(mut ahead) = target.checked_sub(state.pos) else {
        return -1;
    };

    // Consume already-decoded bytes before scheduling the remaining distance.
    // The C have field is bounded by the owned output buffer's uInt length.
    let buffered = u64::from(state.have);
    let Ok(ahead_unsigned) = u64::try_from(ahead) else {
        return -1;
    };
    let take = buffered.min(ahead_unsigned);
    let Ok(take) = usize::try_from(take) else {
        return -1;
    };
    if take != 0 {
        if !state.consume_buffered_output(take) {
            state.invalid_read_cursor();
            return -1;
        }
        let Ok(take_signed) = i64::try_from(take) else {
            return -1;
        };
        state.pos += take_signed;
        ahead -= take_signed;
    }

    state.seek = ahead != 0;
    state.skip = ahead;
    target
}

/// Seek to an uncompressed byte offset of a gzip stream.
///
/// Set the starting position to `offset` relative to `whence` for the next [`gzread`]
/// or [`gzwrite`] on `file`. The `offset` represents a number of bytes in the
/// uncompressed data stream. The `whence` parameter is defined as in `lseek(2)`,
/// but only `SEEK_CUR` (relative to current position) and `SEEK_SET` (absolute from
/// start of the uncompressed data stream) are supported.
///
/// If `file` is open for reading, this function is emulated but can extremely
/// slow (because it operates on the decompressed data stream).  If `file` is open
/// for writing, only forward seeks are supported; `gzseek` then compresses a sequence
/// of zeroes up to the new starting position. If a negative `offset` is specified in
/// write mode, `gzseek` returns -1.
///
/// # Returns
///
/// - The resulting offset location as measured in bytes from the beginning of the uncompressed
///   stream, on success.
/// - `-1` on error.
///
/// # Safety
///
/// - `file`, if non-null, must be an open file handle obtained from [`gzopen`] or [`gzdopen`].
#[cfg_attr(feature = "export-symbols", export_name = prefix!(gzseek))]
pub unsafe extern "C" fn gzseek(file: gzFile, offset: z_off_t, whence: c_int) -> z_off_t {
    // SAFETY: The public gzseek wrapper forwards the live handle and seek arguments to gzseek64 under its matching caller contract.
    z_off_t::try_from(unsafe { gzseek64(file, offset as z_off64_t, whence) }).unwrap_or(-1)
}

/// Rewind `file` to the start. This function is supported only for reading.
///
/// Note: `gzrewind(file)` is equivalent to [`gzseek`]`(file, 0, SEEK_SET)`
///
/// # Returns
///
/// - `0` on success.
/// - `-1` on error.
///
/// # Safety
///
/// - `file`, if non-null, must be an open file handle obtained from [`gzopen`] or [`gzdopen`].
#[cfg_attr(feature = "export-symbols", export_name = prefix!(gzrewind))]
pub unsafe extern "C" fn gzrewind(file: gzFile) -> c_int {
    // SAFETY: The rewind operation receives an exclusive valid gzip state or null; as_mut checks null.
    let Some(state) = (unsafe { file.cast::<GzState>().as_mut() }) else {
        return -1;
    };

    gzrewind_help(state)
}

fn gzrewind_help(state: &mut GzState) -> c_int {
    // Check that we're reading and that there's no error.
    if !state.prepare_read() {
        return -1;
    }

    // Back up and start over.
    if lseek64(state.fd, state.start as _, SEEK_SET) == -1 {
        return -1;
    }
    gz_reset(state);
    0
}

/// Convert, format, compress, and write the variadic arguments `...` to a file under control of the string format, as in `fprintf`.
///
/// # Returns
///
/// Returns the number of uncompressed bytes actually written, or a negative zlib error code in case of error.
/// The number of uncompressed bytes written is limited to 8191, or one less than the buffer size given to [`gzbuffer`].
/// The caller should assure that this limit is not exceeded. If it is exceeded, then [`gzprintf`] will return `0` with nothing written.
///
/// Contrary to other implementations that can use the insecure `vsprintf`, the `zlib-rs` library always uses `vsnprintf`,
/// so attempting to write more bytes than the limit can never run into buffer overflow issues.
///
/// # Safety
///
/// - The `format`  must be a valid C string
/// - The variadic arguments must correspond with the format string in number and type
#[cfg(feature = "gzprintf")]
#[cfg_attr(docsrs, doc(cfg(feature = "gzprintf")))]
#[cfg_attr(feature = "export-symbols", export_name = prefix!(gzprintf))]
pub unsafe extern "C" fn gzprintf(file: gzFile, format: *const c_char, va: ...) -> c_int {
    unsafe { gzvprintf(file, format, va) }
}

/// Convert, format, compress, and write the variable argument list to a file under control of the string format, as in `vfprintf`.
///
/// # Returns
///
/// Returns the number of uncompressed bytes actually written, or a negative zlib error code in case of error.
/// The number of uncompressed bytes written is limited to 8191, or one less than the buffer size given to [`gzbuffer`].
/// The caller should assure that this limit is not exceeded. If it is exceeded, then [`gzvprintf`] will return `0` with nothing written.
///
/// Contrary to other implementations that can use the insecure `vsprintf`, the `zlib-rs` library always uses `vsnprintf`,
/// so attempting to write more bytes than the limit can never run into buffer overflow issues.
///
/// # Safety
///
/// - The `format`  must be a valid C string
/// - The variadic arguments must correspond with the format string in number and type
#[cfg(feature = "gzprintf")]
#[cfg_attr(docsrs, doc(cfg(feature = "gzprintf")))]
#[cfg_attr(feature = "export-symbols", export_name = prefix!(gzvprintf))]
pub unsafe extern "C" fn gzvprintf(
    file: gzFile,
    format: *const c_char,
    va: core::ffi::VaList,
) -> c_int {
    let Some(state) = (unsafe { file.cast::<GzState>().as_mut() }) else {
        return Z_STREAM_ERROR;
    };

    // Check that we're writing and that there's no error.
    if state.mode != GzMode::Write || state.err != Z_OK {
        return Z_STREAM_ERROR;
    }

    // Make sure we have some buffer space.
    if state.input.is_none() && gz_init(state).is_err() {
        return state.err;
    }

    // Check for seek request.
    if state.seek {
        state.seek = false;
        if gz_zero_pending(state).is_err() {
            return state.err;
        }
    }

    // Do the printf() into the input buffer, put length in len -- the input
    // buffer is double-sized just for this function, so there is guaranteed to
    // be state.size bytes available after the current contents
    if state.stream.avail_in == 0 {
        state.stream.next_in = state.input_ptr();
    }

    // The write input allocation is twice `in_size()` specifically for this function,
    // so at least one full `in_size()` span remains available after the current prefix.
    let Some(have) = state.owned_write_input_end() else {
        state.invalid_write_cursor();
        return Z_STREAM_ERROR;
    };
    let Some(input) = state.input.as_ref() else {
        state.invalid_write_cursor();
        return Z_STREAM_ERROR;
    };
    if have > state.in_size() {
        state.invalid_write_cursor();
        return Z_STREAM_ERROR;
    }
    // SAFETY: there is one full input-sized block of reserved owned capacity.
    let next = unsafe { input.as_ptr().add(have) };

    // NOTE: zlib-ng writes a NULL byte to the last position of the input buffer. It must do so
    // because in some cases it falls back to the `vsprintf` function, which contrary to
    // `vsnprintf` does not guarantee NULL-termination.
    //
    // We do not support using `vsprintf`, and therefore don't need to write or check that byte.

    // This function is not currently exposed by libc, because `core::ffi::VaList` is unstable.
    extern "C" {
        fn vsnprintf(
            s: *mut c_char,
            n: libc::size_t,
            format: *const c_char,
            va: core::ffi::VaList,
        ) -> c_int;
    }

    // Safety: as described earlier, there are at least `state.in_size()` bytes available starting at
    // `next`. We forward `format` and `va`, so the caller is responsible for guarenteeing that
    // these are valid.
    let len = unsafe { vsnprintf(next.cast::<c_char>(), state.in_size(), format, va) };

    // Check that printf() results fit in buffer.
    if len == 0 || len as usize >= state.in_size() {
        return 0;
    }

    // Validate the signed logical position before publishing formatted data.
    if state.advance_position(len as usize).is_err() {
        return Z_STREAM_ERROR;
    }
    state.stream.avail_in += len as u32;
    if state.stream.avail_in as usize >= state.in_size() {
        let left = state.stream.avail_in - state.in_size() as u32;
        state.stream.avail_in = state.in_size() as u32;
        if gz_comp(state, Z_NO_FLUSH).is_err() {
            return state.err;
        }
        unsafe {
            core::ptr::copy(
                state.input_ptr().add(state.in_size()),
                state.input_ptr(),
                left as usize,
            )
        };
        state.stream.next_in = state.input_ptr();
        state.stream.avail_in = left;
    }

    len
}

fn lseek64(fd: c_int, offset: z_off64_t, origin: c_int) -> z_off64_t {
    #[cfg(any(target_os = "linux", target_os = "android", target_os = "windows"))]
    {
        // SAFETY: lseek64 accepts a file-descriptor integer and offset; invalid descriptors or origins are returned as errors by libc.
        return unsafe { libc::lseek64(fd, offset as _, origin) as z_off64_t };
    }

    #[cfg(not(any(target_os = "linux", target_os = "android", target_os = "windows")))]
    {
        (unsafe { libc::lseek(fd, offset as _, origin) }) as z_off64_t
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::path::Path;

    fn test_seek(state: &mut GzState, offset: i64, whence: c_int) -> i64 {
        // SAFETY: the state remains live, initialized and uniquely borrowed
        // for the duration of this C ABI call.
        unsafe { gzseek64(core::ptr::from_mut(state).cast(), offset, whence) }
    }

    fn test_tell(state: &GzState) -> i64 {
        // SAFETY: gztell64 only observes this live initialized state.
        unsafe { gztell64(core::ptr::from_ref(state).cast_mut().cast()) }
    }

    fn test_putc(state: &mut GzState, byte: c_int) -> c_int {
        // SAFETY: the state is live and exclusively borrowed during the call.
        unsafe { gzputc(core::ptr::from_mut(state).cast(), byte) }
    }

    fn test_getc(state: &mut GzState) -> c_int {
        // SAFETY: the state is live and exclusively borrowed during the call.
        unsafe { gzgetc(core::ptr::from_mut(state).cast()) }
    }

    fn test_ungetc(state: &mut GzState, byte: c_int) -> c_int {
        // SAFETY: the state is live and exclusively borrowed during the call.
        unsafe { gzungetc(byte, core::ptr::from_mut(state).cast()) }
    }

    fn test_read(state: &mut GzState, output: &mut [u8]) -> c_int {
        let Ok(len) = c_uint::try_from(output.len()) else {
            return -1;
        };
        // SAFETY: output is a live, writable and disjoint caller span.
        unsafe {
            gzread(
                core::ptr::from_mut(state).cast(),
                output.as_mut_ptr().cast(),
                len,
            )
        }
    }

    fn test_write(state: &mut GzState, input: &[u8]) -> c_int {
        let Ok(len) = c_uint::try_from(input.len()) else {
            return 0;
        };
        // SAFETY: input is a live, readable and disjoint caller span.
        unsafe {
            gzwrite(
                core::ptr::from_mut(state).cast(),
                input.as_ptr().cast(),
                len,
            )
        }
    }

    fn test_gets(state: &mut GzState, output: &mut [u8]) -> *mut c_char {
        let Ok(len) = c_int::try_from(output.len()) else {
            return ptr::null_mut();
        };
        // SAFETY: output is a live writable byte span for the call.
        unsafe {
            gzgets(
                core::ptr::from_mut(state).cast(),
                output.as_mut_ptr().cast(),
                len,
            )
        }
    }

    // Generate a file path relative to the project's root
    fn crate_path(file: &str) -> String {
        path(Path::new(env!("CARGO_MANIFEST_DIR")), file)
    }

    fn path(prefix: &Path, file: &str) -> String {
        let mut path_buf = prefix.to_path_buf();
        path_buf.push(file);
        path_buf.to_string_lossy().into_owned()
    }

    #[test]
    fn short_writes_do_not_drop_unwritten_bytes() -> Result<(), ()> {
        let input: Vec<u8> = (0u8..91).collect();
        let mut output = Vec::new();
        let mut steps = 0usize;
        write_all_with(&input, |remaining| {
            steps += 1;
            let count = remaining.len().min(steps % 7 + 1);
            output.extend_from_slice(&remaining[..count]);
            Ok(count)
        })?;
        assert!(steps > 1);
        assert_eq!(output, input);

        // No-progress and impossible writer responses must not spin.
        assert_eq!(write_all_with(&input, |_| Ok(0)), Err(()));
        assert_eq!(write_all_with(&input, |bytes| Ok(bytes.len() + 1)), Err(()));
        assert_eq!(
            write_all_with(&input, |_| Err(IoAttemptError::Failed)),
            Err(())
        );
        assert_eq!(write_all_with(&[], |_| Err(IoAttemptError::Failed)), Ok(()));
        let mut retries = 0;
        let mut written = Vec::new();
        write_all_with(&input, |remaining| {
            retries += 1;
            if retries < 4 {
                return Err(IoAttemptError::Interrupted);
            }
            written.extend_from_slice(remaining);
            Ok(remaining.len())
        })?;
        assert_eq!(written, input);
        assert_eq!(retries, 4);
        Ok(())
    }

    // A complete gzip stream of abc, and a separate valid member for xyz.
    // A split before the first footer forces inflate to produce three
    // bytes before a later input refill encounters EAGAIN.
    const ABC_GZ: [u8; 23] = [
        31, 139, 8, 0, 0, 0, 0, 0, 2, 255, 75, 76, 74, 6, 0, 194, 65, 36, 53, 3, 0, 0, 0,
    ];
    const XYZ_GZ: [u8; 23] = [
        31, 139, 8, 0, 0, 0, 0, 0, 2, 255, 171, 168, 172, 2, 0, 103, 186, 142, 235, 3, 0, 0, 0,
    ];

    #[cfg(unix)]
    #[test]
    fn gzip_read_returns_all_copied_bytes_before_later_would_block() {
        use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd};

        #[derive(Clone, Copy, PartialEq, Eq)]
        enum InputShape {
            Plain,
            CompleteGzip,
            SplitGzip,
        }

        for scenario in [
            InputShape::Plain,
            InputShape::CompleteGzip,
            InputShape::SplitGzip,
        ] {
            for items in [false, true] {
                let mut descriptors = [-1; 2];
                // SAFETY: two valid writable C integer slots.
                assert_eq!(unsafe { libc::pipe(descriptors.as_mut_ptr()) }, 0);
                // SAFETY: pipe returned a unique read descriptor.
                let reader = unsafe { OwnedFd::from_raw_fd(descriptors[0]) };
                // SAFETY: pipe returned a unique write descriptor.
                let writer = unsafe { OwnedFd::from_raw_fd(descriptors[1]) };
                // SAFETY: reader is a live descriptor.
                let flags = unsafe { libc::fcntl(reader.as_raw_fd(), libc::F_GETFL) };
                assert_ne!(flags, -1);
                // SAFETY: new flags are valid for the live reader descriptor.
                let configured = unsafe {
                    libc::fcntl(reader.as_raw_fd(), libc::F_SETFL, flags | libc::O_NONBLOCK)
                };
                assert_eq!(configured, 0);

                let send = |bytes: &[u8]| {
                    // SAFETY: writer remains live and bytes is an initialized readable slice.
                    let written = unsafe {
                        libc::write(writer.as_raw_fd(), bytes.as_ptr().cast(), bytes.len())
                    };
                    assert_eq!(usize::try_from(written), Ok(bytes.len()));
                };
                match scenario {
                    InputShape::Plain => send(b"abc"),
                    InputShape::CompleteGzip => send(&ABC_GZ),
                    InputShape::SplitGzip => send(&ABC_GZ[..15]),
                }

                let fd = reader.into_raw_fd();
                // SAFETY: gzdopen takes ownership of the live read descriptor.
                let file = unsafe { gzdopen(fd, c"rb".as_ptr()) };
                if file.is_null() {
                    // SAFETY: ownership wasn't transferred if gzdopen failed.
                    unsafe { libc::close(fd) };
                }
                assert!(!file.is_null());

                let read = |buffer: &mut [u8; 16]| {
                    if items {
                        // SAFETY: live handle and at least 16 writable single-byte items.
                        let amount = unsafe { gzfread(buffer.as_mut_ptr().cast(), 1, 16, file) };
                        isize::try_from(amount).unwrap_or(-1)
                    } else {
                        // SAFETY: live handle and 16 writable bytes.
                        let amount = unsafe { gzread(file, buffer.as_mut_ptr().cast(), 16) };
                        isize::try_from(amount).unwrap_or(-1)
                    }
                };
                let mut output = [0xa5u8; 16];
                let first = read(&mut output);
                assert_eq!(
                    first, 3,
                    "partial output must not be discarded on subsequent EAGAIN"
                );
                assert_eq!(&output[..3], b"abc");
                assert_eq!(output[3], 0xa5);
                // Verify the success did not incorrectly mark an incomplete
                // compressed member as EOF or rewind its logical position.
                // SAFETY: file remains a live gzip handle.
                assert_eq!(unsafe { gztell64(file) }, 3);

                if scenario == InputShape::Plain {
                    send(b"xyz");
                } else {
                    if scenario == InputShape::SplitGzip {
                        send(&ABC_GZ[15..]);
                    }
                    send(&XYZ_GZ);
                }
                output.fill(0xa5);
                let second = read(&mut output);
                assert_eq!(second, 3);
                assert_eq!(&output[..3], b"xyz");
                assert_eq!(output[3], 0xa5);
                // SAFETY: file remains valid until gzclose consumes it.
                assert_eq!(unsafe { gzclose(file) }, Z_OK);
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn gzip_read_preserves_output_before_corrupted_crc() {
        use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd};

        // A corrupt gzip CRC is a permanent error, unlike a transient
        // EAGAIN. Still return bytes successfully decompressed before that
        // trailer error, but keep the error sticky for subsequent reads.
        let mut corrupt = ABC_GZ;
        corrupt[15] ^= 0x5a;
        let mut descriptors = [-1; 2];
        // SAFETY: descriptors provides exactly two writable integer slots.
        assert_eq!(unsafe { libc::pipe(descriptors.as_mut_ptr()) }, 0);
        // SAFETY: pipe yielded this exclusive read descriptor.
        let reader = unsafe { OwnedFd::from_raw_fd(descriptors[0]) };
        // SAFETY: pipe yielded this exclusive write descriptor.
        let writer = unsafe { OwnedFd::from_raw_fd(descriptors[1]) };
        // SAFETY: writer is live; corrupt contains 23 initialized bytes.
        let written =
            unsafe { libc::write(writer.as_raw_fd(), corrupt.as_ptr().cast(), corrupt.len()) };
        assert_eq!(usize::try_from(written), Ok(corrupt.len()));
        drop(writer);

        let fd = reader.into_raw_fd();
        // SAFETY: gzdopen takes ownership of the live descriptor on success.
        let file = unsafe { gzdopen(fd, c"rb".as_ptr()) };
        if file.is_null() {
            // SAFETY: ownership remains with caller when gzdopen fails.
            unsafe { libc::close(fd) };
        }
        assert!(!file.is_null());
        let mut data = [0xa5u8; 16];
        // SAFETY: file is live; data contains 16 writable bytes.
        let first = unsafe { gzread(file, data.as_mut_ptr().cast(), 16) };
        assert_eq!(first, 3);
        assert_eq!(&data[..3], b"abc");
        assert_eq!(data[3], 0xa5);
        let mut error = 0;
        // SAFETY: file remains live and error is an aligned writable C integer.
        let message = unsafe { gzerror(file, &raw mut error) };
        assert!(!message.is_null());
        assert_eq!(error, Z_DATA_ERROR);
        // SAFETY: file remains live; the same writable destination is valid.
        assert_eq!(unsafe { gzread(file, data.as_mut_ptr().cast(), 16) }, -1);
        // SAFETY: this consumes the live gzip handle and read descriptor.
        assert_eq!(unsafe { gzclose(file) }, Z_OK);
    }

    #[cfg(unix)]
    #[test]
    fn gzip_pushback_before_first_pipe_byte_never_probes_input() {
        use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd};

        const GZIP_ABC: [u8; 23] = [
            31, 139, 8, 0, 0, 0, 0, 0, 2, 255, 75, 76, 74, 6, 0, 194, 65, 36, 53, 3, 0, 0, 0,
        ];

        for compressed in [false, true] {
            let mut pair = [-1; 2];
            // SAFETY: pair provides two writable descriptor slots.
            assert_eq!(unsafe { libc::pipe(pair.as_mut_ptr()) }, 0);
            // SAFETY: pipe returned two new, distinct live descriptors.
            let reader = unsafe { OwnedFd::from_raw_fd(pair[0]) };
            // SAFETY: pipe returned this unique live write descriptor.
            let writer = unsafe { OwnedFd::from_raw_fd(pair[1]) };

            // SAFETY: reader is live and F_GETFL queries its descriptor flags.
            let flags = unsafe { libc::fcntl(reader.as_raw_fd(), libc::F_GETFL) };
            assert_ne!(flags, -1);
            // SAFETY: reader is live and F_SETFL accepts the original flags
            // with O_NONBLOCK enabled.
            let configured =
                unsafe { libc::fcntl(reader.as_raw_fd(), libc::F_SETFL, flags | libc::O_NONBLOCK) };
            assert_eq!(configured, 0);

            let fd = reader.into_raw_fd();
            // SAFETY: gzdopen takes ownership of fd on success; mode is NUL terminated.
            let file = unsafe { gzdopen(fd, c"rb".as_ptr()) };
            if file.is_null() {
                // SAFETY: ownership remains ours if open failed.
                unsafe { libc::close(fd) };
            }
            assert!(!file.is_null());
            // SAFETY: handle is live; buffer size must be set before first read.
            let configured = unsafe { gzbuffer(file, 64) };
            assert_eq!(configured, 0);

            // There is deliberately *no input* in the pipe yet. Pushback must
            // not block on or attempt a read from the underlying descriptor.
            // SAFETY: handle remains exclusively live.
            let pushed = unsafe { gzungetc(c_int::from(b'X'), file) };
            assert_eq!(pushed, c_int::from(b'X'));
            // SAFETY: handle remains live.
            assert_eq!(unsafe { gztell64(file) }, -1);

            // The documented pre-read pushback capacity must also remain
            // available without looking ahead into the source stream.
            let mut pushed = [0u8; 64];
            pushed[0] = b'X';
            for (slot, &byte) in pushed[1..]
                .iter_mut()
                .zip(b"abcdefghijklmnopqrstuvwxyz".iter().cycle())
            {
                *slot = byte;
                // SAFETY: handle remains live and only the supplied byte is buffered.
                let returned = unsafe { gzungetc(c_int::from(byte), file) };
                assert_eq!(returned, c_int::from(byte));
            }
            // SAFETY: handle is live; observing position does not consume data.
            assert_eq!(unsafe { gztell64(file) }, -64);

            let mut out = [0xa5u8; 65];
            // SAFETY: handle is live and out has at least 64 writable bytes.
            let got = unsafe { gzread(file, out.as_mut_ptr().cast(), 64) };
            assert_eq!(got, 64);
            for (index, &expected) in pushed.iter().rev().enumerate() {
                assert_eq!(out[index], expected);
            }
            assert_eq!(out[64], 0xa5);

            let bytes: &[u8] = if compressed { &GZIP_ABC } else { b"abc" };
            // SAFETY: writer is live and bytes is a valid initialized input span.
            let wrote =
                unsafe { libc::write(writer.as_raw_fd(), bytes.as_ptr().cast(), bytes.len()) };
            assert_eq!(usize::try_from(wrote), Ok(bytes.len()));
            drop(writer);

            // SAFETY: handle remains live and out is a writable 65-byte buffer.
            let got = unsafe { gzread(file, out.as_mut_ptr().cast(), 3) };
            assert_eq!(got, 3);
            assert_eq!(&out[..3], b"abc");
            // SAFETY: handle remains live until consumed by gzclose.
            assert_eq!(unsafe { gztell64(file) }, 3);
            // SAFETY: gzclose consumes the handle and owned read descriptor.
            assert_eq!(unsafe { gzclose(file) }, Z_OK);
        }
    }

    #[cfg(unix)]
    #[test]
    fn nonblocking_reads_retry_when_input_arrives_without_clearerr() {
        use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd};

        #[derive(Clone, Copy, PartialEq, Eq)]
        enum ReadMethod {
            Bulk,
            Byte,
            Line,
            Items,
        }

        // A complete, valid gzip member containing six uncompressed bytes.
        // The first two bytes are sent separately to exercise an incomplete
        // header followed by EAGAIN before the rest of the member arrives.
        const GZIP_HELLO: [u8; 26] = [
            31, 139, 8, 0, 0, 0, 0, 0, 2, 255, 203, 72, 205, 201, 201, 231, 2, 0, 32, 48, 58, 54,
            6, 0, 0, 0,
        ];
        for compressed in [false, true] {
            for method in [
                ReadMethod::Bulk,
                ReadMethod::Byte,
                ReadMethod::Line,
                ReadMethod::Items,
            ] {
                let mut descriptors = [-1; 2];
                // SAFETY: descriptors provides two writable C integer slots.
                assert_eq!(unsafe { libc::pipe(descriptors.as_mut_ptr()) }, 0);
                // SAFETY: pipe returned this unique live read descriptor.
                let reader = unsafe { OwnedFd::from_raw_fd(descriptors[0]) };
                // SAFETY: pipe returned this unique live write descriptor.
                let writer = unsafe { OwnedFd::from_raw_fd(descriptors[1]) };
                // SAFETY: reader is a live descriptor.
                let flags = unsafe { libc::fcntl(reader.as_raw_fd(), libc::F_GETFL) };
                assert_ne!(flags, -1);
                // SAFETY: the obtained flags are valid for this live descriptor.
                let configured = unsafe {
                    libc::fcntl(reader.as_raw_fd(), libc::F_SETFL, flags | libc::O_NONBLOCK)
                };
                assert_eq!(configured, 0);
                let fd = reader.into_raw_fd();
                // SAFETY: ownership of the live read descriptor is transferred.
                let file = unsafe { gzdopen(fd, c"rb".as_ptr()) };
                if file.is_null() {
                    // SAFETY: no ownership transfer occurred on failed open.
                    unsafe { libc::close(fd) };
                }
                assert!(!file.is_null());

                let mut data = [0xa5u8; 8];
                let attempt = |data: &mut [u8; 8]| match method {
                    ReadMethod::Bulk => {
                        // SAFETY: file stays live and destination has six writable bytes.
                        unsafe { gzread(file, data.as_mut_ptr().cast(), 6) }
                    }
                    ReadMethod::Byte => {
                        // SAFETY: file stays live until gzclose below.
                        unsafe { gzgetc(file) }
                    }
                    ReadMethod::Line => {
                        // SAFETY: file stays live; data holds an eight-byte line.
                        let result = unsafe { gzgets(file, data.as_mut_ptr().cast(), 8) };
                        if result.is_null() {
                            -1
                        } else {
                            6
                        }
                    }
                    ReadMethod::Items => {
                        // SAFETY: file stays live; data has three writable two-byte items.
                        let count = unsafe { gzfread(data.as_mut_ptr().cast(), 2, 3, file) };
                        if count == 0 {
                            -1
                        } else {
                            6
                        }
                    }
                };

                assert_eq!(attempt(&mut data), -1);
                let mut err = 0;
                // SAFETY: file is live and err is an aligned writable C integer.
                let message = unsafe { gzerror(file, &raw mut err) };
                assert!(!message.is_null());
                assert_eq!(err, Z_ERRNO);

                let send = |bytes: &[u8]| {
                    // SAFETY: writer is live and bytes is a readable slice.
                    let written = unsafe {
                        libc::write(writer.as_raw_fd(), bytes.as_ptr().cast(), bytes.len())
                    };
                    assert_eq!(usize::try_from(written), Ok(bytes.len()));
                };
                if compressed {
                    send(&GZIP_HELLO[..2]);
                    // Two gzip magic bytes do not yet provide a complete member.
                    assert_eq!(attempt(&mut data), -1);
                    // SAFETY: file remains live and the error output is writable.
                    let _ = unsafe { gzerror(file, &raw mut err) };
                    assert_eq!(err, Z_ERRNO);
                    send(&GZIP_HELLO[2..]);
                } else {
                    send(b"hello\n");
                }

                let result = attempt(&mut data);
                if method == ReadMethod::Byte {
                    assert_eq!(result, i32::from(b'h'));
                } else {
                    assert_eq!(result, 6);
                    assert_eq!(&data[..6], b"hello\n");
                }
                // SAFETY: this closes the gzip handle and its owned read descriptor.
                assert_eq!(unsafe { gzclose(file) }, Z_OK);
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn nonblocking_gzip_pipe_preserves_bytes_read_before_would_block() {
        use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd};

        let mut descriptors = [-1; 2];
        // SAFETY: descriptors has exactly two writable C integer slots.
        assert_eq!(unsafe { libc::pipe(descriptors.as_mut_ptr()) }, 0);
        // SAFETY: pipe returned two new descriptors, transferred into RAII owners.
        let reader = unsafe { OwnedFd::from_raw_fd(descriptors[0]) };
        // SAFETY: pipe returned the second unique descriptor.
        let writer = unsafe { OwnedFd::from_raw_fd(descriptors[1]) };
        // SAFETY: reader is a live descriptor, and fcntl only inspects/modifies flags.
        let flags = unsafe { libc::fcntl(reader.as_raw_fd(), libc::F_GETFL) };
        assert_ne!(flags, -1);
        // SAFETY: flags are obtained from this descriptor and O_NONBLOCK is valid.
        let set_flags =
            unsafe { libc::fcntl(reader.as_raw_fd(), libc::F_SETFL, flags | libc::O_NONBLOCK) };
        assert_eq!(set_flags, 0);

        let payload = b"abc";
        // SAFETY: writer is open, and payload is a live three-byte input.
        let wrote =
            unsafe { libc::write(writer.as_raw_fd(), payload.as_ptr().cast(), payload.len()) };
        assert_eq!(usize::try_from(wrote), Ok(payload.len()));
        // Ownership of the read descriptor passes to gzdopen on success.
        let fd = reader.into_raw_fd();
        // SAFETY: the live descriptor was just transferred; mode is a C string.
        let file = unsafe { gzdopen(fd, c"rb".as_ptr()) };
        if file.is_null() {
            // SAFETY: gzdopen did not take ownership on failure.
            unsafe { libc::close(fd) };
        }
        assert!(
            !file.is_null(),
            "gzdopen failed for a live nonblocking pipe"
        );

        let mut data = [0xa5u8; 4];
        // SAFETY: file is live and data provides three writable bytes.
        let got = unsafe { gzread(file, data.as_mut_ptr().cast(), 3) };
        assert_eq!(got, 3, "bytes already read from the pipe must not be lost");
        assert_eq!(&data[..3], payload);
        assert_eq!(data[3], 0xa5);

        // The writer remains open. A fresh read with no new bytes must report
        // the nonblocking syscall error; the earlier successful read did not.
        // SAFETY: file remains live, and the output has one writable byte.
        let got = unsafe { gzread(file, data.as_mut_ptr().cast(), 1) };
        assert_eq!(got, -1);
        let mut err = 0;
        // SAFETY: file is live and err points to a writable C integer.
        let msg = unsafe { gzerror(file, &raw mut err) };
        assert!(!msg.is_null());
        assert_eq!(err, Z_ERRNO);
        // SAFETY: gzclose consumes and releases the file and its read descriptor.
        assert_eq!(unsafe { gzclose(file) }, Z_OK);
    }

    #[test]
    fn interrupted_partial_reads_preserve_initialized_prefix() -> Result<(), ()> {
        let mut output = [MaybeUninit::<u8>::uninit(); 19];
        let mut calls = 0usize;
        let mut next = 0u8;
        let (read, eof) = read_with(&mut output, |dst| {
            calls += 1;
            if calls % 3 == 1 {
                return Err(IoAttemptError::Interrupted);
            }
            let count = dst.len().min(4);
            for byte in &mut dst[..count] {
                byte.write(next);
                next += 1;
            }
            Ok(count)
        })
        .map_err(|_| ())?;
        assert_eq!(read, output.len());
        assert!(!eof);
        assert!(calls > 5);
        for (index, byte) in output.iter().enumerate() {
            // SAFETY: every byte in output was initialized before we reached
            // full length, as asserted above.
            let actual = unsafe { byte.assume_init() };
            assert_eq!(actual, u8::try_from(index).map_err(|_| ())?);
        }

        let mut partial = [MaybeUninit::<u8>::uninit(); 5];
        let mut calls = 0usize;
        let (read, eof) = read_with(&mut partial, |dst| {
            calls += 1;
            if calls == 1 {
                dst[0].write(42);
                Ok(1)
            } else {
                Ok(0)
            }
        })
        .map_err(|_| ())?;
        assert_eq!((read, eof), (1, true));
        assert_eq!(
            read_with(&mut partial, |_| Err(IoAttemptError::Failed)),
            Err(IoAttemptError::Failed)
        );
        // An I/O failure after a successful short read cannot revoke bytes
        // that the kernel already consumed from the input source.
        let mut calls = 0;
        let (read, eof) = read_with(&mut partial, |dst| {
            calls += 1;
            if calls == 1 {
                dst[..2].write_copy_of_slice(b"xy");
                Ok(2)
            } else {
                Err(IoAttemptError::Failed)
            }
        })
        .map_err(|_| ())?;
        assert_eq!((read, eof), (2, false));
        // SAFETY: the successful first call initialized exactly two bytes.
        assert_eq!(unsafe { partial[0].assume_init() }, b'x');
        // SAFETY: the second byte is within the successfully initialized prefix.
        assert_eq!(unsafe { partial[1].assume_init() }, b'y');
        Ok(())
    }

    #[test]
    fn test_configure() -> Result<(), ()> {
        let mut state = GzState::new();

        state.configure(b"r")?;
        assert_eq!(state.mode, GzMode::Read);
        state.configure(b"rw")?;
        assert_eq!(state.mode, GzMode::Write);
        state.configure(b"wr")?;
        assert_eq!(state.mode, GzMode::Read);

        state.configure(b"4")?;
        assert_eq!(state.level, 4);
        state.configure(b"64")?;
        assert_eq!(state.level, 4);

        state.configure(b"f")?;
        assert_eq!(state.strategy, Strategy::Filtered);
        state.configure(b"h")?;
        assert_eq!(state.strategy, Strategy::HuffmanOnly);
        state.configure(b"R")?;
        assert_eq!(state.strategy, Strategy::Rle);
        state.configure(b"F")?;
        assert_eq!(state.strategy, Strategy::Fixed);

        // Unknown characters are ignored.
        state.configure(b"xqz")?;

        // Plus errors (read + write mode is not supported)
        assert!(state.configure(b"123+").is_err());

        assert_eq!(state.configure(b""), Ok((false, false)));
        assert_eq!(state.configure(b"x"), Ok((true, false)));
        assert_eq!(state.configure(b"e"), Ok((false, true)));
        assert_eq!(state.configure(b"xe"), Ok((true, true)));
        Ok(())
    }

    #[test]
    fn gzip_skip_preserves_signed_position_and_buffered_cursor_bounds() -> Result<(), &'static str>
    {
        let mut state = GzState::new();
        state.mode = GzMode::Read;
        state.eof = true;
        let mut output = GzBuffer::allocate(8, 8).ok_or("read output allocation failed")?;
        output.usable_uninit_mut()[..3].write_copy_of_slice(b"abc");
        state.output = Some(output);

        // A huge positive logical seek can exceed usize on a 32-bit host.
        // It must consume the available bytes and stop at EOF, not wrap to zero.
        state.publish_buffered_output(0, 3);
        assert_eq!(gz_skip(&mut state, i64::MAX), Ok(()));
        assert_eq!(state.pos, 3);
        assert_eq!(state.have, 0);

        // A smaller skip only advances the owned read cursor by that amount.
        state.pos = 0;
        state.publish_buffered_output(0, 3);
        assert_eq!(gz_skip(&mut state, 2), Ok(()));
        assert_eq!(state.pos, 2);
        assert_eq!(state.have, 1);
        assert_eq!(state.buffered_output(), Some(&b"c"[..]));

        // Never consume a buffered prefix if it would overflow the signed
        // uncompressed stream position. Fatal errors hide the buffer from gzgetc.
        state.pos = i64::MAX - 1;
        state.publish_buffered_output(0, 3);
        let cursor = state.next;
        assert_eq!(gz_skip(&mut state, 3), Err(()));
        assert_eq!(state.pos, i64::MAX - 1);
        assert_eq!(state.next, cursor);
        assert_eq!(state.have, 0);
        assert_eq!(state.err, Z_STREAM_ERROR);

        // An invalid negative internal request must not start fetching data
        // or reinterpret its signed length as an enormous usize.
        state.err = Z_OK;
        state.pos = 0;
        state.publish_buffered_output(0, 3);
        let cursor = state.next;
        assert_eq!(gz_skip(&mut state, -1), Err(()));
        assert_eq!(state.pos, 0);
        assert_eq!(state.next, cursor);
        assert_eq!(state.have, 0);
        assert_eq!(state.err, Z_STREAM_ERROR);

        // Exercise the same large skip from the public C entrypoints: a read
        // seek is initially virtual, then the next read resolves it at EOF.
        gz_error(&mut state, None);
        state.mode = GzMode::Read;
        state.how = How::Gzip;
        state.pos = 0;
        state.eof = true;
        state.publish_buffered_output(0, 3);
        assert_eq!(test_seek(&mut state, i64::MAX, SEEK_SET), i64::MAX);
        assert_eq!(state.pos, 3);
        assert!(state.seek);
        let mut buf = [0u8; 1];
        assert_eq!(test_read(&mut state, &mut buf), 0);
        assert_eq!(state.pos, 3);
        assert!(!state.seek);
        assert_eq!(state.err, Z_OK);
        Ok(())
    }

    #[test]
    fn pending_write_seek_rejects_negative_offset_without_compression() {
        let mut state = GzState::new();
        state.mode = GzMode::Write;
        state.seek = true;
        state.skip = -1;
        assert_eq!(gz_zero_pending(&mut state), Err(()));
        assert_eq!(state.err, Z_STREAM_ERROR);
        assert_eq!(state.pos, 0);
        assert!(state.input.is_none());

        state.err = Z_OK;
        state.seek = true;
        state.skip = 0;
        assert_eq!(gz_zero_pending(&mut state), Ok(()));
        assert_eq!(state.pos, 0);
        assert_eq!(state.err, Z_OK);
    }

    #[test]
    fn invalid_gzip_level_is_ignored_without_mutating_state() {
        let mut state = GzState::new();
        state.mode = GzMode::Write;
        let pointer: gzFile = core::ptr::from_mut(&mut state).cast();
        let strategy = Strategy::Default as c_int;

        assert_eq!(
            // SAFETY: gzsetparams checks a null handle and returns before accessing state,
            // even when the supplied compression level is invalid.
            unsafe { gzsetparams(ptr::null_mut(), c_int::MAX, strategy) },
            Z_STREAM_ERROR
        );

        // SAFETY: state is a live, aligned, exclusively borrowed gzip handle,
        // and it does not own any compression buffer yet.
        assert_eq!(unsafe { gzsetparams(pointer, c_int::MAX, strategy) }, Z_OK);
        assert_eq!(state.level, -1);
        // SAFETY: the same state remains live and has not been consumed.
        assert_eq!(unsafe { gzsetparams(pointer, 10, strategy) }, Z_OK);
        assert_eq!(state.level, -1);
        assert_eq!(
            // SAFETY: invalid strategy is checked before inspecting the live state
            // exclusively owned by this test.
            unsafe { gzsetparams(pointer, 9, c_int::MAX) },
            Z_STREAM_ERROR
        );
        assert_eq!(state.level, -1);
        // SAFETY: the handle is still live and uninitialized, so this only
        // updates the configuration, without attempting compression.
        assert_eq!(unsafe { gzsetparams(pointer, 9, strategy) }, Z_OK);
        assert_eq!(state.level, 9);
    }

    #[test]
    fn owned_buffers_encode_mode_specific_geometry() -> Result<(), ()> {
        let mut state = GzState::new();
        state.want = 64;

        state.mode = GzMode::Read;
        state.input = Some(state.allocate_input_buffer().ok_or(())?);
        state.output = Some(state.allocate_output_buffer().ok_or(())?);
        assert_eq!(state.in_size(), 64);
        assert_eq!(state.input.as_ref().ok_or(())?.capacity, 64);
        assert_eq!(state.output.as_ref().map_or(0, GzBuffer::usable), 128);
        free_buffers(&mut state);
        assert!(state.input.is_none());
        assert!(state.output.is_none());

        state.mode = GzMode::Write;
        state.input = Some(state.allocate_input_buffer().ok_or(())?);
        state.output = Some(state.allocate_output_buffer().ok_or(())?);
        assert_eq!(state.in_size(), 64);
        assert_eq!(state.input.as_ref().ok_or(())?.capacity, 128);
        assert_eq!(state.output.as_ref().map_or(0, GzBuffer::usable), 64);
        Ok(())
    }

    #[test]
    fn gzip_single_byte_operations_reject_position_overflow_before_mutation() -> Result<(), ()> {
        let mut state = GzState::new();
        state.mode = GzMode::Write;
        state.want = 8;
        state.input = state.allocate_input_buffer();
        assert!(state.input.is_some());
        state.pos = i64::MAX;
        assert_eq!(test_putc(&mut state, 42), -1);
        assert_eq!(state.pos, i64::MAX);
        assert_eq!(state.stream.avail_in, 0);

        state.mode = GzMode::Read;
        state.how = How::Gzip;
        state.output = state.allocate_output_buffer();
        state.output.as_mut().ok_or(())?.usable_uninit_mut()[0].write(67);
        state.publish_buffered_output(0, 1);
        gz_error(&mut state, None);
        assert_eq!(test_getc(&mut state), -1);
        assert_eq!(state.have, 0);
        assert_eq!(state.pos, i64::MAX);
        assert_eq!(state.err, Z_STREAM_ERROR);

        // Reenable the exposed byte and exercise the underflow path itself.
        gz_error(&mut state, None);
        state.publish_buffered_output(0, 1);
        state.pos = i64::MIN;
        assert_eq!(test_ungetc(&mut state, 68), -1);
        assert_eq!(state.have, 0);
        assert_eq!(state.pos, i64::MIN);
        assert_eq!(state.err, Z_STREAM_ERROR);

        // Pushback from position zero to -1 remains valid.
        state.pos = 0;
        gz_error(&mut state, None);
        state.publish_buffered_output(0, 1);
        assert_eq!(test_ungetc(&mut state, 68), 68);
        assert_eq!(state.pos, -1);
        assert_eq!(test_getc(&mut state), 68);
        assert_eq!(state.pos, 0);

        // Return the representable prefix before reporting overflow.
        state.pos = i64::MAX - 1;
        let mut output = [0u8; 2];
        assert_eq!(test_read(&mut state, &mut output), 1);
        assert_eq!(output[0], 67);
        assert_eq!(state.pos, i64::MAX);
        assert_eq!(state.err, Z_STREAM_ERROR);
        Ok(())
    }

    #[test]
    fn pending_zero_fill_initializes_write_buffers_before_access() {
        let mut state = GzState::new();
        state.mode = GzMode::Write;
        state.want = 64;
        // Public gzip APIs clear the lazy seek flag before flushing zeroes.
        state.seek = false;
        state.skip = 12;
        assert!(state.input.is_none());
        assert_eq!(gz_zero_pending(&mut state), Ok(()));
        assert_eq!(state.pos, 12);
        assert!(state.input.is_some());
        assert!(state.output.is_some());
        // SAFETY: successful zero-fill initialized the live deflater. This
        // fixture intentionally abandons the unfinished stream; deflateEnd
        // frees it and reports Z_DATA_ERROR for the incomplete output.
        assert_eq!(unsafe { deflateEnd(&raw mut state.stream) }, Z_DATA_ERROR);
    }

    #[test]
    fn pending_seek_before_first_write_survives_all_zero_fill_entrypoints(
    ) -> Result<(), &'static str> {
        #[derive(Clone, Copy)]
        enum Trigger {
            ChangeParams,
            Flush,
            Putc,
            Close,
        }
        use Trigger::{ChangeParams, Close, Flush, Putc};

        let write_mode = CString::new("wb").map_err(|_| "invalid write mode")?;
        let direct_mode = CString::new("wT").map_err(|_| "invalid direct mode")?;
        let read_mode = CString::new("rb").map_err(|_| "invalid read mode")?;
        let cases = [
            (ChangeParams, false),
            (Flush, false),
            (Putc, false),
            (Close, false),
            (Flush, true),
            (Putc, true),
            (Close, true),
        ];
        for (index, (operation, direct)) in cases.into_iter().enumerate() {
            let path = std::env::temp_dir().join(format!(
                "zlib-rs-fast-lazy-zero-{}-{index}.gz",
                std::process::id()
            ));
            let path_c =
                CString::new(path.to_string_lossy().as_bytes()).map_err(|_| "invalid test path")?;
            let mode = if direct { &direct_mode } else { &write_mode };
            // SAFETY: path and selected mode are live NUL-terminated strings.
            let writer = unsafe { gzopen(path_c.as_ptr(), mode.as_ptr()) };
            assert!(!writer.is_null());
            // SAFETY: this exclusive handle is live; gzbuffer runs before any
            // allocation, and gzseek only schedules the forward distance.
            assert_eq!(unsafe { gzbuffer(writer, 64) }, 0);
            // SAFETY: writer remains a live exclusive gzip handle after gzbuffer.
            assert_eq!(unsafe { gzseek64(writer, 12, SEEK_SET) }, 12);

            match operation {
                ChangeParams => {
                    // SAFETY: the opened writer remains exclusive throughout.
                    let ret = unsafe { gzsetparams(writer, 9, Strategy::Default as c_int) };
                    assert_eq!(ret, Z_OK);
                }
                Flush => {
                    // SAFETY: the opened writer remains exclusive throughout.
                    assert_eq!(unsafe { gzflush(writer, Z_SYNC_FLUSH) }, Z_OK);
                }
                Putc => {
                    // SAFETY: the opened writer remains exclusive throughout.
                    let written = unsafe { gzputc(writer, c_int::from(b'Z')) };
                    assert_eq!(written, c_int::from(b'Z'));
                }
                Close => {
                    // Closing itself must materialize the pending zeroes.
                }
            }
            if matches!(operation, ChangeParams | Flush) {
                // SAFETY: writer is live and the source byte remains valid.
                assert_eq!(unsafe { gzwrite(writer, b"Z".as_ptr().cast(), 1) }, 1);
            }
            // SAFETY: gzclose consumes the live writer handle.
            assert_eq!(unsafe { gzclose(writer) }, Z_OK);

            // SAFETY: test path and read mode are live C strings.
            let reader = unsafe { gzopen(path_c.as_ptr(), read_mode.as_ptr()) };
            assert!(!reader.is_null());
            let mut bytes = [0u8; 14];
            // SAFETY: the reader is live and the destination is an exclusive
            // writable 14-byte buffer.
            let got = unsafe { gzread(reader, bytes.as_mut_ptr().cast(), 14) };
            let want = if matches!(operation, Close) { 12 } else { 13 };
            assert_eq!(got, want);
            assert_eq!(&bytes[..12], &[0u8; 12]);
            if !matches!(operation, Close) {
                assert_eq!(bytes[12], b'Z');
            }
            // SAFETY: gzclose consumes the live reader handle.
            assert_eq!(unsafe { gzclose(reader) }, Z_OK);
            std::fs::remove_file(path).map_err(|_| "failed to remove temporary gzip file")?;
        }
        Ok(())
    }

    #[test]
    fn gzip_multi_byte_write_and_zero_fill_check_position_before_mutation() {
        let mut writer = GzState::new();
        writer.mode = GzMode::Write;
        writer.pos = i64::MAX - 2;
        assert_eq!(test_write(&mut writer, b"abc"), 0);
        assert_eq!(writer.pos, i64::MAX - 2);
        assert!(writer.input.is_none());
        assert_eq!(writer.err, Z_STREAM_ERROR);

        // A pending seek to i64::MAX must not trigger a massive zero-fill
        // before rejecting a following write.
        let mut pending = GzState::new();
        pending.mode = GzMode::Write;
        pending.pos = i64::MAX - 2;
        pending.seek = true;
        pending.skip = 2;
        assert_eq!(test_write(&mut pending, b"a"), 0);
        assert_eq!(pending.pos, i64::MAX - 2);
        assert_eq!(pending.skip, 2);
        assert!(pending.seek);
        assert!(pending.input.is_none());

        // Direct zero-fill also validates before initializing compressors.
        let mut zeros = GzState::new();
        zeros.mode = GzMode::Write;
        zeros.pos = i64::MAX - 1;
        assert_eq!(gz_zero(&mut zeros, 2), Err(()));
        assert_eq!(zeros.pos, i64::MAX - 1);
        assert!(zeros.input.is_none());
    }

    #[test]
    fn gzgets_returns_terminated_prefix_if_subsequent_fetch_fails() -> Result<(), ()> {
        let mut state = GzState::new();
        state.mode = GzMode::Read;
        state.how = How::Copy;
        state.want = 8;
        state.fd = -2; // A later attempt to fetch from this descriptor fails with EBADF.
        state.output = Some(state.allocate_output_buffer().ok_or(())?);
        let output = state.output.as_mut().ok_or(())?;
        output.usable_uninit_mut()[..3].write_copy_of_slice(b"abc");
        state.publish_buffered_output(0, 3);

        let mut line = [0xa5u8; 12];
        let got = test_gets(&mut state, &mut line);
        assert_eq!(
            got,
            line.as_mut_ptr().cast(),
            "a partial line must be returned"
        );
        assert_eq!(&line[..4], b"abc\0");
        assert_eq!(line[4], 0xa5);
        assert_eq!(state.pos, 3);
        assert_eq!(state.err, Z_ERRNO);

        // By contrast, a failure without any data does return NULL, and does
        // not even touch the caller's uninitialized line destination.
        gz_error(&mut state, None);
        state.pos = 0;
        line.fill(0xa5);
        let got = test_gets(&mut state, &mut line);
        assert!(got.is_null());
        assert_eq!(line, [0xa5; 12]);
        assert_eq!(state.err, Z_ERRNO);
        Ok(())
    }

    #[test]
    fn invalid_gzip_input_cursor_fails_before_refill_or_read() -> Result<(), ()> {
        for before in [true, false] {
            let mut state = GzState::new();
            state.mode = GzMode::Read;
            state.want = 8;
            state.input = state.allocate_input_buffer();
            let input = state.input.as_ref().ok_or(())?;
            let base = input.as_ptr();
            state.stream.next_in = if before {
                base.wrapping_sub(1)
            } else {
                base.wrapping_add(input.usable() + 1)
            };
            state.stream.avail_in = 1;
            assert!(state.read_input_range().is_none());
            assert!(state.buffered_read_input().is_none());
            assert_eq!(state.compact_read_input(), None);
            assert_eq!(gz_avail(&mut state), Err(()));
            assert_eq!(state.err, Z_STREAM_ERROR);
        }

        let mut state = GzState::new();
        state.mode = GzMode::Read;
        state.want = 8;
        state.input = state.allocate_input_buffer();
        assert!(!state.publish_read_input(1, state.in_size()));
        let input = state.input.as_mut().ok_or(())?;
        input.usable_uninit_mut()[..4].write_copy_of_slice(b"abcd");
        assert!(state.publish_read_input(0, 4));
        let base = state.input.as_ref().ok_or(())?.as_ptr();
        // SAFETY: the input allocation has at least four initialized bytes.
        state.stream.next_in = unsafe { base.add(1) };
        state.stream.avail_in = 3;
        assert_eq!(state.buffered_read_input(), Some(b"bcd".as_slice()));
        assert_eq!(state.compact_read_input(), Some(3));
        assert_eq!(state.buffered_read_input(), Some(b"bcd".as_slice()));
        Ok(())
    }

    #[test]
    fn invalid_exposed_read_cursors_fail_without_unsafe_pointer_subtraction() -> Result<(), ()> {
        // A gzFile exposes the cursor fields to C's inline gzgetc macro.
        // Construct deliberately inconsistent prefixes without dereferencing them:
        // the Rust core must reject them rather than calling offset_from or
        // forming a slice from an unrelated cursor.
        for before in [true, false] {
            let mut state = GzState::new();
            state.mode = GzMode::Read;
            state.want = 8;
            state.how = How::Gzip;
            state.output = state.allocate_output_buffer();
            let output = state.output.as_mut().ok_or(())?;
            output.usable_uninit_mut()[0].write(123);
            let base = output.as_ptr();
            let invalid = if before {
                base.wrapping_sub(1)
            } else {
                base.wrapping_add(output.usable() + 1)
            };
            state.have = 1;
            state.next = invalid;
            assert!(state.buffered_output().is_none());
            assert!(!state.consume_buffered_output(1));
            assert_eq!(
                state.push_back_buffered_byte(9),
                Err(PushBackError::InvalidCursor)
            );
            assert_eq!(test_ungetc(&mut state, 9), -1);
            assert_eq!(state.err, Z_STREAM_ERROR);
            // Restore the corrupted prefix to independently exercise getc.
            gz_error(&mut state, None);
            state.have = 1;
            state.next = invalid;

            // A real C-exported entry point must signal a normal stream error.
            assert_eq!(test_getc(&mut state), -1);
            assert_eq!(state.err, Z_STREAM_ERROR);
            assert_eq!(state.have, 0);
        }

        // A valid C macro advance still identifies the correct owned range.
        let mut state = GzState::new();
        state.mode = GzMode::Read;
        state.want = 8;
        state.output = state.allocate_output_buffer();
        let output = state.output.as_mut().ok_or(())?;
        output.usable_uninit_mut()[..3].write_copy_of_slice(b"abc");
        state.publish_buffered_output(0, 3);
        assert_eq!(state.buffered_output(), Some(b"abc".as_slice()));
        assert!(state.consume_buffered_output(1));
        assert_eq!(state.buffered_output(), Some(b"bc".as_slice()));
        assert_eq!(state.push_back_buffered_byte(b'x'), Ok(()));
        assert_eq!(state.buffered_output(), Some(b"xbc".as_slice()));
        Ok(())
    }

    #[test]
    fn gzip_item_transfer_rejects_unrepresentable_raw_spans() {
        for mode in [GzMode::Read, GzMode::Write] {
            let mut state = GzState::new();
            state.mode = mode;
            let too_long = (isize::MAX.unsigned_abs()) + 1;
            let oversized_products = [
                (too_long, 1usize),
                (too_long / 2, 3usize),
                (usize::MAX, 1usize),
                (usize::MAX, 2usize),
            ];
            for (size, nitems) in oversized_products {
                gz_error(&mut state, None);
                assert_eq!(checked_gz_transfer_len(&mut state, size, nitems), None);
                assert_eq!(state.err, Z_STREAM_ERROR);
            }
            gz_error(&mut state, None);
            assert_eq!(
                checked_gz_transfer_len(&mut state, isize::MAX.unsigned_abs(), 1),
                Some(isize::MAX.unsigned_abs())
            );
            assert_eq!(checked_gz_transfer_len(&mut state, usize::MAX, 0), Some(0));
            assert_eq!(checked_gz_transfer_len(&mut state, 0, usize::MAX), Some(0));
            assert_eq!(state.err, Z_OK);
        }
    }

    #[test]
    fn gzip_item_ffi_rejects_oversized_spans_before_touching_caller_data() {
        let mut state = GzState::new();
        state.mode = GzMode::Read;
        let mut data = [0u8; 1];
        let too_long = (isize::MAX.unsigned_abs()) + 1;
        // SAFETY: the FFI operation rejects the invalid length before forming
        // any slice or accessing data. The byte array is a live valid pointer.
        let read = unsafe {
            gzfread(
                data.as_mut_ptr().cast(),
                too_long,
                1,
                core::ptr::from_mut(&mut state).cast(),
            )
        };
        assert_eq!(read, 0);
        assert_eq!(state.err, Z_STREAM_ERROR);
        assert!(state.input.is_none());
        assert!(state.output.is_none());

        state.mode = GzMode::Write;
        gz_error(&mut state, None);
        // SAFETY: the positive invalid span is rejected before a raw input
        // cursor is formed; no bytes are read from the byte array.
        let written = unsafe {
            gzfwrite(
                data.as_ptr().cast(),
                too_long,
                1,
                core::ptr::from_mut(&mut state).cast(),
            )
        };
        assert_eq!(written, 0);
        assert_eq!(state.err, Z_STREAM_ERROR);
        assert!(state.input.is_none());
        assert!(state.output.is_none());
    }

    #[test]
    fn gzip_terminal_reads_handle_signed_position_limits() -> Result<(), ()> {
        // Reaching EOF at the maximum position must not raise a false error.
        let mut empty = GzState::new();
        empty.mode = GzMode::Read;
        empty.how = How::Copy;
        empty.pos = i64::MAX;
        empty.eof = true;
        assert_eq!(test_read(&mut empty, &mut [0u8; 1]), 0);
        assert_eq!(empty.err, Z_OK);
        assert!(empty.past);

        // A line read cannot consume a buffered byte past i64::MAX.
        let mut state = GzState::new();
        state.mode = GzMode::Read;
        state.how = How::Gzip;
        state.want = 8;
        state.pos = i64::MAX;
        state.output = state.allocate_output_buffer();
        state.output.as_mut().ok_or(())?.usable_uninit_mut()[0].write(b'A');
        state.publish_buffered_output(0, 1);
        assert!(test_gets(&mut state, &mut [0u8; 3]).is_null());
        assert_eq!(state.have, 0);
        assert_eq!(state.pos, i64::MAX);
        assert_eq!(state.err, Z_STREAM_ERROR);
        Ok(())
    }

    #[test]
    fn gzip_seek_checks_signed_bounds_without_losing_pending_offsets() {
        // Use a live internal state without allocating or writing the huge
        // virtual skip. The C ABI paths may be called before the first write.
        let mut state = GzState::new();
        state.mode = GzMode::Write;
        assert_eq!(test_seek(&mut state, i64::MAX - 3, SEEK_SET), i64::MAX - 3);
        assert_eq!(test_tell(&state), i64::MAX - 3);
        assert_eq!(state.skip, i64::MAX - 3);
        assert!(state.seek);

        // A relative offset must not wrap the virtual position, nor discard
        // the already-pending seek when the new request is invalid.
        assert_eq!(test_seek(&mut state, 7, SEEK_CUR), -1);
        assert_eq!(test_tell(&state), i64::MAX - 3);
        assert!(state.seek);
        assert_eq!(state.skip, i64::MAX - 3);

        assert_eq!(test_seek(&mut state, -2, SEEK_CUR), i64::MAX - 5);
        assert_eq!(test_tell(&state), i64::MAX - 5);
        assert_eq!(test_seek(&mut state, i64::MIN, SEEK_SET), -1);
        assert_eq!(test_tell(&state), i64::MAX - 5);
        assert_eq!(test_seek(&mut state, 0, SEEK_SET), 0);
        assert!(!state.seek);
        assert_eq!(test_tell(&state), 0);
        assert_eq!(test_seek(&mut state, -1, SEEK_CUR), -1);
        assert_eq!(test_tell(&state), 0);

        // The same upper-bound invariant applies to the read-mode cursor;
        // this remains lazy and does not perform any filesystem access.
        state.mode = GzMode::Read;
        state.how = How::Gzip;
        state.pos = i64::MAX - 2;
        assert_eq!(test_seek(&mut state, 3, SEEK_CUR), -1);
        assert_eq!(test_tell(&state), i64::MAX - 2);
        assert_eq!(test_seek(&mut state, i64::MAX, SEEK_SET), i64::MAX);
        assert_eq!(test_tell(&state), i64::MAX);
        assert_eq!(test_seek(&mut state, -1, SEEK_CUR), i64::MAX - 1);
        assert_eq!(test_tell(&state), i64::MAX - 1);

        // A direct (uncompressed) read needs a real descriptor seek. When it
        // fails, the preceding lazy seek must remain available for a retry.
        state.mode = GzMode::Read;
        state.how = How::Copy;
        state.pos = 0;
        state.seek = true;
        state.skip = 20;
        state.fd = -1;
        assert_eq!(test_seek(&mut state, 1, SEEK_CUR), -1);
        assert!(state.seek);
        assert_eq!(state.skip, 20);
        assert_eq!(test_tell(&state), 20);

        // Even a corrupted internal cursor cannot make the public C getter
        // panic through signed overflow.
        state.pos = i64::MAX;
        state.skip = 1;
        assert_eq!(test_tell(&state), -1);
    }

    #[test]
    fn gzip_allocations_enforce_c_counter_geometry() {
        assert!(GzBuffer::allocate(0, 0).is_none());
        assert!(GzBuffer::allocate(8, 9).is_none());

        // The shared constructor must reject an invalid Rust allocation
        // extent even on 32-bit targets where it fits in an unsigned C int.
        let beyond_rust = isize::MAX.unsigned_abs() + 1;
        assert_eq!(GzBuffer::checked_abi_len(beyond_rust), None);
        assert!(GzBuffer::allocate(beyond_rust, 1).is_none());
        let biggest_legal = isize::MAX.unsigned_abs().min(c_uint::MAX as usize);
        assert_eq!(
            GzBuffer::checked_abi_len(biggest_legal),
            c_uint::try_from(biggest_legal).ok()
        );
        assert!(GzState::doubled_buffer_capacity(isize::MAX.unsigned_abs() / 2 + 1).is_none());

        let Ok(too_wide) = usize::try_from((c_uint::MAX / 2) + 1) else {
            return;
        };
        let mut state = GzState::new();
        state.want = too_wide;
        for mode in [GzMode::Read, GzMode::Write] {
            state.mode = mode;
            assert!(state.in_capacity().is_none());
            assert!(state.out_capacity().is_none());
            assert!(state.allocate_input_buffer().is_none());
            assert!(state.allocate_output_buffer().is_none());
        }
        // A representable request retains the mode-specific doubling.
        state.want = 32;
        state.mode = GzMode::Read;
        assert_eq!(state.in_capacity(), Some(32));
        assert_eq!(state.out_capacity(), Some(64));
        state.mode = GzMode::Write;
        assert_eq!(state.in_capacity(), Some(64));
        assert_eq!(state.out_capacity(), Some(32));
    }

    #[test]
    fn large_gzip_read_keeps_abi_cursor_in_owned_output() -> Result<(), ()> {
        let path = std::env::temp_dir().join(format!(
            "zlib-rs-fast-read-cursor-{}.gz",
            std::process::id()
        ));
        let path_c = CString::new(path.to_string_lossy().as_bytes()).map_err(|_| ())?;
        let input = vec![b'A'; 8192];

        let write_mode = CString::new("w").map_err(|_| ())?;
        // SAFETY: the supplied path/mode are either valid NUL-terminated C strings or intentionally null as accepted and checked by gzopen.
        let writer = unsafe { gzopen(path_c.as_ptr(), write_mode.as_ptr()) };
        assert!(!writer.is_null());
        assert_eq!(
            // SAFETY: writer was successfully opened, and input is an initialized readable slice with exactly the length passed to gzwrite.
            unsafe { gzwrite(writer, input.as_ptr().cast(), input.len() as c_uint) },
            input.len() as c_int
        );
        // SAFETY: the handle was returned by gzopen or gzdopen and has not yet been closed; this call consumes the live handle.
        assert_eq!(unsafe { gzclose(writer) }, Z_OK);

        let read_mode = CString::new("r").map_err(|_| ())?;
        // SAFETY: the supplied path/mode are either valid NUL-terminated C strings or intentionally null as accepted and checked by gzopen.
        let reader = unsafe { gzopen(path_c.as_ptr(), read_mode.as_ptr()) };
        assert!(!reader.is_null());
        // SAFETY: the gzip handle is live and exclusively held when non-null; the C API explicitly accepts null and reports an error.
        assert_eq!(unsafe { gzbuffer(reader, 64) }, 0);

        let mut output = vec![0u8; 4096];
        assert_eq!(
            // SAFETY: reader was successfully opened, and output is a writable slice covering all bytes offered to gzread.
            unsafe { gzread(reader, output.as_mut_ptr().cast(), output.len() as c_uint) },
            output.len() as c_int
        );
        assert!(output.iter().all(|&byte| byte == b'A'));

        // SAFETY: the opened gzip handle points to a live, aligned GzState, and this temporary immutable inspection cannot outlive the handle.
        let state = unsafe { reader.cast::<GzState>().as_ref() }.ok_or(())?;
        assert_eq!(state.have, 0);
        let owned = state.output.as_ref().ok_or(())?;
        let owned_start = owned.as_ptr() as usize;
        let owned_end = owned_start + owned.usable();
        let next = state.next as usize;
        assert!((owned_start..=owned_end).contains(&next));

        let caller_start = output.as_ptr() as usize;
        let caller_end = caller_start + output.len();
        assert!(!(caller_start..caller_end).contains(&next));

        // SAFETY: the handle was returned by gzopen or gzdopen and has not yet been closed; this call consumes the live handle.
        assert_eq!(unsafe { gzclose(reader) }, Z_OK);
        std::fs::remove_file(path).map_err(|_| ())?;
        Ok(())
    }

    #[test]
    fn large_write_restores_owned_input_cursor_after_zero_copy() {
        let mut state = GzState::new();
        state.mode = GzMode::Write;
        state.fd = -2;

        assert!(gz_init(&mut state).is_ok());
        let owned_input = state.input_ptr();
        let saved_next_in = state.stream.next_in;
        assert_ne!(owned_input.cast_const(), saved_next_in);

        let input = vec![b'A'; state.in_size()];
        let mut input_cursor = GzWriteInput::from_slice(&input);
        let written = gz_write(&mut state, &mut input_cursor);
        assert_eq!(written as usize, input.len());
        assert_eq!(state.stream.next_in, saved_next_in);
        assert_ne!(state.stream.next_in, input.as_ptr().cast_mut());

        // Safety: `gz_init` initialized the deflate stream above. Ending without
        // `Z_FINISH` reports `Z_DATA_ERROR`, but still releases the deflate state.
        assert_eq!(unsafe { deflateEnd(&raw mut state.stream) }, Z_DATA_ERROR);
    }

    #[test]
    fn invalid_owned_write_input_cursor_fails_without_panic() -> Result<(), ()> {
        for after in [false, true] {
            let mut state = GzState::new();
            state.mode = GzMode::Write;
            state.fd = -2;
            assert_eq!(gz_init(&mut state), Ok(()));
            let input = state.input.as_ref().ok_or(())?;
            state.stream.next_in = if after {
                input.as_ptr().wrapping_add(input.capacity + 1)
            } else {
                input.as_ptr().wrapping_sub(1)
            };
            state.stream.avail_in = 1;
            assert_eq!(state.owned_write_input_end(), None);
            let mut source = GzWriteInput::from_slice(b"a");
            assert_eq!(gz_write(&mut state, &mut source), 0);
            assert_eq!(state.err, Z_STREAM_ERROR);
            // SAFETY: gz_init initialized the compression stream.
            assert_eq!(unsafe { deflateEnd(&raw mut state.stream) }, Z_OK);
        }
        Ok(())
    }

    #[test]
    fn invalid_owned_write_output_cursors_fail_before_deflate() -> Result<(), ()> {
        for wrong_next in [true, false] {
            let mut state = GzState::new();
            state.mode = GzMode::Write;
            state.fd = -2;
            assert_eq!(gz_init(&mut state), Ok(()));
            let output = state.output.as_ref().ok_or(())?;
            if wrong_next {
                state.next = output.as_ptr().wrapping_sub(1);
            } else {
                state.stream.next_out = output.as_ptr().wrapping_add(output.usable() + 1);
                state.stream.avail_out = 0;
            }
            assert!(state.pending_write_output().is_none());
            assert_eq!(gz_comp(&mut state, Z_NO_FLUSH), Err(()));
            assert_eq!(state.err, Z_STREAM_ERROR);
            // SAFETY: gz_init initialized the compression stream.
            assert_eq!(unsafe { deflateEnd(&raw mut state.stream) }, Z_OK);
        }
        Ok(())
    }

    #[test]
    fn empty_direct_flush_never_arithmetics_on_null_input_cursor() {
        let mut state = GzState::new();
        state.mode = GzMode::Write;
        state.direct = true;
        state.fd = -2;
        assert_eq!(gz_comp(&mut state, Z_NO_FLUSH), Ok(()));
        assert!(state.stream.next_in.is_null());
        assert_eq!(state.stream.avail_in, 0);
    }

    #[test]
    fn large_write_error_does_not_retain_caller_input() {
        let mut state = GzState::new();
        state.mode = GzMode::Write;
        state.direct = true;
        state.fd = -2;

        assert!(gz_init(&mut state).is_ok());
        let saved_next_in = state.stream.next_in;
        let input = vec![b'A'; state.in_size()];
        let caller_start = input.as_ptr() as usize;
        let caller_end = caller_start + input.len();

        let mut input_cursor = GzWriteInput::from_slice(&input);
        assert_eq!(gz_write(&mut state, &mut input_cursor), 0);
        assert_eq!(state.err, Z_ERRNO);
        assert_eq!(state.stream.avail_in, 0);
        assert_eq!(state.stream.next_in, saved_next_in);
        assert!(!(caller_start..caller_end).contains(&(state.stream.next_in as usize)));
    }

    // Map a byte string literal to a C string
    // FIXME: switch to c"example" format once MSRV >= 1.77
    macro_rules! c {
        ($s:literal) => {{
            $s.as_ptr().cast::<c_char>()
        }};
    }

    #[test]
    fn gzdopen_invalid_fd() {
        // SAFETY: the mode pointer is a valid, NUL-terminated C string; the descriptor is passed as an integer and may intentionally be invalid.
        assert_eq!(unsafe { gzdopen(-1, c!(b"r\0")) }, core::ptr::null_mut())
    }

    #[test]
    fn gzopen_path_null() {
        assert_eq!(
            // SAFETY: gzopen rejects a null path or mode without reading through that null pointer.
            unsafe { gzopen(core::ptr::null(), c!(b"r\0")) },
            core::ptr::null_mut()
        )
    }

    #[test]
    fn gzopen_mode_null() {
        assert_eq!(
            // SAFETY: gzopen rejects a null path or mode without reading through that null pointer.
            unsafe { gzopen(c!(b"/foo/bar\0"), core::ptr::null(),) },
            core::ptr::null_mut()
        )
    }

    #[test]
    fn owned_c_string_clones_and_concatenates_bytes() -> Result<(), ()> {
        let empty = GzCString::clone_from_c_str(c"").ok_or(())?;
        assert_eq!(empty.as_c_str().to_bytes_with_nul(), b"\0");

        let example = GzCString::clone_from_c_str(c"example").ok_or(())?;
        assert_eq!(example.as_c_str().to_bytes_with_nul(), b"example\0");

        let joined = GzCString::concat(&[b"hello", b"", b",", b"world"]).ok_or(())?;
        assert_eq!(joined.as_c_str().to_bytes_with_nul(), b"hello,world\0");
        Ok(())
    }

    #[test]
    fn owned_c_string_preserves_non_utf8_bytes() -> Result<(), ()> {
        let path = CStr::from_bytes_with_nul(b"bad-\xff-path\0").map_err(|_| ())?;
        let owned = GzCString::clone_from_c_str(path).ok_or(())?;
        assert_eq!(owned.as_c_str().to_bytes_with_nul(), b"bad-\xff-path\0");

        let message =
            GzCString::concat(&[owned.as_c_str().to_bytes(), b": ", b"read error"]).ok_or(())?;
        assert_eq!(
            message.as_c_str().to_bytes_with_nul(),
            b"bad-\xff-path: read error\0"
        );
        Ok(())
    }

    #[test]
    fn test_fd_path() {
        let mut buf = [0u8; 27];
        assert_eq!(fd_path(&mut buf, 0).to_bytes(), b"<fd:0>");
        assert_eq!(fd_path(&mut buf, 9).to_bytes(), b"<fd:9>");
        assert_eq!(fd_path(&mut buf, -1).to_bytes(), b"<fd:-1>");
        assert_eq!(
            fd_path(&mut buf, i32::MIN).to_bytes(),
            format!("<fd:{}>", i32::MIN).as_bytes(),
        );
    }

    #[test]
    #[cfg_attr(
        not(any(target_os = "linux", target_os = "macos")),
        ignore = "lseek is not implemented"
    )]
    fn test_gz_error() -> Result<(), ()> {
        // gzerror(null) should return null.
        // SAFETY: gzerror explicitly checks null file and errnum pointers and returns null.
        assert!(unsafe { gzerror(ptr::null_mut(), ptr::null_mut()) }.is_null());

        // Open a gzip stream with an invalid file handle. Initially, no error
        // status should be set.
        // SAFETY: the mode pointer is a valid, NUL-terminated C string; the descriptor is passed as an integer and may intentionally be invalid.
        let handle = unsafe { gzdopen(-2, c!(b"r\0")) };
        assert!(!handle.is_null());

        // SAFETY: the open gzip handle owns a live, aligned GzState; this test has exclusive access to it and checks for null via ok_or.
        let state = (unsafe { handle.cast::<GzState>().as_mut() }).ok_or(())?;
        assert_eq!(state.err, Z_OK);
        assert!(state.msg.is_none());
        let mut err = Z_ERRNO;
        // SAFETY: the handle is live and the optional err pointer is an aligned writable local integer; the null/null case is explicitly supported.
        let msg = unsafe { gzerror(handle, &raw mut err) };
        // SAFETY: gzerror returned a borrowed pointer to a live, NUL-terminated state or static error message; the handle has not been closed.
        assert_eq!(unsafe { CStr::from_ptr(msg) }.to_bytes_with_nul(), b"\0");
        assert_eq!(err, Z_OK);

        // When an error is set, the path should be prepended to the error message automatically.
        // SAFETY: the open gzip handle owns a live, aligned GzState; this test has exclusive access to it and checks for null via ok_or.
        let state = (unsafe { handle.cast::<GzState>().as_mut() }).ok_or(())?;
        gz_error(state, Some((Z_ERRNO, "example error")));
        assert_eq!(state.err, Z_ERRNO);
        assert_eq!(
            state.msg.as_ref().ok_or(())?.as_c_str().to_bytes_with_nul(),
            b"<fd:-2>: example error\0"
        );
        let mut err = Z_OK;
        // SAFETY: the handle is live and the optional err pointer is an aligned writable local integer; the null/null case is explicitly supported.
        let msg = unsafe { gzerror(handle, &raw mut err) };
        assert_eq!(
            // SAFETY: gzerror returned a borrowed pointer to a live, NUL-terminated state or static error message; the handle has not been closed.
            unsafe { CStr::from_ptr(msg) }.to_bytes_with_nul(),
            b"<fd:-2>: example error\0"
        );
        assert_eq!(err, Z_ERRNO);

        // Setting the error message to null should clear the old error message.
        // SAFETY: the open gzip handle owns a live, aligned GzState; this test has exclusive access to it and checks for null via ok_or.
        let state = (unsafe { handle.cast::<GzState>().as_mut() }).ok_or(())?;
        gz_error(state, None);
        assert_eq!(state.err, Z_OK);
        assert!(state.msg.is_none());
        let mut err = Z_ERRNO;
        // SAFETY: the handle is live and the optional err pointer is an aligned writable local integer; the null/null case is explicitly supported.
        let msg = unsafe { gzerror(handle, &raw mut err) };
        // SAFETY: gzerror returned a borrowed pointer to a live, NUL-terminated state or static error message; the handle has not been closed.
        assert_eq!(unsafe { CStr::from_ptr(msg) }.to_bytes_with_nul(), b"\0");
        assert_eq!(err, Z_OK);

        // Setting the error code to Z_MEM_ERROR should clear the internal error message
        // (because gz_error doesn't try to allocate space for a copy of the message if
        // the reason for the error is that allocations are failing).
        // SAFETY: the open gzip handle owns a live, aligned GzState; this test has exclusive access to it and checks for null via ok_or.
        let state = (unsafe { handle.cast::<GzState>().as_mut() }).ok_or(())?;
        gz_error(state, Some((Z_MEM_ERROR, "should be ignored")));
        assert_eq!(state.err, Z_MEM_ERROR);
        assert!(state.msg.is_none());
        let mut err = Z_OK;
        // SAFETY: the handle is live and the optional err pointer is an aligned writable local integer; the null/null case is explicitly supported.
        let msg = unsafe { gzerror(handle, &raw mut err) };
        assert_eq!(
            // SAFETY: gzerror returned a borrowed pointer to a live, NUL-terminated state or static error message; the handle has not been closed.
            unsafe { CStr::from_ptr(msg) }.to_bytes_with_nul(),
            b"out of memory\0"
        );
        assert_eq!(err, Z_MEM_ERROR);

        // gzclose should return an error because the fd is invalid.
        // SAFETY: the handle was returned by gzopen or gzdopen and has not yet been closed; this call consumes the live handle.
        assert_eq!(unsafe { gzclose(handle) }, Z_ERRNO);
        Ok(())
    }

    #[test]
    fn gz_error_preserves_non_utf8_path_bytes() -> Result<(), ()> {
        let path = CStr::from_bytes_with_nul(b"bad-\xff-path\0").map_err(|_| ())?;
        let mut state = GzState::new();
        state.mode = GzMode::Read;
        state.source = Some(Source::Path(GzCString::clone_from_c_str(path).ok_or(())?));

        gz_error(&mut state, Some((Z_ERRNO, "read error")));

        assert_eq!(state.err, Z_ERRNO);
        assert_eq!(
            state.msg.as_ref().ok_or(())?.as_c_str().to_bytes_with_nul(),
            b"bad-\xff-path: read error\0"
        );
        Ok(())
    }

    #[test]
    #[cfg_attr(
        not(any(target_os = "linux", target_os = "macos")),
        ignore = "lseek is not implemented"
    )]
    fn test_gzclearerr() -> Result<(), ()> {
        // gzclearerr on a null file handle should return quietly.
        // SAFETY: gzclearerr explicitly accepts a null handle and performs no dereference.
        unsafe { gzclearerr(ptr::null_mut()) };

        // Open a gzip stream with an invalid file handle. Initially, no error
        // status should be set.
        // SAFETY: the mode pointer is a valid, NUL-terminated C string; the descriptor is passed as an integer and may intentionally be invalid.
        let handle = unsafe { gzdopen(-2, c!(b"r\0")) };
        assert!(!handle.is_null());

        // gzclearerr should reset the eof and past flags.
        // SAFETY: the open gzip handle owns a live, aligned GzState; this test has exclusive access to it and checks for null via ok_or.
        unsafe { handle.cast::<GzState>().as_mut().ok_or(())?.eof = true };
        // SAFETY: the open gzip handle owns a live, aligned GzState; this test has exclusive access to it and checks for null via ok_or.
        unsafe { handle.cast::<GzState>().as_mut().ok_or(())?.past = true };
        // SAFETY: gzclearerr accepts null, or a live gzip state allocated by gzdopen and held exclusively for this call.
        unsafe { gzclearerr(handle) };
        // SAFETY: the opened gzip handle points to a live, aligned GzState, and this temporary immutable inspection cannot outlive the handle.
        assert!(!unsafe { handle.cast::<GzState>().as_ref().ok_or(())?.eof });
        // SAFETY: the opened gzip handle points to a live, aligned GzState, and this temporary immutable inspection cannot outlive the handle.
        assert!(!unsafe { handle.cast::<GzState>().as_ref().ok_or(())?.past });

        // Set an error flag and message.
        // SAFETY: the handle is live and uniquely borrowed as GzState for gz_error; the checked pointer conversion rejects null.
        unsafe {
            gz_error(
                handle.cast::<GzState>().as_mut().ok_or(())?,
                Some((Z_STREAM_ERROR, "example error")),
            )
        };
        let mut err = Z_OK;
        // SAFETY: the handle is live and the optional err pointer is an aligned writable local integer; the null/null case is explicitly supported.
        let msg = unsafe { gzerror(handle, &raw mut err) };
        assert_eq!(err, Z_STREAM_ERROR);
        assert_eq!(
            // SAFETY: gzerror returned a borrowed pointer to a live, NUL-terminated state or static error message; the handle has not been closed.
            unsafe { CStr::from_ptr(msg) }.to_bytes_with_nul(),
            b"<fd:-2>: example error\0"
        );

        // gzclearerr should clear the error flag and message.
        // SAFETY: gzclearerr accepts null, or a live gzip state allocated by gzdopen and held exclusively for this call.
        unsafe { gzclearerr(handle) };
        // SAFETY: the handle is live and the optional err pointer is an aligned writable local integer; the null/null case is explicitly supported.
        let msg = unsafe { gzerror(handle, &raw mut err) };
        assert_eq!(err, Z_OK);
        // SAFETY: gzerror returned a borrowed pointer to a live, NUL-terminated state or static error message; the handle has not been closed.
        assert_eq!(unsafe { CStr::from_ptr(msg) }.to_bytes_with_nul(), b"\0");

        // gzclose should return an error because the fd is invalid.
        // SAFETY: the handle was returned by gzopen or gzdopen and has not yet been closed; this call consumes the live handle.
        assert_eq!(unsafe { gzclose(handle) }, Z_ERRNO);

        // Test the write and append modes, where gzclearerr should not clear eof or past.
        for mode in [c!(b"w\0"), c!(b"a\0")] {
            // Open a gzip stream for write with an invalid file handle. gzeof should return 0.
            // SAFETY: the mode pointer is a valid, NUL-terminated C string; the descriptor is passed as an integer and may intentionally be invalid.
            let handle = unsafe { gzdopen(-2, mode) };
            assert!(!handle.is_null());
            // SAFETY: gzeof accepts null, or inspects a live handle allocated by gzdopen and not yet closed.
            assert_eq!(unsafe { gzeof(handle) }, 0);

            // gzclearerr should not reset the eof and past flags in write or append mode.
            // SAFETY: the open gzip handle owns a live, aligned GzState; this test has exclusive access to it and checks for null via ok_or.
            unsafe { handle.cast::<GzState>().as_mut().ok_or(())?.eof = true };
            // SAFETY: the open gzip handle owns a live, aligned GzState; this test has exclusive access to it and checks for null via ok_or.
            unsafe { handle.cast::<GzState>().as_mut().ok_or(())?.past = true };
            // SAFETY: gzclearerr accepts null, or a live gzip state allocated by gzdopen and held exclusively for this call.
            unsafe { gzclearerr(handle) };
            // SAFETY: the opened gzip handle points to a live, aligned GzState, and this temporary immutable inspection cannot outlive the handle.
            assert!(unsafe { handle.cast::<GzState>().as_ref().ok_or(())?.eof });
            // SAFETY: the opened gzip handle points to a live, aligned GzState, and this temporary immutable inspection cannot outlive the handle.
            assert!(unsafe { handle.cast::<GzState>().as_ref().ok_or(())?.past });

            // Set an error flag and message.
            // SAFETY: the handle is live and uniquely borrowed as GzState for gz_error; the checked pointer conversion rejects null.
            unsafe {
                gz_error(
                    handle.cast::<GzState>().as_mut().ok_or(())?,
                    Some((Z_STREAM_ERROR, "example error")),
                )
            };

            // gzclearerr should clear the error flag and message.
            // SAFETY: gzclearerr accepts null, or a live gzip state allocated by gzdopen and held exclusively for this call.
            unsafe { gzclearerr(handle) };
            // SAFETY: the handle is live and the optional err pointer is an aligned writable local integer; the null/null case is explicitly supported.
            let msg = unsafe { gzerror(handle, &raw mut err) };
            assert_eq!(err, Z_OK);
            // SAFETY: gzerror returned a borrowed pointer to a live, NUL-terminated state or static error message; the handle has not been closed.
            assert_eq!(unsafe { CStr::from_ptr(msg) }.to_bytes_with_nul(), b"\0");

            // gzclose should return an error because the fd is invalid.
            // SAFETY: the handle was returned by gzopen or gzdopen and has not yet been closed; this call consumes the live handle.
            assert_eq!(unsafe { gzclose(handle) }, Z_ERRNO);
        }
        Ok(())
    }

    #[test]
    #[cfg_attr(
        not(any(target_os = "linux", target_os = "macos")),
        ignore = "lseek is not implemented"
    )]
    fn test_gzeof() -> Result<(), ()> {
        // gzeof on a null file handle should return false.
        // SAFETY: gzeof explicitly checks a null handle and returns 0 without dereferencing it.
        assert_eq!(unsafe { gzeof(ptr::null_mut()) }, 0);

        // Open a gzip stream for read with an invalid file handle. gzeof should return 0.
        // SAFETY: the mode pointer is a valid, NUL-terminated C string; the descriptor is passed as an integer and may intentionally be invalid.
        let handle = unsafe { gzdopen(-2, c!(b"r\0")) };
        assert!(!handle.is_null());
        // SAFETY: gzeof accepts null, or inspects a live handle allocated by gzdopen and not yet closed.
        assert_eq!(unsafe { gzeof(handle) }, 0);

        // gzeof should return 1 only if there was a read attempt past the end of the stream.
        // SAFETY: the open gzip handle owns a live, aligned GzState; this test has exclusive access to it and checks for null via ok_or.
        unsafe { handle.cast::<GzState>().as_mut().ok_or(())?.eof = true };
        // SAFETY: gzeof accepts null, or inspects a live handle allocated by gzdopen and not yet closed.
        assert_eq!(unsafe { gzeof(handle) }, 0);
        // SAFETY: the open gzip handle owns a live, aligned GzState; this test has exclusive access to it and checks for null via ok_or.
        unsafe { handle.cast::<GzState>().as_mut().ok_or(())?.past = true };
        // SAFETY: gzeof accepts null, or inspects a live handle allocated by gzdopen and not yet closed.
        assert_eq!(unsafe { gzeof(handle) }, 1);

        // gzclose should return an error because the fd is invalid.
        // SAFETY: the handle was returned by gzopen or gzdopen and has not yet been closed; this call consumes the live handle.
        assert_eq!(unsafe { gzclose(handle) }, Z_ERRNO);

        // Test the write and append modes, where gzeof should always return 0.
        for mode in [c!(b"w\0"), c!(b"a\0")] {
            // Open a gzip stream for write with an invalid file handle. gzeof should return 0.
            // SAFETY: the mode pointer is a valid, NUL-terminated C string; the descriptor is passed as an integer and may intentionally be invalid.
            let handle = unsafe { gzdopen(-2, mode) };
            assert!(!handle.is_null());
            // SAFETY: gzeof accepts null, or inspects a live handle allocated by gzdopen and not yet closed.
            assert_eq!(unsafe { gzeof(handle) }, 0);

            // Even with the past flag set, gzeof should still return 0 in write or append mode.
            // SAFETY: the open gzip handle owns a live, aligned GzState; this test has exclusive access to it and checks for null via ok_or.
            unsafe { handle.cast::<GzState>().as_mut().ok_or(())?.past = true };
            // SAFETY: gzeof accepts null, or inspects a live handle allocated by gzdopen and not yet closed.
            assert_eq!(unsafe { gzeof(handle) }, 0);

            // gzclose should return an error because the fd is invalid.
            // SAFETY: the handle was returned by gzopen or gzdopen and has not yet been closed; this call consumes the live handle.
            assert_eq!(unsafe { gzclose(handle) }, Z_ERRNO);
        }
        Ok(())
    }

    #[test]
    #[cfg_attr(
        not(any(target_os = "linux", target_os = "macos")),
        ignore = "lseek is not implemented"
    )]
    // Open a gzip file for reading. gzdirect should return 0.
    fn test_gzdirect_gzip_file() -> Result<(), ()> {
        // SAFETY: both path and mode are NUL-terminated CString buffers that remain alive during this gzopen call.
        let file = unsafe {
            gzopen(
                CString::new(crate_path("src/test-data/example.gz"))
                    .map_err(|_| ())?
                    .as_ptr(),
                CString::new("r").map_err(|_| ())?.as_ptr(),
            )
        };
        assert!(!file.is_null());
        // Set a smaller read batch size to exercise the buffer management code paths.
        const FILE_SIZE: usize = 48; // size of test-data/example.gz
        const BLOCK_SIZE: usize = 40;
        // SAFETY: the open gzip handle owns a live, aligned GzState; this test has exclusive access to it and checks for null via ok_or.
        unsafe { file.cast::<GzState>().as_mut().ok_or(())?.want = BLOCK_SIZE };
        // SAFETY: gzdirect receives the live gzip handle opened immediately above, including its owned buffer state.
        assert_eq!(unsafe { gzdirect(file) }, 0);
        // gzdirect should have pulled the first `BLOCK_SIZE` bytes of the file into `file`'s internal input buffer.
        assert_eq!(
            // SAFETY: the opened gzip handle points to a live, aligned GzState, and this temporary immutable inspection cannot outlive the handle.
            unsafe { file.cast::<GzState>().as_ref().ok_or(())?.have },
            0
        );
        assert_eq!(
            // SAFETY: the opened gzip handle points to a live, aligned GzState, and this temporary immutable inspection cannot outlive the handle.
            unsafe { file.cast::<GzState>().as_ref().ok_or(())?.stream.avail_in },
            BLOCK_SIZE as uInt
        );
        // Consume some of the buffered input and call gz_avail. It should move the remaining
        // input to the front of the input buffer.
        // SAFETY: file points to the live owned GzState, and the buffered suffix is checked before advancing its input cursor within the allocated span.
        unsafe {
            let state = file.cast::<GzState>().as_mut().ok_or(())?;
            const CONSUME: usize = 10;
            let buffered = state.buffered_read_input().ok_or(())?;
            let expected_suffix = buffered[CONSUME..].to_vec();
            state.stream.next_in = state.stream.next_in.add(CONSUME);
            state.stream.avail_in -= CONSUME as uInt;
            assert_eq!(
                state.buffered_read_input(),
                Some(expected_suffix.as_slice())
            );

            let expected_avail = BLOCK_SIZE - CONSUME + (FILE_SIZE - BLOCK_SIZE);
            assert_eq!(gz_avail(state), Ok(expected_avail));
            assert_eq!(state.stream.avail_in as usize, expected_avail);
            assert_eq!(state.stream.next_in, state.input_ptr().cast_const());
            let buffered = state.buffered_read_input().ok_or(())?;
            assert_eq!(&buffered[..expected_suffix.len()], expected_suffix);
        };
        // SAFETY: the handle was returned by gzopen or gzdopen and has not yet been closed; this call consumes the live handle.
        assert_eq!(unsafe { gzclose(file) }, Z_OK);
        Ok(())
    }

    // Open a non-gzip file for reading. gzdirect should return 1.
    #[test]
    #[cfg_attr(
        not(any(target_os = "linux", target_os = "macos")),
        ignore = "lseek is not implemented"
    )]
    fn test_gzdirect_non_gzip_file() -> Result<(), ()> {
        // SAFETY: both path and mode are NUL-terminated CString buffers that remain alive during this gzopen call.
        let file = unsafe {
            gzopen(
                CString::new(crate_path("src/test-data/example.txt"))
                    .map_err(|_| ())?
                    .as_ptr(),
                CString::new("r").map_err(|_| ())?.as_ptr(),
            )
        };
        assert!(!file.is_null());
        // SAFETY: gzdirect receives the live gzip handle opened immediately above, including its owned buffer state.
        assert_eq!(unsafe { gzdirect(file) }, 1);
        // gzdirect should have pulled the entire contents of the file (which is smaller than
        // GZBUFSIZE) into `file`'s internal output buffer.
        assert_eq!(
            // SAFETY: the opened gzip handle points to a live, aligned GzState, and this temporary immutable inspection cannot outlive the handle.
            unsafe { file.cast::<GzState>().as_ref().ok_or(())?.have },
            20
        );
        assert_eq!(
            // SAFETY: the opened gzip handle points to a live, aligned GzState, and this temporary immutable inspection cannot outlive the handle.
            unsafe { file.cast::<GzState>().as_ref().ok_or(())?.stream.avail_in },
            0
        );
        // SAFETY: the handle was returned by gzopen or gzdopen and has not yet been closed; this call consumes the live handle.
        assert_eq!(unsafe { gzclose(file) }, Z_OK);

        // Open a file containing only the gzip magic number. gzdirect should return 0.
        // SAFETY: both path and mode are NUL-terminated CString buffers that remain alive during this gzopen call.
        let file = unsafe {
            gzopen(
                CString::new(crate_path("src/test-data/magic-only.gz"))
                    .map_err(|_| ())?
                    .as_ptr(),
                CString::new("r").map_err(|_| ())?.as_ptr(),
            )
        };
        assert!(!file.is_null());
        // SAFETY: gzdirect receives the live gzip handle opened immediately above, including its owned buffer state.
        assert_eq!(unsafe { gzdirect(file) }, 0);
        assert_eq!(
            // SAFETY: the opened gzip handle points to a live, aligned GzState, and this temporary immutable inspection cannot outlive the handle.
            unsafe { file.cast::<GzState>().as_ref().ok_or(())?.have },
            0
        );
        assert_eq!(
            // SAFETY: the opened gzip handle points to a live, aligned GzState, and this temporary immutable inspection cannot outlive the handle.
            unsafe { file.cast::<GzState>().as_ref().ok_or(())?.stream.avail_in },
            2
        );

        // SAFETY: the handle was returned by gzopen or gzdopen and has not yet been closed; this call consumes the live handle.
        assert_eq!(unsafe { gzclose(file) }, Z_OK);

        // Open a file containing only the first byte of the gzip magic number. gzdirect should return 1.
        // SAFETY: both path and mode are NUL-terminated CString buffers that remain alive during this gzopen call.
        let file = unsafe {
            gzopen(
                CString::new(crate_path("src/test-data/incomplete-magic.gz"))
                    .map_err(|_| ())?
                    .as_ptr(),
                CString::new("r").map_err(|_| ())?.as_ptr(),
            )
        };
        assert!(!file.is_null());
        // SAFETY: gzdirect receives the live gzip handle opened immediately above, including its owned buffer state.
        assert_eq!(unsafe { gzdirect(file) }, 1);
        assert_eq!(
            // SAFETY: the opened gzip handle points to a live, aligned GzState, and this temporary immutable inspection cannot outlive the handle.
            unsafe { file.cast::<GzState>().as_ref().ok_or(())?.have },
            1
        );
        assert_eq!(
            // SAFETY: the opened gzip handle points to a live, aligned GzState, and this temporary immutable inspection cannot outlive the handle.
            unsafe { file.cast::<GzState>().as_ref().ok_or(())?.stream.avail_in },
            0
        );
        // SAFETY: the handle was returned by gzopen or gzdopen and has not yet been closed; this call consumes the live handle.
        assert_eq!(unsafe { gzclose(file) }, Z_OK);
        Ok(())
    }

    #[test]
    #[cfg_attr(
        not(any(target_os = "linux", target_os = "macos")),
        ignore = "lseek is not implemented"
    )]
    fn test_gzbuffer() -> Result<(), ()> {
        // gzbuffer on a null file handle should return -1.
        // SAFETY: gzbuffer explicitly checks a null handle and returns -1 without dereferencing it.
        assert_eq!(unsafe { gzbuffer(ptr::null_mut(), 1024) }, -1);

        // Open a valid file handle to test the remaining gzbuffer edge cases.
        // SAFETY: both path and mode are NUL-terminated CString buffers that remain alive during this gzopen call.
        let file = unsafe {
            gzopen(
                CString::new(crate_path("src/test-data/example.txt"))
                    .map_err(|_| ())?
                    .as_ptr(),
                CString::new("r").map_err(|_| ())?.as_ptr(),
            )
        };
        // Temporarily put the file handle in a stat that isn't read or write. gzbuffer should fail.
        // SAFETY: the open gzip handle owns a live, aligned GzState; this test has exclusive access to it and checks for null via ok_or.
        unsafe { file.cast::<GzState>().as_mut().ok_or(())?.mode = GzMode::None };
        // SAFETY: the gzip handle is live and exclusively held when non-null; the C API explicitly accepts null and reports an error.
        assert_eq!(unsafe { gzbuffer(file, 1024) }, -1);
        // Put the file handle back in read mode, and now gzbuffer should work.
        // SAFETY: the open gzip handle owns a live, aligned GzState; this test has exclusive access to it and checks for null via ok_or.
        unsafe { file.cast::<GzState>().as_mut().ok_or(())?.mode = GzMode::Read };
        // SAFETY: the gzip handle is live and exclusively held when non-null; the C API explicitly accepts null and reports an error.
        assert_eq!(unsafe { gzbuffer(file, 1024) }, 0);
        assert_eq!(
            // SAFETY: the opened gzip handle points to a live, aligned GzState, and this temporary immutable inspection cannot outlive the handle.
            unsafe { file.cast::<GzState>().as_ref().ok_or(())?.want },
            1024
        );
        // Request a very small buffer size. gzbuffer should instead use the min size, 8 bytes.
        // SAFETY: the gzip handle is live and exclusively held when non-null; the C API explicitly accepts null and reports an error.
        assert_eq!(unsafe { gzbuffer(file, 5) }, 0);
        assert_eq!(
            // SAFETY: the opened gzip handle points to a live, aligned GzState, and this temporary immutable inspection cannot outlive the handle.
            unsafe { file.cast::<GzState>().as_ref().ok_or(())?.want },
            8
        );
        // Call gzdirect to force the allocation of buffers. After that, gzbuffer should fail.
        // SAFETY: gzdirect receives the live gzip handle opened immediately above, including its owned buffer state.
        assert_eq!(unsafe { gzdirect(file) }, 1);
        // SAFETY: the gzip handle is live and exclusively held when non-null; the C API explicitly accepts null and reports an error.
        assert_eq!(unsafe { gzbuffer(file, 1024) }, -1);
        assert_eq!(
            // SAFETY: the opened gzip handle points to a live, aligned GzState, and this temporary immutable inspection cannot outlive the handle.
            unsafe { file.cast::<GzState>().as_ref().ok_or(())?.want },
            8
        );
        // SAFETY: the handle was returned by gzopen or gzdopen and has not yet been closed; this call consumes the live handle.
        assert_eq!(unsafe { gzclose(file) }, Z_OK);
        Ok(())
    }
}
