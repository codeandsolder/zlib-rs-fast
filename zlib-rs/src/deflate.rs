use core::{ffi::CStr, marker::PhantomData, mem::MaybeUninit, ops::ControlFlow, ptr::NonNull};

use crate::{
    abi_cursor::{advance_abi_bytes, SliceBudget},
    adler32::adler32,
    allocate::{AlignedAllocationBlock, Allocator, ALLOCATION_BLOCK_BYTES},
    c_api::{gz_header, internal_state, z_checksum, z_stream},
    crc32::{crc32, Crc32Fold},
    output_buffer::{assume_init_prefix, WritableBytes},
    state_ptr::StatePtr,
    trace,
    weak_slice::{WeakArrayMut, WeakSliceMut},
    DeflateFlush, ReturnCode, ADLER32_INITIAL_VALUE, CRC32_INITIAL_VALUE, MAX_WBITS, MIN_WBITS,
};

use self::{
    algorithm::{change_requires_block_boundary, level_config},
    hash_calc::{HashCalcVariant, RollHashCalc, StandardHashCalc},
    pending::Pending,
    sym_buf::SymBuf,
    trees_tbl::STATIC_LTREE,
    window::Window,
};

mod algorithm;
#[cfg(feature = "std")]
mod batch;
#[cfg(feature = "gpu")]
mod gpu_match;
mod hash_calc;
mod longest_match;
mod match_compare;
#[cfg(any(test, all(feature = "__internal-api", feature = "rust-allocator")))]
mod near_optimal;
mod pending;
mod slide_hash;
mod sym_buf;
mod trees_tbl;
mod window;

// Position relative to the current window
#[cfg(feature = "gpu")]
pub use batch::compress_slices_batch_gpu;
#[cfg(feature = "parallel")]
pub use batch::BatchCompressor;
#[cfg(feature = "std")]
pub use batch::{
    compress_slices_batch, compress_slices_batch_with_backend, BatchCompressJob,
    BatchCompressResult, BatchMatchSearchBackend,
};
#[cfg(feature = "gpu")]
pub use gpu_match::{GpuMatch, GpuMatchError, GpuMatchSearcher, GpuMatchTable};

pub type Pos = u16;

// SAFETY: This struct must have the same layout as [`z_stream`], so that casts and transmutations
// between the two can work without UB.
#[repr(C)]
pub struct DeflateStream<'a> {
    pub(crate) next_in: *mut crate::c_api::Bytef,
    pub(crate) avail_in: crate::c_api::uInt,
    pub(crate) total_in: crate::c_api::z_size,
    pub(crate) next_out: *mut crate::c_api::Bytef,
    pub(crate) avail_out: crate::c_api::uInt,
    pub(crate) total_out: crate::c_api::z_size,
    pub(crate) msg: *const core::ffi::c_char,
    pub(crate) state: StatePtr<State<'a>>,
    pub(crate) alloc: Allocator<'a>,
    pub(crate) data_type: core::ffi::c_int,
    pub(crate) adler: crate::c_api::z_checksum,
    pub(crate) reserved: crate::c_api::uLong,
}

// `DeflateStream` is a typed view of the public zlib ABI shell. Prove the complete field mapping,
// not merely total size/alignment: the `alloc` field deliberately overlays z_stream's
// zalloc/zfree/opaque triple.
const _: () = {
    use core::mem::{align_of, offset_of, size_of};

    type D = DeflateStream<'static>;
    type Z = z_stream;

    assert!(size_of::<D>() == size_of::<Z>());
    assert!(align_of::<D>() == align_of::<Z>());
    assert!(offset_of!(D, next_in) == offset_of!(Z, next_in));
    assert!(offset_of!(D, avail_in) == offset_of!(Z, avail_in));
    assert!(offset_of!(D, total_in) == offset_of!(Z, total_in));
    assert!(offset_of!(D, next_out) == offset_of!(Z, next_out));
    assert!(offset_of!(D, avail_out) == offset_of!(Z, avail_out));
    assert!(offset_of!(D, total_out) == offset_of!(Z, total_out));
    assert!(offset_of!(D, msg) == offset_of!(Z, msg));
    assert!(offset_of!(D, state) == offset_of!(Z, state));
    assert!(offset_of!(D, alloc) + offset_of!(Allocator<'static>, zalloc) == offset_of!(Z, zalloc));
    assert!(offset_of!(D, alloc) + offset_of!(Allocator<'static>, zfree) == offset_of!(Z, zfree));
    assert!(offset_of!(D, alloc) + offset_of!(Allocator<'static>, opaque) == offset_of!(Z, opaque));
    assert!(offset_of!(D, data_type) == offset_of!(Z, data_type));
    assert!(offset_of!(D, adler) == offset_of!(Z, adler));
    assert!(offset_of!(D, reserved) == offset_of!(Z, reserved));
};

#[derive(Clone, Copy)]
pub struct InputCursor {
    next_in: *mut crate::c_api::Bytef,
    pub(crate) avail_in: crate::c_api::uInt,
    pub(crate) total_in: crate::c_api::z_size,
    // Only bytes consumed during this Compressor invocation can be reread from behind next_in.
    // A prior invocation's buffer may already have been freed by the caller.
    initial_capacity: crate::c_api::uInt,
}

impl InputCursor {
    const fn from_stream(stream: &DeflateStream<'_>) -> Self {
        Self {
            next_in: stream.next_in,
            avail_in: stream.avail_in,
            total_in: stream.total_in,
            initial_capacity: stream.avail_in,
        }
    }

    const fn apply(self, stream: &mut DeflateStream<'_>) {
        stream.next_in = self.next_in;
        stream.avail_in = self.avail_in;
        stream.total_in = self.total_in;
    }

    fn from_slice(bytes: &[u8], total_in: crate::c_api::z_size) -> Option<Self> {
        // The caller supplies the retained dictionary tail (at most 32 KiB). Keep this
        // conversion fallible so a future general slice caller cannot silently truncate it.
        let avail_in = crate::c_api::uInt::try_from(bytes.len()).ok()?;
        Some(Self {
            next_in: bytes.as_ptr().cast_mut(),
            avail_in,
            total_in,
            initial_capacity: avail_in,
        })
    }

    /// Copy live caller input into initialized window storage and consume it.
    fn copy_to_window(&mut self, window: &mut Window<'_>, range: core::ops::Range<usize>) {
        let len = range.len();
        assert!(len <= self.avail_in as usize);
        if len == 0 {
            return;
        }

        // SAFETY: the ABI contract guarantees `next_in` is readable for `avail_in` bytes, and the
        // requested copy is bounded by that count. `Window` checks the destination range.
        unsafe { window.copy_and_initialize(range, self.next_in) };
        self.consume(len);
    }

    /// Copy live caller input directly to output and consume it.
    ///
    /// # Safety
    /// The current caller input span must not overlap the output cursor's writable span.
    unsafe fn copy_to_output<'a>(&mut self, output: &'a mut OutputCursor, len: usize) -> &'a [u8] {
        assert!(len <= self.avail_in as usize);
        // SAFETY: input readability follows from the cursor invariant; non-overlap is required from
        // this method's caller and output bounds are checked by `copy_from_raw`.
        let copied = unsafe { output.copy_from_raw(self.next_in, len) };
        self.consume(len);
        copied
    }

    /// Bytes advanced through this cursor since it was bound to the current caller input.
    const fn consumed_here(&self) -> usize {
        (self.initial_capacity - self.avail_in) as usize
    }

    /// Copy bytes immediately preceding the current input cursor into the LZ history window.
    ///
    /// # Safety
    /// `bytes_back` must not reach before the beginning of the caller input allocation.
    unsafe fn copy_recent_to_window(
        &self,
        window: &mut Window<'_>,
        destination: core::ops::Range<usize>,
        bytes_back: usize,
    ) {
        assert!(destination.len() <= bytes_back);
        // C callers can replace or free their previous input between deflate calls.
        // Stored-mode history recovery must never traverse back into that old buffer.
        assert!(
            bytes_back <= self.consumed_here(),
            "stored history must originate from input consumed in this call"
        );
        let source = self.next_in.wrapping_sub(bytes_back);
        // SAFETY: required by this method's caller; `Window` validates the destination range.
        unsafe { window.copy_and_initialize(destination, source) };
    }

    fn consume(&mut self, len: usize) {
        advance_abi_bytes(
            &mut self.next_in,
            &mut self.avail_in,
            &mut self.total_in,
            len,
        );
    }
}

/// Native C unsigned long can be wider than the zlib/gzip 32-bit wire
/// fields. Those fields intentionally encode the low word, modulo 2^32.
#[inline]
const fn low_wire_u32(value: core::ffi::c_ulong) -> u32 {
    let bytes = value.to_le_bytes();
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

/// RFC 1952 extra-field XLEN uses the low 16 bits of the caller's length.
/// Emit only that many bytes, matching zlib's wire-format contract.
#[inline]
const fn gzip_extra_wire_len(length: usize) -> u16 {
    let bytes = length.to_le_bytes();
    u16::from_le_bytes([bytes[0], bytes[1]])
}

/// Public zlib checksum field detached from caller input traversal.
///
/// zlib overloads `z_stream.adler`: it is the running Adler-32 for zlib streams, temporary gzip
/// header CRC state while emitting an HCRC header, and the final gzip data CRC once finishing.
/// Keeping that ABI-visible value explicit prevents input cursors from silently carrying stream
/// state unrelated to byte traversal.
#[derive(Clone, Copy)]
pub struct StreamChecksum(crate::c_api::z_checksum);

impl StreamChecksum {
    const fn from_stream(stream: &DeflateStream<'_>) -> Self {
        Self(stream.adler)
    }

    const fn apply(self, stream: &mut DeflateStream<'_>) {
        stream.adler = self.0;
    }

    const fn value(self) -> crate::c_api::z_checksum {
        self.0
    }

    const fn set(&mut self, value: crate::c_api::z_checksum) {
        self.0 = value;
    }

    const fn reset_adler(&mut self) {
        self.0 = ADLER32_INITIAL_VALUE as _;
    }

    const fn low_word(self) -> u32 {
        low_wire_u32(self.0)
    }

    fn update_adler(&mut self, bytes: &[u8]) {
        self.0 = adler32(self.low_word(), bytes).into();
    }

    fn update_crc(&mut self, bytes: &[u8]) {
        self.0 = crc32(self.low_word(), bytes).into();
    }
}

/// Shared caller-owned gzip header bound by `set_header`.
///
/// The constructor captures the public API's unsafe lifetime/readability contract once. Header
/// emission and `deflateBound` can then inspect the header and its nested buffers without repeated
/// raw-pointer dereferences.
#[derive(Clone, Copy)]
struct GzipHeaderRef<'a> {
    ptr: NonNull<gz_header>,
    _lifetime: PhantomData<&'a gz_header>,
}

impl<'a> GzipHeaderRef<'a> {
    /// # Safety
    ///
    /// `head` and its non-null `extra`, `name`, and `comment` pointers must satisfy the complete
    /// contract documented on `set_header` for `'a`.
    unsafe fn new(head: &'a mut gz_header) -> Self {
        Self {
            ptr: NonNull::from(head),
            _lifetime: PhantomData,
        }
    }

    const fn header(&self) -> &gz_header {
        // SAFETY: guaranteed once by `new`. The returned reference is deliberately tied to this
        // short borrow, so callers may still mutate the C header between compression operations.
        unsafe { self.ptr.as_ref() }
    }

    const fn flags(self) -> u8 {
        self.header().flags()
    }

    const fn time(self) -> u32 {
        low_wire_u32(self.header().time)
    }

    const fn os(self) -> u8 {
        // The RFC 1952 OS field is the low byte of a signed C int,
        // including a caller-supplied negative value such as -1.
        self.header().os.to_le_bytes()[0]
    }

    const fn hcrc(self) -> bool {
        self.header().hcrc != 0
    }

    const fn extra(&self) -> Option<&[u8]> {
        let header = self.header();
        if header.extra.is_null() {
            return None;
        }

        // SAFETY: `new` requires `extra` to be readable for `extra_len` bytes.
        Some(unsafe { core::slice::from_raw_parts(header.extra, header.extra_len as usize) })
    }

    fn encoded_extra(&self) -> Option<&[u8]> {
        self.extra()
            .map(|extra| &extra[..usize::from(gzip_extra_wire_len(extra.len()))])
    }

    const fn name(&self) -> Option<&CStr> {
        let name = self.header().name;
        if name.is_null() {
            None
        } else {
            // SAFETY: `new` requires non-null `name` to satisfy `CStr::from_ptr`.
            Some(unsafe { CStr::from_ptr(name.cast()) })
        }
    }

    const fn comment(&self) -> Option<&CStr> {
        let comment = self.header().comment;
        if comment.is_null() {
            None
        } else {
            // SAFETY: `new` requires non-null `comment` to satisfy `CStr::from_ptr`.
            Some(unsafe { CStr::from_ptr(comment.cast()) })
        }
    }
}

#[derive(Clone, Copy)]
pub struct OutputCursor {
    next_out: *mut crate::c_api::Bytef,
    pub(crate) avail_out: crate::c_api::uInt,
    pub(crate) total_out: crate::c_api::z_size,
}

impl OutputCursor {
    const fn from_stream(stream: &DeflateStream<'_>) -> Self {
        Self {
            next_out: stream.next_out,
            avail_out: stream.avail_out,
            total_out: stream.total_out,
        }
    }

    const fn apply(self, stream: &mut DeflateStream<'_>) {
        stream.next_out = self.next_out;
        stream.avail_out = self.avail_out;
        stream.total_out = self.total_out;
    }

    fn write_all(&mut self, bytes: &[u8]) {
        let len = bytes.len();
        assert!(len <= self.avail_out as usize);
        if len == 0 {
            return;
        }

        // SAFETY: the cursor represents `avail_out` caller-provided writable bytes, and `len` is
        // checked above. Compression state/pending storage must not overlap the caller output.
        unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), self.next_out, len) };
        self.advance(len);
    }

    fn write_some(&mut self, bytes: &[u8]) -> usize {
        let len = Ord::min(bytes.len(), self.avail_out as usize);
        self.write_all(&bytes[..len]);
        len
    }

    /// Copy caller input into output and expose exactly the bytes just initialized.
    ///
    /// # Safety
    /// `source` must be readable for `len` bytes and must not overlap the cursor's writable output.
    unsafe fn copy_from_raw(&mut self, source: *const u8, len: usize) -> &[u8] {
        assert!(len <= self.avail_out as usize);
        if len == 0 {
            return &[];
        }

        let destination = self.next_out;
        // SAFETY: guaranteed by the caller contract plus the output bound asserted above.
        unsafe { core::ptr::copy_nonoverlapping(source, destination, len) };
        self.advance(len);

        // SAFETY: the copy above initialized exactly this output range, which remains valid for the
        // duration of the cursor borrow.
        unsafe { core::slice::from_raw_parts(destination, len) }
    }

    fn advance(&mut self, len: usize) {
        advance_abi_bytes(
            &mut self.next_out,
            &mut self.avail_out,
            &mut self.total_out,
            len,
        );
    }
}

/// One snapshot of the three independent caller-I/O domains.
///
/// A driver turn borrows the snapshots while compression state is borrowed
/// separately. Every turn commits the input, output, and checksum together,
/// including when it yields for more input, output, or match-search work.
struct CallCursors {
    input: InputCursor,
    output: OutputCursor,
    checksum: StreamChecksum,
}

impl CallCursors {
    const fn from_stream(stream: &DeflateStream<'_>) -> Self {
        Self {
            input: InputCursor::from_stream(stream),
            output: OutputCursor::from_stream(stream),
            checksum: StreamChecksum::from_stream(stream),
        }
    }

    const fn apply(self, stream: &mut DeflateStream<'_>) {
        self.input.apply(stream);
        self.output.apply(stream);
        self.checksum.apply(stream);
    }

    /// The stream's raw ABI fields are only rebound on entry and commit.
    #[inline]
    fn with_state<'state, R>(
        stream: &mut DeflateStream<'state>,
        operation: impl FnOnce(&mut State<'state>, &mut Self) -> R,
    ) -> R {
        let mut cursors = Self::from_stream(stream);
        let result = operation(&mut stream.state, &mut cursors);
        cursors.apply(stream);
        result
    }
}

/// Internal execution context for one compression call.
///
/// `DeflateStream` is constrained by the public zlib ABI. Algorithms operate on this
/// Rust-owned view instead: compression state is borrowed separately from independent input and
/// output cursors, which are committed back to the ABI shell when the algorithm returns.
pub struct Compressor<'a, 'state> {
    pub(crate) state: &'a mut State<'state>,
    pub(crate) input: &'a mut InputCursor,
    pub(crate) output: &'a mut OutputCursor,
    pub(crate) checksum: &'a mut StreamChecksum,
}

impl<'a, 'state> Compressor<'a, 'state> {
    const fn new(
        state: &'a mut State<'state>,
        input: &'a mut InputCursor,
        output: &'a mut OutputCursor,
        checksum: &'a mut StreamChecksum,
    ) -> Self {
        Self {
            state,
            input,
            output,
            checksum,
        }
    }

    pub(crate) fn fill_window(&mut self) {
        fill_window_core(self.state, self.input, self.checksum);
    }

    fn drain_pending(&mut self) -> DrainOutcome {
        drain_pending_output(self.state, self.output)
    }

    pub(crate) fn read_buf_window(&mut self, offset: usize, size: usize) -> usize {
        read_buf_window_core(self.state, self.input, self.checksum, offset, size)
    }

    /// Copy initialized history bytes to the caller's output buffer.
    pub(crate) fn copy_window_to_output(&mut self, range: core::ops::Range<usize>) {
        let len = range.len();
        assert!(len <= self.output.avail_out as usize);
        let src = &self.state.window.filled()[range];

        self.output.write_all(src);
    }

    /// Copy caller input directly to caller output for stored blocks while updating checksums.
    ///
    /// Deliberately does not construct a Rust slice over `next_in`: zlib permits the caller to
    /// mutate input memory concurrently. We checksum the output copy, which is stable after the raw
    /// copy has completed.
    pub(crate) fn copy_input_to_output(&mut self, size: usize) -> usize {
        let len = size.min(self.input.avail_in as usize);
        assert!(len <= self.output.avail_out as usize);
        if len == 0 {
            return 0;
        }

        // SAFETY: zlib's direct stored-copy contract requires the caller input/output spans not to
        // overlap. Input/output bounds and cursor accounting are confined to the cursor methods.
        let copied = unsafe { self.input.copy_to_output(self.output, len) };
        match self.state.wrap {
            2 => self.state.crc_fold.fold(copied),
            1 => self.checksum.update_adler(copied),
            _ => {}
        }

        len
    }

    /// Preserve bytes that were already consumed directly from the caller's input as LZ history.
    pub(crate) fn copy_recent_input_to_window(
        &mut self,
        destination: core::ops::Range<usize>,
        bytes_back: usize,
    ) {
        // SAFETY: callers only request bytes already consumed by `copy_input_to_output`, so the
        // region ending at the current input cursor remains within the caller's input allocation.
        unsafe {
            self.input
                .copy_recent_to_window(&mut self.state.window, destination, bytes_back);
        }
    }
}

fn with_compressor<'state, R>(
    stream: &mut DeflateStream<'state>,
    f: impl FnOnce(&mut Compressor<'_, 'state>) -> R,
) -> R {
    // `data_type` is public z_stream metadata, so honor caller overrides at each ABI entry. The
    // encoder owns the working copy in State and publishes any lazy detection result on return.
    stream.state.data_type = stream.data_type;
    let result = CallCursors::with_state(stream, |state, cursors| {
        let mut compressor = Compressor::new(
            state,
            &mut cursors.input,
            &mut cursors.output,
            &mut cursors.checksum,
        );
        f(&mut compressor)
    });
    stream.data_type = stream.state.data_type;
    result
}

// This ABI view contains non-owning caller pointers and mutable state
// allocations. Thread safety belongs to independently owned compression jobs,
// not to a blanket Send/Sync assertion on the raw-pointer-backed view.

impl<'a> DeflateStream<'a> {
    /// # Safety
    ///
    /// Behavior is undefined if any of the following conditions are violated:
    ///
    /// - `strm` satisfies the conditions of [`pointer::as_mut`]
    /// - if not `NULL`, `strm` as initialized using [`init`] or similar
    ///
    /// [`pointer::as_mut`]: https://doc.rust-lang.org/core/primitive.pointer.html#method.as_mut
    #[inline]
    pub unsafe fn from_stream_mut(strm: *mut z_stream) -> Option<&'a mut Self> {
        {
            // Safety: ptr points to a valid value of type z_stream (if non-null)
            let stream = unsafe { strm.as_ref() }?;

            if stream.zalloc.is_none() || stream.zfree.is_none() {
                return None;
            }

            if stream.state.is_null() {
                return None;
            }
        }

        // SAFETY: DeflateStream has an equivalent layout as z_stream
        unsafe { strm.cast::<DeflateStream>().as_mut() }
    }

    /// # Safety
    ///
    /// Behavior is undefined if any of the following conditions are violated:
    ///
    /// - `strm` satisfies the conditions of [`pointer::as_ref`]
    /// - if not `NULL`, `strm` as initialized using [`init`] or similar
    ///
    /// [`pointer::as_ref`]: https://doc.rust-lang.org/core/primitive.pointer.html#method.as_ref
    #[must_use]
    #[inline]
    pub unsafe fn from_stream_ref(strm: *const z_stream) -> Option<&'a Self> {
        {
            // Safety: ptr points to a valid value of type z_stream (if non-null)
            let stream = unsafe { strm.as_ref() }?;

            if stream.zalloc.is_none() || stream.zfree.is_none() {
                return None;
            }

            if stream.state.is_null() {
                return None;
            }
        }

        // SAFETY: DeflateStream has an equivalent layout as z_stream
        unsafe { strm.cast::<DeflateStream>().as_ref() }
    }

    fn as_z_stream_mut(&mut self) -> &mut z_stream {
        // SAFETY: a valid &mut DeflateStream is also a valid &mut z_stream
        unsafe { &mut *core::ptr::from_mut(self).cast::<z_stream>() }
    }

    #[must_use]
    pub fn pending(&self) -> (usize, u8) {
        (
            self.state.bit_writer.pending.len(),
            self.state.bit_writer.bits_valid,
        )
    }

    /// Last number of used bits when going to a byte boundary.
    #[must_use]
    pub fn bits_used(&self) -> u8 {
        self.state.bit_writer.bits_used
    }

    /// Construct an owned compressor whose allocation is released on drop.
    ///
    /// # Errors
    ///
    /// Returns the initialization status for invalid configuration or
    /// allocation failure, without constructing a partially initialized state.
    pub fn new(config: DeflateConfig) -> Result<Self, ReturnCode> {
        let mut inner = crate::c_api::z_stream::default();
        let status = crate::deflate::init(&mut inner, config);
        if status != ReturnCode::Ok {
            return Err(status);
        }

        // init guarantees callbacks and a live State on success.
        let (Some(zalloc), Some(zfree)) = (inner.zalloc, inner.zfree) else {
            return Err(ReturnCode::StreamError);
        };
        if inner.state.is_null() {
            return Err(ReturnCode::StreamError);
        }

        let state = inner.state.cast::<State<'a>>();
        let alloc = Allocator {
            zalloc,
            zfree,
            opaque: inner.opaque,
            _marker: PhantomData,
        };
        Ok(Self {
            next_in: inner.next_in.cast_mut(),
            avail_in: inner.avail_in,
            total_in: inner.total_in,
            next_out: inner.next_out,
            avail_out: inner.avail_out,
            total_out: inner.total_out,
            msg: inner.msg.cast_const(),
            // SAFETY: successful initialization created the exclusively owned State.
            state: unsafe { StatePtr::from_raw(state) },
            alloc,
            data_type: inner.data_type,
            adler: inner.adler,
            reserved: inner.reserved,
        })
    }
}

impl Drop for DeflateStream<'_> {
    fn drop(&mut self) {
        // End nulls the State pointer before releasing it, so an explicit
        // deflateEnd and this owner cleanup cannot free the same state twice.
        if !self.state.as_ptr().is_null() {
            let _ = end(self);
        }
    }
}

/// number of elements in hash table
pub const HASH_SIZE: usize = 65536;
/// `log2(HASH_SIZE)`
const HASH_BITS: usize = 16;

/// Maximum value for memLevel in deflateInit2
const MAX_MEM_LEVEL: i32 = 9;
pub const DEF_MEM_LEVEL: i32 = if MAX_MEM_LEVEL > 8 { 8 } else { MAX_MEM_LEVEL };

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "__internal-fuzz", derive(arbitrary::Arbitrary))]
pub enum Method {
    #[default]
    Deflated = 8,
}

impl TryFrom<i32> for Method {
    type Error = ();

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            8 => Ok(Self::Deflated),
            _ => Err(()),
        }
    }
}

/// Configuration for compression.
///
/// Used with [`compress_slice`].
///
/// In most cases only the compression level is relevant. We provide three profiles:
///
/// - [`DeflateConfig::best_speed`] provides the fastest compression (at the cost of compression
///   quality)
/// - [`DeflateConfig::default`] tries to find a happy middle
/// - [`DeflateConfig::best_compression`] provides the best compression (at the cost of longer
///   runtime)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "__internal-fuzz", derive(arbitrary::Arbitrary))]
pub struct DeflateConfig {
    pub level: i32,
    pub method: Method,
    pub window_bits: i32,
    pub mem_level: i32,
    pub strategy: Strategy,
}

#[cfg(any(test, feature = "__internal-test"))]
impl quickcheck::Arbitrary for DeflateConfig {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        let mem_levels: Vec<_> = (1..=9).collect();
        let levels: Vec<_> = (0..=9).collect();

        let mut window_bits = Vec::new();
        window_bits.extend(9..=15); // zlib
        window_bits.extend(9 + 16..=15 + 16); // gzip
        window_bits.extend(-15..=-9); // raw

        Self {
            level: *g.choose(&levels).unwrap(),
            method: Method::Deflated,
            window_bits: *g.choose(&window_bits).unwrap(),
            mem_level: *g.choose(&mem_levels).unwrap(),
            strategy: *g
                .choose(&[
                    Strategy::Default,
                    Strategy::Filtered,
                    Strategy::HuffmanOnly,
                    Strategy::Rle,
                    Strategy::Fixed,
                ])
                .unwrap(),
        }
    }
}

impl DeflateConfig {
    #[must_use]
    pub fn new(level: i32) -> Self {
        Self {
            level,
            ..Self::default()
        }
    }

    /// Configure for the best compression (takes longer).
    #[must_use]
    pub fn best_compression() -> Self {
        Self::new(crate::c_api::Z_BEST_COMPRESSION)
    }

    /// Configure for the fastest compression (compresses less well).
    #[must_use]
    pub fn best_speed() -> Self {
        Self::new(crate::c_api::Z_BEST_SPEED)
    }
}

impl Default for DeflateConfig {
    fn default() -> Self {
        Self {
            level: crate::c_api::Z_DEFAULT_COMPRESSION,
            method: Method::Deflated,
            window_bits: MAX_WBITS,
            mem_level: DEF_MEM_LEVEL,
            strategy: Strategy::Default,
        }
    }
}

pub fn init(stream: &mut z_stream, config: DeflateConfig) -> ReturnCode {
    init_with_lit_bufsize(stream, config, None)
}

fn init_with_lit_bufsize(
    stream: &mut z_stream,
    config: DeflateConfig,
    lit_bufsize_override: Option<usize>,
) -> ReturnCode {
    let DeflateConfig {
        mut level,
        method: _,
        mut window_bits,
        mem_level,
        strategy,
    } = config;

    /* Todo: ignore strm->next_in if we use it as window */
    stream.msg = core::ptr::null_mut();

    // for safety we  must really make sure that alloc and free are consistent
    // this is a (slight) deviation from stock zlib. In this crate we pick the rust
    // allocator as the default, but `libz-rs-sys` always explicitly sets an allocator,
    // and can configure the C allocator
    #[cfg(feature = "rust-allocator")]
    if stream.zalloc.is_none() || stream.zfree.is_none() {
        stream.configure_default_rust_allocator();
    }

    #[cfg(feature = "c-allocator")]
    if stream.zalloc.is_none() || stream.zfree.is_none() {
        stream.configure_default_c_allocator();
    }

    let (Some(zalloc), Some(zfree)) = (stream.zalloc, stream.zfree) else {
        return ReturnCode::StreamError;
    };

    if level == crate::c_api::Z_DEFAULT_COMPRESSION {
        level = 6;
    }

    let wrap = if window_bits < 0 {
        if window_bits < -MAX_WBITS {
            return ReturnCode::StreamError;
        }
        window_bits = -window_bits;

        0
    } else if window_bits > MAX_WBITS {
        window_bits -= 16;
        2
    } else {
        1
    };

    if (!(1..=MAX_MEM_LEVEL).contains(&mem_level))
        || !(MIN_WBITS..=MAX_WBITS).contains(&window_bits)
        || !(0..=9).contains(&level)
        || (window_bits == 8 && wrap != 1)
    {
        return ReturnCode::StreamError;
    }

    // Configuration validation above guarantees a positive 8..=15 bit
    // window. Keep the host-address width conversion explicit anyway.
    let Ok(window_bits) = usize::try_from(window_bits) else {
        return ReturnCode::StreamError;
    };
    let window_bits = window_bits.max(9); // zlib's 256-byte window workaround

    let alloc = Allocator {
        zalloc,
        zfree,
        opaque: stream.opaque,
        _marker: PhantomData,
    };

    let lit_bufsize = lit_bufsize_override.unwrap_or(1 << (mem_level + 6));
    // The state representation and 16-bit frequency counters support at most 64K symbol slots.
    // Ordinary zlib-compatible initialization reaches at most 32K (memLevel=9); the experimental
    // near-optimal complete-buffer path uses the full 64K allocation so block boundaries can be
    // chosen independently of the public memLevel buffering policy.
    if lit_bufsize == 0 || lit_bufsize > (1 << 16) {
        return ReturnCode::StreamError;
    }
    // The validated compression level is 0..=9, representable in i8.
    let Ok(level) = i8::try_from(level) else {
        return ReturnCode::StreamError;
    };
    let allocs = DeflateAllocOffsets::new(window_bits, lit_bufsize);
    let Some(state_allocation) = allocs.allocate_state(
        &alloc,
        DeflateStateInit {
            level,
            strategy,
            wrap,
        },
    ) else {
        return ReturnCode::MemError;
    };

    stream.state = state_allocation.as_ptr().cast::<internal_state>();

    // SAFETY: allocation succeeded and initialized stream.state; allocator
    // callbacks were validated above and still belong to the live stream.
    let Some(stream) = (unsafe { DeflateStream::from_stream_mut(stream) }) else {
        if cfg!(debug_assertions) {
            unreachable!("we should have initialized the stream properly");
        }
        return ReturnCode::StreamError;
    };

    reset(stream)
}

/// Return the bytes accumulated since the start of the current DEFLATE block.
///
/// A nonnegative origin identifies resident bytes in the sliding window.
/// When a slide moves that origin out of the window, it becomes negative:
/// its absolute value is the missing prefix length. The accumulated length
/// still includes that prefix, but no stored-block source range exists.
///
/// # Panics
///
/// Panics if the block start lies beyond the current position or the combined
/// distance cannot be represented in the host address-size integer.
const fn unflushed_block_span(strstart: usize, block_start: isize) -> (Option<usize>, usize) {
    let origin = block_start.unsigned_abs();
    if block_start < 0 {
        assert!(
            origin <= usize::MAX - strstart,
            "unflushed block length exceeds host integer range"
        );
        (None, strstart + origin)
    } else {
        assert!(
            origin <= strstart,
            "block start is after the current position"
        );
        (Some(origin), strstart - origin)
    }
}

pub fn params(stream: &mut DeflateStream, level: i32, strategy: Strategy) -> ReturnCode {
    let level = if level == crate::c_api::Z_DEFAULT_COMPRESSION {
        6
    } else {
        level
    };

    if !(0..=9).contains(&level) {
        return ReturnCode::StreamError;
    }

    // The public ABI level is signed and validated above; preserve its
    // representability explicitly before storing it in the compact state.
    let Ok(level) = i8::try_from(level) else {
        unreachable!("validated compression level does not fit i8");
    };

    let state = &mut *stream.state;

    let Some(next_config) = level_config(level) else {
        return ReturnCode::StreamError;
    };
    let Some(requires_boundary) =
        change_requires_block_boundary(state.level, state.strategy, level, strategy)
    else {
        return ReturnCode::StreamError;
    };
    if requires_boundary && state.last_flush != -2 {
        // Flush the last buffer.
        let err = deflate(stream, DeflateFlush::Block);
        if err == ReturnCode::StreamError {
            return err;
        }

        let state = &mut *stream.state;

        if stream.avail_in != 0
            || unflushed_block_span(state.strstart, state.block_start).1 != 0
            || state.lookahead != 0
        {
            return ReturnCode::BufError;
        }
    }

    let state = &mut *stream.state;

    if state.level != level {
        if state.level == 0 && state.matches != 0 {
            if state.matches == 1 {
                self::slide_hash::slide_hash(state);
            } else {
                state.head.as_mut_slice().fill(0);
            }
            state.matches = 0;
        }

        lm_set_level(state, level, next_config);
    }

    state.strategy = strategy;

    ReturnCode::Ok
}

pub fn set_dictionary(stream: &mut DeflateStream, mut dictionary: &[u8]) -> ReturnCode {
    let wrap = stream.state.wrap;

    if wrap == 2
        || (wrap == 1 && stream.state.status != Status::Init)
        || stream.state.lookahead != 0
    {
        return ReturnCode::StreamError;
    }

    let mut checksum = StreamChecksum::from_stream(stream);
    // when using zlib wrappers, compute Adler-32 for provided dictionary
    if wrap == 1 {
        checksum.update_adler(dictionary);
    }

    let state = &mut *stream.state;
    let replace_history = dictionary.len() >= state.w_size;
    if replace_history {
        dictionary = &dictionary[dictionary.len() - state.w_size..];
    }

    // A general input cursor cannot represent more than uInt::MAX bytes in a single call.
    // This retained dictionary tail is bounded by the window, but validate before mutating
    // the compressor state so even an invalid future caller gets a clean error.
    let Some(mut dictionary_input) = InputCursor::from_slice(dictionary, stream.total_in) else {
        return ReturnCode::StreamError;
    };

    if replace_history && wrap == 0 {
        state.head.as_mut_slice().fill(0);
        state.strstart = 0;
        state.block_start = 0;
        state.insert = 0;
        state.window.clear();
    }

    // Adler-32 above covered the complete dictionary before retaining only its window tail.
    // Feed that tail through the ordinary window-refill core without impersonating
    // caller input in the public z_stream. Retained bytes count in total_in; the caller's
    // next_in and avail_in remain unchanged across deflateSetDictionary.
    state.wrap = 0;
    fill_window_core(state, &mut dictionary_input, &mut checksum);

    while state.lookahead >= STD_MIN_MATCH {
        let str = state.strstart;
        let n = state.lookahead - (STD_MIN_MATCH - 1);
        state.insert_string(str, n);
        state.strstart = str + n;
        state.lookahead = STD_MIN_MATCH - 1;
        fill_window_core(state, &mut dictionary_input, &mut checksum);
    }

    state.strstart += state.lookahead;
    state.block_start = state.strstart as _;
    state.insert = state.lookahead;
    state.lookahead = 0;
    state.prev_length = 0;
    state.match_available = false;
    state.wrap = wrap;
    stream.total_in = dictionary_input.total_in;
    checksum.apply(stream);

    ReturnCode::Ok
}

pub fn prime(stream: &mut DeflateStream, bits: i32, value: i32) -> ReturnCode {
    // The public zlib argument is signed, while the bit writer works with
    // byte-sized widths. Reject out-of-range values before changing state.
    let Ok(mut bits) = u8::try_from(bits) else {
        return ReturnCode::BufError;
    };
    if bits > BitWriter::BIT_BUF_SIZE || usize::from(bits) > core::mem::size_of_val(&value) * 8 {
        return ReturnCode::BufError;
    }
    let state = &mut *stream.state;

    // `flush_bits()` writes every complete byte from the combined reservoir. Preflight the
    // complete call so a full pending buffer reports zlib's Z_BUF_ERROR instead of reaching the
    // infallible internal append operation. zlib-ng's 64-bit reservoir accepts up to 32 input
    // bits here (bounded by `value`), despite the public compatibility documentation saying 16.
    let bytes_to_flush = (usize::from(state.bit_writer.bits_valid) + usize::from(bits)) / 8;
    if state.bit_writer.pending.remaining() < bytes_to_flush {
        return ReturnCode::BufError;
    }

    // Only the requested low bits belong to the DEFLATE stream. Mask once
    // before the reservoir can inherit high or sign-extended caller bits.
    let mut value64 = u64::from(value.cast_unsigned()) & ((1u64 << bits) - 1);

    loop {
        let put = (BitWriter::BIT_BUF_SIZE - state.bit_writer.bits_valid).min(bits);

        if state.bit_writer.bits_valid == 0 {
            state.bit_writer.bit_buffer = value64;
        } else {
            state.bit_writer.bit_buffer |=
                (value64 & ((1 << put) - 1)) << state.bit_writer.bits_valid;
        }

        state.bit_writer.bits_valid += put;
        state.bit_writer.flush_bits();
        value64 >>= put;
        bits -= put;

        if bits == 0 {
            break;
        }
    }

    ReturnCode::Ok
}

pub fn copy<'a>(
    dest: &mut MaybeUninit<DeflateStream<'a>>,
    source: &DeflateStream<'a>,
) -> ReturnCode {
    let source_state = &*source.state;
    let allocs = DeflateAllocOffsets::new(source_state.w_bits() as usize, source_state.lit_bufsize);
    let Some(state_allocation) = allocs.clone_state(&source.alloc, source_state) else {
        return ReturnCode::MemError;
    };

    // Build the destination ABI shell explicitly around the independently cloned state allocation.
    dest.write(DeflateStream {
        next_in: source.next_in,
        avail_in: source.avail_in,
        total_in: source.total_in,
        next_out: source.next_out,
        avail_out: source.avail_out,
        total_out: source.total_out,
        msg: source.msg,
        // SAFETY: clone_state allocated and initialized a separate live
        // destination State; this wrapper owns it until deflateEnd.
        state: unsafe { StatePtr::from_raw(state_allocation.as_ptr()) },
        alloc: source.alloc,
        data_type: source.data_type,
        adler: source.adler,
        reserved: source.reserved,
    });

    ReturnCode::Ok
}

/// # Returns
///
/// - Err when deflate is not done. A common cause is insufficient output space
/// - Ok otherwise
///
/// # Errors
///
/// Returns the caller's C stream in the error variant if compression was
/// interrupted or its state has already been released. In both cases the C
/// stream remains caller-owned; an initialized allocation is freed exactly once.
pub fn end<'a>(stream: &'a mut DeflateStream) -> Result<&'a mut z_stream, &'a mut z_stream> {
    if stream.state.as_ptr().is_null() {
        return Err(stream.as_z_stream_mut());
    }

    let status = stream.state.status;
    let allocation_start = stream.state.allocation_start;
    let total_allocation_size = stream.state.total_allocation_size;
    let alloc = stream.alloc;

    let stream = stream.as_z_stream_mut();
    let _ = core::mem::replace(&mut stream.state, core::ptr::null_mut());

    // SAFETY: this stream exclusively owns the live State allocation from
    // init/copy; allocation_start and total_allocation_size came from that
    // allocation, and stream.state was detached immediately above.
    unsafe {
        alloc.deallocate(
            allocation_start.as_ptr(),
            total_allocation_size / DEFLATE_ALLOC_ALIGN,
        );
    }

    match status {
        Status::Busy => Err(stream),
        _ => Ok(stream),
    }
}

pub fn reset(stream: &mut DeflateStream) -> ReturnCode {
    // A damaged or stale state must fail before reset_keep mutates any
    // buffered output, checksums, or the live hash history.
    let Some(config) = level_config(stream.state.level) else {
        return ReturnCode::StreamError;
    };
    let ret = reset_keep(stream);

    if ret == ReturnCode::Ok {
        lm_init(&mut stream.state, config);
    }

    ret
}

pub fn reset_keep(stream: &mut DeflateStream) -> ReturnCode {
    stream.total_in = 0;
    stream.total_out = 0;
    stream.msg = core::ptr::null_mut();
    stream.data_type = crate::c_api::Z_UNKNOWN;

    let state = &mut *stream.state;
    state.data_type = crate::c_api::Z_UNKNOWN;

    state.bit_writer.pending.reset_keep();

    // can be made negative by deflate(..., Z_FINISH);
    state.wrap = state.wrap.abs();

    state.status = match state.wrap {
        2 => Status::GZip,
        _ => Status::Init,
    };

    stream.adler = match state.wrap {
        2 => {
            state.crc_fold = Crc32Fold::new();
            CRC32_INITIAL_VALUE.into()
        }
        _ => ADLER32_INITIAL_VALUE as _,
    };

    state.last_flush = -2;
    state.medium_next_match = MatchCandidate::EMPTY;

    state.zng_tr_init();

    ReturnCode::Ok
}

fn lm_init(state: &mut State, config: algorithm::LevelConfig) {
    state.window_size = 2 * state.w_size;
    state.window.clear();

    // zlib uses CLEAR_HASH here
    state.head.as_mut_slice().fill(0);

    // Hashes are a property of the live history representation, not a tunable
    // search budget. Choose the representation only while the history is empty.
    lm_set_level(state, state.level, config);
    state.hash_calc_variant = HashCalcVariant::for_max_chain_length(state.max_chain_length);

    state.strstart = 0;
    state.block_start = 0;
    state.lookahead = 0;
    state.insert = 0;
    state.prev_length = 0;
    state.match_available = false;
    state.match_start = 0;
    state.ins_h = 0;
}

const fn lm_set_level(state: &mut State, level: i8, config: algorithm::LevelConfig) {
    state.medium_next_match = MatchCandidate::EMPTY;
    state.max_insert_length = config.max_lazy;
    state.max_lazy_match = config.max_lazy;
    state.good_match = config.good_length;
    state.nice_match = config.nice_length;
    state.max_chain_length = config.max_chain;

    // Do not change the hash representation here. deflateParams() may update the
    // level while a nonempty head/prev table still describes earlier input.
    // Reinterpreting those buckets with a different hash algorithm breaks searches.
    state.level = level;
}

pub fn tune(
    stream: &mut DeflateStream,
    good_length: usize,
    max_lazy: usize,
    nice_length: usize,
    max_chain: usize,
) -> ReturnCode {
    // These are comparison thresholds and maximum chain-search budgets, not bitfields:
    // - no DEFLATE match length exceeds 258;
    // - no window search has more than 32,768 distinct candidate positions.
    // A larger tuning parameter therefore has the same effect as u16::MAX; wrapping
    // it to zero would instead disable matching or change compression decisions.
    let bound = |value| u16::try_from(value).unwrap_or(u16::MAX);
    stream.state.good_match = bound(good_length);
    stream.state.medium_next_match = MatchCandidate::EMPTY;
    // zlib exposes one tuning knob for both historical uses of max_lazy. Keep the
    // public contract while representing the two internal concepts independently.
    let lazy = bound(max_lazy);
    stream.state.max_insert_length = lazy;
    stream.state.max_lazy_match = lazy;
    stream.state.nice_match = bound(nice_length);
    stream.state.max_chain_length = bound(max_chain);

    ReturnCode::Ok
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Value {
    a: u16,
    b: u16,
}

impl Value {
    pub(crate) const fn new(a: u16, b: u16) -> Self {
        Self { a, b }
    }

    pub(crate) const fn freq_mut(&mut self) -> &mut u16 {
        &mut self.a
    }

    pub(crate) const fn code_mut(&mut self) -> &mut u16 {
        &mut self.a
    }

    pub(crate) const fn dad_mut(&mut self) -> &mut u16 {
        &mut self.b
    }

    pub(crate) const fn len_mut(&mut self) -> &mut u16 {
        &mut self.b
    }

    #[inline]
    pub(crate) const fn freq(self) -> u16 {
        self.a
    }

    #[inline]
    pub(crate) const fn code(self) -> u16 {
        self.a
    }

    #[inline]
    pub(crate) const fn dad(self) -> u16 {
        self.b
    }

    #[inline]
    pub(crate) const fn len(self) -> u16 {
        self.b
    }
}

/// number of length codes, not counting the special `END_BLOCK` code
pub const LENGTH_CODES: usize = 29;

/// number of literal bytes 0..255
const LITERALS: usize = 256;

/// number of Literal or Length codes, including the `END_BLOCK` code
pub const L_CODES: usize = LITERALS + 1 + LENGTH_CODES;

/// number of distance codes
pub const D_CODES: usize = 30;

/// number of codes used to transfer the bit lengths
const BL_CODES: usize = 19;

/// maximum heap size
const HEAP_SIZE: usize = 2 * L_CODES + 1;

/// all codes must not exceed `MAX_BITS` bits
const MAX_BITS: usize = 15;

/// Bit length codes must not exceed `MAX_BL_BITS` bits
const MAX_BL_BITS: usize = 7;

pub const DIST_CODE_LEN: usize = 512;

struct BitWriter<'a> {
    pub(crate) pending: Pending<'a>, // output still pending

    /// Output buffer. bits are inserted starting at the bottom (least significant bits).
    pub(crate) bit_buffer: u64,

    /// Number of valid bits in `bit_buffer`. All bits above the last valid bit are always zero.
    pub(crate) bits_valid: u8,

    /// Last number of used bits when going to a byte boundary.
    pub(crate) bits_used: u8,

    /// total bit length of compressed file (NOTE: zlib-ng uses a 32-bit integer here)
    #[cfg(feature = "ZLIB_DEBUG")]
    compressed_len: usize,
    /// bit length of compressed data sent (NOTE: zlib-ng uses a 32-bit integer here)
    #[cfg(feature = "ZLIB_DEBUG")]
    bits_sent: usize,
}

/// A bounded, least-significant-bit-first DEFLATE code or combined match.
///
/// The representation maintains the width beside the bits it describes,
/// instead of narrowing unbounded bit counts at the eventual writer call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PackedBits {
    value: u64,
    len: u8,
}

impl PackedBits {
    const fn new(value: u64, len: u8) -> Self {
        // Only internal Huffman tables and checked extra-bit widths construct
        // these sequences. Validate their contract in debug builds without
        // adding per-match branches to release compression.
        debug_assert!(len <= 64, "packed DEFLATE sequence exceeds reservoir width");
        debug_assert!(
            len == 64 || value >> len == 0,
            "packed DEFLATE sequence contains bits beyond its width"
        );
        Self { value, len }
    }

    const fn code(symbol: Value) -> Self {
        // Huffman tree nodes reuse a 16-bit field for parent indices and
        // final lengths. Assert the <=15-bit wire contract *before* taking
        // the low byte, so an invalid node cannot become a plausible code
        // length even in a release build. This remains const on Rust 1.99.
        let length = symbol.len();
        assert!(length <= 15, "invalid DEFLATE Huffman code length");
        let [length, _] = length.to_le_bytes();
        Self::new(symbol.code() as u64, length)
    }

    const fn append(&mut self, suffix: Self) {
        debug_assert!(
            suffix.len <= 64 && self.len <= 64 - suffix.len,
            "packed DEFLATE sequence exceeds reservoir width"
        );
        // Wrapping shift makes appending a zero-width suffix to a full-width
        // value an ordinary no-op, with no branch in the match hot path.
        self.value |= suffix.value.wrapping_shl(self.len as u32);
        self.len += suffix.len;
    }
}

#[inline]
const fn encode_len(ltree: &[Value], lc: u8) -> PackedBits {
    let mut lc = lc as usize;

    /* Send the length code, len is the match length - STD_MIN_MATCH */
    let code = self::trees_tbl::LENGTH_CODE[lc] as usize;
    let c = code + LITERALS + 1;
    assert!(c < L_CODES, "bad l_code");

    let mut encoded = PackedBits::code(ltree[c]);
    let extra = StaticTreeDesc::EXTRA_LBITS[code];
    if extra != 0 {
        lc -= self::trees_tbl::BASE_LENGTH[code] as usize;
        encoded.append(PackedBits::new(lc as u64, extra));
    }

    encoded
}

#[inline]
const fn encode_dist(dtree: &[Value], dist: u16) -> PackedBits {
    assert!(dist != 0, "DEFLATE match distance must be nonzero");
    let mut dist = dist - 1;
    let code = State::d_code(dist as usize) as usize;
    assert!(code < D_CODES, "bad d_code");

    let mut encoded = PackedBits::code(dtree[code]);
    let extra = StaticTreeDesc::EXTRA_DBITS[code];
    if extra != 0 {
        dist -= self::trees_tbl::BASE_DIST[code];
        encoded.append(PackedBits::new(dist as u64, extra));
    }

    encoded
}

impl<'a> BitWriter<'a> {
    pub(crate) const BIT_BUF_SIZE: u8 = 64;

    const fn from_pending(pending: Pending<'a>) -> Self {
        Self {
            pending,
            bit_buffer: 0,
            bits_valid: 0,
            bits_used: 0,

            #[cfg(feature = "ZLIB_DEBUG")]
            compressed_len: 0,
            #[cfg(feature = "ZLIB_DEBUG")]
            bits_sent: 0,
        }
    }

    const fn clone_with_pending(&self, pending: Pending<'a>) -> Self {
        Self {
            pending,
            bit_buffer: self.bit_buffer,
            bits_valid: self.bits_valid,
            bits_used: self.bits_used,

            #[cfg(feature = "ZLIB_DEBUG")]
            compressed_len: self.compressed_len,
            #[cfg(feature = "ZLIB_DEBUG")]
            bits_sent: self.bits_sent,
        }
    }

    fn flush_bits(&mut self) {
        debug_assert!(self.bits_valid <= 64);
        let removed = self.bits_valid.saturating_sub(7).next_multiple_of(8);
        let keep_bytes = self.bits_valid / 8; // can never divide by zero

        let src = &self.bit_buffer.to_le_bytes();
        self.pending.extend(&src[..keep_bytes as usize]);

        self.bits_valid -= removed;
        self.bit_buffer = self.bit_buffer.checked_shr(u32::from(removed)).unwrap_or(0);
    }

    fn emit_align(&mut self) {
        debug_assert!(self.bits_valid <= 64);
        let keep_bytes = self.bits_valid.div_ceil(8);
        let src = &self.bit_buffer.to_le_bytes();
        self.pending.extend(&src[..keep_bytes as usize]);

        self.bits_used = match self.bits_valid {
            0 => 8,
            _ => ((self.bits_valid - 1) & 7) + 1,
        };
        self.bit_buffer = 0;
        self.bits_valid = 0;

        #[cfg(feature = "ZLIB_DEBUG")]
        self.sent_bits_align();
    }

    // These counters do not exist in release builds. Keep their updates
    // feature-gated rather than exposing release-build no-op methods.
    #[cfg(feature = "ZLIB_DEBUG")]
    fn cmpr_bits_add(&mut self, len: usize) {
        self.compressed_len += len;
    }

    #[cfg(feature = "ZLIB_DEBUG")]
    fn cmpr_bits_align(&mut self) {
        self.compressed_len = self.compressed_len.next_multiple_of(8);
    }

    #[cfg(feature = "ZLIB_DEBUG")]
    fn sent_bits_add(&mut self, len: usize) {
        self.bits_sent += len;
    }

    #[cfg(feature = "ZLIB_DEBUG")]
    fn sent_bits_align(&mut self) {
        self.bits_sent = self.bits_sent.next_multiple_of(8);
    }

    #[inline]
    fn send_bits(&mut self, val: u64, len: u8) {
        debug_assert!(len <= 64);
        debug_assert!(self.bits_valid <= 64);
        debug_assert!(
            len == 64 || val >> len == 0,
            "bit values must fit their width"
        );

        let total_bits = len + self.bits_valid;

        trace!(" l {:>2} v {:>4x} ", len, val);
        #[cfg(feature = "ZLIB_DEBUG")]
        self.sent_bits_add(usize::from(len));

        if total_bits < Self::BIT_BUF_SIZE {
            self.bit_buffer |= val << self.bits_valid;
            self.bits_valid = total_bits;
        } else {
            self.send_bits_overflow(val, total_bits);
        }
    }

    fn send_bits_overflow(&mut self, val: u64, total_bits: u8) {
        if self.bits_valid == Self::BIT_BUF_SIZE {
            self.pending.extend(&self.bit_buffer.to_le_bytes());
            self.bit_buffer = val;
        } else {
            self.bit_buffer |= val << self.bits_valid;
            self.pending.extend(&self.bit_buffer.to_le_bytes());
            self.bit_buffer = val >> (Self::BIT_BUF_SIZE - self.bits_valid);
        }
        self.bits_valid = total_bits - Self::BIT_BUF_SIZE;
    }

    fn send_code(&mut self, code: usize, tree: &[Value]) {
        let bits = PackedBits::code(tree[code]);
        self.send_bits(bits.value, bits.len);
    }

    /// Send one empty static block to give enough lookahead for inflate.
    /// This takes 10 bits, of which 7 may remain in the bit buffer.
    pub fn align(&mut self) {
        self.emit_tree(BlockType::StaticTrees, false);
        self.emit_end_block(&STATIC_LTREE, false);
        self.flush_bits();
    }

    pub(crate) fn emit_tree(&mut self, block_type: BlockType, is_last_block: bool) {
        let header_bits = ((block_type as u64) << 1) | u64::from(is_last_block);
        self.send_bits(header_bits, 3);
        trace!("\n--- Emit Tree: Last: {}\n", is_last_block as u8);
    }

    pub(crate) fn emit_end_block_and_align(&mut self, ltree: &[Value], is_last_block: bool) {
        self.emit_end_block(ltree, is_last_block);

        if is_last_block {
            self.emit_align();
        }
    }

    fn emit_end_block(&mut self, ltree: &[Value], _is_last_block: bool) {
        const END_BLOCK: usize = 256;
        self.send_code(END_BLOCK, ltree);

        trace!(
            "\n+++ Emit End Block: Last: {} Pending: {} Total Out: {}\n",
            _is_last_block as u8,
            self.pending.pending().len(),
            "<unknown>"
        );
    }

    pub(crate) fn emit_lit(&mut self, ltree: &[Value], c: u8) -> u16 {
        self.send_code(c as usize, ltree);

        #[cfg(feature = "ZLIB_DEBUG")]
        if let Some(c) = char::from_u32(c as u32) {
            if isgraph(c as u8) {
                trace!(" '{}' ", c);
            }
        }

        ltree[c as usize].len()
    }

    pub(crate) fn emit_dist(
        &mut self,
        ltree: &[Value],
        dtree: &[Value],
        lc: u8,
        dist: u16,
    ) -> usize {
        let mut bits = encode_len(ltree, lc);
        bits.append(encode_dist(dtree, dist));
        self.send_bits(bits.value, bits.len);
        usize::from(bits.len)
    }

    pub(crate) fn emit_dist_static(&mut self, lc: usize, dist: u16) -> usize {
        let mut bits = PackedBits::code(trees_tbl::STATIC_LTREE_ENCODINGS[lc]);
        bits.append(encode_dist(&trees_tbl::STATIC_DTREE, dist));
        self.send_bits(bits.value, bits.len);
        usize::from(bits.len)
    }

    fn compress_block_help(&mut self, sym_buf: &SymBuf, ltree: &[Value], dtree: &[Value]) {
        for (dist, lc) in sym_buf.iter() {
            match dist {
                0 => self.emit_lit(ltree, lc) as usize,
                _ => self.emit_dist(ltree, dtree, lc, dist),
            };
        }

        self.emit_end_block(ltree, false);
    }

    fn send_tree(&mut self, tree: &[Value], bl_tree: &[Value], max_code: usize) {
        for_each_code_length_run(tree, max_code, |run| match run {
            CodeLengthRun::Literal { length, count } => {
                for _ in 0..count {
                    self.send_code(usize::from(length), bl_tree);
                }
            }
            CodeLengthRun::RepeatPrevious {
                length,
                count,
                emit_initial,
            } => {
                if emit_initial {
                    self.send_code(usize::from(length), bl_tree);
                }
                self.send_code(REP_3_6, bl_tree);
                self.send_bits(u64::from(count - 3), 2);
            }
            CodeLengthRun::RepeatZero { code, count } => {
                self.send_code(code, bl_tree);
                let (base, bits) = if code == REPZ_3_10 { (3, 3) } else { (11, 7) };
                self.send_bits(u64::from(count - base), bits);
            }
        });
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatchCandidate {
    pub(crate) match_start: u16,
    pub(crate) match_length: u16,
    pub(crate) strstart: u16,
    pub(crate) orgstart: u16,
}

impl MatchCandidate {
    pub(crate) const EMPTY: Self = Self {
        match_start: 0,
        match_length: 0,
        strstart: 0,
        orgstart: 0,
    };
}

#[repr(C, align(64))]
pub struct State<'a> {
    status: Status,

    last_flush: i8, /* value of flush param for previous deflate call */

    pub(crate) wrap: i8, /* bit 0 true for zlib, bit 1 true for gzip */

    pub(crate) strategy: Strategy,
    pub(crate) level: i8,

    /// Whether or not a block is currently open for the QUICK deflation scheme.
    /// 0 if the block is closed, 1 if there is an active block, or 2 if there
    /// is an active block and it is the last block.
    pub(crate) block_open: u8,

    pub(crate) hash_calc_variant: HashCalcVariant,

    pub(crate) match_available: bool, /* set if previous match exists */

    /// Use a faster search when the previous match is longer than this
    pub(crate) good_match: u16,

    /// Stop searching when current match exceeds this
    pub(crate) nice_match: u16,

    pub(crate) match_start: Pos, /* start of matching string */
    pub(crate) prev_match: Pos,  /* previous match */
    pub(crate) strstart: usize,  /* start of string to insert */

    pub(crate) window: Window<'a>,
    pub(crate) w_size: usize, /* LZ77 window size (32K by default) */

    pub(crate) lookahead: usize, /* number of valid bytes ahead in window */

    _cache_line_0: (),

    /// prev[N], where N is an offset in the current window, contains the offset in the window
    /// of the previous 4-byte sequence that hashes to the same value as the 4-byte sequence
    /// starting at N. Together with head, prev forms a chained hash table that can be used
    /// to find earlier strings in the window that are potential matches for new input being
    /// deflated.
    pub(crate) prev: WeakSliceMut<'a, u16>,
    /// head[H] contains the offset of the last 4-character sequence seen so far in
    /// the current window that hashes to H (as calculated using the `hash_calc_variant`).
    pub(crate) head: WeakArrayMut<'a, u16, HASH_SIZE>,

    /// Length of the best match at previous step. Matches not greater than this
    /// are discarded. This is used in the lazy match evaluation.
    pub(crate) prev_length: u16,

    /// To speed up deflation, hash chains are never searched beyond this length.
    /// A higher limit improves compression ratio but degrades the speed.
    pub(crate) max_chain_length: u16,

    /// Insert new strings in the hash table only when the current match is no longer than this.
    /// This saves work at lower compression levels at the cost of some compression ratio.
    pub(crate) max_insert_length: u16,

    /// number of string matches in current block
    /// NOTE: this is a saturating 8-bit counter, to help keep the struct compact. The code that
    /// makes decisions based on this field only cares whether the count is greater than 2, so
    /// an 8-bit counter is sufficient.
    pub(crate) matches: u8,

    /// Window position at the beginning of the current output block. Gets
    /// negative when the window is moved backwards.
    pub(crate) block_start: isize,

    pub(crate) sym_buf: SymBuf<'a>,

    _cache_line_1: (),

    /// Size of match buffer for literals/lengths.  There are 4 reasons for
    /// limiting `lit_bufsize` to 64K:
    ///   - frequencies can be kept in 16 bit counters
    ///   - if compression is not successful for the first block, all input
    ///     data is still in the window so we can still emit a stored block even
    ///     when input comes from standard input.  (This can also be done for
    ///     all blocks if `lit_bufsize` is not greater than 32K.)
    ///   - if compression is not successful for a file smaller than 64K, we can
    ///     even emit a stored file instead of a stored block (saving 5 bytes).
    ///     This is applicable only for zip (not gzip or zlib).
    ///   - creating new Huffman trees less frequently may not provide fast
    ///     adaptation to changes in the input data statistics. (Take for
    ///     example a binary file with poorly compressible code followed by
    ///     a highly compressible string table.) Smaller buffer sizes give
    ///     fast adaptation but have of course the overhead of transmitting
    ///     trees more frequently.
    ///   - I can't count above 4
    lit_bufsize: usize,

    /// Actual size of window: 2*`w_size`, except when the user input buffer is directly used as sliding window.
    pub(crate) window_size: usize,

    bit_writer: BitWriter<'a>,

    _cache_line_2: (),

    /// bit length of current block with optimal trees
    opt_len: usize,
    /// bit length of current block with static trees
    static_len: usize,

    /// bytes at end of window left to insert
    pub(crate) insert: usize,

    ///  hash index of string to be inserted
    pub(crate) ins_h: u32,

    /// Attempt to find a better lazy match only while the current match is shorter than this.
    /// Slow compression uses this less frequently than the hot insertion limit above, so keep it
    /// in the colder state line rather than perturbing the hand-packed match-search cache line.
    pub(crate) max_lazy_match: u16,

    /// Caller-owned gzip header. zlib permits `deflateCopy()` to duplicate this pointer, so this
    /// must not be represented as a long-lived Rust `&mut` reference.
    gzhead: Option<GzipHeaderRef<'a>>,
    gzindex: usize,

    /// Medium's speculative next match must survive an output-blocking return. Keeping it here
    /// makes compression decisions independent of caller output chunking. It occupies bytes that
    /// were previously explicit alignment padding, so the hand-packed cache-line layout is stable.
    pub(crate) medium_next_match: MatchCandidate,
    /// Working copy of `z_stream::data_type`. Kept in compression state so block encoding is
    /// independent of caller I/O and synchronized once at the outer compression-call boundary.
    pub(crate) data_type: core::ffi::c_int,
    padding_0: [u8; 4],

    _cache_line_3: (),

    crc_fold: crate::crc32::Crc32Fold,

    /// Original aligned storage pointer, retained for matching typed deallocation.
    allocation_start: NonNull<AlignedAllocationBlock>,
    /// Total size of the aligned storage allocation in bytes.
    total_allocation_size: usize,

    l_desc: TreeDesc<HEAP_SIZE>,             /* literal and length tree */
    d_desc: TreeDesc<{ 2 * D_CODES + 1 }>,   /* distance tree */
    bl_desc: TreeDesc<{ 2 * BL_CODES + 1 }>, /* Huffman tree for bit lengths */
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
#[cfg_attr(feature = "__internal-fuzz", derive(arbitrary::Arbitrary))]
pub enum Strategy {
    #[default]
    Default = 0,
    Filtered = 1,
    HuffmanOnly = 2,
    Rle = 3,
    Fixed = 4,
}

impl TryFrom<i32> for Strategy {
    type Error = ();

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Default),
            1 => Ok(Self::Filtered),
            2 => Ok(Self::HuffmanOnly),
            3 => Ok(Self::Rle),
            4 => Ok(Self::Fixed),
            _ => Err(()),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum DataType {
    Binary = 0,
    Text = 1,
    Unknown = 2,
}

impl<'a> State<'a> {
    pub const BIT_BUF_SIZE: u8 = BitWriter::BIT_BUF_SIZE;

    /// Clone stream state while rebinding every view that belongs to the destination allocation.
    ///
    /// Keeping this as an explicit field-wise construction makes new state fields a compile-time
    /// clone-policy decision instead of silently including them in a byte-for-byte state copy.
    fn clone_rebound(
        &self,
        buffers: DeflateAllocBuffers<'a>,
        allocation_start: NonNull<AlignedAllocationBlock>,
        total_allocation_size: usize,
    ) -> Self {
        let DeflateAllocBuffers {
            state_ptr: _,
            window,
            prev,
            head,
            pending,
            sym_buf,
        } = buffers;
        Self {
            status: self.status,
            last_flush: self.last_flush,
            wrap: self.wrap,
            strategy: self.strategy,
            level: self.level,
            block_open: self.block_open,
            hash_calc_variant: self.hash_calc_variant,
            match_available: self.match_available,
            good_match: self.good_match,
            nice_match: self.nice_match,
            match_start: self.match_start,
            prev_match: self.prev_match,
            strstart: self.strstart,
            window,
            w_size: self.w_size,
            lookahead: self.lookahead,
            _cache_line_0: (),
            prev,
            head,
            prev_length: self.prev_length,
            max_chain_length: self.max_chain_length,
            max_insert_length: self.max_insert_length,
            matches: self.matches,
            block_start: self.block_start,
            sym_buf,
            _cache_line_1: (),
            lit_bufsize: self.lit_bufsize,
            window_size: self.window_size,
            bit_writer: self.bit_writer.clone_with_pending(pending),
            _cache_line_2: (),
            opt_len: self.opt_len,
            static_len: self.static_len,
            insert: self.insert,
            ins_h: self.ins_h,
            max_lazy_match: self.max_lazy_match,
            gzhead: self.gzhead,
            gzindex: self.gzindex,
            medium_next_match: self.medium_next_match,
            data_type: self.data_type,
            padding_0: self.padding_0,
            _cache_line_3: (),
            crc_fold: self.crc_fold,
            allocation_start,
            total_allocation_size,
            l_desc: self.l_desc.clone(),
            d_desc: self.d_desc.clone(),
            bl_desc: self.bl_desc.clone(),
        }
    }

    /// Encode the resident token block into pending bytes without touching caller I/O.
    pub(crate) fn encode_block(&mut self, is_last: bool) {
        encode_block_core(self, is_last);
    }

    // log2(w_size)  (in the range MIN_WBITS..=MAX_WBITS)
    pub(crate) const fn w_bits(&self) -> u32 {
        self.w_size.trailing_zeros()
    }

    pub(crate) const fn w_mask(&self) -> usize {
        self.w_size - 1
    }

    pub(crate) const fn max_dist(&self) -> usize {
        self.w_size - MIN_LOOKAHEAD
    }

    /// Return the representable backward distance to an existing hash-chain position.
    ///
    /// Hash position zero may be used by the quick path as an actual history index;
    /// other match producers separately reject it when treating zero as an empty sentinel.
    /// Rejection is conservative: absent/out-of-range entries cannot yield invalid
    /// or unrepresentable backreferences.
    #[inline]
    pub(crate) fn recent_hash_distance(&self, head: u16) -> Option<u16> {
        let distance = self.strstart.checked_sub(usize::from(head))?;
        if distance == 0 || distance > self.max_dist() {
            return None;
        }
        u16::try_from(distance).ok()
    }

    #[inline]
    pub(crate) const fn update_hash(&self, h: u32, val: u32) -> u32 {
        match self.hash_calc_variant {
            HashCalcVariant::Standard => StandardHashCalc::update_hash(h, val),
            HashCalcVariant::Roll => RollHashCalc::update_hash(h, val),
        }
    }

    #[inline]
    pub(crate) fn quick_insert_string(&mut self, string: usize) -> u16 {
        match self.hash_calc_variant {
            HashCalcVariant::Standard => StandardHashCalc::quick_insert_string(self, string),
            HashCalcVariant::Roll => RollHashCalc::quick_insert_string(self, string),
        }
    }

    #[inline]
    pub(crate) fn insert_string(&mut self, string: usize, count: usize) {
        match self.hash_calc_variant {
            HashCalcVariant::Standard => StandardHashCalc::insert_string(self, string, count),
            HashCalcVariant::Roll => RollHashCalc::insert_string(self, string, count),
        }
    }

    #[inline]
    pub(crate) fn tally_lit(&mut self, unmatched: u8) -> bool {
        Self::tally_lit_help(&mut self.sym_buf, &mut self.l_desc, unmatched)
    }

    // This helper is to work around an ownership issue in algorithm/medium.
    pub(crate) fn tally_lit_help(
        sym_buf: &mut SymBuf,
        l_desc: &mut TreeDesc<HEAP_SIZE>,
        unmatched: u8,
    ) -> bool {
        const _VERIFY: () = {
            // Verify during compilation that even the largest possible value
            // of unmatched will fit within the expected range.
            assert!(
                u8::MAX as usize <= STD_MAX_MATCH - STD_MIN_MATCH,
                "tally_lit: bad literal"
            );
        };

        sym_buf.push_lit(unmatched);

        *l_desc.dyn_tree[unmatched as usize].freq_mut() += 1;

        // signal that the current block should be flushed
        sym_buf.should_flush_block()
    }

    const fn d_code(dist: usize) -> u8 {
        const _VERIFY: () = {
            // Verify during compilation that every DIST_CODE value is < D_CODES.
            let mut i = 0;
            while i < trees_tbl::DIST_CODE.len() {
                assert!((trees_tbl::DIST_CODE[i] as usize) < D_CODES);
                i += 1;
            }
        };

        let index = if dist < 256 { dist } else { 256 + (dist >> 7) };
        self::trees_tbl::DIST_CODE[index]
    }

    #[inline]
    pub(crate) fn tally_dist(&mut self, dist: usize, len: usize) -> bool {
        let max_distance = self.max_dist();
        self.tally_dist_with_limit(dist, len, max_distance)
    }

    /// Tally a match found by a complete-buffer matcher that can safely use the entire configured
    /// DEFLATE history window rather than the streaming matchfinder's lookahead-reserved limit.
    #[inline]
    pub(crate) fn tally_dist_full_window(&mut self, dist: usize, len: usize) -> bool {
        let max_distance = self.w_mask() + 1;
        self.tally_dist_with_limit(dist, len, max_distance)
    }

    #[inline]
    fn tally_dist_with_limit(&mut self, dist: usize, len: usize, max_distance: usize) -> bool {
        // Validate the entire token before writing any of its three bytes
        // or incrementing Huffman frequencies. A distance is one-based and
        // must fit both the configured DEFLATE window and the wire encoding.
        assert!(
            (1..=max_distance.min(1usize << crate::MAX_WBITS)).contains(&dist),
            "tally_dist: bad match"
        );
        assert!(
            len <= STD_MAX_MATCH - STD_MIN_MATCH,
            "tally_dist: bad match length"
        );
        let Ok(encoded_dist) = u16::try_from(dist) else {
            unreachable!("distance was checked against u16::MAX");
        };
        let Ok(encoded_len) = u8::try_from(len) else {
            unreachable!("match length was checked against the 255-byte DEFLATE delta");
        };

        self.sym_buf.push_dist(encoded_dist, encoded_len);
        self.matches = self.matches.saturating_add(1);
        let dist = dist - 1;

        let index = usize::from(self::trees_tbl::LENGTH_CODE[len]) + LITERALS + 1;
        *self.l_desc.dyn_tree[index].freq_mut() += 1;

        *self.d_desc.dyn_tree[Self::d_code(dist) as usize].freq_mut() += 1;

        // signal that the current block should be flushed
        self.sym_buf.should_flush_block()
    }

    fn detect_data_type(dyn_tree: &[Value]) -> DataType {
        // set bits 0..6, 14..25, and 28..31
        // 0xf3ffc07f = binary 11110011111111111100000001111111
        const NON_TEXT: u64 = 0xf3ff_c07f;
        let mut mask = NON_TEXT;

        /* Check for non-textual bytes. */
        for value in &dyn_tree[0..32] {
            if (mask & 1) != 0 && value.freq() != 0 {
                return DataType::Binary;
            }

            mask >>= 1;
        }

        /* Check for textual bytes. */
        if dyn_tree[9].freq() != 0 || dyn_tree[10].freq() != 0 || dyn_tree[13].freq() != 0 {
            return DataType::Text;
        }

        if dyn_tree[32..LITERALS].iter().any(|v| v.freq() != 0) {
            return DataType::Text;
        }

        // there are no explicit text or non-text bytes. The stream is either empty or has only
        // tolerated bytes
        DataType::Binary
    }

    fn compress_block_static_trees(&mut self) {
        let ltree = self::trees_tbl::STATIC_LTREE.as_slice();
        for (dist, lc) in self.sym_buf.iter() {
            match dist {
                0 => self.bit_writer.emit_lit(ltree, lc) as usize,
                _ => self.bit_writer.emit_dist_static(usize::from(lc), dist),
            };
        }

        self.bit_writer.emit_end_block(ltree, false);
    }

    fn compress_block_dynamic_trees(&mut self) {
        self.bit_writer.compress_block_help(
            &self.sym_buf,
            &self.l_desc.dyn_tree,
            &self.d_desc.dyn_tree,
        );
    }

    fn header(&self) -> u16 {
        // preset dictionary flag in zlib header
        const PRESET_DICT: u16 = 0x20;

        // The deflate compression method (the only one supported in this version)
        const Z_DEFLATED: u16 = 8;

        let dict = match self.strstart {
            0 => 0,
            _ => PRESET_DICT,
        };

        let h = ((Z_DEFLATED + ((self.w_bits() as u16 - 8) << 4)) << 8)
            | (self.level_flags() << 6)
            | dict;

        h + 31 - (h % 31)
    }

    fn level_flags(&self) -> u16 {
        if self.strategy >= Strategy::HuffmanOnly || self.level < 2 {
            0
        } else if self.level < 6 {
            1
        } else if self.level == 6 {
            2
        } else {
            3
        }
    }

    fn zng_tr_init(&mut self) {
        // Descriptors are associated with their actual alphabets at
        // construction and retain them across reset/copy operations.
        self.bit_writer.bit_buffer = 0;
        self.bit_writer.bits_valid = 0;
        self.bit_writer.bits_used = 0;

        #[cfg(feature = "ZLIB_DEBUG")]
        {
            self.bit_writer.compressed_len = 0;
            self.bit_writer.bits_sent = 0;
        }

        // Initialize the first block of the first file:
        self.init_block();
    }

    /// initializes a new block
    fn init_block(&mut self) {
        // Initialize the trees.
        // TODO would a memset also work here?

        for value in &mut self.l_desc.dyn_tree[..L_CODES] {
            *value.freq_mut() = 0;
        }

        for value in &mut self.d_desc.dyn_tree[..D_CODES] {
            *value.freq_mut() = 0;
        }

        for value in &mut self.bl_desc.dyn_tree[..BL_CODES] {
            *value.freq_mut() = 0;
        }

        // end of block literal code
        const END_BLOCK: usize = 256;

        *self.l_desc.dyn_tree[END_BLOCK].freq_mut() = 1;
        self.opt_len = 0;
        self.static_len = 0;
        self.sym_buf.clear();
        self.matches = 0;
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    Init = 1,

    GZip = 4,
    Extra = 5,
    Name = 6,
    Comment = 7,
    Hcrc = 8,

    Busy = 2,
    Finish = 3,
}

const fn rank_flush(f: i8) -> i8 {
    // rank Z_BLOCK between Z_NO_FLUSH and Z_PARTIAL_FLUSH
    ((f) * 2) - (if (f) > 4 { 9 } else { 0 })
}

#[derive(Debug)]
pub enum BlockState {
    /// block not completed, need more input or more output
    NeedMore = 0,
    /// block flush performed
    BlockDone = 1,
    /// finish started, need only more output at next deflate
    FinishStarted = 2,
    /// finish done, accept no more input or output
    FinishDone = 3,
    /// The compressor or scheduler encountered an invalid internal state.
    InvalidState = 4,
}

// Maximum stored block length in deflate format (not including header).
pub const MAX_STORED: usize = 65535; // so u16::max

/// Serialize the length and one's complement shared by pending and direct
/// stored-block emission. Validate the complete field before either caller
/// modifies its output: a stored block can hold at most 65,535 bytes.
#[inline]
fn stored_block_length_fields(length: usize) -> [u8; 4] {
    assert!(
        length <= MAX_STORED,
        "DEFLATE stored block payload must fit in 16 bits"
    );
    let bytes = length.to_le_bytes();
    let complement = (!length).to_le_bytes();
    [bytes[0], bytes[1], complement[0], complement[1]]
}

fn read_buf_window_core(
    state: &mut State<'_>,
    input: &mut InputCursor,
    checksum: &mut StreamChecksum,
    offset: usize,
    size: usize,
) -> usize {
    let len = Ord::min(input.avail_in as usize, size);

    if len == 0 {
        return 0;
    }

    // Copy first, then checksum the stable window bytes. We intentionally never construct a slice
    // over live caller input because zlib permits it to be concurrently mutated.
    input.copy_to_window(&mut state.window, offset..offset + len);
    let data = &state.window.filled()[offset..][..len];

    if state.wrap == 2 {
        state.crc_fold.fold(data);
    } else if state.wrap == 1 {
        checksum.update_adler(data);
    }

    len
}

pub enum BlockType {
    StoredBlock = 0,
    StaticTrees = 1,
    DynamicTrees = 2,
}

/// Emit the header for a stored block whose payload will be written separately.
///
/// Validates the 16-bit LEN/NLEN field before altering the pending bits.
/// Both the direct-output and pending-output implementations use this same path.
#[inline]
fn emit_stored_block_header(state: &mut State, stored_len: usize, is_last: bool) {
    let length_fields = stored_block_length_fields(stored_len);
    state.bit_writer.emit_tree(BlockType::StoredBlock, is_last);
    state.bit_writer.emit_align();
    #[cfg(feature = "ZLIB_DEBUG")]
    state.bit_writer.cmpr_bits_align();
    state.bit_writer.pending.extend(&length_fields);
    #[cfg(feature = "ZLIB_DEBUG")]
    state.bit_writer.cmpr_bits_add(32);
    #[cfg(feature = "ZLIB_DEBUG")]
    state.bit_writer.sent_bits_add(32);
}

/// Emit a single DEFLATE stored block using initialized bytes from the window.
///
/// # Panics
///
/// Panics if the input range is outside the initialized window, the stored payload is
/// larger than the 16-bit DEFLATE length field, or the pending-output buffer is too small.
pub fn zng_tr_stored_block(
    state: &mut State,
    window_range: core::ops::Range<usize>,
    is_last: bool,
) {
    // Validate both the window range and format width before emitting the header.
    let stored_len = state.window.filled()[window_range.clone()].len();
    emit_stored_block_header(state, stored_len, is_last);
    if stored_len > 0 {
        state
            .bit_writer
            .pending
            .extend(&state.window.filled()[window_range]);
        #[cfg(feature = "ZLIB_DEBUG")]
        {
            let payload_bits = stored_len << 3;
            state.bit_writer.cmpr_bits_add(payload_bits);
            state.bit_writer.sent_bits_add(payload_bits);
        }
    }
}

/// The minimum match length mandated by the deflate standard
pub const STD_MIN_MATCH: usize = 3;
/// The maximum match length mandated by the deflate standard
pub const STD_MAX_MATCH: usize = 258;

/// The minimum wanted match length, affects `deflate_quick`, `deflate_fast`, `deflate_medium` and `deflate_slow`
pub const WANT_MIN_MATCH: usize = 4;

pub const MIN_LOOKAHEAD: usize = STD_MAX_MATCH + STD_MIN_MATCH + 1;

#[inline]
pub fn fill_window_core(
    state: &mut State<'_>,
    input: &mut InputCursor,
    checksum: &mut StreamChecksum,
) {
    fill_window_core_with_hash_maintenance(state, input, checksum, true);
}

#[inline]
pub fn fill_window_core_with_hash_maintenance(
    state: &mut State<'_>,
    input: &mut InputCursor,
    checksum: &mut StreamChecksum,
    maintain_hash_chain: bool,
) {
    debug_assert!(state.lookahead < MIN_LOOKAHEAD);

    let wsize = state.w_size;

    loop {
        let mut more = state.window_size - state.lookahead - state.strstart;

        // If the window is almost full and there is insufficient lookahead,
        // move the upper half to the lower one to make room in the upper half.
        if state.strstart >= wsize + state.max_dist() {
            // Shift only the logically initialized upper-half bytes down.
            state.window.slide_down(wsize);

            // A match before the retained half of the window is no longer
            // valid. Sliding itself uses a full-width displacement and
            // saturates the 16-bit position, just like the hash chains.
            if usize::from(state.match_start) < wsize {
                state.prev_length = 0;
            }
            state.match_start = self::slide_hash::slide_position(state.match_start, wsize);

            state.strstart -= wsize; /* we now have strstart >= MAX_DIST */
            let Some(slid_origin) = state.block_start.checked_sub_unsigned(wsize) else {
                unreachable!("window slide exceeds signed block-origin range");
            };
            state.block_start = slid_origin;
            state.insert = Ord::min(state.insert, state.strstart);

            if maintain_hash_chain {
                self::slide_hash::slide_hash(state);
            }

            more += wsize;
        }

        if input.avail_in == 0 {
            break;
        }

        // If there was no sliding:
        //    strstart <= WSIZE+MAX_DIST-1 && lookahead <= MIN_LOOKAHEAD - 1 &&
        //    more == window_size - lookahead - strstart
        // => more >= window_size - (MIN_LOOKAHEAD-1 + WSIZE + MAX_DIST-1)
        // => more >= window_size - 2*WSIZE + 2
        // In the BIG_MEM or MMAP case (not yet supported),
        //   window_size == input_size + MIN_LOOKAHEAD  &&
        //   strstart + s->lookahead <= input_size => more >= MIN_LOOKAHEAD.
        // Otherwise, window_size == 2*WSIZE so more >= 2.
        // If there was sliding, more >= WSIZE. So in all cases, more >= 2.
        assert!(more >= 2, "more < 2");

        let n = read_buf_window_core(
            state,
            input,
            checksum,
            state.strstart + state.lookahead,
            more,
        );

        state.lookahead += n;

        // Initialize the hash value now that we have some input:
        if maintain_hash_chain && state.lookahead + state.insert >= STD_MIN_MATCH {
            let string = state.strstart - state.insert;
            // The rolling seed must match the hash representation used to index
            // the live head/prev tables, not the current (possibly tuned) chain budget.
            if matches!(state.hash_calc_variant, HashCalcVariant::Roll) {
                let v0 = u32::from(state.window.filled()[string]);
                let v1 = u32::from(state.window.filled()[string + 1]);
                state.ins_h = state.update_hash(v0, v1);
            } else if string >= 1 {
                state.quick_insert_string(string + 2 - STD_MIN_MATCH);
            }
            let mut count = state.insert;
            if state.lookahead == 1 {
                count -= 1;
            }
            if count > 0 {
                state.insert_string(string, count);
                state.insert -= count;
            }
        }

        // If the whole input has less than STD_MIN_MATCH bytes, ins_h is garbage,
        // but this is not important since only literal bytes will be emitted.

        if !(state.lookahead < MIN_LOOKAHEAD && input.avail_in != 0) {
            break;
        }
    }

    assert!(
        state.strstart <= state.window_size - MIN_LOOKAHEAD,
        "not enough room for search"
    );
}

pub struct StaticTreeDesc {
    /// static tree or NULL
    pub(crate) static_tree: &'static [Value],
    /// extra bits for each code or NULL
    extra_bits: &'static [u8],
    /// base index for `extra_bits`
    extra_base: usize,
    /// max number of elements in the tree
    elems: usize,
    /// max bit length for the codes
    max_length: u16,
}

impl StaticTreeDesc {
    /// extra bits for each length code
    const EXTRA_LBITS: [u8; LENGTH_CODES] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
    ];

    /// extra bits for each distance code
    const EXTRA_DBITS: [u8; D_CODES] = [
        0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12,
        13, 13,
    ];

    /// extra bits for each bit length code
    const EXTRA_BLBITS: [u8; BL_CODES] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 7];

    /// The lengths of the bit length codes are sent in order of decreasing
    /// probability, to avoid transmitting the lengths for unused bit length codes.
    const BL_ORDER: [u8; BL_CODES] = [
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    ];

    pub(crate) const L: Self = Self {
        static_tree: &self::trees_tbl::STATIC_LTREE,
        extra_bits: &Self::EXTRA_LBITS,
        extra_base: LITERALS + 1,
        elems: L_CODES,
        max_length: MAX_BITS as u16,
    };

    pub(crate) const D: Self = Self {
        static_tree: &self::trees_tbl::STATIC_DTREE,
        extra_bits: &Self::EXTRA_DBITS,
        extra_base: 0,
        elems: D_CODES,
        max_length: MAX_BITS as u16,
    };

    pub(crate) const BL: Self = Self {
        static_tree: &[],
        extra_bits: &Self::EXTRA_BLBITS,
        extra_base: 0,
        elems: BL_CODES,
        max_length: MAX_BL_BITS as u16,
    };
}

#[derive(Clone)]
pub struct TreeDesc<const N: usize> {
    dyn_tree: [Value; N],
    max_code: usize,
    stat_desc: &'static StaticTreeDesc,
}

impl<const N: usize> TreeDesc<N> {
    /// Construct a tree with its fixed alphabet descriptor from the start.
    /// No transient "empty descriptor" is possible during compression or
    /// when a stream is cloned or reset.
    const fn new(stat_desc: &'static StaticTreeDesc) -> Self {
        assert!(stat_desc.elems >= 2 && stat_desc.elems <= L_CODES && N >= 2 * stat_desc.elems - 1);
        Self {
            dyn_tree: [Value::new(0, 0); N],
            max_code: 0,
            stat_desc,
        }
    }
}

fn build_tree<const N: usize>(
    desc: &mut TreeDesc<N>,
    optimal_bits: &mut usize,
    static_bits: &mut usize,
) {
    let tree = &mut desc.dyn_tree;
    let stree = desc.stat_desc.static_tree;
    let elements = desc.stat_desc.elems;
    // Each internal Huffman node consumes one more slot than the leaves.
    // The descriptor must have space for every leaf and combined parent.
    assert!(
        elements >= 2 && elements <= L_CODES && N >= 2 * elements - 1,
        "Huffman tree descriptor does not fit its alphabet"
    );
    assert!(
        (1..=MAX_BITS).contains(&usize::from(desc.stat_desc.max_length)),
        "Huffman tree descriptor has an invalid code-length bound"
    );

    let mut heap = Heap::new();
    let mut max_code = heap.initialize(&mut tree[..elements]);

    // The pkzip format requires that at least one distance code exists,
    // and that at least one bit should be sent even if there is only one
    // possible code. So to avoid special checks later on we force at least
    // two codes of non zero frequency. These synthetic symbols are needed to
    // construct the tree but are never emitted, so discount their payload
    // cost after bit lengths have been generated. Doing that afterwards also
    // avoids the C implementation's transient unsigned underflow on empty trees.
    let mut synthetic_opt_len = 0usize;
    let mut synthetic_static_len = 0usize;
    while heap.active_len < 2 {
        let node = if max_code.is_none_or(|code| code < 2) {
            let next = max_code.map_or(0, |code| code + 1);
            max_code = Some(next);
            next
        } else {
            0
        };

        heap.push_leaf(node);
        *tree[node].freq_mut() = 1;
        synthetic_opt_len += 1;
        if !stree.is_empty() {
            synthetic_static_len += usize::from(stree[node].len());
        }
        /* node is 0 or 1 so it does not have extra bits */
    }

    // A DEFLATE Huffman alphabet always needs at least two symbols,
    // including synthetic leaves for an initially empty alphabet.
    let Some(max_code) = max_code else {
        unreachable!("the two-leaf Huffman tree must have a largest symbol");
    };
    desc.max_code = max_code;

    // The elements heap[heap_len/2+1 .. heap_len] are leaves of the tree,
    // establish sub-heaps of increasing lengths:
    heap.heapify(tree);

    heap.construct_huffman_tree(tree, elements);

    // At this point, the fields freq and dad are set. We can now
    // generate the bit lengths.
    let bl_count = gen_bitlen(&heap, desc, optimal_bits, static_bits);
    *optimal_bits -= synthetic_opt_len;
    *static_bits -= synthetic_static_len;

    // The field len is now set, we can generate the bit codes
    gen_codes(&mut desc.dyn_tree, max_code, &bl_count);
}

fn gen_bitlen<const N: usize>(
    heap: &Heap,
    desc: &mut TreeDesc<N>,
    optimal_bits: &mut usize,
    static_bits: &mut usize,
) -> [u16; MAX_BITS + 1] {
    let tree = &mut desc.dyn_tree;
    let max_code = desc.max_code;
    let stree = desc.stat_desc.static_tree;
    let extra = desc.stat_desc.extra_bits;
    let base = desc.stat_desc.extra_base;
    let max_length = desc.stat_desc.max_length;

    let mut bl_count = [0u16; MAX_BITS + 1];

    // In a first pass, compute the optimal bit lengths (which may
    // overflow in the case of the bit length tree).
    *tree[heap.root()].len_mut() = 0; /* root of the completed tree */

    // number of elements with bit length too large
    let mut overflow: i32 = 0;

    for &n in heap.non_root_nodes() {
        let mut bits = tree[usize::from(tree[n].dad())].len() + 1;

        if bits > max_length {
            bits = max_length;
            overflow += 1;
        }

        // We overwrite tree[n].Dad which is no longer needed
        *tree[n].len_mut() = bits;

        // not a leaf node
        if n > max_code {
            continue;
        }

        bl_count[usize::from(bits)] += 1;
        let mut xbits = 0;
        if n >= base {
            xbits = extra[n - base] as usize;
        }

        let f = tree[n].freq() as usize;
        *optimal_bits += f * (bits as usize + xbits);

        if !stree.is_empty() {
            *static_bits += f * (stree[n].len() as usize + xbits);
        }
    }

    if overflow == 0 {
        return bl_count;
    }

    /* Find the first bit length which could increase: */
    loop {
        let mut bits = usize::from(max_length) - 1;
        while bl_count[bits] == 0 {
            bits -= 1;
        }
        bl_count[bits] -= 1; /* move one leaf down the tree */
        bl_count[bits + 1] += 2; /* move one overflow item as its brother */
        bl_count[usize::from(max_length)] -= 1;
        /* The brother of the overflow item also moves one step up,
         * but this does not affect bl_count[max_length]
         */
        overflow -= 2;

        if overflow <= 0 {
            break;
        }
    }

    // Now recompute all bit lengths, scanning in increasing frequency.
    // h is still equal to HEAP_SIZE. (It is simpler to reconstruct all
    // lengths instead of fixing only the wrong ones. This idea is taken
    // from 'ar' written by Haruhiko Okumura.)
    let mut h = HEAP_SIZE;
    for bits in (1..=max_length).rev() {
        let mut n = bl_count[bits as usize];
        while n != 0 {
            h -= 1;
            let m = heap.nodes[h];
            if m > max_code {
                continue;
            }

            if tree[m].len() != bits {
                // Tracev((stderr, "code %d bits %d->%u\n", m, tree[m].Len, bits));
                // Cost arithmetic is in the host index domain. Multiplying
                // two u16 values *before* widening can overflow even though
                // the final bit cost is a perfectly valid usize.
                let frequency = usize::from(tree[m].freq());
                *optimal_bits += usize::from(bits) * frequency;
                *optimal_bits -= usize::from(tree[m].len()) * frequency;
                *tree[m].len_mut() = bits;
            }

            n -= 1;
        }
    }
    bl_count
}

/// Checks that symbol is a printing character (excluding space)
#[cfg(feature = "ZLIB_DEBUG")]
fn isgraph(c: u8) -> bool {
    (c > 0x20) && (c <= 0x7E)
}

fn gen_codes(tree: &mut [Value], max_code: usize, bl_count: &[u16]) {
    /* tree: the tree to decorate */
    /* max_code: largest code with non zero frequency */
    /* bl_count: number of codes at each bit length */
    let mut next_code = [0; MAX_BITS + 1]; /* next code value for each bit length */
    let mut code = 0; /* running code value */

    /* The distribution counts are first used to generate the code values
     * without bit reversal.
     */
    for bits in 1..=MAX_BITS {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }

    /* Check that the bit counts in bl_count are consistent. The last code
     * must be all ones.
     */
    assert!(
        code + bl_count[MAX_BITS] - 1 == (1 << MAX_BITS) - 1,
        "inconsistent bit counts"
    );

    trace!("\ngen_codes: max_code {max_code} ");

    for n in 0..=max_code {
        let len = tree[n].len();
        if len == 0 {
            continue;
        }

        /* Now reverse the bits */
        assert!((1..=15).contains(&len), "code length must be 1-15");
        *tree[n].code_mut() = next_code[len as usize].reverse_bits() >> (16 - len);
        next_code[len as usize] += 1;

        if tree != self::trees_tbl::STATIC_LTREE.as_slice() {
            trace!(
                "\nn {:>3} {} l {:>2} c {:>4x} ({:x}) ",
                n,
                u8::try_from(n)
                    .ok()
                    .filter(|&symbol| isgraph(symbol))
                    .map(char::from)
                    .unwrap_or(' '),
                len,
                tree[n].code(),
                next_code[len as usize] - 1
            );
        }
    }
}

/// repeat previous bit length 3-6 times (2 bits of repeat count)
const REP_3_6: usize = 16;

/// repeat a zero length 3-10 times  (3 bits of repeat count)
const REPZ_3_10: usize = 17;

/// repeat a zero length 11-138 times  (7 bits of repeat count)
const REPZ_11_138: usize = 18;

/// One shared representation of the RFC 1951 code-length RLE decisions.
///
/// The frequency pass and bitstream pass must agree on these events. Previously
/// they independently maintained their own zero/nonzero repeat state, which
/// allowed the Huffman model to drift from what the encoder actually emitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CodeLengthRun {
    Literal {
        length: u16,
        count: u16,
    },
    RepeatPrevious {
        length: u16,
        count: u16,
        emit_initial: bool,
    },
    RepeatZero {
        code: usize,
        count: u16,
    },
}

/// Visit code-length RLE decisions in wire order. The final sentinel is
/// virtual: callers no longer need to mutate the tree's guard entry.
fn for_each_code_length_run(tree: &[Value], max_code: usize, mut visit: impl FnMut(CodeLengthRun)) {
    assert!(max_code < tree.len(), "code-length tree bound is invalid");
    let mut previous = None;
    let mut next_length = tree[0].len();
    let mut count = 0u16;
    let (mut max_count, mut min_count) = if next_length == 0 {
        (138u16, 3u16)
    } else {
        (7u16, 4u16)
    };

    for index in 0..=max_code {
        let length = next_length;
        next_length = if index == max_code {
            u16::MAX
        } else {
            tree[index + 1].len()
        };
        count += 1;
        if count < max_count && length == next_length {
            continue;
        }

        let run = if count < min_count {
            CodeLengthRun::Literal { length, count }
        } else if length != 0 {
            let emit_initial = previous != Some(length);
            let repeat_count = count - u16::from(emit_initial);
            assert!(
                (3..=6).contains(&repeat_count),
                "nonzero code-length repetition exceeds DEFLATE bounds"
            );
            CodeLengthRun::RepeatPrevious {
                length,
                count: repeat_count,
                emit_initial,
            }
        } else if count <= 10 {
            CodeLengthRun::RepeatZero {
                code: REPZ_3_10,
                count,
            }
        } else {
            CodeLengthRun::RepeatZero {
                code: REPZ_11_138,
                count,
            }
        };
        visit(run);

        count = 0;
        previous = Some(length);
        if next_length == 0 {
            max_count = 138;
            min_count = 3;
        } else if length == next_length {
            max_count = 6;
            min_count = 3;
        } else {
            max_count = 7;
            min_count = 4;
        }
    }
}

fn scan_tree(bl_desc: &mut TreeDesc<{ 2 * BL_CODES + 1 }>, tree: &[Value], max_code: usize) {
    let bl_tree = &mut bl_desc.dyn_tree;
    for_each_code_length_run(tree, max_code, |run| match run {
        CodeLengthRun::Literal { length, count } => {
            *bl_tree[usize::from(length)].freq_mut() += count;
        }
        CodeLengthRun::RepeatPrevious {
            length,
            emit_initial,
            ..
        } => {
            if emit_initial {
                *bl_tree[usize::from(length)].freq_mut() += 1;
            }
            *bl_tree[REP_3_6].freq_mut() += 1;
        }
        CodeLengthRun::RepeatZero { code, .. } => {
            *bl_tree[code].freq_mut() += 1;
        }
    });
}

fn send_all_trees(state: &mut State, lcodes: usize, dcodes: usize, blcodes: usize) {
    assert!(
        lcodes >= 257 && dcodes >= 1 && blcodes >= 4,
        "not enough codes"
    );
    assert!(
        lcodes <= L_CODES && dcodes <= D_CODES && blcodes <= BL_CODES,
        "too many codes"
    );

    trace!("\nbl counts: ");
    state.bit_writer.send_bits(lcodes as u64 - 257, 5); /* not +255 as stated in appnote.txt */
    state.bit_writer.send_bits(dcodes as u64 - 1, 5);
    state.bit_writer.send_bits(blcodes as u64 - 4, 4); /* not -3 as stated in appnote.txt */

    for rank in 0..blcodes {
        trace!("\nbl code {:>2} ", StaticTreeDesc::BL_ORDER[rank]);
        state.bit_writer.send_bits(
            u64::from(state.bl_desc.dyn_tree[usize::from(StaticTreeDesc::BL_ORDER[rank])].len()),
            3,
        );
    }
    trace!("\nbl tree: sent {}", state.bit_writer.bits_sent);

    // literal tree
    state
        .bit_writer
        .send_tree(&state.l_desc.dyn_tree, &state.bl_desc.dyn_tree, lcodes - 1);
    trace!("\nlit tree: sent {}", state.bit_writer.bits_sent);

    // distance tree
    state
        .bit_writer
        .send_tree(&state.d_desc.dyn_tree, &state.bl_desc.dyn_tree, dcodes - 1);
    trace!("\ndist tree: sent {}", state.bit_writer.bits_sent);
}

/// Construct the Huffman tree for the bit lengths and return the index in
/// `bl_order` of the last bit length code to send.
fn build_bl_tree(state: &mut State) -> usize {
    /* Determine the bit length frequencies for literal and distance trees */

    scan_tree(
        &mut state.bl_desc,
        &state.l_desc.dyn_tree,
        state.l_desc.max_code,
    );

    scan_tree(
        &mut state.bl_desc,
        &state.d_desc.dyn_tree,
        state.d_desc.max_code,
    );

    /* Build the bit length tree in place. */
    build_tree(
        &mut state.bl_desc,
        &mut state.opt_len,
        &mut state.static_len,
    );

    /* opt_len now includes the length of the tree representations, except
     * the lengths of the bit lengths codes and the 5+5+4 bits for the counts.
     */

    /* Determine the number of bit length codes to send. The pkzip format
     * requires that at least 4 bit length codes be sent. (appnote.txt says
     * 3 but the actual value used is 4.)
     */
    let mut max_blindex = BL_CODES - 1;
    while max_blindex >= 3 {
        let index = StaticTreeDesc::BL_ORDER[max_blindex] as usize;
        if state.bl_desc.dyn_tree[index].len() != 0 {
            break;
        }

        max_blindex -= 1;
    }

    /* Update opt_len to include the bit length tree and counts */
    state.opt_len += 3 * (max_blindex + 1) + 5 + 5 + 4;
    trace!(
        "\ndyn trees: dyn {}, stat {}",
        state.opt_len,
        state.static_len
    );

    max_blindex
}

fn zng_tr_flush_block(
    state: &mut State<'_>,
    window_offset: Option<usize>,
    stored_len: u32,
    last: bool,
) {
    /* window_offset: offset of the input block into the window */
    /* stored_len: length of input block */
    /* last: one if this is the last block for a file */

    let mut opt_lenb;
    let static_lenb;
    let mut max_blindex = 0;

    if state.sym_buf.is_empty() {
        opt_lenb = 0;
        static_lenb = 0;
        state.static_len = 7;
    } else if state.level > 0 {
        if state.data_type == DataType::Unknown as i32 {
            state.data_type = State::detect_data_type(&state.l_desc.dyn_tree) as i32;
        }

        build_tree(&mut state.l_desc, &mut state.opt_len, &mut state.static_len);
        trace!(
            "\nlit data: dyn {}, stat {}",
            state.opt_len,
            state.static_len
        );

        build_tree(&mut state.d_desc, &mut state.opt_len, &mut state.static_len);
        trace!(
            "\ndist data: dyn {}, stat {}",
            state.opt_len,
            state.static_len
        );

        // Build the bit length tree for the above two trees, and get the index
        // in bl_order of the last bit length code to send.
        max_blindex = build_bl_tree(state);

        // Determine the best encoding. Compute the block lengths in bytes.
        opt_lenb = (state.opt_len + 3 + 7) >> 3;
        static_lenb = (state.static_len + 3 + 7) >> 3;

        trace!(
            "\nopt {}({}) stat {}({}) stored {} lit {} ",
            opt_lenb,
            state.opt_len,
            static_lenb,
            state.static_len,
            stored_len,
            state.sym_buf.iter().count()
        );

        if static_lenb <= opt_lenb || state.strategy == Strategy::Fixed {
            opt_lenb = static_lenb;
        }
    } else {
        assert!(window_offset.is_some(), "lost buf");
        /* force a stored block */
        opt_lenb = stored_len as usize + 5;
        static_lenb = stored_len as usize + 5;
    }

    if let Some(window_offset) = window_offset.filter(|_| stored_len as usize + 4 <= opt_lenb) {
        /* 4: two words for the lengths
         * The test buf != NULL is only necessary if LIT_BUFSIZE > WSIZE.
         * Otherwise we can't have processed more than WSIZE input bytes since
         * the last block flush, because compression would have been
         * successful. If LIT_BUFSIZE <= WSIZE, it is never too late to
         * transform a block into a stored block.
         */
        let range = window_offset..window_offset + stored_len as usize;
        zng_tr_stored_block(state, range, last);
    } else if static_lenb == opt_lenb {
        state.bit_writer.emit_tree(BlockType::StaticTrees, last);
        state.compress_block_static_trees();
    // cmpr_bits_add(s, s.static_len);
    } else {
        state.bit_writer.emit_tree(BlockType::DynamicTrees, last);
        send_all_trees(
            state,
            state.l_desc.max_code + 1,
            state.d_desc.max_code + 1,
            max_blindex + 1,
        );

        state.compress_block_dynamic_trees();
    }

    // TODO
    // This check is made mod 2^32, for files larger than 512 MB and unsigned long implemented on 32 bits.
    // assert_eq!(state.compressed_len, state.bits_sent, "bad compressed size");

    state.init_block();
    if last {
        state.bit_writer.emit_align();
    }

    // Tracev((stderr, "\ncomprlen {}(%lu) ", s->compressed_len>>3, s->compressed_len-7*last));
}

fn encode_block_core(state: &mut State<'_>, is_last: bool) {
    let (window_offset, unflushed_len) = unflushed_block_span(state.strstart, state.block_start);
    // This count describes the current not-yet-encoded token block, not a
    // lifetime total; token-buffer capacity bounds it well below u32::MAX.
    let Ok(stored_len) = u32::try_from(unflushed_len) else {
        unreachable!("unflushed DEFLATE block exceeds u32 length");
    };
    zng_tr_flush_block(state, window_offset, stored_len, is_last);

    let Ok(next_origin) = isize::try_from(state.strstart) else {
        unreachable!("current window position exceeds signed block-origin range");
    };
    state.block_start = next_origin;
}

/// Append an optional gzip metadata field incrementally. The caller supplies
/// only the not-yet-buffered suffix, using state.gzindex after an output pause.
/// FHCRC's own two bytes must not become part of the header CRC accumulator.
fn flush_bytes(
    stream: &mut Compressor<'_, '_>,
    mut bytes: &[u8],
    update_header_crc: bool,
) -> ControlFlow<ReturnCode> {
    let mut state = &mut *stream.state;

    // we'll be using the pending buffer as temporary storage
    let mut beg = state.bit_writer.pending.pending().len(); /* start of bytes to update crc */

    while state.bit_writer.pending.remaining() < bytes.len() {
        let copy = state.bit_writer.pending.remaining();

        state.bit_writer.pending.extend(&bytes[..copy]);

        if update_header_crc {
            stream
                .checksum
                .update_crc(&state.bit_writer.pending.pending()[beg..]);
        }

        state.gzindex += copy;
        let _ = stream.drain_pending();
        state = &mut *stream.state;

        // could not flush all the pending output
        if !state.bit_writer.pending.is_empty() {
            state.last_flush = -1;
            return ControlFlow::Break(ReturnCode::Ok);
        }

        beg = 0;
        bytes = &bytes[copy..];
    }

    state.bit_writer.pending.extend(bytes);

    if update_header_crc {
        stream
            .checksum
            .update_crc(&state.bit_writer.pending.pending()[beg..]);
    }
    state.gzindex = 0;

    ControlFlow::Continue(())
}

pub fn deflate(stream: &mut DeflateStream, flush: DeflateFlush) -> ReturnCode {
    let result = if stream.next_out.is_null()
        || (stream.avail_in != 0 && stream.next_in.is_null())
        || (stream.state.status == Status::Finish && flush != DeflateFlush::Finish)
    {
        ReturnCode::StreamError
    } else if stream.avail_out == 0 {
        ReturnCode::BufError
    } else {
        with_compressor(stream, |compressor| deflate_core(compressor, flush))
    };

    if matches!(result, ReturnCode::StreamError | ReturnCode::BufError) {
        stream.msg = result.error_message();
    }

    result
}

/// Emit the RFC 1950/zlib or RFC 1952/gzip wrapper prefix, including
/// optional gzip metadata. Incomplete output is not an error: the caller
/// retains its flush mode and resumes this exact header state.
/// Resume gzip extra/name/comment fields and the optional FHCRC from
/// their saved metadata cursor. No payload compression occurs here.
fn emit_gzip_metadata(stream: &mut Compressor<'_, '_>) -> ControlFlow<ReturnCode> {
    if stream.state.status == Status::Extra {
        if let Some(gzhead) = stream.state.gzhead {
            if let Some(extra) = gzhead.encoded_extra() {
                if let ControlFlow::Break(err) =
                    flush_bytes(stream, &extra[stream.state.gzindex..], true)
                {
                    return ControlFlow::Break(err);
                }
            }
        }
        stream.state.status = Status::Name;
    }

    if stream.state.status == Status::Name {
        if let Some(gzhead) = stream.state.gzhead {
            if let Some(name) = gzhead.name() {
                if let ControlFlow::Break(err) = flush_bytes(
                    stream,
                    &name.to_bytes_with_nul()[stream.state.gzindex..],
                    true,
                ) {
                    return ControlFlow::Break(err);
                }
            }
            stream.state.status = Status::Comment;
        }
    }

    if stream.state.status == Status::Comment {
        if let Some(gzhead) = stream.state.gzhead {
            if let Some(comment) = gzhead.comment() {
                if let ControlFlow::Break(err) = flush_bytes(
                    stream,
                    &comment.to_bytes_with_nul()[stream.state.gzindex..],
                    true,
                ) {
                    return ControlFlow::Break(err);
                }
            }
            stream.state.status = Status::Hcrc;
        }
    }

    if stream.state.status == Status::Hcrc {
        if let Some(gzhead) = stream.state.gzhead {
            if gzhead.hcrc() {
                let word = stream.checksum.low_word().to_le_bytes();
                let bytes = [word[0], word[1]];
                if let ControlFlow::Break(err) =
                    flush_bytes(stream, &bytes[stream.state.gzindex..], false)
                {
                    return ControlFlow::Break(err);
                }
            }
        }

        stream.state.status = Status::Busy;

        // compression must start with an empty pending buffer
        let _ = stream.drain_pending();
        if !stream.state.bit_writer.pending.is_empty() {
            stream.state.last_flush = -1;
            return ControlFlow::Break(ReturnCode::Ok);
        }
    }

    ControlFlow::Continue(())
}

fn emit_wrapper_header(stream: &mut Compressor<'_, '_>) -> ControlFlow<ReturnCode> {
    /* Write the header */
    if stream.state.status == Status::Init && stream.state.wrap == 0 {
        stream.state.status = Status::Busy;
    }

    if stream.state.status == Status::Init {
        let header = stream.state.header();
        stream
            .state
            .bit_writer
            .pending
            .extend(&header.to_be_bytes());

        /* Save the adler32 of the preset dictionary: */
        if stream.state.strstart != 0 {
            let adler = stream.checksum.low_word();
            stream.state.bit_writer.pending.extend(&adler.to_be_bytes());
        }

        stream.checksum.reset_adler();
        stream.state.status = Status::Busy;

        // compression must start with an empty pending buffer
        let _ = stream.drain_pending();

        if !stream.state.bit_writer.pending.is_empty() {
            stream.state.last_flush = -1;

            return ControlFlow::Break(ReturnCode::Ok);
        }
    }

    if stream.state.status == Status::GZip {
        /* gzip header */
        stream.state.crc_fold = Crc32Fold::new();

        stream.state.bit_writer.pending.extend(&[31, 139, 8]);

        let extra_flags = if stream.state.level == 9 {
            2
        } else if stream.state.strategy >= Strategy::HuffmanOnly || stream.state.level < 2 {
            4
        } else {
            0
        };

        match stream.state.gzhead {
            None => {
                let bytes = [0, 0, 0, 0, 0, extra_flags, gz_header::OS_CODE];
                stream.state.bit_writer.pending.extend(&bytes);
                stream.state.status = Status::Busy;

                /* Compression must start with an empty pending buffer */
                let _ = stream.drain_pending();
                if !stream.state.bit_writer.pending.is_empty() {
                    stream.state.last_flush = -1;
                    return ControlFlow::Break(ReturnCode::Ok);
                }
            }
            Some(gzhead) => {
                stream.state.bit_writer.pending.extend(&[gzhead.flags()]);
                stream
                    .state
                    .bit_writer
                    .pending
                    .extend(&gzhead.time().to_le_bytes());
                stream
                    .state
                    .bit_writer
                    .pending
                    .extend(&[extra_flags, gzhead.os()]);

                if let Some(extra) = gzhead.encoded_extra() {
                    let bytes = gzip_extra_wire_len(extra.len()).to_le_bytes();
                    stream.state.bit_writer.pending.extend(&bytes);
                }

                if gzhead.hcrc() {
                    stream
                        .checksum
                        .update_crc(stream.state.bit_writer.pending.pending());
                }

                stream.state.gzindex = 0;
                stream.state.status = Status::Extra;
            }
        }
    }

    emit_gzip_metadata(stream)
}

fn deflate_core(stream: &mut Compressor<'_, '_>, flush: DeflateFlush) -> ReturnCode {
    let old_flush = stream.state.last_flush;
    stream.state.last_flush = flush as i8;

    /* Flush as much pending output as possible */
    if !stream.state.bit_writer.pending.is_empty() {
        let _ = stream.drain_pending();
        if stream.output.avail_out == 0 {
            /* Since avail_out is 0, deflate will be called again with
             * more output space, but possibly with both pending and
             * avail_in equal to zero. There won't be anything to do,
             * but this is not an error situation so make sure we
             * return OK instead of BUF_ERROR at next call of deflate:
             */
            stream.state.last_flush = -1;
            return ReturnCode::Ok;
        }

        /* Make sure there is something to do and avoid duplicate consecutive
         * flushes. For repeated and useless calls with Z_FINISH, we keep
         * returning Z_STREAM_END instead of Z_BUF_ERROR.
         */
    } else if stream.input.avail_in == 0
        && rank_flush(flush as i8) <= rank_flush(old_flush)
        && flush != DeflateFlush::Finish
    {
        return ReturnCode::BufError;
    }

    /* User must not provide more input after the first FINISH: */
    if stream.state.status == Status::Finish && stream.input.avail_in != 0 {
        return ReturnCode::BufError;
    }

    if let ControlFlow::Break(code) = emit_wrapper_header(stream) {
        return code;
    }

    // Start a new block or continue the current one.
    let state = &mut *stream.state;
    if stream.input.avail_in != 0
        || state.lookahead != 0
        || (flush != DeflateFlush::NoFlush && state.status != Status::Finish)
    {
        let bstate = self::algorithm::run(stream, flush);

        let state = &mut *stream.state;

        if matches!(bstate, BlockState::FinishStarted | BlockState::FinishDone) {
            state.status = Status::Finish;
        }

        match bstate {
            BlockState::InvalidState => return ReturnCode::StreamError,
            BlockState::NeedMore | BlockState::FinishStarted => {
                if stream.output.avail_out == 0 {
                    state.last_flush = -1; /* avoid BUF_ERROR next call, see above */
                }
                return ReturnCode::Ok;
                /* If flush != Z_NO_FLUSH && avail_out == 0, the next call
                 * of deflate should use the same flush parameter to make sure
                 * that the flush is complete. So we don't have to output an
                 * empty block here, this will be done at next call. This also
                 * ensures that for a very small output buffer, we emit at most
                 * one empty block.
                 */
            }
            BlockState::BlockDone => {
                match flush {
                    DeflateFlush::NoFlush => unreachable!("condition of inner surrounding if"),
                    DeflateFlush::PartialFlush => {
                        state.bit_writer.align();
                    }
                    DeflateFlush::SyncFlush => {
                        // add an empty stored block that is marked as not final. This is useful for
                        // parallel deflate where we want to make sure the intermediate blocks are not
                        // marked as "last block".
                        zng_tr_stored_block(state, 0..0, false);
                    }
                    DeflateFlush::FullFlush => {
                        // add an empty stored block that is marked as not final. This is useful for
                        // parallel deflate where we want to make sure the intermediate blocks are not
                        // marked as "last block".
                        zng_tr_stored_block(state, 0..0, false);

                        state.head.as_mut_slice().fill(0); // forget history

                        if state.lookahead == 0 {
                            state.strstart = 0;
                            state.block_start = 0;
                            state.insert = 0;
                            state.window.clear();
                        }
                    }
                    DeflateFlush::Block => { /* fall through */ }
                    DeflateFlush::Finish => unreachable!("condition of outer surrounding if"),
                }

                let _ = stream.drain_pending();

                if stream.output.avail_out == 0 {
                    stream.state.last_flush = -1; /* avoid BUF_ERROR at next call, see above */
                    return ReturnCode::Ok;
                }
            }
            BlockState::FinishDone => { /* do nothing */ }
        }
    }

    if flush != DeflateFlush::Finish {
        return ReturnCode::Ok;
    }

    // write the trailer
    if stream.state.wrap == 2 {
        let crc_fold = core::mem::take(&mut stream.state.crc_fold);
        stream.checksum.set(z_checksum::from(crc_fold.finish()));

        let adler = stream.checksum.low_word();
        stream.state.bit_writer.pending.extend(&adler.to_le_bytes());

        let total_in = low_wire_u32(stream.input.total_in);
        stream
            .state
            .bit_writer
            .pending
            .extend(&total_in.to_le_bytes());
    } else if stream.state.wrap == 1 {
        let adler = stream.checksum.low_word();
        stream.state.bit_writer.pending.extend(&adler.to_be_bytes());
    }

    let _ = stream.drain_pending();

    // If avail_out is zero, the application will call deflate again to flush the rest.
    if stream.state.wrap > 0 {
        stream.state.wrap = -stream.state.wrap; /* write the trailer only once! */
    }

    if stream.state.bit_writer.pending.is_empty() {
        assert_eq!(stream.state.bit_writer.bits_valid, 0, "bi_buf not flushed");
        return ReturnCode::StreamEnd;
    }
    ReturnCode::Ok
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrainOutcome {
    Ready,
    OutputFull,
}

/// Drain pending compressed bytes using only compression state and the caller output cursor.
///
/// `OutputFull` deliberately mirrors zlib's streaming boundary: callers return once the supplied
/// output buffer reaches zero, even when that write happened to consume the final pending byte.
pub fn drain_pending_output(state: &mut State<'_>, output: &mut OutputCursor) -> DrainOutcome {
    state.bit_writer.flush_bits();

    let pending = state.bit_writer.pending.pending();
    let len = output.write_some(pending);

    if len != 0 {
        trace!("\n[FLUSH {len} bytes]");
        state.bit_writer.pending.advance(len);
    }

    if output.avail_out == 0 {
        DrainOutcome::OutputFull
    } else {
        DrainOutcome::Ready
    }
}

/// Drain all complete bytes currently produced by the complete-buffer near-optimal path while
/// preserving any sub-byte DEFLATE prefix for the next block.
#[cfg(any(test, all(feature = "__internal-api", feature = "rust-allocator")))]
fn drain_near_optimal_pending(
    state: &mut State<'_>,
    output: &mut [u8],
    produced: &mut usize,
) -> bool {
    state.bit_writer.flush_bits();
    let pending = state.bit_writer.pending.pending();
    let remaining = output.len().saturating_sub(*produced);
    if pending.len() > remaining {
        return false;
    }
    let len = pending.len();
    output[*produced..*produced + len].copy_from_slice(pending);
    state.bit_writer.pending.advance(len);
    *produced += len;
    true
}

#[cfg(any(test, all(feature = "__internal-api", feature = "rust-allocator")))]
fn emit_planned_near_optimal_block(
    state: &mut State<'_>,
    input: &[u8],
    block: &near_optimal::PlannedBlock,
    is_last: bool,
    output: &mut [u8],
    produced: &mut usize,
) -> bool {
    near_optimal::tally_tokens(state, &block.tokens);

    state.window.clear();
    if block.range.len() <= MAX_STORED {
        // Short blocks keep the production stored/static/dynamic choice. Only the current block
        // bytes are needed for stored fallback; LZ77 history was already accounted for while
        // planning tokens.
        if !block.range.is_empty() {
            // SAFETY: this exact block slice is initialized and readable, the state owns its window
            // uniquely, and a stored-capable block is no larger than the 64 KiB window allocation.
            unsafe {
                state
                    .window
                    .copy_and_initialize(0..block.range.len(), input[block.range.clone()].as_ptr())
            };
        }
        state.block_start = 0;
        state.strstart = block.range.len();
        state.lookahead = 0;
        state.encode_block(is_last);
    } else {
        // DEFLATE Huffman blocks have no byte-length field. A long, sufficiently compressible block
        // can therefore span more than 65,535 input bytes as long as its token count fits SymBuf.
        // There is no single stored-block fallback for such a span, so deliberately omit window
        // backing and force the existing tree chooser to select static or dynamic Huffman coding.
        let stored_len = u32::try_from(block.range.len())
            .expect("near-optimal token capacity bounds input span below u32::MAX");
        zng_tr_flush_block(state, None, stored_len, is_last);
    }
    drain_near_optimal_pending(state, output, produced)
}

/// Experimental complete-buffer near-optimal compressor with adaptive block splitting.
///
/// This remains behind `__internal-api` while the cost model and splitter are being developed. It
/// emits a normal zlib-wrapped DEFLATE stream, keeps LZ77 history across Huffman block boundaries,
/// and reuses the production tree builder / stored-static-dynamic block chooser / bit writer.
/// `optimization_passes` counts minimum-cost path searches per selected block; zero is retained only
/// as a fixed-Huffman diagnostic baseline.
#[cfg(any(test, all(feature = "__internal-api", feature = "rust-allocator")))]
pub fn compress_slice_near_optimal<'a>(
    output: &'a mut [u8],
    input: &[u8],
    optimization_passes: usize,
) -> (&'a mut [u8], ReturnCode) {
    let mut raw = z_stream::default();
    let init_result = init_with_lit_bufsize(
        &mut raw,
        DeflateConfig::new(9),
        Some(near_optimal::NEAR_OPTIMAL_LIT_BUFSIZE),
    );
    if init_result != ReturnCode::Ok {
        return (&mut output[..0], init_result);
    }

    let produced = {
        let stream = unsafe { DeflateStream::from_stream_mut(&raw mut raw) }
            .expect("successful deflate init creates typed state");
        let state = &mut *stream.state;
        // Whole-input near-optimal matching is not constrained by zlib's ring-buffer lookahead
        // reservation, so every distance representable by the configured DEFLATE window is legal.
        let max_distance = state.w_mask() + 1;
        let ranges = near_optimal::choose_block_ranges(input, max_distance);
        debug_assert!(ranges
            .iter()
            .all(|range| range.len() <= near_optimal::MAX_COARSE_BLOCK_INPUT));

        // Emit the wrapper while strstart still describes the true initial state. Moving this after
        // synthetic block bookkeeping would incorrectly set zlib's PRESET_DICT flag.
        state
            .bit_writer
            .pending
            .extend(&state.header().to_be_bytes());
        let mut produced = 0usize;
        let mut ok = drain_near_optimal_pending(state, output, &mut produced);

        let mut ranges = ranges.into_iter();
        let mut current = ranges.next().map(|range| {
            near_optimal::parse_planned_block_source(
                state,
                input,
                range,
                max_distance,
                optimization_passes,
                None,
            )
            .into_parts()
        });

        for range in ranges {
            if !ok {
                break;
            }
            let previous_model = current.as_ref().map(|(_, model)| model);
            let mut next = near_optimal::parse_planned_block_source(
                state,
                input,
                range,
                max_distance,
                optimization_passes,
                previous_model,
            );
            let (current_block, current_model) = current
                .as_mut()
                .expect("the splitter always returns at least one block");
            match near_optimal::optimize_token_partition(state, current_block, &mut next.block) {
                near_optimal::BoundaryOptimization::Merged(merged) => {
                    next.block = merged;
                    let _ = next.reparse_with_existing_candidates(
                        state,
                        input,
                        optimization_passes,
                        None,
                    );
                    current = Some(next.into_parts());
                    continue;
                }
                near_optimal::BoundaryOptimization::Shifted => {
                    *current_model = near_optimal::model_from_tokens(state, &current_block.tokens);
                    let _ = next.reparse_with_existing_candidates(
                        state,
                        input,
                        optimization_passes,
                        Some(current_model),
                    );
                }
                near_optimal::BoundaryOptimization::Unchanged => {}
            }
            ok = emit_planned_near_optimal_block(
                state,
                input,
                current_block,
                false,
                output,
                &mut produced,
            );
            current = Some(next.into_parts());
        }

        if ok {
            if let Some((final_block, _)) = current.as_ref() {
                ok = emit_planned_near_optimal_block(
                    state,
                    input,
                    final_block,
                    true,
                    output,
                    &mut produced,
                );
            }
        }

        if ok {
            let checksum = adler32(ADLER32_INITIAL_VALUE, input);
            state.bit_writer.pending.extend(&checksum.to_be_bytes());
            ok = drain_near_optimal_pending(state, output, &mut produced);
        }
        ok.then_some(produced)
    };

    if let Some(stream) = unsafe { DeflateStream::from_stream_mut(&raw mut raw) } {
        stream.state.status = Status::Finish;
        let _ = end(stream);
    }

    match produced {
        Some(len) => (&mut output[..len], ReturnCode::Ok),
        None => (&mut output[..0], ReturnCode::BufError),
    }
}

/// Compresses `input` into the provided `output` buffer.
///
/// Returns a subslice of `output` containing the compressed bytes and a
/// [`ReturnCode`] indicating the result of the operation. Returns [`ReturnCode::BufError`] if
/// there is insufficient output space.
///
/// Use [`compress_bound`] for an upper bound on how large the output buffer needs to be.
///
/// # Example
///
/// ```
/// # use zlib_rs::*;
/// # fn foo(input: &[u8]) {
/// let mut buf = vec![0u8; compress_bound(input.len())];
/// let (compressed, rc) = compress_slice(&mut buf, input, DeflateConfig::default());
/// # }
/// ```
pub fn compress_slice<'a>(
    output: &'a mut [u8],
    input: &[u8],
    config: DeflateConfig,
) -> (&'a mut [u8], ReturnCode) {
    compress_slice_with_flush(output, input, config, DeflateFlush::Finish)
}

pub fn compress<'a>(
    output: &'a mut [MaybeUninit<u8>],
    input: &[u8],
    config: DeflateConfig,
) -> (&'a mut [u8], ReturnCode) {
    compress_with_flush(output, input, config, DeflateFlush::Finish)
}

pub fn compress_slice_with_flush<'a>(
    output: &'a mut [u8],
    input: &[u8],
    config: DeflateConfig,
    flush: DeflateFlush,
) -> (&'a mut [u8], ReturnCode) {
    let (produced, code) = compress_into(output, input, config, flush);
    (&mut output[..produced], code)
}

pub fn compress_with_flush<'a>(
    output: &'a mut [MaybeUninit<u8>],
    input: &[u8],
    config: DeflateConfig,
    final_flush: DeflateFlush,
) -> (&'a mut [u8], ReturnCode) {
    let (produced, code) = compress_into(output, input, config, final_flush);
    // SAFETY: `compress_into` reports only bytes written through the zlib output cursor.
    let initialized = unsafe { assume_init_prefix(output, produced) };
    (initialized, code)
}

fn compress_into<O: WritableBytes + ?Sized>(
    output: &mut O,
    input: &[u8],
    config: DeflateConfig,
    final_flush: DeflateFlush,
) -> (usize, ReturnCode) {
    let output_len = output.len();
    let mut stream = z_stream {
        next_in: input.as_ptr().cast_mut(),
        avail_in: 0, // for special logic in the first iteration
        total_in: 0,
        next_out: output.byte_ptr(),
        avail_out: 0, // for special logic on the first iteration
        total_out: 0,
        msg: core::ptr::null_mut(),
        state: core::ptr::null_mut(),
        zalloc: None,
        zfree: None,
        opaque: core::ptr::null_mut(),
        data_type: 0,
        adler: 0,
        reserved: 0,
    };

    let err = init(&mut stream, config);
    if err != ReturnCode::Ok {
        return (0, err);
    }

    let mut output_budget = SliceBudget::new(output_len);
    let mut input_budget = SliceBudget::new(input.len());
    let Some(stream) = (unsafe { DeflateStream::from_stream_mut(&raw mut stream) }) else {
        return (0, ReturnCode::StreamError);
    };

    let return_code = loop {
        output_budget.refill(&mut stream.avail_out);
        input_budget.refill(&mut stream.avail_in);

        let flush = if input_budget.pending() > 0 {
            DeflateFlush::NoFlush
        } else {
            final_flush
        };
        let err = deflate(stream, flush);

        match err {
            ReturnCode::Ok => continue,
            ReturnCode::StreamEnd => break ReturnCode::Ok,
            _ => break err,
        }
    };

    // The public zlib total_out counter may wrap on 64-bit Windows (32-bit c_ulong).
    // Recover the exact Rust-slice length from caller output budgets instead.
    let produced = output_budget.processed(stream.avail_out);
    let _ = end(stream);

    (produced, return_code)
}

/// Returns the upper bound on the compressed size for an input of `source_len` bytes.
///
/// When compression has this much space available, it will never fail because of insufficient
/// output space. Returns `usize::MAX` if the mathematical bound cannot be represented
/// as `usize`; no finite output slice can satisfy that bound.
///
/// # Example
///
/// ```
/// # use zlib_rs::*;
///
/// assert_eq!(compress_bound(1024), 1161);
/// assert_eq!(compress_bound(4096), 4617);
/// assert_eq!(compress_bound(65536), 73737);
///
/// # fn foo(input: &[u8]) {
/// let mut buf = vec![0u8; compress_bound(input.len())];
/// let (compressed, rc) = compress_slice(&mut buf, input, DeflateConfig::default());
/// # }
/// ```
#[must_use]
pub const fn compress_bound(source_len: usize) -> usize {
    // div_ceil avoids the overflow in (source_len + 7) / 8; saturating additions
    // avoid understating bounds that cannot fit in usize.
    let quick_overhead = source_len.div_ceil(8);

    source_len
        .saturating_add(if source_len == 0 { 1 } else { 0 })
        .saturating_add(if source_len < 9 { 1 } else { 0 })
        .saturating_add(quick_overhead)
        .saturating_add(DEFLATE_BLOCK_OVERHEAD)
        .saturating_add(ZLIB_WRAPLEN)
}

/// zlib-ng-compatible C ABI bound, including its unsigned wrapping semantics.
///
/// C callers already observe the original zlib-ng integer behavior at extreme
/// lengths; the Rust-facing `compress_bound` instead supplies a true saturating
/// upper bound. This function intentionally stays separate from that safe API.
#[must_use]
pub const fn compress_bound_abi(source_len: usize) -> usize {
    compress_bound_help(source_len, ZLIB_WRAPLEN)
}

const fn compress_bound_help(source_len: usize, wrap_len: usize) -> usize {
    source_len // The source size itself */
        // Always at least one byte for any input
        .wrapping_add(if source_len == 0 { 1 } else { 0 })
        // One extra byte for lengths less than 9
        .wrapping_add(if source_len < 9 { 1 } else { 0 })
        // Source encoding overhead, padded to next full byte
        .wrapping_add(deflate_quick_overhead(source_len))
        // Deflate block overhead bytes
        .wrapping_add(DEFLATE_BLOCK_OVERHEAD)
        // none, zlib or gzip wrapper
        .wrapping_add(wrap_len)
}

///  heap used to build the Huffman trees
///
/// The sons of heap[n] are heap[2*n] and heap[2*n+1]. heap[0] is not used.
/// The same heap array is used to build all trees.
#[derive(Clone)]
struct Heap {
    // Native indices address the heap and tree arrays, not DEFLATE wire fields.
    nodes: [usize; HEAP_SIZE],
    /// Number of active entries in the binary min-heap (indexed from 1).
    active_len: usize,
    /// First occupied position in the completed tree's sorted-node suffix.
    sorted_start: usize,
    depth: [u8; HEAP_SIZE],
}

impl Heap {
    /// A fresh heap; rebuilding a tree resets the active and sorted spans.
    const fn new() -> Self {
        Self {
            nodes: [0; HEAP_SIZE],
            active_len: 0,
            sorted_start: 0,
            depth: [0; HEAP_SIZE],
        }
    }

    /// The sorted suffix contains the root followed by its children.
    const fn root(&self) -> usize {
        self.nodes[self.sorted_start]
    }

    fn non_root_nodes(&self) -> &[usize] {
        &self.nodes[self.sorted_start + 1..]
    }

    fn push_leaf(&mut self, code: usize) {
        assert!(self.active_len + 1 < HEAP_SIZE);
        self.active_len += 1;
        self.nodes[self.active_len] = code;
        self.depth[code] = 0;
    }

    fn heapify(&mut self, tree: &[Value]) {
        for index in (1..=self.active_len / 2).rev() {
            self.pqdownheap(tree, index);
        }
    }

    /// Compare the frequency before the tie-breaking depth of two nodes.
    #[inline]
    fn priority(&self, tree: &[Value], node: usize) -> u32 {
        (u32::from(tree[node].freq()) << 8) | u32::from(self.depth[node])
    }

    /// Construct the initial heap, with least frequent element in
    /// heap[SMALLEST]. The sons of heap[n] are heap[2*n] and heap[2*n+1]. heap[0] is not used.
    fn initialize(&mut self, tree: &mut [Value]) -> Option<usize> {
        let mut max_code = None;

        self.active_len = 0;
        self.sorted_start = HEAP_SIZE;

        for (n, node) in tree.iter_mut().enumerate() {
            if node.freq() > 0 {
                self.push_leaf(n);
                max_code = Some(n);
            } else {
                *node.len_mut() = 0;
            }
        }

        max_code
    }

    /// Index within the heap array of least frequent node in the Huffman tree
    const SMALLEST: usize = 1;

    fn pqdownheap(&mut self, tree: &[Value], mut k: usize) {
        /* tree: the tree to restore */
        /* k: node to move down */

        let v = self.nodes[k];
        let v_val = self.priority(tree, v);
        let mut j = k << 1; /* left son of k */

        while j <= self.active_len {
            /* Set j to the smallest of the two sons: */
            let mut j_val = self.priority(tree, self.nodes[j]);
            if j < self.active_len {
                let next_val = self.priority(tree, self.nodes[j + 1]);
                if next_val <= j_val {
                    j += 1;
                    j_val = next_val;
                }
            }

            /* Exit if v is smaller than both sons */
            if v_val <= j_val {
                break;
            }

            /* Exchange v with the smallest son */
            self.nodes[k] = self.nodes[j];
            k = j;

            /* And continue down the tree, setting j to the left son of k */
            j <<= 1;
        }

        self.nodes[k] = v;
    }

    /// Remove the smallest element from the heap and recreate the heap with
    /// one less element. Updates heap and `heap_len`.
    fn pqremove(&mut self, tree: &[Value]) -> usize {
        let top = self.nodes[Self::SMALLEST];
        self.nodes[Self::SMALLEST] = self.nodes[self.active_len];
        self.active_len -= 1;

        self.pqdownheap(tree, Self::SMALLEST);

        top
    }

    /// Construct the Huffman tree by repeatedly combining the least two frequent nodes.
    fn construct_huffman_tree(&mut self, tree: &mut [Value], mut node: usize) {
        loop {
            let n = self.pqremove(tree); /* n = node of least frequency */
            let m = self.nodes[Self::SMALLEST]; /* m = node of next least frequency */

            self.sorted_start -= 1;
            self.nodes[self.sorted_start] = n; /* keep the nodes sorted by frequency */
            self.sorted_start -= 1;
            self.nodes[self.sorted_start] = m;

            /* Create a new node father of n and m */
            let combined = u32::from(tree[n].freq()) + u32::from(tree[m].freq());
            let Ok(frequency) = u16::try_from(combined) else {
                unreachable!("Huffman parent frequency exceeds the bounded token count");
            };
            *tree[node].freq_mut() = frequency;
            let Some(depth) = self.depth[n].max(self.depth[m]).checked_add(1) else {
                unreachable!("Huffman tree depth exceeds its packed priority range");
            };
            self.depth[node] = depth;

            let Ok(parent) = u16::try_from(node) else {
                unreachable!("the fixed Huffman alphabet cannot exceed u16 parent indices");
            };
            *tree[n].dad_mut() = parent;
            *tree[m].dad_mut() = parent;

            /* and insert the new node in the heap */
            self.nodes[Self::SMALLEST] = node;
            node += 1;

            self.pqdownheap(tree, Self::SMALLEST);

            if self.active_len < 2 {
                break;
            }
        }

        self.sorted_start -= 1;
        self.nodes[self.sorted_start] = self.nodes[Self::SMALLEST];
    }
}

/// # Safety
///
/// The caller must guarantee:
///
/// * If `head` is `Some`, the header remains valid while this stream may still emit it, and
///   is not mutated concurrently with an operation on the stream that reads it.
///     - `head.extra` is `NULL` or is readable for at least `head.extra_len` bytes
///     - `head.name` is `NULL` or satisfies the requirements of [`core::ffi::CStr::from_ptr`]
///     - `head.comment` is `NULL` or satisfies the requirements of [`core::ffi::CStr::from_ptr`]
pub unsafe fn set_header<'a>(
    stream: &mut DeflateStream<'a>,
    head: Option<&'a mut gz_header>,
) -> ReturnCode {
    if stream.state.wrap != 2 {
        ReturnCode::StreamError
    } else {
        stream.state.gzhead = head.map(|head| {
            // SAFETY: this function's contract establishes all invariants required by the view.
            unsafe { GzipHeaderRef::new(head) }
        });
        ReturnCode::Ok
    }
}

// zlib format overhead
const ZLIB_WRAPLEN: usize = 6;
// gzip format overhead
const GZIP_WRAPLEN: usize = 18;

const DEFLATE_HEADER_BITS: usize = 3;
const DEFLATE_EOBS_BITS: usize = 15;
const DEFLATE_PAD_BITS: usize = 6;
const DEFLATE_BLOCK_OVERHEAD: usize =
    (DEFLATE_HEADER_BITS + DEFLATE_EOBS_BITS + DEFLATE_PAD_BITS) >> 3;

const DEFLATE_QUICK_LIT_MAX_BITS: usize = 9;
const fn deflate_quick_overhead(x: usize) -> usize {
    let sum = x
        .wrapping_mul(DEFLATE_QUICK_LIT_MAX_BITS - 8)
        .wrapping_add(7);

    // imitate zlib-ng rounding behavior (on windows, c_ulong is 32 bits)
    (sum as core::ffi::c_ulong >> 3) as usize
}

/// For the default windowBits of 15 and memLevel of 8, this function returns
/// a close to exact, as well as small, upper bound on the compressed size.
/// They are coded as constants here for a reason--if the #define's are
/// changed, then this function needs to be changed as well.  The return
/// value for 15 and 8 only works for those exact settings.
///
/// For any setting other than those defaults for windowBits and memLevel,
/// the value returned is a conservative worst case for the maximum expansion
/// resulting from using fixed blocks instead of stored blocks, which deflate
/// can emit on compressed data for some combinations of the parameters.
///
/// This function could be more sophisticated to provide closer upper bounds for
/// every combination of windowBits and memLevel.  But even the conservative
/// upper bound of about 14% expansion does not seem onerous for output buffer
/// allocation.
pub fn bound(stream: Option<&DeflateStream>, source_len: usize) -> usize {
    // on windows, c_ulong is only a 32-bit integer
    let mask = core::ffi::c_ulong::MAX as usize;

    // conservative upper bound for compressed data
    let comp_len = source_len
        .wrapping_add((source_len.wrapping_add(7) & mask) >> 3)
        .wrapping_add((source_len.wrapping_add(63) & mask) >> 6)
        .wrapping_add(5);

    let Some(stream) = stream else {
        // return conservative bound plus zlib wrapper
        return comp_len.wrapping_add(6);
    };

    /* compute wrapper length */
    let wrap_len = match stream.state.wrap {
        0 => {
            // raw deflate
            0
        }
        1 => {
            // zlib wrapper
            if stream.state.strstart != 0 {
                ZLIB_WRAPLEN + 4
            } else {
                ZLIB_WRAPLEN
            }
        }
        2 => {
            // gzip wrapper
            let mut gz_wrap_len = GZIP_WRAPLEN;

            if let Some(header) = stream.state.gzhead {
                if let Some(extra) = header.encoded_extra() {
                    gz_wrap_len += 2 + extra.len();
                }
                if let Some(name) = header.name() {
                    gz_wrap_len += name.to_bytes_with_nul().len();
                }
                if let Some(comment) = header.comment() {
                    gz_wrap_len += comment.to_bytes_with_nul().len();
                }
                if header.hcrc() {
                    gz_wrap_len += 2;
                }
            }

            gz_wrap_len
        }
        _ => {
            // default
            ZLIB_WRAPLEN
        }
    };

    if stream.state.w_bits() != MAX_WBITS as u32 || HASH_BITS < 15 {
        if stream.state.level == 0 {
            /* upper bound for stored blocks with length 127 (memLevel == 1) ~4% overhead plus a small constant */
            source_len
                .wrapping_add(source_len >> 5)
                .wrapping_add(source_len >> 7)
                .wrapping_add(source_len >> 11)
                .wrapping_add(7)
                .wrapping_add(wrap_len)
        } else {
            comp_len.wrapping_add(wrap_len)
        }
    } else {
        compress_bound_help(source_len, wrap_len)
    }
}

/// Return the sliding dictionary in logical input order.
///
/// Raw caller-buffer handling belongs to the C ABI layer; the compression core exposes only a
/// checked slice over its initialized window storage.
pub fn dictionary<'stream>(stream: &'stream DeflateStream<'_>) -> &'stream [u8] {
    let state = &*stream.state;
    state
        .window
        .recent_history(state.strstart + state.lookahead, state.w_size)
}

const DEFLATE_ALLOC_ALIGN: usize = ALLOCATION_BLOCK_BYTES;
const LIT_BUFS: usize = 4;

// Every typed view into the shared deflate allocation must fit within the
// cache-line-aligned base and offsets established by DeflateAllocOffsets.
const _: () = {
    assert!(DEFLATE_ALLOC_ALIGN.is_power_of_two());
    assert!(core::mem::size_of::<AlignedAllocationBlock>() == DEFLATE_ALLOC_ALIGN);
    assert!(core::mem::align_of::<AlignedAllocationBlock>() == DEFLATE_ALLOC_ALIGN);
    assert!(core::mem::align_of::<State<'static>>() <= DEFLATE_ALLOC_ALIGN);
    assert!(core::mem::align_of::<Pos>() <= DEFLATE_ALLOC_ALIGN);
    assert!(core::mem::align_of::<[Pos; HASH_SIZE]>() <= DEFLATE_ALLOC_ALIGN);
    assert!(core::mem::align_of::<MaybeUninit<u8>>() <= DEFLATE_ALLOC_ALIGN);
};

/// All raw region pointers derive from the same aligned allocation base.
/// This is the sole place where byte offsets become typed deflate regions.
struct DeflateAllocRegions<'a> {
    state: *mut State<'a>,
    window: *mut u8,
    prev: *mut Pos,
    head: *mut [Pos; HASH_SIZE],
    pending: *mut u8,
    sym_buf: *mut u8,
}

#[derive(Clone, Copy)]
struct DeflateStateInit {
    level: i8,
    strategy: Strategy,
    wrap: i8,
}

struct DeflateAllocBuffers<'a> {
    state_ptr: *mut State<'a>,
    window: Window<'a>,
    prev: WeakSliceMut<'a, Pos>,
    head: WeakArrayMut<'a, Pos, HASH_SIZE>,
    pending: Pending<'a>,
    sym_buf: SymBuf<'a>,
}

struct DeflateAllocOffsets {
    window_bits: usize,
    lit_bufsize: usize,
    total_size: usize,
    state_pos: usize,
    window_pos: usize,
    pending_pos: usize,
    sym_buf_pos: usize,
    prev_pos: usize,
    head_pos: usize,
}

impl DeflateAllocOffsets {
    const fn new(window_bits: usize, lit_bufsize: usize) -> Self {
        use core::mem::size_of;

        let mut curr_size = 0usize;

        /* Define sizes */
        let state_size = size_of::<State>();
        // Allocate a second window worth of space to avoid the need to shift the data constantly.
        let window_size = (1 << window_bits) * 2;
        let prev_size = (1 << window_bits) * size_of::<Pos>();
        let head_size = HASH_SIZE * size_of::<Pos>();
        let pending_size = lit_bufsize * LIT_BUFS;
        let sym_buf_size = lit_bufsize * (LIT_BUFS - 1);
        // let alloc_size = size_of::<DeflateAlloc>();

        /* Calculate relative buffer positions and paddings */
        let state_pos = curr_size.next_multiple_of(DEFLATE_ALLOC_ALIGN);
        curr_size = state_pos + state_size;

        let window_pos = curr_size.next_multiple_of(DEFLATE_ALLOC_ALIGN);
        curr_size = window_pos + window_size;

        let prev_pos = curr_size.next_multiple_of(DEFLATE_ALLOC_ALIGN);
        curr_size = prev_pos + prev_size;

        let head_pos = curr_size.next_multiple_of(DEFLATE_ALLOC_ALIGN);
        curr_size = head_pos + head_size;

        let pending_pos = curr_size.next_multiple_of(DEFLATE_ALLOC_ALIGN);
        curr_size = pending_pos + pending_size;

        let sym_buf_pos = curr_size.next_multiple_of(DEFLATE_ALLOC_ALIGN);
        curr_size = sym_buf_pos + sym_buf_size;

        // The typed storage allocation is already 64-byte aligned; only trailing
        // padding to a whole storage unit is necessary.
        let total_size = curr_size.next_multiple_of(DEFLATE_ALLOC_ALIGN);

        Self {
            window_bits,
            lit_bufsize,
            total_size,
            state_pos,
            window_pos,
            pending_pos,
            sym_buf_pos,
            prev_pos,
            head_pos,
        }
    }

    /// Allocate cache-line-aligned typed storage through the common allocator.
    ///
    /// The allocator handles alignment even when custom C callbacks return
    /// unaligned memory. No redundant adjustment or headroom is required.
    fn allocate_storage(&self, alloc: &Allocator<'_>) -> Option<NonNull<AlignedAllocationBlock>> {
        alloc.allocate_slice_raw::<AlignedAllocationBlock>(self.total_size / DEFLATE_ALLOC_ALIGN)
    }

    /// Allocate, initialize, and place one fresh deflate state using this layout.
    ///
    /// Raw allocation alignment, allocation-backed buffer construction, state placement, and
    /// deallocation metadata are all established here before the state pointer can escape.
    fn allocate_state<'a>(
        &self,
        alloc: &Allocator<'a>,
        init: DeflateStateInit,
    ) -> Option<NonNull<State<'a>>> {
        let allocation_start = self.allocate_storage(alloc)?;
        let aligned = allocation_start.cast();
        // SAFETY: aligned is the fresh 64-byte-aligned typed storage allocation and
        // remains owned by the state until matching typed deallocation in end().
        let DeflateAllocBuffers {
            state_ptr,
            window,
            prev,
            head,
            pending,
            sym_buf,
        } = unsafe { self.initialize_buffers(aligned) };

        let state = State {
            status: Status::Init,
            last_flush: 0,
            wrap: init.wrap,
            strategy: init.strategy,
            level: init.level,
            block_open: 0,
            hash_calc_variant: HashCalcVariant::Standard,
            match_available: false,
            good_match: 0,
            nice_match: 0,
            match_start: 0,
            prev_match: 0,
            strstart: 0,
            window,
            w_size: 1 << self.window_bits,
            lookahead: 0,
            _cache_line_0: (),
            prev,
            head,
            prev_length: 0,
            max_chain_length: 0,
            max_insert_length: 0,
            matches: 0,
            block_start: 0,
            sym_buf,
            _cache_line_1: (),
            lit_bufsize: self.lit_bufsize,
            window_size: 0,
            bit_writer: BitWriter::from_pending(pending),
            _cache_line_2: (),
            opt_len: 0,
            static_len: 0,
            insert: 0,
            ins_h: 0,
            max_lazy_match: 0,
            gzhead: None,
            gzindex: 0,
            medium_next_match: MatchCandidate::EMPTY,
            data_type: crate::c_api::Z_UNKNOWN,
            padding_0: [0; 4],
            _cache_line_3: (),
            crc_fold: Crc32Fold::new(),
            allocation_start,
            total_allocation_size: self.total_size,
            l_desc: TreeDesc::new(&StaticTreeDesc::L),
            d_desc: TreeDesc::new(&StaticTreeDesc::D),
            bl_desc: TreeDesc::new(&StaticTreeDesc::BL),
        };

        // SAFETY: `state_ptr` is the aligned State region in this fresh allocation and no State
        // has previously been materialized there.
        unsafe { state_ptr.write(state) };
        NonNull::new(state_ptr)
    }

    /// Resolve the six nonoverlapping regions of the layout from its aligned allocation base.
    ///
    /// # Safety
    ///
    /// The aligned pointer must designate the 64-byte-aligned start of a writable
    /// typed allocation large enough for all six regions, and remain live while used.
    const unsafe fn regions<'a>(&self, aligned: NonNull<u8>) -> DeflateAllocRegions<'a> {
        let base = aligned.as_ptr();

        // SAFETY: offsets are within this same allocated region and multiples of
        // DEFLATE_ALLOC_ALIGN. The base alignment and const assertions establish the
        // alignment of every typed target.
        unsafe {
            DeflateAllocRegions {
                state: base.add(self.state_pos).cast(),
                window: base.add(self.window_pos),
                prev: base.add(self.prev_pos).cast(),
                head: base.add(self.head_pos).cast(),
                pending: base.add(self.pending_pos).cast(),
                sym_buf: base.add(self.sym_buf_pos),
            }
        }
    }

    /// Resolve this layout inside one raw allocator block and initialize every region whose typed
    /// wrapper requires initialized values before constructing that wrapper.
    ///
    /// # Safety
    ///
    /// `aligned` must be the aligned interior base of a writable allocation described by this
    /// layout, remain alive for `'a`, and not be aliased by another live deflate state.
    const unsafe fn initialize_buffers<'a>(&self, aligned: NonNull<u8>) -> DeflateAllocBuffers<'a> {
        // SAFETY: the caller supplies the aligned base of an independent allocation described
        // by this layout; each returned region has the required alignment and valid extent.
        let DeflateAllocRegions {
            state: state_ptr,
            window: window_ptr,
            prev: prev_ptr,
            head: head_ptr,
            pending: pending_ptr,
            sym_buf: sym_buf_ptr,
        } = unsafe { self.regions(aligned) };
        let w_size = 1usize << self.window_bits;

        // SAFETY: all regions are disjoint, adequately sized, and backed by this live allocation.
        // The position tables are initialized before creating their typed views.
        unsafe {
            let window = Window::from_raw_parts(window_ptr, self.window_bits);
            prev_ptr.write_bytes(0, w_size);
            let prev = WeakSliceMut::from_raw_parts_mut(prev_ptr, w_size);
            head_ptr.write_bytes(0, 1);
            let head = WeakArrayMut::<Pos, HASH_SIZE>::from_ptr(head_ptr);
            let pending = Pending::from_raw_parts(pending_ptr.cast(), self.lit_bufsize * LIT_BUFS);
            let sym_buf = SymBuf::from_raw_parts(sym_buf_ptr, self.lit_bufsize * (LIT_BUFS - 1));

            DeflateAllocBuffers {
                state_ptr,
                window,
                prev,
                head,
                pending,
                sym_buf,
            }
        }
    }

    /// Resolve this layout inside a new raw allocation and clone the source buffer regions before
    /// constructing typed wrappers over them.
    ///
    /// # Safety
    ///
    /// `aligned` must satisfy the same allocation/lifetime requirements as
    /// [`Self::initialize_buffers`] and must not overlap any source-state allocation.
    unsafe fn clone_buffers<'a>(
        &self,
        aligned: NonNull<u8>,
        source: &State<'_>,
    ) -> DeflateAllocBuffers<'a> {
        // SAFETY: caller guarantees a fresh destination allocation disjoint from source.
        let DeflateAllocRegions {
            state: state_ptr,
            window: window_ptr,
            prev: prev_ptr,
            head: head_ptr,
            pending: pending_ptr,
            sym_buf: sym_buf_ptr,
        } = unsafe { self.regions(aligned) };
        let w_size = 1usize << self.window_bits;
        debug_assert_eq!(source.w_size, w_size);
        debug_assert_eq!(source.lit_bufsize, self.lit_bufsize);

        // SAFETY: all destination regions are sized and aligned by the shared layout; none
        // overlap the source buffers. Each typed view is created only after its contents
        // have been cloned into the new allocation.
        unsafe {
            let window = source.window.clone_to(window_ptr);
            prev_ptr.copy_from_nonoverlapping(source.prev.as_ptr(), source.prev.len());
            let prev = WeakSliceMut::from_raw_parts_mut(prev_ptr, w_size);
            head_ptr.copy_from_nonoverlapping(source.head.as_ptr(), 1);
            let head = WeakArrayMut::<Pos, HASH_SIZE>::from_ptr(head_ptr);
            let pending = source.bit_writer.pending.clone_to(pending_ptr);
            let sym_buf = source.sym_buf.clone_to(sym_buf_ptr);

            DeflateAllocBuffers {
                state_ptr,
                window,
                prev,
                head,
                pending,
                sym_buf,
            }
        }
    }

    /// Clone a complete deflate state into one independent allocation and rebind every view that
    /// points into the allocation. Caller-owned ABI pointers (for example `gzhead`) intentionally
    /// retain zlib's shallow-copy semantics.
    fn clone_state<'a>(
        &self,
        alloc: &Allocator<'_>,
        source: &State<'a>,
    ) -> Option<NonNull<State<'a>>> {
        assert_eq!(
            self.window_bits,
            source.w_bits() as usize,
            "deflate clone layout must match source window geometry"
        );
        assert_eq!(
            self.lit_bufsize, source.lit_bufsize,
            "deflate clone layout must match source symbol-buffer geometry"
        );
        assert_eq!(self.total_size, source.total_allocation_size);

        let allocation_start = self.allocate_storage(alloc)?;
        let aligned = allocation_start.cast();

        // SAFETY: `aligned` is a fresh layout base and the cloned buffer views below are the only
        // destination views over their respective allocation regions.
        let buffers = unsafe { self.clone_buffers(aligned, source) };
        // The storage pointer must be saved before moving the live views into
        // the destination State. Every view is tied to this same allocation.
        let state_ptr = buffers.state_ptr;
        let state = source.clone_rebound(buffers, allocation_start, self.total_size);

        // SAFETY: `state_ptr` is the aligned State region in this fresh allocation and no State
        // has previously been materialized there.
        unsafe { state_ptr.write(state) };
        NonNull::new(state_ptr)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        inflate::{decompress_slice, InflateConfig, InflateStream},
        InflateFlush,
    };

    use super::*;
    use crate::allocate::{misaligned_alloc_callback, misaligned_free_callback, MisalignedAlloc};

    use core::{ffi::CStr, sync::atomic::AtomicUsize};

    #[test]
    fn owned_stream_rejects_invalid_config_and_does_not_double_free_after_end(
    ) -> Result<(), ReturnCode> {
        let invalid = DeflateConfig {
            level: i32::MAX,
            ..DeflateConfig::default()
        };
        assert!(matches!(
            DeflateStream::new(invalid),
            Err(ReturnCode::StreamError)
        ));

        let mut owned = DeflateStream::new(DeflateConfig::default())?;
        assert!(!owned.state.as_ptr().is_null());
        assert!(end(&mut owned).is_ok());
        assert!(owned.state.as_ptr().is_null());
        // Drop must not release state a second time after explicit deflateEnd.
        drop(owned);
        Ok(())
    }

    #[test]
    fn gzip_extra_wire_length_uses_low_sixteen_bits_and_exact_prefix() -> Result<(), &'static str> {
        for (length, expected) in [
            (0usize, 0u16),
            (1, 1),
            (usize::from(u16::MAX), u16::MAX),
            (usize::from(u16::MAX) + 1, 0),
            (usize::from(u16::MAX) + 2, 1),
        ] {
            assert_eq!(gzip_extra_wire_len(length), expected);
            let extra = vec![0x5au8; length];
            let mut header = gz_header {
                extra: extra.as_ptr().cast_mut(),
                extra_len: crate::c_api::uInt::try_from(length)
                    .map_err(|_| "gzip XLEN fixture exceeds C unsigned")?,
                ..gz_header::default()
            };
            // SAFETY: extra is live for the length declared by header;
            // the metadata pointer is borrowed only for this test scope.
            let view = unsafe { GzipHeaderRef::new(&mut header) };
            let emitted = view.encoded_extra();
            assert_eq!(emitted.map(<[u8]>::len), Some(usize::from(expected)));
            assert!(emitted.is_some_and(|bytes| bytes.iter().all(|&value| value == 0x5a)));
        }
        Ok(())
    }

    #[test]
    fn gzip_bound_tracks_encoded_extra_length_not_original_buffer() -> Result<(), &'static str> {
        let extra = vec![0x5au8; usize::from(u16::MAX) + 2];
        let mut reference_bound = None;
        let mut reference_bytes = None;

        for advertised_length in [1usize, extra.len()] {
            let mut header = gz_header {
                extra: extra.as_ptr().cast_mut(),
                extra_len: crate::c_api::uInt::try_from(advertised_length)
                    .map_err(|_| "extra fixture length exceeds C unsigned")?,
                ..gz_header::default()
            };
            let config = DeflateConfig {
                window_bits: 16 + crate::MAX_WBITS,
                ..DeflateConfig::default()
            };
            let mut raw = z_stream::default();
            assert_eq!(init(&mut raw, config), ReturnCode::Ok);
            // SAFETY: init created a uniquely owned, live deflate state.
            let stream = (unsafe { DeflateStream::from_stream_mut(&raw mut raw) })
                .ok_or("initialized gzip stream lacks a state")?;
            // SAFETY: the header and extra backing buffer remain live, readable,
            // and unmodified throughout compression.
            let header_status = unsafe { set_header(stream, Some(&mut header)) };
            assert_eq!(header_status, ReturnCode::Ok);
            let predicted = bound(Some(stream), 0);
            let mut encoded = [0u8; 128];
            stream.next_out = encoded.as_mut_ptr();
            stream.avail_out = crate::c_api::uInt::try_from(encoded.len())
                .map_err(|_| "output buffer exceeds C unsigned")?;
            assert_eq!(deflate(stream, DeflateFlush::Finish), ReturnCode::StreamEnd);
            let written = encoded.len()
                - usize::try_from(stream.avail_out)
                    .map_err(|_| "output remaining count exceeds pointer width")?;
            assert!(written <= predicted);
            assert_eq!(&encoded[10..13], &[1, 0, 0x5a]);
            assert!(end(stream).is_ok());

            if let Some(expected) = reference_bound {
                assert_eq!(predicted, expected);
                assert_eq!(
                    &encoded[..written],
                    reference_bytes.as_deref().ok_or("missing baseline")?
                );
            } else {
                reference_bound = Some(predicted);
                reference_bytes = Some(encoded[..written].to_vec());
            }
        }
        Ok(())
    }

    #[test]
    fn gzip_trailer_serializes_modular_c_input_total_without_large_buffers(
    ) -> Result<(), &'static str> {
        let mut raw = z_stream::default();
        let config = DeflateConfig {
            window_bits: 16 + crate::MAX_WBITS,
            ..DeflateConfig::default()
        };
        assert_eq!(init(&mut raw, config), ReturnCode::Ok);
        // SAFETY: initialized deflate state is live and borrowed exclusively.
        let stream = (unsafe { DeflateStream::from_stream_mut(&raw mut raw) })
            .ok_or("initialized stream missing deflate state")?;
        // On 64-bit Unix this sets high bits without allocating a 4 GiB input;
        // on a 32-bit C ABI it still validates the same modulo wire contract.
        let total = crate::c_api::z_size::MAX - crate::c_api::z_size::from(u32::MAX)
            + crate::c_api::z_size::from(19u32);
        stream.total_in = total;

        let mut encoded = [0u8; 128];
        stream.next_out = encoded.as_mut_ptr();
        stream.avail_out = crate::c_api::uInt::try_from(encoded.len())
            .map_err(|_| "output fixture exceeds C unsigned")?;
        assert_eq!(deflate(stream, DeflateFlush::Finish), ReturnCode::StreamEnd);
        let written = encoded.len()
            - usize::try_from(stream.avail_out)
                .map_err(|_| "output remaining count exceeds pointer width")?;
        assert!(written >= 18);
        assert_eq!(&encoded[written - 8..written - 4], &[0, 0, 0, 0]);
        assert_eq!(&encoded[written - 4..written], &[19, 0, 0, 0]);
        assert!(end(stream).is_ok());
        Ok(())
    }

    #[test]
    fn gzip_header_and_checksum_preserve_low_wire_bits() {
        let mut header = gz_header {
            time: crate::c_api::z_checksum::MAX,
            os: -1,
            hcrc: 1,
            ..gz_header::default()
        };
        // SAFETY: all header metadata pointers are null; the live C header
        // outlives the borrowed view in this test.
        let view = unsafe { GzipHeaderRef::new(&mut header) };
        assert_eq!(view.time(), u32::MAX);
        assert_eq!(view.os(), u8::MAX);
        assert!(view.hcrc());
        assert_eq!(view.flags() & 2, 2);

        let bytes = b"stream checksum high-bits";
        for initial in [
            crate::c_api::z_checksum::from(u32::MAX),
            crate::c_api::z_checksum::MAX,
        ] {
            let mut adler = StreamChecksum(initial);
            adler.update_adler(bytes);
            assert_eq!(
                adler.value(),
                crate::c_api::z_checksum::from(adler32(u32::MAX, bytes))
            );
            let mut crc = StreamChecksum(initial);
            crc.update_crc(bytes);
            assert_eq!(
                crc.value(),
                crate::c_api::z_checksum::from(crc32(u32::MAX, bytes))
            );
        }
    }

    #[test]
    fn compress_bound_never_wraps_below_its_input_length() {
        // No allocation: these are mathematical length queries only.
        let cases = [
            0,
            1,
            8,
            9,
            1_024,
            usize::MAX / 9,
            usize::MAX / 2,
            usize::MAX - 64,
            usize::MAX - 1,
            usize::MAX,
        ];
        for n in cases {
            let bound = compress_bound(n);
            assert!(bound >= n, "bound wrapped for source length {n}");
        }
        assert_eq!(compress_bound(usize::MAX), usize::MAX);

        // Existing zlib-sized requests keep exactly their historical bound.
        for n in [0, 1, 8, 9, 1_024, 65_536, 1_000_000] {
            assert_eq!(compress_bound(n), compress_bound_abi(n));
        }
    }

    #[test]
    fn huffman_tree_histograms_preserve_codes_and_bit_costs() {
        fn validate<const N: usize>(descriptor: &'static StaticTreeDesc, seed: u32) {
            let mut tree = TreeDesc::<N>::new(descriptor);
            let mut frequencies = [0u16; N];
            let mut value = seed.wrapping_add(0x9e37_79b9);
            for (index, frequency) in frequencies[..descriptor.elems].iter_mut().enumerate() {
                value ^= value << 13;
                value ^= value >> 17;
                value ^= value << 5;
                *frequency = if (seed as usize + index) % 7 == 0 {
                    0
                } else {
                    u16::from(value.to_le_bytes()[0] % 64) + 1
                };
                *tree.dyn_tree[index].freq_mut() = *frequency;
            }

            let mut optimal_bits = 0usize;
            let mut static_bits = 0usize;
            build_tree(&mut tree, &mut optimal_bits, &mut static_bits);
            let maximum = usize::from(descriptor.max_length);
            let mut kraft = 0usize;
            let mut expected_optimal = 0usize;
            let mut expected_static = 0usize;
            let mut codes = Vec::new();
            for (index, (&frequency, symbol)) in frequencies[..descriptor.elems]
                .iter()
                .zip(&tree.dyn_tree[..descriptor.elems])
                .enumerate()
            {
                let length = usize::from(symbol.len());
                if length > 0 {
                    assert!(length <= maximum);
                    kraft += 1usize << (maximum - length);
                    codes.push((symbol.code(), length));
                } else {
                    assert_eq!(frequency, 0, "a used symbol lacks a Huffman code");
                }
                if frequency != 0 {
                    let extra = descriptor
                        .extra_bits
                        .get(index.saturating_sub(descriptor.extra_base))
                        .filter(|_| index >= descriptor.extra_base)
                        .copied()
                        .map_or(0, usize::from);
                    expected_optimal += usize::from(frequency) * (length + extra);
                    if !descriptor.static_tree.is_empty() {
                        expected_static += usize::from(frequency)
                            * (usize::from(descriptor.static_tree[index].len()) + extra);
                    }
                }
            }
            assert_eq!(kraft, 1usize << maximum, "incomplete Huffman alphabet");
            assert_eq!(optimal_bits, expected_optimal);
            assert_eq!(static_bits, expected_static);
            for (position, &(left_code, left_length)) in codes.iter().enumerate() {
                for &(right_code, right_length) in &codes[position + 1..] {
                    let shared = left_length.min(right_length);
                    let mask = (1u16 << shared) - 1;
                    assert_ne!(left_code & mask, right_code & mask, "prefix collision");
                }
            }
        }

        for seed in 0..16 {
            validate::<HEAP_SIZE>(&StaticTreeDesc::L, seed);
            validate::<{ 2 * D_CODES + 1 }>(&StaticTreeDesc::D, seed);
            validate::<{ 2 * BL_CODES + 1 }>(&StaticTreeDesc::BL, seed);
        }
    }

    #[test]
    fn skewed_huffman_tree_respects_length_limit_and_exact_cost() {
        // Fibonacci frequencies produce a tree much deeper than the seven-bit
        // wire limit of the 19-symbol code-length alphabet. This exercises
        // the overflow repair rather than only the ordinary heap path.
        let mut desc = TreeDesc::<{ 2 * BL_CODES + 1 }>::new(&StaticTreeDesc::BL);
        let (mut previous, mut current) = (1u16, 1u16);
        let mut original_frequencies = [0u16; BL_CODES];
        for (symbol, frequency) in original_frequencies.iter_mut().enumerate() {
            *frequency = previous;
            *desc.dyn_tree[symbol].freq_mut() = previous;
            (previous, current) = (current, previous + current);
        }

        let (mut optimal_bits, mut static_bits) = (0usize, 0usize);
        build_tree(&mut desc, &mut optimal_bits, &mut static_bits);
        assert_eq!(desc.max_code, BL_CODES - 1);
        assert_eq!(static_bits, 0);

        let mut kraft = 0usize;
        let mut exact_cost = 0usize;
        let max_length = usize::from(StaticTreeDesc::BL.max_length);
        for symbol in 0..BL_CODES {
            let value = desc.dyn_tree[symbol];
            let length = usize::from(value.len());
            assert!((1..=max_length).contains(&length));
            kraft += 1usize << (max_length - length);
            // After gen_codes, Value::a holds the codeword, not frequency.
            // Compute the cost using the independently retained histogram.
            exact_cost += usize::from(original_frequencies[symbol])
                * (length + usize::from(StaticTreeDesc::EXTRA_BLBITS[symbol]));
        }
        assert_eq!(
            kraft,
            1usize << max_length,
            "Huffman prefix space is incomplete"
        );
        assert_eq!(
            optimal_bits, exact_cost,
            "overflow correction lost bit cost"
        );
        // The assigned bit-reversed codewords must remain prefix-free.
        for left in 0..BL_CODES {
            for right in left + 1..BL_CODES {
                let a = desc.dyn_tree[left];
                let b = desc.dyn_tree[right];
                let shared = usize::from(a.len().min(b.len()));
                let mask = (1u16 << shared) - 1;
                assert_ne!(
                    a.code() & mask,
                    b.code() & mask,
                    "Huffman codes for {left} and {right} have a prefix collision"
                );
            }
        }
    }

    #[test]
    fn huffman_heap_initialization_has_explicit_empty_and_sparse_origins(
    ) -> Result<(), crate::ReturnCode> {
        for active in [None, Some(0usize), Some(1), Some(D_CODES - 1)] {
            let mut leaves = [Value::new(0, 0); D_CODES];
            if let Some(code) = active {
                *leaves[code].freq_mut() = 1;
            }

            let mut heap = Heap::new();
            assert_eq!(heap.initialize(&mut leaves), active);
            assert_eq!(heap.active_len, usize::from(active.is_some()));
            if let Some(code) = active {
                assert_eq!(heap.nodes[Heap::SMALLEST], code);
            }

            let mut stream = DeflateStream::new(DeflateConfig::new(6))?;
            let mut desc = TreeDesc::<{ 2 * D_CODES + 1 }>::new(&StaticTreeDesc::D);
            if let Some(code) = active {
                *desc.dyn_tree[code].freq_mut() = 1;
            }
            let state = &mut *stream.state;
            build_tree(&mut desc, &mut state.opt_len, &mut state.static_len);
            let expected_max = match active {
                None | Some(0) => 1,
                Some(1) => 2,
                Some(code) => code,
            };
            assert_eq!(desc.max_code, expected_max);
            let used: Vec<usize> = desc.dyn_tree[..D_CODES]
                .iter()
                .enumerate()
                .filter_map(|(i, v)| (v.len() != 0).then_some(i))
                .collect();
            assert_eq!(used.len(), 2, "sparse Huffman alphabet {active:?}");
            if let Some(code) = active {
                assert!(used.contains(&code));
            } else {
                assert_eq!(used, [0, 1]);
            }
            for &code in &used {
                assert_eq!(desc.dyn_tree[code].len(), 1);
            }
            assert!(end(&mut stream).is_ok());
        }
        Ok(())
    }

    #[test]
    fn packed_match_bits_preserve_every_static_length_and_distance_boundary() {
        for length in 0..=u8::MAX {
            let length_bits = encode_len(&trees_tbl::STATIC_LTREE, length);
            let precalculated = trees_tbl::STATIC_LTREE_ENCODINGS[usize::from(length)];
            assert_eq!(u64::from(precalculated.code()), length_bits.value);
            assert_eq!(
                u8::try_from(precalculated.len()).ok(),
                Some(length_bits.len)
            );

            // Exhaust every valid distance; no width is narrowed after
            // concatenation. The complete match always fits the 64-bit writer.
            for distance in 1..=32768u16 {
                let distance_bits = encode_dist(&trees_tbl::STATIC_DTREE, distance);
                let mut combined = length_bits;
                combined.append(distance_bits);
                assert!(combined.len <= 64);
                if combined.len != 64 {
                    assert_eq!(combined.value >> combined.len, 0);
                }
                assert_eq!(
                    combined.value & ((1u64 << length_bits.len) - 1),
                    length_bits.value
                );
            }
        }
    }

    #[test]
    fn bit_writer_matches_independent_lsb_first_stream_across_reservoir_boundaries(
    ) -> Result<(), crate::ReturnCode> {
        let mut stream = DeflateStream::new(DeflateConfig::new(6))?;
        let writer = &mut stream.state.bit_writer;
        let mut expected_bits = Vec::new();
        for len in [0u8, 1, 7, 8, 15, 16, 31, 32, 33, 63, 64, 1, 64, 8, 0, 7, 63] {
            let value = if len == 0 { 0 } else { u64::MAX >> (64 - len) };
            let packed = PackedBits::new(value, len);
            writer.send_bits(packed.value, packed.len);
            for bit in 0..len {
                expected_bits.push(((value >> bit) & 1) != 0);
            }
        }
        writer.emit_align();
        let mut expected = vec![0u8; expected_bits.len().div_ceil(8)];
        for (index, bit) in expected_bits.iter().copied().enumerate() {
            if bit {
                expected[index / 8] |= 1 << (index % 8);
            }
        }
        assert_eq!(writer.pending.pending(), expected);
        assert_eq!(writer.bits_valid, 0);
        assert_eq!(writer.bit_buffer, 0);
        let last_bits = expected_bits.len() % 8;
        assert_eq!(
            usize::from(writer.bits_used),
            if last_bits == 0 { 8 } else { last_bits }
        );
        Ok(())
    }

    #[test]
    #[should_panic(expected = "packed DEFLATE sequence contains bits beyond its width")]
    fn packed_bits_reject_unaccounted_high_bits() {
        let _ = PackedBits::new(0b100, 2);
    }

    #[test]
    #[should_panic(expected = "invalid DEFLATE Huffman code length")]
    fn packed_huffman_length_is_checked_before_narrowing() {
        // The Huffman node length shares storage with its 16-bit parent index.
        // A malformed width of 256 must not silently encode as zero bits.
        let _ = PackedBits::code(Value::new(0, 256));
    }

    #[test]
    #[should_panic(expected = "packed DEFLATE sequence exceeds reservoir width")]
    fn packed_bits_reject_overwide_concatenation() {
        let mut bits = PackedBits::new(u64::MAX, 64);
        bits.append(PackedBits::new(1, 1));
    }

    #[test]
    fn code_length_runs_cover_literal_repeat_and_zero_boundaries() {
        use CodeLengthRun::{Literal, RepeatPrevious, RepeatZero};

        let cases: &[(Vec<u16>, Vec<CodeLengthRun>)] = &[
            (
                vec![0; 2],
                vec![Literal {
                    length: 0,
                    count: 2,
                }],
            ),
            (
                vec![0; 3],
                vec![RepeatZero {
                    code: REPZ_3_10,
                    count: 3,
                }],
            ),
            (
                vec![0; 10],
                vec![RepeatZero {
                    code: REPZ_3_10,
                    count: 10,
                }],
            ),
            (
                vec![0; 11],
                vec![RepeatZero {
                    code: REPZ_11_138,
                    count: 11,
                }],
            ),
            (
                vec![0; 138],
                vec![RepeatZero {
                    code: REPZ_11_138,
                    count: 138,
                }],
            ),
            (
                vec![1; 3],
                vec![Literal {
                    length: 1,
                    count: 3,
                }],
            ),
            (
                vec![1; 6],
                vec![RepeatPrevious {
                    length: 1,
                    count: 5,
                    emit_initial: true,
                }],
            ),
            (
                vec![1; 10],
                vec![
                    RepeatPrevious {
                        length: 1,
                        count: 6,
                        emit_initial: true,
                    },
                    RepeatPrevious {
                        length: 1,
                        count: 3,
                        emit_initial: false,
                    },
                ],
            ),
        ];
        for (lengths, expected) in cases {
            let tree: Vec<Value> = lengths.iter().map(|&len| Value::new(0, len)).collect();
            let mut actual = Vec::new();
            for_each_code_length_run(&tree, lengths.len() - 1, |run| actual.push(run));
            assert_eq!(&actual, expected, "RLE boundary {lengths:?}");
        }

        // Every run expands to the original lengths, even when zeros and
        // nonzero codes alternate and split across repeat boundaries.
        for seed in 0..32u32 {
            let mut state = seed ^ 0x9e37_79b9;
            let mut lengths = Vec::with_capacity(512);
            for _ in 0..512 {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let raw = state.to_le_bytes()[0];
                lengths.push(if raw & 3 == 0 { 0 } else { u16::from(raw % 16) });
            }
            let tree: Vec<Value> = lengths.iter().map(|&len| Value::new(0, len)).collect();
            let mut rebuilt = Vec::new();
            for_each_code_length_run(&tree, lengths.len() - 1, |run| match run {
                Literal { length, count } => {
                    rebuilt.extend(core::iter::repeat_n(length, usize::from(count)));
                }
                RepeatPrevious {
                    length,
                    count,
                    emit_initial,
                } => {
                    if emit_initial {
                        rebuilt.push(length);
                    } else {
                        assert_eq!(rebuilt.last().copied(), Some(length));
                    }
                    rebuilt.extend(core::iter::repeat_n(length, usize::from(count)));
                }
                RepeatZero { count, .. } => {
                    rebuilt.extend(core::iter::repeat_n(0, usize::from(count)));
                }
            });
            assert_eq!(rebuilt, lengths, "random RLE case seed {seed}");
        }
    }

    #[test]
    fn unflushed_block_span_tracks_slid_out_prefixes() {
        let window = 1usize << crate::MAX_WBITS;
        let slid_origin = -(1isize << crate::MAX_WBITS);
        assert_eq!(unflushed_block_span(100, 10), (Some(10), 90));
        assert_eq!(unflushed_block_span(100, 100), (Some(100), 0));
        assert_eq!(unflushed_block_span(100, -1), (None, 101));
        assert_eq!(
            unflushed_block_span(window, slid_origin),
            (None, window * 2)
        );
        assert_eq!(
            unflushed_block_span(window + 4, slid_origin),
            (None, window * 2 + 4)
        );
    }

    #[test]
    #[should_panic(expected = "block start is after the current position")]
    fn unflushed_block_span_rejects_future_origin() {
        let _ = unflushed_block_span(10, 11);
    }

    #[test]
    #[should_panic(expected = "unflushed block length exceeds host integer range")]
    fn unflushed_block_span_rejects_overflow() {
        let _ = unflushed_block_span(usize::MAX, isize::MIN);
    }

    #[test]
    fn compressed_block_survives_multiple_window_slides_before_flush() -> Result<(), &'static str> {
        // Compressible input produces few tokens but crosses several window
        // slides before the DEFLATE block is emitted. Its origin must remain
        // outside the current window and its length must not narrow or wrap.
        let source = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcd".repeat(8_000);
        for level in [1, 6, 9] {
            let mut stream = DeflateStream::new(DeflateConfig::new(level))
                .map_err(|_| "valid deflate initialization failed")?;
            let mut output = vec![0u8; compress_bound(source.len()) + 64];
            stream.next_in = source.as_ptr().cast_mut();
            stream.avail_in = crate::c_api::uInt::try_from(source.len())
                .map_err(|_| "test input exceeds ABI length")?;
            stream.next_out = output.as_mut_ptr();
            stream.avail_out = crate::c_api::uInt::try_from(output.len())
                .map_err(|_| "test output exceeds ABI length")?;

            assert_eq!(deflate(&mut stream, DeflateFlush::NoFlush), ReturnCode::Ok);
            assert_eq!(stream.avail_in, 0);
            assert!(
                stream.state.block_start < 0,
                "level {level} did not leave a block spanning a window slide"
            );
            let (window_offset, block_len) =
                unflushed_block_span(stream.state.strstart, stream.state.block_start);
            assert_eq!(window_offset, None);
            assert!(block_len > stream.state.w_size);
            assert!(block_len <= source.len());

            assert_eq!(
                deflate(&mut stream, DeflateFlush::Finish),
                ReturnCode::StreamEnd
            );
            let written = usize::try_from(stream.total_out)
                .map_err(|_| "test output count exceeds host usize")?;
            assert!(end(&mut stream).is_ok());
            let mut decoded = vec![0u8; source.len()];
            let (restored, code) = crate::decompress_slice(
                &mut decoded,
                &output[..written],
                crate::InflateConfig::default(),
            );
            assert_eq!(code, ReturnCode::Ok, "level {level} did not decode");
            assert_eq!(restored, source, "level {level} round-trip mismatch");
        }
        Ok(())
    }

    #[test]
    fn all_valid_window_bits_slide_positions_and_round_trip() -> Result<(), &'static str> {
        // Cross multiple window slides at every supported zlib window size.
        // Exercising the encoder and inflater catches stale match positions
        // that a direct hash-table test cannot observe.
        let source = b"ABCDxyzXYZ01234abcdefghij".repeat(6_000);
        for window_bits in 9..=crate::MAX_WBITS {
            for level in [1, 6, 9] {
                let config = DeflateConfig {
                    level,
                    window_bits,
                    ..DeflateConfig::default()
                };
                let mut stream = DeflateStream::new(config)
                    .map_err(|_| "valid window configuration failed to initialize")?;
                let mut output = vec![0u8; compress_bound(source.len()) + 64];
                stream.next_in = source.as_ptr().cast_mut();
                stream.avail_in = crate::c_api::uInt::try_from(source.len())
                    .map_err(|_| "test input exceeds ABI length")?;
                stream.next_out = output.as_mut_ptr();
                stream.avail_out = crate::c_api::uInt::try_from(output.len())
                    .map_err(|_| "test output exceeds ABI length")?;

                assert_eq!(
                    deflate(&mut stream, DeflateFlush::Finish),
                    ReturnCode::StreamEnd,
                    "window bits {window_bits}, level {level}"
                );
                assert_eq!(stream.avail_in, 0);
                let written = usize::try_from(stream.total_out)
                    .map_err(|_| "test output count exceeds host usize")?;
                assert!(end(&mut stream).is_ok());
                let mut decoded = vec![0u8; source.len()];
                let (restored, status) = crate::decompress_slice(
                    &mut decoded,
                    &output[..written],
                    crate::InflateConfig::default(),
                );
                assert_eq!(
                    status,
                    ReturnCode::Ok,
                    "window bits {window_bits}, level {level}"
                );
                assert_eq!(
                    restored, source,
                    "round trip window bits {window_bits}, level {level}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn resetting_a_state_with_invalid_compression_level_preserves_live_cursors(
    ) -> Result<(), ReturnCode> {
        let mut stream = DeflateStream::new(DeflateConfig::new(6))?;
        let original_level = stream.state.level;
        let original_total_in = stream.total_in;
        let original_total_out = stream.total_out;
        let original_start = stream.state.strstart;
        for invalid in [i8::MIN, -1, 10, i8::MAX] {
            stream.state.level = invalid;
            assert_eq!(reset(&mut stream), ReturnCode::StreamError);
            assert_eq!(stream.total_in, original_total_in);
            assert_eq!(stream.total_out, original_total_out);
            assert_eq!(stream.state.strstart, original_start);
        }
        stream.state.level = original_level;
        assert_eq!(reset(&mut stream), ReturnCode::Ok);
        let _ = end(&mut stream);
        Ok(())
    }

    #[test]
    fn level_change_flushes_block_with_origin_outside_window() -> Result<(), &'static str> {
        let source = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcd".repeat(8_000);
        for (initial_level, next_level) in [(1, 6), (6, 9), (9, 1)] {
            let mut stream = DeflateStream::new(DeflateConfig::new(initial_level))
                .map_err(|_| "valid deflate initialization failed")?;
            let mut output = vec![0u8; compress_bound(source.len()) + 256];
            stream.next_in = source.as_ptr().cast_mut();
            stream.avail_in = crate::c_api::uInt::try_from(source.len())
                .map_err(|_| "test input exceeds ABI length")?;
            stream.next_out = output.as_mut_ptr();
            stream.avail_out = crate::c_api::uInt::try_from(output.len())
                .map_err(|_| "test output exceeds ABI length")?;

            assert_eq!(deflate(&mut stream, DeflateFlush::NoFlush), ReturnCode::Ok);
            assert_eq!(stream.avail_in, 0);
            assert!(stream.state.block_start < 0);
            assert!(
                unflushed_block_span(stream.state.strstart, stream.state.block_start).1
                    > stream.state.w_size
            );

            assert_eq!(
                params(&mut stream, next_level, Strategy::Default),
                ReturnCode::Ok,
                "level transition {initial_level} -> {next_level} failed"
            );
            assert_eq!(i32::from(stream.state.level), next_level);
            assert!(stream.state.block_start >= 0);
            assert_eq!(
                stream.state.block_start.unsigned_abs(),
                stream.state.strstart
            );
            assert_eq!(
                deflate(&mut stream, DeflateFlush::Finish),
                ReturnCode::StreamEnd
            );
            let written =
                usize::try_from(stream.total_out).map_err(|_| "test output exceeds host usize")?;
            assert!(end(&mut stream).is_ok());
            let mut decoded = vec![0u8; source.len()];
            let (restored, status) = crate::decompress_slice(
                &mut decoded,
                &output[..written],
                crate::InflateConfig::default(),
            );
            assert_eq!(status, ReturnCode::Ok);
            assert_eq!(restored, source);
        }
        Ok(())
    }

    #[test]
    fn stored_header_emits_real_length_without_payload_or_rewrite() -> Result<(), crate::ReturnCode>
    {
        for len in [0usize, 1, 255, 256, 32_768, MAX_STORED] {
            let mut stream = DeflateStream::new(DeflateConfig::new(0))?;
            emit_stored_block_header(&mut stream.state, len, true);
            let fields = stored_block_length_fields(len);
            let pending = stream.state.bit_writer.pending.pending();
            assert_eq!(pending, &[&[1u8][..], &fields[..]].concat());
            #[cfg(feature = "ZLIB_DEBUG")]
            {
                assert_eq!(stream.state.bit_writer.compressed_len, 32);
                assert_eq!(stream.state.bit_writer.bits_sent, 40);
            }
            assert!(end(&mut stream).is_ok());
        }
        Ok(())
    }

    #[test]
    fn stored_length_fields_cover_entire_wire_range() {
        for length in [0usize, 1, 255, 256, 8_192, 32_768, MAX_STORED] {
            let fields = stored_block_length_fields(length);
            assert_eq!(&fields[..2], &length.to_le_bytes()[..2]);
            assert_eq!(&fields[2..], &(!length).to_le_bytes()[..2]);
        }
    }

    #[test]
    #[should_panic(expected = "DEFLATE stored block payload must fit in 16 bits")]
    fn stored_length_fields_reject_overflow_before_writing() {
        let _ = stored_block_length_fields(MAX_STORED + 1);
    }

    #[test]
    fn stored_streaming_fragmentation_preserves_bytes_across_window_slides() {
        let input: Vec<u8> = (0usize..190_001)
            .map(|index| ((index * 37) ^ (index / 29)).to_le_bytes()[0])
            .collect();

        for window_bits in [-crate::MAX_WBITS, crate::MAX_WBITS, 16 + crate::MAX_WBITS] {
            for mem_level in [1, 8] {
                let config = DeflateConfig {
                    window_bits,
                    mem_level,
                    ..DeflateConfig::new(0)
                };
                for chunk_size in [7, 509, 32_768] {
                    let compressed = compress_with_output_chunks(&input, config, chunk_size);
                    let mut restored = vec![0u8; input.len()];
                    let (decoded, status) = crate::decompress_slice(
                        &mut restored,
                        &compressed,
                        InflateConfig { window_bits },
                    );
                    assert_eq!(
                        status, ReturnCode::Ok,
                        "stored roundtrip status for window_bits={window_bits}, mem_level={mem_level}, out_chunk={chunk_size}"
                    );
                    assert_eq!(
                        decoded, input,
                        "stored roundtrip mismatch for window_bits={window_bits}, mem_level={mem_level}, out_chunk={chunk_size}"
                    );
                }
            }
        }
    }

    #[test]
    fn stored_and_compressed_level_transitions_preserve_long_history() -> Result<(), &'static str> {
        let before: Vec<u8> = (0usize..160_000)
            .map(|index| ((index * 17) ^ (index / 43)).to_le_bytes()[0])
            .collect();
        let after = b"level-transition-stored-and-huffman/".repeat(3_000);
        for (old_level, new_level) in [(0, 6), (6, 0)] {
            let mut raw = z_stream::default();
            let config = DeflateConfig {
                mem_level: 1,
                ..DeflateConfig::new(old_level)
            };
            assert_eq!(init(&mut raw, config), ReturnCode::Ok);
            // SAFETY: init created the uniquely owned live compression state.
            let stream = unsafe { DeflateStream::from_stream_mut(&raw mut raw) }
                .ok_or("initialized compression stream missing its state")?;
            let mut output = vec![0u8; compress_bound(before.len() + after.len()) + 1024];
            stream.next_in = before.as_ptr().cast_mut();
            stream.avail_in = before.len().try_into().map_err(|_| "input too long")?;
            stream.next_out = output.as_mut_ptr();
            stream.avail_out = output.len().try_into().map_err(|_| "output too long")?;

            while stream.avail_in > 0 {
                assert_eq!(deflate(stream, DeflateFlush::NoFlush), ReturnCode::Ok);
                assert_ne!(stream.avail_out, 0);
            }
            assert_eq!(params(stream, new_level, Strategy::Default), ReturnCode::Ok);
            assert_eq!(i32::from(stream.state.level), new_level);

            stream.next_in = after.as_ptr().cast_mut();
            stream.avail_in = after.len().try_into().map_err(|_| "tail too long")?;
            loop {
                match deflate(stream, DeflateFlush::Finish) {
                    ReturnCode::Ok => assert_ne!(stream.avail_out, 0),
                    ReturnCode::StreamEnd => break,
                    other => panic!("stored-level transition {old_level}->{new_level}: {other:?}"),
                }
            }
            let written =
                usize::try_from(stream.total_out).map_err(|_| "output size exceeds usize")?;
            assert_eq!(stream.avail_in, 0);
            assert!(end(stream).is_ok());

            let mut original = before.clone();
            original.extend_from_slice(&after);
            let mut restored = vec![0u8; original.len()];
            let (decoded, code) = crate::decompress_slice(
                &mut restored,
                &output[..written],
                InflateConfig::default(),
            );
            assert_eq!(code, ReturnCode::Ok);
            assert_eq!(decoded, original, "level {old_level}->{new_level}");
        }
        Ok(())
    }

    #[test]
    fn stored_mode_never_reads_a_slid_out_block_start() -> Result<(), crate::ReturnCode> {
        let mut stream = DeflateStream::new(DeflateConfig::new(0))?;
        stream.state.block_start = -1;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_compressor(&mut stream, |compressor| {
                self::algorithm::run(compressor, DeflateFlush::Finish)
            })
        }));
        assert!(
            result.is_err(),
            "unavailable stored history must fail before accessing output"
        );
        stream.state.block_start = 0;
        assert!(end(&mut stream).is_ok());
        Ok(())
    }

    #[test]
    fn stored_block_large_lengths_preserve_bytes_and_debug_bit_counts(
    ) -> Result<(), crate::ReturnCode> {
        for len in [8_191usize, 8_192, 16_384, 32_768, usize::from(u16::MAX)] {
            let source: Vec<u8> = (0..len)
                .map(|i| (i * 29 + i / 13).to_le_bytes()[0])
                .collect();
            // A maximum-sized stored block needs five bytes of header space beyond 64 KiB.
            let mut stream = DeflateStream::new(DeflateConfig {
                mem_level: 9,
                ..DeflateConfig::new(0)
            })?;
            // SAFETY: source is an initialized, non-overlapping buffer and its bytes
            // remain live while the stream copies the complete window prefix.
            unsafe {
                stream
                    .state
                    .window
                    .copy_and_initialize(0..len, source.as_ptr());
            }
            zng_tr_stored_block(&mut stream.state, 0..len, true);

            let pending = stream.state.bit_writer.pending.pending();
            assert_eq!(pending.len(), len + 5);
            assert_eq!(pending[0], 1);
            assert_eq!(&pending[1..3], &len.to_le_bytes()[..2]);
            assert_eq!(&pending[3..5], &(!len).to_le_bytes()[..2]);
            assert_eq!(&pending[5..], source.as_slice());

            #[cfg(feature = "ZLIB_DEBUG")]
            {
                assert_eq!(
                    stream.state.bit_writer.compressed_len,
                    32 + len * 8,
                    "stored {len}-byte payload must count its full width"
                );
                assert_eq!(
                    stream.state.bit_writer.bits_sent,
                    40 + len * 8,
                    "stored {len}-byte payload and header must count exactly"
                );
            }
            assert!(end(&mut stream).is_ok());
        }
        Ok(())
    }

    #[test]
    fn window_refill_seeds_hash_by_history_representation_not_tuned_chain(
    ) -> Result<(), &'static str> {
        let source = b"ABCDABCD0123456789".repeat(32);
        let rolling_seed = RollHashCalc::update_hash(u32::from(source[0]), u32::from(source[1]));

        for (level, chain, expected_seed) in [
            (7, 4_096, 0),
            (7, 128, 0),
            (9, 128, rolling_seed),
            (9, 4_096, rolling_seed),
        ] {
            let mut stream = DeflateStream::new(DeflateConfig::new(level))
                .map_err(|_| "valid deflate initialization failed")?;
            assert_eq!(tune(&mut stream, 8, 16, 128, chain), ReturnCode::Ok);
            let mut input = InputCursor::from_slice(&source, stream.total_in)
                .ok_or("test source length must fit zlib uInt")?;
            let mut checksum = StreamChecksum::from_stream(&stream);
            fill_window_core(&mut stream.state, &mut input, &mut checksum);
            assert_eq!(input.avail_in, 0, "the refill should consume all the input");
            assert_eq!(
                stream.state.ins_h, expected_seed,
                "level {level}, tuned chain {chain}, hash mode {:?}",
                stream.state.hash_calc_variant
            );
            assert!(end(&mut stream).is_ok());
        }
        Ok(())
    }

    #[test]
    fn level_changes_preserve_live_hash_table_representation() -> Result<(), crate::ReturnCode> {
        let dictionary: Vec<u8> = (0..4_096usize)
            .map(|i| (i * 29 + i / 97).to_le_bytes()[0])
            .collect();

        for (old_level, new_level, expected_roll) in [(7, 9, false), (9, 7, true), (6, 9, false)] {
            let mut stream = DeflateStream::new(DeflateConfig::new(old_level))?;
            assert_eq!(set_dictionary(&mut stream, &dictionary), ReturnCode::Ok);
            assert!(stream.state.strstart > 0);
            assert!(stream.state.head.as_slice().iter().any(|entry| *entry != 0));
            assert_eq!(
                params(&mut stream, new_level, Strategy::Default),
                ReturnCode::Ok
            );

            assert_eq!(
                matches!(stream.state.hash_calc_variant, HashCalcVariant::Roll),
                expected_roll,
                "level change {old_level} to {new_level} reinterpreted existing hash history",
            );
            assert!(end(&mut stream).is_ok());
        }
        Ok(())
    }

    #[test]
    fn tune_wide_parameters_do_not_wrap_to_zero() -> Result<(), &'static str> {
        let mut raw = z_stream::default();
        assert_eq!(init(&mut raw, DeflateConfig::new(9)), ReturnCode::Ok);
        // SAFETY: init succeeded and its state is live, uniquely borrowed by this test.
        let stream = unsafe { DeflateStream::from_stream_mut(&raw mut raw) }
            .ok_or("deflate state missing after init")?;

        // All four tunables are upper bounds or comparison thresholds. A positive
        // value beyond u16 range must not become zero after narrowing.
        assert_eq!(tune(stream, 65_536, 65_536, 65_536, 65_536), ReturnCode::Ok);
        assert_eq!(stream.state.good_match, u16::MAX);
        assert_eq!(stream.state.max_lazy_match, u16::MAX);
        assert_eq!(stream.state.max_insert_length, u16::MAX);
        assert_eq!(stream.state.nice_match, u16::MAX);
        assert_eq!(stream.state.max_chain_length, u16::MAX);

        // A negative C i32 forwarded by the C wrapper becomes a large usize.
        // It must remain a high threshold rather than wrapping into a short one.
        assert_eq!(
            tune(stream, usize::MAX, usize::MAX, usize::MAX, usize::MAX),
            ReturnCode::Ok
        );
        assert_eq!(stream.state.max_chain_length, u16::MAX);
        assert!(end(stream).is_ok());
        Ok(())
    }

    #[test]
    fn call_cursors_commit_all_abi_fields_and_rebind_history_per_turn(
    ) -> Result<(), crate::ReturnCode> {
        let source = *b"cursorruntime";
        let mut destination = [0u8; 16];
        let mut stream = DeflateStream::new(DeflateConfig::new(6))?;
        stream.next_in = source.as_ptr().cast_mut();
        stream.avail_in = 8;
        stream.total_in = crate::c_api::z_size::MAX - 2;
        stream.next_out = destination.as_mut_ptr();
        stream.avail_out = 16;
        stream.total_out = crate::c_api::z_size::MAX - 1;
        stream.data_type = crate::c_api::Z_TEXT;

        let first = with_compressor(&mut stream, |compressor| {
            assert_eq!(compressor.state.data_type, crate::c_api::Z_TEXT);
            assert_eq!(compressor.input.consumed_here(), 0);
            compressor
                .input
                .copy_to_window(&mut compressor.state.window, 0..5);
            compressor.output.write_all(&source[..3]);
            compressor.checksum.update_adler(&source[..5]);
            compressor.state.data_type = crate::c_api::Z_BINARY;
            compressor.input.consumed_here()
        });
        assert_eq!(first, 5);
        assert_eq!(stream.avail_in, 3);
        assert_eq!(stream.avail_out, 13);
        assert_eq!(stream.total_in, 2);
        assert_eq!(stream.total_out, 1);
        assert_eq!(stream.next_in, source.as_ptr().wrapping_add(5).cast_mut());
        assert_eq!(stream.next_out, destination.as_mut_ptr().wrapping_add(3));
        assert_eq!(&destination[..3], &source[..3]);
        assert_eq!(&stream.state.window.filled()[..5], &source[..5]);
        assert_eq!(
            stream.adler,
            crate::adler32::adler32(1u32, &source[..5]).into()
        );
        assert_eq!(stream.data_type, crate::c_api::Z_BINARY);

        // A second invocation must not reuse the previous call's readable
        // history allowance or reset the newly committed public counters.
        let stream_adler = stream.adler;
        let second = with_compressor(&mut stream, |compressor| {
            assert_eq!(compressor.input.consumed_here(), 0);
            assert_eq!(compressor.input.avail_in, 3);
            assert_eq!(compressor.output.avail_out, 13);
            assert_eq!(compressor.checksum.value(), stream_adler);
            compressor.input.consume(2);
            compressor.output.write_all(&source[5..7]);
            (
                compressor.input.consumed_here(),
                compressor.output.total_out,
            )
        });
        assert_eq!(second, (2, 3));
        assert_eq!(stream.avail_in, 1);
        assert_eq!(stream.avail_out, 11);
        assert_eq!(stream.total_in, 4);
        assert_eq!(stream.total_out, 3);
        assert_eq!(&destination[..3], &source[..3]);
        assert_eq!(&destination[3..5], &source[5..7]);
        assert!(end(&mut stream).is_ok());
        Ok(())
    }

    #[test]
    fn input_cursor_history_is_scoped_to_current_call() -> Result<(), &'static str> {
        let source = b"two separate caller buffers";
        let mut first = InputCursor::from_slice(&source[..8], crate::c_api::z_size::MAX - 1)
            .ok_or("the eight-byte input must fit in zlib uInt")?;
        assert_eq!(first.consumed_here(), 0);
        first.consume(5);
        assert_eq!(first.consumed_here(), 5);
        assert_eq!(first.avail_in, 3);
        assert_eq!(first.total_in, 3);

        // Changing the caller input buffer resets the readable history budget, even
        // though total_in carries the wrapped counter from the previous invocation.
        let next = InputCursor::from_slice(&source[8..], first.total_in)
            .ok_or("the second input must fit in zlib uInt")?;
        assert_eq!(next.consumed_here(), 0);
        assert_eq!(next.total_in, 3);
        Ok(())
    }

    #[test]
    fn deflate_total_out_wraps_like_unsigned_c_counter() -> Result<(), &'static str> {
        let mut raw = z_stream::default();
        assert_eq!(init(&mut raw, DeflateConfig::default()), ReturnCode::Ok);
        let mut output = [0u8; 128];
        // SAFETY: init succeeded and the stream owns a live, exclusively borrowed deflate state.
        let stream = unsafe { DeflateStream::from_stream_mut(&raw mut raw) }
            .ok_or("freshly initialized deflate stream was not usable")?;
        stream.total_out = crate::c_api::z_size::MAX - 1;
        stream.next_out = output.as_mut_ptr();
        stream.avail_out = 128; // Fixed-size output array.

        assert_eq!(deflate(stream, DeflateFlush::Finish), ReturnCode::StreamEnd);
        let written = output.len() - stream.avail_out as usize;
        assert!(written >= 8);
        assert_eq!(
            stream.total_out,
            (crate::c_api::z_size::MAX - 1).wrapping_add(written as crate::c_api::z_size)
        );
        assert!(end(stream).is_ok());
        Ok(())
    }

    #[test]
    fn invalid_match_tokens_do_not_modify_encoder_state() -> Result<(), &'static str> {
        let mut raw = z_stream::default();
        assert_eq!(init(&mut raw, DeflateConfig::default()), ReturnCode::Ok);
        // SAFETY: init created the live uniquely owned Deflate state.
        let stream = unsafe { DeflateStream::from_stream_mut(&raw mut raw) }
            .ok_or("initialized stream has no state")?;
        {
            let state = &mut *stream.state;
            let max_distance = state.w_mask() + 1;
            let original = (
                state.matches,
                state.sym_buf.iter().collect::<Vec<_>>(),
                state
                    .l_desc
                    .dyn_tree
                    .iter()
                    .map(|value| value.freq())
                    .collect::<Vec<_>>(),
                state
                    .d_desc
                    .dyn_tree
                    .iter()
                    .map(|value| value.freq())
                    .collect::<Vec<_>>(),
            );

            for (distance, len, limit) in [
                (0, 0, max_distance),
                (max_distance + 1, 0, max_distance),
                (usize::from(u16::MAX) + 1, 0, max_distance),
                (1, usize::from(u8::MAX) + 1, max_distance),
                (max_distance + 1, 0, max_distance + 1),
            ] {
                assert!(
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        state.tally_dist_with_limit(distance, len, limit);
                    }))
                    .is_err(),
                    "invalid match {distance}/{len} must be rejected"
                );
                assert_eq!(
                    (
                        state.matches,
                        state.sym_buf.iter().collect::<Vec<_>>(),
                        state
                            .l_desc
                            .dyn_tree
                            .iter()
                            .map(|value| value.freq())
                            .collect::<Vec<_>>(),
                        state
                            .d_desc
                            .dyn_tree
                            .iter()
                            .map(|value| value.freq())
                            .collect::<Vec<_>>(),
                    ),
                    original,
                    "failed match must leave tokens and Huffman frequencies intact"
                );
            }

            // Full window distance and the 258-byte maximum are both legal.
            let _ = state.tally_dist_with_limit(
                max_distance,
                STD_MAX_MATCH - STD_MIN_MATCH,
                max_distance,
            );
            assert_eq!(state.matches, original.0 + 1);
            assert_eq!(
                state
                    .sym_buf
                    .iter()
                    .last()
                    .map(|(dist, len)| (usize::from(dist), len)),
                Some((max_distance, u8::MAX))
            );
        }
        assert!(end(stream).is_ok());
        Ok(())
    }

    #[test]
    fn detect_data_type_basic() {
        let empty = || [Value::new(0, 0); LITERALS];

        assert_eq!(State::detect_data_type(&empty()), DataType::Binary);

        let mut binary = empty();
        binary[0] = Value::new(1, 0);
        assert_eq!(State::detect_data_type(&binary), DataType::Binary);

        let mut text = empty();
        text[b'\r' as usize] = Value::new(1, 0);
        assert_eq!(State::detect_data_type(&text), DataType::Text);

        let mut text = empty();
        text[b'a' as usize] = Value::new(1, 0);
        assert_eq!(State::detect_data_type(&text), DataType::Text);

        let mut non_text = empty();
        non_text[7] = Value::new(1, 0);
        assert_eq!(State::detect_data_type(&non_text), DataType::Binary);
    }

    #[test]
    fn data_type_override_round_trips_through_state_only_encoder() {
        let mut raw = z_stream::default();
        assert_eq!(init(&mut raw, DeflateConfig::default()), ReturnCode::Ok);
        let stream = unsafe { DeflateStream::from_stream_mut(&raw mut raw) }.unwrap();

        // zlib only auto-detects when this field is Z_UNKNOWN. Preserve arbitrary caller metadata.
        const CALLER_DATA_TYPE: i32 = 12_345;
        stream.data_type = CALLER_DATA_TYPE;

        let input = b"plain text that is compressible plain text that is compressible";
        let mut output = [0u8; 1024];
        stream.next_in = input.as_ptr().cast_mut();
        stream.avail_in = input.len() as _;
        stream.next_out = output.as_mut_ptr();
        stream.avail_out = output.len() as _;

        assert_eq!(deflate(stream, DeflateFlush::Finish), ReturnCode::StreamEnd);
        assert_eq!(stream.data_type, CALLER_DATA_TYPE);
        assert_eq!(stream.state.data_type, CALLER_DATA_TYPE);
        assert!(end(stream).is_ok());
    }

    #[test]
    fn from_stream_mut() {
        unsafe {
            assert!(DeflateStream::from_stream_mut(core::ptr::null_mut()).is_none());

            let mut stream = z_stream::default();
            assert!(DeflateStream::from_stream_mut(&raw mut stream).is_none());

            // state is still NULL
            assert!(DeflateStream::from_stream_mut(&raw mut stream).is_none());

            init(&mut stream, DeflateConfig::default());
            let stream = DeflateStream::from_stream_mut(&raw mut stream);
            assert!(stream.is_some());

            assert!(end(stream.unwrap()).is_ok());
        }
    }

    #[cfg(feature = "c-allocator")]
    unsafe extern "C" fn fail_nth_allocation<const N: usize>(
        opaque: crate::c_api::voidpf,
        items: crate::c_api::uInt,
        size: crate::c_api::uInt,
    ) -> crate::c_api::voidpf {
        let count = unsafe { &*(opaque as *const AtomicUsize) };

        if count.fetch_add(1, core::sync::atomic::Ordering::Relaxed) != N {
            // must use the C allocator internally because (de)allocation is based on function
            // pointer values and because we don't use the rust allocator directly, the allocation
            // logic will store the pointer to the start at the start of the allocation.
            unsafe { (crate::allocate::C.zalloc)(opaque, items, size) }
        } else {
            core::ptr::null_mut()
        }
    }

    #[test]
    #[cfg(feature = "c-allocator")]
    fn init_invalid_allocator() {
        {
            let atomic = AtomicUsize::new(0);
            let mut stream = z_stream {
                zalloc: Some(fail_nth_allocation::<0>),
                zfree: Some(crate::allocate::C.zfree),
                opaque: (&raw const atomic).cast::<core::ffi::c_void>().cast_mut(),
                ..z_stream::default()
            };
            assert_eq!(
                init(&mut stream, DeflateConfig::default()),
                ReturnCode::MemError
            );
        }
    }

    #[test]
    fn near_optimal_partition_can_merge_beyond_stored_limit() {
        fn repeated_tokens(mut bytes: usize) -> Vec<near_optimal::ParseToken> {
            let mut tokens = Vec::new();
            tokens.push(near_optimal::ParseToken::Literal(b'a'));
            bytes -= 1;
            while bytes >= STD_MAX_MATCH {
                tokens.push(near_optimal::ParseToken::Match {
                    length: STD_MAX_MATCH as u16,
                    distance: 1,
                });
                bytes -= STD_MAX_MATCH;
            }
            if bytes >= STD_MIN_MATCH {
                tokens.push(near_optimal::ParseToken::Match {
                    length: bytes as u16,
                    distance: 1,
                });
            } else {
                tokens.extend(core::iter::repeat_n(
                    near_optimal::ParseToken::Literal(b'a'),
                    bytes,
                ));
            }
            tokens
        }

        let mut raw = z_stream::default();
        assert_eq!(
            init_with_lit_bufsize(
                &mut raw,
                DeflateConfig::new(9),
                Some(near_optimal::NEAR_OPTIMAL_LIT_BUFSIZE),
            ),
            ReturnCode::Ok
        );
        let stream = unsafe { DeflateStream::from_stream_mut(&raw mut raw) }.unwrap();
        let mut left = near_optimal::PlannedBlock {
            range: 0..60_000,
            tokens: repeated_tokens(60_000),
        };
        let mut right = near_optimal::PlannedBlock {
            range: 60_000..120_000,
            tokens: repeated_tokens(60_000),
        };

        let outcome =
            near_optimal::optimize_token_partition(&mut stream.state, &mut left, &mut right);
        let near_optimal::BoundaryOptimization::Merged(merged) = outcome else {
            panic!("expected identical compressible blocks to merge");
        };
        assert_eq!(merged.range, 0..120_000);
        assert!(merged.tokens.len() <= near_optimal::MAX_BLOCK_TOKENS);
        assert_eq!(left.tokens.as_slice(), &[]);
        assert_eq!(right.tokens.as_slice(), &[]);
        assert!(end(stream).is_ok());
    }

    #[test]
    fn near_optimal_partition_respects_huffman_token_capacity() {
        fn literal_block(range: core::ops::Range<usize>) -> near_optimal::PlannedBlock {
            near_optimal::PlannedBlock {
                tokens: vec![near_optimal::ParseToken::Literal(b'a'); range.len()],
                range,
            }
        }

        let mut raw = z_stream::default();
        assert_eq!(
            init_with_lit_bufsize(
                &mut raw,
                DeflateConfig::new(9),
                Some(near_optimal::NEAR_OPTIMAL_LIT_BUFSIZE),
            ),
            ReturnCode::Ok
        );
        let stream = unsafe { DeflateStream::from_stream_mut(&raw mut raw) }.unwrap();

        let half = near_optimal::MAX_BLOCK_TOKENS / 2;
        let mut left = literal_block(0..half);
        let mut right = literal_block(half..near_optimal::MAX_BLOCK_TOKENS);
        let outcome =
            near_optimal::optimize_token_partition(&mut stream.state, &mut left, &mut right);
        let near_optimal::BoundaryOptimization::Merged(merged) = outcome else {
            panic!("the exact 65,534-token Huffman capacity should remain mergeable");
        };
        assert_eq!(merged.tokens.len(), near_optimal::MAX_BLOCK_TOKENS);

        let mut left = literal_block(0..half);
        let mut right = literal_block(half..near_optimal::MAX_BLOCK_TOKENS + 1);
        let outcome =
            near_optimal::optimize_token_partition(&mut stream.state, &mut left, &mut right);
        assert!(!matches!(
            outcome,
            near_optimal::BoundaryOptimization::Merged(_)
        ));
        assert_eq!(
            left.tokens.len() + right.tokens.len(),
            near_optimal::MAX_BLOCK_TOKENS + 1
        );
        assert!(end(stream).is_ok());
    }

    #[test]
    fn near_optimal_emits_exact_huffman_token_capacity() {
        let input = vec![b'a'; near_optimal::MAX_BLOCK_TOKENS];
        let block = near_optimal::PlannedBlock {
            range: 0..input.len(),
            tokens: vec![near_optimal::ParseToken::Literal(b'a'); input.len()],
        };
        let mut output = vec![0u8; compress_bound(input.len())];
        let mut raw = z_stream::default();
        assert_eq!(
            init_with_lit_bufsize(
                &mut raw,
                DeflateConfig::new(9),
                Some(near_optimal::NEAR_OPTIMAL_LIT_BUFSIZE),
            ),
            ReturnCode::Ok
        );

        let produced = {
            let stream = unsafe { DeflateStream::from_stream_mut(&raw mut raw) }.unwrap();
            let state = &mut *stream.state;
            state
                .bit_writer
                .pending
                .extend(&state.header().to_be_bytes());
            let mut produced = 0usize;
            assert!(drain_near_optimal_pending(
                state,
                &mut output,
                &mut produced
            ));
            assert!(emit_planned_near_optimal_block(
                state,
                &input,
                &block,
                true,
                &mut output,
                &mut produced,
            ));
            state
                .bit_writer
                .pending
                .extend(&adler32(ADLER32_INITIAL_VALUE, &input).to_be_bytes());
            assert!(drain_near_optimal_pending(
                state,
                &mut output,
                &mut produced
            ));
            produced
        };

        let stream = unsafe { DeflateStream::from_stream_mut(&raw mut raw) }.unwrap();
        assert!(end(stream).is_ok());
        let mut decoded = vec![0u8; input.len()];
        let (decoded, code) =
            decompress_slice(&mut decoded, &output[..produced], InflateConfig::default());
        assert_eq!(code, ReturnCode::Ok);
        assert_eq!(decoded, input);
    }

    #[test]
    fn init_allocation_geometry_matches_config() {
        for (window_bits, mem_level) in [(MIN_WBITS, 1), (MAX_WBITS, MAX_MEM_LEVEL)] {
            let mut raw = z_stream::default();
            let config = DeflateConfig {
                window_bits,
                mem_level,
                ..Default::default()
            };
            assert_eq!(init(&mut raw, config), ReturnCode::Ok);
            let stream = unsafe { DeflateStream::from_stream_mut(&raw mut raw) }.unwrap();

            let effective_window_bits = if window_bits == 8 { 9 } else { window_bits };
            let w_size = 1usize << effective_window_bits;
            let lit_bufsize = 1usize << (mem_level + 6);
            let layout = DeflateAllocOffsets::new(effective_window_bits as usize, lit_bufsize);

            assert_eq!(stream.state.w_size, w_size);
            assert_eq!(stream.state.window.capacity(), 2 * w_size);
            assert_eq!(stream.state.prev.len(), w_size);
            assert_eq!(stream.state.head.as_slice().len(), HASH_SIZE);
            assert_eq!(
                stream.state.bit_writer.pending.capacity(),
                lit_bufsize * LIT_BUFS
            );
            assert!(stream.state.sym_buf.is_empty());
            assert_eq!(stream.state.total_allocation_size, layout.total_size);
            let logical_end = layout.sym_buf_pos + lit_bufsize * (LIT_BUFS - 1);
            assert_eq!(
                layout.total_size,
                logical_end.next_multiple_of(DEFLATE_ALLOC_ALIGN),
                "typed allocation has no redundant leading alignment slack"
            );
            assert_eq!(
                stream.state.as_ptr() as usize % DEFLATE_ALLOC_ALIGN,
                0,
                "State placement must retain its cache-line alignment"
            );

            assert!(end(stream).is_ok());
        }
    }

    /// A valid zalloc callback is not required to return cache-line-aligned
    /// memory. Cover every possible remainder and both initialization/copy.
    #[test]
    fn custom_unaligned_allocator_preserves_state_layout_and_copy() -> Result<(), &'static str> {
        for remainder in 0..DEFLATE_ALLOC_ALIGN {
            let mut ctx = MisalignedAlloc {
                remainder,
                modulus: ALLOCATION_BLOCK_BYTES,
                allocations: 0,
                frees: 0,
                buffers: Vec::new(),
            };
            let mut raw = z_stream {
                zalloc: Some(misaligned_alloc_callback),
                zfree: Some(misaligned_free_callback),
                opaque: (&raw mut ctx).cast(),
                ..z_stream::default()
            };
            let config = DeflateConfig {
                window_bits: if remainder % 2 == 0 {
                    MIN_WBITS
                } else {
                    MAX_WBITS
                },
                mem_level: if remainder % 2 == 0 { 1 } else { MAX_MEM_LEVEL },
                ..DeflateConfig::default()
            };
            assert_eq!(init(&mut raw, config), ReturnCode::Ok);
            // SAFETY: successful initialization owns one state and its allocation.
            let source = (unsafe { DeflateStream::from_stream_mut(&raw mut raw) })
                .ok_or("state missing after initialization")?;
            assert_eq!(source.state.as_ptr() as usize % DEFLATE_ALLOC_ALIGN, 0);
            assert_eq!(
                source.state.window.as_ptr() as usize % DEFLATE_ALLOC_ALIGN,
                0
            );
            assert_eq!(source.state.prev.as_ptr() as usize % DEFLATE_ALLOC_ALIGN, 0);
            assert_eq!(source.state.head.as_ptr() as usize % DEFLATE_ALLOC_ALIGN, 0);
            source.state.prev.as_mut_slice()[0] = 0x1234;
            source.state.head.as_mut_slice()[0] = 0x4321;
            source.state.bit_writer.pending.extend(&[11, 22, 33]);

            let mut dest = MaybeUninit::<DeflateStream>::uninit();
            assert_eq!(copy(&mut dest, source), ReturnCode::Ok);
            // SAFETY: successful deflate copy initializes a complete independent state.
            let cloned = unsafe { dest.assume_init_mut() };
            assert_eq!(cloned.state.as_ptr() as usize % DEFLATE_ALLOC_ALIGN, 0);
            assert_eq!(
                cloned.state.window.as_ptr() as usize % DEFLATE_ALLOC_ALIGN,
                0
            );
            assert_eq!(cloned.state.prev.as_ptr() as usize % DEFLATE_ALLOC_ALIGN, 0);
            assert_eq!(cloned.state.head.as_ptr() as usize % DEFLATE_ALLOC_ALIGN, 0);
            assert_ne!(cloned.state.as_ptr(), source.state.as_ptr());
            assert_ne!(cloned.state.window.as_ptr(), source.state.window.as_ptr());
            assert_eq!(cloned.state.prev.as_slice()[0], 0x1234);
            assert_eq!(cloned.state.head.as_slice()[0], 0x4321);
            assert_eq!(cloned.state.bit_writer.pending.pending(), &[11, 22, 33]);
            source.state.prev.as_mut_slice()[0] = 0xabcd;
            assert_eq!(cloned.state.prev.as_slice()[0], 0x1234);
            assert!(end(source).is_ok());
            assert!(end(cloned).is_ok());
            assert_eq!(
                ctx.allocations, 2,
                "allocation count at remainder {remainder}"
            );
            assert_eq!(ctx.frees, 2, "free count at remainder {remainder}");
            assert!(
                ctx.buffers.is_empty(),
                "leaked buffers at remainder {remainder}"
            );
        }
        Ok(())
    }

    #[test]
    fn set_dictionary_keeps_abi_input_cursor_detached() {
        let mut raw = z_stream::default();
        assert_eq!(init(&mut raw, DeflateConfig::default()), ReturnCode::Ok);
        let stream = unsafe { DeflateStream::from_stream_mut(&raw mut raw) }.unwrap();

        // A caller may have already populated the next input cursor even though no bytes have been
        // consumed yet. Dictionary setup must not temporarily publish its own slice through these
        // ABI fields or disturb the caller's cursor.
        let caller_input = b"future caller input";
        stream.next_in = caller_input.as_ptr().cast_mut();
        stream.avail_in = caller_input.len() as _;
        let original_next = stream.next_in;
        let original_avail = stream.avail_in;
        let total_before = stream.total_in;

        let retained = stream.state.w_size;
        let dictionary = vec![0x5a; retained + 1];
        assert_eq!(set_dictionary(stream, &dictionary), ReturnCode::Ok);

        assert_eq!(stream.next_in, original_next);
        assert_eq!(stream.avail_in, original_avail);
        assert_eq!(
            stream.total_in,
            total_before + retained as crate::c_api::z_size,
            "only the logical-window tail is ingested into compression history"
        );
        assert_eq!(
            stream.adler as u32,
            adler32(ADLER32_INITIAL_VALUE, &dictionary),
            "zlib wrapper dictionary id still covers the entire supplied dictionary"
        );

        assert!(end(stream).is_ok());
    }

    #[test]
    #[cfg(feature = "c-allocator")]
    fn copy_invalid_allocator() {
        let mut stream = z_stream::default();

        let atomic = AtomicUsize::new(0);
        stream.opaque = (&raw const atomic).cast::<core::ffi::c_void>().cast_mut();
        stream.zalloc = Some(fail_nth_allocation::<1>);
        stream.zfree = Some(crate::allocate::C.zfree);

        // Init consumes the first state allocation; fail the second allocation used by copy.
        assert_eq!(init(&mut stream, DeflateConfig::default()), ReturnCode::Ok);

        let Some(stream) = (unsafe { DeflateStream::from_stream_mut(&raw mut stream) }) else {
            unreachable!()
        };

        let mut stream_copy = MaybeUninit::<DeflateStream>::zeroed();

        assert_eq!(copy(&mut stream_copy, stream), ReturnCode::MemError);

        assert!(end(stream).is_ok());
    }

    mod invalid_deflate_config {
        use super::*;

        #[test]
        fn extreme_signed_parameters_reject_without_allocated_state() {
            let invalid = [
                DeflateConfig {
                    level: i32::MIN,
                    ..DeflateConfig::default()
                },
                DeflateConfig {
                    level: i32::MAX,
                    ..DeflateConfig::default()
                },
                DeflateConfig {
                    window_bits: i32::MIN,
                    ..DeflateConfig::default()
                },
                DeflateConfig {
                    window_bits: i32::MAX,
                    ..DeflateConfig::default()
                },
                DeflateConfig {
                    mem_level: i32::MIN,
                    ..DeflateConfig::default()
                },
                DeflateConfig {
                    mem_level: i32::MAX,
                    ..DeflateConfig::default()
                },
            ];
            for config in invalid {
                let mut stream = z_stream::default();
                assert_eq!(init(&mut stream, config), ReturnCode::StreamError);
                assert!(stream.state.is_null(), "invalid config allocated a state");
            }
        }

        #[test]
        fn sanity_check() {
            let mut stream = z_stream::default();
            assert_eq!(init(&mut stream, DeflateConfig::default()), ReturnCode::Ok);

            assert!(stream.zalloc.is_some());
            assert!(stream.zfree.is_some());

            // this should be the default level
            let stream = unsafe { DeflateStream::from_stream_mut(&raw mut stream) }.unwrap();
            assert_eq!(stream.state.level, 6);

            assert!(end(stream).is_ok());
        }

        #[test]
        fn window_bits_correction() {
            // window_bits of 8 gets turned into 9 internally
            let mut stream = z_stream::default();
            let config = DeflateConfig {
                window_bits: 8,
                ..Default::default()
            };
            assert_eq!(init(&mut stream, config), ReturnCode::Ok);
            let stream = unsafe { DeflateStream::from_stream_mut(&raw mut stream) }.unwrap();
            assert_eq!(stream.state.w_bits(), 9);

            assert!(end(stream).is_ok());
        }

        #[test]
        fn window_bits_too_low() {
            let mut stream = z_stream::default();
            let config = DeflateConfig {
                window_bits: -16,
                ..Default::default()
            };
            assert_eq!(init(&mut stream, config), ReturnCode::StreamError);
        }

        #[test]
        fn window_bits_too_high() {
            // window bits too high
            let mut stream = z_stream::default();
            let config = DeflateConfig {
                window_bits: 42,
                ..Default::default()
            };
            assert_eq!(init(&mut stream, config), ReturnCode::StreamError);
        }
    }

    #[test]
    fn end_leaves_null_abi_state_pointer() {
        let mut raw = z_stream::default();
        assert_eq!(init(&mut raw, DeflateConfig::default()), ReturnCode::Ok);
        let stream = unsafe { DeflateStream::from_stream_mut(&raw mut raw) }.unwrap();

        assert!(end(stream).is_ok());
        assert!(stream.as_z_stream_mut().state.is_null());
        assert!(end(stream).is_err());
        assert!(stream.as_z_stream_mut().state.is_null());
    }

    #[test]
    fn end_data_error() {
        let mut stream = z_stream::default();
        assert_eq!(init(&mut stream, DeflateConfig::default()), ReturnCode::Ok);
        let stream = unsafe { DeflateStream::from_stream_mut(&raw mut stream) }.unwrap();

        // next deflate into too little space
        let input = b"Hello World\n";
        stream.next_in = input.as_ptr().cast_mut();
        stream.avail_in = input.len() as _;
        let output = &mut [0, 0, 0];
        stream.next_out = output.as_mut_ptr();
        stream.avail_out = output.len() as _;

        // the deflate is fine
        assert_eq!(deflate(stream, DeflateFlush::NoFlush), ReturnCode::Ok);

        // but end is not
        assert!(end(stream).is_err());
    }

    #[test]
    fn test_reset_keep() {
        let mut stream = z_stream::default();
        assert_eq!(init(&mut stream, DeflateConfig::default()), ReturnCode::Ok);
        let stream = unsafe { DeflateStream::from_stream_mut(&raw mut stream) }.unwrap();

        // next deflate into too little space
        let input = b"Hello World\n";
        stream.next_in = input.as_ptr().cast_mut();
        stream.avail_in = input.len() as _;

        let output = &mut [0; 1024];
        stream.next_out = output.as_mut_ptr();
        stream.avail_out = output.len() as _;
        assert_eq!(deflate(stream, DeflateFlush::Finish), ReturnCode::StreamEnd);

        assert_eq!(reset_keep(stream), ReturnCode::Ok);

        let output = &mut [0; 1024];
        stream.next_out = output.as_mut_ptr();
        stream.avail_out = output.len() as _;
        assert_eq!(deflate(stream, DeflateFlush::Finish), ReturnCode::StreamEnd);

        assert!(end(stream).is_ok());
    }

    #[test]
    fn reset_keep_rewinds_partially_drained_pending_output() {
        let mut raw = z_stream::default();
        let config = DeflateConfig {
            window_bits: 31,
            mem_level: 1,
            ..Default::default()
        };
        assert_eq!(init(&mut raw, config), ReturnCode::Ok);
        let stream = unsafe { DeflateStream::from_stream_mut(&raw mut raw) }.unwrap();

        // Make the gzip name larger than the small mem_level=1 pending buffer. One deflate call
        // fills that buffer and drains only part of it, leaving `Pending::out` advanced while
        // bytes are still pending.
        let mut name = vec![b'n'; 600];
        name.push(0);
        let mut header = gz_header {
            text: 0,
            time: 0,
            xflags: 0,
            os: 3,
            extra: core::ptr::null_mut(),
            extra_len: 0,
            extra_max: 0,
            name: name.as_mut_ptr(),
            name_max: 0,
            comment: core::ptr::null_mut(),
            comm_max: 0,
            hcrc: 0,
            done: 0,
        };
        assert_eq!(
            unsafe { set_header(stream, Some(&mut header)) },
            ReturnCode::Ok
        );

        let mut output = [0u8; 400];
        stream.next_in = core::ptr::null_mut();
        stream.avail_in = 0;
        stream.next_out = output.as_mut_ptr();
        stream.avail_out = output.len() as _;
        assert_eq!(deflate(stream, DeflateFlush::NoFlush), ReturnCode::Ok);

        let capacity = stream.state.bit_writer.pending.capacity();
        assert!(!stream.state.bit_writer.pending.is_empty());
        assert!(stream.state.bit_writer.pending.remaining() < capacity);

        assert_eq!(reset_keep(stream), ReturnCode::Ok);
        assert!(stream.state.bit_writer.pending.is_empty());
        assert_eq!(
            stream.state.bit_writer.pending.remaining(),
            capacity,
            "deflateResetKeep must rewind pending_out to pending_buf"
        );

        assert!(end(stream).is_ok());
    }

    #[test]
    fn prime_invalid_width_preserves_existing_bit_reservoir() -> Result<(), &'static str> {
        let mut raw = z_stream::default();
        assert_eq!(init(&mut raw, DeflateConfig::default()), ReturnCode::Ok);
        // SAFETY: init created the live uniquely owned Deflate state.
        let stream = unsafe { DeflateStream::from_stream_mut(&raw mut raw) }
            .ok_or("initialized stream has no state")?;
        assert_eq!(prime(stream, 3, 0b101), ReturnCode::Ok);

        let writer = &stream.state.bit_writer;
        let original = (
            writer.bit_buffer,
            writer.bits_valid,
            writer.pending.pending().to_vec(),
        );
        for invalid_bits in [-1, 33, i32::MAX] {
            assert_eq!(prime(stream, invalid_bits, -1), ReturnCode::BufError);
            let writer = &stream.state.bit_writer;
            assert_eq!(
                (
                    writer.bit_buffer,
                    writer.bits_valid,
                    writer.pending.pending().to_vec()
                ),
                original
            );
        }
        assert_eq!(prime(stream, 0, -1), ReturnCode::Ok);
        assert!(end(stream).is_ok());
        Ok(())
    }

    #[test]
    fn prime_reports_full_pending_buffer() {
        let mut raw = z_stream::default();
        let config = DeflateConfig {
            window_bits: -15,
            mem_level: 1,
            ..Default::default()
        };
        assert_eq!(init(&mut raw, config), ReturnCode::Ok);
        let stream = unsafe { DeflateStream::from_stream_mut(&raw mut raw) }.unwrap();

        let calls = stream.state.bit_writer.pending.capacity() / 2;
        for _ in 0..calls {
            assert_eq!(prime(stream, 16, 0xa55a), ReturnCode::Ok);
        }
        assert_eq!(stream.state.bit_writer.pending.remaining(), 0);
        assert_eq!(prime(stream, 16, 0xa55a), ReturnCode::BufError);

        assert!(end(stream).is_ok());
    }

    #[test]
    fn uninitialized_output_exposes_exactly_the_produced_prefix() {
        let input = b"MaybeUninit output should expose only bytes actually written by deflate";
        let config = DeflateConfig::default();
        let capacity = compress_bound(input.len());

        let mut initialized = vec![0u8; capacity];
        let (expected, expected_code) = compress_slice(&mut initialized, input, config);
        assert_eq!(expected_code, ReturnCode::Ok);
        let expected = expected.to_vec();

        let mut uninitialized = vec![MaybeUninit::<u8>::uninit(); capacity];
        let (actual, actual_code) = compress(&mut uninitialized, input, config);
        assert_eq!(actual_code, ReturnCode::Ok);
        assert_eq!(actual, expected);
        assert_eq!(actual.len(), expected.len());
        assert!(actual.len() < capacity);
    }

    #[test]
    fn hello_world_huffman_only() {
        const EXPECTED: &[u8] = &[
            0x78, 0x01, 0xf3, 0x48, 0xcd, 0xc9, 0xc9, 0x57, 0x08, 0xcf, 0x2f, 0xca, 0x49, 0x51,
            0xe4, 0x02, 0x00, 0x20, 0x91, 0x04, 0x48,
        ];

        let input = "Hello World!\n";

        let mut output = vec![0; 128];

        let config = DeflateConfig {
            level: 6,
            method: Method::Deflated,
            window_bits: crate::MAX_WBITS,
            mem_level: DEF_MEM_LEVEL,
            strategy: Strategy::HuffmanOnly,
        };

        let (output, err) = compress_slice(&mut output, input.as_bytes(), config);

        assert_eq!(err, ReturnCode::Ok);

        assert_eq!(output.len(), EXPECTED.len());

        assert_eq!(EXPECTED, output);
    }

    #[test]
    fn hello_world_quick() {
        const EXPECTED: &[u8] = &[
            0x78, 0x01, 0xf3, 0x48, 0xcd, 0xc9, 0xc9, 0x57, 0x08, 0xcf, 0x2f, 0xca, 0x49, 0x51,
            0xe4, 0x02, 0x00, 0x20, 0x91, 0x04, 0x48,
        ];

        let input = "Hello World!\n";

        let mut output = vec![0; 128];

        let config = DeflateConfig {
            level: 1,
            method: Method::Deflated,
            window_bits: crate::MAX_WBITS,
            mem_level: DEF_MEM_LEVEL,
            strategy: Strategy::Default,
        };

        let (output, err) = compress_slice(&mut output, input.as_bytes(), config);

        assert_eq!(err, ReturnCode::Ok);

        assert_eq!(output.len(), EXPECTED.len());

        assert_eq!(EXPECTED, output);
    }

    #[test]
    fn hello_world_quick_random() {
        const EXPECTED: &[u8] = &[
            0x78, 0x01, 0x53, 0xe1, 0x50, 0x51, 0xe1, 0x52, 0x51, 0x51, 0x01, 0x00, 0x03, 0xec,
            0x00, 0xeb,
        ];

        let input = "$\u{8}$$\n$$$";

        let mut output = vec![0; 128];

        let config = DeflateConfig {
            level: 1,
            method: Method::Deflated,
            window_bits: crate::MAX_WBITS,
            mem_level: DEF_MEM_LEVEL,
            strategy: Strategy::Default,
        };

        let (output, err) = compress_slice(&mut output, input.as_bytes(), config);

        assert_eq!(err, ReturnCode::Ok);

        assert_eq!(output.len(), EXPECTED.len());

        assert_eq!(EXPECTED, output);
    }

    fn compress_with_input_chunks(
        input: &[u8],
        config: DeflateConfig,
        chunk_size: usize,
    ) -> Vec<u8> {
        let mut raw = z_stream::default();
        assert_eq!(init(&mut raw, config), ReturnCode::Ok);
        let stream = unsafe { DeflateStream::from_stream_mut(&raw mut raw) }.unwrap();

        let mut output = vec![0u8; compress_bound(input.len()) + 64];
        stream.next_out = output.as_mut_ptr();
        stream.avail_out = output.len() as _;

        for chunk in input.chunks(chunk_size) {
            stream.next_in = chunk.as_ptr().cast_mut();
            stream.avail_in = chunk.len() as _;
            while stream.avail_in != 0 {
                assert_eq!(deflate(stream, DeflateFlush::NoFlush), ReturnCode::Ok);
                assert_ne!(
                    stream.avail_out, 0,
                    "compress_bound-sized output unexpectedly filled"
                );
            }
        }

        stream.next_in = core::ptr::null_mut();
        stream.avail_in = 0;
        loop {
            match deflate(stream, DeflateFlush::Finish) {
                ReturnCode::Ok => assert_ne!(
                    stream.avail_out, 0,
                    "compress_bound-sized output unexpectedly filled while finishing"
                ),
                ReturnCode::StreamEnd => break,
                other => panic!("input-chunked deflate failed: {other:?}"),
            }
        }

        output.truncate(stream.total_out as usize);
        assert!(end(stream).is_ok());
        output
    }

    fn compress_with_output_chunks(
        input: &[u8],
        config: DeflateConfig,
        chunk_size: usize,
    ) -> Vec<u8> {
        let mut raw = z_stream::default();
        assert_eq!(init(&mut raw, config), ReturnCode::Ok);
        let stream = unsafe { DeflateStream::from_stream_mut(&raw mut raw) }.unwrap();
        stream.next_in = input.as_ptr().cast_mut();
        stream.avail_in = input.len() as _;

        let mut output = Vec::new();
        loop {
            let mut chunk = vec![0u8; chunk_size];
            stream.next_out = chunk.as_mut_ptr();
            stream.avail_out = chunk.len() as _;
            let before = stream.total_out;
            let rc = deflate(stream, DeflateFlush::Finish);
            let produced = (stream.total_out - before) as usize;
            output.extend_from_slice(&chunk[..produced]);

            match rc {
                ReturnCode::Ok => {}
                ReturnCode::StreamEnd => break,
                other => panic!("chunked deflate failed: {other:?}"),
            }
        }

        assert_eq!(stream.avail_in, 0);
        assert!(end(stream).is_ok());
        output
    }

    struct RoundRobinCore {
        raw: z_stream,
        output: Vec<u8>,
        driver: algorithm::StandardDriver,
        done: bool,
        drain_turns: usize,
        output_full_count: usize,
        pending_match_result: Option<crate::deflate::longest_match::MatchSearchResult>,
        match_search_turns: usize,
        match_result_turns: usize,
    }

    impl RoundRobinCore {
        const OUTPUT_CHUNK: usize = 17;

        fn new(input: &[u8], config: DeflateConfig, algorithm: algorithm::Algorithm) -> Self {
            let mut raw = z_stream::default();
            assert_eq!(init(&mut raw, config), ReturnCode::Ok);

            let driver = {
                let stream = unsafe { DeflateStream::from_stream_mut(&raw mut raw) }.unwrap();
                stream.next_in = input.as_ptr().cast_mut();
                stream.avail_in = input.len() as _;
                algorithm::StandardDriver::for_algorithm(algorithm, &stream.state).unwrap()
            };

            Self {
                raw,
                output: Vec::new(),
                driver,
                done: false,
                drain_turns: 0,
                output_full_count: 0,
                pending_match_result: None,
                match_search_turns: 0,
                match_result_turns: 0,
            }
        }

        /// Perform exactly one production scheduler turn using the same driver as the zlib path.
        fn step(
            &mut self,
        ) -> Result<
            Option<crate::deflate::longest_match::MatchSearchWork>,
            crate::deflate::longest_match::MatchSearchError,
        > {
            assert!(!self.done);
            let stream = unsafe { DeflateStream::from_stream_mut(&raw mut self.raw) }.unwrap();

            // Match computation and result delivery are distinct scheduler turns. This keeps the
            // test honest: a batch scheduler may collect work from several paused streams before
            // executing searches, then resume each producer later with its owned result.
            if let Some(result) = self.pending_match_result.take() {
                self.driver
                    .complete_match_search(&mut stream.state, result)?;
                self.match_result_turns += 1;
                return Ok(None);
            }

            let mut chunk = [0u8; Self::OUTPUT_CHUNK];
            stream.next_out = chunk.as_mut_ptr();
            stream.avail_out = chunk.len() as _;
            let before = stream.total_out;

            let mut input = InputCursor::from_stream(stream);
            let mut output = OutputCursor::from_stream(stream);
            let mut checksum = StreamChecksum::from_stream(stream);
            let outcome = self.driver.step(
                &mut stream.state,
                &mut input,
                &mut output,
                &mut checksum,
                algorithm::StandardSegmentMode::Final,
            );
            input.apply(stream);
            output.apply(stream);
            checksum.apply(stream);

            let produced = (stream.total_out - before) as usize;
            if produced != 0 {
                self.output.extend_from_slice(&chunk[..produced]);
                self.drain_turns += 1;
            }

            // Do not leave a raw cursor into the stack-local chunk between scheduler turns.
            stream.next_out = core::ptr::null_mut();
            stream.avail_out = 0;

            match outcome {
                algorithm::StandardTurnOutcome::Progress => {}
                algorithm::StandardTurnOutcome::MatchSearch(work) => {
                    self.match_search_turns += 1;
                    return Ok(Some(work));
                }
                algorithm::StandardTurnOutcome::MatchSearchError(error) => return Err(error),
                algorithm::StandardTurnOutcome::InvalidSegmentMode => {
                    return Err(
                        crate::deflate::longest_match::MatchSearchError::InvalidSegmentMode,
                    );
                }
                algorithm::StandardTurnOutcome::NeedMoreCallerInput => {
                    panic!("Final mode with all input supplied unexpectedly requested more input")
                }
                algorithm::StandardTurnOutcome::NeedMoreOutput { .. } => {
                    self.output_full_count += 1;
                }
                algorithm::StandardTurnOutcome::SegmentDone(algorithm::EndBlockKind::Final) => {
                    self.done = true;
                }
                algorithm::StandardTurnOutcome::SegmentDone(
                    algorithm::EndBlockKind::FlushBoundary,
                ) => panic!("final batch job ended as a non-final flush boundary"),
            }
            Ok(None)
        }

        fn match_request(
            &self,
            work: crate::deflate::longest_match::MatchSearchWork,
        ) -> Result<
            crate::deflate::longest_match::MatchSearchRequest<'_>,
            crate::deflate::longest_match::MatchSearchError,
        > {
            let stream = unsafe { DeflateStream::from_stream_ref(&raw const self.raw) }.unwrap();
            work.request(&stream.state)
        }

        fn into_output(mut self) -> Vec<u8> {
            assert!(self.done);
            let stream = unsafe { DeflateStream::from_stream_mut(&raw mut self.raw) }.unwrap();
            assert_eq!(stream.avail_in, 0);
            assert_eq!(stream.total_out as usize, self.output.len());
            assert!(end(stream).is_ok());
            self.output
        }
    }

    fn direct_algorithm_output(input: &[u8], config: DeflateConfig) -> Vec<u8> {
        let mut raw = z_stream::default();
        assert_eq!(init(&mut raw, config), ReturnCode::Ok);
        let mut output = vec![0u8; compress_bound(input.len()) + 64];

        let stream = unsafe { DeflateStream::from_stream_mut(&raw mut raw) }.unwrap();
        stream.next_in = input.as_ptr().cast_mut();
        stream.avail_in = input.len() as _;
        stream.next_out = output.as_mut_ptr();
        stream.avail_out = output.len() as _;

        assert!(matches!(
            with_compressor(stream, |compressor| {
                algorithm::run(compressor, DeflateFlush::Finish)
            }),
            BlockState::FinishDone
        ));
        let len = stream.total_out as usize;
        assert_eq!(stream.avail_in, 0);
        assert!(end(stream).is_ok());
        output.truncate(len);
        output
    }

    fn framed_standard_driver_output(
        input: &[u8],
        config: DeflateConfig,
    ) -> Result<Vec<u8>, crate::deflate::longest_match::MatchSearchError> {
        let mut raw = z_stream::default();
        assert_eq!(init(&mut raw, config), ReturnCode::Ok);
        let mut output = vec![0u8; compress_bound(input.len()) + 64];
        let stream = unsafe { DeflateStream::from_stream_mut(&raw mut raw) }.unwrap();

        // Use the real zlib framing/status path with no caller input attached. `reset_keep()`
        // initializes last_flush to zlib's -2 sentinel, so this emits any zlib/gzip header without
        // entering the compression algorithm.
        stream.next_in = core::ptr::null_mut();
        stream.avail_in = 0;
        stream.next_out = output.as_mut_ptr();
        stream.avail_out = output.len() as _;
        assert_eq!(deflate(stream, DeflateFlush::NoFlush), ReturnCode::Ok);
        assert!(stream.state.bit_writer.pending.is_empty());

        let algorithm = algorithm::Algorithm::for_stream(stream.state.level, stream.state.strategy)
            .ok_or(crate::deflate::longest_match::MatchSearchError::InvalidCompressionLevel)?;
        let mut driver = algorithm::StandardDriver::for_algorithm(algorithm, &stream.state)
            .expect("proof only covers the five standard token-producing algorithms");

        stream.next_in = input.as_ptr().cast_mut();
        stream.avail_in = input.len().try_into().expect("proof input fits zlib uInt");

        let mut turns = 0usize;
        loop {
            turns += 1;
            assert!(
                turns < 100_000,
                "framed StandardDriver failed to make progress"
            );

            let mut input_cursor = InputCursor::from_stream(stream);
            let mut output_cursor = OutputCursor::from_stream(stream);
            let mut checksum = StreamChecksum::from_stream(stream);
            let outcome = driver.step(
                &mut stream.state,
                &mut input_cursor,
                &mut output_cursor,
                &mut checksum,
                algorithm::StandardSegmentMode::Final,
            );
            input_cursor.apply(stream);
            output_cursor.apply(stream);
            checksum.apply(stream);

            match outcome {
                algorithm::StandardTurnOutcome::Progress => {}
                algorithm::StandardTurnOutcome::MatchSearch(work) => {
                    let request = work.request(&stream.state)?;
                    let mut results = [crate::deflate::longest_match::MatchSearchResult {
                        length: 0,
                        start: 0,
                    }];
                    crate::deflate::longest_match::execute_match_search_batch(
                        core::slice::from_ref(&request),
                        &mut results,
                    );
                    driver.complete_match_search(&mut stream.state, results[0])?;
                }
                algorithm::StandardTurnOutcome::MatchSearchError(error) => return Err(error),
                algorithm::StandardTurnOutcome::InvalidSegmentMode => {
                    return Err(
                        crate::deflate::longest_match::MatchSearchError::InvalidSegmentMode,
                    );
                }
                algorithm::StandardTurnOutcome::NeedMoreCallerInput => {
                    panic!("final proof supplied the complete caller input")
                }
                algorithm::StandardTurnOutcome::NeedMoreOutput { .. } => {
                    panic!("compress_bound-sized proof output unexpectedly filled")
                }
                algorithm::StandardTurnOutcome::SegmentDone(algorithm::EndBlockKind::Final) => {
                    break;
                }
                algorithm::StandardTurnOutcome::SegmentDone(
                    algorithm::EndBlockKind::FlushBoundary,
                ) => panic!("final proof ended at a non-final boundary"),
            }
        }

        assert_eq!(stream.avail_in, 0);
        assert!(stream.state.bit_writer.pending.is_empty());

        // The StandardDriver already encoded the final DEFLATE block. Mark only that fact in the
        // outer zlib status machine, then let the real deflate path emit the wrapper trailer.
        stream.state.status = Status::Finish;
        stream.next_in = core::ptr::null_mut();
        stream.avail_in = 0;
        assert_eq!(deflate(stream, DeflateFlush::Finish), ReturnCode::StreamEnd);

        let len = stream.total_out as usize;
        assert!(end(stream).is_ok());
        output.truncate(len);
        Ok(output)
    }

    #[test]
    fn standard_driver_middle_preserves_full_stream_framing(
    ) -> Result<(), crate::deflate::longest_match::MatchSearchError> {
        let mut input = Vec::with_capacity(120_000);
        for i in 0..3_000usize {
            input.extend_from_slice(b"framed-standard-driver/");
            input.extend_from_slice(&(i as u32).rotate_left(9).to_le_bytes());
            input.extend(core::iter::repeat_n(((i * 29) % 251) as u8, 3 + i % 31));
        }

        let algorithms = [
            (2, Strategy::Default),
            (6, Strategy::Default),
            (9, Strategy::Default),
            (6, Strategy::HuffmanOnly),
            (6, Strategy::Rle),
        ];
        let wrappers = [-crate::MAX_WBITS, crate::MAX_WBITS, 16 + crate::MAX_WBITS];

        for window_bits in wrappers {
            for (level, strategy) in algorithms {
                let config = DeflateConfig {
                    level,
                    method: Method::Deflated,
                    window_bits,
                    mem_level: 1,
                    strategy,
                };
                let mut reference_buf = vec![0u8; compress_bound(input.len()) + 64];
                let (reference, rc) = compress_slice(&mut reference_buf, &input, config);
                assert_eq!(rc, ReturnCode::Ok);

                let scheduled = framed_standard_driver_output(&input, config)?;
                assert_eq!(
                    scheduled, reference,
                    "framed StandardDriver changed output for level {level}, strategy {strategy:?}, window_bits {window_bits}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn standard_core_events_can_be_round_robin_scheduled_across_streams(
    ) -> Result<(), crate::deflate::longest_match::MatchSearchError> {
        let mut input = Vec::with_capacity(160_000);
        for i in 0..4_000usize {
            input.extend_from_slice(b"round-robin-standard-core/");
            input.extend_from_slice(&(i as u32).rotate_left(7).to_le_bytes());
            input.extend(core::iter::repeat_n(((i * 17) % 251) as u8, 5 + i % 37));
        }

        let cases = [
            (algorithm::Algorithm::Fast, 2, Strategy::Default),
            (algorithm::Algorithm::Medium, 3, Strategy::Default),
            (algorithm::Algorithm::Medium, 6, Strategy::Default),
            (algorithm::Algorithm::Slow, 7, Strategy::Default),
            (algorithm::Algorithm::Slow, 9, Strategy::Default),
            (algorithm::Algorithm::Huffman, 6, Strategy::HuffmanOnly),
            (algorithm::Algorithm::Rle, 6, Strategy::Rle),
        ];

        let mut jobs = Vec::new();
        let mut references = Vec::new();
        for (algorithm, level, strategy) in cases {
            let config = || DeflateConfig {
                level,
                method: Method::Deflated,
                window_bits: crate::MAX_WBITS,
                mem_level: 1,
                strategy,
            };
            references.push(direct_algorithm_output(&input, config()));
            jobs.push(RoundRobinCore::new(&input, config(), algorithm));
        }

        let mut scheduler_turns = 0usize;
        let mut match_batch_count = 0usize;
        let mut max_match_batch = 0usize;
        while jobs.iter().any(|job| !job.done) {
            let mut match_work = Vec::new();
            for (index, job) in jobs.iter_mut().enumerate() {
                if !job.done {
                    if let Some(work) = job.step()? {
                        match_work.push((index, work));
                    }
                    scheduler_turns += 1;
                }
            }

            if !match_work.is_empty() {
                let requests = match_work
                    .iter()
                    .map(|(index, work)| jobs[*index].match_request(*work))
                    .collect::<Result<Vec<_>, _>>()?;
                let mut results = vec![
                    crate::deflate::longest_match::MatchSearchResult {
                        length: 0,
                        start: 0,
                    };
                    requests.len()
                ];
                crate::deflate::longest_match::execute_match_search_batch(&requests, &mut results);
                drop(requests);

                match_batch_count += 1;
                max_match_batch = max_match_batch.max(results.len());
                for ((index, _), result) in match_work.into_iter().zip(results) {
                    assert!(jobs[index].pending_match_result.replace(result).is_none());
                }
            }

            assert!(
                scheduler_turns < 100_000,
                "round-robin scheduler failed to make progress"
            );
        }

        assert!(
            match_batch_count > 0,
            "scheduler must dispatch match-search batches"
        );
        assert!(
            max_match_batch >= 2,
            "scheduler regression must batch match searches from multiple paused streams"
        );

        assert_eq!(
            jobs[0].match_search_turns, 0,
            "Fast's tiny searches should stay local instead of paying a scheduler handoff"
        );
        assert_eq!(
            jobs[1].match_search_turns, 0,
            "low-chain Medium searches should stay local instead of paying a scheduler handoff"
        );
        assert_eq!(
            jobs[2].match_search_turns, 0,
            "level-6 Medium searches should stay local instead of paying a scheduler handoff"
        );
        assert!(
            jobs[3].match_search_turns > 0,
            "level-7 Slow must still yield detached standard-search work"
        );
        assert!(
            jobs[4].match_search_turns > 0,
            "level-9 Slow must still yield detached slow-search work"
        );

        for (job, reference) in jobs.into_iter().zip(references) {
            assert!(
                job.drain_turns > 1,
                "test must schedule multiple output drains"
            );
            assert!(
                job.output_full_count > 0,
                "tiny output chunks must exercise resumable OutputFull"
            );
            assert_eq!(
                job.match_search_turns, job.match_result_turns,
                "every yielded match search must resume on a later scheduler turn"
            );
            assert_eq!(job.into_output(), reference);
        }
        Ok(())
    }

    #[test]
    fn tuned_medium_long_chain_still_yields_detached_match_search(
    ) -> Result<(), crate::deflate::longest_match::MatchSearchError> {
        let input_bytes = b"tuned-medium-detached-search/".repeat(2_000);
        let config = DeflateConfig {
            level: 6,
            method: Method::Deflated,
            window_bits: crate::MAX_WBITS,
            mem_level: 1,
            strategy: Strategy::Default,
        };
        let mut raw = z_stream::default();
        assert_eq!(init(&mut raw, config), ReturnCode::Ok);

        let stream = unsafe { DeflateStream::from_stream_mut(&raw mut raw) }.unwrap();
        assert_eq!(tune(stream, 8, 16, 128, 129), ReturnCode::Ok);
        stream.next_in = input_bytes.as_ptr().cast_mut();
        stream.avail_in = input_bytes.len() as _;
        let mut core =
            algorithm::StandardCore::for_algorithm(algorithm::Algorithm::Medium, &stream.state)
                .unwrap();

        let search = loop {
            match core.next_event(&mut stream.state) {
                algorithm::CoreEvent::NeedInput(demand) => {
                    let mut input = InputCursor::from_stream(stream);
                    let mut checksum = StreamChecksum::from_stream(stream);
                    let outcome = core.satisfy_input(
                        &mut stream.state,
                        &mut input,
                        &mut checksum,
                        demand,
                        algorithm::InputCompletion::EndOfSegment,
                    );
                    input.apply(stream);
                    checksum.apply(stream);
                    assert!(matches!(
                        outcome,
                        algorithm::RefillOutcome::Ready | algorithm::RefillOutcome::FinalTail
                    ));
                }
                algorithm::CoreEvent::MatchSearch(work) => break work,
                algorithm::CoreEvent::MatchSearchError(error) => return Err(error),
                algorithm::CoreEvent::TokenBlockReady(_) => {
                    panic!("tuned Medium filled a token block before yielding match-search work")
                }
                algorithm::CoreEvent::EndOfInput(_) => {
                    panic!("test input exhausted before tuned Medium yielded match-search work")
                }
            }
        };

        let result = search.request(&stream.state)?.execute();
        core.complete_match_search(&mut stream.state, result)?;
        let _ = end(stream);
        Ok(())
    }

    #[test]
    fn standard_core_rejects_production_before_match_search_completes(
    ) -> Result<(), crate::deflate::longest_match::MatchSearchError> {
        let input_bytes = b"detached-match-search/".repeat(2_000);
        let config = DeflateConfig {
            level: 7,
            method: Method::Deflated,
            window_bits: crate::MAX_WBITS,
            mem_level: 1,
            strategy: Strategy::Default,
        };
        let mut raw = z_stream::default();
        assert_eq!(init(&mut raw, config), ReturnCode::Ok);

        let stream = unsafe { DeflateStream::from_stream_mut(&raw mut raw) }.unwrap();
        stream.next_in = input_bytes.as_ptr().cast_mut();
        stream.avail_in = input_bytes.len() as _;
        let mut core =
            algorithm::StandardCore::for_algorithm(algorithm::Algorithm::Slow, &stream.state)
                .unwrap();

        let search = loop {
            match core.next_event(&mut stream.state) {
                algorithm::CoreEvent::NeedInput(demand) => {
                    let mut input = InputCursor::from_stream(stream);
                    let mut checksum = StreamChecksum::from_stream(stream);
                    let outcome = core.satisfy_input(
                        &mut stream.state,
                        &mut input,
                        &mut checksum,
                        demand,
                        algorithm::InputCompletion::EndOfSegment,
                    );
                    input.apply(stream);
                    checksum.apply(stream);
                    assert!(matches!(
                        outcome,
                        algorithm::RefillOutcome::Ready | algorithm::RefillOutcome::FinalTail
                    ));
                }
                algorithm::CoreEvent::MatchSearch(work) => break work,
                algorithm::CoreEvent::MatchSearchError(error) => return Err(error),
                algorithm::CoreEvent::TokenBlockReady(_) => {
                    panic!("Slow filled a token block before yielding match-search work")
                }
                algorithm::CoreEvent::EndOfInput(_) => {
                    panic!("test input exhausted before yielding match-search work")
                }
            }
        };

        let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = core.next_event(&mut stream.state);
        }));
        assert!(
            rejected.is_err(),
            "production advanced with an outstanding match search"
        );

        let result = search.request(&stream.state)?.execute();
        core.complete_match_search(&mut stream.state, result)?;
        assert!(matches!(
            core.next_event(&mut stream.state),
            algorithm::CoreEvent::MatchSearch(_)
                | algorithm::CoreEvent::TokenBlockReady(_)
                | algorithm::CoreEvent::NeedInput(_)
                | algorithm::CoreEvent::EndOfInput(_)
        ));

        let _ = end(stream);
        Ok(())
    }

    #[test]
    fn standard_core_rejects_production_before_pending_output_is_drained(
    ) -> Result<(), crate::deflate::longest_match::MatchSearchError> {
        let mut input_bytes = Vec::with_capacity(40_000);
        for i in 0..2_000usize {
            input_bytes.extend_from_slice(b"must-drain-before-producing/");
            input_bytes.extend_from_slice(&(i as u32).to_le_bytes());
        }

        let config = DeflateConfig {
            level: 2,
            method: Method::Deflated,
            window_bits: crate::MAX_WBITS,
            mem_level: 1,
            strategy: Strategy::Default,
        };
        let mut raw = z_stream::default();
        assert_eq!(init(&mut raw, config), ReturnCode::Ok);

        let stream = unsafe { DeflateStream::from_stream_mut(&raw mut raw) }.unwrap();
        stream.next_in = input_bytes.as_ptr().cast_mut();
        stream.avail_in = input_bytes.len() as _;
        let mut core =
            algorithm::StandardCore::for_algorithm(algorithm::Algorithm::Fast, &stream.state)
                .unwrap();

        let pending = loop {
            match core.next_event(&mut stream.state) {
                algorithm::CoreEvent::NeedInput(demand) => {
                    let mut input = InputCursor::from_stream(stream);
                    let mut checksum = StreamChecksum::from_stream(stream);
                    let outcome = core.satisfy_input(
                        &mut stream.state,
                        &mut input,
                        &mut checksum,
                        demand,
                        algorithm::InputCompletion::EndOfSegment,
                    );
                    input.apply(stream);
                    checksum.apply(stream);
                    assert!(matches!(
                        outcome,
                        algorithm::RefillOutcome::Ready | algorithm::RefillOutcome::FinalTail
                    ));
                }
                algorithm::CoreEvent::MatchSearch(work) => {
                    let result = work.request(&stream.state)?.execute();
                    core.complete_match_search(&mut stream.state, result)?;
                }
                algorithm::CoreEvent::MatchSearchError(error) => return Err(error),
                algorithm::CoreEvent::TokenBlockReady(work) => {
                    break core.encode_token_block(&mut stream.state, work);
                }
                algorithm::CoreEvent::EndOfInput(_) => {
                    panic!("test input exhausted before producing a token block")
                }
            }
        };

        let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = core.next_event(&mut stream.state);
        }));
        assert!(
            rejected.is_err(),
            "production advanced with undrained output"
        );

        let mut output_bytes = vec![0u8; compress_bound(input_bytes.len()) + 64];
        stream.next_out = output_bytes.as_mut_ptr();
        stream.avail_out = output_bytes.len() as _;
        let mut output = OutputCursor::from_stream(stream);
        assert!(matches!(
            core.drain_output(&mut stream.state, &mut output, pending),
            algorithm::PendingOutputOutcome::Ready
        ));
        output.apply(stream);

        // This test bypasses the outer zlib status machine; only ensure the allocation is released.
        let _ = end(stream);
        Ok(())
    }

    #[test]
    fn fast_deflate_is_input_chunking_independent() {
        let mut input = Vec::with_capacity(100_000);
        for i in 0..3_000usize {
            input.extend_from_slice(b"logical-window-tail/");
            input.extend(core::iter::repeat_n((i % 251) as u8, 1 + i % 19));
            input.extend_from_slice(&(i as u32).to_le_bytes());
        }

        let config = DeflateConfig {
            level: 2,
            method: Method::Deflated,
            window_bits: crate::MAX_WBITS,
            mem_level: 8,
            strategy: Strategy::Default,
        };
        let mut reference_buf = vec![0u8; compress_bound(input.len())];
        let (reference, rc) = compress_slice(&mut reference_buf, &input, config);
        assert_eq!(rc, ReturnCode::Ok);
        let reference = reference.to_vec();

        for chunk_size in [1, 2, 3, 4, 7, 31, 257] {
            assert_eq!(
                compress_with_input_chunks(&input, config, chunk_size),
                reference,
                "fast output changed with {chunk_size}-byte input chunks"
            );
        }
    }

    #[test]
    fn token_producers_are_output_chunking_independent_across_block_boundaries() {
        let mut input = Vec::with_capacity(160_000);
        for i in 0..4_000usize {
            input.extend_from_slice(b"portable-simd::token-block/");
            for _ in 0..(8 + i % 23) {
                input.push((i % 251) as u8);
            }
            input.extend_from_slice(&(i as u32).to_le_bytes());
        }

        let configurations = [
            (2, Strategy::Default),
            (6, Strategy::Default),
            (9, Strategy::Default),
            (6, Strategy::HuffmanOnly),
            (6, Strategy::Rle),
        ];

        for (level, strategy) in configurations {
            let config = DeflateConfig {
                level,
                method: Method::Deflated,
                window_bits: crate::MAX_WBITS,
                mem_level: 1,
                strategy,
            };
            let mut reference_buf = vec![0u8; compress_bound(input.len())];
            let (reference, rc) = compress_slice(&mut reference_buf, &input, config);
            assert_eq!(rc, ReturnCode::Ok);
            let reference = reference.to_vec();

            for chunk_size in [1, 7, 23, 257] {
                let chunked = compress_with_output_chunks(&input, config, chunk_size);
                assert_eq!(
                    chunked, reference,
                    "output changed with {chunk_size}-byte output chunks for level {level}, strategy {strategy:?}"
                );
            }
        }
    }

    fn fuzz_based_test(input: &[u8], config: DeflateConfig, expected: &[u8]) {
        let mut output_rs = [0; 1 << 17];
        let (output_rs, err) = compress_slice(&mut output_rs, input, config);
        assert_eq!(err, ReturnCode::Ok);

        assert_eq!(output_rs, expected);
    }

    #[test]
    fn simple_rle() {
        fuzz_based_test(
            "\0\0\0\0\u{6}".as_bytes(),
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
    fn reset_preserves_custom_gzip_header() {
        let name = b"survives-reset\0";
        let mut header = gz_header {
            text: 1,
            time: 0x1234_5678,
            xflags: 0,
            os: 3,
            extra: core::ptr::null_mut(),
            extra_len: 0,
            extra_max: 0,
            name: name.as_ptr().cast_mut(),
            name_max: 0,
            comment: core::ptr::null_mut(),
            comm_max: 0,
            hcrc: 1,
            done: 0,
        };

        let config = DeflateConfig {
            window_bits: 31,
            ..Default::default()
        };
        let mut stream = z_stream::default();
        assert_eq!(init(&mut stream, config), ReturnCode::Ok);
        let stream = unsafe { DeflateStream::from_stream_mut(&raw mut stream) }.unwrap();
        assert_eq!(
            unsafe { set_header(stream, Some(&mut header)) },
            ReturnCode::Ok
        );

        assert_eq!(reset(stream), ReturnCode::Ok);
        assert!(stream.state.gzhead.is_some());

        let input = b"gzip after reset";
        let mut output = [0u8; 128];
        stream.next_in = input.as_ptr().cast_mut();
        stream.avail_in = input.len() as _;
        stream.next_out = output.as_mut_ptr();
        stream.avail_out = output.len() as _;
        assert_eq!(deflate(stream, DeflateFlush::Finish), ReturnCode::StreamEnd);

        // zlib and zlib-ng retain the caller's custom header across both reset variants.
        assert_eq!(
            &output[..10],
            &[31, 139, 8, 0b1011, 0x78, 0x56, 0x34, 0x12, 0, 3]
        );
        assert_eq!(&output[10..10 + name.len()], name);

        assert!(end(stream).is_ok());
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

    fn encode_custom_gzip_header(output_chunk: usize) -> Result<Vec<u8>, &'static str> {
        // All three metadata fields exceed the mem-level-1 pending buffer.
        // Every state must therefore handle an interrupted header flush.
        let extra = vec![0x5au8; 701];
        let mut name = vec![b'n'; 709];
        name.push(0);
        // Make the gzip header length 511 modulo 512, so FHCRC itself
        // must also resume across an exhausted pending buffer.
        let mut comment = vec![b'c'; 1_135];
        comment.push(0);
        let mut header = gz_header {
            text: 1,
            time: 0x1020_3040,
            os: 3,
            extra: extra.as_ptr().cast_mut(),
            extra_len: crate::c_api::uInt::try_from(extra.len())
                .map_err(|_| "extra header fixture exceeds C length")?,
            name: name.as_mut_ptr(),
            comment: comment.as_mut_ptr(),
            hcrc: 1,
            ..gz_header::default()
        };

        let config = DeflateConfig {
            window_bits: 16 + crate::MAX_WBITS,
            mem_level: 1,
            ..DeflateConfig::default()
        };
        let mut stream =
            DeflateStream::new(config).map_err(|_| "valid gzip compressor configuration failed")?;
        // SAFETY: header and all three backing buffers remain live and
        // unmodified until deflateEnd, including every resumed call.
        let header_status = unsafe { set_header(&mut stream, Some(&mut header)) };
        assert_eq!(header_status, ReturnCode::Ok);
        let source = b"chunked gzip header then actual deflate payload".repeat(8);
        stream.next_in = source.as_ptr().cast_mut();
        stream.avail_in = crate::c_api::uInt::try_from(source.len())
            .map_err(|_| "source fixture exceeds C length")?;

        let limit = bound(Some(&stream), source.len());
        let mut encoded = vec![0u8; limit];
        let mut written = 0usize;
        let mut finished = false;
        let mut paused_header_crc = false;
        for _ in 0..limit.saturating_add(8) {
            let count = output_chunk.min(encoded.len() - written);
            if count == 0 {
                break;
            }
            stream.next_out = encoded[written..].as_mut_ptr();
            stream.avail_out = crate::c_api::uInt::try_from(count)
                .map_err(|_| "output fixture exceeds C length")?;
            let status = deflate(&mut stream, DeflateFlush::Finish);
            paused_header_crc |= stream.state.status == Status::Hcrc && stream.state.gzindex != 0;
            let remaining = usize::try_from(stream.avail_out)
                .map_err(|_| "output cursor exceeds host pointer width")?;
            written += count - remaining;
            if status == ReturnCode::StreamEnd {
                finished = true;
                break;
            }
            assert_eq!(
                status,
                ReturnCode::Ok,
                "gzip streaming failed at chunk {output_chunk}"
            );
        }
        assert!(
        finished,
        "gzip chunk {output_chunk} stuck: produced {written}/{limit}, status={:?}, gzindex={}, pending={}, avail_in={}, last_flush={}",
        stream.state.status,
        stream.state.gzindex,
        stream.state.bit_writer.pending.pending().len(),
        stream.avail_in,
        stream.state.last_flush
    );
        assert_eq!(stream.avail_in, 0);
        if output_chunk == 1 {
            assert!(
                paused_header_crc,
                "the one-byte output fixture must interrupt FHCRC"
            );
        }
        assert!(end(&mut stream).is_ok());
        encoded.truncate(written);
        assert!(encoded.starts_with(&[31, 139, 8, 0x1f]));
        Ok(encoded)
    }

    #[test]
    fn gzip_custom_header_is_identical_across_tiny_output_chunks() -> Result<(), &'static str> {
        let baseline = encode_custom_gzip_header(16_384)?;
        let source = b"chunked gzip header then actual deflate payload".repeat(8);
        let mut restored = vec![0u8; source.len()];
        let (decoded, status) = crate::decompress_slice(
            &mut restored,
            &baseline,
            crate::InflateConfig { window_bits: 31 },
        );
        assert_eq!(status, ReturnCode::Ok);
        assert_eq!(decoded, source);
        let hcrc_at = 10 + 2 + 701 + 710 + 1_136;
        let expected_crc = crc32(0, &baseline[..hcrc_at]).to_le_bytes();
        assert_eq!(
            &baseline[hcrc_at..hcrc_at + 2],
            &expected_crc[..2],
            "FHCRC must cover the header without including its own bytes"
        );
        for output_chunk in [1usize, 2, 3, 7, 31, 127, 511] {
            let actual = encode_custom_gzip_header(output_chunk)?;
            assert_eq!(actual, baseline, "gzip output chunk size {output_chunk}");
        }
        Ok(())
    }

    #[test]
    fn gzip_header_pending_flush() {
        let extra = "aaaaaaaaaaaaaaaaaaaa\0";
        let name = "bbbbbbbbbbbbbbbbbbbb\0";
        let comment = "cccccccccccccccccccc\0";

        let mut header = gz_header {
            text: 0,
            time: 0,
            xflags: 0,
            os: 0,
            extra: extra.as_ptr().cast_mut(),
            extra_len: extra.len() as _,
            extra_max: 0,
            name: name.as_ptr().cast_mut(),
            name_max: 0,
            comment: comment.as_ptr().cast_mut(),
            comm_max: 0,
            hcrc: 1,
            done: 0,
        };

        let config = DeflateConfig {
            window_bits: 31,
            mem_level: 1,
            ..Default::default()
        };

        let mut stream = z_stream::default();
        assert_eq!(init(&mut stream, config), ReturnCode::Ok);

        let Some(stream) = (unsafe { DeflateStream::from_stream_mut(&raw mut stream) }) else {
            unreachable!()
        };

        unsafe { set_header(stream, Some(&mut header)) };

        let input = b"Hello World\n";
        stream.next_in = input.as_ptr().cast_mut();
        stream.avail_in = input.len() as _;

        let mut output = [0u8; 1024];
        stream.next_out = output.as_mut_ptr();
        stream.avail_out = 100;

        assert_eq!(stream.state.bit_writer.pending.capacity(), 512);

        // only 12 bytes remain, so to write the name the pending buffer must be flushed.
        // but there is insufficient output space to flush (only 100 bytes)
        stream.state.bit_writer.pending.extend(&[0; 500]);

        assert_eq!(deflate(stream, DeflateFlush::Finish), ReturnCode::Ok);

        // now try that again but with sufficient output space
        stream.avail_out = output.len() as _;
        assert_eq!(deflate(stream, DeflateFlush::Finish), ReturnCode::StreamEnd);

        let n = stream.total_out as usize;

        assert!(end(stream).is_ok());

        let output_rs = &mut output[..n];

        assert_eq!(output_rs.len(), 500 + 99);
    }

    #[test]
    fn deflate_copy_owns_independent_state_and_buffer_storage() {
        let mut raw = z_stream::default();
        assert_eq!(init(&mut raw, DeflateConfig::default()), ReturnCode::Ok);
        let source = unsafe { DeflateStream::from_stream_mut(&raw mut raw) }.unwrap();

        let initial_window_byte = 0x11u8;
        unsafe {
            source
                .state
                .window
                .copy_and_initialize(0..1, &raw const initial_window_byte)
        };
        source.state.prev.as_mut_slice()[0] = 0x1234;
        source.state.head.as_mut_slice()[0] = 0x2345;
        source.state.bit_writer.pending.extend(&[1, 2, 3]);
        source.state.sym_buf.push_lit(0xa5);

        let source_state = source.state.as_ptr();
        let source_window = source.state.window.as_ptr();
        let source_prev = source.state.prev.as_ptr();
        let source_head = source.state.head.as_ptr();

        let mut copied = MaybeUninit::<DeflateStream>::uninit();
        assert_eq!(copy(&mut copied, source), ReturnCode::Ok);
        let copied = unsafe { copied.assume_init_mut() };

        assert_ne!(source_state, copied.state.as_ptr());
        assert_ne!(source_window, copied.state.window.as_ptr());
        assert_ne!(source_prev, copied.state.prev.as_ptr());
        assert_ne!(source_head, copied.state.head.as_ptr());
        assert_eq!(
            source.state.total_allocation_size,
            copied.state.total_allocation_size
        );

        source.state.window.filled_mut()[0] = 0x99;
        source.state.prev.as_mut_slice()[0] = 0x3456;
        source.state.head.as_mut_slice()[0] = 0x4567;
        source.state.bit_writer.pending.advance(3);
        source.state.bit_writer.pending.extend(&[9, 9, 9]);
        source.state.sym_buf.clear();

        assert_eq!(copied.state.window.filled()[0], 0x11);
        assert_eq!(copied.state.prev.as_slice()[0], 0x1234);
        assert_eq!(copied.state.head.as_slice()[0], 0x2345);
        assert_eq!(copied.state.bit_writer.pending.pending(), &[1, 2, 3]);
        {
            let mut copied_symbols = copied.state.sym_buf.iter();
            assert_eq!(copied_symbols.next(), Some((0, 0xa5)));
            assert_eq!(copied_symbols.next(), None);
        }

        assert!(end(source).is_ok());
        assert!(end(copied).is_ok());
    }

    #[test]
    fn gzip_header_is_shared_safely_across_deflate_copy() {
        let extra = b"copy-extra\0";
        let name = b"copy-name\0";
        let comment = b"copy-comment\0";
        let mut header = gz_header {
            text: 1,
            time: 0x1234_5678,
            xflags: 0,
            os: 3,
            extra: extra.as_ptr().cast_mut(),
            extra_len: extra.len() as _,
            extra_max: 0,
            name: name.as_ptr().cast_mut(),
            name_max: 0,
            comment: comment.as_ptr().cast_mut(),
            comm_max: 0,
            hcrc: 1,
            done: 0,
        };

        let config = DeflateConfig {
            window_bits: 31,
            ..Default::default()
        };
        let mut source_stream = z_stream::default();
        assert_eq!(init(&mut source_stream, config), ReturnCode::Ok);
        let source = unsafe { DeflateStream::from_stream_mut(&raw mut source_stream) }.unwrap();
        assert_eq!(
            unsafe { set_header(source, Some(&mut header)) },
            ReturnCode::Ok
        );

        // Copy before either stream starts emitting the caller-owned header. zlib semantics require
        // both compressors to retain the same header pointer.
        let mut copied = MaybeUninit::<DeflateStream>::uninit();
        assert_eq!(copy(&mut copied, source), ReturnCode::Ok);
        let copied = unsafe { copied.assume_init_mut() };

        let input = b"same payload through both copied streams";
        let mut source_output = [0u8; 256];
        source.next_in = input.as_ptr().cast_mut();
        source.avail_in = input.len() as _;
        source.next_out = source_output.as_mut_ptr();
        source.avail_out = source_output.len() as _;

        let mut copied_output = [0u8; 256];
        copied.next_in = input.as_ptr().cast_mut();
        copied.avail_in = input.len() as _;
        copied.next_out = copied_output.as_mut_ptr();
        copied.avail_out = copied_output.len() as _;

        assert_eq!(deflate(source, DeflateFlush::Finish), ReturnCode::StreamEnd);
        assert_eq!(deflate(copied, DeflateFlush::Finish), ReturnCode::StreamEnd);

        let source_len = source.total_out as usize;
        let copied_len = copied.total_out as usize;
        assert_eq!(source_len, copied_len);
        assert_eq!(&source_output[..source_len], &copied_output[..copied_len]);

        assert!(end(source).is_ok());
        assert!(end(copied).is_ok());
    }

    #[test]
    fn gzip_with_header() {
        // this test is here mostly so we get some MIRI action on the gzip header. A test that
        // compares behavior with zlib-ng is in the libz-rs-sys test suite

        let extra = "some extra stuff\0";
        let name = "nomen est omen\0";
        let comment = "such comment\0";

        let mut header = gz_header {
            text: 0,
            time: 0,
            xflags: 0,
            os: 0,
            extra: extra.as_ptr().cast_mut(),
            extra_len: extra.len() as _,
            extra_max: 0,
            name: name.as_ptr().cast_mut(),
            name_max: 0,
            comment: comment.as_ptr().cast_mut(),
            comm_max: 0,
            hcrc: 1,
            done: 0,
        };

        let config = DeflateConfig {
            window_bits: 31,
            ..Default::default()
        };

        let mut stream = z_stream::default();
        assert_eq!(init(&mut stream, config), ReturnCode::Ok);

        let Some(stream) = (unsafe { DeflateStream::from_stream_mut(&raw mut stream) }) else {
            unreachable!()
        };

        unsafe { set_header(stream, Some(&mut header)) };

        let input = b"Hello World\n";
        stream.next_in = input.as_ptr().cast_mut();
        stream.avail_in = input.len() as _;

        let mut output = [0u8; 256];
        stream.next_out = output.as_mut_ptr();
        stream.avail_out = output.len() as _;

        assert_eq!(deflate(stream, DeflateFlush::Finish), ReturnCode::StreamEnd);

        let n = stream.total_out as usize;

        assert!(end(stream).is_ok());

        let output_rs = &mut output[..n];

        assert_eq!(output_rs.len(), 81);

        {
            let mut stream = z_stream::default();

            let config = InflateConfig {
                window_bits: config.window_bits,
            };

            assert_eq!(crate::inflate::init(&mut stream, config), ReturnCode::Ok);

            let Some(stream) = (unsafe { InflateStream::from_stream_mut(&raw mut stream) }) else {
                unreachable!();
            };

            stream.next_in = output_rs.as_mut_ptr() as _;
            stream.avail_in = output_rs.len() as _;

            let mut output = [0u8; 12];
            stream.next_out = output.as_mut_ptr();
            stream.avail_out = output.len() as _;

            let mut extra_buf = [0u8; 64];
            let mut name_buf = [0u8; 64];
            let mut comment_buf = [0u8; 64];

            let mut header = gz_header {
                text: 0,
                time: 0,
                xflags: 0,
                os: 0,
                extra: extra_buf.as_mut_ptr(),
                extra_len: 0,
                extra_max: extra_buf.len() as _,
                name: name_buf.as_mut_ptr(),
                name_max: name_buf.len() as _,
                comment: comment_buf.as_mut_ptr(),
                comm_max: comment_buf.len() as _,
                hcrc: 0,
                done: 0,
            };

            assert_eq!(
                unsafe { crate::inflate::get_header(stream, Some(&mut header)) },
                ReturnCode::Ok
            );

            assert_eq!(
                unsafe { crate::inflate::inflate(stream, InflateFlush::Finish) },
                ReturnCode::StreamEnd
            );

            crate::inflate::end(stream);

            assert!(!header.comment.is_null());
            assert_eq!(
                unsafe { CStr::from_ptr(header.comment.cast()) }
                    .to_str()
                    .unwrap(),
                comment.trim_end_matches('\0')
            );

            assert!(!header.name.is_null());
            assert_eq!(
                unsafe { CStr::from_ptr(header.name.cast()) }
                    .to_str()
                    .unwrap(),
                name.trim_end_matches('\0')
            );

            assert!(!header.extra.is_null());
            assert_eq!(
                unsafe { CStr::from_ptr(header.extra.cast()) }
                    .to_str()
                    .unwrap(),
                extra.trim_end_matches('\0')
            );
        }
    }

    #[test]
    fn insufficient_compress_space() {
        const DATA: &[u8] = include_bytes!("deflate/test-data/inflate_buf_error.dat");

        fn helper(deflate_buf: &mut [u8], deflate_err: ReturnCode) -> ReturnCode {
            let config = DeflateConfig {
                level: 0,
                method: Method::Deflated,
                window_bits: 10,
                mem_level: 6,
                strategy: Strategy::Default,
            };

            let (output, err) = compress_slice(deflate_buf, DATA, config);
            assert_eq!(err, deflate_err);

            let config = InflateConfig {
                window_bits: config.window_bits,
            };

            let mut uncompr = [0; 1 << 17];
            let (uncompr, err) = decompress_slice(&mut uncompr, output, config);

            if err == ReturnCode::Ok {
                assert_eq!(DATA, uncompr);
            }

            err
        }

        let mut output = [0; 1 << 17];

        // this is too little space
        assert_eq!(
            helper(&mut output[..1 << 16], ReturnCode::BufError),
            ReturnCode::DataError
        );

        // this is sufficient space
        assert_eq!(helper(&mut output, ReturnCode::Ok), ReturnCode::Ok);
    }

    fn test_flush(flush: DeflateFlush, expected: &[u8]) {
        let input = b"Hello World!\n";

        let config = DeflateConfig {
            level: 6, // use gzip
            method: Method::Deflated,
            window_bits: 16 + crate::MAX_WBITS,
            mem_level: DEF_MEM_LEVEL,
            strategy: Strategy::Default,
        };

        let mut output_rs = vec![0; 128];

        // with the flush modes that we test here, the deflate process still has `Status::Busy`,
        // and the `deflate` function will return `BufError` because more input is needed before
        // the flush can occur.
        let expected_err = ReturnCode::BufError;

        let (rs, err) = compress_slice_with_flush(&mut output_rs, input, config, flush);
        assert_eq!(expected_err, err);

        assert_eq!(rs, expected);
    }

    #[test]
    #[rustfmt::skip]
    fn sync_flush() {
        test_flush(
            DeflateFlush::SyncFlush,
            &[
                31, 139, 8, 0, 0, 0, 0, 0, 0, gz_header::OS_CODE, 242, 72, 205, 201, 201, 87, 8, 207, 47, 202, 73,
                81, 228, 2, 0, 0, 0, 255, 255,
            ],
        )
    }

    #[test]
    #[rustfmt::skip]
    fn partial_flush() {
        test_flush(
            DeflateFlush::PartialFlush,
            &[
                31, 139, 8, 0, 0, 0, 0, 0, 0, gz_header::OS_CODE, 242, 72, 205, 201, 201, 87, 8, 207, 47, 202, 73,
                81, 228, 2, 8,
            ],
        );
    }

    #[test]
    #[rustfmt::skip]
    fn full_flush() {
        test_flush(
            DeflateFlush::FullFlush,
            &[
                31, 139, 8, 0, 0, 0, 0, 0, 0, gz_header::OS_CODE, 242, 72, 205, 201, 201, 87, 8, 207, 47, 202, 73,
                81, 228, 2, 0, 0, 0, 255, 255,
            ],
        );
    }

    #[test]
    #[rustfmt::skip]
    fn block_flush() {
        test_flush(
            DeflateFlush::Block,
            &[
                31, 139, 8, 0, 0, 0, 0, 0, 0, gz_header::OS_CODE, 242, 72, 205, 201, 201, 87, 8, 207, 47, 202, 73,
                81, 228, 2,
            ],
        );
    }

    #[test]
    // splits the input into two, deflates them seperately and then joins the deflated byte streams
    // into something that can be correctly inflated again. This is the basic idea behind pigz, and
    // allows for parallel compression.
    fn split_deflate() {
        let input = "Hello World!\n";

        let (input1, input2) = input.split_at(6);

        let mut output1 = vec![0; 128];
        let mut output2 = vec![0; 128];

        let config = DeflateConfig {
            level: 6, // use gzip
            method: Method::Deflated,
            window_bits: 16 + crate::MAX_WBITS,
            mem_level: DEF_MEM_LEVEL,
            strategy: Strategy::Default,
        };

        // see also the docs on `SyncFlush`. it makes sure everything is flushed, ends on a byte
        // boundary, and that the final block does not have the "last block" bit set.
        let (prefix, err) = compress_slice_with_flush(
            &mut output1,
            input1.as_bytes(),
            config,
            DeflateFlush::SyncFlush,
        );
        assert_eq!(err, ReturnCode::BufError);

        let (output2, err) = compress_slice_with_flush(
            &mut output2,
            input2.as_bytes(),
            config,
            DeflateFlush::Finish,
        );
        assert_eq!(err, ReturnCode::Ok);

        let inflate_config = crate::inflate::InflateConfig {
            window_bits: 16 + 15,
        };

        // cuts off the length and crc
        let (suffix, end) = output2.split_at(output2.len() - 8);
        let (crc2, len2) = end.split_at(4);
        let crc2 = u32::from_le_bytes(crc2.try_into().unwrap());

        // cuts off the gzip header (10 bytes) from the front
        let suffix = &suffix[10..];

        let mut result: Vec<u8> = Vec::new();
        result.extend(prefix.iter());
        result.extend(suffix);

        // it would be more proper to use `stream.total_in` here, but the slice helpers hide the
        // stream so we're cheating a bit here
        let len1 = input1.len() as u32;
        let len2 = u32::from_le_bytes(len2.try_into().unwrap());
        assert_eq!(len2 as usize, input2.len());

        let crc1 = crate::crc32::crc32(0, input1.as_bytes());
        let crc = crate::crc32::crc32_combine(crc1, crc2, u64::from(len2));

        // combined crc of the parts should be the crc of the whole
        let crc_cheating = crate::crc32::crc32(0, input.as_bytes());
        assert_eq!(crc, crc_cheating);

        // write the trailer
        result.extend(crc.to_le_bytes());
        result.extend((len1 + len2).to_le_bytes());

        let mut output = vec![0; 128];
        let (output, err) = crate::inflate::decompress_slice(&mut output, &result, inflate_config);
        assert_eq!(err, ReturnCode::Ok);

        assert_eq!(output, input.as_bytes());
    }

    #[test]
    fn inflate_window_copy_slice() {
        let uncompressed = [
            9, 126, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 76, 33, 8, 2, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12,
            10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 76, 33, 8, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 12, 10, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69,
            69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 14, 0, 0, 0, 0, 0, 69, 69, 69, 69, 69, 69, 69,
            69, 69, 69, 69, 69, 69, 69, 69, 9, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69,
            69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69,
            69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69,
            69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 12, 28, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 12, 10, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69,
            69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 14, 0, 0, 0, 0, 0, 69, 69, 69, 69, 69, 69, 69,
            69, 69, 69, 69, 69, 69, 69, 69, 9, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69,
            69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69,
            69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69,
            69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 12, 28, 0, 2, 0, 0, 0, 63, 1, 0, 12, 2,
            36, 0, 28, 0, 0, 0, 1, 0, 0, 63, 63, 13, 0, 0, 0, 0, 0, 0, 0, 63, 63, 63, 63, 0, 0, 0,
            0, 0, 0, 65, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 91, 0, 0, 0, 9, 0, 0, 0, 9, 0, 0, 12, 33, 2, 0, 0, 8,
            0, 4, 0, 0, 0, 12, 10, 41, 12, 10, 47,
        ];

        let compressed = &[
            31, 139, 8, 0, 0, 0, 0, 0, 4, 3, 181, 193, 49, 14, 194, 32, 24, 128, 209, 175, 192, 0,
            228, 151, 232, 206, 66, 226, 226, 96, 60, 2, 113, 96, 235, 13, 188, 139, 103, 23, 106,
            104, 108, 100, 49, 169, 239, 185, 39, 11, 199, 7, 51, 39, 171, 248, 118, 226, 63, 52,
            157, 120, 86, 102, 78, 86, 209, 104, 58, 241, 84, 129, 166, 12, 4, 154, 178, 229, 202,
            30, 36, 130, 166, 19, 79, 21, 104, 202, 64, 160, 41, 91, 174, 236, 65, 34, 10, 200, 19,
            162, 206, 68, 96, 130, 156, 15, 188, 229, 138, 197, 157, 161, 35, 3, 87, 126, 245, 0,
            28, 224, 64, 146, 2, 139, 1, 196, 95, 196, 223, 94, 10, 96, 92, 33, 86, 2, 0, 0,
        ];

        let config = InflateConfig { window_bits: 25 };

        let mut dest_vec_rs = vec![0u8; uncompressed.len()];
        let (output_rs, error) =
            crate::inflate::decompress_slice(&mut dest_vec_rs, compressed, config);

        assert_eq!(ReturnCode::Ok, error);
        assert_eq!(output_rs, uncompressed);
    }

    #[test]
    fn hash_calc_difference() {
        let input = [
            0, 0, 0, 0, 0, 43, 0, 0, 0, 0, 0, 0, 43, 0, 0, 0, 0, 0, 0, 0, 0, 48, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 0, 0,
            0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 0, 64, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 102, 102, 102, 102, 102,
            102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102,
            102, 102, 102, 102, 102, 102, 102, 102, 112, 102, 102, 102, 102, 102, 102, 102, 102,
            102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 0,
            0, 0, 0, 0, 0, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 0, 64, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42, 0, 0, 0,
            50, 0,
        ];

        let config = DeflateConfig {
            level: 6,
            method: Method::Deflated,
            window_bits: 9,
            mem_level: 8,
            strategy: Strategy::Default,
        };

        let expected = [
            24, 149, 99, 96, 96, 96, 96, 208, 6, 17, 112, 138, 129, 193, 128, 1, 29, 24, 50, 208,
            1, 200, 146, 169, 79, 24, 74, 59, 96, 147, 52, 71, 22, 70, 246, 88, 26, 94, 80, 128,
            83, 6, 162, 219, 144, 76, 183, 210, 5, 8, 67, 105, 36, 159, 35, 128, 57, 118, 97, 100,
            160, 197, 192, 192, 96, 196, 0, 0, 3, 228, 25, 128,
        ];

        fuzz_based_test(&input, config, &expected);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    mod _cache_lines {
        use super::State;
        // FIXME: once zlib-rs Minimum Supported Rust Version >= 1.77, switch to core::mem::offset_of
        // and move this _cache_lines module from up a level from tests to super::
        use memoffset::offset_of;

        const _: () = assert!(offset_of!(State, status) == 0);
        const _: () = assert!(offset_of!(State, _cache_line_0) == 64);
        const _: () = assert!(offset_of!(State, _cache_line_1) == 128);
        #[cfg(not(feature = "ZLIB_DEBUG"))] // ZLIB_DEBUG adds 16 bytes
        const _: () = assert!(offset_of!(State, _cache_line_2) == 192);
        #[cfg(not(feature = "ZLIB_DEBUG"))] // ZLIB_DEBUG adds 16 bytes
        const _: () = assert!(offset_of!(State, _cache_line_3) == 256);
    }
}
