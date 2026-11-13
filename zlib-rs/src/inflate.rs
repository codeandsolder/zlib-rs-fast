use core::ffi::{c_char, c_int, c_long, c_void};
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ops::ControlFlow;
use core::ptr::NonNull;

mod back_io;
mod bitreader;
mod infback;
mod inffixed_tbl;
mod inftrees;
mod window;
mod writer;

use crate::abi_cursor::{advance_abi_bytes, SliceBudget};
use crate::allocate::Allocator;
use crate::c_api::{in_func, internal_state, out_func};
use crate::output_buffer::{assume_init_prefix, WritableBytes};
use crate::state_ptr::StatePtr;
use crate::{
    adler32::adler32,
    c_api::{gz_header, z_checksum, z_size, z_stream, Z_DEFLATED},
    inflate::writer::Writer,
    traceln, Code, InflateFlush, ReturnCode, DEF_WBITS, MAX_WBITS, MIN_WBITS,
};

use crate::crc32::{crc32, Crc32Fold};

pub use self::infback::{back_end, back_init};
use self::window::BackWindow;
pub use self::window::Window;
use self::{
    bitreader::{BitBuffer, BitReader},
    inftrees::{inflate_table, CodeType, InflateTable},
};

const INFLATE_STRICT: bool = false;

// RFC 1951 §3.2.7: the transmitted code-length alphabet uses this permutation.
// Both ordinary and callback inflate must use the same layout.
const CODE_LENGTH_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

/// Run-length instructions in the dynamic code-length alphabet (RFC 1951).
/// Extra bits are still read by the owning input cursor, so an incomplete
/// input chunk never partly mutates the decoder's code-length sequence.
#[derive(Clone, Copy)]
struct CodeLengthRepeat {
    extra_bits: u8,
    base_count: usize,
    previous: bool,
}

impl CodeLengthRepeat {
    const fn from_symbol(symbol: u16) -> Option<Self> {
        match symbol {
            16 => Some(Self {
                extra_bits: 2,
                base_count: 3,
                previous: true,
            }),
            17 => Some(Self {
                extra_bits: 3,
                base_count: 3,
                previous: false,
            }),
            18 => Some(Self {
                extra_bits: 7,
                base_count: 11,
                previous: false,
            }),
            _ => None,
        }
    }
}

/// Initialize callback-driven inflate directly from caller-owned scratch storage.
///
/// This is the raw boundary used by `libz-rs-sys::inflateBackInit_`: the caller window is allowed
/// to begin uninitialized and therefore must not first be represented as an ordinary [`Window`].
///
/// # Safety
///
/// When non-null, `window` must be writable for `1 << config.window_bits` bytes and remain uniquely
/// owned until `inflateBackEnd` finishes with the stream. Invalid window-bit values are rejected
/// before the required span length is calculated.
#[cfg(any(feature = "__internal-api", test))]
#[doc(hidden)]
pub unsafe fn back_init_raw(
    stream: &mut z_stream,
    config: InflateConfig,
    window: *mut u8,
) -> ReturnCode {
    if !(MIN_WBITS..=MAX_WBITS).contains(&config.window_bits) || window.is_null() {
        return ReturnCode::StreamError;
    }
    let len = 1usize << config.window_bits;
    // SAFETY: upheld by this function's caller contract after validating the requested length.
    let window = unsafe { BackWindow::from_raw_parts(window, len) };
    infback::back_init_window(stream, config, window)
}

/// Run callback-driven raw DEFLATE decompression.
///
/// # Safety
/// The initial `strm.next_in` span must be readable for `strm.avail_in` bytes when non-null.
/// `in_` must provide readable, unchanged spans until its next invocation or this function returns.
/// `out` must not mutate or invalidate the caller-provided output window while this function runs.
/// Both descriptors must remain valid for their callback's use for the duration of the call.
pub unsafe fn back(
    strm: &mut InflateStream,
    in_: in_func,
    in_desc: *mut c_void,
    out: out_func,
    out_desc: *mut c_void,
) -> ReturnCode {
    if strm.state.kind != InflateStateKind::Back {
        return ReturnCode::StreamError;
    }

    // SAFETY: `back` inherits zlib's input-span and callback contracts from its caller for this
    // invocation. Once bound, `decode_back` uses only safe cursor operations.
    let input =
        unsafe { back_io::BackInput::new(strm.next_in.cast_const(), strm.avail_in, in_, in_desc) };
    // SAFETY: same contract for the output callback and caller-owned window.
    let output = unsafe { back_io::BackOutput::new(out, out_desc) };

    infback::decode_back(strm, input, output)
}

// SAFETY: This struct must have the same layout as [`z_stream`], so that casts and transmutations
// between the two can work without UB.
#[repr(C)]
pub struct InflateStream<'a> {
    pub(crate) next_in: *mut crate::c_api::Bytef,
    pub(crate) avail_in: crate::c_api::uInt,
    pub(crate) total_in: crate::c_api::z_size,
    pub(crate) next_out: *mut crate::c_api::Bytef,
    pub(crate) avail_out: crate::c_api::uInt,
    pub(crate) total_out: crate::c_api::z_size,
    pub(crate) msg: *mut c_char,
    pub(crate) state: StatePtr<State<'a>>,
    pub(crate) alloc: Allocator<'a>,
    pub(crate) data_type: c_int,
    pub(crate) adler: crate::c_api::z_checksum,
    pub(crate) reserved: crate::c_api::uLong,
}

// `InflateStream` overlays the public z_stream ABI exactly. `alloc` is the typed representation of
// z_stream's adjacent zalloc/zfree/opaque fields, so verify those interior offsets as well.
const _: () = {
    use core::mem::{align_of, offset_of, size_of};

    type I = InflateStream<'static>;
    type Z = z_stream;

    assert!(size_of::<I>() == size_of::<Z>());
    assert!(align_of::<I>() == align_of::<Z>());
    assert!(offset_of!(I, next_in) == offset_of!(Z, next_in));
    assert!(offset_of!(I, avail_in) == offset_of!(Z, avail_in));
    assert!(offset_of!(I, total_in) == offset_of!(Z, total_in));
    assert!(offset_of!(I, next_out) == offset_of!(Z, next_out));
    assert!(offset_of!(I, avail_out) == offset_of!(Z, avail_out));
    assert!(offset_of!(I, total_out) == offset_of!(Z, total_out));
    assert!(offset_of!(I, msg) == offset_of!(Z, msg));
    assert!(offset_of!(I, state) == offset_of!(Z, state));
    assert!(offset_of!(I, alloc) + offset_of!(Allocator<'static>, zalloc) == offset_of!(Z, zalloc));
    assert!(offset_of!(I, alloc) + offset_of!(Allocator<'static>, zfree) == offset_of!(Z, zfree));
    assert!(offset_of!(I, alloc) + offset_of!(Allocator<'static>, opaque) == offset_of!(Z, opaque));
    assert!(offset_of!(I, data_type) == offset_of!(Z, data_type));
    assert!(offset_of!(I, adler) == offset_of!(Z, adler));
    assert!(offset_of!(I, reserved) == offset_of!(Z, reserved));
};

// This C ABI view contains caller-owned pointers and mutable inflate state.
// It deliberately does not claim Send or Sync; independent owned streams can
// be scheduled concurrently without sharing this borrowed raw-pointer view.

#[cfg(feature = "__internal-test")]
#[doc(hidden)]
pub const INFLATE_STATE_SIZE: usize = core::mem::size_of::<crate::inflate::State>();

#[cfg(feature = "__internal-test")]
#[doc(hidden)]
/// # Safety
///
/// The stream must be initialized for inflate, and its state pointer must
/// refer to a live, exclusively borrowed inflate State allocation.
pub unsafe fn set_mode_dict(strm: &mut z_stream) {
    // SAFETY: the caller guarantees an initialized inflate state, exclusively
    // borrowed through this z_stream for the duration of the assignment.
    unsafe { (*strm.state.cast::<State>()).mode = Mode::Dict };
}

#[cfg(feature = "__internal-test")]
#[doc(hidden)]
/// # Safety
///
/// The pointer must be non-null, aligned, and valid for exclusive access
/// to a live initialized inflate stream and its state allocation.
pub unsafe fn set_mode_sync(strm: *mut z_stream) {
    // SAFETY: the caller guarantees a non-null, exclusive z_stream and its
    // initialized inflate state, both live for the duration of this call.
    unsafe { (*(*strm).state.cast::<State>()).mode = Mode::Sync };
}

impl<'a> InflateStream<'a> {
    /// # Safety
    ///
    /// Behavior is undefined if any of the following conditions are violated:
    ///
    /// - `strm` satisfies the conditions of [`pointer::as_ref`]
    /// - if not `NULL`, `strm` as initialized using [`init`] or similar
    ///
    /// [`pointer::as_ref`]: https://doc.rust-lang.org/core/primitive.pointer.html#method.as_ref
    #[inline]
    unsafe fn from_stream_ref_kind(
        strm: *const z_stream,
        kind: InflateStateKind,
    ) -> Option<&'a Self> {
        // SAFETY: caller guarantees `strm` is either null or points to a valid z_stream.
        let stream = unsafe { strm.as_ref() }?;
        if stream.zalloc.is_none() || stream.zfree.is_none() || stream.state.is_null() {
            return None;
        }

        // SAFETY: initialized inflate streams store a `State` behind the ABI `internal_state`
        // pointer. Checking the kind before materializing `InflateStream` keeps the regular and
        // callback-driven APIs from accidentally sharing incompatible window/storage invariants.
        let state = unsafe { stream.state.cast::<State>().as_ref() }?;
        if state.kind != kind {
            return None;
        }

        // SAFETY: InflateStream has an equivalent layout as z_stream.
        unsafe { strm.cast::<InflateStream>().as_ref() }
    }

    #[inline]
    unsafe fn from_stream_mut_kind(
        strm: *mut z_stream,
        kind: InflateStateKind,
    ) -> Option<&'a mut Self> {
        // SAFETY: the caller guarantees a live initialized stream and
        // exclusive ownership; temporarily inspect its kind through a
        // shared borrow before constructing the mutable ABI view.
        unsafe { Self::from_stream_ref_kind(strm.cast_const(), kind)? };
        // SAFETY: InflateStream has an equivalent layout as z_stream and the caller supplied a
        // mutable raw pointer satisfying `pointer::as_mut`.
        unsafe { strm.cast::<InflateStream>().as_mut() }
    }

    /// Convert a regular `inflateInit*` stream. `BackInit` streams are deliberately rejected.
    #[inline]
    /// # Safety
    ///
    /// If non-null, the stream pointer must be valid, initialized for inflate,
    /// and remain alive for the returned lifetime. There must be no concurrent
    /// exclusive access to the returned stream view.
    pub unsafe fn from_stream_ref(strm: *const z_stream) -> Option<&'a Self> {
        unsafe { Self::from_stream_ref_kind(strm, InflateStateKind::Regular) }
    }

    /// Convert a regular `inflateInit*` stream. `BackInit` streams are deliberately rejected.
    #[inline]
    /// # Safety
    ///
    /// If non-null, the stream pointer must reference a live, initialized
    /// inflate stream for which the caller guarantees exclusive access
    /// throughout the returned lifetime.
    pub unsafe fn from_stream_mut(strm: *mut z_stream) -> Option<&'a mut Self> {
        unsafe { Self::from_stream_mut_kind(strm, InflateStateKind::Regular) }
    }

    /// Convert an `inflateBackInit*` stream. Regular inflate streams are rejected.
    #[inline]
    /// # Safety
    ///
    /// If non-null, the pointer must reference a valid initialized `BackInit`
    /// stream and callback-owned window. The caller must preserve both
    /// lifetimes and exclusive access while the returned view is borrowed.
    pub unsafe fn from_back_stream_mut(strm: *mut z_stream) -> Option<&'a mut Self> {
        unsafe { Self::from_stream_mut_kind(strm, InflateStateKind::Back) }
    }

    fn as_z_stream_mut(&mut self) -> &mut z_stream {
        // SAFETY: compile-time size, alignment and every field-offset
        // assertion establish ABI layout equivalence. The exclusive borrow
        // of Self permits exactly one mutable z_stream view.
        unsafe { &mut *core::ptr::from_mut(self).cast::<z_stream>() }
    }

    /// Allocate and initialize a regular inflate stream.
    ///
    /// # Errors
    ///
    /// Reports invalid configuration/allocator contracts or failed allocation.
    pub fn new(config: InflateConfig) -> Result<Self, ReturnCode> {
        let mut inner = crate::c_api::z_stream::default();
        match crate::inflate::init(&mut inner, config) {
            ReturnCode::Ok => {}
            error => return Err(error),
        }

        // A successful init has selected both allocator callbacks and
        // installed a live, regular inflate state.
        let (Some(zalloc), Some(zfree)) = (inner.zalloc, inner.zfree) else {
            return Err(ReturnCode::StreamError);
        };
        let Some(state) = core::ptr::NonNull::new(inner.state.cast::<State<'a>>()) else {
            return Err(ReturnCode::StreamError);
        };
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
            msg: inner.msg,
            // SAFETY: successful init created a live regular-inflate State.
            state: unsafe { StatePtr::from_raw(state.as_ptr()) },
            alloc,
            data_type: inner.data_type,
            adler: inner.adler,
            reserved: inner.reserved,
        })
    }
}

const MAX_BITS: u8 = 15; // maximum number of bits in a code
const MAX_DIST_EXTRA_BITS: u8 = 13; // maximum number of extra distance bits

/// Decompresses `input` into the provided `output` buffer.
///
/// Returns a subslice of `output` containing the decompressed bytes and a
/// [`ReturnCode`] indicating the result of the operation. Returns [`ReturnCode::BufError`] if
/// there is insufficient output space.
///
/// # Example
///
/// ```
/// # use zlib_rs::*;
/// # fn foo(compressed: &[u8]) {
/// let mut buffer = [0u8; 1024];
/// let (decompressed, rc) = decompress_slice(&mut buffer, compressed, InflateConfig::default());
/// assert_eq!(rc, ReturnCode::Ok);
/// # }
/// ```
pub fn decompress_slice<'a>(
    output: &'a mut [u8],
    input: &[u8],
    config: InflateConfig,
) -> (&'a mut [u8], ReturnCode) {
    let (_consumed, produced, ret) = uncompress2_into(output, input, config);
    (&mut output[..produced], ret)
}

/// Inflates `source` into `dest`, and writes the final inflated size into `dest_len`.
pub fn uncompress<'a>(
    output: &'a mut [MaybeUninit<u8>],
    input: &[u8],
    config: InflateConfig,
) -> (&'a mut [u8], ReturnCode) {
    let (_consumed, produced, ret) = uncompress2_into(output, input, config);
    // SAFETY: `uncompress2_into` reports only bytes initialized by the inflate output cursor.
    let output = unsafe { assume_init_prefix(output, produced) };
    (output, ret)
}

/// Decompress into caller-owned storage, returning the number of *input bytes consumed*.
///
/// Any bytes after the end of the compressed stream are excluded from the count.
pub fn uncompress2<'a>(
    output: &'a mut [MaybeUninit<u8>],
    input: &[u8],
    config: InflateConfig,
) -> (usize, &'a mut [u8], ReturnCode) {
    let (consumed, produced, ret) = uncompress2_into(output, input, config);
    // SAFETY: `uncompress2_into` reports only bytes initialized by the inflate output cursor.
    let output = unsafe { assume_init_prefix(output, produced) };
    (consumed, output, ret)
}

fn uncompress2_into<O: WritableBytes + ?Sized>(
    output: &mut O,
    input: &[u8],
    config: InflateConfig,
) -> (usize, usize, ReturnCode) {
    let output_len = output.len();

    // For zero-capacity output, zlib needs one scratch byte to distinguish an empty
    // stream from a stream that would produce data. This byte is never exposed to Rust.
    let mut scratch = [0u8];
    let mut output_budget = SliceBudget::new(if output_len == 0 { 1 } else { output_len });
    let mut input_budget = SliceBudget::new(input.len());
    let dest = if output_len == 0 {
        scratch.as_mut_ptr()
    } else {
        output.byte_ptr()
    };

    let mut stream = z_stream {
        next_in: input.as_ptr().cast_mut(),
        avail_in: 0,

        zalloc: None,
        zfree: None,
        opaque: core::ptr::null_mut(),

        ..z_stream::default()
    };

    let err = init(&mut stream, config);
    if err != ReturnCode::Ok {
        return (0, 0, err);
    }

    stream.next_out = dest;
    stream.avail_out = 0;

    let Some(stream) = (unsafe { InflateStream::from_stream_mut(&raw mut stream) }) else {
        return (0, 0, ReturnCode::StreamError);
    };

    let err = loop {
        output_budget.refill(&mut stream.avail_out);
        input_budget.refill(&mut stream.avail_in);

        let err = unsafe { inflate(stream, InflateFlush::NoFlush) };

        if err != ReturnCode::Ok {
            break err;
        }
    };

    // Unoffered or unconsumed input is not part of the processed prefix.
    let consumed = input_budget.processed(stream.avail_in);
    // Preserve full Rust slice lengths even if the C total_out counter wraps.
    let produced = if output_len == 0 {
        0
    } else {
        output_budget.processed(stream.avail_out)
    };

    // A scratch byte means a zero-capacity caller destination was too small.
    let scratch_overflow = output_len == 0 && stream.total_out != 0;
    let capacity_left = output_budget.pending() != 0 || stream.avail_out != 0;
    end(stream);

    let ret = match err {
        ReturnCode::StreamEnd => ReturnCode::Ok,
        ReturnCode::NeedDict => ReturnCode::DataError,
        ReturnCode::BufError if capacity_left || scratch_overflow => ReturnCode::DataError,
        _ => err,
    };

    (consumed, produced, ret)
}

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum Mode {
    Head,
    Flags,
    Time,
    Os,
    ExLen,
    Extra,
    Name,
    Comment,
    HCrc,
    Sync,
    Mem,
    Length,
    Type,
    TypeDo,
    Stored,
    CopyBlock,
    Check,
    Len_,
    Len,
    Lit,
    LenExt,
    Dist,
    DistExt,
    Match,
    Table,
    LenLens,
    CodeLens,
    DictId,
    Dict,
    Done,
    Bad,
}

#[derive(Default, Clone, Copy)]
enum Codes {
    #[default]
    Fixed,
    CodeLengths,
    Len,
    Dist,
}

#[derive(Default, Clone, Copy)]
struct Table {
    codes: Codes,
    bits: usize,
}

#[derive(Clone, Copy)]
struct Flags(u8);

impl Default for Flags {
    fn default() -> Self {
        Self::SANE
    }
}

impl Flags {
    /// set if currently processing the last block
    const IS_LAST_BLOCK: Self = Self(0b0000_0001);

    /// set if a custom dictionary was provided
    const HAVE_DICT: Self = Self(0b0000_0010);

    /// if false, allow invalid distance too far
    const SANE: Self = Self(0b0000_0100);

    pub(crate) const fn contains(self, other: Self) -> bool {
        debug_assert!(other.0.count_ones() == 1);

        self.0 & other.0 != 0
    }

    #[inline]
    pub(crate) const fn update(&mut self, other: Self, value: bool) {
        if value {
            *self = Self(self.0 | other.0);
        } else {
            *self = Self(self.0 & !other.0);
        }
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InflateStateKind {
    Regular,
    Back,
}

/// Writable caller-owned gzip-header destination bound by `get_header`.
///
/// This deliberately stores only a raw handle rather than a long-lived `&mut gz_header`:
/// `inflateCopy()` shallow-copies the destination exactly like zlib. Each parser operation creates
/// only the short mutable access it needs, and all nested extra/name/comment pointer writes are
/// confined here.
#[derive(Clone, Copy)]
enum GzipStringField {
    Name,
    Comment,
}

#[repr(transparent)]
#[derive(Clone, Copy)]
struct GzipHeaderOutput(NonNull<gz_header>);

impl GzipHeaderOutput {
    /// # Safety
    ///
    /// `head` and its non-null nested output buffers must satisfy the complete contract documented
    /// on `get_header` for as long as any stream (including `inflateCopy` clones) may populate it.
    unsafe fn new(head: &mut gz_header) -> Self {
        Self(NonNull::from(head))
    }

    fn with_header_mut<R>(&mut self, f: impl FnOnce(&mut gz_header) -> R) -> R {
        // SAFETY: established by `new`. The callback shape keeps the mutable reference scoped to
        // this operation even though the raw handle itself is intentionally copyable for
        // `inflateCopy` compatibility.
        f(unsafe { self.0.as_mut() })
    }

    fn set_done(&mut self, done: i32) {
        self.with_header_mut(|head| head.done = done);
    }

    fn set_text(&mut self, text: i32) {
        self.with_header_mut(|head| head.text = text);
    }

    fn set_time(&mut self, time: z_size) {
        self.with_header_mut(|head| head.time = time);
    }

    fn set_os(&mut self, xflags: i32, os: i32) {
        self.with_header_mut(|head| {
            head.xflags = xflags;
            head.os = os;
        });
    }

    fn set_extra_len(&mut self, len: u16) {
        self.with_header_mut(|head| head.extra_len = u32::from(len));
    }

    fn clear_extra(&mut self) {
        self.with_header_mut(|head| head.extra = core::ptr::null_mut());
    }

    fn clear_name(&mut self) {
        self.with_header_mut(|head| head.name = core::ptr::null_mut());
    }

    fn clear_comment(&mut self) {
        self.with_header_mut(|head| head.comment = core::ptr::null_mut());
    }

    fn finish(&mut self, hcrc: i32) {
        self.with_header_mut(|head| {
            head.hcrc = hcrc;
            head.done = 1;
        });
    }

    /// Copy into a caller-owned gzip metadata field without crossing its capacity.
    ///
    /// The nested buffer contracts are established by the unsafe `get_header`
    /// boundary, while this helper confines raw writes to one checked operation.
    fn write_bounded(ptr: *mut u8, capacity: usize, offset: usize, bytes: &[u8]) -> usize {
        if ptr.is_null() || offset >= capacity {
            return 0;
        }
        let count = bytes.len().min(capacity - offset);
        if count != 0 {
            // SAFETY: get_header requires this nested destination to be writable for
            // capacity bytes, disjoint from the caller input. The offset and count
            // bound this copy to that allocation.
            unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.add(offset), count) };
        }
        count
    }

    fn write_extra(&mut self, remaining_before: usize, bytes: &[u8]) -> Result<(), ReturnCode> {
        self.with_header_mut(|head| {
            // The wire length is u16. Reject inconsistent progress before
            // touching caller-owned memory or consuming more compressed input.
            let declared = usize::try_from(head.extra_len).map_err(|_| ReturnCode::StreamError)?;
            let Some(written) = declared.checked_sub(remaining_before) else {
                return Err(ReturnCode::StreamError);
            };
            if bytes.len() > remaining_before {
                return Err(ReturnCode::StreamError);
            }
            Self::write_bounded(head.extra, head.extra_max as usize, written, bytes);
            Ok(())
        })
    }

    fn write_string(&mut self, field: GzipStringField, offset: usize, bytes: &[u8]) -> usize {
        self.with_header_mut(|head| {
            let (ptr, capacity) = match field {
                GzipStringField::Name => (head.name, head.name_max as usize),
                GzipStringField::Comment => (head.comment, head.comm_max as usize),
            };
            Self::write_bounded(ptr, capacity, offset, bytes)
        })
    }
}

/// Consume a gzip NUL-terminated metadata field across any number of caller
/// input fragments. The stored count reflects only bytes that fit the destination,
/// while the input cursor and optional header CRC always account for the entire field.
fn consume_gzip_header_string(
    reader: &mut BitReader<'_>,
    header: Option<GzipHeaderOutput>,
    field: GzipStringField,
    stored: &mut usize,
    checksum: &mut u32,
    update_crc: bool,
) -> bool {
    let available = reader.as_slice();
    let terminator = available.iter().position(|&byte| byte == 0);
    let count = terminator.map_or(available.len(), |position| position + 1);
    let bytes = &available[..count];

    if let Some(mut header) = header {
        *stored += header.write_string(field, *stored, bytes);
    }
    if update_crc {
        *checksum = crc32(*checksum, bytes);
    }
    reader.advance(count);

    terminator.is_some()
}

#[repr(C, align(64))]
pub struct State<'a> {
    kind: InflateStateKind,
    /// Current inflate mode
    mode: Mode,

    flags: Flags,

    /// log base 2 of requested window size
    wbits: u8,

    /// bitflag
    ///
    /// - bit 0 true if zlib
    /// - bit 1 true if gzip
    /// - bit 2 true to validate check value
    wrap: u8,

    // allocated window if needed (capacity == 0 if unused)
    window: Window<'a>,

    //
    /// number of code length code lengths
    ncode: usize,
    /// number of length code lengths
    nlen: usize,
    /// number of distance code lengths
    ndist: usize,
    /// number of code lengths in lens[]
    have: usize,
    /// next available space in codes[]
    next: usize, // represented as an index, don't want a self-referential structure here

    // Only buffered bits persist across calls; caller input pointers are call-local.
    bit_buffer: BitBuffer,

    total: usize,

    /// length of a block to copy
    length: usize,
    /// distance back to copy the string from
    offset: usize,

    /// DEFLATE extra-bit count (at most 15 bits), stored in its native width.
    extra: u8,

    /// bits back of last unprocessed length/lit
    back: usize,

    /// initial length of match
    was: usize,

    gzip_flags: i32,

    checksum: u32,
    crc_fold: Crc32Fold,

    error_message: Option<&'static str>,

    /// Caller-owned gzip-header destination. zlib permits `inflateCopy()` to duplicate this
    /// handle, so it must not be represented as a long-lived Rust `&mut` reference.
    head: Option<GzipHeaderOutput>,
    dmax: usize,

    /// table for length/literal codes
    len_table: Table,

    /// table for dist codes
    dist_table: Table,

    codes_codes: [Code; crate::ENOUGH_LENS],
    len_codes: [Code; crate::ENOUGH_LENS],
    dist_codes: [Code; crate::ENOUGH_DISTS],

    /// temporary storage space for code lengths
    lens: [u16; 320],
    /// work area for code table building
    work: [u16; 288],

    allocation_start: *mut u8,
    total_allocation_size: usize,

    // Back-only state stays at the cold tail so regular decode field offsets do not move.
    back_window: BackWindow<'a>,
}

impl<'a> State<'a> {
    fn new(kind: InflateStateKind) -> Self {
        Self {
            kind,
            flags: Flags::default(),
            wrap: 0,
            mode: Mode::Head,
            length: 0,

            len_table: Table::default(),
            dist_table: Table::default(),

            wbits: 0,
            offset: 0,
            extra: 0,
            back: 0,
            was: 0,

            bit_buffer: BitBuffer::default(),

            total: 0,

            window: Window::empty(),
            back_window: BackWindow::empty(),
            head: None,

            lens: [0u16; 320],
            work: [0u16; 288],

            ncode: 0,
            nlen: 0,
            ndist: 0,
            have: 0,
            next: 0,

            error_message: None,

            checksum: 0,
            crc_fold: Crc32Fold::new(),

            dmax: 0,
            gzip_flags: 0,

            codes_codes: [Code::default(); crate::ENOUGH_LENS],
            len_codes: [Code::default(); crate::ENOUGH_LENS],
            dist_codes: [Code::default(); crate::ENOUGH_DISTS],

            allocation_start: core::ptr::null_mut(),
            total_allocation_size: 0,
        }
    }

    /// Clone regular inflate state while rebinding the destination-owned history window.
    ///
    /// The explicit field list keeps `inflateCopy` semantics reviewable: adding a new state field
    /// requires choosing whether it is copied or rebound instead of silently joining a raw memcpy.
    fn clone_rebound(
        &self,
        window: Window<'a>,
        allocation_start: *mut u8,
        total_allocation_size: usize,
    ) -> Self {
        debug_assert!(self.back_window.is_empty());
        Self {
            kind: self.kind,
            mode: self.mode,
            flags: self.flags,
            wbits: self.wbits,
            wrap: self.wrap,
            window,
            back_window: BackWindow::empty(),
            ncode: self.ncode,
            nlen: self.nlen,
            ndist: self.ndist,
            have: self.have,
            next: self.next,
            bit_buffer: self.bit_buffer,
            total: self.total,
            length: self.length,
            offset: self.offset,
            extra: self.extra,
            back: self.back,
            was: self.was,
            gzip_flags: self.gzip_flags,
            checksum: self.checksum,
            crc_fold: self.crc_fold,
            error_message: self.error_message,
            head: self.head,
            dmax: self.dmax,
            len_table: self.len_table,
            dist_table: self.dist_table,
            codes_codes: self.codes_codes,
            len_codes: self.len_codes,
            dist_codes: self.dist_codes,
            lens: self.lens,
            work: self.work,
            allocation_start,
            total_allocation_size,
        }
    }

    /// Record one transmitted code-length-alphabet entry in RFC 1951 order.
    const fn append_code_length_alphabet(&mut self, length: u16) {
        self.lens[CODE_LENGTH_ORDER[self.have]] = length;
        self.have += 1;
    }

    /// Append a decoded repeat instruction only if the complete run is valid.
    /// No partially initialized code lengths become visible on failure.
    fn append_code_length_repeat(
        &mut self,
        repeat: CodeLengthRepeat,
        extra_value: u64,
    ) -> Result<(), &'static str> {
        const BAD_REPEAT: &str = "invalid bit length repeat\0";
        let remaining = (self.nlen + self.ndist)
            .checked_sub(self.have)
            .ok_or(BAD_REPEAT)?;
        let extra = usize::try_from(extra_value).map_err(|_| BAD_REPEAT)?;
        let count = repeat.base_count.checked_add(extra).ok_or(BAD_REPEAT)?;
        if count > remaining {
            return Err(BAD_REPEAT);
        }
        let value = if repeat.previous {
            if self.have == 0 {
                return Err(BAD_REPEAT);
            }
            self.lens[self.have - 1]
        } else {
            0
        };
        self.lens[self.have..self.have + count].fill(value);
        self.have += count;
        Ok(())
    }

    /// Complete the code-length alphabet and construct its canonical table.
    /// Both inflate front-ends use the same validation and table metadata.
    fn finish_code_length_alphabet(&mut self) -> Result<(), &'static str> {
        while self.have < CODE_LENGTH_ORDER.len() {
            self.append_code_length_alphabet(0);
        }

        let InflateTable::Success { root, used } = inflate_table(
            CodeType::Codes,
            &self.lens[..CODE_LENGTH_ORDER.len()],
            &mut self.codes_codes,
            7,
            &mut self.work,
        ) else {
            return Err("invalid code lengths set\0");
        };

        self.next = used;
        self.len_table.codes = Codes::CodeLengths;
        self.len_table.bits = root;
        self.have = 0;
        Ok(())
    }

    /// Build the dynamic literal/length and distance tables after both length
    /// alphabets are decoded. The ordinary and callback decoders share the
    /// exact same validation, code-table sizes and error strings.
    fn finish_dynamic_trees(&mut self) -> Result<(), &'static str> {
        if self.lens[256] == 0 {
            return Err("invalid code -- missing end-of-block\0");
        }

        let InflateTable::Success { root, used } = inflate_table(
            CodeType::Lens,
            &self.lens[..self.nlen],
            &mut self.len_codes,
            10,
            &mut self.work,
        ) else {
            return Err("invalid literal/lengths set\0");
        };
        self.len_table.codes = Codes::Len;
        self.len_table.bits = root;
        self.next = used;

        let InflateTable::Success { root, used } = inflate_table(
            CodeType::Dists,
            &self.lens[self.nlen..][..self.ndist],
            &mut self.dist_codes,
            9,
            &mut self.work,
        ) else {
            return Err("invalid distances set\0");
        };
        self.dist_table.codes = Codes::Dist;
        self.dist_table.bits = root;
        self.next += used;
        Ok(())
    }

    const fn len_table_ref(&self) -> &[Code] {
        match self.len_table.codes {
            Codes::Fixed => &self::inffixed_tbl::LENFIX,
            Codes::CodeLengths => &self.codes_codes,
            Codes::Len => &self.len_codes,
            Codes::Dist => &self.dist_codes,
        }
    }

    const fn dist_table_ref(&self) -> &[Code] {
        match self.dist_table.codes {
            Codes::Fixed => &self::inffixed_tbl::DISTFIX,
            Codes::CodeLengths => &self.codes_codes,
            Codes::Len => &self.len_codes,
            Codes::Dist => &self.dist_codes,
        }
    }

    const fn len_table_get(&self, index: usize) -> Code {
        self.len_table_ref()[index]
    }

    const fn dist_table_get(&self, index: usize) -> Code {
        self.dist_table_ref()[index]
    }
}

// swaps endianness
const fn zswap32(q: u32) -> u32 {
    u32::from_be(q.to_le())
}

const INFLATE_FAST_MIN_HAVE: usize = 15;
const INFLATE_FAST_MIN_LEFT: usize = 260;

/// Caller-output cursor for one ordinary inflate call.
///
/// The writer points into `z_stream.next_out` and therefore must not live in persistent `State`.
/// `accounted_remaining` preserves zlib's trailer bookkeeping: when checksum processing folds
/// bytes already produced in this call into `State::total`, later return accounting starts from
/// that checkpoint instead of counting those bytes twice.
struct InflateOutput<'a> {
    writer: Writer<'a>,
    accounted_remaining: usize,
}

impl<'a> InflateOutput<'a> {
    const unsafe fn new_uninit(ptr: *mut u8, len: usize) -> Self {
        Self {
            writer: unsafe { Writer::new_uninit(ptr, len) },
            accounted_remaining: len,
        }
    }

    #[inline]
    const fn mark_accounted(&mut self) {
        self.accounted_remaining = self.writer.remaining();
    }
    #[inline]
    const fn written_since_accounted(&self) -> usize {
        self.accounted_remaining - self.writer.remaining()
    }
    #[inline]
    const fn remaining(&self) -> usize {
        self.writer.remaining()
    }
    #[inline]
    const fn len(&self) -> usize {
        self.writer.len()
    }
    #[inline]
    const fn is_full(&self) -> bool {
        self.writer.is_full()
    }
    #[inline]
    fn filled(&self) -> &[u8] {
        self.writer.filled()
    }
    #[inline]
    const fn next_out(&mut self) -> *mut core::mem::MaybeUninit<u8> {
        self.writer.next_out()
    }
    #[inline]
    const fn push(&mut self, byte: u8) {
        self.writer.push(byte);
    }
    #[inline]
    fn extend(&mut self, bytes: &[u8]) {
        self.writer.extend(bytes);
    }
    #[inline]
    fn fill_zero(&mut self, len: usize) {
        self.writer.fill_zero(len);
    }
    #[inline]
    fn extend_from_window(&mut self, window: &Window<'_>, range: core::ops::Range<usize>) {
        self.writer.extend_from_window(window, range);
    }
    #[inline]
    fn copy_match(&mut self, offset_from_end: usize, length: usize) {
        self.writer.copy_match(offset_from_end, length);
    }
}

#[derive(Debug, Clone, Copy)]
struct DispatchExit {
    mode: Mode,
    code: ReturnCode,
}

#[derive(Debug, Clone, Copy)]
struct LenDecodeOutcome<'input> {
    mode: Mode,
    reader: BitReader<'input>,
    flow: ControlFlow<ReturnCode, ()>,
}

impl<'input> LenDecodeOutcome<'input> {
    const fn continue_with(mode: Mode, reader: BitReader<'input>) -> Self {
        Self {
            mode,
            reader,
            flow: ControlFlow::Continue(()),
        }
    }

    const fn break_with(mode: Mode, reader: BitReader<'input>, code: ReturnCode) -> Self {
        Self {
            mode,
            reader,
            flow: ControlFlow::Break(code),
        }
    }
}

#[inline]
fn need_bits_for_mode(
    reader: &mut BitReader<'_>,
    count: usize,
    mode: Mode,
) -> Result<(), DispatchExit> {
    reader
        .need_bits(count)
        .map_err(|code| DispatchExit { mode, code })
}

#[inline]
fn pull_byte_for_mode(reader: &mut BitReader<'_>, mode: Mode) -> Result<u8, DispatchExit> {
    reader
        .pull_byte()
        .map_err(|code| DispatchExit { mode, code })
}

#[derive(Debug, Clone, Copy)]
enum DistanceDecodeError {
    Input(ReturnCode),
    InvalidCode,
}

#[derive(Debug, Clone, Copy)]
struct DecodedDistance {
    offset: usize,
    extra: u8,
    back_bits: usize,
}

#[inline]
const fn emit_literal(output: &mut InflateOutput<'_>, literal: u8) -> bool {
    if output.is_full() {
        return false;
    }
    output.push(literal);
    true
}

#[inline]
fn consume_extra_bits(
    reader: &mut BitReader<'_>,
    extra: u8,
    value: &mut usize,
    back: &mut usize,
) -> Result<(), ReturnCode> {
    if extra != 0 {
        let width = usize::from(extra);
        reader.need_bits(width)?;
        *value += reader.bits_usize(width);
        reader.drop_bits(extra);
        *back += width;
    }
    Ok(())
}

/// Validate the two little-endian 16-bit words that prefix a stored
/// DEFLATE block. A failed complement check is a format error, not a
/// truncated or partially accepted length.
#[inline]
pub(super) fn stored_block_length(header: u32) -> Option<u16> {
    let bytes = header.to_le_bytes();
    let len = u16::from_le_bytes([bytes[0], bytes[1]]);
    let complement = u16::from_le_bytes([bytes[2], bytes[3]]);
    (len == !complement).then_some(len)
}

/// gzip's ISIZE trailer is defined modulo 2^32, even when a decompression
/// has produced more than 4 GiB. Preserve the low four bytes explicitly on
/// both 32- and 64-bit hosts rather than requiring a fallible cast.
#[inline]
const fn gzip_isize(total: usize) -> u32 {
    let bytes = total.to_le_bytes();
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

#[inline]
fn decode_distance_code(
    reader: &mut BitReader<'_>,
    table: &[Code],
    table_bits: usize,
) -> Result<DecodedDistance, DistanceDecodeError> {
    let mut here;
    loop {
        here = table[reader.bits_usize(table_bits)];
        if here.bits <= reader.bits_in_buffer() {
            break;
        }
        reader.pull_byte().map_err(DistanceDecodeError::Input)?;
    }

    let mut back_bits = 0;
    if here.op & 0xf0 == 0 {
        let last = here;
        loop {
            let bits = reader.bits_usize(usize::from(last.bits + last.op));
            here = table[usize::from(last.val) + (bits >> last.bits)];
            if last.bits + here.bits <= reader.bits_in_buffer() {
                break;
            }
            reader.pull_byte().map_err(DistanceDecodeError::Input)?;
        }
        reader.drop_bits(last.bits);
        back_bits = last.bits as usize;
    }

    reader.drop_bits(here.bits);
    if here.op & 64 != 0 {
        return Err(DistanceDecodeError::InvalidCode);
    }

    Ok(DecodedDistance {
        offset: here.val as usize,
        extra: here.op & MAX_BITS,
        back_bits,
    })
}

#[inline]
fn copy_match_bytes(
    window: &Window<'_>,
    sane: bool,
    offset: usize,
    length: &mut usize,
    output: &mut InflateOutput<'_>,
) -> Result<bool, &'static str> {
    let left = output.remaining();
    let output_len = output.len();

    let copied = if offset > output_len {
        let mut from_end = offset - output_len;
        if from_end > window.have() {
            if sane {
                return Err("invalid distance too far back\0");
            }
            // Non-strict inflateUndermine is defined to synthesize zero
            // bytes for the history that precedes the initialized window.
            // The resumed Match state re-enters here with the remaining length.
            let missing = from_end - window.have();
            let copied = missing.min(*length).min(left);
            output.fill_zero(copied);
            *length -= copied;
            return Ok(*length == 0);
        }

        let wnext = window.next();
        let wsize = window.size();
        let from = if from_end > wnext {
            from_end -= wnext;
            wsize - from_end
        } else {
            wnext - from_end
        };

        let copied = Ord::min(Ord::min(from_end, *length), left);
        output.extend_from_window(window, from..from + copied);
        copied
    } else {
        let copied = Ord::min(*length, left);
        output.copy_match(offset, copied);
        copied
    };

    *length -= copied;
    Ok(*length == 0)
}

impl State<'_> {
    // This logic is split into its own function for two reasons
    //
    // - We get to load state to the stack; doing this in all cases is expensive, but doing it just
    //      for Len and related states is very helpful.
    // - The `-Cllvm-args=-enable-dfa-jump-thread` llvm arg is able to optimize this function, but
    //      not the entirity of `dispatch`. We get a massive boost from that pass.
    //
    // Resumed substates still enter directly through `dispatch`: routing all of them back through
    // this function measurably regresses tiny-chunk inflate. Shared inline primitives below this
    // boundary remove the duplicated decode/copy mechanics without changing that call shape.
    fn len_and_friends(
        &mut self,
        reader: &mut BitReader<'_>,
        output: &mut InflateOutput<'_>,
    ) -> ControlFlow<ReturnCode, ()> {
        let avail_in = reader.bytes_remaining();
        let avail_out = output.remaining();

        if avail_in >= INFLATE_FAST_MIN_HAVE && avail_out >= INFLATE_FAST_MIN_LEFT {
            inflate_fast_help_impl::<true, _>(self, reader, output);
            match self.mode {
                Mode::Len => {}
                _ => return ControlFlow::Continue(()),
            }
        }

        let outcome = self.len_and_friends_inner(*reader, output);
        self.mode = outcome.mode;
        *reader = outcome.reader;
        outcome.flow
    }

    fn len_and_friends_inner<'input>(
        &mut self,
        mut bit_reader: BitReader<'input>,
        output: &mut InflateOutput<'_>,
    ) -> LenDecodeOutcome<'input> {
        let mut mode = self.mode;

        let len_table = match self.len_table.codes {
            Codes::Fixed => &self::inffixed_tbl::LENFIX[..],
            Codes::CodeLengths => &self.codes_codes,
            Codes::Len => &self.len_codes,
            Codes::Dist => &self.dist_codes,
        };

        let dist_table = match self.dist_table.codes {
            Codes::Fixed => &self::inffixed_tbl::DISTFIX[..],
            Codes::CodeLengths => &self.codes_codes,
            Codes::Len => &self.len_codes,
            Codes::Dist => &self.dist_codes,
        };

        loop {
            mode = 'top: {
                match mode {
                    Mode::Len => {
                        let avail_in = bit_reader.bytes_remaining();
                        let avail_out = output.remaining();

                        // The fast decoder may emit a maximum-length DEFLATE match before it
                        // rechecks output space, so keep the complete fast-path reserve available.
                        if avail_in >= INFLATE_FAST_MIN_HAVE && avail_out >= INFLATE_FAST_MIN_LEFT {
                            self.mode = mode;
                            inflate_fast_help_impl::<true, _>(self, &mut bit_reader, output);
                            return LenDecodeOutcome::continue_with(self.mode, bit_reader);
                        }

                        self.back = 0;

                        // get a literal, length, or end-of-block code
                        let mut here;
                        loop {
                            let bits = bit_reader.bits_usize(self.len_table.bits);
                            here = len_table[bits];

                            if here.bits <= bit_reader.bits_in_buffer() {
                                break;
                            }

                            if let Err(return_code) = bit_reader.pull_byte() {
                                return LenDecodeOutcome::break_with(mode, bit_reader, return_code);
                            };
                        }

                        if here.op != 0 && here.op & 0xf0 == 0 {
                            let last = here;
                            loop {
                                let bits = bit_reader.bits_u16(usize::from(last.bits + last.op));
                                here = len_table[usize::from(last.val + (bits >> last.bits))];
                                if last.bits + here.bits <= bit_reader.bits_in_buffer() {
                                    break;
                                }

                                if let Err(return_code) = bit_reader.pull_byte() {
                                    return LenDecodeOutcome::break_with(
                                        mode,
                                        bit_reader,
                                        return_code,
                                    );
                                };
                            }

                            bit_reader.drop_bits(last.bits);
                            self.back += last.bits as usize;
                        }

                        bit_reader.drop_bits(here.bits);
                        self.back += here.bits as usize;
                        self.length = here.val as usize;

                        if here.op == 0 {
                            break 'top Mode::Lit;
                        } else if here.op & 32 != 0 {
                            // end of block

                            traceln!("inflate:         end of block");

                            self.back = usize::MAX;
                            mode = Mode::Type;

                            return LenDecodeOutcome::continue_with(mode, bit_reader);
                        } else if here.op & 64 != 0 {
                            mode = Mode::Bad;
                            {
                                let msg: &'static str = "invalid literal/length code\0";
                                self.error_message = Some(msg);
                                return LenDecodeOutcome::break_with(
                                    mode,
                                    bit_reader,
                                    ReturnCode::DataError,
                                );
                            }
                        } else {
                            // length code
                            self.extra = here.op & MAX_BITS;
                            break 'top Mode::LenExt;
                        }
                    }
                    Mode::Lit => {
                        if !emit_literal(output, self.length as u8) {
                            traceln!("Ok: writer is full ({} bytes)", output.writer.capacity());
                            return LenDecodeOutcome::break_with(
                                Mode::Lit,
                                bit_reader,
                                ReturnCode::Ok,
                            );
                        }
                        break 'top Mode::Len;
                    }
                    Mode::LenExt => {
                        if let Err(code) = consume_extra_bits(
                            &mut bit_reader,
                            self.extra,
                            &mut self.length,
                            &mut self.back,
                        ) {
                            return LenDecodeOutcome::break_with(Mode::LenExt, bit_reader, code);
                        }
                        traceln!("inflate: length {}", self.length);
                        self.was = self.length;
                        break 'top Mode::Dist;
                    }
                    Mode::Dist => {
                        match decode_distance_code(
                            &mut bit_reader,
                            dist_table,
                            self.dist_table.bits,
                        ) {
                            Ok(decoded) => {
                                self.back += decoded.back_bits;
                                self.offset = decoded.offset;
                                self.extra = decoded.extra;
                                break 'top Mode::DistExt;
                            }
                            Err(DistanceDecodeError::Input(code)) => {
                                return LenDecodeOutcome::break_with(Mode::Dist, bit_reader, code);
                            }
                            Err(DistanceDecodeError::InvalidCode) => {
                                self.error_message = Some("invalid distance code\0");
                                return LenDecodeOutcome::break_with(
                                    Mode::Bad,
                                    bit_reader,
                                    ReturnCode::DataError,
                                );
                            }
                        }
                    }
                    Mode::DistExt => {
                        if let Err(code) = consume_extra_bits(
                            &mut bit_reader,
                            self.extra,
                            &mut self.offset,
                            &mut self.back,
                        ) {
                            return LenDecodeOutcome::break_with(Mode::DistExt, bit_reader, code);
                        }
                        if INFLATE_STRICT && self.offset > self.dmax {
                            self.error_message = Some("invalid distance code too far back\0");
                            return LenDecodeOutcome::break_with(
                                Mode::Bad,
                                bit_reader,
                                ReturnCode::DataError,
                            );
                        }
                        traceln!("inflate: distance {}", self.offset);
                        break 'top Mode::Match;
                    }
                    Mode::Match => {
                        if output.is_full() {
                            traceln!(
                                "BufError: writer is full ({} bytes)",
                                output.writer.capacity()
                            );
                            return LenDecodeOutcome::break_with(
                                Mode::Match,
                                bit_reader,
                                ReturnCode::Ok,
                            );
                        }
                        match copy_match_bytes(
                            &self.window,
                            self.flags.contains(Flags::SANE),
                            self.offset,
                            &mut self.length,
                            output,
                        ) {
                            Ok(true) => break 'top Mode::Len,
                            Ok(false) => break 'top Mode::Match,
                            Err(msg) => {
                                self.error_message = Some(msg);
                                return LenDecodeOutcome::break_with(
                                    Mode::Bad,
                                    bit_reader,
                                    ReturnCode::DataError,
                                );
                            }
                        }
                    }
                    other => unreachable!("len decoder entered non-literal/match state: {other:?}"),
                }
            }
        }
    }

    /// Header errors must publish the Bad mode on return, including when
    /// parsing is resumed after a partial input chunk.
    const fn invalid_header(&mut self, msg: &'static str) -> DispatchExit {
        DispatchExit {
            mode: Mode::Bad,
            code: self.bad(msg),
        }
    }

    /// Read the wrapper's leading two bytes, dispatching zlib or gzip without
    /// touching payload input. Header-only work remains off the inflate hot path.
    fn decode_wrapper_head(
        &mut self,
        bit_reader: &mut BitReader<'_>,
    ) -> Result<Mode, DispatchExit> {
        if self.wrap == 0 {
            return Ok(Mode::TypeDo);
        }
        need_bits_for_mode(bit_reader, 16, Mode::Head)?;

        if (self.wrap & 2) != 0 && bit_reader.hold() == 0x8b1f {
            if self.wbits == 0 {
                self.wbits = 15;
            }
            self.checksum = crc32(
                crate::CRC32_INITIAL_VALUE,
                &bit_reader.bits_u16(16).to_le_bytes(),
            );
            bit_reader.init_bits();
            return Ok(Mode::Flags);
        }

        if let Some(mut header) = self.head {
            header.set_done(-1);
        }

        if (self.wrap & 1) == 0 || ((bit_reader.bits(8) << 8) + (bit_reader.hold() >> 8)) % 31 != 0
        {
            return Err(self.invalid_header("incorrect header check\0"));
        }
        if i32::from(bit_reader.bits_u16(4)) != Z_DEFLATED {
            return Err(self.invalid_header("unknown compression method\0"));
        }

        bit_reader.drop_bits(4);
        let len = bit_reader.bits_u16(4).to_le_bytes()[0] + 8;
        if self.wbits == 0 {
            self.wbits = len;
        }
        if i32::from(len) > MAX_WBITS || len > self.wbits {
            return Err(self.invalid_header("invalid window size\0"));
        }

        self.dmax = 1 << len;
        self.gzip_flags = 0;
        self.checksum = crate::ADLER32_INITIAL_VALUE;
        let has_dictionary = bit_reader.hold() & 0x200 != 0;
        bit_reader.init_bits();
        Ok(if has_dictionary {
            Mode::DictId
        } else {
            Mode::Type
        })
    }

    fn decode_gzip_flags(&mut self, bit_reader: &mut BitReader<'_>) -> Result<Mode, DispatchExit> {
        need_bits_for_mode(bit_reader, 16, Mode::Flags)?;
        let bytes = bit_reader.bits_u16(16).to_le_bytes();
        self.gzip_flags = i32::from(u16::from_le_bytes(bytes));
        if self.gzip_flags & 0xff != Z_DEFLATED {
            return Err(self.invalid_header("unknown compression method\0"));
        }
        if self.gzip_flags & 0xe000 != 0 {
            return Err(self.invalid_header("unknown header flags set\0"));
        }

        if let Some(mut head) = self.head {
            head.set_text(i32::from(bytes[1] & 1));
        }
        if self.gzip_flags & 0x0200 != 0 && self.wrap & 4 != 0 {
            self.checksum = crc32(self.checksum, &bytes);
        }
        bit_reader.init_bits();
        Ok(Mode::Time)
    }

    fn decode_gzip_time(&mut self, bit_reader: &mut BitReader<'_>) -> Result<Mode, DispatchExit> {
        need_bits_for_mode(bit_reader, 32, Mode::Time)?;
        let timestamp = bit_reader.bits_u32(32);
        if let Some(mut head) = self.head {
            head.set_time(z_size::from(timestamp));
        }
        if self.gzip_flags & 0x0200 != 0 && self.wrap & 4 != 0 {
            self.checksum = crc32(self.checksum, &timestamp.to_le_bytes());
        }
        bit_reader.init_bits();
        Ok(Mode::Os)
    }

    fn decode_gzip_os(&mut self, bit_reader: &mut BitReader<'_>) -> Result<Mode, DispatchExit> {
        need_bits_for_mode(bit_reader, 16, Mode::Os)?;
        let bytes = bit_reader.bits_u16(16).to_le_bytes();
        if let Some(mut head) = self.head {
            head.set_os(i32::from(bytes[0]), i32::from(bytes[1]));
        }
        if self.gzip_flags & 0x0200 != 0 && self.wrap & 4 != 0 {
            self.checksum = crc32(self.checksum, &bytes);
        }
        bit_reader.init_bits();
        Ok(Mode::ExLen)
    }

    fn decode_gzip_extra_length(
        &mut self,
        reader: &mut BitReader<'_>,
    ) -> Result<Mode, DispatchExit> {
        if self.gzip_flags & 0x0400 != 0 {
            need_bits_for_mode(reader, 16, Mode::ExLen)?;
            let extra_len = reader.bits_u16(16);
            self.length = usize::from(extra_len);
            if let Some(mut head) = self.head {
                head.set_extra_len(extra_len);
            }
            if self.gzip_flags & 0x0200 != 0 && self.wrap & 4 != 0 {
                self.checksum = crc32(self.checksum, &extra_len.to_le_bytes());
            }
            reader.init_bits();
        } else if let Some(mut head) = self.head {
            head.clear_extra();
        }
        Ok(Mode::Extra)
    }

    /// None means the field is still incomplete, so the caller must return
    /// Ok without advancing the persistent gzip header mode.
    fn decode_gzip_extra(
        &mut self,
        reader: &mut BitReader<'_>,
    ) -> Result<Option<Mode>, DispatchExit> {
        if self.gzip_flags & 0x0400 != 0 {
            let available = self.length.min(reader.bytes_remaining());
            if available > 0 {
                let bytes = &reader.as_slice()[..available];
                if let Some(mut head) = self.head {
                    head.write_extra(self.length, bytes)
                        .map_err(|code| DispatchExit {
                            mode: Mode::Extra,
                            code,
                        })?;
                }
                if self.gzip_flags & 0x0200 != 0 && self.wrap & 4 != 0 {
                    self.checksum = crc32(self.checksum, bytes);
                }
                reader.advance(available);
                self.length -= available;
            }
            if self.length != 0 {
                return Ok(None);
            }
        }
        self.length = 0;
        Ok(Some(Mode::Name))
    }

    /// Consume the optional name or comment until its NUL byte appears.
    /// Incomplete metadata yields None with its mode and cursor intact.
    fn decode_gzip_string(
        &mut self,
        reader: &mut BitReader<'_>,
        field: GzipStringField,
    ) -> Option<Mode> {
        let (flag, next) = match field {
            GzipStringField::Name => (0x0800, Mode::Comment),
            GzipStringField::Comment => (0x1000, Mode::HCrc),
        };
        if self.gzip_flags & flag != 0 {
            if !consume_gzip_header_string(
                reader,
                self.head,
                field,
                &mut self.length,
                &mut self.checksum,
                self.gzip_flags & 0x0200 != 0 && self.wrap & 4 != 0,
            ) {
                return None;
            }
        } else if let Some(mut head) = self.head {
            match field {
                GzipStringField::Name => head.clear_name(),
                GzipStringField::Comment => head.clear_comment(),
            }
        }
        if matches!(field, GzipStringField::Name) {
            self.length = 0;
        }
        Some(next)
    }

    fn decode_gzip_header_crc(&mut self, reader: &mut BitReader<'_>) -> Result<Mode, DispatchExit> {
        if self.gzip_flags & 0x0200 != 0 {
            need_bits_for_mode(reader, 16, Mode::HCrc)?;
            let checksum_bytes = self.checksum.to_le_bytes();
            let expected = u16::from_le_bytes([checksum_bytes[0], checksum_bytes[1]]);
            if self.wrap & 4 != 0 && reader.bits_u16(16) != expected {
                return Err(self.invalid_header("header crc mismatch\0"));
            }
            reader.init_bits();
        }

        if let Some(mut head) = self.head {
            head.finish((self.gzip_flags >> 9) & 1);
        }

        // Header CRC accumulation ends here; initialize data CRC state.
        if self.wrap & 4 != 0 && self.gzip_flags != 0 {
            self.crc_fold = Crc32Fold::new();
            self.checksum = crate::CRC32_INITIAL_VALUE;
        }
        Ok(Mode::Type)
    }

    fn dispatch(
        &mut self,
        bit_reader: &mut BitReader<'_>,
        output: &mut InflateOutput<'_>,
        flush: InflateFlush,
    ) -> ReturnCode {
        let mode = self.mode;
        match self.dispatch_inner(bit_reader, output, flush, mode) {
            Ok((mode, code)) | Err(DispatchExit { mode, code }) => {
                self.mode = mode;
                code
            }
        }
    }

    /// Decode the next DEFLATE block header. The final-block marker and
    /// Huffman table selection belong to this state transition, not its caller.
    fn decode_block_type(&mut self, bit_reader: &mut BitReader<'_>) -> Result<Mode, DispatchExit> {
        if self.flags.contains(Flags::IS_LAST_BLOCK) {
            bit_reader.next_byte_boundary();
            return Ok(Mode::Check);
        }

        need_bits_for_mode(bit_reader, 3, Mode::TypeDo)?;
        let last = bit_reader.bits(1) != 0;
        self.flags.update(Flags::IS_LAST_BLOCK, last);
        bit_reader.drop_bits(1);
        let block_type = bit_reader.bits(2);
        bit_reader.drop_bits(2);

        match block_type {
            0b00 => {
                traceln!("inflate:     stored block (last = {last})");
                Ok(Mode::Stored)
            }
            0b01 => {
                traceln!("inflate:     fixed codes block (last = {last})");
                self.len_table = Table {
                    codes: Codes::Fixed,
                    bits: 9,
                };
                self.dist_table = Table {
                    codes: Codes::Fixed,
                    bits: 5,
                };
                Ok(Mode::Len_)
            }
            0b10 => {
                traceln!("inflate:     dynamic codes block (last = {last})");
                Ok(Mode::Table)
            }
            0b11 => {
                traceln!("inflate:     invalid block type");
                Err(DispatchExit {
                    mode: Mode::Bad,
                    code: self.bad("invalid block type\0"),
                })
            }
            _ => unreachable!("two buffered bits yield only values 0..=3"),
        }
    }

    /// Verify the trailer checksum only after its complete 32-bit field is
    /// available. Account for the current output exactly once on resume.
    fn decode_data_checksum(
        &mut self,
        bit_reader: &mut BitReader<'_>,
        output: &mut InflateOutput<'_>,
    ) -> Result<Mode, DispatchExit> {
        if self.wrap != 0 {
            need_bits_for_mode(bit_reader, 32, Mode::Check)?;
            let verify_checksum =
                self.wrap & 4 != 0 && !cfg!(feature = "__internal-fuzz-disable-checksum");
            // Even without verification, the trailer and output count must advance.
            self.total = self.total.wrapping_add(output.len());

            if verify_checksum {
                if self.gzip_flags != 0 {
                    self.crc_fold.fold(output.filled());
                    self.checksum = self.crc_fold.finish();
                } else {
                    self.checksum = adler32(self.checksum, output.filled());
                }
            }

            let given_checksum = if self.gzip_flags != 0 {
                bit_reader.bits_u32(32)
            } else {
                zswap32(bit_reader.bits_u32(32))
            };
            output.mark_accounted();

            if verify_checksum && given_checksum != self.checksum {
                return Err(DispatchExit {
                    mode: Mode::Bad,
                    code: self.bad("incorrect data check\0"),
                });
            }
            bit_reader.init_bits();
        }
        Ok(Mode::Length)
    }

    /// Complete the dynamic literal/length and distance alphabets. Input
    /// starvation preserves the CodeLens mode and the partial symbol count.
    fn decode_code_lengths(&mut self, bit_reader: &mut BitReader<'_>) -> Result<(), DispatchExit> {
        while self.have < self.nlen + self.ndist {
            let here = loop {
                let bits = bit_reader.bits_usize(self.len_table.bits);
                let here = self.len_table_get(bits);
                if here.bits <= bit_reader.bits_in_buffer() {
                    break here;
                }

                pull_byte_for_mode(bit_reader, Mode::CodeLens)?;
            };

            let here_bits = here.bits;
            match here.val {
                0..=15 => {
                    bit_reader.drop_bits(here_bits);
                    self.lens[self.have] = here.val;
                    self.have += 1;
                }
                symbol => {
                    let Some(repeat) = CodeLengthRepeat::from_symbol(symbol) else {
                        return Err(DispatchExit {
                            mode: Mode::Bad,
                            code: self.bad("invalid bit length repeat\0"),
                        });
                    };
                    need_bits_for_mode(
                        bit_reader,
                        usize::from(here_bits) + usize::from(repeat.extra_bits),
                        Mode::CodeLens,
                    )?;
                    bit_reader.drop_bits(here_bits);
                    let extra = bit_reader.bits(usize::from(repeat.extra_bits));
                    bit_reader.drop_bits(repeat.extra_bits);
                    if let Err(msg) = self.append_code_length_repeat(repeat, extra) {
                        return Err(DispatchExit {
                            mode: Mode::Bad,
                            code: self.bad(msg),
                        });
                    }
                }
            }
        }

        if let Err(msg) = self.finish_dynamic_trees() {
            return Err(DispatchExit {
                mode: Mode::Bad,
                code: self.bad(msg),
            });
        }
        Ok(())
    }

    fn dispatch_inner(
        &mut self,
        bit_reader: &mut BitReader<'_>,
        output: &mut InflateOutput<'_>,
        flush: InflateFlush,
        mut mode: Mode,
    ) -> Result<(Mode, ReturnCode), DispatchExit> {
        let ret = 'label: loop {
            mode = 'blk: {
                match mode {
                    Mode::Head => break 'blk self.decode_wrapper_head(bit_reader)?,
                    Mode::Flags => break 'blk self.decode_gzip_flags(bit_reader)?,
                    Mode::Time => break 'blk self.decode_gzip_time(bit_reader)?,
                    Mode::Os => break 'blk self.decode_gzip_os(bit_reader)?,
                    Mode::ExLen => break 'blk self.decode_gzip_extra_length(bit_reader)?,
                    Mode::Extra => match self.decode_gzip_extra(bit_reader)? {
                        Some(next) => break 'blk next,
                        None => break 'label ReturnCode::Ok,
                    },
                    Mode::Name => {
                        match self.decode_gzip_string(bit_reader, GzipStringField::Name) {
                            Some(next) => break 'blk next,
                            None => break 'label ReturnCode::Ok,
                        }
                    }
                    Mode::Comment => {
                        match self.decode_gzip_string(bit_reader, GzipStringField::Comment) {
                            Some(next) => break 'blk next,
                            None => break 'label ReturnCode::Ok,
                        }
                    }
                    Mode::HCrc => break 'blk self.decode_gzip_header_crc(bit_reader)?,
                    Mode::Type => {
                        use InflateFlush::*;

                        match flush {
                            Block | Trees => break 'label ReturnCode::Ok,
                            NoFlush | SyncFlush | Finish => {
                                // NOTE: this is slightly different to what zlib-rs does!
                                break 'blk Mode::TypeDo;
                            }
                        }
                    }
                    Mode::TypeDo => {
                        let next = self.decode_block_type(bit_reader)?;
                        if flush == InflateFlush::Trees && matches!(next, Mode::Len_) {
                            mode = next;
                            break 'label ReturnCode::Ok;
                        }
                        break 'blk next;
                    }
                    Mode::Stored => {
                        bit_reader.next_byte_boundary();

                        need_bits_for_mode(bit_reader, 32, mode)?;

                        let hold = bit_reader.bits_u32(32);

                        traceln!("hold {hold:#x}");

                        let Some(length) = stored_block_length(hold) else {
                            mode = Mode::Bad;
                            break 'label self.bad("invalid stored block lengths\0");
                        };

                        self.length = usize::from(length);
                        traceln!("inflate:     stored length {}", self.length);

                        bit_reader.init_bits();

                        if flush == InflateFlush::Trees {
                            // Z_TREES stops after the validated LEN/NLEN header,
                            // so the next call must resume at the first stored byte.
                            mode = Mode::CopyBlock;
                            break 'label ReturnCode::Ok;
                        }
                        break 'blk Mode::CopyBlock;
                    }
                    Mode::CopyBlock => {
                        loop {
                            let mut copy = self.length;

                            if copy == 0 {
                                break;
                            }

                            copy = Ord::min(copy, output.remaining());
                            copy = Ord::min(copy, bit_reader.bytes_remaining());

                            if copy == 0 {
                                break 'label ReturnCode::Ok;
                            }

                            output.extend(&bit_reader.as_slice()[..copy]);
                            bit_reader.advance(copy);

                            self.length -= copy;
                        }

                        break 'blk Mode::Type;
                    }
                    Mode::Check => {
                        break 'blk self.decode_data_checksum(bit_reader, output)?;
                    }
                    Mode::Len_ => {
                        break 'blk Mode::Len;
                    }
                    Mode::Len => {
                        self.mode = mode;
                        let val = self.len_and_friends(bit_reader, output);
                        mode = self.mode;
                        match val {
                            ControlFlow::Break(return_code) => break 'label return_code,
                            ControlFlow::Continue(()) => continue 'label,
                        }
                    }
                    Mode::LenExt => {
                        if let Err(code) = consume_extra_bits(
                            bit_reader,
                            self.extra,
                            &mut self.length,
                            &mut self.back,
                        ) {
                            return Err(DispatchExit {
                                mode: Mode::LenExt,
                                code,
                            });
                        }
                        traceln!("inflate: length {}", self.length);
                        self.was = self.length;
                        break 'blk Mode::Dist;
                    }
                    Mode::Lit => {
                        if !emit_literal(output, self.length as u8) {
                            traceln!("Ok: writer is full ({} bytes)", output.writer.capacity());
                            break 'label ReturnCode::Ok;
                        }
                        break 'blk Mode::Len;
                    }
                    Mode::Dist => {
                        let decoded = {
                            let table = self.dist_table_ref();
                            decode_distance_code(bit_reader, table, self.dist_table.bits)
                        };
                        match decoded {
                            Ok(decoded) => {
                                self.back += decoded.back_bits;
                                self.offset = decoded.offset;
                                self.extra = decoded.extra;
                                break 'blk Mode::DistExt;
                            }
                            Err(DistanceDecodeError::Input(code)) => {
                                return Err(DispatchExit {
                                    mode: Mode::Dist,
                                    code,
                                });
                            }
                            Err(DistanceDecodeError::InvalidCode) => {
                                mode = Mode::Bad;
                                break 'label self.bad("invalid distance code\0");
                            }
                        }
                    }
                    Mode::DistExt => {
                        if let Err(code) = consume_extra_bits(
                            bit_reader,
                            self.extra,
                            &mut self.offset,
                            &mut self.back,
                        ) {
                            return Err(DispatchExit {
                                mode: Mode::DistExt,
                                code,
                            });
                        }
                        if INFLATE_STRICT && self.offset > self.dmax {
                            mode = Mode::Bad;
                            break 'label self.bad("invalid distance code too far back\0");
                        }
                        traceln!("inflate: distance {}", self.offset);
                        break 'blk Mode::Match;
                    }
                    Mode::Match => 'match_: loop {
                        if output.is_full() {
                            traceln!(
                                "BufError: writer is full ({} bytes)",
                                output.writer.capacity()
                            );
                            break 'label ReturnCode::Ok;
                        }
                        match copy_match_bytes(
                            &self.window,
                            self.flags.contains(Flags::SANE),
                            self.offset,
                            &mut self.length,
                            output,
                        ) {
                            Ok(true) => break 'blk Mode::Len,
                            Ok(false) => continue 'match_,
                            Err(msg) => {
                                mode = Mode::Bad;
                                break 'label self.bad(msg);
                            }
                        }
                    },
                    Mode::Table => {
                        need_bits_for_mode(bit_reader, 14, mode)?;
                        self.nlen = bit_reader.bits_usize(5) + 257;
                        bit_reader.drop_bits(5);
                        self.ndist = bit_reader.bits_usize(5) + 1;
                        bit_reader.drop_bits(5);
                        self.ncode = bit_reader.bits_usize(4) + 4;
                        bit_reader.drop_bits(4);

                        // TODO pkzit_bug_workaround
                        if self.nlen > 286 || self.ndist > 30 {
                            mode = Mode::Bad;
                            break 'label self.bad("too many length or distance symbols\0");
                        }

                        self.have = 0;

                        break 'blk Mode::LenLens;
                    }
                    Mode::LenLens => {
                        while self.have < self.ncode {
                            need_bits_for_mode(bit_reader, 3, mode)?;
                            self.append_code_length_alphabet(bit_reader.bits_u16(3));
                            bit_reader.drop_bits(3);
                        }

                        if let Err(msg) = self.finish_code_length_alphabet() {
                            mode = Mode::Bad;
                            break 'label self.bad(msg);
                        }
                        break 'blk Mode::CodeLens;
                    }
                    Mode::CodeLens => {
                        self.decode_code_lengths(bit_reader)?;
                        mode = Mode::Len_;
                        if flush == InflateFlush::Trees {
                            break 'label ReturnCode::Ok;
                        }
                        break 'blk Mode::Len_;
                    }
                    Mode::Dict => {
                        if !self.flags.contains(Flags::HAVE_DICT) {
                            break 'label ReturnCode::NeedDict;
                        }

                        self.checksum = crate::ADLER32_INITIAL_VALUE;

                        break 'blk Mode::Type;
                    }
                    Mode::DictId => {
                        need_bits_for_mode(bit_reader, 32, mode)?;

                        self.checksum = zswap32(bit_reader.bits_u32(32));

                        bit_reader.init_bits();

                        break 'blk Mode::Dict;
                    }
                    Mode::Done => {
                        // Inflate stream terminated properly.
                        break 'label ReturnCode::StreamEnd;
                    }
                    Mode::Bad => {
                        let msg = "repeated call with bad state\0";
                        self.error_message = Some(msg);

                        break 'label ReturnCode::DataError;
                    }
                    Mode::Mem => {
                        break 'label ReturnCode::MemError;
                    }
                    Mode::Sync => {
                        break 'label ReturnCode::StreamError;
                    }
                    Mode::Length => {
                        // for gzip, last bytes contain LENGTH
                        if self.wrap != 0 && self.gzip_flags != 0 {
                            need_bits_for_mode(bit_reader, 32, mode)?;
                            if (self.wrap & 0b100) != 0
                                && bit_reader.bits_u32(32) != gzip_isize(self.total)
                            {
                                mode = Mode::Bad;
                                break 'label self.bad("incorrect length check\0");
                            }

                            bit_reader.init_bits();
                        }

                        mode = Mode::Done;
                        // Inflate stream terminated properly.
                        break 'label ReturnCode::StreamEnd;
                    }
                };
            }
        };

        Ok((mode, ret))
    }

    const fn bad(&mut self, msg: &'static str) -> ReturnCode {
        self.error_message = Some(msg);
        ReturnCode::DataError
    }

    /// Stored in the `z_stream.data_type` field
    fn decoding_state(&self) -> i32 {
        let bit_reader_bits = i32::from(self.bit_buffer.bits_in_buffer());
        debug_assert!(bit_reader_bits < 64);

        let last = if self.flags.contains(Flags::IS_LAST_BLOCK) {
            64
        } else {
            0
        };

        let mode = match self.mode {
            Mode::Type => 128,
            Mode::Len_ | Mode::CopyBlock => 256,
            _ => 0,
        };

        bit_reader_bits | last | mode
    }
}

/// Compile-time output policy for the shared DEFLATE fast decoder.
///
/// Ordinary inflate may consult circular history, while inflateBack uses the
/// caller's previous full output window. Generic dispatch specializes the hot loop.
trait FastDecodeOutput {
    fn push(&mut self, byte: u8);
    fn remaining(&self) -> usize;
    fn copy_back_reference(
        &mut self,
        state: &State<'_>,
        distance: usize,
        length: usize,
    ) -> Result<(), &'static str>;
}

impl FastDecodeOutput for InflateOutput<'_> {
    #[inline]
    fn push(&mut self, byte: u8) {
        self.writer.push(byte);
    }

    #[inline]
    fn remaining(&self) -> usize {
        self.writer.remaining()
    }

    #[inline]
    fn copy_back_reference(
        &mut self,
        state: &State<'_>,
        dist: usize,
        mut len: usize,
    ) -> Result<(), &'static str> {
        let window_size = state.window.size();
        // max distance in output
        let written = self.len();

        if dist > written {
            // Copy from the circular history window
            if (dist - written) > state.window.have() {
                if state.flags.contains(Flags::SANE) {
                    return Err("invalid distance too far back\0");
                }

                // This cold branch follows the same initialized-history
                // reconstruction as the ordinary Match decoder. Fast inflate
                // reserves sufficient output for a whole DEFLATE match.
                let mut remaining = len;
                while remaining != 0 {
                    copy_match_bytes(&state.window, false, dist, &mut remaining, self)?;
                }
                return Ok(());
            }

            let mut op = dist - written;
            let mut from;

            let window_next = state.window.next();

            if window_next == 0 {
                // This case is hit when the window has just wrapped around
                // by logic in `Window::extend`. It is special-cased because
                // apparently this is quite common.
                //
                // the match is at the end of the window, even though the next
                // position has now wrapped around.
                from = window_size - op;
            } else if window_next >= op {
                // the standard case: a contiguous copy from the window, no wrapping
                from = window_next - op;
            } else {
                // This case is hit when the window has recently wrapped around
                // by logic in `Window::extend`.
                //
                // The match is (partially) at the end of the window
                op -= window_next;
                from = window_size - op;

                if op < len {
                    // This case is hit when part of the match is at the end of the
                    // window, and part of it has wrapped around to the start. Copy
                    // the end section here, the start section will be copied below.
                    len -= op;
                    self.extend_from_window(&state.window, from..from + op);
                    from = 0;
                    op = window_next;
                }
            }

            let copy = Ord::min(op, len);
            self.extend_from_window(&state.window, from..from + copy);

            if op < len {
                // here we need some bytes from the output itself
                self.copy_match(dist, len - op);
            }
        } else {
            self.copy_match(dist, len);
        }
        Ok(())
    }
}

impl FastDecodeOutput for writer::BackWriter<'_> {
    #[inline]
    fn push(&mut self, byte: u8) {
        writer::BackWriter::push(self, byte);
    }

    #[inline]
    fn remaining(&self) -> usize {
        writer::BackWriter::remaining(self)
    }

    #[inline]
    fn copy_back_reference(
        &mut self,
        state: &State<'_>,
        distance: usize,
        length: usize,
    ) -> Result<(), &'static str> {
        self.copy_match(state.flags.contains(Flags::SANE), distance, length)
    }
}

/// Fast literal/length/distance decoder.
///
/// Callers enter only when at least `INFLATE_FAST_MIN_HAVE` bytes and
/// `INFLATE_FAST_MIN_LEFT` output bytes are available. The helper retains those fast-path
/// invariants with assertions in the checked reader/output primitives rather than an unsafe
/// caller contract.
#[inline]
fn inflate_fast_help_impl<const COUNT_BUFFERED_INPUT: bool, O: FastDecodeOutput>(
    state: &mut State,
    bit_reader: &mut BitReader<'_>,
    output: &mut O,
) {
    debug_assert!(bit_reader.bytes_remaining() >= 15);

    let lcode = state.len_table_ref();
    let dcode = state.dist_table_ref();

    // Validate table widths once per decoder invocation; retain bounded
    // u16 masks for the entire hot loop, avoiding per-symbol validation.
    let (Some(lmask), Some(dmask)) = (
        BitBuffer::huffman_mask(state.len_table.bits),
        BitBuffer::huffman_mask(state.dist_table.bits),
    ) else {
        state.mode = Mode::Bad;
        state.error_message = Some("invalid Huffman decode table width\0");
        return;
    };

    let mut bad = None;

    if bit_reader.bits_in_buffer() < 10 {
        debug_assert!(bit_reader.bytes_remaining() >= 15);
        // The fast-path entry invariant guarantees at least eight bytes remain for `refill`.
        bit_reader.refill();
    }
    // We had at least 15 bytes in the slice, plus whatever was in the buffer. After filling the
    // buffer from the slice, we now have at least 8 bytes remaining in the slice, plus a full buffer.
    debug_assert!(
        bit_reader.bytes_remaining() >= 8 && bit_reader.bytes_remaining_including_buffer() >= 15
    );

    'outer: loop {
        // This condition is ensured above for the first iteration of the `outer` loop. For
        // subsequent iterations, the loop continuation condition is
        // `bit_reader.bytes_remaining_including_buffer() > 15`. And because the buffer
        // contributes at most 7 bytes to the result of bit_reader.bytes_remaining_including_buffer(),
        // that means that the slice contains at least 8 bytes.
        debug_assert!(
            bit_reader.bytes_remaining() >= 8
                && bit_reader.bytes_remaining_including_buffer() >= 15
        );

        let mut here = {
            let bits = bit_reader.bits_in_buffer();
            let hold = bit_reader.hold();

            // Safety: As described in the comments for the debug_assert at the start of
            // the `outer` loop, it is guaranteed that `bit_reader.bytes_remaining() >= 8` here,
            // which satisfies `refill`'s eight-byte requirement. And, because the total
            // number of bytes in `bit_reader`'s buffer plus its slice is at least 15, and
            // `refill` moves at most 7 bytes from the slice to the buffer, the slice will still
            // contain at least 8 bytes after this `refill` call.
            bit_reader.refill();
            // After the refill, there will be at least 8 bytes left in the bit_reader's slice.
            debug_assert!(bit_reader.bytes_remaining() >= 8);

            // in most cases, the read can be interleaved with the logic
            // based on benchmarks this matters in practice. wild.
            if bits as usize >= state.len_table.bits {
                lcode[BitBuffer::lookup_index(hold, lmask)]
            } else {
                lcode[BitBuffer::lookup_index(bit_reader.hold(), lmask)]
            }
        };

        if here.op == 0 {
            output.push(here.val as u8);
            bit_reader.drop_bits(here.bits);
            here = lcode[BitBuffer::lookup_index(bit_reader.hold(), lmask)];

            if here.op == 0 {
                output.push(here.val as u8);
                bit_reader.drop_bits(here.bits);
                here = lcode[BitBuffer::lookup_index(bit_reader.hold(), lmask)];
            }
        }

        'dolen: loop {
            bit_reader.drop_bits(here.bits);
            let op = here.op;

            if op == 0 {
                output.push(here.val as u8);
            } else if op & 16 != 0 {
                let op = op & MAX_BITS;
                let len = here.val + bit_reader.bits_u16(usize::from(op));
                bit_reader.drop_bits(op);

                here = dcode[BitBuffer::lookup_index(bit_reader.hold(), dmask)];

                // we have two fast-path loads: 10+10 + 15+5 = 40,
                // but we may need to refill here in the worst case
                if bit_reader.bits_in_buffer() < MAX_BITS + MAX_DIST_EXTRA_BITS {
                    debug_assert!(bit_reader.bytes_remaining() >= 8);
                    // Safety: On the first iteration of the `dolen` loop, we can rely on the
                    // invariant documented for the previous `refill` call above: after that
                    // operation, `bit_reader.bytes_remining >= 8`, which satisfies the safety
                    // requirement for this call. For subsequent iterations, this invariant
                    // remains true because nothing else within the `dolen` loop consumes data
                    // from the slice.
                    bit_reader.refill();
                }

                'dodist: loop {
                    bit_reader.drop_bits(here.bits);
                    let op = here.op;

                    if op & 16 != 0 {
                        let op = op & MAX_BITS;
                        let dist = here.val + bit_reader.bits_u16(usize::from(op));

                        if INFLATE_STRICT && dist as usize > state.dmax {
                            bad = Some("invalid distance too far back\0");
                            state.mode = Mode::Bad;
                            break 'outer;
                        }

                        bit_reader.drop_bits(op);

                        if let Err(msg) =
                            output.copy_back_reference(state, usize::from(dist), usize::from(len))
                        {
                            bad = Some(msg);
                            state.mode = Mode::Bad;
                            break 'outer;
                        }
                    } else if (op & 64) == 0 {
                        // 2nd level distance code
                        here = dcode[usize::from(here.val + bit_reader.bits_u16(usize::from(op)))];
                        continue 'dodist;
                    } else {
                        bad = Some("invalid distance code\0");
                        state.mode = Mode::Bad;
                        break 'outer;
                    }

                    break 'dodist;
                }
            } else if (op & 64) == 0 {
                // 2nd level length code
                here = lcode[usize::from(here.val + bit_reader.bits_u16(usize::from(op)))];
                continue 'dolen;
            } else if op & 32 != 0 {
                // end of block
                state.mode = Mode::Type;
                break 'outer;
            } else {
                bad = Some("invalid literal/length code\0");
                state.mode = Mode::Bad;
                break 'outer;
            }

            break 'dolen;
        }

        // For normal `inflate`, include the bits in the bit_reader buffer in the count of available bytes.
        // Ordinary inflate considers retained bytes in the bit reservoir for
        // its fast-loop threshold, while the callback decoder counts only the
        // unread callback span. Specialize that input policy at compile time.
        let remaining = if COUNT_BUFFERED_INPUT {
            bit_reader.bytes_remaining_including_buffer()
        } else {
            bit_reader.bytes_remaining()
        };
        if remaining >= INFLATE_FAST_MIN_HAVE && output.remaining() >= INFLATE_FAST_MIN_LEFT {
            continue;
        }

        break 'outer;
    }

    // return unused bytes (on entry, bits < 8, so in won't go too far back)
    bit_reader.return_unused_bytes();

    if let Some(error_message) = bad {
        debug_assert!(matches!(state.mode, Mode::Bad));
        state.bad(error_message);
    }
}

pub fn prime(stream: &mut InflateStream, bits: i32, value: i32) -> ReturnCode {
    if bits < 0 {
        stream.state.bit_buffer.init_bits();
    } else if bits != 0 {
        let Ok(width) = u8::try_from(bits) else {
            return ReturnCode::StreamError;
        };
        if width > 16
            || usize::from(stream.state.bit_buffer.bits_in_buffer()) + usize::from(width) > 32
        {
            return ReturnCode::StreamError;
        }
        // Preserve the signed C argument's bit pattern before extending it
        // to the unsigned bit-reservoir representation.
        stream
            .state
            .bit_buffer
            .prime(width, u64::from(value.cast_unsigned()));
    }

    ReturnCode::Ok
}

struct InflateAllocOffsets {
    total_size: usize,
    state_pos: usize,
    window_pos: Option<usize>,
}

impl InflateAllocOffsets {
    const ALIGN_SIZE: usize = 64;

    fn with_window() -> Self {
        Self::new(true)
    }

    fn state_only() -> Self {
        Self::new(false)
    }

    fn new(with_window: bool) -> Self {
        use core::mem::size_of;

        let mut curr_size = 0usize;
        let state_pos = curr_size.next_multiple_of(Self::ALIGN_SIZE);
        curr_size = state_pos + size_of::<State>();

        let window_pos = with_window.then(|| {
            let pos = curr_size.next_multiple_of(Self::ALIGN_SIZE);
            curr_size = pos + (1 << MAX_WBITS);
            pos
        });

        // Reserve enough headroom to align the allocation start to 64 bytes, then round the whole
        // allocation up to the same granularity.
        let total_size = (curr_size + (Self::ALIGN_SIZE - 1)).next_multiple_of(Self::ALIGN_SIZE);

        Self {
            total_size,
            state_pos,
            window_pos,
        }
    }

    const fn window_pos(&self) -> usize {
        self.window_pos
            .expect("inflate allocation layout includes an owned window")
    }

    /// Allocate raw storage and return both the original allocation pointer (needed for freeing)
    /// and its 64-byte-aligned interior base used by this layout.
    fn allocate_storage<'a>(&self, alloc: &Allocator<'a>) -> Option<(NonNull<u8>, NonNull<u8>)> {
        let allocation_start = alloc.allocate_slice_raw::<u8>(self.total_size)?;
        let address = allocation_start.as_ptr() as usize;
        let align_offset = address.next_multiple_of(Self::ALIGN_SIZE) - address;
        // SAFETY: `total_size` reserves `ALIGN_SIZE - 1` bytes of alignment headroom, and offsetting
        // a non-null allocation pointer within that allocation cannot produce null.
        let aligned =
            unsafe { NonNull::new_unchecked(allocation_start.as_ptr().add(align_offset)) };
        Some((allocation_start, aligned))
    }

    /// Allocate and place one inflate state using this layout.
    ///
    /// For the ordinary layout this also creates the owned history window. For the
    /// state-only layout (used by `inflateBack`) any caller-owned back window already stored in
    /// `state` is preserved. Raw alignment, placement, and allocation metadata stay confined here.
    fn allocate_state<'a>(
        &self,
        alloc: &Allocator<'a>,
        mut state: State<'a>,
    ) -> Option<NonNull<State<'a>>> {
        let (allocation_start, aligned) = self.allocate_storage(alloc)?;

        if let Some(window_pos) = self.window_pos {
            // SAFETY: `window_pos` is computed inside this allocation and leaves enough room for
            // the complete regular history window. The backing bytes begin uninitialized; Window
            // exposes only history proven initialized by `extend`.
            let window_allocation = unsafe { aligned.as_ptr().add(window_pos) };
            state.window = unsafe { Window::from_raw_parts(window_allocation, 1 << MAX_WBITS) };
        }

        state.allocation_start = allocation_start.as_ptr();
        state.total_allocation_size = self.total_size;

        // SAFETY: `state_pos` is aligned for `State`, lies inside this fresh allocation, and no
        // `State` has previously been materialized there.
        let state_ptr = unsafe { aligned.as_ptr().add(self.state_pos).cast::<State<'a>>() };
        unsafe { state_ptr.write(state) };
        NonNull::new(state_ptr)
    }

    /// Snapshot a live regular inflate state into independent state + window storage.
    fn clone_state<'a>(
        &self,
        alloc: &Allocator<'a>,
        source: &State<'a>,
    ) -> Option<NonNull<State<'a>>> {
        debug_assert_eq!(source.kind, InflateStateKind::Regular);
        debug_assert_eq!(self.total_size, source.total_allocation_size);
        debug_assert!(source.back_window.is_empty());

        // A regular inflate state always owns the canonical MAX_WBITS backing window. This is a
        // memory-safety invariant for the raw copy below, so keep it as a release assertion rather
        // than a debug-only check.
        assert_eq!(
            source.window.size(),
            1 << MAX_WBITS,
            "regular inflate state owns the canonical history window"
        );

        let (allocation_start, aligned) = self.allocate_storage(alloc)?;
        let window_pos = self.window_pos();
        // SAFETY: the ordinary layout reserves exactly the source window backing length at
        // `window_pos`, and the fresh destination allocation cannot overlap the source.
        let window_allocation = unsafe { aligned.as_ptr().add(window_pos) };
        let window = unsafe { source.window.clone_to(window_allocation) };

        let state = source.clone_rebound(window, allocation_start.as_ptr(), self.total_size);

        // SAFETY: `state_pos` is aligned and in-bounds in this fresh allocation, and no State has
        // previously been materialized there.
        let state_ptr = unsafe { aligned.as_ptr().add(self.state_pos).cast::<State<'a>>() };
        unsafe { state_ptr.write(state) };
        NonNull::new(state_ptr)
    }
}

/// Configuration for decompresssion.
///
/// Used with [`decompress_slice`].
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct InflateConfig {
    pub window_bits: i32,
}

impl Default for InflateConfig {
    fn default() -> Self {
        Self {
            window_bits: DEF_WBITS,
        }
    }
}

/// Validated DEFLATE window selection, including the zlib wrapper mode.
/// Four wrapper modes exist: raw (0), zlib (5), gzip (6),
/// and zlib/gzip autodetection (7). The gzip and autodetect flag forms
/// each encode their window size in the low four bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct InflateWindowSelection {
    wrap: u8,
    wbits: u8,
}

fn parse_inflate_window_bits(bits: i32) -> Option<InflateWindowSelection> {
    let (wrap, width) = match bits {
        -15..=-8 => (0, -bits),
        0 | 8..=15 => (5, bits),
        24..=31 => (6, bits - 16),
        // 32 is zlib's autodetect form with window size inferred from the header.
        32 | 40..=47 => (7, bits - 32),
        _ => return None,
    };
    Some(InflateWindowSelection {
        wrap,
        wbits: u8::try_from(width).ok()?,
    })
}

/// Initialize the stream in an inflate state
pub fn init(stream: &mut z_stream, config: InflateConfig) -> ReturnCode {
    stream.msg = core::ptr::null_mut();

    // Reject invalid formats before selecting allocators or allocating state.
    if parse_inflate_window_bits(config.window_bits).is_none() {
        return ReturnCode::StreamError;
    }

    // for safety we must really make sure that alloc and free are consistent
    // this is a (slight) deviation from stock zlib. In this crate we pick the rust
    // allocator as the default, but `libz-rs-sys` configures the C allocator
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

    let state = State::new(InflateStateKind::Regular);

    let alloc = Allocator {
        zalloc,
        zfree,
        opaque: stream.opaque,
        _marker: PhantomData,
    };
    let allocs = InflateAllocOffsets::with_window();
    let Some(state_allocation) = allocs.allocate_state(&alloc, state) else {
        return ReturnCode::MemError;
    };
    stream.state = state_allocation.as_ptr().cast::<internal_state>();

    // SAFETY: `allocate_state` initialized a regular `State` behind the ABI state pointer.
    if let Some(stream) = unsafe { InflateStream::from_stream_mut(stream) } {
        let ret = reset_with_config(stream, config);

        if ret != ReturnCode::Ok {
            end(stream);
        }

        ret
    } else {
        ReturnCode::StreamError
    }
}

pub fn reset_with_config(stream: &mut InflateStream, config: InflateConfig) -> ReturnCode {
    let Some(window) = parse_inflate_window_bits(config.window_bits) else {
        return ReturnCode::StreamError;
    };

    // Do not mutate either field until both the wrapper kind and its
    // encoded history-window width have been fully validated.
    stream.state.wrap = window.wrap;
    stream.state.wbits = window.wbits;

    reset(stream)
}

pub fn reset(stream: &mut InflateStream) -> ReturnCode {
    // reset the state of the window
    stream.state.window.clear();

    stream.state.error_message = None;

    reset_keep(stream)
}

pub fn reset_keep(stream: &mut InflateStream) -> ReturnCode {
    stream.total_in = 0;
    stream.total_out = 0;
    stream.state.total = 0;

    stream.msg = core::ptr::null_mut();

    let state = &mut *stream.state;

    if state.wrap != 0 {
        // to support ill-conceived Java test suite
        stream.adler = (state.wrap & 1).into();
    }

    state.mode = Mode::Head;
    state.checksum = crate::ADLER32_INITIAL_VALUE;

    state.flags.update(Flags::IS_LAST_BLOCK, false);
    state.flags.update(Flags::HAVE_DICT, false);
    state.flags.update(Flags::SANE, true);
    state.gzip_flags = -1;
    state.dmax = 32768;
    state.head = None;
    state.bit_buffer = BitBuffer::default();

    state.next = 0;
    state.len_table = Table::default();
    state.dist_table = Table::default();

    state.back = usize::MAX;

    ReturnCode::Ok
}

pub fn codes_used(stream: &InflateStream) -> usize {
    stream.state.next
}

/// Bind a caller-owned input span; null-and-empty is a valid C representation.
///
/// # Safety
/// A nonnull pointer with a nonzero length must remain readable for the
/// returned borrow, whose lifetime belongs to the caller.
const unsafe fn caller_input<'a>(
    pointer: *const u8,
    available: crate::c_api::uInt,
) -> Option<&'a [u8]> {
    if available == 0 {
        return Some(&[]);
    }
    if pointer.is_null() {
        return None;
    }
    // SAFETY: the caller supplies readable input of the advertised length.
    Some(unsafe { core::slice::from_raw_parts(pointer, available as usize) })
}

/// Bind one zlib ABI input/output pair and run ordinary inflate.
///
/// # Safety
/// Any non-null input span must be readable for `avail_in` bytes and any non-null output span must
/// be uniquely writable for `avail_out` bytes. The input/output spans must not overlap each other
/// or the live stream/state objects. Null input with zero length is valid; null output is rejected
/// before it is dereferenced.
pub unsafe fn inflate(stream: &mut InflateStream, flush: InflateFlush) -> ReturnCode {
    if stream.next_out.is_null() {
        return ReturnCode::StreamError;
    }
    // SAFETY: the public inflate contract guarantees a live readable input span.
    let Some(input) = (unsafe { caller_input(stream.next_in, stream.avail_in) }) else {
        return ReturnCode::StreamError;
    };
    // SAFETY: the public inflate contract guarantees a live writable output span.
    let output =
        unsafe { InflateOutput::new_uninit(stream.next_out.cast(), stream.avail_out as usize) };

    inflate_bound_io(stream, input, output, flush)
}

/// Safe per-call inflate core after the caller's raw input/output spans have been bound to typed
/// cursors. No raw caller buffer is dereferenced or materialized below this boundary.
fn inflate_bound_io(
    stream: &mut InflateStream,
    input: &[u8],
    mut output: InflateOutput<'_>,
    flush: InflateFlush,
) -> ReturnCode {
    debug_assert_eq!(stream.avail_in as usize, input.len());
    let initial_avail_in = input.len();
    let state = &mut *stream.state;

    // skip check
    if let Mode::Type = state.mode {
        state.mode = Mode::TypeDo;
    }

    let mut bit_reader = BitReader::from_buffer(state.bit_buffer, input);
    let err = state.dispatch(&mut bit_reader, &mut output, flush);

    let in_read = initial_avail_in - bit_reader.bytes_remaining();
    let out_written = output.written_since_accounted();
    // The shared ABI cursor preserves null input on zero consumption and wraps C totals.
    advance_abi_bytes(
        &mut stream.next_in,
        &mut stream.avail_in,
        &mut stream.total_in,
        in_read,
    );
    state.total = state.total.wrapping_add(out_written);
    stream.total_out = state.total as _;

    state.bit_buffer = bit_reader.into_buffer();

    stream.avail_out = output.remaining() as u32;
    stream.next_out = output.next_out() as *mut u8;

    stream.adler = z_checksum::from(state.checksum);

    let valid_mode = |mode| !matches!(mode, Mode::Bad | Mode::Mem | Mode::Sync);
    let not_done = |mode| {
        !matches!(
            mode,
            Mode::Check | Mode::Length | Mode::Bad | Mode::Mem | Mode::Sync
        )
    };

    let must_update_window = state.window.size() != 0
        || (out_written != 0
            && valid_mode(state.mode)
            && (not_done(state.mode) || !matches!(flush, InflateFlush::Finish)));

    let update_checksum =
        state.wrap & 4 != 0 && !cfg!(feature = "__internal-fuzz-disable-checksum");

    if must_update_window {
        state.window.extend(
            &output.filled()[..out_written],
            state.gzip_flags,
            update_checksum,
            &mut state.checksum,
            &mut state.crc_fold,
        );
    }

    if let Some(msg) = state.error_message {
        assert!(msg.ends_with('\0'));
        stream.msg = msg.as_ptr().cast_mut() as *mut core::ffi::c_char;
    }

    stream.data_type = state.decoding_state();

    if ((in_read == 0 && out_written == 0) || flush == InflateFlush::Finish)
        && err == ReturnCode::Ok
    {
        ReturnCode::BufError
    } else {
        err
    }
}

const fn syncsearch(mut got: usize, buf: &[u8]) -> (usize, usize) {
    let len = buf.len();
    let mut next = 0;

    while next < len && got < 4 {
        if buf[next] == if got < 2 { 0 } else { 0xff } {
            got += 1;
        } else if buf[next] != 0 {
            got = 0;
        } else {
            got = 4 - got;
        }
        next += 1;
    }

    (got, next)
}

/// Find a recoverable full-flush marker in the caller's raw input.
///
/// # Safety
/// If `avail_in` is nonzero, `next_in` must point to that many readable bytes.
/// Buffers may be replaced between calls.
pub unsafe fn sync(stream: &mut InflateStream) -> ReturnCode {
    let state = &mut *stream.state;

    // SAFETY: the caller supplies a live readable input span.
    let Some(input) = (unsafe { caller_input(stream.next_in, stream.avail_in) }) else {
        return ReturnCode::StreamError;
    };
    if input.is_empty() && state.bit_buffer.bits_in_buffer() < 8 {
        return ReturnCode::BufError;
    }
    /* if first time, start search in bit buffer */
    if !matches!(state.mode, Mode::Sync) {
        state.mode = Mode::Sync;

        let (buf, len) = state.bit_buffer.start_sync_search();

        (state.have, _) = syncsearch(0, &buf[..len]);
    }

    let len;
    (state.have, len) = syncsearch(state.have, input);
    advance_abi_bytes(
        &mut stream.next_in,
        &mut stream.avail_in,
        &mut stream.total_in,
        len,
    );

    /* return no joy or set up to restart inflate() on a new block */
    if state.have != 4 {
        return ReturnCode::DataError;
    }

    if state.gzip_flags == -1 {
        state.wrap = 0; /* if no header yet, treat as raw */
    } else {
        state.wrap &= !4; /* no point in computing a check value now */
    }

    let flags = state.gzip_flags;
    // The next inflate call reconstructs total_out from this full-width count.
    // reset() clears it, so preserve it alongside the public ABI counters.
    let total = state.total;
    let total_in = stream.total_in;
    let total_out = stream.total_out;

    reset(stream);

    stream.total_in = total_in;
    stream.total_out = total_out;
    stream.state.total = total;

    stream.state.gzip_flags = flags;
    stream.state.mode = Mode::Type;

    ReturnCode::Ok
}

/*
  Returns true if inflate is currently at the end of a block generated by
  Z_SYNC_FLUSH or Z_FULL_FLUSH. This function is used by one PPP
  implementation to provide an additional safety check. PPP uses
  Z_SYNC_FLUSH but removes the length bytes of the resulting empty stored
  block. When decompressing, PPP checks that at the end of input packet,
  inflate is waiting for these length bytes.
*/
pub fn sync_point(stream: &InflateStream) -> bool {
    matches!(stream.state.mode, Mode::Stored) && stream.state.bit_buffer.bits_in_buffer() == 0
}

pub fn copy<'a>(
    dest: &mut MaybeUninit<InflateStream<'a>>,
    source: &InflateStream<'a>,
) -> ReturnCode {
    if source.state.kind != InflateStateKind::Regular {
        return ReturnCode::StreamError;
    }

    // `inflateCopy` snapshots decoder state; zlib does not require live input/output pointers.
    // The allocation helper keeps the transient duplicated raw window handle private and returns
    // only after the copied state owns independent window storage.
    let allocs = InflateAllocOffsets::with_window();
    let Some(copy) = allocs.clone_state(&source.alloc, &source.state) else {
        return ReturnCode::MemError;
    };

    // Build the destination ABI shell explicitly around the cloned state allocation.
    dest.write(InflateStream {
        next_in: source.next_in,
        avail_in: source.avail_in,
        total_in: source.total_in,
        next_out: source.next_out,
        avail_out: source.avail_out,
        total_out: source.total_out,
        msg: source.msg,
        // SAFETY: `clone_state` returns a live initialized `State` in an allocation owned by this
        // destination stream.
        state: unsafe { StatePtr::from_raw(copy.as_ptr()) },
        alloc: source.alloc,
        data_type: source.data_type,
        adler: source.adler,
        reserved: source.reserved,
    });

    ReturnCode::Ok
}

pub fn undermine(stream: &mut InflateStream, subvert: i32) -> ReturnCode {
    stream.state.flags.update(Flags::SANE, subvert == 0);

    ReturnCode::Ok
}

/// Configures whether the checksum is calculated and checked.
pub fn validate(stream: &mut InflateStream, check: bool) -> ReturnCode {
    if check && stream.state.wrap != 0 {
        stream.state.wrap |= 0b100;
    } else {
        stream.state.wrap &= !0b100;
    }

    ReturnCode::Ok
}

#[must_use]
pub fn mark(stream: &InflateStream) -> c_long {
    // inflateMark inspects only the initialized state machine. The caller may
    // legitimately detach or replace input/output buffers between inflate calls.
    let state = &*stream.state;

    let length = match state.mode {
        Mode::CopyBlock => state.length,
        Mode::Match => state.was - state.length,
        _ => 0,
    };

    // The high part is a count of bits back, with usize::MAX representing -1
    // before the first block. Valid code lengths and stored-block lengths fit
    // in the low 16 bits, regardless of the platform's C long width.
    let back = if state.back == usize::MAX {
        -1
    } else {
        let Ok(back) = c_long::try_from(state.back) else {
            return -65_536;
        };
        back
    };
    let Ok(length) = c_long::try_from(length) else {
        return -65_536;
    };
    back.wrapping_shl(16).wrapping_add(length)
}

pub fn set_dictionary(stream: &mut InflateStream, dictionary: &[u8]) -> ReturnCode {
    if stream.state.wrap != 0 && !matches!(stream.state.mode, Mode::Dict) {
        return ReturnCode::StreamError;
    }

    // check for correct dictionary identifier
    if matches!(stream.state.mode, Mode::Dict) {
        let dictid = adler32(1, dictionary);

        if dictid != stream.state.checksum {
            return ReturnCode::DataError;
        }
    }

    let state = &mut *stream.state;
    state.window.extend(
        dictionary,
        state.gzip_flags,
        false,
        &mut state.checksum,
        &mut state.crc_fold,
    );

    state.flags.update(Flags::HAVE_DICT, true);

    ReturnCode::Ok
}

pub fn end<'a>(stream: &'a mut InflateStream<'_>) -> &'a mut z_stream {
    if stream.state.as_ptr().is_null() {
        return stream.as_z_stream_mut();
    }

    let alloc = stream.alloc;
    let allocation_start = stream.state.allocation_start;
    let total_allocation_size = stream.state.total_allocation_size;

    let mut window = Window::empty();
    core::mem::swap(&mut window, &mut stream.state.window);

    let stream = stream.as_z_stream_mut();
    let _ = core::mem::replace(&mut stream.state, core::ptr::null_mut());

    unsafe { alloc.deallocate(allocation_start, total_allocation_size) };

    stream
}

/// # Safety
///
/// The caller must guarantee:
///
/// * If `head` is `Some`, the header remains valid and exclusively writable while this stream or
///   any `inflateCopy` clone may still populate it. The caller must not access the destination
///   concurrently with such an inflate operation, and copied streams sharing it must not inflate
///   concurrently with each other.
///     - If `head.extra` is not NULL, it must be writable for at least `head.extra_max` bytes
///     - if `head.name` is not NULL, it must be writable for at least `head.name_max` bytes
///     - if `head.comment` is not NULL, it must be writable for at least `head.comm_max` bytes
/// * The nested output buffers must not overlap the compressed input span of any inflate call
///   that writes into them: metadata copies use non-overlapping memory semantics.
pub unsafe fn get_header(
    stream: &mut InflateStream<'_>,
    head: Option<&mut gz_header>,
) -> ReturnCode {
    if (stream.state.wrap & 2) == 0 {
        return ReturnCode::StreamError;
    }

    stream.state.head = head.map(|head| {
        head.done = 0;
        // SAFETY: this function's contract establishes the header and nested-buffer invariants.
        unsafe { GzipHeaderOutput::new(head) }
    });
    ReturnCode::Ok
}

/// Return the sliding dictionary as at most two contiguous segments in logical input order.
///
/// Raw caller-buffer handling belongs to the C ABI layer; the inflate core exposes only checked
/// slices over the initialized circular window.
pub fn dictionary<'stream>(stream: &'stream InflateStream<'_>) -> (&'stream [u8], &'stream [u8]) {
    stream.state.window.dictionary_segments()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_decoder_rejects_invalid_table_width_before_consuming_input_or_output() {
        let input = [0xa5u8; 32];
        let mut output_bytes = [core::mem::MaybeUninit::<u8>::uninit(); 260];

        for (len_bits, dist_bits) in [(16, 5), (9, 16), (usize::MAX, 5)] {
            let mut state = State::new(InflateStateKind::Regular);
            state.len_table = Table {
                codes: Codes::Fixed,
                bits: len_bits,
            };
            state.dist_table = Table {
                codes: Codes::Fixed,
                bits: dist_bits,
            };
            let mut reader = BitReader::from_buffer(BitBuffer::default(), &input);
            // SAFETY: the stack array is writable for exactly its declared length;
            // InflateOutput tracks the initialized prefix and no other borrow exists.
            let mut output = unsafe {
                InflateOutput::new_uninit(output_bytes.as_mut_ptr().cast(), output_bytes.len())
            };

            inflate_fast_help_impl::<true, _>(&mut state, &mut reader, &mut output);

            assert!(matches!(state.mode, Mode::Bad));
            assert_eq!(
                state.error_message,
                Some("invalid Huffman decode table width\0")
            );
            assert_eq!(reader.bytes_remaining(), input.len());
            assert_eq!(reader.bits_in_buffer(), 0);
            assert_eq!(output.len(), 0);
        }
    }

    #[test]
    fn stored_block_length_matches_all_65536_wire_lengths() {
        for value in u16::MIN..=u16::MAX {
            let header = u32::from(value) | (u32::from(!value) << 16);
            assert_eq!(stored_block_length(header), Some(value));
            assert_eq!(stored_block_length(header ^ (1 << 16)), None);
            assert_eq!(stored_block_length(header ^ 1), None);
        }
    }

    #[test]
    fn trees_flush_resumes_stored_payload_after_length_header() {
        let input = [3, 0, 0xfc, 0xff, b'a', b'b', b'c'];
        let mut state = State::new(InflateStateKind::Regular);
        state.mode = Mode::Stored;
        let mut reader = BitReader::from_buffer(BitBuffer::default(), &input);
        let mut destination = [0u8; 8];
        // SAFETY: destination is exclusively writable for its full stack allocation;
        // only its initialized prefix is observed before returning.
        let mut output =
            unsafe { InflateOutput::new_uninit(destination.as_mut_ptr(), destination.len()) };

        assert_eq!(
            state.dispatch(&mut reader, &mut output, InflateFlush::Trees),
            ReturnCode::Ok
        );
        assert!(matches!(state.mode, Mode::CopyBlock));
        assert_eq!(state.length, 3);
        assert_eq!(reader.as_slice(), b"abc");
        assert_eq!(output.len(), 0);

        assert_eq!(
            state.dispatch(&mut reader, &mut output, InflateFlush::NoFlush),
            ReturnCode::Ok
        );
        assert_eq!(output.filled(), b"abc");
        assert_eq!(state.length, 0);
    }

    #[test]
    fn trees_flush_stored_payload_resumes_through_stream_api() -> Result<(), &'static str> {
        // A final raw stored block containing abc, delivered in two input spans.
        let mut prefix = [0x01u8];
        let mut payload = [3, 0, 0xfc, 0xff, b'a', b'b', b'c'];
        let mut destination = [0xa5u8; 8];
        let mut raw = z_stream::default();
        assert_eq!(
            init(&mut raw, InflateConfig { window_bits: -15 }),
            ReturnCode::Ok
        );
        // SAFETY: init created an exclusively borrowed regular inflate state.
        let stream = unsafe { InflateStream::from_stream_mut(&raw mut raw) }
            .ok_or("raw inflate stream was not initialized")?;
        stream.next_out = destination.as_mut_ptr();
        stream.avail_out = 8;

        stream.next_in = prefix.as_mut_ptr();
        stream.avail_in = 1;
        // SAFETY: both input spans and output allocation stay live and disjoint.
        assert_eq!(
            unsafe { inflate(stream, InflateFlush::NoFlush) },
            ReturnCode::Ok
        );
        assert!(matches!(stream.state.mode, Mode::Stored));
        assert_eq!(stream.avail_in, 0);

        stream.next_in = payload.as_mut_ptr();
        stream.avail_in = 7;
        // SAFETY: payload and destination are live, disjoint, and lengths are exact.
        assert_eq!(
            unsafe { inflate(stream, InflateFlush::Trees) },
            ReturnCode::Ok
        );
        assert!(matches!(stream.state.mode, Mode::CopyBlock));
        assert_eq!(stream.avail_in, 3);
        assert_eq!(stream.total_out, 0);

        // SAFETY: the ABI pointers now address the valid unconsumed payload suffix.
        assert_eq!(
            unsafe { inflate(stream, InflateFlush::NoFlush) },
            ReturnCode::StreamEnd
        );
        assert_eq!(stream.avail_in, 0);
        assert_eq!(stream.total_out, 3);
        assert_eq!(&destination[..3], b"abc");
        assert_eq!(&destination[3..], &[0xa5; 5]);
        assert!(end(stream).state.is_null());
        Ok(())
    }

    #[test]
    fn gzip_isize_uses_modulo_32_bit_wire_length() {
        assert_eq!(gzip_isize(0), 0);
        assert_eq!(gzip_isize(usize::from(u16::MAX)), u32::from(u16::MAX));
        #[cfg(target_pointer_width = "64")]
        {
            let over_4gib = (u64::from(u32::MAX) + 4097) as usize;
            assert_eq!(gzip_isize(over_4gib), 4096);
        }
    }

    #[test]
    fn dynamic_code_length_runs_validate_without_partial_writes() -> Result<(), &'static str> {
        let mut state = State::new(InflateStateKind::Regular);
        state.nlen = 257;
        state.ndist = 1;
        state.lens[0] = 7;
        state.have = 1;

        let previous = CodeLengthRepeat::from_symbol(16).ok_or("missing previous repeat")?;
        let short_zero = CodeLengthRepeat::from_symbol(17).ok_or("missing short zero repeat")?;
        let long_zero = CodeLengthRepeat::from_symbol(18).ok_or("missing long zero repeat")?;

        state.append_code_length_repeat(previous, 0)?;
        assert_eq!(&state.lens[..4], &[7, 7, 7, 7]);
        state.append_code_length_repeat(previous, 3)?;
        assert_eq!(&state.lens[4..10], &[7; 6]);

        state.append_code_length_repeat(short_zero, 7)?;
        assert_eq!(&state.lens[10..20], &[0; 10]);
        state.append_code_length_repeat(long_zero, 127)?;
        assert_eq!(&state.lens[20..158], &[0; 138]);
        assert_eq!(state.have, 158);

        // Both long and short repeats reject a run crossing the alphabet
        // boundary without changing the preceding decoded prefix or cursor.
        state.have = 257;
        state.lens[257] = 0xa5;
        assert_eq!(
            state.append_code_length_repeat(short_zero, 0),
            Err("invalid bit length repeat\0")
        );
        assert_eq!(
            state.append_code_length_repeat(long_zero, 0),
            Err("invalid bit length repeat\0")
        );
        assert_eq!(state.have, 257);
        assert_eq!(state.lens[257], 0xa5);

        state.have = 0;
        assert_eq!(
            state.append_code_length_repeat(previous, 0),
            Err("invalid bit length repeat\0")
        );
        assert_eq!(state.have, 0);
        assert!(CodeLengthRepeat::from_symbol(19).is_none());
        Ok(())
    }

    #[test]
    fn inflate_prime_keeps_signed_bit_patterns_and_rejects_overflow_without_mutation(
    ) -> Result<(), &'static str> {
        let mut raw = z_stream::default();
        assert_eq!(init(&mut raw, InflateConfig::default()), ReturnCode::Ok);
        // SAFETY: init succeeded and the state is exclusively borrowed for this test.
        let stream = unsafe { InflateStream::from_stream_mut(&raw mut raw) }
            .ok_or("freshly initialized inflate stream was not usable")?;

        for width in [17, 32, 256, i32::MAX] {
            let before = stream.state.bit_buffer;
            assert_eq!(prime(stream, width, -1), ReturnCode::StreamError);
            assert_eq!(stream.state.bit_buffer, before);
        }

        assert_eq!(prime(stream, 16, -1), ReturnCode::Ok);
        assert_eq!(stream.state.bit_buffer.hold(), 0xffff);
        assert_eq!(stream.state.bit_buffer.bits_in_buffer(), 16);

        assert_eq!(prime(stream, 16, i32::MIN), ReturnCode::Ok);
        assert_eq!(stream.state.bit_buffer.hold(), 0xffff);
        assert_eq!(stream.state.bit_buffer.bits_in_buffer(), 32);

        let before = stream.state.bit_buffer;
        assert_eq!(prime(stream, 1, 1), ReturnCode::StreamError);
        assert_eq!(stream.state.bit_buffer, before);
        assert_eq!(prime(stream, 0, 5), ReturnCode::Ok);
        assert_eq!(stream.state.bit_buffer, before);

        assert_eq!(prime(stream, -1, 0), ReturnCode::Ok);
        assert_eq!(stream.state.bit_buffer, BitBuffer::default());
        assert_eq!(prime(stream, 16, i32::MIN), ReturnCode::Ok);
        assert_eq!(stream.state.bit_buffer.hold(), 0);
        assert_eq!(end(stream).state, core::ptr::null_mut());
        Ok(())
    }

    #[test]
    fn inflate_window_selection_accepts_only_complete_format_encodings() {
        for bits in -64..=100 {
            let expected = matches!(bits, -15..=-8 | 0 | 8..=15 | 24..=31 | 32 | 40..=47);
            assert_eq!(
                parse_inflate_window_bits(bits).is_some(),
                expected,
                "window bits {bits}",
            );

            if !expected {
                let mut raw = z_stream::default();
                // z_stream::default may already select an allocator under
                // rust-allocator/c-allocator features. Rejection must not
                // replace that choice, or allocate a new state.
                let alloc_before = (raw.zalloc.is_some(), raw.zfree.is_some(), raw.opaque);
                assert_eq!(
                    init(&mut raw, InflateConfig { window_bits: bits }),
                    ReturnCode::StreamError,
                );
                assert!(raw.state.is_null());
                assert_eq!(
                    (raw.zalloc.is_some(), raw.zfree.is_some(), raw.opaque),
                    alloc_before
                );
                assert!(matches!(
                    InflateStream::new(InflateConfig { window_bits: bits }),
                    Err(ReturnCode::StreamError)
                ));
            }
        }

        for (bits, wrap, wbits) in [
            (-15, 0, 15),
            (-8, 0, 8),
            (0, 5, 0),
            (8, 5, 8),
            (15, 5, 15),
            (24, 6, 8),
            (31, 6, 15),
            (32, 7, 0),
            (40, 7, 8),
            (47, 7, 15),
        ] {
            assert_eq!(
                parse_inflate_window_bits(bits),
                Some(InflateWindowSelection { wrap, wbits }),
            );
        }
        assert!(parse_inflate_window_bits(i32::MIN).is_none());
        assert!(parse_inflate_window_bits(i32::MAX).is_none());
    }

    #[test]
    fn inflate_init_reports_allocator_failure_without_exposing_state() {
        unsafe extern "C" fn reject(
            _context: *mut c_void,
            _count: core::ffi::c_uint,
            _size: core::ffi::c_uint,
        ) -> *mut c_void {
            core::ptr::null_mut()
        }
        unsafe extern "C" fn unused_free(_context: *mut c_void, _ptr: *mut c_void) {}

        let mut raw = z_stream {
            zalloc: Some(reject),
            zfree: Some(unused_free),
            ..z_stream::default()
        };

        assert_eq!(
            init(&mut raw, InflateConfig::default()),
            ReturnCode::MemError
        );
        assert!(raw.state.is_null());
        assert!(raw.msg.is_null());
    }

    #[test]
    fn successful_fallible_inflate_stream_construction_can_be_ended() -> Result<(), ReturnCode> {
        let mut stream = InflateStream::new(InflateConfig::default())?;
        assert_eq!(end(&mut stream).state, core::ptr::null_mut());
        Ok(())
    }

    #[test]
    fn invalid_inflate_reconfiguration_preserves_existing_state() -> Result<(), &'static str> {
        let mut raw = z_stream::default();
        assert_eq!(init(&mut raw, InflateConfig::default()), ReturnCode::Ok);
        // SAFETY: init succeeded and the state is exclusively borrowed for this test.
        let stream = unsafe { InflateStream::from_stream_mut(&raw mut raw) }
            .ok_or("freshly initialized inflate stream was not usable")?;
        let initial_wrap = stream.state.wrap;
        let initial_window_bits = stream.state.wbits;
        let initial_buffer = stream.state.bit_buffer;

        for window_bits in [
            -16,
            -7,
            -1,
            1,
            7,
            16,
            17,
            23,
            33,
            39,
            48,
            64,
            i32::MIN,
            i32::MAX,
        ] {
            assert_eq!(
                reset_with_config(stream, InflateConfig { window_bits }),
                ReturnCode::StreamError,
                "invalid window bits {window_bits}",
            );
            assert_eq!(stream.state.wrap, initial_wrap);
            assert_eq!(stream.state.wbits, initial_window_bits);
            assert_eq!(stream.state.bit_buffer, initial_buffer);
        }
        assert_eq!(end(stream).state, core::ptr::null_mut());
        Ok(())
    }

    #[test]
    fn gzip_header_fields_clip_storage_but_consume_all_fragments() {
        let mut extra = [0xa5_u8; 6];
        let mut name = [0xa5_u8; 5];
        let mut comment = [0xa5_u8; 6];
        let mut header = gz_header {
            extra: extra.as_mut_ptr(),
            extra_len: 8,
            extra_max: 4,
            name: name.as_mut_ptr(),
            name_max: 3,
            comment: comment.as_mut_ptr(),
            comm_max: 4,
            ..gz_header::default()
        };

        // SAFETY: every nested pointer refers to its independent, uniquely
        // writable stack buffer, with capacities no larger than those buffers.
        let mut destination = unsafe { GzipHeaderOutput::new(&mut header) };
        assert_eq!(destination.write_extra(8, b"abc"), Ok(()));
        assert_eq!(destination.write_extra(5, b"defgh"), Ok(()));
        // Invalid fragment ordering and overlong fragments must leave the
        // already-written caller buffer untouched.
        assert_eq!(
            destination.write_extra(9, b"x"),
            Err(ReturnCode::StreamError)
        );
        assert_eq!(
            destination.write_extra(1, b"xy"),
            Err(ReturnCode::StreamError)
        );
        assert_eq!(extra, *b"abcd\xa5\xa5");
        assert_eq!(header.extra_len, 8);

        let mut checksum = crate::CRC32_INITIAL_VALUE;
        let mut stored = 0;
        let mut first = BitReader::from_buffer(BitBuffer::default(), b"abc");
        assert!(!consume_gzip_header_string(
            &mut first,
            Some(destination),
            GzipStringField::Name,
            &mut stored,
            &mut checksum,
            true,
        ));
        assert_eq!(first.as_slice(), []);
        assert_eq!(stored, 3);

        let mut second = BitReader::from_buffer(BitBuffer::default(), b"def\0trailing");
        assert!(consume_gzip_header_string(
            &mut second,
            Some(destination),
            GzipStringField::Name,
            &mut stored,
            &mut checksum,
            true,
        ));
        assert_eq!(second.as_slice(), b"trailing");
        assert_eq!(stored, 3);
        assert_eq!(name, *b"abc\xa5\xa5");
        assert_eq!(checksum, crc32(crate::CRC32_INITIAL_VALUE, b"abcdef\0"));

        stored = 0;
        let mut third = BitReader::from_buffer(BitBuffer::default(), b"you\0unused");
        assert!(consume_gzip_header_string(
            &mut third,
            Some(destination),
            GzipStringField::Comment,
            &mut stored,
            &mut checksum,
            false,
        ));
        assert_eq!(third.as_slice(), b"unused");
        assert_eq!(stored, 4);
        assert_eq!(comment, *b"you\0\xa5\xa5");
        assert_eq!(checksum, crc32(crate::CRC32_INITIAL_VALUE, b"abcdef\0"));

        // The same bounded write must handle a missing destination without
        // touching memory or losing the input cursor's progress.
        destination.clear_comment();
        let mut missing = BitReader::from_buffer(BitBuffer::default(), b"skip\0rest");
        stored = 0;
        assert!(consume_gzip_header_string(
            &mut missing,
            Some(destination),
            GzipStringField::Comment,
            &mut stored,
            &mut checksum,
            false,
        ));
        assert_eq!(stored, 0);
        assert_eq!(missing.as_slice(), b"rest");
    }

    #[test]
    fn non_strict_missing_history_is_initialized_as_zero_without_panic() -> Result<(), &'static str>
    {
        let mut output_bytes = [0xa5u8; 32];
        // SAFETY: the stack array is valid and uniquely writable for 32 bytes.
        let mut output = unsafe { InflateOutput::new_uninit(output_bytes.as_mut_ptr(), 32) };
        let window = Window::empty();
        let mut remaining = 10;
        assert_eq!(
            copy_match_bytes(&window, true, 3, &mut remaining, &mut output),
            Err("invalid distance too far back\0"),
        );
        assert_eq!(output.len(), 0);

        while remaining != 0 {
            copy_match_bytes(&window, false, 3, &mut remaining, &mut output)?;
        }
        assert_eq!(output.filled(), &[0; 10]);
        assert_eq!(&output_bytes[10..], &[0xa5; 22]);
        Ok(())
    }

    #[test]
    fn non_strict_missing_history_rejoins_initialized_window_and_output() -> Result<(), &'static str>
    {
        let mut window_storage = [0xa5u8; 8];
        let mut window = Window::from_initialized_slice(&mut window_storage);
        let mut checksum = 1;
        window.extend(
            &[10, 11, 12, 13],
            0,
            false,
            &mut checksum,
            &mut Crc32Fold::new(),
        );
        let mut output_bytes = [0xa5u8; 16];
        // SAFETY: the stack array is valid and uniquely writable for 16 bytes.
        let mut output = unsafe { InflateOutput::new_uninit(output_bytes.as_mut_ptr(), 16) };
        let mut remaining = 9;
        while remaining != 0 {
            copy_match_bytes(&window, false, 6, &mut remaining, &mut output)?;
        }
        assert_eq!(output.filled(), &[0, 0, 10, 11, 12, 13, 0, 0, 10]);
        assert_eq!(&output_bytes[9..], &[0xa5; 7]);
        Ok(())
    }

    #[test]
    fn uncompress2_reports_bytes_consumed_including_with_trailing_input() {
        let plain = b"input accounting must report bytes consumed, not bytes remaining";
        let mut storage = vec![0; crate::compress_bound(plain.len())];
        let (compressed, code) =
            crate::compress_slice(&mut storage, plain, crate::DeflateConfig::default());
        assert_eq!(code, ReturnCode::Ok);
        let compressed = compressed.to_vec();

        for trailing in [0, 3, 13] {
            let mut input = compressed.clone();
            input.extend(core::iter::repeat_n(0xa5, trailing));
            let mut storage = vec![MaybeUninit::<u8>::uninit(); plain.len()];
            let (consumed, decoded, code) =
                uncompress2(&mut storage, &input, InflateConfig::default());
            assert_eq!(code, ReturnCode::Ok);
            assert_eq!(decoded, plain);
            assert_eq!(consumed, compressed.len(), "trailing {trailing} bytes");
        }
    }

    #[test]
    fn whole_slice_inflate_distinguishes_empty_stream_from_exhausted_output() {
        // RFC1950 stream containing no uncompressed bytes.
        const EMPTY_STREAM: &[u8] = &[0x78, 0x9c, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01];
        let mut empty_dest = [MaybeUninit::<u8>::uninit(); 0];
        let (consumed, decoded, code) =
            uncompress2(&mut empty_dest, EMPTY_STREAM, InflateConfig::default());
        assert_eq!(code, ReturnCode::Ok);
        assert_eq!(consumed, EMPTY_STREAM.len());
        assert_eq!(decoded, []);

        let mut compressed = [0u8; 128];
        let (encoded, code) = crate::compress_slice(
            &mut compressed,
            b"nonempty",
            crate::DeflateConfig::default(),
        );
        assert_eq!(code, ReturnCode::Ok);
        let (consumed, decoded, code) =
            uncompress2(&mut empty_dest, encoded, InflateConfig::default());
        assert_eq!(code, ReturnCode::DataError);
        assert!(consumed <= encoded.len());
        assert_eq!(decoded, []);
    }

    #[test]
    fn inflate_sync_total_in_wraps_like_unsigned_c_counter() -> Result<(), &'static str> {
        let mut raw = z_stream::default();
        assert_eq!(init(&mut raw, InflateConfig::default()), ReturnCode::Ok);
        let mut marker = [0u8, 0, 0xff, 0xff];
        // SAFETY: init succeeded and the stream owns a live, exclusively borrowed inflate state.
        let stream = unsafe { InflateStream::from_stream_mut(&raw mut raw) }
            .ok_or("freshly initialized inflate stream was not usable")?;
        stream.next_in = marker.as_mut_ptr();
        stream.avail_in = 4; // Four-byte stored-block sync marker.
        stream.total_in = z_size::MAX - 1;

        // SAFETY: the marker is a live, readable caller input span.
        assert_eq!(unsafe { sync(stream) }, ReturnCode::Ok);
        assert_eq!(stream.total_in, 2);
        assert_eq!(stream.avail_in, 0);
        assert_eq!(end(stream).state, core::ptr::null_mut());
        Ok(())
    }

    #[test]
    fn inflate_sync_rebinds_disjoint_buffers_and_preserves_null_input() -> Result<(), &'static str>
    {
        let mut raw = z_stream::default();
        assert_eq!(init(&mut raw, InflateConfig::default()), ReturnCode::Ok);
        let mut marker = [0u8, 0, 0xff, 0xff];
        // SAFETY: the initialized stream owns a live exclusively borrowed inflate state.
        let stream = unsafe { InflateStream::from_stream_mut(&raw mut raw) }
            .ok_or("initialized inflate stream is unavailable")?;
        stream.total_in = z_size::MAX - 2;

        for index in 0..marker.len() {
            stream.next_in = marker[index..].as_mut_ptr();
            stream.avail_in = 1;
            // SAFETY: the supplied byte is readable for this call.
            let result = unsafe { sync(stream) };
            assert_eq!(
                result,
                if index == 3 {
                    ReturnCode::Ok
                } else {
                    ReturnCode::DataError
                }
            );
            assert_eq!(stream.avail_in, 0);
            assert_eq!(stream.next_in, marker[index..].as_mut_ptr().wrapping_add(1));
            assert_eq!(
                stream.total_in,
                (z_size::MAX - 2).wrapping_add(index as z_size + 1)
            );

            if index == 0 {
                stream.next_in = core::ptr::null_mut();
                // SAFETY: zero available bytes require no input allocation.
                assert_eq!(unsafe { sync(stream) }, ReturnCode::BufError);
                assert!(stream.next_in.is_null());
                assert_eq!(stream.avail_in, 0);
                assert_eq!(stream.total_in, z_size::MAX - 1);
            }
        }
        assert_eq!(end(stream).state, core::ptr::null_mut());
        Ok(())
    }

    #[test]
    fn inflate_total_in_wraps_like_unsigned_c_counter() -> Result<(), &'static str> {
        // An empty RFC1950 zlib stream is eight bytes long, and creates no output.
        let mut input = [0x78, 0x9c, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01];
        let mut output = [0u8; 1];
        let mut raw = z_stream::default();
        assert_eq!(init(&mut raw, InflateConfig::default()), ReturnCode::Ok);
        // SAFETY: init succeeded and the stream owns a live, exclusively borrowed inflate state.
        let stream = unsafe { InflateStream::from_stream_mut(&raw mut raw) }
            .ok_or("freshly initialized inflate stream was not usable")?;
        stream.next_in = input.as_mut_ptr();
        stream.avail_in = 8; // Eight-byte RFC1950 empty stream.
        stream.next_out = output.as_mut_ptr();
        stream.avail_out = 1;
        stream.total_in = z_size::MAX - 1;

        // SAFETY: the input and output spans remain live and non-overlapping for this call.
        assert_eq!(
            unsafe { inflate(stream, InflateFlush::Finish) },
            ReturnCode::StreamEnd
        );
        assert_eq!(stream.total_in, 6);
        assert_eq!(stream.avail_in, 0);
        assert_eq!(end(stream).state, core::ptr::null_mut());
        Ok(())
    }

    #[test]
    fn end_leaves_null_abi_state_pointer() {
        let mut raw = z_stream::default();
        assert_eq!(init(&mut raw, InflateConfig::default()), ReturnCode::Ok);
        let stream = unsafe { InflateStream::from_stream_mut(&raw mut raw) }.unwrap();

        assert!(end(stream).state.is_null());
        assert!(end(stream).state.is_null());
    }

    #[test]
    fn inflate_copy_owns_independent_state_and_window_storage() {
        let mut raw = z_stream::default();
        assert_eq!(init(&mut raw, InflateConfig::default()), ReturnCode::Ok);
        let source = unsafe { InflateStream::from_stream_mut(&raw mut raw) }.unwrap();
        let source_state = source.state.as_ptr();
        let source_window = source.state.window.as_ptr();

        let mut copied = MaybeUninit::<InflateStream>::uninit();
        assert_eq!(copy(&mut copied, source), ReturnCode::Ok);
        let copied = unsafe { copied.assume_init_mut() };

        assert_ne!(source_state, copied.state.as_ptr());
        assert_ne!(source_window, copied.state.window.as_ptr());
        assert_eq!(
            source.state.total_allocation_size,
            copied.state.total_allocation_size
        );

        end(copied);
        end(source);
    }

    #[test]
    fn extended_gzip_header_resumes_across_every_small_chunk_and_rejects_bad_crc(
    ) -> Result<(), &'static str> {
        // Extended gzip prefix including FTEXT/FHCRC/FEXTRA/FNAME/FCOMMENT,
        // with an independently generated CRC16 at the end. The decoder
        // must finish metadata parsing before attempting the payload.
        let header = [
            0x1f, 0x8b, 0x08, 0x1f, 0x44, 0x0a, 0x45, 0x65, 0x00, 0x03, 0x0e, 0x00, 0x54, 0x47,
            0x0a, 0x00, 0x45, 0x58, 0x54, 0x52, 0x41, 0x20, 0x44, 0x41, 0x54, 0x41, 0x74, 0x65,
            0x73, 0x74, 0x2e, 0x74, 0x78, 0x74, 0x00, 0x41, 0x20, 0x63, 0x6f, 0x6d, 0x6d, 0x65,
            0x6e, 0x74, 0x00, 0x3b, 0x92,
        ];
        for chunk in [1usize, 2, 3, 5, 7, 13, header.len()] {
            for corrupt_crc in [false, true] {
                let mut input = header;
                if corrupt_crc {
                    let last = input.len() - 1;
                    input[last] ^= 1;
                }
                let mut extra = [0u8; 14];
                let mut name = [0u8; 9];
                let mut comment = [0u8; 10];
                let mut destination = gz_header {
                    extra: extra.as_mut_ptr(),
                    extra_max: 14,
                    name: name.as_mut_ptr(),
                    name_max: 9,
                    comment: comment.as_mut_ptr(),
                    comm_max: 10,
                    ..gz_header::default()
                };
                let mut raw = z_stream::default();
                assert_eq!(
                    init(&mut raw, InflateConfig { window_bits: 47 }),
                    ReturnCode::Ok
                );
                // SAFETY: successful init created an exclusively borrowed
                // regular inflate state; destination buffers remain live.
                let stream = unsafe { InflateStream::from_stream_mut(&raw mut raw) }
                    .ok_or("initialized gzip state is unavailable")?;
                // SAFETY: the independent nested arrays are live, disjoint,
                // and writable for exactly the configured capacities.
                assert_eq!(
                    unsafe { get_header(stream, Some(&mut destination)) },
                    ReturnCode::Ok
                );

                let mut consumed = 0;
                let mut scratch = [0u8; 1];
                let mut status = ReturnCode::Ok;
                while consumed < input.len() && destination.done == 0 {
                    let offered = chunk.min(input.len() - consumed);
                    stream.next_in = input[consumed..].as_ptr().cast_mut();
                    stream.avail_in = u32::try_from(offered).map_err(|_| "chunk too large")?;
                    stream.next_out = scratch.as_mut_ptr();
                    stream.avail_out = 1;

                    // SAFETY: both ABI slices are live and bounded by the
                    // exact current per-call avail_in/avail_out counts.
                    status = unsafe { inflate(stream, InflateFlush::NoFlush) };
                    let taken = offered
                        - usize::try_from(stream.avail_in)
                            .map_err(|_| "invalid input remainder")?;
                    assert!(taken > 0, "gzip header parser stalled at {consumed}");
                    consumed += taken;
                    if matches!(status, ReturnCode::DataError) {
                        break;
                    }
                }

                assert_eq!(consumed, input.len());
                if corrupt_crc {
                    assert_eq!(status, ReturnCode::DataError);
                    assert_eq!(destination.done, 0);
                } else {
                    assert_eq!(destination.done, 1);
                    assert_eq!(destination.text, 1);
                    assert_eq!(destination.time, 1_699_023_428);
                    assert_eq!(destination.os, 3);
                    assert_eq!(destination.extra_len, 14);
                    assert_eq!(destination.hcrc, 1);
                    assert_eq!(extra, *b"TG\n\0EXTRA DATA");
                    assert_eq!(name, *b"test.txt\0");
                    assert_eq!(comment, *b"A comment\0");
                }
                assert_eq!(end(stream).state, core::ptr::null_mut());
            }
        }
        Ok(())
    }

    #[test]
    fn gzip_header_destination_is_shared_safely_across_inflate_copy() {
        // Extended gzip header with FTEXT/FHCRC/FEXTRA/FNAME/FCOMMENT set. The payload starts after
        // the header; this test only needs both copied streams to finish parsing the header.
        let input: &[u8] = &[
            0x1f, 0x8b, 0x08, 0x1f, 0x44, 0x0a, 0x45, 0x65, 0x00, 0x03, 0x0e, 0x00, 0x54, 0x47,
            0x0a, 0x00, 0x45, 0x58, 0x54, 0x52, 0x41, 0x20, 0x44, 0x41, 0x54, 0x41, 0x74, 0x65,
            0x73, 0x74, 0x2e, 0x74, 0x78, 0x74, 0x00, 0x41, 0x20, 0x63, 0x6f, 0x6d, 0x6d, 0x65,
            0x6e, 0x74, 0x00, 0x3b, 0x92,
        ];
        let mut extra = [0u8; 14];
        let mut name = [0u8; 9];
        let mut comment = [0u8; 10];
        let mut header = gz_header {
            text: 0,
            time: 0,
            xflags: 0,
            os: 0,
            extra: extra.as_mut_ptr(),
            extra_len: 0,
            extra_max: extra.len() as _,
            name: name.as_mut_ptr(),
            name_max: name.len() as _,
            comment: comment.as_mut_ptr(),
            comm_max: comment.len() as _,
            hcrc: 0,
            done: 0,
        };

        let mut source_stream = z_stream::default();
        assert_eq!(
            init(&mut source_stream, InflateConfig { window_bits: 47 }),
            ReturnCode::Ok
        );
        let source = unsafe { InflateStream::from_stream_mut(&raw mut source_stream) }.unwrap();
        assert_eq!(
            unsafe { get_header(source, Some(&mut header)) },
            ReturnCode::Ok
        );

        // Configure I/O before copying, but do not call inflate yet: both states must retain the
        // same caller-owned header destination just as zlib's copied C pointer does.
        let mut source_out = [0u8; 1];
        source.next_in = input.as_ptr().cast_mut();
        source.avail_in = input.len() as _;
        source.next_out = source_out.as_mut_ptr();
        source.avail_out = source_out.len() as _;

        let mut copied = MaybeUninit::<InflateStream>::uninit();
        assert_eq!(copy(&mut copied, source), ReturnCode::Ok);
        let copied = unsafe { copied.assume_init_mut() };
        let mut copied_out = [0u8; 1];
        copied.next_out = copied_out.as_mut_ptr();
        copied.avail_out = copied_out.len() as _;

        let source_ret = unsafe { inflate(source, InflateFlush::NoFlush) };
        assert!(matches!(source_ret, ReturnCode::Ok | ReturnCode::StreamEnd));
        assert_eq!(header.done, 1);
        assert_eq!(header.text, 1);
        assert_eq!(header.time, 1_699_023_428);
        assert_eq!(header.os, 3);
        assert_eq!(header.hcrc, 1);

        let copied_ret = unsafe { inflate(copied, InflateFlush::NoFlush) };
        assert_eq!(copied_ret, source_ret);
        assert_eq!(header.done, 1);
        assert_eq!(extra, *b"TG\n\0EXTRA DATA");
        assert_eq!(&name, b"test.txt\0");
        assert_eq!(&comment, b"A comment\0");

        end(source);
        end(copied);
    }

    fn inflate_fragmented(
        compressed: &[u8],
        expected: &[u8],
        input_chunk: usize,
        output_chunk: usize,
    ) {
        let mut raw = z_stream::default();
        assert_eq!(init(&mut raw, InflateConfig::default()), ReturnCode::Ok);
        let stream = unsafe { InflateStream::from_stream_mut(&raw mut raw) }.unwrap();

        let mut output = vec![0u8; expected.len() + output_chunk];
        let mut next_input = 0usize;
        let mut next_output = 0usize;

        loop {
            if stream.avail_in == 0 && next_input < compressed.len() {
                let len = input_chunk.min(compressed.len() - next_input);
                stream.next_in = compressed[next_input..].as_ptr().cast_mut();
                stream.avail_in = len as u32;
                next_input += len;
            }

            if stream.avail_out == 0 {
                let len = output_chunk.min(output.len() - next_output);
                assert!(
                    len > 0,
                    "fragmented inflate exhausted its output allocation"
                );
                stream.next_out = output[next_output..].as_mut_ptr();
                stream.avail_out = len as u32;
                next_output += len;
            }

            match unsafe { inflate(stream, InflateFlush::NoFlush) } {
                ReturnCode::Ok => {}
                ReturnCode::StreamEnd => break,
                code => panic!("fragmented inflate returned {code:?}"),
            }
        }

        let written = stream.total_out as usize;
        assert_eq!(written, expected.len());
        assert_eq!(&output[..written], expected);
        end(stream);
    }

    #[test]
    fn inflate_resumes_across_fragmented_input_and_output() {
        const PATTERN: &[u8] =
            b"the quick brown fox jumps over the lazy dog | 0123456789 | DEFLATE!";
        let input: Vec<u8> = (0..16 * 1024)
            .map(|i| PATTERN[(i * 17 + i / 97) % PATTERN.len()])
            .collect();
        let mut compressed_storage = vec![0u8; crate::compress_bound(input.len())];
        let (compressed, code) = crate::compress_slice(
            &mut compressed_storage,
            &input,
            crate::DeflateConfig::default(),
        );
        assert_eq!(code, ReturnCode::Ok);
        let compressed = compressed.to_vec();

        for input_chunk in [1, 2, 3, 5, 7, 13] {
            for output_chunk in [1, 3, 7, 31] {
                inflate_fragmented(&compressed, &input, input_chunk, output_chunk);
            }
        }
    }

    #[test]
    fn uninitialized_output_exposes_exactly_the_decompressed_prefix() {
        let input: Vec<u8> = (0..257)
            .flat_map(|i| {
                let byte = (i % 251) as u8;
                [b"typed-inflate-output/".as_slice(), &[byte]].concat()
            })
            .collect();
        let mut compressed_storage = vec![0u8; crate::compress_bound(input.len())];
        let (compressed, code) = crate::compress_slice(
            &mut compressed_storage,
            &input,
            crate::DeflateConfig::default(),
        );
        assert_eq!(code, ReturnCode::Ok);
        let compressed = compressed.to_vec();

        let capacity = input.len() + 97;
        let mut initialized = vec![0u8; capacity];
        let (expected, expected_code) =
            decompress_slice(&mut initialized, &compressed, InflateConfig::default());
        assert_eq!(expected_code, ReturnCode::Ok);
        assert_eq!(expected, input);
        let expected = expected.to_vec();

        let mut uninitialized = vec![MaybeUninit::<u8>::uninit(); capacity];
        let (actual, actual_code) =
            uncompress(&mut uninitialized, &compressed, InflateConfig::default());
        assert_eq!(actual_code, ReturnCode::Ok);
        assert_eq!(actual, expected);
        assert_eq!(actual.len(), input.len());
        assert!(actual.len() < capacity);
    }

    #[test]
    fn uncompress_buffer_overflow() {
        let mut output = [0; 1 << 13];
        let input = [
            72, 137, 58, 0, 3, 39, 255, 255, 255, 255, 255, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
            14, 14, 184, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 184, 14, 14,
            14, 14, 14, 14, 14, 63, 14, 14, 14, 14, 14, 14, 14, 14, 184, 14, 14, 255, 14, 103, 14,
            14, 14, 14, 14, 14, 61, 14, 255, 255, 63, 14, 14, 14, 14, 14, 14, 14, 14, 184, 14, 14,
            255, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 6, 14, 14, 14, 14, 14, 14, 14, 14, 71,
            4, 137, 106,
        ];

        let config = InflateConfig { window_bits: 15 };

        let (_decompressed, err) = decompress_slice(&mut output, &input, config);
        assert_eq!(err, ReturnCode::DataError);
    }

    #[test]
    fn back_init_raw_rejects_null_window() {
        let mut stream = z_stream::default();
        assert_eq!(
            unsafe {
                back_init_raw(
                    &mut stream,
                    InflateConfig {
                        window_bits: MIN_WBITS,
                    },
                    core::ptr::null_mut(),
                )
            },
            ReturnCode::StreamError,
        );
    }
}
