#![allow(non_snake_case)] // TODO ultimately remove this
#![allow(clippy::missing_safety_doc)] // obviously needs to be fixed long-term

use core::ffi::{c_char, c_int, c_long, c_ulong};
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ops::ControlFlow;
use core::ptr::NonNull;

mod bitreader;
mod infback;
mod inffixed_tbl;
mod inftrees;
mod window;
mod writer;

use crate::allocate::Allocator;
use crate::c_api::internal_state;
use crate::output_buffer::{assume_init_prefix, WritableBytes};
use crate::state_ptr::StatePtr;
use crate::{
    adler32::adler32,
    c_api::{gz_header, z_checksum, z_size, z_stream, Z_DEFLATED},
    inflate::writer::Writer,
    traceln, Code, InflateFlush, ReturnCode, DEF_WBITS, MAX_WBITS, MIN_WBITS,
};

use crate::crc32::{crc32, Crc32Fold};

pub use self::infback::{back, back_end, back_init};
pub use self::window::Window;
use self::{
    bitreader::{BitBuffer, BitReader},
    inftrees::{inflate_table, CodeType, InflateTable},
};

const INFLATE_STRICT: bool = false;

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

unsafe impl Sync for InflateStream<'_> {}
unsafe impl Send for InflateStream<'_> {}

#[cfg(feature = "__internal-test")]
#[doc(hidden)]
pub const INFLATE_STATE_SIZE: usize = core::mem::size_of::<crate::inflate::State>();

#[cfg(feature = "__internal-test")]
#[doc(hidden)]
pub unsafe fn set_mode_dict(strm: &mut z_stream) {
    unsafe {
        (*(strm.state as *mut State)).mode = Mode::Dict;
    }
}

#[cfg(feature = "__internal-test")]
#[doc(hidden)]
pub unsafe fn set_mode_sync(strm: *mut z_stream) {
    unsafe {
        (*((*strm).state as *mut State)).mode = Mode::Sync;
    }
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
    #[inline(always)]
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

    #[inline(always)]
    unsafe fn from_stream_mut_kind(
        strm: *mut z_stream,
        kind: InflateStateKind,
    ) -> Option<&'a mut Self> {
        // Validate through a shared borrow before constructing the exclusive ABI view.
        unsafe { Self::from_stream_ref_kind(strm.cast_const(), kind)? };
        // SAFETY: InflateStream has an equivalent layout as z_stream and the caller supplied a
        // mutable raw pointer satisfying `pointer::as_mut`.
        unsafe { strm.cast::<InflateStream>().as_mut() }
    }

    /// Convert a regular `inflateInit*` stream. BackInit streams are deliberately rejected.
    #[inline(always)]
    pub unsafe fn from_stream_ref(strm: *const z_stream) -> Option<&'a Self> {
        unsafe { Self::from_stream_ref_kind(strm, InflateStateKind::Regular) }
    }

    /// Convert a regular `inflateInit*` stream. BackInit streams are deliberately rejected.
    #[inline(always)]
    pub unsafe fn from_stream_mut(strm: *mut z_stream) -> Option<&'a mut Self> {
        unsafe { Self::from_stream_mut_kind(strm, InflateStateKind::Regular) }
    }

    /// Convert an `inflateBackInit*` stream. Regular inflate streams are rejected.
    #[inline(always)]
    pub unsafe fn from_back_stream_mut(strm: *mut z_stream) -> Option<&'a mut Self> {
        unsafe { Self::from_stream_mut_kind(strm, InflateStateKind::Back) }
    }

    fn as_z_stream_mut(&mut self) -> &mut z_stream {
        // safety: a valid &mut InflateStream is also a valid &mut z_stream
        unsafe { &mut *(self as *mut _ as *mut z_stream) }
    }

    pub fn new(config: InflateConfig) -> Self {
        let mut inner = crate::c_api::z_stream::default();

        let ret = crate::inflate::init(&mut inner, config);
        assert_eq!(ret, ReturnCode::Ok);

        let state = inner.state.cast::<State<'a>>();
        let alloc = Allocator {
            zalloc: inner
                .zalloc
                .expect("initialized inflate stream has an allocator"),
            zfree: inner
                .zfree
                .expect("initialized inflate stream has a deallocator"),
            opaque: inner.opaque,
            _marker: PhantomData,
        };

        Self {
            next_in: inner.next_in.cast_mut(),
            avail_in: inner.avail_in,
            total_in: inner.total_in,
            next_out: inner.next_out,
            avail_out: inner.avail_out,
            total_out: inner.total_out,
            msg: inner.msg,
            // SAFETY: successful `init` created a live regular-inflate State allocation.
            state: unsafe { StatePtr::from_raw(state) },
            alloc,
            data_type: inner.data_type,
            adler: inner.adler,
            reserved: inner.reserved,
        }
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

pub fn uncompress2<'a>(
    output: &'a mut [MaybeUninit<u8>],
    input: &[u8],
    config: InflateConfig,
) -> (u64, &'a mut [u8], ReturnCode) {
    let (consumed, produced, ret) = uncompress2_into(output, input, config);
    // SAFETY: `uncompress2_into` reports only bytes initialized by the inflate output cursor.
    let output = unsafe { assume_init_prefix(output, produced) };
    (consumed, output, ret)
}

fn uncompress2_into<O: WritableBytes + ?Sized>(
    output: &mut O,
    input: &[u8],
    config: InflateConfig,
) -> (u64, usize, ReturnCode) {
    let output_len = output.len();
    let mut dest_len = output_len as z_checksum;

    // for detection of incomplete stream when *destLen == 0
    let mut buf = [0u8];

    let mut left;
    let mut len = input.len() as u64;

    let dest = if output_len == 0 {
        left = 1;
        buf.as_mut_ptr()
    } else {
        left = output_len as u64;
        dest_len = 0;
        output.byte_ptr()
    };

    let mut stream = z_stream {
        next_in: input.as_ptr() as *mut u8,
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

    let Some(stream) = (unsafe { InflateStream::from_stream_mut(&mut stream) }) else {
        return (0, 0, ReturnCode::StreamError);
    };

    let err = loop {
        if stream.avail_out == 0 {
            stream.avail_out = Ord::min(left, u32::MAX as u64) as u32;
            left -= stream.avail_out as u64;
        }

        if stream.avail_in == 0 {
            stream.avail_in = Ord::min(len, u32::MAX as u64) as u32;
            len -= stream.avail_in as u64;
        }

        let err = unsafe { inflate(stream, InflateFlush::NoFlush) };

        if err != ReturnCode::Ok {
            break err;
        }
    };

    let consumed = len + u64::from(stream.avail_in);
    if output_len != 0 {
        dest_len = stream.total_out;
    } else if stream.total_out != 0 && err == ReturnCode::BufError {
        left = 1;
    }

    let avail_out = stream.avail_out;
    end(stream);

    let ret = match err {
        ReturnCode::StreamEnd => ReturnCode::Ok,
        ReturnCode::NeedDict => ReturnCode::DataError,
        ReturnCode::BufError if (left + avail_out as u64) != 0 => ReturnCode::DataError,
        _ => err,
    };

    let produced = dest_len as usize;
    assert!(
        produced <= output_len,
        "inflate reported output beyond caller storage"
    );
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
#[allow(clippy::enum_variant_names)]
enum Codes {
    #[default]
    Fixed,
    Codes,
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

    #[inline(always)]
    pub(crate) fn update(&mut self, other: Self, value: bool) {
        if value {
            *self = Self(self.0 | other.0);
        } else {
            *self = Self(self.0 & !other.0);
        }
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InflateStateKind {
    Regular,
    Back,
}

/// Writable caller-owned gzip-header destination bound by `get_header`.
///
/// This deliberately stores only a raw handle rather than a long-lived `&mut gz_header`:
/// `inflateCopy()` shallow-copies the destination exactly like zlib. Each parser operation creates
/// only the short mutable access it needs, and all nested extra/name/comment pointer writes are
/// confined here.
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

    fn set_extra_len(&mut self, len: usize) {
        self.with_header_mut(|head| head.extra_len = len as u32);
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

    fn write_extra(&mut self, remaining_before: usize, bytes: &[u8]) {
        self.with_header_mut(|head| {
            if head.extra.is_null() {
                return;
            }

            let written = (head.extra_len as usize)
                .checked_sub(remaining_before)
                .expect("extra-field progress stays within declared length");
            let capacity = head.extra_max as usize;
            let count = Ord::min(bytes.len(), capacity.saturating_sub(written));
            if count == 0 {
                return;
            }
            let offset = Ord::min(written, capacity);

            // SAFETY: `get_header` requires `extra` to be writable for `extra_max` bytes. `offset`
            // and `count` are bounded to that capacity and `bytes` is readable for `count` bytes.
            unsafe {
                core::ptr::copy_nonoverlapping(bytes.as_ptr(), head.extra.add(offset), count)
            };
        });
    }

    fn write_name(&mut self, offset: usize, bytes: &[u8]) -> usize {
        self.with_header_mut(|head| {
            if head.name.is_null() {
                return 0;
            }

            let remaining = (head.name_max as usize)
                .checked_sub(offset)
                .expect("name output offset stays within capacity");
            let count = Ord::min(bytes.len(), remaining);
            if count != 0 {
                // SAFETY: `get_header` requires `name` writable for `name_max` bytes, and the
                // checked offset plus bounded count stays within that allocation.
                unsafe {
                    core::ptr::copy_nonoverlapping(bytes.as_ptr(), head.name.add(offset), count)
                };
            }
            count
        })
    }

    fn write_comment(&mut self, offset: usize, bytes: &[u8]) -> usize {
        self.with_header_mut(|head| {
            if head.comment.is_null() {
                return 0;
            }

            let remaining = (head.comm_max as usize)
                .checked_sub(offset)
                .expect("comment output offset stays within capacity");
            let count = Ord::min(bytes.len(), remaining);
            if count != 0 {
                // SAFETY: `get_header` requires `comment` writable for `comm_max` bytes, and the
                // checked offset plus bounded count stays within that allocation.
                unsafe {
                    core::ptr::copy_nonoverlapping(bytes.as_ptr(), head.comment.add(offset), count)
                };
            }
            count
        })
    }
}

#[repr(C, align(64))]
pub(crate) struct State<'a> {
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

    /// extra bits needed
    extra: usize,

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
        Self {
            kind: self.kind,
            mode: self.mode,
            flags: self.flags,
            wbits: self.wbits,
            wrap: self.wrap,
            window,
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

    fn len_table_ref(&self) -> &[Code] {
        match self.len_table.codes {
            Codes::Fixed => &self::inffixed_tbl::LENFIX,
            Codes::Codes => &self.codes_codes,
            Codes::Len => &self.len_codes,
            Codes::Dist => &self.dist_codes,
        }
    }

    fn dist_table_ref(&self) -> &[Code] {
        match self.dist_table.codes {
            Codes::Fixed => &self::inffixed_tbl::DISTFIX,
            Codes::Codes => &self.codes_codes,
            Codes::Len => &self.len_codes,
            Codes::Dist => &self.dist_codes,
        }
    }

    fn len_table_get(&self, index: usize) -> Code {
        self.len_table_ref()[index]
    }

    fn dist_table_get(&self, index: usize) -> Code {
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
    unsafe fn new_uninit(ptr: *mut u8, len: usize) -> Self {
        Self {
            writer: unsafe { Writer::new_uninit(ptr, len) },
            accounted_remaining: len,
        }
    }

    #[inline]
    fn mark_accounted(&mut self) {
        self.accounted_remaining = self.writer.remaining();
    }
    #[inline]
    fn written_since_accounted(&self) -> usize {
        self.accounted_remaining - self.writer.remaining()
    }
    #[inline]
    fn remaining(&self) -> usize {
        self.writer.remaining()
    }
    #[inline]
    fn len(&self) -> usize {
        self.writer.len()
    }
    #[inline]
    fn is_full(&self) -> bool {
        self.writer.is_full()
    }
    #[inline]
    fn filled(&self) -> &[u8] {
        self.writer.filled()
    }
    #[inline]
    fn next_out(&mut self) -> *mut core::mem::MaybeUninit<u8> {
        self.writer.next_out()
    }
    #[inline]
    fn push(&mut self, byte: u8) {
        self.writer.push(byte)
    }
    #[inline]
    fn extend(&mut self, bytes: &[u8]) {
        self.writer.extend(bytes)
    }
    #[inline]
    fn extend_from_window(&mut self, window: &Window<'_>, range: core::ops::Range<usize>) {
        self.writer.extend_from_window(window, range)
    }
    #[inline]
    fn copy_match(&mut self, offset_from_end: usize, length: usize) {
        self.writer.copy_match(offset_from_end, length)
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
    fn continue_with(mode: Mode, reader: BitReader<'input>) -> Self {
        Self {
            mode,
            reader,
            flow: ControlFlow::Continue(()),
        }
    }

    fn break_with(mode: Mode, reader: BitReader<'input>, code: ReturnCode) -> Self {
        Self {
            mode,
            reader,
            flow: ControlFlow::Break(code),
        }
    }
}

#[inline(always)]
fn need_bits_for_mode(
    reader: &mut BitReader<'_>,
    count: usize,
    mode: Mode,
) -> Result<(), DispatchExit> {
    reader
        .need_bits(count)
        .map_err(|code| DispatchExit { mode, code })
}

#[inline(always)]
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
    extra: usize,
    back_bits: usize,
}

#[inline(always)]
fn emit_literal(output: &mut InflateOutput<'_>, literal: u8) -> bool {
    if output.is_full() {
        return false;
    }
    output.push(literal);
    true
}

#[inline(always)]
fn consume_extra_bits(
    reader: &mut BitReader<'_>,
    extra: usize,
    value: &mut usize,
    back: &mut usize,
) -> Result<(), ReturnCode> {
    if extra != 0 {
        reader.need_bits(extra)?;
        *value += reader.bits(extra) as usize;
        reader.drop_bits(extra as u8);
        *back += extra;
    }
    Ok(())
}

#[inline(always)]
fn decode_distance_code(
    reader: &mut BitReader<'_>,
    table: &[Code],
    table_bits: usize,
) -> Result<DecodedDistance, DistanceDecodeError> {
    let mut here;
    loop {
        here = table[reader.bits(table_bits) as usize];
        if here.bits <= reader.bits_in_buffer() {
            break;
        }
        reader.pull_byte().map_err(DistanceDecodeError::Input)?;
    }

    let mut back_bits = 0;
    if here.op & 0xf0 == 0 {
        let last = here;
        loop {
            let bits = reader.bits((last.bits + last.op) as usize) as usize;
            here = table[last.val as usize + (bits >> last.bits)];
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
        extra: (here.op & MAX_BITS) as usize,
        back_bits,
    })
}

#[inline(always)]
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
            panic!("INFLATE_ALLOW_INVALID_DISTANCE_TOOFAR_ARRR")
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
            inflate_fast_help(self, reader, output);
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
            Codes::Codes => &self.codes_codes,
            Codes::Len => &self.len_codes,
            Codes::Dist => &self.dist_codes,
        };

        let dist_table = match self.dist_table.codes {
            Codes::Fixed => &self::inffixed_tbl::DISTFIX[..],
            Codes::Codes => &self.codes_codes,
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
                            inflate_fast_help(self, &mut bit_reader, output);
                            return LenDecodeOutcome::continue_with(self.mode, bit_reader);
                        }

                        self.back = 0;

                        // get a literal, length, or end-of-block code
                        let mut here;
                        loop {
                            let bits = bit_reader.bits(self.len_table.bits);
                            here = len_table[bits as usize];

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
                                let bits = bit_reader.bits((last.bits + last.op) as usize) as u16;
                                here = len_table[(last.val + (bits >> last.bits)) as usize];
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
                                #[cfg(all(feature = "std", test))]
                                dbg!(msg);
                                self.error_message = Some(msg);
                                return LenDecodeOutcome::break_with(
                                    mode,
                                    bit_reader,
                                    ReturnCode::DataError,
                                );
                            }
                        } else {
                            // length code
                            self.extra = (here.op & MAX_BITS) as usize;
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

    fn dispatch(
        &mut self,
        bit_reader: &mut BitReader<'_>,
        output: &mut InflateOutput<'_>,
        flush: InflateFlush,
    ) -> ReturnCode {
        let mode = self.mode;
        match self.dispatch_inner(bit_reader, output, flush, mode) {
            Ok((mode, code)) => {
                self.mode = mode;
                code
            }
            Err(DispatchExit { mode, code }) => {
                self.mode = mode;
                code
            }
        }
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
                    Mode::Head => {
                        if self.wrap == 0 {
                            break 'blk Mode::TypeDo;
                        }

                        need_bits_for_mode(bit_reader, 16, mode)?;

                        // Gzip
                        if (self.wrap & 2) != 0 && bit_reader.hold() == 0x8b1f {
                            if self.wbits == 0 {
                                self.wbits = 15;
                            }

                            let b0 = bit_reader.bits(8) as u8;
                            let b1 = (bit_reader.hold() >> 8) as u8;
                            self.checksum = crc32(crate::CRC32_INITIAL_VALUE, &[b0, b1]);
                            bit_reader.init_bits();

                            break 'blk Mode::Flags;
                        }

                        if let Some(mut header) = self.head {
                            header.set_done(-1);
                        }

                        // check if zlib header is allowed
                        if (self.wrap & 1) == 0
                            || ((bit_reader.bits(8) << 8) + (bit_reader.hold() >> 8)) % 31 != 0
                        {
                            mode = Mode::Bad;
                            break 'label self.bad("incorrect header check\0");
                        }

                        if bit_reader.bits(4) != Z_DEFLATED as u64 {
                            mode = Mode::Bad;
                            break 'label self.bad("unknown compression method\0");
                        }

                        bit_reader.drop_bits(4);
                        let len = bit_reader.bits(4) as u8 + 8;

                        if self.wbits == 0 {
                            self.wbits = len;
                        }

                        if len as i32 > MAX_WBITS || len > self.wbits {
                            mode = Mode::Bad;
                            break 'label self.bad("invalid window size\0");
                        }

                        self.dmax = 1 << len;
                        self.gzip_flags = 0; // indicate zlib header
                        self.checksum = crate::ADLER32_INITIAL_VALUE as _;

                        if bit_reader.hold() & 0x200 != 0 {
                            bit_reader.init_bits();

                            break 'blk Mode::DictId;
                        } else {
                            bit_reader.init_bits();

                            break 'blk Mode::Type;
                        }
                    }
                    Mode::Flags => {
                        need_bits_for_mode(bit_reader, 16, mode)?;
                        self.gzip_flags = bit_reader.hold() as i32;

                        // Z_DEFLATED = 8 is the only supported method
                        if self.gzip_flags & 0xff != Z_DEFLATED {
                            mode = Mode::Bad;
                            break 'label self.bad("unknown compression method\0");
                        }

                        if self.gzip_flags & 0xe000 != 0 {
                            mode = Mode::Bad;
                            break 'label self.bad("unknown header flags set\0");
                        }

                        if let Some(mut head) = self.head {
                            head.set_text(((bit_reader.hold() >> 8) & 1) as i32);
                        }

                        if (self.gzip_flags & 0x0200) != 0 && (self.wrap & 4) != 0 {
                            let b0 = bit_reader.bits(8) as u8;
                            let b1 = (bit_reader.hold() >> 8) as u8;
                            self.checksum = crc32(self.checksum, &[b0, b1]);
                        }

                        bit_reader.init_bits();

                        break 'blk Mode::Time;
                    }
                    Mode::Time => {
                        need_bits_for_mode(bit_reader, 32, mode)?;
                        if let Some(mut head) = self.head {
                            head.set_time(bit_reader.hold() as z_size);
                        }

                        if (self.gzip_flags & 0x0200) != 0 && (self.wrap & 4) != 0 {
                            let bytes = (bit_reader.hold() as u32).to_le_bytes();
                            self.checksum = crc32(self.checksum, &bytes);
                        }

                        bit_reader.init_bits();

                        break 'blk Mode::Os;
                    }
                    Mode::Os => {
                        need_bits_for_mode(bit_reader, 16, mode)?;
                        if let Some(mut head) = self.head {
                            head.set_os(
                                (bit_reader.hold() & 0xff) as i32,
                                (bit_reader.hold() >> 8) as i32,
                            );
                        }

                        if (self.gzip_flags & 0x0200) != 0 && (self.wrap & 4) != 0 {
                            let bytes = (bit_reader.hold() as u16).to_le_bytes();
                            self.checksum = crc32(self.checksum, &bytes);
                        }

                        bit_reader.init_bits();

                        break 'blk Mode::ExLen;
                    }
                    Mode::ExLen => {
                        if (self.gzip_flags & 0x0400) != 0 {
                            need_bits_for_mode(bit_reader, 16, mode)?;

                            // self.length (and head.extra_len) represent the length of the extra field
                            self.length = bit_reader.hold() as usize;
                            if let Some(mut head) = self.head {
                                head.set_extra_len(self.length);
                            }

                            if (self.gzip_flags & 0x0200) != 0 && (self.wrap & 4) != 0 {
                                let bytes = (bit_reader.hold() as u16).to_le_bytes();
                                self.checksum = crc32(self.checksum, &bytes);
                            }
                            bit_reader.init_bits();
                        } else if let Some(mut head) = self.head {
                            head.clear_extra();
                        }

                        break 'blk Mode::Extra;
                    }
                    Mode::Extra => {
                        if (self.gzip_flags & 0x0400) != 0 {
                            // self.length is the number of remaining `extra` bytes. But they may not all be available
                            let extra_available =
                                Ord::min(self.length, bit_reader.bytes_remaining());

                            if extra_available > 0 {
                                if let Some(mut head) = self.head {
                                    head.write_extra(
                                        self.length,
                                        &bit_reader.as_slice()[..extra_available],
                                    );
                                }

                                // Checksum
                                if (self.gzip_flags & 0x0200) != 0 && (self.wrap & 4) != 0 {
                                    let extra_slice = &bit_reader.as_slice()[..extra_available];
                                    self.checksum = crc32(self.checksum, extra_slice)
                                }

                                bit_reader.advance(extra_available);
                                self.length -= extra_available;
                            }

                            // Checks for errors occur after returning
                            if self.length != 0 {
                                break 'label ReturnCode::Ok;
                            }
                        }

                        self.length = 0;

                        break 'blk Mode::Name;
                    }
                    Mode::Name => {
                        if (self.gzip_flags & 0x0800) != 0 {
                            if bit_reader.bytes_remaining() == 0 {
                                break 'label ReturnCode::Ok;
                            }

                            // the name string will always be null-terminated, but might be longer than we have
                            // space for in the header struct. Nonetheless, we read the whole thing.
                            let slice = bit_reader.as_slice();
                            let null_terminator_index = slice.iter().position(|c| *c == 0);

                            // we include the null terminator if it exists
                            let name_slice = match null_terminator_index {
                                Some(i) => &slice[..=i],
                                None => slice,
                            };

                            // if the header has space, store as much as possible in there
                            if let Some(mut head) = self.head {
                                self.length += head.write_name(self.length, name_slice);
                            }

                            if (self.gzip_flags & 0x0200) != 0 && (self.wrap & 4) != 0 {
                                self.checksum = crc32(self.checksum, name_slice);
                            }

                            let reached_end = name_slice.last() == Some(&0);
                            bit_reader.advance(name_slice.len());

                            if !reached_end && bit_reader.bytes_remaining() == 0 {
                                break 'label ReturnCode::Ok;
                            }
                        } else if let Some(mut head) = self.head {
                            head.clear_name();
                        }

                        self.length = 0;

                        break 'blk Mode::Comment;
                    }
                    Mode::Comment => {
                        if (self.gzip_flags & 0x01000) != 0 {
                            if bit_reader.bytes_remaining() == 0 {
                                break 'label ReturnCode::Ok;
                            }

                            // the comment string will always be null-terminated, but might be longer than we have
                            // space for in the header struct. Nonetheless, we read the whole thing.
                            let slice = bit_reader.as_slice();
                            let null_terminator_index = slice.iter().position(|c| *c == 0);

                            // we include the null terminator if it exists
                            let comment_slice = match null_terminator_index {
                                Some(i) => &slice[..=i],
                                None => slice,
                            };

                            // if the header has space, store as much as possible in there
                            if let Some(mut head) = self.head {
                                self.length += head.write_comment(self.length, comment_slice);
                            }

                            if (self.gzip_flags & 0x0200) != 0 && (self.wrap & 4) != 0 {
                                self.checksum = crc32(self.checksum, comment_slice);
                            }

                            let reached_end = comment_slice.last() == Some(&0);
                            bit_reader.advance(comment_slice.len());

                            if !reached_end && bit_reader.bytes_remaining() == 0 {
                                break 'label ReturnCode::Ok;
                            }
                        } else if let Some(mut head) = self.head {
                            head.clear_comment();
                        }

                        break 'blk Mode::HCrc;
                    }
                    Mode::HCrc => {
                        if (self.gzip_flags & 0x0200) != 0 {
                            need_bits_for_mode(bit_reader, 16, mode)?;

                            if (self.wrap & 4) != 0
                                && bit_reader.hold() as u32 != (self.checksum & 0xffff)
                            {
                                mode = Mode::Bad;
                                break 'label self.bad("header crc mismatch\0");
                            }

                            bit_reader.init_bits();
                        }

                        if let Some(mut head) = self.head {
                            head.finish((self.gzip_flags >> 9) & 1);
                        }

                        // compute crc32 checksum if not in raw mode
                        if (self.wrap & 4 != 0) && self.gzip_flags != 0 {
                            self.crc_fold = Crc32Fold::new();
                            self.checksum = crate::CRC32_INITIAL_VALUE;
                        }

                        break 'blk Mode::Type;
                    }
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
                        if self.flags.contains(Flags::IS_LAST_BLOCK) {
                            bit_reader.next_byte_boundary();
                            break 'blk Mode::Check;
                        }

                        need_bits_for_mode(bit_reader, 3, mode)?;
                        let last = bit_reader.bits(1) != 0;
                        self.flags.update(Flags::IS_LAST_BLOCK, last);
                        bit_reader.drop_bits(1);

                        match bit_reader.bits(2) {
                            0b00 => {
                                traceln!("inflate:     stored block (last = {last})");

                                bit_reader.drop_bits(2);

                                break 'blk Mode::Stored;
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

                                mode = Mode::Len_;

                                bit_reader.drop_bits(2);

                                if let InflateFlush::Trees = flush {
                                    break 'label ReturnCode::Ok;
                                } else {
                                    break 'blk Mode::Len_;
                                }
                            }
                            0b10 => {
                                traceln!("inflate:     dynamic codes block (last = {last})");

                                bit_reader.drop_bits(2);

                                break 'blk Mode::Table;
                            }
                            0b11 => {
                                traceln!("inflate:     invalid block type");

                                bit_reader.drop_bits(2);

                                mode = Mode::Bad;
                                break 'label self.bad("invalid block type\0");
                            }
                            _ => {
                                // LLVM will optimize this branch away
                                unreachable!("BitReader::bits(2) only yields a value of two bits, so this match is already exhaustive")
                            }
                        }
                    }
                    Mode::Stored => {
                        bit_reader.next_byte_boundary();

                        need_bits_for_mode(bit_reader, 32, mode)?;

                        let hold = bit_reader.bits(32) as u32;

                        traceln!("hold {hold:#x}");

                        if hold as u16 != !((hold >> 16) as u16) {
                            mode = Mode::Bad;
                            break 'label self.bad("invalid stored block lengths\0");
                        }

                        self.length = hold as usize & 0xFFFF;
                        traceln!("inflate:     stored length {}", self.length);

                        bit_reader.init_bits();

                        if let InflateFlush::Trees = flush {
                            break 'label ReturnCode::Ok;
                        } else {
                            break 'blk Mode::CopyBlock;
                        }
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
                        if !cfg!(feature = "__internal-fuzz-disable-checksum") && self.wrap != 0 {
                            need_bits_for_mode(bit_reader, 32, mode)?;

                            self.total += output.len();

                            if self.wrap & 4 != 0 {
                                if self.gzip_flags != 0 {
                                    self.crc_fold.fold(output.filled());
                                    self.checksum = self.crc_fold.finish();
                                } else {
                                    self.checksum = adler32(self.checksum, output.filled());
                                }
                            }

                            let given_checksum = if self.gzip_flags != 0 {
                                bit_reader.hold() as u32
                            } else {
                                zswap32(bit_reader.hold() as u32)
                            };

                            output.mark_accounted();

                            if self.wrap & 4 != 0 && given_checksum != self.checksum {
                                mode = Mode::Bad;
                                break 'label self.bad("incorrect data check\0");
                            }

                            bit_reader.init_bits();
                        }

                        break 'blk Mode::Length;
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
                        self.nlen = bit_reader.bits(5) as usize + 257;
                        bit_reader.drop_bits(5);
                        self.ndist = bit_reader.bits(5) as usize + 1;
                        bit_reader.drop_bits(5);
                        self.ncode = bit_reader.bits(4) as usize + 4;
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
                        // permutation of code lengths ;
                        const ORDER: [u8; 19] = [
                            16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
                        ];

                        while self.have < self.ncode {
                            need_bits_for_mode(bit_reader, 3, mode)?;
                            self.lens[usize::from(ORDER[self.have])] = bit_reader.bits(3) as u16;
                            self.have += 1;
                            bit_reader.drop_bits(3);
                        }

                        while self.have < 19 {
                            self.lens[usize::from(ORDER[self.have])] = 0;
                            self.have += 1;
                        }

                        let InflateTable::Success { root, used } = inflate_table(
                            CodeType::Codes,
                            &self.lens[..19],
                            &mut self.codes_codes,
                            7,
                            &mut self.work,
                        ) else {
                            mode = Mode::Bad;
                            break 'label self.bad("invalid code lengths set\0");
                        };

                        self.next = used;
                        self.len_table.codes = Codes::Codes;
                        self.len_table.bits = root;

                        self.have = 0;

                        break 'blk Mode::CodeLens;
                    }
                    Mode::CodeLens => {
                        while self.have < self.nlen + self.ndist {
                            let here = loop {
                                let bits = bit_reader.bits(self.len_table.bits);
                                let here = self.len_table_get(bits as usize);
                                if here.bits <= bit_reader.bits_in_buffer() {
                                    break here;
                                }

                                pull_byte_for_mode(bit_reader, mode)?;
                            };

                            let here_bits = here.bits;

                            match here.val {
                                0..=15 => {
                                    bit_reader.drop_bits(here_bits);
                                    self.lens[self.have] = here.val;
                                    self.have += 1;
                                }
                                16 => {
                                    need_bits_for_mode(
                                        bit_reader,
                                        usize::from(here_bits) + 2,
                                        mode,
                                    )?;
                                    bit_reader.drop_bits(here_bits);
                                    if self.have == 0 {
                                        mode = Mode::Bad;
                                        break 'label self.bad("invalid bit length repeat\0");
                                    }

                                    let len = self.lens[self.have - 1];
                                    let copy = 3 + bit_reader.bits(2) as usize;
                                    bit_reader.drop_bits(2);

                                    if self.have + copy > self.nlen + self.ndist {
                                        mode = Mode::Bad;
                                        break 'label self.bad("invalid bit length repeat\0");
                                    }

                                    self.lens[self.have..][..copy].fill(len);
                                    self.have += copy;
                                }
                                17 => {
                                    need_bits_for_mode(
                                        bit_reader,
                                        usize::from(here_bits) + 3,
                                        mode,
                                    )?;
                                    bit_reader.drop_bits(here_bits);
                                    let copy = 3 + bit_reader.bits(3) as usize;
                                    bit_reader.drop_bits(3);

                                    if self.have + copy > self.nlen + self.ndist {
                                        mode = Mode::Bad;
                                        break 'label self.bad("invalid bit length repeat\0");
                                    }

                                    self.lens[self.have..][..copy].fill(0);
                                    self.have += copy;
                                }
                                18.. => {
                                    need_bits_for_mode(
                                        bit_reader,
                                        usize::from(here_bits) + 7,
                                        mode,
                                    )?;
                                    bit_reader.drop_bits(here_bits);
                                    let copy = 11 + bit_reader.bits(7) as usize;
                                    bit_reader.drop_bits(7);

                                    if self.have + copy > self.nlen + self.ndist {
                                        mode = Mode::Bad;
                                        break 'label self.bad("invalid bit length repeat\0");
                                    }

                                    self.lens[self.have..][..copy].fill(0);
                                    self.have += copy;
                                }
                            }
                        }

                        // check for end-of-block code (better have one)
                        if self.lens[256] == 0 {
                            mode = Mode::Bad;
                            break 'label self.bad("invalid code -- missing end-of-block\0");
                        }

                        // build code tables

                        let InflateTable::Success { root, used } = inflate_table(
                            CodeType::Lens,
                            &self.lens[..self.nlen],
                            &mut self.len_codes,
                            10,
                            &mut self.work,
                        ) else {
                            mode = Mode::Bad;
                            break 'label self.bad("invalid literal/lengths set\0");
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
                            mode = Mode::Bad;
                            break 'label self.bad("invalid distances set\0");
                        };

                        self.dist_table.bits = root;
                        self.dist_table.codes = Codes::Dist;
                        self.next += used;

                        mode = Mode::Len_;

                        if matches!(flush, InflateFlush::Trees) {
                            break 'label ReturnCode::Ok;
                        }

                        break 'blk Mode::Len_;
                    }
                    Mode::Dict => {
                        if !self.flags.contains(Flags::HAVE_DICT) {
                            break 'label ReturnCode::NeedDict;
                        }

                        self.checksum = crate::ADLER32_INITIAL_VALUE as _;

                        break 'blk Mode::Type;
                    }
                    Mode::DictId => {
                        need_bits_for_mode(bit_reader, 32, mode)?;

                        self.checksum = zswap32(bit_reader.hold() as u32);

                        bit_reader.init_bits();

                        break 'blk Mode::Dict;
                    }
                    Mode::Done => {
                        // Inflate stream terminated properly.
                        break 'label ReturnCode::StreamEnd;
                    }
                    Mode::Bad => {
                        let msg = "repeated call with bad state\0";
                        #[cfg(all(feature = "std", test))]
                        dbg!(msg);
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
                                && bit_reader.hold() as u32 != self.total as u32
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

    fn bad(&mut self, msg: &'static str) -> ReturnCode {
        #[cfg(all(feature = "std", test))]
        dbg!(msg);
        self.error_message = Some(msg);
        ReturnCode::DataError
    }

    /// Stored in the `z_stream.data_type` field
    fn decoding_state(&self) -> i32 {
        let bit_reader_bits = self.bit_buffer.bits_in_buffer() as i32;
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

/// Fast literal/length/distance decoder.
///
/// Callers enter only when at least `INFLATE_FAST_MIN_HAVE` bytes and
/// `INFLATE_FAST_MIN_LEFT` output bytes are available. The helper retains those fast-path
/// invariants with assertions in the checked reader/output primitives rather than an unsafe
/// caller contract.
fn inflate_fast_help(
    state: &mut State,
    bit_reader: &mut BitReader<'_>,
    output: &mut InflateOutput<'_>,
) {
    inflate_fast_help_impl(state, bit_reader, output);
}

#[inline(always)]
fn inflate_fast_help_impl(
    state: &mut State,
    bit_reader: &mut BitReader<'_>,
    output: &mut InflateOutput<'_>,
) {
    debug_assert!(bit_reader.bytes_remaining() >= 15);

    let lcode = state.len_table_ref();
    let dcode = state.dist_table_ref();

    // IDEA: use const generics for the bits here?
    let lmask = (1u64 << state.len_table.bits) - 1;
    let dmask = (1u64 << state.dist_table.bits) - 1;

    let window_size = state.window.size();

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
                lcode[(hold & lmask) as usize]
            } else {
                lcode[(bit_reader.hold() & lmask) as usize]
            }
        };

        if here.op == 0 {
            output.push(here.val as u8);
            bit_reader.drop_bits(here.bits);
            here = lcode[(bit_reader.hold() & lmask) as usize];

            if here.op == 0 {
                output.push(here.val as u8);
                bit_reader.drop_bits(here.bits);
                here = lcode[(bit_reader.hold() & lmask) as usize];
            }
        }

        'dolen: loop {
            bit_reader.drop_bits(here.bits);
            let op = here.op;

            if op == 0 {
                output.push(here.val as u8);
            } else if op & 16 != 0 {
                let op = op & MAX_BITS;
                let mut len = here.val + bit_reader.bits(op as usize) as u16;
                bit_reader.drop_bits(op);

                here = dcode[(bit_reader.hold() & dmask) as usize];

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
                        let dist = here.val + bit_reader.bits(op as usize) as u16;

                        if INFLATE_STRICT && dist as usize > state.dmax {
                            bad = Some("invalid distance too far back\0");
                            state.mode = Mode::Bad;
                            break 'outer;
                        }

                        bit_reader.drop_bits(op);

                        // max distance in output
                        let written = output.len();

                        if dist as usize > written {
                            // copy fropm the window
                            if (dist as usize - written) > state.window.have() {
                                if state.flags.contains(Flags::SANE) {
                                    bad = Some("invalid distance too far back\0");
                                    state.mode = Mode::Bad;
                                    break 'outer;
                                }

                                panic!("INFLATE_ALLOW_INVALID_DISTANCE_TOOFAR_ARRR")
                            }

                            let mut op = dist as usize - written;
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

                                if op < len as usize {
                                    // This case is hit when part of the match is at the end of the
                                    // window, and part of it has wrapped around to the start. Copy
                                    // the end section here, the start section will be copied below.
                                    len -= op as u16;
                                    output.extend_from_window(&state.window, from..from + op);
                                    from = 0;
                                    op = window_next;
                                }
                            }

                            let copy = Ord::min(op, len as usize);
                            output.extend_from_window(&state.window, from..from + copy);

                            if op < len as usize {
                                // here we need some bytes from the output itself
                                output.copy_match(dist as usize, len as usize - op);
                            }
                        } else {
                            output.copy_match(dist as usize, len as usize)
                        }
                    } else if (op & 64) == 0 {
                        // 2nd level distance code
                        here = dcode[(here.val + bit_reader.bits(op as usize) as u16) as usize];
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
                here = lcode[(here.val + bit_reader.bits(op as usize) as u16) as usize];
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
        let remaining = bit_reader.bytes_remaining_including_buffer();
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
    if bits == 0 {
        /* fall through */
    } else if bits < 0 {
        stream.state.bit_buffer.init_bits();
    } else if bits > 16 || stream.state.bit_buffer.bits_in_buffer() + bits as u8 > 32 {
        return ReturnCode::StreamError;
    } else {
        stream.state.bit_buffer.prime(bits as u8, value as u64);
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
    const WINDOW_PAD_SIZE: usize = 64;

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
            curr_size = pos + (1 << MAX_WBITS) + Self::WINDOW_PAD_SIZE;
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

    fn window_pos(&self) -> usize {
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
    /// For the ordinary layout this also creates the owned padded history window. For the
    /// state-only layout (used by `inflateBack`) any caller-owned window already stored in `state`
    /// is preserved. Raw alignment, placement, and allocation metadata stay confined here.
    fn allocate_state<'a>(
        &self,
        alloc: &Allocator<'a>,
        mut state: State<'a>,
    ) -> Option<NonNull<State<'a>>> {
        let (allocation_start, aligned) = self.allocate_storage(alloc)?;

        if let Some(window_pos) = self.window_pos {
            // SAFETY: `window_pos` is computed inside this allocation and leaves enough room for
            // the full padded window.
            let window_allocation = unsafe { aligned.as_ptr().add(window_pos) };
            state.window = unsafe {
                Window::from_uninitialized_raw_parts(
                    window_allocation,
                    (1 << MAX_WBITS) + Self::WINDOW_PAD_SIZE,
                )
            };
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
        // SAFETY: the ordinary layout reserves exactly the source window's full padded backing
        // length at `window_pos`, and the fresh destination allocation cannot overlap the source.
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

/// Initialize the stream in an inflate state
pub fn init(stream: &mut z_stream, config: InflateConfig) -> ReturnCode {
    stream.msg = core::ptr::null_mut();

    // for safety we must really make sure that alloc and free are consistent
    // this is a (slight) deviation from stock zlib. In this crate we pick the rust
    // allocator as the default, but `libz-rs-sys` configures the C allocator
    #[cfg(feature = "rust-allocator")]
    if stream.zalloc.is_none() || stream.zfree.is_none() {
        stream.configure_default_rust_allocator()
    }

    #[cfg(feature = "c-allocator")]
    if stream.zalloc.is_none() || stream.zfree.is_none() {
        stream.configure_default_c_allocator()
    }

    if stream.zalloc.is_none() || stream.zfree.is_none() {
        return ReturnCode::StreamError;
    }

    let state = State::new(InflateStateKind::Regular);

    let alloc = Allocator {
        zalloc: stream.zalloc.unwrap(),
        zfree: stream.zfree.unwrap(),
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
    let mut window_bits = config.window_bits;
    let wrap;

    if window_bits < 0 {
        wrap = 0;

        if window_bits < -MAX_WBITS {
            return ReturnCode::StreamError;
        }

        window_bits = -window_bits;
    } else {
        wrap = (window_bits >> 4) + 5; // TODO wth?

        if window_bits < 48 {
            window_bits &= MAX_WBITS;
        }
    }

    if window_bits != 0 && !(MIN_WBITS..=MAX_WBITS).contains(&window_bits) {
        traceln!("invalid windowBits");
        return ReturnCode::StreamError;
    }

    stream.state.wrap = wrap as u8;
    stream.state.wbits = window_bits as _;

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
        stream.adler = (state.wrap & 1) as _;
    }

    state.mode = Mode::Head;
    state.checksum = crate::ADLER32_INITIAL_VALUE as u32;

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

/// Bind one zlib ABI input/output pair and run ordinary inflate.
///
/// # Safety
/// Any non-null input span must be readable for `avail_in` bytes and any non-null output span must
/// be uniquely writable for `avail_out` bytes. The input/output spans must not overlap each other
/// or the live stream/state objects. Null input with zero length is valid; null output is rejected
/// before it is dereferenced.
pub unsafe fn inflate(stream: &mut InflateStream, flush: InflateFlush) -> ReturnCode {
    if stream.next_out.is_null() || (stream.next_in.is_null() && stream.avail_in != 0) {
        return ReturnCode::StreamError;
    }

    let input = if stream.avail_in == 0 {
        &[][..]
    } else {
        // SAFETY: `inflate` requires `next_in` to address `avail_in` readable bytes. The slice is
        // call-local and cannot survive this invocation.
        unsafe { core::slice::from_raw_parts(stream.next_in, stream.avail_in as usize) }
    };
    // SAFETY: `stream.next_out` is non-null and points to at least `stream.avail_out` writable bytes.
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

    let avail_in = bit_reader.bytes_remaining();
    let in_read = initial_avail_in - avail_in;
    // Preserve the exact caller pointer when no input was consumed. This matters for zlib's legal
    // NULL + zero-length input representation; Rust's canonical empty slice has a non-null sentinel
    // pointer that must not leak back through the C ABI.
    let next_in = (in_read != 0).then(|| bit_reader.next_input_ptr());
    let out_written = output.written_since_accounted();

    stream.total_in += in_read as z_size;
    state.total = state.total.wrapping_add(out_written);
    stream.total_out = state.total as _;

    stream.avail_in = avail_in as u32;
    if let Some(next_in) = next_in {
        stream.next_in = next_in as *mut u8;
    }
    state.bit_buffer = bit_reader.into_buffer();

    stream.avail_out = output.remaining() as u32;
    stream.next_out = output.next_out() as *mut u8;

    stream.adler = state.checksum as z_checksum;

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

    let update_checksum = state.wrap & 4 != 0;

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
        stream.msg = msg.as_ptr() as *mut u8 as *mut core::ffi::c_char;
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

fn syncsearch(mut got: usize, buf: &[u8]) -> (usize, usize) {
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

pub fn sync(stream: &mut InflateStream) -> ReturnCode {
    let state = &mut *stream.state;

    if stream.next_in.is_null() && stream.avail_in != 0 {
        return ReturnCode::StreamError;
    }
    if stream.avail_in == 0 && state.bit_buffer.bits_in_buffer() < 8 {
        return ReturnCode::BufError;
    }
    /* if first time, start search in bit buffer */
    if !matches!(state.mode, Mode::Sync) {
        state.mode = Mode::Sync;

        let (buf, len) = state.bit_buffer.start_sync_search();

        (state.have, _) = syncsearch(0, &buf[..len]);
    }

    // Search available input. A zlib caller may legitimately provide NULL when `avail_in == 0`;
    // Rust slices still require a non-null pointer even at length zero, so represent that case with
    // the canonical empty slice instead of materializing `from_raw_parts(NULL, 0)`.
    let slice = if stream.avail_in == 0 {
        &[][..]
    } else {
        // SAFETY: the public FFI contract requires the non-null input span to be readable for
        // `avail_in` bytes, and the null/nonzero case was rejected above.
        unsafe { core::slice::from_raw_parts(stream.next_in, stream.avail_in as usize) }
    };

    let len;
    (state.have, len) = syncsearch(state.have, slice);
    if len != 0 {
        // SAFETY: `syncsearch` returns an in-bounds consumed prefix; `len != 0` also proves the
        // source pointer came from the non-empty, non-null branch above.
        stream.next_in = unsafe { stream.next_in.add(len) };
    }
    stream.avail_in -= len as u32;
    stream.total_in += len as z_size;

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
    let total_in = stream.total_in;
    let total_out = stream.total_out;

    reset(stream);

    stream.total_in = total_in;
    stream.total_out = total_out;

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
    stream.state.flags.update(Flags::SANE, (!subvert) != 0);

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

pub fn mark(stream: &InflateStream) -> c_long {
    if stream.next_out.is_null() || (stream.next_in.is_null() && stream.avail_in != 0) {
        return c_long::MIN;
    }

    let state = &*stream.state;

    let length = match state.mode {
        Mode::CopyBlock => state.length,
        Mode::Match => state.was - state.length,
        _ => 0,
    };

    (((state.back as c_long) as c_ulong) << 16) as c_long + length as c_long
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
    fn end_leaves_null_abi_state_pointer() {
        let mut raw = z_stream::default();
        assert_eq!(init(&mut raw, InflateConfig::default()), ReturnCode::Ok);
        let stream = unsafe { InflateStream::from_stream_mut(&mut raw) }.unwrap();

        assert!(end(stream).state.is_null());
        assert!(end(stream).state.is_null());
    }

    #[test]
    fn inflate_copy_owns_independent_state_and_window_storage() {
        let mut raw = z_stream::default();
        assert_eq!(init(&mut raw, InflateConfig::default()), ReturnCode::Ok);
        let source = unsafe { InflateStream::from_stream_mut(&mut raw) }.unwrap();
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
        let source = unsafe { InflateStream::from_stream_mut(&mut source_stream) }.unwrap();
        assert_eq!(
            unsafe { get_header(source, Some(&mut header)) },
            ReturnCode::Ok
        );

        // Configure I/O before copying, but do not call inflate yet: both states must retain the
        // same caller-owned header destination just as zlib's copied C pointer does.
        let mut source_out = [0u8; 1];
        source.next_in = input.as_ptr() as *mut _;
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
        let stream = unsafe { InflateStream::from_stream_mut(&mut raw) }.unwrap();

        let mut output = vec![0u8; expected.len() + output_chunk];
        let mut next_input = 0usize;
        let mut next_output = 0usize;

        loop {
            if stream.avail_in == 0 && next_input < compressed.len() {
                let len = input_chunk.min(compressed.len() - next_input);
                stream.next_in = compressed[next_input..].as_ptr() as *mut u8;
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
}
