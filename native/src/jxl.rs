//! JPEG XL, the one format pixport doesn't implement itself: read and written with libjxl, the
//! reference implementation (built from source by build.rs and called through the C functions
//! declared in `ffi` below).
//!
//! Decoding gives the first frame (an animation's frame count and durations come along) with the
//! orientation applied and alpha straight (not premultiplied). Pictures with an ordinary (SDR)
//! transfer curve come out in their own colour space, as 8- or 16-bit values or float32 for
//! floating-point pictures, with the ICC profile that describes the space unless it's sRGB; CMYK
//! pictures are converted to sRGB. HDR (PQ, HLG) and linear floating-point pictures come out as
//! float32 linear light in sRGB/Rec.709 primaries (values above 1 and, for wider gamuts, below 0
//! are kept).

use crate::{raise, Damage, Res};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::os::raw::{c_int, c_void};
use std::sync::atomic::{AtomicIsize, Ordering};

// the caps every pixport reader keeps to
const MAX_SIDE: u64 = 50_000;
const MAX_PIXELS: u64 = 1 << 28; // pixport.limits' default; pixport passes its own budget when it differs
// metadata boxes and ICC profiles bigger than this are damage
const MAX_BOX: usize = 64 << 20;

#[allow(dead_code)]
mod ffi {
    use std::os::raw::{c_char, c_int, c_void};

    pub type JxlBool = c_int;

    pub const TYPE_FLOAT: c_int = 0;
    pub const TYPE_UINT8: c_int = 2;
    pub const TYPE_UINT16: c_int = 3;

    pub const ENC_SUCCESS: c_int = 0;
    pub const ENC_NEED_MORE_OUTPUT: c_int = 2;
    pub const SETTING_EFFORT: c_int = 0;

    pub const DEC_SUCCESS: c_int = 0;
    pub const DEC_ERROR: c_int = 1;
    pub const DEC_NEED_MORE_INPUT: c_int = 2;
    pub const DEC_NEED_IMAGE_OUT_BUFFER: c_int = 5;
    pub const DEC_JPEG_NEED_MORE_OUTPUT: c_int = 6;
    pub const DEC_BOX_NEED_MORE_OUTPUT: c_int = 7;
    pub const DEC_BASIC_INFO: c_int = 0x40;
    pub const DEC_COLOR_ENCODING: c_int = 0x100;
    pub const DEC_FRAME: c_int = 0x400;
    pub const DEC_FULL_IMAGE: c_int = 0x1000;
    pub const DEC_JPEG_RECONSTRUCTION: c_int = 0x2000;
    pub const DEC_BOX: c_int = 0x4000;
    pub const TARGET_ORIGINAL: c_int = 0;
    pub const TARGET_DATA: c_int = 1;

    pub const SPACE_RGB: c_int = 0;
    pub const SPACE_GRAY: c_int = 1;
    pub const WHITE_D65: c_int = 1;
    pub const PRIMARIES_SRGB: c_int = 1;
    pub const TF_LINEAR: c_int = 8;
    pub const TF_SRGB: c_int = 13;
    pub const TF_PQ: c_int = 16;
    pub const TF_HLG: c_int = 18;

    #[repr(C)]
    pub struct PreviewHeader {
        pub xsize: u32,
        pub ysize: u32,
    }

    #[repr(C)]
    pub struct AnimationHeader {
        pub tps_numerator: u32,
        pub tps_denominator: u32,
        pub num_loops: u32,
        pub have_timecodes: JxlBool,
    }

    #[repr(C)]
    pub struct BasicInfo {
        pub have_container: JxlBool,
        pub xsize: u32,
        pub ysize: u32,
        pub bits_per_sample: u32,
        pub exponent_bits_per_sample: u32,
        pub intensity_target: f32,
        pub min_nits: f32,
        pub relative_to_max_display: JxlBool,
        pub linear_below: f32,
        pub uses_original_profile: JxlBool,
        pub have_preview: JxlBool,
        pub have_animation: JxlBool,
        pub orientation: c_int,
        pub num_color_channels: u32,
        pub num_extra_channels: u32,
        pub alpha_bits: u32,
        pub alpha_exponent_bits: u32,
        pub alpha_premultiplied: JxlBool,
        pub preview: PreviewHeader,
        pub animation: AnimationHeader,
        pub intrinsic_xsize: u32,
        pub intrinsic_ysize: u32,
        pub padding: [u8; 100],
    }

    #[repr(C)]
    pub struct PixelFormat {
        pub num_channels: u32,
        pub data_type: c_int,
        pub endianness: c_int,
        pub align: usize,
    }

    #[repr(C)]
    pub struct ColorEncoding {
        pub color_space: c_int,
        pub white_point: c_int,
        pub white_point_xy: [f64; 2],
        pub primaries: c_int,
        pub primaries_red_xy: [f64; 2],
        pub primaries_green_xy: [f64; 2],
        pub primaries_blue_xy: [f64; 2],
        pub transfer_function: c_int,
        pub gamma: f64,
        pub rendering_intent: c_int,
    }

    #[repr(C)]
    pub struct BlendInfo {
        pub blendmode: c_int,
        pub source: u32,
        pub alpha: u32,
        pub clamp: JxlBool,
    }

    #[repr(C)]
    pub struct LayerInfo {
        pub have_crop: JxlBool,
        pub crop_x0: i32,
        pub crop_y0: i32,
        pub xsize: u32,
        pub ysize: u32,
        pub blend_info: BlendInfo,
        pub save_as_reference: u32,
    }

    #[repr(C)]
    pub struct FrameHeader {
        pub duration: u32,
        pub timecode: u32,
        pub name_length: u32,
        pub is_last: JxlBool,
        pub layer_info: LayerInfo,
    }

    #[repr(C)]
    pub struct MemoryManager {
        pub opaque: *mut c_void,
        pub alloc: unsafe extern "C" fn(*mut c_void, usize) -> *mut c_void,
        pub free: unsafe extern "C" fn(*mut c_void, *mut c_void),
    }

    /// libjxl's colour management (skcms), passed by value to the decoder.
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct CmsInterface {
        pub set_fields_data: *mut c_void,
        pub set_fields_from_icc: *const c_void,
        pub init_data: *mut c_void,
        pub init: *const c_void,
        pub get_src_buf: *const c_void,
        pub get_dst_buf: *const c_void,
        pub run: *const c_void,
        pub destroy: *const c_void,
    }

    pub type ParallelRunner = unsafe extern "C" fn(*mut c_void, *mut c_void, *const c_void, *const c_void, u32, u32) -> c_int;

    extern "C" {
        pub fn JxlEncoderVersion() -> u32;
        pub fn JxlEncoderCreate(memory_manager: *const MemoryManager) -> *mut c_void;
        pub fn JxlEncoderDestroy(enc: *mut c_void);
        pub fn JxlEncoderSetParallelRunner(enc: *mut c_void, runner: ParallelRunner, opaque: *mut c_void) -> c_int;
        pub fn JxlEncoderGetError(enc: *mut c_void) -> c_int;
        pub fn JxlEncoderProcessOutput(enc: *mut c_void, next_out: *mut *mut u8, avail_out: *mut usize) -> c_int;
        pub fn JxlEncoderAddImageFrame(fs: *const c_void, format: *const PixelFormat, buffer: *const c_void, size: usize) -> c_int;
        pub fn JxlEncoderAddJPEGFrame(fs: *const c_void, buffer: *const u8, size: usize) -> c_int;
        pub fn JxlEncoderStoreJPEGMetadata(enc: *mut c_void, store: JxlBool) -> c_int;
        pub fn JxlEncoderAddBox(enc: *mut c_void, kind: *const c_char, contents: *const u8, size: usize, compress: JxlBool) -> c_int;
        pub fn JxlEncoderUseBoxes(enc: *mut c_void) -> c_int;
        pub fn JxlEncoderCloseBoxes(enc: *mut c_void);
        pub fn JxlEncoderCloseInput(enc: *mut c_void);
        pub fn JxlEncoderSetColorEncoding(enc: *mut c_void, color: *const ColorEncoding) -> c_int;
        pub fn JxlEncoderSetICCProfile(enc: *mut c_void, icc: *const u8, size: usize) -> c_int;
        pub fn JxlEncoderInitBasicInfo(info: *mut BasicInfo);
        pub fn JxlEncoderSetBasicInfo(enc: *mut c_void, info: *const BasicInfo) -> c_int;
        pub fn JxlEncoderFrameSettingsSetOption(fs: *mut c_void, option: c_int, value: i64) -> c_int;
        pub fn JxlEncoderSetFrameLossless(fs: *mut c_void, lossless: JxlBool) -> c_int;
        pub fn JxlEncoderSetFrameDistance(fs: *mut c_void, distance: f32) -> c_int;
        pub fn JxlEncoderFrameSettingsCreate(enc: *mut c_void, source: *const c_void) -> *mut c_void;
        pub fn JxlColorEncodingSetToSRGB(color: *mut ColorEncoding, is_gray: JxlBool);
        pub fn JxlColorEncodingSetToLinearSRGB(color: *mut ColorEncoding, is_gray: JxlBool);

        pub fn JxlDecoderCreate(memory_manager: *const MemoryManager) -> *mut c_void;
        pub fn JxlDecoderDestroy(dec: *mut c_void);
        pub fn JxlDecoderSetParallelRunner(dec: *mut c_void, runner: ParallelRunner, opaque: *mut c_void) -> c_int;
        pub fn JxlDecoderSubscribeEvents(dec: *mut c_void, events: c_int) -> c_int;
        pub fn JxlDecoderSetUnpremultiplyAlpha(dec: *mut c_void, on: JxlBool) -> c_int;
        pub fn JxlDecoderSetDecompressBoxes(dec: *mut c_void, on: JxlBool) -> c_int;
        pub fn JxlDecoderSetCms(dec: *mut c_void, cms: CmsInterface) -> c_int;
        pub fn JxlDecoderSetInput(dec: *mut c_void, data: *const u8, size: usize) -> c_int;
        pub fn JxlDecoderCloseInput(dec: *mut c_void);
        pub fn JxlDecoderProcessInput(dec: *mut c_void) -> c_int;
        pub fn JxlDecoderGetBasicInfo(dec: *const c_void, info: *mut BasicInfo) -> c_int;
        pub fn JxlDecoderGetColorAsEncodedProfile(dec: *const c_void, target: c_int, enc: *mut ColorEncoding) -> c_int;
        pub fn JxlDecoderGetICCProfileSize(dec: *const c_void, target: c_int, size: *mut usize) -> c_int;
        pub fn JxlDecoderGetColorAsICCProfile(dec: *const c_void, target: c_int, icc: *mut u8, size: usize) -> c_int;
        pub fn JxlDecoderSetOutputColorProfile(dec: *mut c_void, enc: *const ColorEncoding, icc: *const u8, size: usize) -> c_int;
        pub fn JxlDecoderGetFrameHeader(dec: *const c_void, header: *mut FrameHeader) -> c_int;
        pub fn JxlDecoderImageOutBufferSize(dec: *const c_void, format: *const PixelFormat, size: *mut usize) -> c_int;
        pub fn JxlDecoderSetImageOutBuffer(dec: *mut c_void, format: *const PixelFormat, buf: *mut c_void, size: usize) -> c_int;
        pub fn JxlDecoderGetBoxType(dec: *mut c_void, kind: *mut c_char, decompressed: JxlBool) -> c_int;
        pub fn JxlDecoderSetBoxBuffer(dec: *mut c_void, data: *mut u8, size: usize) -> c_int;
        pub fn JxlDecoderReleaseBoxBuffer(dec: *mut c_void) -> usize;
        pub fn JxlDecoderSetJPEGBuffer(dec: *mut c_void, data: *mut u8, size: usize) -> c_int;
        pub fn JxlDecoderReleaseJPEGBuffer(dec: *mut c_void) -> usize;

        pub fn JxlGetDefaultCms() -> *const CmsInterface;

        pub fn JxlThreadParallelRunner(runner: *mut c_void, opaque: *mut c_void, init: *const c_void, func: *const c_void, start: u32,
                                       end: u32) -> c_int;
        pub fn JxlThreadParallelRunnerCreate(memory_manager: *const MemoryManager, num_threads: usize) -> *mut c_void;
        pub fn JxlThreadParallelRunnerDestroy(runner: *mut c_void);
        pub fn JxlThreadParallelRunnerDefaultNumWorkerThreads() -> usize;
    }
}

// ---------------------------------------------------------------------------------------------
// what libjxl may allocate: a budget set from the picture's size, so a hostile header ends in a
// clean error instead of exhausting memory

struct Budget {
    limit: AtomicIsize,
    used: AtomicIsize,
}

const HEAD: usize = 16;

unsafe extern "C" fn budget_alloc(opaque: *mut c_void, size: usize) -> *mut c_void {
    // SAFETY: opaque is the Budget boxed by Session, alive while libjxl runs
    let b = unsafe { &*(opaque as *const Budget) };
    let Some(total) = size.checked_add(HEAD) else { return std::ptr::null_mut() };
    let Ok(layout) = std::alloc::Layout::from_size_align(total, HEAD) else { return std::ptr::null_mut() };
    let cost = total as isize;
    let before = b.used.fetch_add(cost, Ordering::Relaxed);
    if before.saturating_add(cost) > b.limit.load(Ordering::Relaxed) {
        b.used.fetch_sub(cost, Ordering::Relaxed);
        return std::ptr::null_mut();
    }
    // SAFETY: layout has a non-zero size; the size is kept in front of the block for budget_free
    unsafe {
        let p = std::alloc::alloc(layout);
        if p.is_null() {
            b.used.fetch_sub(cost, Ordering::Relaxed);
            return std::ptr::null_mut();
        }
        (p as *mut usize).write(total);
        p.add(HEAD).cast()
    }
}

unsafe extern "C" fn budget_free(opaque: *mut c_void, address: *mut c_void) {
    if address.is_null() {
        return;
    }
    // SAFETY: address came from budget_alloc, which stored the block's size just before it
    unsafe {
        let b = &*(opaque as *const Budget);
        let p = (address as *mut u8).sub(HEAD);
        let total = (p as *const usize).read();
        std::alloc::dealloc(p, std::alloc::Layout::from_size_align_unchecked(total, HEAD));
        b.used.fetch_sub(total as isize, Ordering::Relaxed);
    }
}

/// A libjxl decoder or encoder with its thread pool and memory budget, released however it ends.
struct Session {
    handle: *mut c_void,
    runner: *mut c_void,
    encoder: bool,
    budget: Box<Budget>,
    _mm: Box<ffi::MemoryManager>,
}

impl Session {
    fn new(encoder: bool, budget: isize) -> Res<Self> {
        let budget = Box::new(Budget { limit: AtomicIsize::new(budget), used: AtomicIsize::new(0) });
        let mm = Box::new(ffi::MemoryManager {
            opaque: (&*budget as *const Budget).cast_mut().cast(),
            alloc: budget_alloc,
            free: budget_free,
        });
        // SAFETY: mm and budget are boxed and dropped after the handle and runner
        unsafe {
            let handle = if encoder { ffi::JxlEncoderCreate(&*mm) } else { ffi::JxlDecoderCreate(&*mm) };
            if handle.is_null() {
                return Err(Damage::new("JPEG XL: can't start libjxl"));
            }
            let runner = ffi::JxlThreadParallelRunnerCreate(std::ptr::null(), threads());
            let s = Session { handle, runner, encoder, budget, _mm: mm };
            if !runner.is_null() {
                let status = if encoder {
                    ffi::JxlEncoderSetParallelRunner(handle, ffi::JxlThreadParallelRunner, runner)
                } else {
                    ffi::JxlDecoderSetParallelRunner(handle, ffi::JxlThreadParallelRunner, runner)
                };
                if status != 0 {
                    return Err(Damage::new("JPEG XL: can't start libjxl's threads"));
                }
            }
            Ok(s)
        }
    }

    /// A new limit for everything libjxl holds, what it has already allocated included.
    fn set_budget(&self, bytes: u64) {
        self.budget.limit.store(isize::try_from(bytes).unwrap_or(isize::MAX), Ordering::Relaxed);
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // SAFETY: made by libjxl, released once
        unsafe {
            if self.encoder {
                ffi::JxlEncoderDestroy(self.handle);
            } else {
                ffi::JxlDecoderDestroy(self.handle);
            }
            if !self.runner.is_null() {
                ffi::JxlThreadParallelRunnerDestroy(self.runner);
            }
        }
    }
}

/// One thread per core, or fewer when PIXPORT_THREADS says so.
fn threads() -> usize {
    // SAFETY: no arguments, no state
    let cores = unsafe { ffi::JxlThreadParallelRunnerDefaultNumWorkerThreads() };
    match std::env::var("PIXPORT_THREADS").ok().and_then(|v| v.trim().parse::<usize>().ok()) {
        Some(n) if n > 0 => n.min(cores.max(1)),
        _ => cores,
    }
}

fn check_size(w: u64, h: u64, max_pixels: u64) -> Res<()> {
    if w == 0 || h == 0 || w > MAX_SIDE || h > MAX_SIDE || w * h > max_pixels {
        return Err(Damage::new(format!("JPEG XL: {w}x{h} pixels is past pixport's limits")));
    }
    Ok(())
}

/// A zeroed buffer of n values, or an error when the memory isn't there (a plain vec! would end
/// the whole process).
fn zeroed<T: Clone + Default>(n: usize) -> Res<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(|_| bad("not enough memory for the picture"))?;
    v.resize(n, T::default());
    Ok(v)
}

fn encoding(srgb: bool, grey: bool) -> ffi::ColorEncoding {
    // SAFETY: a plain struct libjxl fills in
    unsafe {
        let mut c: ffi::ColorEncoding = std::mem::zeroed();
        if srgb {
            ffi::JxlColorEncodingSetToSRGB(&mut c, grey as ffi::JxlBool);
        } else {
            ffi::JxlColorEncodingSetToLinearSRGB(&mut c, grey as ffi::JxlBool);
        }
        c
    }
}

// ---------------------------------------------------------------------------------------------
// decoding

enum Pixels {
    U8(Vec<u8>),
    U16(Vec<u16>),
    F32(Vec<f32>),
}

struct Decoded {
    width: usize,
    height: usize,
    channels: usize,
    pixels: Pixels,
    icc: Option<Vec<u8>>,
    exif: Option<Vec<u8>>,
    xmp: Option<Vec<u8>>,
    linear: bool,
    hdr: Option<&'static str>,
    original_bits: u32,
    float: bool,
    animated: bool,
    frames: usize,
    durations: Vec<f64>,
    loops: Option<u32>,
    jpeg: bool,
}

fn bad(what: &str) -> Damage {
    Damage::new(format!("JPEG XL: {what}"))
}

/// An Exif or XMP box being copied out of the file.
struct BoxReader {
    kind: u8,
    buf: Vec<u8>,
}

impl BoxReader {
    /// The box's bytes, once libjxl has moved on.
    fn finish(&mut self, dec: *mut c_void, exif: &mut Option<Vec<u8>>, xmp: &mut Option<Vec<u8>>) {
        if self.kind == 0 {
            return;
        }
        // SAFETY: a box buffer is set while kind != 0
        let rest = unsafe { ffi::JxlDecoderReleaseBoxBuffer(dec) };
        let mut b = std::mem::take(&mut self.buf);
        b.truncate(b.len().saturating_sub(rest));
        if self.kind == 1 {
            // an Exif box starts with the offset of the TIFF header
            if b.len() >= 4 && exif.is_none() {
                let off = u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as usize;
                *exif = b.get(4usize.saturating_add(off)..).map(<[u8]>::to_vec);
            }
        } else if xmp.is_none() {
            *xmp = Some(b);
        }
        self.kind = 0;
    }
}

/// Decode a file: the first frame's pixels and the metadata boxes, plus a second, pixel-less
/// pass over animations to count their frames. An XYB picture whose ICC profile libjxl can't
/// convert back to comes out as sRGB.
fn decode(data: &[u8], to_srgb: bool, max_pixels: u64) -> Res<Decoded> {
    let mut d = match decode_pass(data, to_srgb, true, max_pixels) {
        // only a picture whose own colours libjxl can't give is tried again, as sRGB
        Err(e) if !to_srgb && e.0 == COLOUR_FAIL => decode_pass(data, true, true, max_pixels)?,
        r => r?,
    };
    if d.animated {
        let frames = decode_pass(data, to_srgb, false, max_pixels)?;
        d.frames = frames.frames;
        d.durations = frames.durations;
    }
    Ok(d)
}

const COLOUR_FAIL: &str = "JPEG XL: can't convert the colours";

fn decode_pass(data: &[u8], to_srgb: bool, pixels: bool, max_pixels: u64) -> Res<Decoded> {
    use ffi::*;
    let events = if pixels {
        DEC_BASIC_INFO | DEC_COLOR_ENCODING | DEC_FULL_IMAGE | DEC_JPEG_RECONSTRUCTION | DEC_BOX
    } else {
        DEC_BASIC_INFO | DEC_FRAME
    };
    let s = Session::new(false, 256 << 20)?;
    let dec = s.handle;
    // SAFETY: a zeroed BasicInfo is valid; libjxl fills it in
    let mut info: BasicInfo = unsafe { std::mem::zeroed() };
    let mut format = PixelFormat { num_channels: 0, data_type: TYPE_UINT8, endianness: 0, align: 0 };
    let mut out = Pixels::U8(Vec::new());
    let (mut linear, mut hdr, mut float, mut jpeg, mut have_image) = (false, None, false, false, false);
    let mut icc: Option<Vec<u8>> = None;
    let (mut frames, mut durations, mut tick) = (0usize, Vec::new(), 0.0f64);
    let (mut exif, mut xmp): (Option<Vec<u8>>, Option<Vec<u8>>) = (None, None);
    let mut reading = BoxReader { kind: 0, buf: Vec::new() };
    // SAFETY: every buffer handed to libjxl outlives its use (they live until the loop ends)
    unsafe {
        if JxlDecoderSubscribeEvents(dec, events) != DEC_SUCCESS {
            return Err(bad("decoder setup"));
        }
        JxlDecoderSetUnpremultiplyAlpha(dec, 1);
        JxlDecoderSetDecompressBoxes(dec, 1);
        JxlDecoderSetCms(dec, *JxlGetDefaultCms());
        JxlDecoderSetInput(dec, data.as_ptr(), data.len());
        JxlDecoderCloseInput(dec);
        loop {
            match JxlDecoderProcessInput(dec) {
                DEC_SUCCESS => {
                    reading.finish(dec, &mut exif, &mut xmp);
                    break;
                }
                DEC_ERROR => return Err(bad("the file is damaged")),
                DEC_NEED_MORE_INPUT => return Err(bad("the file is cut short")),
                DEC_BASIC_INFO => {
                    if JxlDecoderGetBasicInfo(dec, &mut info) != DEC_SUCCESS {
                        return Err(bad("damaged header"));
                    }
                    let (w, h) = (info.xsize as u64, info.ysize as u64);
                    check_size(w, h, max_pixels)?;
                    // frames, reference frames and the render pipeline work on float planes
                    let planes = (info.num_color_channels.max(3) + info.num_extra_channels) as u64;
                    s.set_budget((w * h * planes * 4).saturating_mul(8).saturating_add(512 << 20));
                    float = info.exponent_bits_per_sample > 0;
                    if info.have_animation != 0 && info.animation.tps_numerator > 0 {
                        tick = info.animation.tps_denominator as f64 / info.animation.tps_numerator as f64;
                    }
                }
                DEC_COLOR_ENCODING => {
                    let grey = info.num_color_channels == 1;
                    let mut orig: ColorEncoding = std::mem::zeroed();
                    let is_enum = JxlDecoderGetColorAsEncodedProfile(dec, TARGET_ORIGINAL, &mut orig) == DEC_SUCCESS;
                    let orig_icc = profile(dec, TARGET_ORIGINAL);
                    let cmyk = orig_icc.as_deref().and_then(|p| p.get(16..20)) == Some(b"CMYK");
                    let tf = if is_enum { orig.transfer_function } else { 0 };
                    hdr = match tf {
                        TF_PQ => Some("pq"),
                        TF_HLG => Some("hlg"),
                        _ => None,
                    };
                    linear = hdr.is_some() || (float && tf == TF_LINEAR);
                    let status = if linear {
                        JxlDecoderSetOutputColorProfile(dec, &encoding(false, grey), std::ptr::null(), 0)
                    } else if cmyk || to_srgb {
                        JxlDecoderSetOutputColorProfile(dec, &encoding(true, grey && !cmyk), std::ptr::null(), 0)
                    } else if !is_enum && info.uses_original_profile == 0 {
                        // XYB pictures come back in their original ICC space
                        match &orig_icc {
                            Some(p) => JxlDecoderSetOutputColorProfile(dec, std::ptr::null(), p.as_ptr(), p.len()),
                            None => DEC_SUCCESS,
                        }
                    } else {
                        DEC_SUCCESS
                    };
                    if status != DEC_SUCCESS {
                        return Err(Damage::new(COLOUR_FAIL));
                    }
                    let plain = is_enum
                        && orig.white_point == WHITE_D65
                        && orig.transfer_function == TF_SRGB
                        && (orig.color_space == SPACE_GRAY || (orig.color_space == SPACE_RGB && orig.primaries == PRIMARIES_SRGB));
                    if !(linear || cmyk || to_srgb || plain) {
                        icc = profile(dec, TARGET_DATA);
                    }
                    format.num_channels = if grey && !cmyk { 1 } else { 3 } + (info.alpha_bits > 0) as u32;
                    format.data_type = if linear || float {
                        TYPE_FLOAT
                    } else if info.bits_per_sample <= 8 {
                        TYPE_UINT8
                    } else {
                        TYPE_UINT16
                    };
                }
                DEC_FRAME => {
                    let mut fh: FrameHeader = std::mem::zeroed();
                    if JxlDecoderGetFrameHeader(dec, &mut fh) == DEC_SUCCESS {
                        durations.push(fh.duration as f64 * tick);
                    }
                    frames += 1;
                }
                DEC_NEED_IMAGE_OUT_BUFFER => {
                    let mut size = 0usize;
                    if have_image || JxlDecoderImageOutBufferSize(dec, &format, &mut size) != DEC_SUCCESS {
                        return Err(bad("output buffer"));
                    }
                    let ptr: *mut c_void = match format.data_type {
                        TYPE_UINT8 => {
                            let mut v = zeroed::<u8>(size)?;
                            let p = v.as_mut_ptr().cast();
                            out = Pixels::U8(v);
                            p
                        }
                        TYPE_UINT16 => {
                            let mut v = zeroed::<u16>(size.div_ceil(2))?;
                            let p = v.as_mut_ptr().cast();
                            out = Pixels::U16(v);
                            p
                        }
                        _ => {
                            let mut v = zeroed::<f32>(size.div_ceil(4))?;
                            let p = v.as_mut_ptr().cast();
                            out = Pixels::F32(v);
                            p
                        }
                    };
                    if JxlDecoderSetImageOutBuffer(dec, &format, ptr, size) != DEC_SUCCESS {
                        return Err(bad("output buffer"));
                    }
                }
                DEC_FULL_IMAGE => {
                    have_image = true;
                    if info.have_animation != 0 {
                        // the other frames are only counted, in a pass of their own
                        break;
                    }
                }
                DEC_JPEG_RECONSTRUCTION => jpeg = true,
                DEC_BOX => {
                    reading.finish(dec, &mut exif, &mut xmp);
                    let mut kind = [0 as std::os::raw::c_char; 4];
                    if JxlDecoderGetBoxType(dec, kind.as_mut_ptr(), 1) == DEC_SUCCESS {
                        reading.kind = match &kind.map(|c| c as u8) {
                            b"Exif" if exif.is_none() => 1,
                            b"xml " if xmp.is_none() => 2,
                            _ => 0,
                        };
                        if reading.kind != 0 {
                            reading.buf = vec![0u8; 1 << 16];
                            JxlDecoderSetBoxBuffer(dec, reading.buf.as_mut_ptr(), reading.buf.len());
                        }
                    }
                }
                DEC_BOX_NEED_MORE_OUTPUT => {
                    let rest = JxlDecoderReleaseBoxBuffer(dec);
                    let filled = reading.buf.len().saturating_sub(rest);
                    if reading.buf.len() >= MAX_BOX {
                        return Err(bad("a metadata box is too big"));
                    }
                    let grow = reading.buf.len();
                    reading.buf.resize(reading.buf.len() + grow, 0);
                    JxlDecoderSetBoxBuffer(dec, reading.buf.as_mut_ptr().add(filled), reading.buf.len() - filled);
                }
                _ => return Err(bad("unexpected decoder state")),
            }
        }
    }
    if pixels && !have_image {
        return Err(bad("no picture in the file"));
    }
    if tick == 0.0 {
        durations.clear();
    }
    // the orientation has been applied: width and height are the ones shown
    let (mut width, mut height) = (info.xsize as usize, info.ysize as usize);
    if info.orientation > 4 {
        std::mem::swap(&mut width, &mut height);
    }
    let animated = info.have_animation != 0;
    Ok(Decoded {
        width,
        height,
        channels: format.num_channels as usize,
        pixels: out,
        icc,
        exif,
        xmp,
        linear,
        hdr,
        original_bits: info.bits_per_sample,
        float,
        animated,
        frames: if animated { frames } else { 1 },
        durations,
        loops: animated.then_some(info.animation.num_loops),
        jpeg,
    })
}

/// The ICC profile libjxl gives for the original or the output colours.
fn profile(dec: *mut c_void, target: c_int) -> Option<Vec<u8>> {
    let mut n = 0usize;
    // SAFETY: the buffer is n bytes, as libjxl asked for
    unsafe {
        if ffi::JxlDecoderGetICCProfileSize(dec, target, &mut n) != ffi::DEC_SUCCESS || n == 0 || n > MAX_BOX {
            return None;
        }
        let mut v = vec![0u8; n];
        (ffi::JxlDecoderGetColorAsICCProfile(dec, target, v.as_mut_ptr(), n) == ffi::DEC_SUCCESS).then_some(v)
    }
}

/// Decode a JPEG XL file's first frame: (width, height, channels, bits, pixels, icc, exif, xmp,
/// info). bits is 8 (uint8 pixels), 16 (uint16) or 32 (float32); pixels is (height, width,
/// channels): 1 grey, 2 grey + alpha, 3 RGB, 4 RGBA. icc describes the pixels' colour space when
/// it isn't sRGB; exif is TIFF-structured; xmp is the XMP packet. info is a dict: frames,
/// durations (seconds, for animations), loops (0 = forever), linear (float32 linear light in sRGB
/// primaries), hdr ("pq", "hlg" or None), original_bits, float, jpeg (the original JPEG can be
/// rebuilt with jxl_to_jpeg). With `srgb`, other colour spaces are converted to sRGB.
#[pyfunction]
#[pyo3(signature = (data, srgb=false, max_pixels=None))]
#[allow(clippy::type_complexity)]
pub fn jxl_decode<'py>(
    py: Python<'py>,
    data: &[u8],
    srgb: bool,
    max_pixels: Option<u64>,
) -> PyResult<(usize, usize, usize, u32, Bound<'py, PyAny>, Option<Bound<'py, PyBytes>>, Option<Bound<'py, PyBytes>>,
    Option<Bound<'py, PyBytes>>, Bound<'py, PyDict>)> {
    let max = max_pixels.unwrap_or(MAX_PIXELS).max(1);
    let d = py.detach(|| decode(data, srgb, max)).map_err(|e| raise(py, e))?;
    let shape = [d.height, d.width, d.channels];
    let n = d.height * d.width * d.channels;
    let (bits, arr) = match d.pixels {
        Pixels::U8(mut v) => {
            v.truncate(n);
            (8, PyArray1::from_vec(py, v).reshape(shape)?.into_any())
        }
        Pixels::U16(mut v) => {
            v.truncate(n);
            (16, PyArray1::from_vec(py, v).reshape(shape)?.into_any())
        }
        Pixels::F32(mut v) => {
            v.truncate(n);
            (32, PyArray1::from_vec(py, v).reshape(shape)?.into_any())
        }
    };
    let info = PyDict::new(py);
    info.set_item("frames", d.frames)?;
    info.set_item("durations", d.durations)?;
    info.set_item("loops", d.loops)?;
    info.set_item("linear", d.linear)?;
    info.set_item("hdr", d.hdr)?;
    info.set_item("original_bits", d.original_bits)?;
    info.set_item("float", d.float)?;
    info.set_item("jpeg", d.jpeg)?;
    let b = |v: Option<Vec<u8>>| v.map(|v| PyBytes::new(py, &v));
    Ok((d.width, d.height, d.channels, bits, arr, b(d.icc), b(d.exif), b(d.xmp), info))
}

/// The JPEG a JPEG XL file was losslessly recompressed from, rebuilt byte for byte.
#[pyfunction]
#[pyo3(signature = (data, max_pixels=None))]
pub fn jxl_to_jpeg<'py>(py: Python<'py>, data: &[u8], max_pixels: Option<u64>) -> PyResult<Bound<'py, PyBytes>> {
    let max = max_pixels.unwrap_or(MAX_PIXELS).max(1);
    let out = py.detach(|| to_jpeg(data, max)).map_err(|e| raise(py, e))?;
    Ok(PyBytes::new(py, &out))
}

fn to_jpeg(data: &[u8], max_pixels: u64) -> Res<Vec<u8>> {
    use ffi::*;
    let s = Session::new(false, 256 << 20)?;
    let dec = s.handle;
    let mut out: Vec<u8> = Vec::new();
    let mut started = false;
    // SAFETY: the JPEG buffer stays alive and is set again whenever it moves
    unsafe {
        if JxlDecoderSubscribeEvents(dec, DEC_BASIC_INFO | DEC_JPEG_RECONSTRUCTION | DEC_FULL_IMAGE) != DEC_SUCCESS {
            return Err(bad("decoder setup"));
        }
        JxlDecoderSetInput(dec, data.as_ptr(), data.len());
        JxlDecoderCloseInput(dec);
        loop {
            match JxlDecoderProcessInput(dec) {
                DEC_BASIC_INFO => {
                    let mut info: BasicInfo = std::mem::zeroed();
                    if JxlDecoderGetBasicInfo(dec, &mut info) != DEC_SUCCESS {
                        return Err(bad("damaged header"));
                    }
                    check_size(info.xsize as u64, info.ysize as u64, max_pixels)?;
                    let planes = (info.num_color_channels.max(3) + info.num_extra_channels) as u64;
                    s.set_budget((info.xsize as u64 * info.ysize as u64 * planes * 4).saturating_mul(4).saturating_add(512 << 20));
                }
                DEC_JPEG_RECONSTRUCTION => {
                    started = true;
                    out = zeroed::<u8>(data.len().saturating_mul(2).clamp(1 << 16, 1 << 30))?;
                    JxlDecoderSetJPEGBuffer(dec, out.as_mut_ptr(), out.len());
                }
                DEC_JPEG_NEED_MORE_OUTPUT => {
                    let rest = JxlDecoderReleaseJPEGBuffer(dec);
                    let filled = out.len().saturating_sub(rest);
                    if out.len() >= (1 << 31) {
                        return Err(bad("the JPEG is too big"));
                    }
                    let grow = out.len();
                    out.try_reserve_exact(grow).map_err(|_| bad("not enough memory for the JPEG"))?;
                    out.resize(out.len() + grow, 0);
                    JxlDecoderSetJPEGBuffer(dec, out.as_mut_ptr().add(filled), out.len() - filled);
                }
                DEC_FULL_IMAGE | DEC_SUCCESS if started => {
                    let rest = JxlDecoderReleaseJPEGBuffer(dec);
                    out.truncate(out.len().saturating_sub(rest));
                    return Ok(out);
                }
                DEC_NEED_IMAGE_OUT_BUFFER | DEC_FULL_IMAGE | DEC_SUCCESS => return Err(bad("this file wasn't made from a JPEG")),
                DEC_NEED_MORE_INPUT => return Err(bad("the file is cut short")),
                _ => return Err(bad("the file is damaged")),
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// encoding

impl Session {
    fn check(&self, status: c_int, what: &str) -> Res<()> {
        if status == ffi::ENC_SUCCESS {
            return Ok(());
        }
        // SAFETY: the encoder is live
        let code = unsafe { ffi::JxlEncoderGetError(self.handle) };
        let why = match code {
            2 => "out of memory",
            3 => "the JPEG can't be recompressed losslessly",
            4 => "bad input",
            0x80 => "not supported",
            0x81 => "API misuse",
            _ => "encoder error",
        };
        Err(Damage::new(format!("JPEG XL: {what}: {why}")))
    }

    fn settings(&self, effort: i64) -> Res<*mut c_void> {
        // SAFETY: frame settings belong to the live encoder
        unsafe {
            let fs = ffi::JxlEncoderFrameSettingsCreate(self.handle, std::ptr::null());
            if fs.is_null() {
                return Err(Damage::new("JPEG XL: can't make frame settings"));
            }
            self.check(ffi::JxlEncoderFrameSettingsSetOption(fs, ffi::SETTING_EFFORT, effort.clamp(1, 10)), "effort")?;
            Ok(fs)
        }
    }

    fn boxes(&self, exif: Option<&[u8]>, xmp: Option<&[u8]>) -> Res<()> {
        if exif.is_none() && xmp.is_none() {
            return Ok(());
        }
        // SAFETY: the encoder is live; libjxl copies the contents
        unsafe {
            self.check(ffi::JxlEncoderUseBoxes(self.handle), "boxes")?;
            if let Some(tiff) = exif {
                // an Exif box starts with the offset of the TIFF header
                let mut b = Vec::with_capacity(tiff.len() + 4);
                b.extend_from_slice(&[0, 0, 0, 0]);
                b.extend_from_slice(tiff);
                self.check(ffi::JxlEncoderAddBox(self.handle, c"Exif".as_ptr(), b.as_ptr(), b.len(), 0), "Exif")?;
            }
            if let Some(x) = xmp {
                self.check(ffi::JxlEncoderAddBox(self.handle, c"xml ".as_ptr(), x.as_ptr(), x.len(), 0), "XMP")?;
            }
            ffi::JxlEncoderCloseBoxes(self.handle);
        }
        Ok(())
    }

    fn finish(&self) -> Res<Vec<u8>> {
        let mut out: Vec<u8> = vec![0; 1 << 16];
        let mut used = 0usize;
        // SAFETY: next/avail always describe the unused tail of `out`
        unsafe {
            ffi::JxlEncoderCloseInput(self.handle);
            loop {
                let mut next = out.as_mut_ptr().add(used);
                let mut avail = out.len() - used;
                let status = ffi::JxlEncoderProcessOutput(self.handle, &mut next, &mut avail);
                used = out.len() - avail;
                if status == ffi::ENC_NEED_MORE_OUTPUT {
                    let grow = out.len();
                    out.try_reserve_exact(grow).map_err(|_| bad("not enough memory for the file"))?;
                    out.resize(out.len() + grow, 0);
                    continue;
                }
                self.check(status, "encoding")?;
                break;
            }
        }
        out.truncate(used);
        Ok(out)
    }
}

struct Frame<'a> {
    width: usize,
    height: usize,
    channels: usize,
    data_type: c_int,
    bytes: &'a [u8],
}

struct Settings<'a> {
    lossless: bool,
    distance: f32,
    effort: i64,
    linear: bool,
    icc: Option<&'a [u8]>,
    exif: Option<&'a [u8]>,
    xmp: Option<&'a [u8]>,
}

fn encode(f: &Frame<'_>, o: &Settings<'_>) -> Res<Vec<u8>> {
    let (w, h, c) = (f.width as u64, f.height as u64, f.channels);
    check_size(w, h, MAX_SIDE * MAX_SIDE)?; // pixels already in memory: only the sides are capped
    if !(1..=4).contains(&c) {
        return Err(Damage::new(format!("JPEG XL: can't write pixels with {c} channels")));
    }
    let s = Session::new(true, isize::MAX)?;
    let alpha = c == 2 || c == 4;
    let (bits, exp_bits) = match f.data_type {
        ffi::TYPE_UINT8 => (8, 0),
        ffi::TYPE_UINT16 => (16, 0),
        _ => (32, 8),
    };
    // SAFETY: plain structs libjxl fills in; the encoder is live; libjxl copies the pixels
    unsafe {
        let mut info: ffi::BasicInfo = std::mem::zeroed();
        ffi::JxlEncoderInitBasicInfo(&mut info);
        info.xsize = w as u32;
        info.ysize = h as u32;
        info.bits_per_sample = bits;
        info.exponent_bits_per_sample = exp_bits;
        info.num_color_channels = if c <= 2 { 1 } else { 3 };
        info.num_extra_channels = alpha as u32;
        info.alpha_bits = if alpha { bits } else { 0 };
        info.alpha_exponent_bits = if alpha { exp_bits } else { 0 };
        // lossless keeps the samples as they are; lossy works in XYB
        info.uses_original_profile = o.lossless as ffi::JxlBool;
        s.check(ffi::JxlEncoderSetBasicInfo(s.handle, &info), "picture header")?;
        match o.icc {
            Some(p) if !o.linear => s.check(ffi::JxlEncoderSetICCProfile(s.handle, p.as_ptr(), p.len()), "ICC profile")?,
            _ => s.check(ffi::JxlEncoderSetColorEncoding(s.handle, &encoding(!o.linear, c <= 2)), "colour space")?,
        }
        s.boxes(o.exif, o.xmp)?;
        let fs = s.settings(o.effort)?;
        if o.lossless {
            s.check(ffi::JxlEncoderSetFrameLossless(fs, 1), "lossless")?;
        } else {
            s.check(ffi::JxlEncoderSetFrameDistance(fs, o.distance.clamp(0.0, 25.0)), "distance")?;
        }
        let format = ffi::PixelFormat { num_channels: c as u32, data_type: f.data_type, endianness: 0, align: 0 };
        s.check(ffi::JxlEncoderAddImageFrame(fs, &format, f.bytes.as_ptr().cast(), f.bytes.len()), "pixels")?;
    }
    s.finish()
}

fn as_bytes<'a, T: numpy::Element>(arr: &'a PyReadonlyArray3<'_, T>) -> PyResult<&'a [u8]> {
    let s = arr.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("pixels must be C-contiguous"))?;
    // SAFETY: plain numbers viewed as their bytes
    Ok(unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<u8>(), std::mem::size_of_val(s)) })
}

/// Encode (height, width, channels) uint8, uint16 or float32 pixels (1 grey, 2 grey + alpha,
/// 3 RGB, 4 RGBA) as JPEG XL: lossy at `distance` (Butteraugli units: 1.0 is visually lossless;
/// 0.0 to 25.0) or `lossless`, at `effort` 1 (fastest) to 10. The values are sRGB-coded, or in
/// the `icc` profile's space, or with `linear`, float32 linear light in sRGB primaries. `exif` is
/// TIFF-structured EXIF and `xmp` an XMP packet, both stored in the file.
#[pyfunction]
#[pyo3(signature = (pixels, lossless=false, distance=1.0, effort=7, icc=None, exif=None, xmp=None, linear=false))]
#[allow(clippy::too_many_arguments)]
pub fn jxl_encode<'py>(
    py: Python<'py>,
    pixels: &Bound<'py, PyAny>,
    lossless: bool,
    distance: f32,
    effort: i64,
    icc: Option<&[u8]>,
    exif: Option<&[u8]>,
    xmp: Option<&[u8]>,
    linear: bool,
) -> PyResult<Bound<'py, PyBytes>> {
    let o = Settings { lossless, distance, effort, linear, icc, exif, xmp };
    macro_rules! go {
        ($arr:expr, $ty:expr) => {{
            let arr = $arr;
            let shape = arr.shape().to_vec();
            let frame = Frame { width: shape[1], height: shape[0], channels: shape[2], data_type: $ty, bytes: as_bytes(&arr)? };
            py.detach(|| encode(&frame, &o))
        }};
    }
    let out = if let Ok(a) = pixels.extract::<PyReadonlyArray3<'py, u8>>() {
        go!(a, ffi::TYPE_UINT8)
    } else if let Ok(a) = pixels.extract::<PyReadonlyArray3<'py, u16>>() {
        go!(a, ffi::TYPE_UINT16)
    } else if let Ok(a) = pixels.extract::<PyReadonlyArray3<'py, f32>>() {
        go!(a, ffi::TYPE_FLOAT)
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err("pixels must be a (height, width, channels) uint8, uint16 or float32 array"));
    };
    Ok(PyBytes::new(py, &out.map_err(|e| raise(py, e))?))
}

/// A JPEG recompressed losslessly as JPEG XL: about a fifth smaller, and jxl_to_jpeg gives the
/// JPEG back byte for byte. `effort` 1 to 10.
#[pyfunction]
#[pyo3(signature = (data, effort=7))]
pub fn jxl_from_jpeg<'py>(py: Python<'py>, data: &[u8], effort: i64) -> PyResult<Bound<'py, PyBytes>> {
    let out = py
        .detach(|| -> Res<Vec<u8>> {
            let s = Session::new(true, isize::MAX)?;
            // SAFETY: the encoder is live; libjxl copies the JPEG
            unsafe {
                s.check(ffi::JxlEncoderStoreJPEGMetadata(s.handle, 1), "JPEG metadata")?;
                let fs = s.settings(effort)?;
                s.check(ffi::JxlEncoderAddJPEGFrame(fs, data.as_ptr(), data.len()), "JPEG")?;
            }
            s.finish()
        })
        .map_err(|e| raise(py, e))?;
    Ok(PyBytes::new(py, &out))
}

/// The libjxl version inside, as "major.minor.patch".
#[pyfunction]
pub fn jxl_version() -> String {
    // SAFETY: no arguments, no state
    let v = unsafe { ffi::JxlEncoderVersion() };
    format!("{}.{}.{}", v / 1_000_000, (v / 1000) % 1000, v % 1000)
}
