"""GIMP's XCF, written from GIMP's description of the format (devel-docs/xcf.txt, now on the
GIMP developer site), with layers composited the way GIMP shows them.

The file: "gimp xcf " and a version tag, the canvas size, the colour mode (RGB, grey, indexed),
the precision (XCF 4 and later: 8/16/32-bit integers, half, float or double, linear or
perceptual), properties (type, length, payload; ended by type 0), then pointers to the layers
(topmost first), channels and paths. Pointers are 32-bit up to XCF 10 and 64-bit from XCF 11.
A layer has a size, type, name, properties (offsets, opacity, visibility, mode, group membership,
blend and composite spaces) and pointers to its pixels and mask. Pixels sit in a hierarchy whose
first level holds 64 x 64 tiles, stored raw, run-length coded (each byte of the pixel as its own
stream) or zlib-compressed; samples are big-endian.

Compositing follows GIMP 2.10 and later: each layer mode has a blend function, a space it blends
in, a space it composites in and a composite mode (union, clip to backdrop, clip to layer,
intersection); the file records the last three for each layer. The legacy modes (before GIMP
2.10) blend and composite in perceptual sRGB and never raise the backdrop's alpha.
"""

from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import limits
from ._accel import native
from .errors import PixportError, UnsupportedError
from .image import Image, srgb_to_linear

MAX_SIDE = 50_000
TILE = 64

# precision codes (XCF 7 and later): (bytes per sample, kind, linear)
_PRECISION = {
    100: (1, "u", True), 150: (1, "u", False), 200: (2, "u", True), 250: (2, "u", False),
    300: (4, "u", True), 350: (4, "u", False), 500: (2, "f", True), 550: (2, "f", False),
    600: (4, "f", True), 650: (4, "f", False), 700: (8, "f", True), 750: (8, "f", False),
}
_PRECISION_V4 = {0: 150, 1: 250, 2: 300, 3: 500, 4: 600}
_PRECISION_V5 = {100: 100, 150: 150, 200: 200, 250: 250, 300: 300, 350: 350, 400: 500, 450: 550, 500: 600, 550: 650}

# composite spaces
LINEAR, PERCEPTUAL, LAB = 1, 2, 3
# composite modes
UNION, CLIP_BACKDROP, CLIP_LAYER = 1, 2, 3

# layer modes
(NORMAL_LEGACY, DISSOLVE, BEHIND_LEGACY, MULTIPLY_LEGACY, SCREEN_LEGACY, OVERLAY_LEGACY, DIFFERENCE_LEGACY,
 ADDITION_LEGACY, SUBTRACT_LEGACY, DARKEN_LEGACY, LIGHTEN_LEGACY, HUE_LEGACY, SATURATION_LEGACY, COLOR_LEGACY,
 VALUE_LEGACY, DIVIDE_LEGACY, DODGE_LEGACY, BURN_LEGACY, HARDLIGHT_LEGACY, SOFTLIGHT_LEGACY, GRAIN_EXTRACT_LEGACY,
 GRAIN_MERGE_LEGACY, COLOR_ERASE_LEGACY, OVERLAY, LCH_HUE, LCH_CHROMA, LCH_COLOR, LCH_LIGHTNESS, NORMAL, BEHIND,
 MULTIPLY, SCREEN, DIFFERENCE, ADDITION, SUBTRACT, DARKEN, LIGHTEN, HSV_HUE, HSV_SATURATION, HSL_COLOR, HSV_VALUE,
 DIVIDE, DODGE, BURN, HARDLIGHT, SOFTLIGHT, GRAIN_EXTRACT, GRAIN_MERGE, VIVID_LIGHT, PIN_LIGHT, LINEAR_LIGHT,
 HARD_MIX, EXCLUSION, LINEAR_BURN, LUMA_DARKEN, LUMA_LIGHTEN, LUMINANCE, COLOR_ERASE, ERASE, MERGE, SPLIT,
 PASS_THROUGH) = range(62)

# GIMP's defaults ("Auto") for files that don't record them: (blend space, composite space, composite mode)
_PERC = {OVERLAY, SCREEN, HSV_HUE, HSV_SATURATION, HSL_COLOR, HSV_VALUE, DODGE, BURN, HARDLIGHT, SOFTLIGHT,
         GRAIN_EXTRACT, GRAIN_MERGE, VIVID_LIGHT, PIN_LIGHT, LINEAR_LIGHT, HARD_MIX, EXCLUSION, LINEAR_BURN}
_UNION_MODES = {NORMAL, BEHIND, DISSOLVE, NORMAL_LEGACY, BEHIND_LEGACY, ERASE, MERGE, SPLIT, PASS_THROUGH,
                COLOR_ERASE, COLOR_ERASE_LEGACY}


def _default_spaces(mode: int) -> tuple[int, int, int]:
    if mode <= COLOR_ERASE_LEGACY:
        return PERCEPTUAL, PERCEPTUAL, UNION
    blend = LAB if LCH_HUE <= mode <= LCH_LIGHTNESS else (PERCEPTUAL if mode in _PERC else LINEAR)
    comp = UNION if mode in _UNION_MODES else CLIP_BACKDROP
    return blend, LINEAR, comp


@dataclass
class Layer:
    name: str
    width: int
    height: int
    type: int
    hptr: int
    mptr: int
    x: int = 0
    y: int = 0
    opacity: float = 1.0
    visible: bool = True
    mode: int = NORMAL_LEGACY
    blend_space: int = 0
    composite_space: int = 0
    composite_mode: int = 0
    group: bool = False
    path: list = field(default_factory=list)
    apply_mask: bool = True
    floating: bool = False
    children: list = field(default_factory=list)


class _Reader:
    def __init__(self, data: bytes):
        self.d = data
        self.n = len(data)

    def u32(self, pos: int) -> int:
        if pos + 4 > self.n:
            raise PixportError("The XCF file is cut short")
        return struct.unpack_from(">I", self.d, pos)[0]

    def string(self, pos: int) -> tuple[str, int]:
        k = self.u32(pos)
        pos += 4
        if k == 0:
            return "", pos
        if k > 1 << 20 or pos + k > self.n:
            raise PixportError("The XCF file is damaged (a string runs past the end)")
        return self.d[pos:pos + k - 1].decode("utf-8", "replace"), pos + k


def _parasites(r: _Reader, pos: int, end: int) -> dict[str, bytes]:
    out = {}
    while pos < end:
        name, pos = r.string(pos)
        if pos + 8 > end:
            break
        _flags, size = struct.unpack_from(">II", r.d, pos)
        pos += 8
        if pos + size > end:
            break
        out[name] = r.d[pos:pos + size]
        pos += size
    return out


def _props(r: _Reader, pos: int, version: int, image: bool = False) -> tuple[dict, int]:
    """A property list: {type: payload bytes} (parasites merged under 21), and where it ends."""
    props: dict = {}
    for _ in range(100000):
        typ = r.u32(pos)
        size = r.u32(pos + 4)
        pos += 8
        if typ == 0:
            return props, pos
        if image and typ == 1:  # PROP_COLORMAP: the length word can't be trusted
            ncol = r.u32(pos)
            if ncol > 256:
                raise PixportError("The XCF colour map is damaged")
            if pos + 4 + 3 * ncol > r.n:
                raise PixportError("The XCF file is cut short")
            props[1] = r.d[pos:pos + 4 + 3 * ncol]
            pos += 4 + 3 * ncol
            continue
        if image and typ == 24:  # PROP_USER_UNIT: nor this one's
            p = pos + 8
            for _k in range(3 if version >= 21 else 5):
                _s, p = r.string(p)
            pos = p
            continue
        if pos + size > r.n:
            raise PixportError("The XCF file is damaged (a property runs past the end)")
        payload = r.d[pos:pos + size]
        if typ == 21:
            props.setdefault(21, {}).update(_parasites(r, pos, pos + size))
        else:
            props[typ] = payload
        pos += size
    raise PixportError("The XCF file is damaged (a property list with no end)")


def _u32(b: bytes, default: int = 0) -> int:
    return struct.unpack_from(">I", b)[0] if len(b) >= 4 else default


def _i32(b: bytes, default: int = 0) -> int:
    return struct.unpack_from(">i", b)[0] if len(b) >= 4 else default


@dataclass
class _Image:
    version: int
    width: int
    height: int
    base: int
    precision: int
    compression: int
    colormap: np.ndarray | None
    props: dict
    layers: list
    ptr: int  # pointer size


def _header(data: bytes) -> tuple[_Image, int]:
    r = _Reader(data)
    if not data.startswith(b"gimp xcf "):
        raise PixportError("Not a GIMP XCF file")
    tag = data[9:13]
    if tag == b"file":
        version = 0
    elif tag[:1] == b"v" and tag[1:].isdigit():
        version = int(tag[1:])
    else:
        raise PixportError("The XCF version tag is damaged")
    if version >= 100:
        raise UnsupportedError("This is a CinePaint XCF file, which pixport doesn't read")
    pos = 14
    width, height, base = r.u32(pos), r.u32(pos + 4), r.u32(pos + 8)
    pos += 12
    if width < 1 or height < 1 or width > MAX_SIDE or height > MAX_SIDE:
        raise PixportError(f"The XCF canvas ({width} x {height}) is out of range")
    limits.check(width, height, "XCF canvas")
    if base > 2:
        raise PixportError("The XCF colour mode is damaged")
    precision = 150
    if version >= 4:
        p = r.u32(pos)
        pos += 4
        if version == 4:
            precision = _PRECISION_V4.get(p, -1)
        elif version in (5, 6):
            precision = _PRECISION_V5.get(p, -1)
        else:
            precision = p
        if precision not in _PRECISION:
            raise UnsupportedError(f"pixport doesn't read XCF precision {p}")
    props, pos = _props(r, pos, version, image=True)
    comp = props.get(17, b"\0")[0] if props.get(17) else 0
    cmap = None
    if 1 in props:
        b = props[1]
        n = _u32(b)
        cmap = np.frombuffer(b, np.uint8, 3 * n, 4).reshape(n, 3) if n else np.zeros((0, 3), np.uint8)
    ptr = 8 if version >= 11 else 4
    img = _Image(version, width, height, base, precision, comp, cmap, props, [], ptr)
    return img, pos


def _pointer(r: _Reader, pos: int, size: int) -> int:
    if pos + size > r.n:
        raise PixportError("The XCF file is cut short")
    return struct.unpack_from(">Q" if size == 8 else ">I", r.d, pos)[0]


def _layer(r: _Reader, img: _Image, pos: int) -> Layer:
    if not 0 < pos < r.n:
        raise PixportError("The XCF file is damaged (a layer pointer outside the file)")
    w, h, typ = r.u32(pos), r.u32(pos + 4), r.u32(pos + 8)
    name, pos = r.string(pos + 12)
    if w > MAX_SIDE or h > MAX_SIDE or typ > 5:
        raise PixportError(f"The XCF layer {name!r} is damaged (size or type)")
    limits.check(w, h, f"XCF layer {name!r}")
    props, pos = _props(r, pos, img.version)
    hptr = _pointer(r, pos, img.ptr)
    mptr = _pointer(r, pos + img.ptr, img.ptr)
    lay = Layer(name, w, h, typ, hptr, mptr)
    if 15 in props and len(props[15]) >= 8:
        lay.x, lay.y = struct.unpack_from(">ii", props[15])
    if 6 in props:
        lay.opacity = min(_u32(props[6], 255), 255) / 255.0
    if 33 in props and len(props[33]) >= 4:
        v = struct.unpack_from(">f", props[33])[0]
        if not math.isnan(v):
            lay.opacity = min(max(v, 0.0), 1.0)
    if 8 in props:
        lay.visible = bool(_u32(props[8], 1))
    if 7 in props:
        lay.mode = _u32(props[7])
    if 35 in props:
        lay.composite_mode = abs(_i32(props[35]))
    if 36 in props:
        lay.composite_space = abs(_i32(props[36]))
    if 37 in props:
        lay.blend_space = abs(_i32(props[37]))
    lay.group = 29 in props
    if 30 in props:
        b = props[30]
        lay.path = list(struct.unpack_from(f">{len(b) // 4}I", b))
    if 11 in props:
        lay.apply_mask = bool(_u32(props[11], 1))
    lay.floating = 5 in props
    return lay


# ---------------------------------------------------------------- pixels

def rle_tile(data: bytes, pos: int, end: int, bpp: int, npix: int) -> tuple[bytes, int]:
    """One RLE tile: bpp streams of npix bytes each (the first byte of every pixel, then the
    second, ...), returned pixel-interleaved, and where the tile's data ended. (With pixport-native
    a whole level is decoded at once instead, by xcf_rle_level.)"""
    starts, lengths, steps = [], [], []
    for _ in range(bpp):
        left = npix
        while left > 0:
            if pos >= end:
                raise PixportError("The XCF tile data is cut short")
            op = data[pos]
            if op < 127:
                k = op + 1
                if pos + 2 > end or k > left:
                    raise PixportError("The XCF tile data is damaged (a run overflows its tile)")
                starts.append(pos + 1)
                steps.append(0)
                pos += 2
            elif op == 127:
                if pos + 4 > end:
                    raise PixportError("The XCF tile data is cut short")
                k = data[pos + 1] << 8 | data[pos + 2]
                if k > left:
                    raise PixportError("The XCF tile data is damaged (a run overflows its tile)")
                starts.append(pos + 3)
                steps.append(0)
                pos += 4
            elif op == 128:
                if pos + 3 > end:
                    raise PixportError("The XCF tile data is cut short")
                k = data[pos + 1] << 8 | data[pos + 2]
                if k > left or pos + 3 + k > end:
                    raise PixportError("The XCF tile data is damaged (a literal overflows its tile)")
                starts.append(pos + 3)
                steps.append(1)
                pos += 3 + k
            else:
                k = 256 - op
                if k > left or pos + 1 + k > end:
                    raise PixportError("The XCF tile data is damaged (a literal overflows its tile)")
                starts.append(pos + 1)
                steps.append(1)
                pos += 1 + k
            if k:
                lengths.append(k)
            else:
                starts.pop()
                steps.pop()
            left -= k
    src = np.frombuffer(data, np.uint8)
    st = np.asarray(starts, np.int64)
    ln = np.asarray(lengths, np.int64)
    sp = np.asarray(steps, np.int64)
    total = bpp * npix
    first = np.repeat(np.cumsum(ln) - ln, ln)
    ramp = np.arange(total, dtype=np.int64) - first
    planes = src[np.repeat(st, ln) + ramp * np.repeat(sp, ln)].reshape(bpp, npix)
    return planes.T.tobytes(), pos


def _level_tiles(r: _Reader, img: _Image, hptr: int, bpp_want: int, w: int, h: int) -> np.ndarray:
    """A drawable's pixel bytes as (h, w, bpp) uint8, from its hierarchy's first level."""
    if not 0 < hptr < r.n:
        raise PixportError("The XCF file is damaged (a hierarchy pointer outside the file)")
    hw, hh, bpp = r.u32(hptr), r.u32(hptr + 4), r.u32(hptr + 8)
    if (hw, hh) != (w, h) or bpp != bpp_want:
        raise PixportError("The XCF file is damaged (a hierarchy of the wrong size)")
    lptr = _pointer(r, hptr + 12, img.ptr)
    if not 0 < lptr < r.n:
        raise PixportError("The XCF file is damaged (a level pointer outside the file)")
    lw, lh = r.u32(lptr), r.u32(lptr + 4)
    if (lw, lh) != (w, h):
        raise PixportError("The XCF file is damaged (a level of the wrong size)")
    ntx, nty = -(-w // TILE), -(-h // TILE)
    nt = ntx * nty
    ptrs = []
    pos = lptr + 8
    for _ in range(nt + 1):
        ptrs.append(_pointer(r, pos, img.ptr))
        pos += img.ptr
    if any(not 0 < p < r.n for p in ptrs[:nt]):
        raise PixportError("The XCF file is damaged (a tile pointer outside the file)")
    if img.compression == 1 and native is not None and hasattr(native, "xcf_rle_level"):
        return np.frombuffer(native.xcf_rle_level(r.d, ptrs, w, h, bpp), np.uint8).reshape(h, w, bpp)
    out = np.zeros((h, w, bpp), np.uint8)
    d = r.d
    for t in range(nt):
        ty, tx = divmod(t, ntx)
        x0, y0 = tx * TILE, ty * TILE
        tw, th = min(TILE, w - x0), min(TILE, h - y0)
        npix = tw * th
        start = ptrs[t]
        nxt = ptrs[t + 1]
        end = nxt if start < nxt <= r.n else r.n
        if img.compression == 0:
            if start + npix * bpp > r.n:
                raise PixportError("The XCF tile data is cut short")
            raw = d[start:start + npix * bpp]
        elif img.compression == 1:
            raw, _ = rle_tile(d, start, end, bpp, npix)
        elif img.compression == 2:
            z = zlib.decompressobj()
            raw = z.decompress(d[start:end], npix * bpp)
            if len(raw) != npix * bpp:
                raise PixportError("The XCF tile data is damaged (zlib)")
        else:
            raise UnsupportedError(f"pixport doesn't read XCF compression {img.compression}")
        out[y0:y0 + th, x0:x0 + tw] = np.frombuffer(raw, np.uint8).reshape(th, tw, bpp)
    return out


def _samples(raw: np.ndarray, size: int, kind: str) -> np.ndarray:
    """Big-endian samples to float32 (integers scaled to 0..1)."""
    h, w, nb = raw.shape
    flat = np.ascontiguousarray(raw).reshape(h, w, nb // size, size)
    if kind == "u":
        dt = {1: np.uint8, 2: ">u2", 4: ">u4"}[size]
        v = flat.reshape(h, w, -1).view(dt).astype(np.float64 if size == 4 else np.float32)
        return (v / float((1 << (8 * size)) - 1)).astype(np.float32)
    dt = {2: ">f2", 4: ">f4", 8: ">f8"}[size]
    return flat.reshape(h, w, -1).view(dt).astype(np.float32)


_LUTS: dict = {}


def _srgb_lut(size: int) -> np.ndarray:
    """Perceptual integer samples to linear light, for every code."""
    if size not in _LUTS:
        top = (1 << (8 * size)) - 1
        _LUTS[size] = srgb_to_linear(np.arange(top + 1, dtype=np.float64) / top).astype(np.float32)
    return _LUTS[size]


def _layer_pixels(r: _Reader, img: _Image, lay: Layer, linear: bool = False) -> np.ndarray:
    """(h, w, 4) float32 RGBA, straight alpha: in the image's own encoding, or with linear=True
    in linear light whatever the image's precision."""
    size, kind, lin = _PRECISION[img.precision]
    to_lin = linear and not lin
    has_alpha = lay.type in (1, 3, 5)
    if lay.type in (4, 5):
        ncomp, size, kind = 1, 1, "u"
    else:
        ncomp = 3 if lay.type in (0, 1) else 1
    nch = ncomp + has_alpha
    raw = _level_tiles(r, img, lay.hptr, nch * size, lay.width, lay.height)
    if lay.type in (4, 5):
        cmap = img.colormap if img.colormap is not None and len(img.colormap) else np.zeros((1, 3), np.uint8)
        idx = np.minimum(raw[:, :, 0], len(cmap) - 1)
        rgb = _srgb_lut(1)[cmap[idx]] if to_lin else cmap[idx].astype(np.float32) / 255.0
        a = (raw[:, :, 1] >= 128).astype(np.float32) if has_alpha else np.ones(raw.shape[:2], np.float32)
        return np.concatenate([rgb, a[:, :, None]], axis=2)
    if to_lin and kind == "u" and size <= 2:
        h, w = raw.shape[:2]
        ints = np.ascontiguousarray(raw).view(np.uint8 if size == 1 else ">u2").reshape(h, w, nch)
        v = np.empty((h, w, nch), np.float32)
        v[..., :ncomp] = _srgb_lut(size)[ints[..., :ncomp]]
        if has_alpha:
            v[..., ncomp] = ints[..., ncomp] * np.float32(1 / ((1 << (8 * size)) - 1))
    else:
        v = _samples(raw, size, kind)
        if to_lin:
            v[..., :ncomp] = _srgb_to_linear_ext(v[..., :ncomp])
    if ncomp == 1:
        v = np.concatenate([np.repeat(v[:, :, :1], 3, axis=2), v[:, :, 1:]], axis=2)
    if not has_alpha:
        v = np.concatenate([v, np.ones(v.shape[:2] + (1,), np.float32)], axis=2)
    return v


def _mask_pixels(r: _Reader, img: _Image, lay: Layer) -> np.ndarray | None:
    if not lay.mptr or not lay.apply_mask:
        return None
    p = lay.mptr
    if not 0 < p < r.n:
        raise PixportError("The XCF file is damaged (a mask pointer outside the file)")
    w, h = r.u32(p), r.u32(p + 4)
    _name, pos = r.string(p + 8)
    pos = _props(r, pos, img.version)[1]
    hptr = _pointer(r, pos, img.ptr)
    if (w, h) != (lay.width, lay.height):
        raise PixportError("The XCF layer mask doesn't match its layer")
    size, kind, _lin = _PRECISION[img.precision]
    return _samples(_level_tiles(r, img, hptr, size, w, h), size, kind)[:, :, 0]


# ---------------------------------------------------------------- colour spaces

def _to_space(rgb: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Converts colour between linear and perceptual (sRGB-curved) encodings."""
    if src == dst or dst == LAB or src == LAB:
        return rgb
    if dst == LINEAR:
        return _srgb_to_linear_ext(rgb)
    return _linear_to_srgb_ext(rgb)


def _linear_to_srgb_ext(x: np.ndarray) -> np.ndarray:
    """The sRGB curve extended to negative values (mirrored), in one float32 array: the power
    is only taken where it's used."""
    a = np.abs(x, dtype=np.float32)
    hi = a > np.float32(0.0031308)
    np.power(a, np.float32(1 / 2.4), out=a, where=hi)
    np.multiply(a, np.float32(1.055), out=a, where=hi)
    np.subtract(a, np.float32(0.055), out=a, where=hi)
    np.multiply(a, np.float32(12.92), out=a, where=np.logical_not(hi, out=hi))
    return np.copysign(a, x, out=a)


def _srgb_to_linear_ext(x: np.ndarray) -> np.ndarray:
    a = np.abs(x, dtype=np.float32)
    hi = a > np.float32(0.04045)
    np.add(a, np.float32(0.055), out=a, where=hi)
    np.divide(a, np.float32(1.055), out=a, where=hi)
    np.power(a, np.float32(2.4), out=a, where=hi)
    np.divide(a, np.float32(12.92), out=a, where=np.logical_not(hi, out=hi))
    return np.copysign(a, x, out=a)


_D65 = np.array([0.95047, 1.0, 1.08883], np.float32)
_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                     [0.0193339, 0.1191920, 0.9503041]], np.float32)
_XYZ2RGB = np.linalg.inv(_RGB2XYZ.astype(np.float64)).astype(np.float32)


def _lab(lin: np.ndarray) -> np.ndarray:
    """Linear sRGB to CIE L*a*b* (D65)."""
    xyz = lin @ _RGB2XYZ.T / _D65
    e = 216 / 24389
    k = 24389 / 27
    f = np.where(xyz > e, np.cbrt(np.maximum(xyz, 0)), (k * xyz + 16) / 116).astype(np.float32)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def _unlab(lab: np.ndarray) -> np.ndarray:
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    e = 216 / 24389
    k = 24389 / 27

    def inv(f):
        return np.where(f ** 3 > e, f ** 3, (116 * f - 16) / k)
    xyz = np.stack([inv(fx), np.where(lab[..., 0] > k * e, fy ** 3, lab[..., 0] / k), inv(fz)], axis=-1) * _D65
    return (xyz @ _XYZ2RGB.T).astype(np.float32)


# ---------------------------------------------------------------- blend functions (on colour only)

def _lum(c: np.ndarray) -> np.ndarray:
    return c[..., 0] * 0.2126 + c[..., 1] * 0.7152 + c[..., 2] * 0.0722


def _safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(b) > 1e-7, a / np.where(np.abs(b) > 1e-7, b, 1), np.where(a > 0, 1e7, 0) * 1.0)


def _hsv(c):
    mx = c.max(axis=-1)
    mn = c.min(axis=-1)
    d = mx - mn
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(mx > 0, d / np.where(mx > 0, mx, 1), 0)
        r, g, b = c[..., 0], c[..., 1], c[..., 2]
        dd = np.where(d > 0, d, 1)
        h = np.where(mx == r, (g - b) / dd, np.where(mx == g, 2 + (b - r) / dd, 4 + (r - g) / dd))
    h = np.where(d > 0, (h / 6) % 1.0, 0)
    return h, s, mx


def _from_hsv(h, s, v):
    h6 = (h % 1.0) * 6
    i = np.floor(h6).astype(np.int32) % 6
    f = h6 - np.floor(h6)
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    choices = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]
    out = np.zeros(h.shape + (3,), np.float32)
    for k, (a, b, c) in enumerate(choices):
        m = i == k
        out[m, 0], out[m, 1], out[m, 2] = a[m], b[m], c[m]
    return out


def _hsl(c):
    mx = c.max(axis=-1)
    mn = c.min(axis=-1)
    l_ = (mx + mn) / 2
    d = mx - mn
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(d > 0, d / np.where(l_ <= 0.5, np.maximum(mx + mn, 1e-12), np.maximum(2 - mx - mn, 1e-12)), 0)
    h, _s, _v = _hsv(c)
    return h, s, l_


def _from_hsl(h, s, l_):
    q = np.where(l_ <= 0.5, l_ * (1 + s), l_ + s - l_ * s)
    p = 2 * l_ - q

    def hue(t):
        t = t % 1.0
        return np.where(t < 1 / 6, p + (q - p) * 6 * t,
                        np.where(t < 0.5, q, np.where(t < 2 / 3, p + (q - p) * (2 / 3 - t) * 6, p)))
    out = np.stack([hue(h + 1 / 3), hue(h), hue(h - 1 / 3)], axis=-1)
    return np.where((s == 0)[..., None], l_[..., None], out).astype(np.float32)


def _blend(mode: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The blend function of a mode: backdrop colour a, layer colour b, both in the blend space."""
    legacy = mode <= COLOR_ERASE_LEGACY
    if mode in (MULTIPLY, MULTIPLY_LEGACY):
        return a * b
    if mode in (SCREEN, SCREEN_LEGACY):
        return 1 - (1 - a) * (1 - b)
    if mode in (DIFFERENCE, DIFFERENCE_LEGACY):
        return np.abs(a - b)
    if mode == ADDITION:
        return a + b
    if mode == ADDITION_LEGACY:
        return np.clip(a + b, 0, 1)
    if mode == SUBTRACT:
        return a - b
    if mode == SUBTRACT_LEGACY:
        return np.clip(a - b, 0, 1)
    if mode in (DARKEN, DARKEN_LEGACY):
        return np.minimum(a, b)
    if mode in (LIGHTEN, LIGHTEN_LEGACY):
        return np.maximum(a, b)
    if mode == DIVIDE:
        return _safe_div(a, b)
    if mode == DIVIDE_LEGACY:
        return np.clip((256 / 255) * a / (1 / 255 + b), 0, 1)
    if mode == DODGE:
        return _safe_div(a, 1 - b)
    if mode == DODGE_LEGACY:
        return np.clip((256 / 255) * a / (256 / 255 - b), 0, 1)
    if mode == BURN:
        return 1 - _safe_div(1 - a, b)
    if mode == BURN_LEGACY:
        return np.clip(1 - (256 / 255) * (1 - a) / (1 / 255 + b), 0, 1)
    if mode in (HARDLIGHT, HARDLIGHT_LEGACY):
        out = np.where(b > 0.5, 1 - (1 - a) * (1 - (b - 0.5) * 2), a * b * 2)
        return np.clip(out, 0, 1) if legacy else out
    if mode in (SOFTLIGHT_LEGACY, OVERLAY_LEGACY, SOFTLIGHT):
        m = a * b
        s = 1 - (1 - a) * (1 - b)
        return (1 - a) * m + a * s
    if mode == OVERLAY:
        return np.where(a < 0.5, 2 * a * b, 1 - 2 * (1 - a) * (1 - b))
    if mode == GRAIN_EXTRACT:
        return a - b + 0.5
    if mode == GRAIN_EXTRACT_LEGACY:
        return np.clip(a - b + 128 / 255, 0, 1)
    if mode == GRAIN_MERGE:
        return a + b - 0.5
    if mode == GRAIN_MERGE_LEGACY:
        return np.clip(a + b - 128 / 255, 0, 1)
    if mode == VIVID_LIGHT:
        return np.where(b <= 0.5, 1 - _safe_div(1 - a, 2 * b), _safe_div(a, 2 * (1 - b)))
    if mode == PIN_LIGHT:
        return np.where(b > 0.5, np.maximum(a, 2 * b - 1), np.minimum(a, 2 * b))
    if mode == LINEAR_LIGHT:
        return a + 2 * b - 1
    if mode == HARD_MIX:
        return np.where(a + b < 1, 0.0, 1.0).astype(np.float32)
    if mode == EXCLUSION:
        return 0.5 - 2 * (a - 0.5) * (b - 0.5)
    if mode == LINEAR_BURN:
        return a + b - 1
    if mode in (LUMA_DARKEN, LUMA_LIGHTEN):
        pick_a = _lum(a) <= _lum(b) if mode == LUMA_DARKEN else _lum(a) >= _lum(b)
        return np.where(pick_a[..., None], a, b)
    if mode == LUMINANCE:
        la, lb = _lum(a), _lum(b)
        return a * _safe_div(lb, la)[..., None]
    if mode in (HSV_HUE, HUE_LEGACY, HSV_SATURATION, SATURATION_LEGACY, HSV_VALUE, VALUE_LEGACY):
        ha, sa, va = _hsv(a)
        hb, sb, vb = _hsv(b)
        if mode in (HSV_HUE, HUE_LEGACY):
            out = _from_hsv(np.where(sb > 0, hb, ha), sa, va)
            return np.where((sb > 0)[..., None], out, a)
        if mode in (HSV_SATURATION, SATURATION_LEGACY):
            return _from_hsv(ha, sb, va)
        return _from_hsv(ha, sa, vb)
    if mode in (HSL_COLOR, COLOR_LEGACY):
        _ha, _sa, la = _hsl(a)
        hb, sb, _lb = _hsl(b)
        return _from_hsl(hb, sb, la)
    if LCH_HUE <= mode <= LCH_LIGHTNESS:
        la, lb = a, b  # already L*a*b*
        ca = np.hypot(la[..., 1], la[..., 2])
        cb = np.hypot(lb[..., 1], lb[..., 2])
        out = la.copy()
        if mode == LCH_HUE:
            with np.errstate(divide="ignore", invalid="ignore"):
                k = np.where(cb > 0, ca / np.where(cb > 0, cb, 1), 0)
            hue = np.where((cb > 0)[..., None], lb[..., 1:] * k[..., None], la[..., 1:])
            out[..., 1:] = hue
        elif mode == LCH_CHROMA:
            with np.errstate(divide="ignore", invalid="ignore"):
                k = np.where(ca > 0, cb / np.where(ca > 0, ca, 1), 0)
            out[..., 1:] = la[..., 1:] * k[..., None]
        elif mode == LCH_COLOR:
            out[..., 1:] = lb[..., 1:]
        else:
            out[..., 0] = lb[..., 0]
        return out
    return b


# ---------------------------------------------------------------- compositing

def _hash_noise(h: int, w: int, y0: int, x0: int) -> np.ndarray:
    """A fixed per-pixel pseudo-random field in [0, 1) for Dissolve."""
    yy, xx = np.mgrid[y0:y0 + h, x0:x0 + w]
    v = (xx.astype(np.uint64) * np.uint64(0x9E3779B1) ^ yy.astype(np.uint64) * np.uint64(0x85EBCA77)) & np.uint64(0xFFFFFFFF)
    v ^= v >> np.uint64(15)
    v = (v * np.uint64(0x2C1B3C6D)) & np.uint64(0xFFFFFFFF)
    v ^= v >> np.uint64(12)
    return (v & np.uint64(0xFFFFFF)).astype(np.float32) / np.float32(1 << 24)


def _composite(back: np.ndarray, layer: np.ndarray, opacity: np.ndarray, mode: int, blend_space: int,
               comp_space: int, comp_mode: int, work: int, y0: int = 0, x0: int = 0) -> np.ndarray:
    """One layer over a backdrop. back and layer: (h, w, 4) RGBA in the working encoding (work:
    LINEAR or PERCEPTUAL), straight alpha; opacity: (h, w) layer opacity times its mask.
    Returns the new backdrop."""
    ia = back[..., 3]
    la = layer[..., 3] * opacity
    legacy = mode <= COLOR_ERASE_LEGACY and mode not in (NORMAL_LEGACY, DISSOLVE, BEHIND_LEGACY)
    if mode == DISSOLVE:
        keep = _hash_noise(la.shape[0], la.shape[1], y0, x0) < la
        la = keep.astype(np.float32)
        mode = NORMAL
    # Normal with a composite mode other than union (clip to backdrop or layer, intersection) is
    # composited as the other modes are, its blend being the layer itself
    normal_clip = mode == NORMAL and comp_mode not in (0, UNION)
    if not normal_clip and mode in (NORMAL_LEGACY, NORMAL, DISSOLVE, BEHIND, BEHIND_LEGACY, ERASE, MERGE, SPLIT,
                                    COLOR_ERASE, COLOR_ERASE_LEGACY):
        cspace = comp_space or (PERCEPTUAL if mode <= COLOR_ERASE_LEGACY else LINEAR)
        if mode <= COLOR_ERASE_LEGACY:
            cspace = PERCEPTUAL
        bc = _to_space(back[..., :3], work, cspace)
        lc = _to_space(layer[..., :3], work, cspace)
        out = np.empty_like(back)
        if mode in (BEHIND, BEHIND_LEGACY):
            na = ia + la * (1 - ia)
            with np.errstate(divide="ignore", invalid="ignore"):
                c = np.where(na[..., None] > 0, (bc * ia[..., None] + lc * (la * (1 - ia))[..., None]) / na[..., None], bc)
            out[..., :3], out[..., 3] = c, na
        elif mode == ERASE:
            out[..., :3], out[..., 3] = bc, ia * (1 - la)
        elif mode == MERGE:
            na = np.minimum(ia + la, 1.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                c = np.where(na[..., None] > 0, (bc * (ia * (1 - la) / np.maximum(na, 1e-12))[..., None]
                                                 + lc * (la / np.maximum(na, 1e-12))[..., None]), bc)
            out[..., :3], out[..., 3] = c, na
        elif mode == SPLIT:
            out[..., :3], out[..., 3] = bc, np.maximum(ia - la, 0)
        elif mode in (COLOR_ERASE, COLOR_ERASE_LEGACY):
            out[..., :3], out[..., 3] = _color_erase(bc, ia, lc, la)
        elif la.min() >= 1:  # an opaque layer covers everything below
            out[..., :3], out[..., 3] = lc, 1
        elif ia.max() <= 0:  # nothing below
            out[..., :3], out[..., 3] = lc, la
        elif ia.min() >= 1:  # an opaque backdrop
            out[..., :3] = bc + (lc - bc) * la[..., None]
            out[..., 3] = 1
        else:
            na = la + ia * (1 - la)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(na > 0, la / np.where(na > 0, na, 1), 0)[..., None]
            out[..., :3] = bc + ratio * (lc - bc)
            out[..., 3] = na
        out[..., :3] = _to_space(out[..., :3], cspace, work)
        return out
    if legacy:
        bc = _to_space(back[..., :3], work, PERCEPTUAL)
        lc = _to_space(layer[..., :3], work, PERCEPTUAL)
        comp = _blend(mode, bc, lc)
        ca = np.minimum(ia, layer[..., 3]) * opacity
        na = ia + (1 - ia) * ca
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where((ca > 0) & (na > 0), ca / np.where(na > 0, na, 1), 0)[..., None]
        out = np.empty_like(back)
        out[..., :3] = _to_space(comp * ratio + bc * (1 - ratio), PERCEPTUAL, work)
        out[..., 3] = ia
        return out
    dflt = _default_spaces(mode)
    bspace = blend_space or dflt[0]
    cspace = comp_space or dflt[1]
    cmode = comp_mode or dflt[2]
    if normal_clip:
        comp = _to_space(layer[..., :3], work, cspace)
    elif bspace == LAB:
        ba = _lab(_to_space(back[..., :3], work, LINEAR))
        lb = _lab(_to_space(layer[..., :3], work, LINEAR))
        comp = _to_space(_unlab(_blend(mode, ba, lb)), LINEAR, cspace)
    else:
        comp = _to_space(_blend(mode, _to_space(back[..., :3], work, bspace), _to_space(layer[..., :3], work, bspace)),
                         bspace, cspace)
    bc = _to_space(back[..., :3], work, cspace)
    lc = _to_space(layer[..., :3], work, cspace)
    out = np.empty_like(back)
    if cmode == UNION:
        na = la + (1 - la) * ia
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(na > 0, la / np.where(na > 0, na, 1), 0)[..., None]
        c = ratio * (ia[..., None] * (comp - lc) + lc - bc) + bc
        out[..., 3] = na
    elif cmode == CLIP_BACKDROP:
        c = bc + (comp - bc) * la[..., None]
        out[..., 3] = ia
    elif cmode == CLIP_LAYER:
        c = lc + (comp - lc) * ia[..., None]
        out[..., 3] = la
    else:
        c = comp
        out[..., 3] = ia * la
    out[..., :3] = _to_space(c, cspace, work)
    return out


def _color_erase(bc, ia, lc, la):
    """GIMP's colour erase: removes as much of the layer's colour from the backdrop as can be
    explained as that colour over something more transparent."""
    eps = 1e-6
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.zeros(bc.shape, np.float32)
        for k in range(3):
            c, l_ = bc[..., k], lc[..., k]
            alpha[..., k] = np.where(c > l_ + eps, (c - l_) / np.maximum(1 - l_, eps),
                                     np.where(c < l_ - eps, (l_ - c) / np.maximum(l_, eps), 0))
        a = alpha.max(axis=-1)
        new_a = ia * (1 - la + la * a)
        col = np.where(a[..., None] > eps, (bc - lc) / np.maximum(a, eps)[..., None] + lc, bc)
        col = bc + (col - bc) * la[..., None]
    return col, new_a


def _rect(lay: Layer, w: int, h: int):
    x0, y0 = max(lay.x, 0), max(lay.y, 0)
    x1, y1 = min(lay.x + lay.width, w), min(lay.y + lay.height, h)
    return x0, y0, x1, y1


def _identity_outside(mode: int, comp_mode: int) -> bool:
    """Whether a layer leaves the backdrop alone where it has no pixels."""
    if mode in (PASS_THROUGH,):
        return False
    cm = comp_mode or _default_spaces(mode)[2]
    if mode == NORMAL and cm not in (UNION, CLIP_BACKDROP):  # clipped to the layer: nothing is left outside it
        return False
    if mode <= COLOR_ERASE_LEGACY or mode in (NORMAL, BEHIND, ERASE, MERGE, SPLIT, COLOR_ERASE):
        return True
    return cm in (UNION, CLIP_BACKDROP)


class _Compositor:
    def __init__(self, r: _Reader, img: _Image, work: int):
        self.r = r
        self.img = img
        self.work = work
        self.w = img.width
        self.h = img.height

    def pixels(self, lay: Layer) -> np.ndarray:
        return _layer_pixels(self.r, self.img, lay, linear=self.work == LINEAR)

    def stack(self, layers: list, back: np.ndarray | None = None) -> np.ndarray:
        """Composites a stack (topmost first) onto back (or onto transparency)."""
        canvas = np.zeros((self.h, self.w, 4), np.float32) if back is None else back
        visible = [lay for lay in layers if lay.visible and not lay.floating]
        for i, lay in enumerate(reversed(visible)):
            bottom = back is None and i == 0
            canvas = self.layer(lay, canvas, bottom)
        return canvas

    def layer(self, lay: Layer, canvas: np.ndarray, bottom: bool) -> np.ndarray:
        w, h = self.w, self.h
        mode = lay.mode
        if mode == PASS_THROUGH and not lay.group:
            mode = NORMAL
        if bottom and mode not in (DISSOLVE, PASS_THROUGH):
            mode = NORMAL if mode > COLOR_ERASE_LEGACY else NORMAL_LEGACY
            comp_mode = UNION
        else:
            comp_mode = lay.composite_mode
        mask = None
        if lay.mptr and lay.apply_mask:
            m = _mask_pixels(self.r, self.img, lay)
            if m is not None:
                mask = np.zeros((h, w), np.float32)
                x0, y0, x1, y1 = _rect(lay, w, h)
                if x0 < x1 and y0 < y1:
                    mask[y0:y1, x0:x1] = m[y0 - lay.y:y1 - lay.y, x0 - lay.x:x1 - lay.x]
        if lay.group:
            if mode == PASS_THROUGH:
                result = self.stack(lay.children, canvas.copy())
                op = np.full((h, w), lay.opacity, np.float32)
                if mask is not None:
                    op *= mask
                return canvas + (result - canvas) * op[..., None]
            src = self.stack(lay.children)
            region = (0, 0, w, h)
        else:
            px = self.pixels(lay)
            src = None
            region = _rect(lay, w, h)
        if _identity_outside(mode, comp_mode) and not lay.group:
            x0, y0, x1, y1 = region
            if x0 >= x1 or y0 >= y1:
                return canvas
        else:
            x0, y0, x1, y1 = 0, 0, w, h
        if src is None:
            src = np.zeros((y1 - y0, x1 - x0, 4), np.float32)
            lx0, ly0, lx1, ly1 = _rect(lay, w, h)
            if lx0 < lx1 and ly0 < ly1:
                src[ly0 - y0:ly1 - y0, lx0 - x0:lx1 - x0] = px[ly0 - lay.y:ly1 - lay.y, lx0 - lay.x:lx1 - lay.x]
        else:
            src = src[y0:y1, x0:x1]
        op = np.full((y1 - y0, x1 - x0), lay.opacity, np.float32)
        if mask is not None:
            op *= mask[y0:y1, x0:x1]
        out = canvas
        out[y0:y1, x0:x1] = _composite(canvas[y0:y1, x0:x1], src, op, mode, lay.blend_space, lay.composite_space,
                                       comp_mode, self.work, y0, x0)
        return out


def _tree(layers: list[Layer]) -> list[Layer]:
    """Nests layers by their item paths. GIMP lists the layers depth first and gives only the
    nested ones a path; the others are top-level, in file order."""
    top: list = []
    by_path: dict = {}
    for lay in layers:
        if len(lay.path) > 1:
            parent = by_path.get(tuple(lay.path[:-1]))
            if parent is not None and parent.group:
                parent.children.append(lay)
                by_path[tuple(lay.path)] = lay
                continue
        by_path[(len(top),)] = lay
        top.append(lay)

    def order(lst):
        lst.sort(key=lambda la: la.path[-1] if la.path else 0)
        for la in lst:
            if la.children:
                order(la.children)
    for lay in top:
        if lay.children:
            order(lay.children)
    return top


def parse(data: bytes) -> tuple[_Image, _Reader]:
    img, pos = _header(data)
    r = _Reader(data)
    ptrs = []
    for _ in range(100000):
        p = _pointer(r, pos, img.ptr)
        pos += img.ptr
        if p == 0:
            break
        ptrs.append(p)
    else:
        raise PixportError("The XCF file is damaged (a layer list with no end)")
    seen = set()
    for p in ptrs:
        if p in seen:
            raise PixportError("The XCF file is damaged (a layer listed twice)")
        seen.add(p)
        img.layers.append(_layer(r, img, p))
    return img, r


def read(path: Path, fmt: str = "xcf", **_options) -> Image:
    data = Path(path).read_bytes()
    img, r = parse(data)
    linear = _PRECISION[img.precision][2]
    # composited in linear light (GIMP's default for most modes); perceptual pictures go back to
    # their curve at the end
    comp = _Compositor(r, img, LINEAR)
    tree = _tree(img.layers)
    canvas = comp.stack(tree)
    # the result is made in the canvas itself: a full-size copy less at the peak
    np.clip(canvas[..., 3], 0.0, 1.0, out=canvas[..., 3])
    if not linear:
        canvas[..., :3] = _linear_to_srgb_ext(canvas[..., :3])
    size = _PRECISION[img.precision][0]
    meta: dict = {"bits": {1: 8, 2: 16}.get(size, 32),
                  "xcf_version": img.version, "xcf_precision": img.precision,
                  "xcf_layers": [lay.name for lay in img.layers]}
    par = img.props.get(21, {})
    if isinstance(par, dict):
        if "icc-profile" in par:
            meta["icc"] = bytes(par["icc-profile"])
        if "gimp-comment" in par:
            meta["comment"] = par["gimp-comment"].rstrip(b"\0").decode("utf-8", "replace")
    if 19 in img.props and len(img.props[19]) >= 8:
        xr, yr = struct.unpack_from(">ff", img.props[19])
        if xr > 0 and yr > 0:
            meta["dpi"] = (xr, yr)
    px = canvas[..., [0, 3]] if img.base == 1 else canvas
    return Image(np.ascontiguousarray(px, dtype=np.float32), "linear" if linear else "srgb", meta, "xcf")
