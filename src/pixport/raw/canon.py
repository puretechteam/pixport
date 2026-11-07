"""Canon CR2: a TIFF whose fourth IFD holds the sensor data as lossless JPEG, cut into vertical
slices (tag 0xC640), and whose maker note has the sensor's borders (0x00E0) and the colour data
(0x4001: white balance). sRAW and mRAW hold YCbCr with subsampled chroma instead of a mosaic."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..errors import PixportError, UnsupportedError
from . import RawData, colour, ljpeg
from .common import check_size, guess_bayer, masked_black, saturation
from .tiff import Tiff, camera_metadata

SLICES, SRAW_TYPE = 0xC640, 0xC6C5


def _raw_ifd(t: Tiff):
    """The sensor data's IFD: the first after the previews (IFD3; a Dual Pixel RAW adds the
    second sub-image as IFD4)."""
    for ifd in t.ifds[1:]:
        if ifd.int(259) == 6 and (SLICES in ifd or ifd.int(SRAW_TYPE) in (1, 4)):
            return ifd
    if len(t.ifds) >= 4 and t.ifds[3].int(259) == 6:
        return t.ifds[3]
    raise PixportError("This CR2 has no sensor data")


def colour_data(mn) -> np.ndarray | None:
    e = mn.get(0x4001) if mn is not None else None
    if e is None:
        return None
    if e.type == 7:
        return np.frombuffer(e.raw()[: e.count & ~1], np.dtype("u2").newbyteorder(e.order)).astype(np.int64)
    return e.values().astype(np.int64)


def as_shot_wb(cd: np.ndarray | None) -> tuple[float, float, float] | None:
    """WB_RGGBLevelsAsShot: at index 63 in the colour data of the CR2 generation (25 or 34 in
    the oldest layouts, 71 in the PowerShots' 5120-byte one)."""
    if cd is None:
        return None
    n = len(cd)
    at = {582: 25, 653: 34, 2560: 71}.get(n, 63)
    if at + 4 > n:
        return None
    r, g1, g2, b = (int(x) for x in cd[at:at + 4])
    g = (g1 + g2) / 2
    if min(r, g, b) <= 0:
        return None
    return (r / g, 1.0, b / g)


def _preview(t: Tiff) -> np.ndarray | None:
    """The small uncompressed 16-bit RGB picture CR2s carry (IFD2; some PowerShots label it
    compression 6 although it is stored plain), or None."""
    for ifd in t.ifds[1:3]:
        w, h, bps = ifd.int(256), ifd.int(257), ifd.ints(258)
        off, size = ifd.int(273), ifd.int(279)
        if not w or not h or bps != [16, 16, 16] or off is None or size is None:
            continue
        n = w * h * 3
        if (ifd.int(259) != 1 and size != n * 2) or n * 2 > size or off + n * 2 > len(t.data) or w * h > 4_000_000:
            continue
        return np.frombuffer(t.data, np.dtype("u2").newbyteorder(t.order), n, off).reshape(h, w, 3)
    return None


def decode(path: Path, fmt: str = "cr2") -> RawData:
    data = path.read_bytes()
    t = Tiff(data)
    ifd = _raw_ifd(t)
    off, size = ifd.int(273), ifd.int(279)
    if off is None or size is None or off >= len(data):
        raise PixportError("This CR2's sensor data is missing")
    end = min(len(data), off + size)
    sraw = ifd.int(SRAW_TYPE) == 4
    samples, frame = ljpeg.decode(data, off, end, chain=sraw)
    slices = ifd.ints(SLICES)
    sl = tuple(slices[:3]) if slices and len(slices) >= 3 else None
    width = (sl[0] * sl[1] + sl[2]) if sl and sl[0] else samples.shape[1]
    if width <= 0 or samples.size % width:
        raise PixportError("This CR2's slices don't match its data")
    height = samples.size // width
    check_size(width, height)
    img = ljpeg.unslice(samples, height, width, sl)

    mk = t.makernote()
    mn = mk[1] if mk and mk[0] == "canon" else None
    cd = colour_data(mn)
    md = camera_metadata(t)
    wb = as_shot_wb(cd) or (2.0, 1.0, 1.5)
    orient = t.ifds[0].int(274, 1) if t.ifds else 1
    if len(frame.comps) == 3 and frame.slots != [0, 1, 2]:
        raw = _sraw(img, frame, cd, wb, orient, md)
        colour.attach(raw, t)
        return raw

    raw = RawData(cfa=img, wb=wb, orientation=orient, metadata=md)
    info = mn.ints(0xE0) if mn is not None else None
    if info and len(info) >= 9 and info[1] == width and info[2] == height:
        left, top, right, bottom = info[5], info[6], info[7], info[8]
        if 0 <= left < right < width and 0 <= top < bottom < height:
            raw.crop = (top, left, bottom - top + 1, right - left + 1)
    if raw.crop is None:
        raw.crop = (0, 0, height, width)
    top, left = raw.crop[0], raw.crop[1]
    mask = info[9:13] if info and len(info) >= 13 else None
    raw.black = masked_black(img, np.zeros((2, 2)), left=left, top=top, mask=mask)
    # CR2 doesn't record its Bayer layout (it varies by model): read it from the picture
    raw.pattern = guess_bayer(img, raw.crop, wb, raw.black, _preview(t))
    bits = frame.precision
    raw.white = float(saturation(img, raw.crop, (1 << bits) - 1, raw.black))
    raw.metadata["bits"] = bits
    colour.attach(raw, t)
    return raw


def _interp(a: np.ndarray, axis: int) -> np.ndarray:
    """Double ``a`` along ``axis``: the samples at even positions, the means of their neighbours
    between (the last one repeated)."""
    a = a.astype(np.float32)
    n = a.shape[axis]
    out_shape = list(a.shape)
    out_shape[axis] = 2 * n
    out = np.empty(out_shape, np.float32)
    even = [slice(None)] * a.ndim
    odd = [slice(None)] * a.ndim
    even[axis] = slice(0, None, 2)
    odd[axis] = slice(1, None, 2)
    out[tuple(even)] = a
    nxt = np.concatenate([np.take(a, range(1, n), axis=axis), np.take(a, [n - 1], axis=axis)], axis=axis)
    out[tuple(odd)] = (a + nxt) * np.float32(0.5)
    return out


def _sraw(img, frame, cd, wb, orient, md) -> RawData:
    """sRAW/mRAW: luma for each pixel, chroma (offset by 16384) once per 2x1 or 2x2 block, made
    from white-balanced camera RGB. The colour data's version tells the two encodings apart:
    versions 6 and 7 (2008-2010 bodies) use ITU-R BT.601 with the chroma divided by four, the others
    R = Y + Cr, B = Y + Cb, G = Y - 0.19 Cb - 0.5 Cr."""
    slots = frame.slots
    k = len(slots)
    nl = slots.count(0)
    if nl not in (2, 4) or k != nl + 2:
        raise UnsupportedError("This Canon sRAW layout isn't supported")
    m = img.reshape(img.shape[0], -1, k)
    rows, cols = m.shape[0], m.shape[1]
    vert = 2 if nl == 4 else 1
    y = np.empty((rows * vert, cols * 2), np.float32)
    for i in range(nl):
        y[(i >> 1)::vert, (i & 1)::2] = m[:, :, i]
    cb = _interp(m[:, :, nl].astype(np.float32) - 16384, 1)
    cr = _interp(m[:, :, nl + 1].astype(np.float32) - 16384, 1)
    if vert == 2:
        cb, cr = _interp(cb, 0), _interp(cr, 0)
    version = int(cd[0]) if cd is not None and len(cd) else 0
    enc = wb
    if version in (6, 7):
        cb, cr = cb * 4, cr * 4
        # these bodies balance an sRAW's colours with WB_RGGBLevelsMeasured (colour data index
        # 78), not the as-shot levels (measured against LibRaw: within 0.2%)
        if len(cd) >= 82 and min(int(x) for x in cd[78:82]) > 0:
            r0, g1, g2, b0 = (int(x) for x in cd[78:82])
            enc = ((g1 + g2) / 2 / r0, 1.0, (g1 + g2) / 2 / b0)
        r = y + np.float32(1.402) * cr
        g = y - np.float32(0.344136) * cb - np.float32(0.714136) * cr
        b = y + np.float32(1.772) * cb
    else:
        r = y + cr
        g = y - np.float32(0.19) * cb - np.float32(0.5) * cr
        b = y + cb
    # back to camera RGB before white balance (develop applies it again)
    rgb = np.stack([r / np.float32(enc[0]), g, b / np.float32(enc[2])], axis=2)
    np.maximum(rgb, 0, out=rgb)
    raw = RawData(linear=rgb, wb=wb, orientation=orient, metadata=md)
    raw.white = float(saturation(y.astype(np.uint16), (0, 0, y.shape[0], y.shape[1]), 16383.0))
    raw.metadata["bits"] = 14
    raw.metadata["sraw"] = "sRAW" if vert == 1 else "mRAW"
    return raw
