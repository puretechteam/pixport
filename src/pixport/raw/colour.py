"""Where a camera's colour matrix comes from, for files that aren't DNGs: the maker's own matrix
when the file has one (Olympus, Sony), else one fitted from the camera's own small preview of the
same picture (raw/preview.py); develop() records which in ``metadata["colour"]``."""

from __future__ import annotations

import struct

import numpy as np

from . import preview

# linear sRGB (D65) <- XYZ, IEC 61966-2-1
SRGB_XYZ = np.array([[3.2404542, -1.5371385, -0.4985314],
                     [-0.9692660, 1.8760108, 0.0415560],
                     [0.0556434, -0.2040259, 1.0572252]])

FROM_FILE = "matrix: the camera's own, from the file's maker note"
FROM_DNG = "matrix: the file's DNG colour matrices"
FROM_PREVIEW = "matrix: fitted from the camera's preview"
NONE = "camera colour (no colour matrix found)"


def from_rgb_cam(m) -> np.ndarray | None:
    """The XYZ -> camera matrix equivalent to a camera (white balanced) -> sRGB one, as develop
    expects: rows scaled to sum 1 (white stays white), inverted, times sRGB-from-XYZ."""
    m = np.asarray(m, dtype=np.float64).reshape(3, 3)
    sums = m.sum(axis=1, keepdims=True)
    if np.any(np.abs(sums) < 1e-6):
        return None
    try:
        return np.linalg.inv(m / sums) @ SRGB_XYZ
    except np.linalg.LinAlgError:
        return None


def from_image(img: np.ndarray, raw) -> np.ndarray | None:
    """XYZ -> camera fitted from an 8-bit RGB rendering of the picture (the camera's thumbnail),
    through preview.fit_matrix (which cuts letterbox bands and turns an upright preview back)."""
    m = preview.fit_matrix(raw, img)
    return None if m is None else preview.cam_xyz(m, raw.wb)


def exif_thumbnail(jpeg: bytes, base: int = 0) -> tuple[int, int] | None:
    """(offset, size) in ``jpeg`` of the small JPEG its EXIF (APP1) carries, if any."""
    from ..errors import PixportError
    from .tiff import Tiff
    if jpeg[base:base + 2] != b"\xff\xd8":
        return None
    p = base + 2
    while p + 4 <= len(jpeg) and jpeg[p] == 0xFF:
        marker, length = jpeg[p + 1], struct.unpack_from(">H", jpeg, p + 2)[0]
        if marker == 0xE1 and jpeg[p + 4:p + 10] == b"Exif\0\0":
            try:
                t = Tiff(jpeg, p + 10)
            except PixportError:
                return None
            if len(t.ifds) > 1:
                o, n = t.ifds[1].int(513), t.ifds[1].int(514)
                if o is not None and n and p + 10 + o + n <= len(jpeg):
                    return (p + 10 + o, n)
            return None
        if marker == 0xDA:
            break
        p += 2 + length
    return None


def from_thumbnail(t, raw, ifds=None, extra: list | None = None) -> np.ndarray | None:
    """Fit from the smallest usable thumbnail in the file's IFDs (or ``extra`` (offset, size)
    JPEGs): a JPEG (JPEGInterchangeFormat or compression 6/7 strips) or an uncompressed 8-bit RGB
    strip; one that can't be matched is passed over for the next."""
    cands = [("jpeg", o, n) for o, n in (extra or []) if 0 <= o and o + n <= len(t.data)]
    for ifd in ifds if ifds is not None else t.all_ifds():
        w, h = ifd.int(256), ifd.int(257)
        jo, jl = ifd.int(513), ifd.int(514)
        so, sl = ifd.int(273), ifd.int(279)
        comp = ifd.int(259, 1)
        cand = None
        if jo is not None and jl and jo + jl <= len(t.data):
            cand = ("jpeg", jo, jl)
        elif so is not None and sl and so + sl <= len(t.data) and comp in (6, 7) and t.data[so:so + 2] == b"\xff\xd8":
            cand = ("jpeg", so, sl)
        elif comp == 1 and w and h and ifd.ints(258) == [8, 8, 8] and so is not None and sl == w * h * 3 \
                and so + sl <= len(t.data):
            cand = ("rgb", so, sl, w, h)
        if cand is not None and cand[2] <= 400_000:
            cands.append(cand)
    for best in sorted(cands, key=lambda c: c[2]):
        try:
            if best[0] == "jpeg":
                img = preview.decode_jpeg(t.data[best[1]:best[1] + best[2]])
            else:
                img = np.frombuffer(t.data, np.uint8, best[2], best[1]).reshape(best[4], best[3], 3)
        except (ValueError, IndexError, KeyError, struct.error):
            continue
        m = from_image(img, raw)
        if m is not None:
            return m
    return None


def attach(raw, t, ifds=None, extra=None) -> None:
    """Give ``raw`` a matrix fitted from the file's thumbnail when it has none yet, and say in its
    metadata where its colour comes from."""
    if raw.cam_xyz is None and (raw.cfa is not None or raw.linear is not None):
        m = from_thumbnail(t, raw, ifds, extra)
        if m is not None:
            raw.cam_xyz = m
            raw.metadata["colour"] = FROM_PREVIEW
    raw.metadata.setdefault("colour", NONE if raw.cam_xyz is None else FROM_FILE)
