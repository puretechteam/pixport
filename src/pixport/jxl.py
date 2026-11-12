"""JPEG XL (ISO/IEC 18181), the one format pixport doesn't implement itself: it's read and written
through pixport-native, which carries libjxl, the reference implementation (BSD-3-Clause).

Reading gives the first frame (an animation's frame count and durations go in the metadata) with
the orientation applied and straight alpha. Ordinary pictures come back as ``"srgb"`` values in
their own colour space, with ``metadata["icc"]`` when that isn't sRGB (``srgb=True`` converts
them to sRGB instead); HDR (PQ, HLG) and linear floating-point pictures come back as ``"linear"``
light in sRGB primaries. Writing is lossy at ``quality`` 90 by default, as ``magick in.png
out.jxl`` does (libjxl's distance 1.0, effort 7), or ``lossless=True``. EXIF, XMP and the ICC
profile are carried both ways. A JPEG can also be recompressed losslessly (``from_jpeg``), and
such a file turned back into the very same JPEG (``to_jpeg``).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import _accel, limits
from .errors import PixportError, UnsupportedError
from .image import Image, upright_exif

DEFAULT_QUALITY = 90
DEFAULT_EFFORT = 7
NATIVE_MAX_PIXELS = 1 << 28  # the budget pixport-native keeps to when it is given none


def _native():
    n = _accel.native
    if n is None or not hasattr(n, "jxl_decode"):
        raise UnsupportedError("JPEG XL needs pixport-native (pip install pixport-native)")
    return n


def available() -> bool:
    """Whether JPEG XL can be read and written here (pixport-native is installed)."""
    n = _accel.native
    return n is not None and hasattr(n, "jxl_decode")


def distance_from_quality(quality: float) -> float:
    """libjxl's mapping from a JPEG-like quality (0 to 100) to its Butteraugli distance:
    90 is 1.0 (visually lossless), 100 is 0.0 (lossless)."""
    q = float(quality)
    if q >= 100:
        return 0.0
    if q >= 30:
        return 0.1 + (100 - q) * 0.09
    return 53 / 3000 * q * q - 23 / 20 * q + 25


def _budget() -> dict:
    """pixport.limits' budget for pixport-native, when it isn't the default it already keeps to."""
    n = limits.max_pixels()
    return {} if n == NATIVE_MAX_PIXELS else {"max_pixels": n}


def decode(data: bytes, srgb: bool = False) -> Image:
    """A JPEG XL file's bytes as an Image (see the module notes)."""
    _w, _h, _c, bits, px, icc, exif, xmp, info = _native().jxl_decode(bytes(data), srgb=srgb, **_budget())
    if bits == 8:
        pixels = px.astype(np.float32) / np.float32(255)
    elif bits == 16:
        pixels = px.astype(np.float32) / np.float32(65535)
    else:
        pixels = px
    meta: dict = {"bits": info["original_bits"], "jxl_float": info["float"]}
    for key, value in (("icc", icc), ("exif", exif), ("xmp", xmp)):
        if value:
            meta[key] = value
    if info["frames"] > 1 or info["loops"] is not None:
        meta["frames"] = info["frames"]
        meta["durations"] = info["durations"]
        meta["loops"] = info["loops"]
    if info["hdr"]:
        meta["hdr"] = info["hdr"]
    if info["jpeg"]:
        meta["jxl_jpeg"] = True
    return Image(pixels, "linear" if info["linear"] else "srgb", meta, "jxl")


def read(path: str | os.PathLike, fmt: str = "jxl", srgb: bool = False, **_options) -> Image:
    """Decode a .jxl file: the first frame; ``srgb=True`` converts other colour spaces to sRGB."""
    _native()
    return decode(Path(path).read_bytes(), srgb=srgb)


def _samples(img: Image, lossless: bool) -> tuple[np.ndarray, bool]:
    """The pixels as the uint8, uint16 or float32 array libjxl takes, and whether it's linear."""
    px = np.asarray(img.pixels)
    if px.ndim == 2:
        px = px[:, :, None]
    if px.ndim != 3 or px.shape[2] not in (1, 2, 3, 4):
        raise PixportError(f"Can't write a picture shaped {px.shape} as JPEG XL")
    if px.shape[0] < 1 or px.shape[1] < 1:
        raise PixportError("Can't write an empty picture")
    if img.transfer == "linear":
        return np.ascontiguousarray(px, dtype=np.float32), True
    if px.dtype in (np.uint8, np.uint16):
        return np.ascontiguousarray(px), False
    px = np.asarray(px, dtype=np.float32)
    if img.metadata.get("jxl_float") and lossless:
        return np.ascontiguousarray(px), False
    if int(img.metadata.get("bits") or 8) > 8:
        return np.rint(np.clip(px, 0, 1) * 65535).astype(np.uint16), False
    return np.rint(np.clip(px, 0, 1) * 255).astype(np.uint8), False


def encode(img: Image, quality: float = DEFAULT_QUALITY, lossless: bool = False, distance: float | None = None,
           effort: int = DEFAULT_EFFORT) -> bytes:
    """An Image as JPEG XL bytes: lossy at ``quality`` (0 to 100, 90 by default, like ImageMagick)
    or at a Butteraugli ``distance`` (0.0 to 25.0; 1.0 is visually lossless), or ``lossless``
    (also what quality 100 means); ``effort`` 1 (fastest) to 10 (smallest). sRGB values are
    written with 8 bits, or 16 when ``metadata["bits"]`` is more than 8; linear pictures as
    32-bit float, so HDR survives. EXIF, XMP and the ICC profile in the metadata are kept."""
    n = _native()
    if distance is None:
        distance = distance_from_quality(quality)
    lossless = bool(lossless) or distance <= 0
    px, linear = _samples(img, lossless)
    md = img.metadata
    return n.jxl_encode(px, lossless=lossless, distance=float(distance), effort=int(effort),
                        icc=None if linear else md.get("icc"), exif=upright_exif(md.get("exif")), xmp=md.get("xmp"),
                        linear=linear)


def write(img: Image, path: str | os.PathLike, quality: float = DEFAULT_QUALITY, lossless: bool = False,
          distance: float | None = None, effort: int = DEFAULT_EFFORT, **_options) -> None:
    """Encode img to a .jxl file (the options are ``encode``'s)."""
    Path(path).write_bytes(encode(img, quality=quality, lossless=lossless, distance=distance, effort=effort))


def from_jpeg(data: bytes, effort: int = DEFAULT_EFFORT) -> bytes:
    """A JPEG's bytes recompressed losslessly as JPEG XL: typically a fifth smaller, every pixel
    the same, and ``to_jpeg`` gives the original file back. Raises PixportError for the rare
    JPEGs libjxl can't take this way (then decode the JPEG and ``encode`` its pixels)."""
    return _native().jxl_from_jpeg(bytes(data), effort=int(effort))


def to_jpeg(data: bytes) -> bytes:
    """The JPEG a ``from_jpeg`` file was made from, byte for byte."""
    return _native().jxl_to_jpeg(bytes(data), **_budget())
