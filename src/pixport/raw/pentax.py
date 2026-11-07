"""Pentax PEF: a TIFF whose IFD0 holds the mosaic, uncompressed or Huffman coded (compression
65535): JPEG-style difference coding, MSB-first without stuffing, with a code table from the maker
note (0x0220: code values and lengths) or a fixed default; each row runs two horizontal predictors
seeded from four vertical ones (row and column parity). The maker note also has the black levels
(0x0200) and the white balance (0x0201)."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .. import _accel
from ..errors import PixportError
from . import RawData, colour
from .common import check_size, enough, masked_edges, need
from .ljpeg import build_lut
from .tiff import Tiff, camera_metadata

DEFAULT = ((0, 2, 3, 1, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0), (3, 4, 2, 5, 1, 6, 0, 7, 8, 9, 10, 11, 12))


def table(meta: bytes | None, order: str) -> np.ndarray:
    """The 16-bit lookup table (length << 8 | symbol) from tag 0x0220, or the default code."""
    if meta is None or len(meta) < 14:
        return build_lut(*DEFAULT)
    depth = (struct.unpack_from(order + "H", meta, 0)[0] + 12) & 15
    if len(meta) < 14 + depth * 3:
        raise PixportError("The PEF Huffman table is cut short")
    codes = struct.unpack_from(order + f"{depth}H", meta, 14)
    lengths = meta[14 + depth * 2:14 + depth * 3]
    t12 = np.zeros(4096, np.uint16)
    for c in range(depth):
        ln = lengths[c]
        if not 1 <= ln <= 12:
            continue
        lo = codes[c] & 4095
        span = 4096 >> ln
        idx = (lo + np.arange(span)) & 4095
        t12[idx] = (ln << 8) | c
    return t12[np.arange(65536) >> 4]


def _decode(data: bytes, off: int, width: int, height: int, lut: np.ndarray) -> np.ndarray:
    enough(data, off, width * height, 1, "PEF data")
    if _accel.has("pentax_decode"):
        return np.asarray(_accel.native.pentax_decode(data, off, width, height, lut)).reshape(height, width)
    buf = data[off:] + b"\0" * 8
    tab = lut.tolist()
    diffs = np.empty(width * height, np.int64)
    pos = 0
    frm = int.from_bytes
    for i in range(width * height):
        b = pos >> 3
        sh = pos & 7
        w = frm(buf[b:b + 5], "big") if b + 5 <= len(buf) else frm((buf[b:] + b"\0" * 5)[:5], "big")
        e = tab[(w >> (24 - sh)) & 0xFFFF]
        if not e:
            raise PixportError("The PEF data has a bad Huffman code")
        ln, s = e >> 8, e & 0xFF
        pos += ln
        if s == 0:
            diffs[i] = 0
        elif s >= 16:
            diffs[i] = -32768
        else:
            v = (w >> (40 - sh - ln - s)) & ((1 << s) - 1)
            diffs[i] = v if v >> (s - 1) else v - (1 << s) + 1
            pos += s
    d = diffs.reshape(height, width)
    out = np.empty((height, width), np.int64)
    for par in (0, 1):
        for c in (0, 1):
            out[par::2, c] = np.cumsum(d[par::2, c])
    for c in (0, 1):
        out[:, c + 2::2] = out[:, c:c + 1] + np.cumsum(d[:, c + 2::2], axis=1)
    return (out & 0xFFFF).astype(np.uint16)


def decode(path: Path, fmt: str = "pef") -> RawData:
    data = path.read_bytes()
    t = Tiff(data)
    ifd = t.ifds[0] if t.ifds else None
    if ifd is None or not ifd.int(256):
        raise PixportError("This PEF has no sensor data")
    width, height, bits = ifd.int(256), ifd.int(257), ifd.int(258, 12)
    check_size(width, height)
    comp = ifd.int(259, 1)
    off = (ifd.ints(273) or [0])[0]
    size = sum(ifd.ints(279) or [0])
    need(data, off, 1)
    size = min(size, len(data) - off)
    mk = t.makernote()
    mn = mk[1] if mk and mk[0] == "pentax" else None
    n = width * height
    if comp == 65535:
        meta = mn.get(0x0220) if mn is not None else None
        lut = table(meta.raw() if meta is not None else None, meta.order if meta is not None else ">")
        cfa = _decode(data, off, width, height, lut)
    elif size >= n * 2:
        cfa = np.frombuffer(data, "<u2" if t.order == "<" else ">u2", n, off).reshape(height, width).astype(np.uint16)
    else:
        raise PixportError("This PEF's sensor data is smaller than its size")
    md = camera_metadata(t)
    pat = ifd.ints(33422) or [0, 1, 1, 2]
    raw = RawData(cfa=cfa, pattern=np.array(pat[:4], np.uint8).reshape(2, 2), white=float((1 << bits) - 1), metadata=md)
    raw.orientation = ifd.int(274, 1) or 1
    if mn is not None:
        bl = mn.ints(0x0200)
        if bl and len(bl) >= 4:
            raw.black = np.array([[bl[0], bl[1]], [bl[2], bl[3]]], dtype=np.float64)
        wb = mn.ints(0x0201)
        if wb and len(wb) >= 4 and min(wb[:4]) > 0:
            g = (wb[1] + wb[2]) / 2
            raw.wb = (wb[0] / g, 1.0, wb[3] / g)
    raw.crop = masked_edges(cfa, raw.black, raw.white)
    raw.metadata["bits"] = bits
    colour.attach(raw, t)
    return raw
