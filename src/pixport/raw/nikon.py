"""Nikon NEF: a TIFF whose raw SubIFD holds the mosaic, uncompressed (12/14-bit packed or
16-bit) or Huffman coded (compression 34713, "lossy" and "lossless"); the maker note (an embedded
TIFF) has the compression's predictors and linearization curve (0x96), the white balance (0x0C),
black level (0x3D) and the camera's crop (0x45).

The coded stream is MSB-first without byte stuffing. Each photosite's difference is coded as a
Huffman symbol (low nibble: length, high nibble: shift) plus that many bits; each row continues
two running predictors (one per CFA column parity) and the first two columns continue per-row-
parity vertical predictors. "Lossy" files then map the 12/14-bit values through the curve."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .. import _accel
from ..errors import PixportError, UnsupportedError
from . import RawData, colour
from .common import check_size, enough, need, unpack_bits
from .ljpeg import build_lut
from .tiff import Tiff, camera_metadata

# Nikon's Huffman tables (code counts for lengths 1-16, then symbols): 12-bit lossy, 12-bit
# lossy after the split row, 12-bit lossless, then the same three for 14 bits. They are fixed by
# the format (no NEF carries its own), so every NEF reader holds these same numbers.
TREES = (
    ((0, 1, 5, 1, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0), (5, 4, 3, 6, 2, 7, 1, 0, 8, 9, 11, 10, 12, 0)),
    ((0, 1, 5, 1, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0), (0x39, 0x5A, 0x38, 0x27, 0x16, 5, 4, 3, 2, 1, 0, 11, 12, 12)),
    ((0, 1, 4, 2, 3, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0), (5, 4, 6, 3, 7, 2, 8, 1, 9, 0, 10, 11, 12)),
    ((0, 1, 4, 3, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0), (5, 6, 4, 7, 8, 3, 9, 2, 1, 0, 10, 11, 12, 13, 14)),
    ((0, 1, 5, 1, 1, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0), (8, 0x5C, 0x4B, 0x3A, 0x29, 7, 6, 5, 4, 3, 2, 1, 0, 13, 14)),
    ((0, 1, 4, 2, 2, 3, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0), (7, 6, 8, 5, 9, 4, 10, 3, 11, 12, 2, 0, 1, 13, 14)),
)


def tree_lut(n: int) -> np.ndarray:
    counts, syms = TREES[n]
    syms = list(syms) + [0] * max(0, sum(counts) - len(syms))
    return build_lut(counts, syms)


def _symbols(data: bytes, start: int, n: int, luts: list, switch_at: int) -> np.ndarray:
    """``n`` coded differences (switching table at sample ``switch_at``); bits past the end
    read as zeros."""
    buf = data[start:] + b"\0" * 8
    out = np.empty(n, dtype=np.int32)
    lut = luts[0].tolist()
    lut2 = luts[1].tolist() if len(luts) > 1 else lut
    pos = 0
    frm = int.from_bytes
    for i in range(n):
        if i == switch_at:
            lut = lut2
        b = pos >> 3
        sh = pos & 7
        w = frm(buf[b:b + 5], "big")
        e = lut[(w >> (24 - sh)) & 0xFFFF]
        if not e:
            raise PixportError("The NEF data has a bad Huffman code")
        cl = e >> 8
        sym = e & 0xFF
        ln, shl = sym & 15, sym >> 4
        nb = ln - shl
        pos += cl
        if nb > 0:
            v = (w >> (40 - sh - cl - nb)) & ((1 << nb) - 1)
            pos += nb
        else:
            v = 0
        diff = (((v << 1) + 1) << shl) >> 1
        if ln and not diff & (1 << (ln - 1)):
            diff -= (1 << ln) - (0 if shl else 1)
        elif not ln:
            diff = 0
        out[i] = diff
    return out


def _predict(diffs: np.ndarray, height: int, width: int, vpred: list[int]) -> np.ndarray:
    """Running sums: the first two columns down the rows of each parity (from ``vpred``), the
    rest along the row; 16-bit wrap-around, then read as signed and limited to 0..16383."""
    d = diffs.reshape(height, width).astype(np.int64)
    out = np.empty((height, width), dtype=np.int64)
    for par in (0, 1):
        for c in (0, 1):
            col = np.cumsum(d[par::2, c]) + vpred[par * 2 + c]
            out[par::2, c] = col
    for c in (0, 1):
        out[:, c + 2::2] = out[:, c:c + 1] + np.cumsum(d[:, c + 2::2], axis=1)
    out = ((out + 0x8000) & 0xFFFF) - 0x8000
    return np.clip(out, 0, 0x3FFF).astype(np.uint16)


def _curve(meta: bytes, order: str, bits: int) -> tuple[np.ndarray, int, int, list[int], int]:
    """(curve, tree, split row, vpred, white) from the linearization table (0x96)."""
    if len(meta) < 12:
        raise PixportError("The NEF linearization table is cut short")
    ver0, ver1 = meta[0], meta[1]
    p = 2
    if ver0 == 0x49 or ver1 == 0x58:
        p += 2110
    tree = (2 if ver0 == 0x46 else 0) + (3 if bits == 14 else 0)
    if p + 10 > len(meta):
        raise PixportError("The NEF linearization table is cut short")
    vpred = list(struct.unpack_from(order + "4h", meta, p))
    csize = struct.unpack_from(order + "H", meta, p + 8)[0]
    top = (1 << bits) & 0x7FFF
    curve = np.arange(0x10000, dtype=np.int64)
    step = top // (csize - 1) if csize > 1 else 0
    split = 0
    if ver0 == 0x44 and (ver1 == 0x20 or (ver1 == 0x40 and bits == 14)) and step > 0:
        if ver1 == 0x40:
            # the newer 14-bit lossy files code 12-bit values: the curve spans a quarter
            step, top = step // 4, top // 4
            if step < 1:
                raise PixportError("The NEF curve is damaged (too many points)")
        n = min(csize, (len(meta) - p - 10) // 2)
        pts = np.array(struct.unpack_from(order + f"{n}H", meta, p + 10), dtype=np.int64)
        if n < csize:
            raise PixportError("The NEF curve is cut short")
        knots = np.zeros(top + step + 1, dtype=np.int64)
        knots[np.arange(csize) * step] = pts
        i = np.arange(top)
        lo = i - i % step
        curve[:top] = (knots[lo] * (step - i % step) + knots[np.minimum(lo + step, len(knots) - 1)] * (i % step)) // step
        if len(meta) >= 564:
            split = struct.unpack_from(order + "H", meta, 562)[0]
    elif ver0 != 0x46 and csize <= 0x4001:
        n = min(csize, (len(meta) - p - 10) // 2)
        curve[:n] = struct.unpack_from(order + f"{n}H", meta, p + 10)
        top = n
    while top > 2 and curve[top - 2] == curve[top - 1]:
        top -= 1
    return curve.astype(np.uint16), tree, split, vpred, int(curve[max(0, top - 1)])


def decode_compressed(data: bytes, off: int, width: int, height: int, meta: bytes, order: str, bits: int):
    """The mosaic of a Huffman-coded NEF: (uint16 array, white level)."""
    enough(data, off, width * height, 2, "NEF data")
    curve, tree, split, vpred, white = _curve(meta, order, bits)
    if _accel.has("nikon_decode"):
        out = _accel.native.nikon_decode(data, off, width, height, tree_lut(tree),
                                         tree_lut(tree + 1) if split else tree_lut(tree), split, np.array(vpred, np.int64))
        raw = np.asarray(out).reshape(height, width)
    else:
        luts = [tree_lut(tree)] + ([tree_lut(tree + 1)] if split else [])
        switch = split * width if 0 < split < height else -1
        diffs = _symbols(data, off, width * height, luts, switch)
        raw = _predict(diffs, height, width, vpred)
    return curve[raw], white


def decode(path: Path, fmt: str = "nef") -> RawData:
    data = path.read_bytes()
    t = Tiff(data)
    ifd = None
    for cand in t.all_ifds():
        if cand.int(254, 0) == 0 and cand.int(262) == 32803 and cand.int(256):
            ifd = cand
            break
    if ifd is None:
        raise PixportError("This NEF has no sensor data")
    width, height, bits = ifd.int(256), ifd.int(257), ifd.int(258, 12)
    check_size(width, height)
    comp = ifd.int(259, 1)
    offs, counts = ifd.ints(273) or [], ifd.ints(279) or []
    if not offs or not counts:
        raise PixportError("This NEF's sensor data is missing")
    off, size = offs[0], sum(counts) if len(offs) == 1 or offs == sorted(offs) else counts[0]
    size = min(size, len(data) - off)
    need(data, off, 1)
    mk = t.makernote()
    mn = mk[1] if mk and mk[0] == "nikon" else None
    md = camera_metadata(t)
    white = float((1 << bits) - 1)
    n = width * height
    kind = mn.int(0x93) if mn is not None else None  # NEFCompression
    if kind in (13, 14):
        raise UnsupportedError("Nikon's High Efficiency RAW (HE and HE*) isn't supported")
    if kind == 8:
        raise UnsupportedError("Nikon's small NEF (sRAW) isn't supported")
    if comp == 34713 and size * 8 < n * bits:
        meta = mn[0x96].raw() if mn is not None and 0x96 in mn else None
        if meta is None:
            raise PixportError("This compressed NEF has no linearization table")
        cfa, white = decode_compressed(data, off, width, height, meta, mn[0x96].order, bits)
        white = float(white)
    elif size >= n * 2:
        row = max(size // height, width * 2)
        a = np.frombuffer(data, "<u2" if t.order == "<" else ">u2", n if row == width * 2 else height * row // 2, off)
        cfa = (a.reshape(height, -1)[:, :width] if row != width * 2 else a.reshape(height, width)).astype(np.uint16)
    elif size * 8 >= n * bits:
        # packed: 12 or 14 bits back to back, rows possibly padded to 16 bytes
        row_bytes = size // height
        pack = width * bits // 8
        raw = np.frombuffer(data, np.uint8, row_bytes * height, off).reshape(height, row_bytes)[:, :pack]
        cfa = unpack_bits(raw.tobytes(), n, bits, "msb" if t.order == ">" or bits == 12 else "lsb").reshape(height, width)
    else:
        raise PixportError("This NEF's sensor data is smaller than its size")

    pat = ifd.ints(33422) or [0, 1, 1, 2]
    dims = ifd.ints(33421) or [2, 2]
    pattern = np.array(pat[:4], dtype=np.uint8).reshape(2, 2) if dims[:2] == [2, 2] and len(pat) >= 4 else \
        np.array([[0, 1], [1, 2]], np.uint8)
    raw = RawData(cfa=cfa, pattern=pattern, white=white, metadata=md)
    raw.orientation = t.ifds[0].int(274, 1) if t.ifds else 1
    if mn is not None:
        wbv = mn.floats(0x0C)
        if wbv and len(wbv) >= 2 and wbv[0] > 0 and wbv[1] > 0:
            raw.wb = (wbv[0], 1.0, wbv[1])
        blk = mn.ints(0x3D)
        if blk and len(blk) >= 4:
            shift = max(0, 14 - bits)
            raw.black = np.array([[blk[0], blk[1]], [blk[2], blk[3]]], dtype=np.float64) / (1 << shift)
        crop = mn.ints(0x45)
        if crop and len(crop) >= 4 and crop[2] > 0 and crop[3] > 0 and crop[0] + crop[2] <= width \
                and crop[1] + crop[3] <= height:
            raw.crop = (crop[1], crop[0], crop[3], crop[2])
    raw.metadata["bits"] = bits
    colour.attach(raw, t)
    return raw
