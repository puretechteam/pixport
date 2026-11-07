"""A small XCF writer for pixport's tests: layers (with masks, groups and modes), every precision
and tile compression, written the way GIMP lays a file out."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field

import numpy as np

PRECISION_DTYPE = {100: "u1", 150: "u1", 200: ">u2", 250: ">u2", 300: ">u4", 350: ">u4", 500: ">f2", 550: ">f2",
                   600: ">f4", 650: ">f4", 700: ">f8", 750: ">f8"}


@dataclass
class L:
    name: str
    pixels: np.ndarray | None = None  # (h, w, channels) already in the file's sample type; None for groups
    x: int = 0
    y: int = 0
    opacity: float = 1.0
    visible: bool = True
    mode: int = 28
    blend_space: int | None = None
    composite_space: int | None = None
    composite_mode: int | None = None
    mask: np.ndarray | None = None  # (h, w) in the sample type
    apply_mask: bool = True
    children: list = field(default_factory=list)
    type: int | None = None
    size: tuple | None = None  # (w, h) for groups


def _prop(t: int, payload: bytes) -> bytes:
    return struct.pack(">II", t, len(payload)) + payload


def _string(s: str) -> bytes:
    b = s.encode() + b"\0"
    return struct.pack(">I", len(b)) + b


def rle_stream(b: bytes) -> bytes:
    """GIMP's tile RLE for one stream, using every opcode."""
    out = bytearray()
    i, n = 0, len(b)
    while i < n:
        j = i + 1
        while j < n and b[j] == b[i]:
            j += 1
        run = j - i
        if run >= 3:
            if run <= 127:
                out += bytes((run - 1, b[i]))
            else:
                out += bytes((127, run >> 8, run & 255, b[i]))
            i = j
            continue
        j = i
        while j < n and not (j + 2 < n and b[j] == b[j + 1] == b[j + 2]):
            j += 1
        lit = b[i:j]
        if len(lit) < 128:
            out += bytes((256 - len(lit),)) + lit
        else:
            out += bytes((128, len(lit) >> 8, len(lit) & 255)) + lit
        i = j
    return bytes(out)


class XcfWriter:
    def __init__(self, width: int, height: int, base: int = 0, precision: int = 150, compression: int = 1,
                 version: int | None = None, colormap: np.ndarray | None = None, icc: bytes | None = None,
                 resolution: tuple | None = None):
        self.w, self.h, self.base, self.prec, self.comp = width, height, base, precision, compression
        self.version = version if version is not None else (11 if precision == 150 else 12)
        self.ptr = 8 if self.version >= 11 else 4
        self.colormap, self.icc, self.res = colormap, icc, resolution
        self.buf = bytearray()

    def p(self, v: int) -> bytes:
        return struct.pack(">Q" if self.ptr == 8 else ">I", v)

    def _tiles(self, px: np.ndarray) -> int:
        """Writes a hierarchy for (h, w, c) samples; returns its offset."""
        h, w, _c = px.shape
        raw = np.ascontiguousarray(px).view(np.uint8).reshape(h, w, -1)
        bpp = raw.shape[2]
        tiles = []
        for ty in range(0, h, 64):
            for tx in range(0, w, 64):
                t = raw[ty:ty + 64, tx:tx + 64].reshape(-1, bpp)
                if self.comp == 0:
                    tiles.append(t.tobytes())
                elif self.comp == 1:
                    tiles.append(b"".join(rle_stream(t[:, k].tobytes()) for k in range(bpp)))
                else:
                    tiles.append(zlib.compress(t.tobytes()))
        level_at = len(self.buf) + 12 + 2 * self.ptr
        hier = struct.pack(">III", w, h, bpp) + self.p(level_at) + self.p(0)
        self.buf += hier
        start = level_at + 8 + self.ptr * (len(tiles) + 1)
        offs = []
        for t in tiles:
            offs.append(start)
            start += len(t)
        self.buf += struct.pack(">II", w, h) + b"".join(self.p(o) for o in offs) + self.p(0)
        for t in tiles:
            self.buf += t
        return level_at - 12 - 2 * self.ptr

    def _layer_props(self, lay: L, path: list | None) -> bytes:
        props = _prop(6, struct.pack(">I", round(lay.opacity * 255)))
        props += _prop(33, struct.pack(">f", lay.opacity))
        props += _prop(8, struct.pack(">I", int(lay.visible)))
        props += _prop(7, struct.pack(">I", lay.mode))
        props += _prop(15, struct.pack(">ii", lay.x, lay.y))
        if lay.blend_space is not None:
            props += _prop(37, struct.pack(">i", lay.blend_space))
        if lay.composite_space is not None:
            props += _prop(36, struct.pack(">i", lay.composite_space))
        if lay.composite_mode is not None:
            props += _prop(35, struct.pack(">i", lay.composite_mode))
        if lay.mask is not None:
            props += _prop(11, struct.pack(">I", int(lay.apply_mask)))
        if lay.pixels is None:
            props += _prop(29, b"") + _prop(31, struct.pack(">I", 1))
        if path and len(path) > 1:
            props += _prop(30, struct.pack(f">{len(path)}I", *path))
        return props + _prop(0, b"")

    def _write_layer(self, lay: L, path: list) -> int:
        group = lay.pixels is None
        if group:
            w, h = lay.size or (self.w, self.h)
            nch = 4 if self.base == 0 else 2
            px = np.zeros((h, w, nch), PRECISION_DTYPE[self.prec] if self.base != 2 else "u1")
        else:
            px = lay.pixels
            h, w = px.shape[:2]
        typ = lay.type if lay.type is not None else {0: 0, 1: 2, 2: 4}[self.base] + (1 if px.shape[2] in (2, 4) else 0)
        hptr = self._tiles(px)
        mptr = 0
        if lay.mask is not None:
            mh = self._tiles(lay.mask[:, :, None])
            mptr = len(self.buf)
            self.buf += struct.pack(">II", w, h) + _string("mask") + _prop(0, b"") + self.p(mh)
        at = len(self.buf)
        self.buf += struct.pack(">III", w, h, typ) + _string(lay.name) + self._layer_props(lay, path)
        self.buf += self.p(hptr) + self.p(mptr)
        if self.version >= 20:
            self.buf += self.p(0)
        return at

    def build(self, layers: list[L]) -> bytes:
        """layers: topmost first; groups carry children (topmost first)."""
        order: list[tuple[L, list]] = []

        def walk(items, prefix):
            for i, lay in enumerate(items):
                order.append((lay, prefix + [i]))
                if lay.pixels is None:
                    walk(lay.children, prefix + [i])
        walk(layers, [])
        tag = b"file" if self.version == 0 else b"v%03d" % self.version
        head = b"gimp xcf " + tag + b"\0" + struct.pack(">III", self.w, self.h, self.base)
        if self.version >= 4:
            head += struct.pack(">I", self.prec)
        props = _prop(17, bytes([self.comp]))
        if self.colormap is not None:
            cm = np.asarray(self.colormap, np.uint8)
            props += _prop(1, struct.pack(">I", len(cm)) + cm.tobytes())
        if self.res is not None:
            props += _prop(19, struct.pack(">ff", *self.res))
        if self.icc is not None:
            par = _string("icc-profile") + struct.pack(">II", 1, len(self.icc)) + self.icc
            props += _prop(21, par)
        props += _prop(0, b"")
        head += props
        table_size = self.ptr * (len(order) + 3)
        self.buf = bytearray(len(head) + table_size)
        ptrs = [self._write_layer(lay, path) for lay, path in order]
        table = b"".join(self.p(x) for x in ptrs) + self.p(0) + self.p(0) + self.p(0)
        self.buf[:len(head) + table_size] = head + table
        return bytes(self.buf)


def u8(a) -> np.ndarray:
    return np.asarray(np.rint(np.asarray(a, np.float64) * 255), np.uint8)
