"""OpenEXR, written from the OpenEXR file layout document and the descriptions of its compressions.

A file is a magic number, a version word (with flags for tiles, long names, deep data and several
parts), one header per part (attributes: ``name\\0 type\\0 size value``), a table of chunk offsets
per part, then the chunks. A chunk holds a block of scanlines or one tile; inside, each line holds
each channel's samples in turn (channels sorted by name, little-endian half / float / uint).

Compressions: NONE; RLE and ZIP(S) (bytes split into two halves and delta coded, then run-length
or zlib); PIZ (a 16-bit look-up table, a Haar wavelet per channel and a Huffman code); PXR24
(floats cut to 24 bits, bytes split into planes, delta coded, zlib); B44/B44A (4x4 blocks of
halves in 14 bytes, or 3 for a flat block); DWAA/DWAB (8x8 DCT of perceptually coded halves,
Huffman-coded AC, zlib DC, run-length alpha).

``read`` returns the main picture: the first image part (or the one named like a beauty pass),
its unprefixed R G B A (or Y with RY BY), placed in the display window, in linear light with
sRGB primaries and straight alpha. ``read_channels`` gives every channel of a part as stored.
"""

from __future__ import annotations

import math
import os
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import limits
from ._accel import native
from .errors import PixportError, UnsupportedError
from .hdr import _expand, linearise
from .image import Image, each

MAGIC = b"\x76\x2f\x31\x01"
MAX_SIDE = 50_000
PAD_LIMIT = 1 << 24  # display-window pixels past which empty padding around the data isn't made
UINT, HALF, FLOAT = 0, 1, 2
_SIZE = {UINT: 4, HALF: 2, FLOAT: 4}
_DTYPE = {UINT: np.dtype("<u4"), HALF: np.dtype("<f2"), FLOAT: np.dtype("<f4")}
NONE, RLE, ZIPS, ZIP, PIZ, PXR24, B44, B44A, DWAA, DWAB = range(10)
COMPRESSION_NAMES = ["none", "rle", "zips", "zip", "piz", "pxr24", "b44", "b44a", "dwaa", "dwab", "zstd",
                     "htj2k256", "htj2k32"]
_LINES = {NONE: 1, RLE: 1, ZIPS: 1, ZIP: 16, PIZ: 32, PXR24: 16, B44: 32, B44A: 32, DWAA: 32, DWAB: 256}

_REC709 = (0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290)


# ---------------------------------------------------------------- colour

def rgb_to_xyz(prim) -> np.ndarray:
    """The 3x3 matrix taking RGB with these chromaticities (rx ry gx gy bx by wx wy) to XYZ,
    white at Y = 1. Primaries may lie on the edge of the diagram (an XYZ file's blue at y = 0)."""
    rx, ry, gx, gy, bx, by, wx, wy = (float(v) for v in prim)
    if wy <= 0:
        raise ValueError("bad chromaticities")
    p = np.array([[rx, gx, bx], [ry, gy, by], [1 - rx - ry, 1 - gx - gy, 1 - bx - by]])
    w = np.array([wx / wy, 1.0, (1 - wx - wy) / wy])
    return p * np.linalg.solve(p, w)


XYZ_TO_SRGB = np.linalg.inv(rgb_to_xyz(_REC709))


_BRADFORD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])


def _white_xyz(wx: float, wy: float) -> np.ndarray:
    return np.array([wx / wy, 1.0, (1 - wx - wy) / wy])


def _adapt_to_d65(prim) -> np.ndarray:
    """The Bradford matrix taking XYZ under these primaries' white to XYZ under sRGB's D65 (the
    identity when the white already is D65, to four decimals)."""
    if abs(prim[6] - _REC709[6]) < 1e-4 and abs(prim[7] - _REC709[7]) < 1e-4:
        return np.eye(3)
    src = _BRADFORD @ _white_xyz(prim[6], prim[7])
    dst = _BRADFORD @ _white_xyz(_REC709[6], _REC709[7])
    return np.linalg.inv(_BRADFORD) @ np.diag(dst / src) @ _BRADFORD


def primaries_matrix(prim) -> np.ndarray | None:
    """The matrix taking RGB in these primaries to linear sRGB, or None when they're already
    sRGB's (to four decimals). A white other than D65 (ACES's D60, an XYZ file's equal-energy
    white) is adapted to D65 with the Bradford transform, so a neutral stays neutral."""
    try:
        prim = tuple(float(v) for v in prim)
        if len(prim) != 8 or not all(math.isfinite(v) for v in prim):
            return None
        if max(abs(a - b) for a, b in zip(prim, _REC709)) < 1e-4:
            return None
        out = XYZ_TO_SRGB @ _adapt_to_d65(prim) @ rgb_to_xyz(prim)
    except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
        return None
    return out if np.all(np.isfinite(out)) else None


# ---------------------------------------------------------------- header

@dataclass
class Channel:
    name: str
    type: int
    plinear: bool = False
    xs: int = 1
    ys: int = 1


@dataclass
class Part:
    attrs: dict
    channels: list
    dw: tuple  # x0, y0, x1, y1 (inclusive)
    disp: tuple
    compression: int
    kind: str  # scanlineimage, tiledimage, deepscanline, deeptile
    tile: tuple | None = None  # (width, height, level mode, rounding)
    chunks: int = 0
    offsets: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint64))

    @property
    def name(self) -> str:
        return self.attrs.get("name", "")


def _cstr(data: bytes, pos: int, limit: int = 256) -> tuple[str, int]:
    end = data.find(b"\0", pos, pos + limit)
    if end < 0:
        raise PixportError("The EXR header is damaged (a name with no end)")
    return data[pos:end].decode("latin-1"), end + 1


def _attr(typ: str, v: bytes):
    n = len(v)
    if typ == "box2i" and n == 16:
        return struct.unpack("<4i", v)
    if typ == "box2f" and n == 16:
        return struct.unpack("<4f", v)
    if typ in ("compression", "lineOrder", "envmap", "deepImageState") and n == 1:
        return v[0]
    if typ == "int" and n == 4:
        return struct.unpack("<i", v)[0]
    if typ == "float" and n == 4:
        return struct.unpack("<f", v)[0]
    if typ == "double" and n == 8:
        return struct.unpack("<d", v)[0]
    if typ in ("v2i", "v2f") and n == 8:
        return struct.unpack("<2i" if typ == "v2i" else "<2f", v)
    if typ in ("v3i", "v3f") and n == 12:
        return struct.unpack("<3i" if typ == "v3i" else "<3f", v)
    if typ == "chromaticities" and n == 32:
        return struct.unpack("<8f", v)
    if typ == "tiledesc" and n == 9:
        x, y, mode = struct.unpack("<IIB", v)
        return (x, y, mode & 15, mode >> 4)
    if typ == "rational" and n == 8:
        return struct.unpack("<iI", v)
    if typ == "string":
        return v.decode("utf-8", "replace")
    if typ == "stringvector":
        out, pos = [], 0
        while pos + 4 <= n:
            k = struct.unpack_from("<i", v, pos)[0]
            pos += 4
            if k < 0 or pos + k > n:
                break
            out.append(v[pos:pos + k].decode("utf-8", "replace"))
            pos += k
        return out
    if typ == "chlist":
        chans, pos = [], 0
        while pos < n and v[pos] != 0:
            name, pos = _cstr(v, pos)
            if pos + 16 > n:
                raise PixportError("The EXR channel list is cut short")
            t, plin, xs, ys = struct.unpack_from("<iB3xii", v, pos)
            pos += 16
            chans.append(Channel(name, t, bool(plin), xs, ys))
        return chans
    return v


def parse(data: bytes) -> tuple[int, list[Part], int]:
    """The version flags, the parts and where the offset tables start."""
    if len(data) < 8 or data[:4] != MAGIC:
        raise PixportError("Not an OpenEXR file (bad magic number)")
    flags = struct.unpack_from("<I", data, 4)[0]
    if flags & 0xFF != 2:
        raise UnsupportedError(f"pixport doesn't read OpenEXR version {flags & 0xFF} files")
    if flags & ~0x1EFF:
        raise UnsupportedError("This OpenEXR file uses format flags pixport doesn't know")
    multi = bool(flags & 0x1000)
    pos = 8
    headers: list[dict] = []
    n = len(data)
    while True:
        attrs: dict = {}
        types: dict = {}
        while True:
            if pos >= n:
                raise PixportError("The EXR header is cut short")
            if data[pos] == 0:
                pos += 1
                break
            name, pos = _cstr(data, pos)
            typ, pos = _cstr(data, pos)
            if pos + 4 > n:
                raise PixportError("The EXR header is cut short")
            size = struct.unpack_from("<i", data, pos)[0]
            pos += 4
            if size < 0 or pos + size > n:
                raise PixportError(f"The EXR header is damaged (attribute {name!r} runs past the end)")
            attrs[name] = _attr(typ, data[pos:pos + size])
            types[name] = typ
            pos += size
            if len(attrs) > 10000:
                raise PixportError("The EXR header is damaged (too many attributes)")
        headers.append(attrs)
        if not multi:
            break
        if pos >= n:
            raise PixportError("The EXR header is cut short")
        if data[pos] == 0:
            pos += 1
            break
        if len(headers) > 10000:
            raise PixportError("The EXR file is damaged (too many parts)")
    parts = []
    for attrs in headers:
        parts.append(_part(attrs, flags, multi))
    return flags, parts, pos


def _part(attrs: dict, flags: int, multi: bool) -> Part:
    for key in ("channels", "compression", "dataWindow", "displayWindow"):
        if key not in attrs:
            raise PixportError(f"The EXR header is damaged (no {key})")
    chans = attrs["channels"]
    dw, disp = attrs["dataWindow"], attrs["displayWindow"]
    if (not isinstance(chans, list) or not isinstance(dw, tuple) or not isinstance(disp, tuple) or len(dw) != 4
            or len(disp) != 4 or not all(isinstance(v, int) for v in dw + disp)):
        raise PixportError("The EXR header is damaged (its data or display window)")
    kind = attrs.get("type")
    if not isinstance(kind, str):
        kind = "tiledimage" if flags & 0x200 else ("deepscanline" if flags & 0x800 else "scanlineimage")
    comp = attrs["compression"]
    if not isinstance(comp, int):
        raise PixportError("The EXR header is damaged (compression)")
    w, h = dw[2] - dw[0] + 1, dw[3] - dw[1] + 1
    if w < 1 or h < 1 or w > MAX_SIDE or h > MAX_SIDE:
        raise PixportError(f"The EXR data window ({w} x {h}) is out of range")
    limits.check(w, h, "EXR data window")
    for c in chans:
        if c.type not in _SIZE:
            raise PixportError(f"The EXR channel {c.name!r} has an unknown pixel type ({c.type})")
        if not (1 <= c.xs <= w and 1 <= c.ys <= h):
            raise PixportError(f"The EXR channel {c.name!r} has a bad sampling rate")
    tile = None
    if kind in ("tiledimage", "deeptile"):
        t = attrs.get("tiles")
        if not isinstance(t, tuple) or t[0] < 1 or t[1] < 1 or t[0] > 65536 or t[1] > 65536:
            raise PixportError("The EXR tile description is missing or damaged")
        tile = t
    part = Part(attrs, chans, tuple(dw), tuple(disp), comp, kind, tile)
    if multi:
        cc = attrs.get("chunkCount")
        if not isinstance(cc, int) or cc < 0:
            raise PixportError("The EXR part has no chunk count")
        part.chunks = cc
    else:
        part.chunks = _chunk_count(part)
    return part


def _level_size(full: int, level: int, rounding: int) -> int:
    size = full // (1 << level) if rounding == 0 else -(-full // (1 << level))
    return max(size, 1)


def _chunk_count(part: Part) -> int:
    x0, y0, x1, y1 = part.dw
    w, h = x1 - x0 + 1, y1 - y0 + 1
    if part.tile is None:
        lines = _LINES.get(part.compression, 1)
        return -(-h // lines)
    tw, th, mode, rnd = part.tile
    if mode == 0:
        return (-(-w // tw)) * (-(-h // th))
    levels_of = (lambda size: (math.floor(math.log2(size)) if rnd == 0 else math.ceil(math.log2(size))) + 1)
    if mode == 1:
        n = levels_of(max(w, h))
        return sum((-(-_level_size(w, k, rnd) // tw)) * (-(-_level_size(h, k, rnd) // th)) for k in range(n))
    if mode == 2:
        nx, ny = levels_of(w), levels_of(h)
        return sum((-(-_level_size(w, i, rnd) // tw)) * (-(-_level_size(h, j, rnd) // th))
                   for j in range(ny) for i in range(nx))
    raise PixportError("The EXR tile description has an unknown level mode")


def _offsets(data: bytes, parts: list[Part], pos: int) -> None:
    for part in parts:
        n = part.chunks
        if n > (len(data) - pos) // 8:
            raise PixportError("The EXR offset table is cut short")
        part.offsets = np.frombuffer(data, "<u8", n, pos)
        pos += 8 * n


# ---------------------------------------------------------------- blocks

@dataclass
class _Block:
    x0: int
    y0: int
    x1: int
    y1: int
    chans: list  # (Channel, nx, ny)

    @property
    def raw_size(self) -> int:
        return sum(nx * ny * _SIZE[c.type] for c, nx, ny in self.chans)


def _count(lo: int, hi: int, s: int) -> int:
    """How many multiples of s lie in [lo, hi]."""
    return hi // s - (lo - 1) // s


def _block(part: Part, x0: int, y0: int, x1: int, y1: int) -> _Block:
    return _Block(x0, y0, x1, y1, [(c, _count(x0, x1, c.xs), _count(y0, y1, c.ys)) for c in part.channels])


def _sub(blk: _Block) -> bool:
    return any(c.ys != 1 for c, _nx, _ny in blk.chans)


def _split(buf, blk: _Block) -> list[np.ndarray]:
    """Per-channel (ny, nx) arrays from a block's bytes in file order (line by line, each line
    channel by channel)."""
    b = np.frombuffer(buf, np.uint8)
    if not _sub(blk):
        lines = blk.y1 - blk.y0 + 1
        rows = b.reshape(lines, -1)
        out, off = [], 0
        for c, nx, _ny in blk.chans:
            k = nx * _SIZE[c.type]
            out.append(np.ascontiguousarray(rows[:, off:off + k]).view(_DTYPE[c.type]).reshape(lines, nx))
            off += k
        return out
    parts: list[list] = [[] for _ in blk.chans]
    off = 0
    for y in range(blk.y0, blk.y1 + 1):
        for i, (c, nx, _ny) in enumerate(blk.chans):
            if y % c.ys == 0:
                k = nx * _SIZE[c.type]
                parts[i].append(b[off:off + k])
                off += k
    out = []
    for (c, nx, ny), rows in zip(blk.chans, parts):
        arr = np.concatenate(rows) if rows else np.zeros(0, np.uint8)
        out.append(np.ascontiguousarray(arr).view(_DTYPE[c.type]).reshape(ny, nx))
    return out


def _planar_u16(u16: np.ndarray, blk: _Block) -> list[np.ndarray]:
    """Per-channel arrays from channel-after-channel 16-bit words (as PIZ and B44 leave them)."""
    out, off = [], 0
    raw = np.ascontiguousarray(u16, dtype="<u2")
    for c, nx, ny in blk.chans:
        k = nx * ny * _SIZE[c.type] // 2
        out.append(raw[off:off + k].view(_DTYPE[c.type]).reshape(ny, nx))
        off += k
    return out


def _inflate(raw: bytes, size: int) -> bytes:
    d = zlib.decompressobj()
    out = d.decompress(raw, size)
    if len(out) != size:
        raise PixportError("The EXR data is damaged (zlib data of the wrong size)")
    return out


def _unpredict(buf: bytes) -> bytes:
    """Undoes ZIP and RLE's preprocessing: byte deltas (+128), then the two interleaved halves."""
    t = np.frombuffer(buf, np.uint8)
    if t.size == 0:
        return b""
    d = t - np.uint8(128)
    d[0] = t[0]
    t = np.cumsum(d, dtype=np.uint8)
    out = np.empty_like(t)
    half = (t.size + 1) // 2
    out[0::2] = t[:half]
    out[1::2] = t[half:]
    return out.tobytes()


def _predict(buf: bytes) -> bytes:
    t = np.frombuffer(buf, np.uint8)
    if t.size == 0:
        return b""
    t = np.concatenate([t[0::2], t[1::2]])
    d = np.empty_like(t)
    d[0] = t[0]
    d[1:] = t[1:] - t[:-1] + np.uint8(128)
    return d.tobytes()


def rle_unpack(raw: bytes, size: int) -> bytes:
    """OpenEXR's run-length code: a signed count; negative: that many literal bytes follow,
    otherwise the next byte repeats count + 1 times."""
    if native is not None and hasattr(native, "exr_rle"):
        return native.exr_rle(raw, size)
    starts, lengths, steps = [], [], []
    pos, n, total = 0, len(raw), 0
    while pos < n and total < size:
        c = raw[pos]
        if c >= 128:
            k = 256 - c
            if pos + 1 + k > n:
                raise PixportError("The EXR run-length data is cut short")
            starts.append(pos + 1)
            steps.append(1)
            pos += 1 + k
        else:
            k = c + 1
            if pos + 1 >= n:
                raise PixportError("The EXR run-length data is cut short")
            starts.append(pos + 1)
            steps.append(0)
            pos += 2
        lengths.append(k)
        total += k
    if total != size:
        raise PixportError("The EXR run-length data is damaged (wrong size)")
    return _expand(raw, starts, lengths, steps, size).tobytes()


# ---------------------------------------------------------------- Huffman (PIZ, DWA)

def _huf_table(data: bytes, pos: int, im: int, iM: int) -> tuple[np.ndarray, int]:
    """The code lengths of symbols im..iM: 6 bits each, 59-62 a short run of zero lengths, 63 a
    long one (8 more bits). Returns the lengths and where the coded bits start."""
    lengths = np.zeros(iM + 1, np.int64)
    n = len(data)
    acc = 0
    have = 0
    i = im
    while i <= iM:
        while have < 6:
            if pos >= n:
                raise PixportError("The EXR Huffman table is cut short")
            acc = (acc << 8 | data[pos]) & 0xFFFFFFFF
            pos += 1
            have += 8
        have -= 6
        ln = (acc >> have) & 63
        if ln == 63:
            while have < 8:
                if pos >= n:
                    raise PixportError("The EXR Huffman table is cut short")
                acc = (acc << 8 | data[pos]) & 0xFFFFFFFF
                pos += 1
                have += 8
            have -= 8
            run = ((acc >> have) & 255) + 6
            if i + run > iM + 1:
                raise PixportError("The EXR Huffman table is damaged")
            i += run
        elif ln >= 59:
            run = ln - 59 + 2
            if i + run > iM + 1:
                raise PixportError("The EXR Huffman table is damaged")
            i += run
        else:
            lengths[i] = ln
            i += 1
    return lengths, pos


def _canonical(lengths: np.ndarray) -> np.ndarray:
    """OpenEXR's canonical codes: within a length, codes rise with the symbol; longer codes take
    the smaller values."""
    counts = np.bincount(lengths, minlength=59)[:59]
    start = np.zeros(59, np.int64)
    c = 0
    for ln in range(58, 0, -1):
        start[ln] = c
        c = (c + int(counts[ln])) >> 1
    codes = np.zeros(lengths.size, np.int64)
    used = lengths > 0
    syms = np.flatnonzero(used)
    ls = lengths[syms]
    order = np.lexsort((syms, ls))
    rank = np.empty(syms.size, np.int64)
    sorted_l = ls[order]
    first = np.searchsorted(sorted_l, sorted_l, side="left")
    rank[order] = np.arange(syms.size) - first
    codes[syms] = start[ls] + rank
    return codes


def huf_decode(data: bytes, nraw: int) -> np.ndarray:
    """nraw 16-bit values from OpenEXR's Huffman stream (a 20-byte header: lowest and highest
    symbol, table length, bit count; the code lengths; the bits). The highest symbol is the run
    code: 8 more bits say how many times the value before repeats."""
    if nraw == 0:
        return np.zeros(0, np.uint16)
    if native is not None and hasattr(native, "exr_huf"):
        return np.frombuffer(native.exr_huf(data, nraw), "<u2")
    if len(data) < 20:
        raise PixportError("The EXR Huffman data is cut short")
    im, iM, _tlen, nbits, _ = struct.unpack_from("<5I", data)
    if im >= 65537 or iM >= 65537 or im > iM:
        raise PixportError("The EXR Huffman data is damaged (symbol range)")
    lengths, pos = _huf_table(data, 20, im, iM)
    if nbits > 8 * (len(data) - pos):
        raise PixportError("The EXR Huffman data is cut short")
    rlc = iM
    codes = _canonical(lengths)
    used = np.flatnonzero(lengths)
    if used.size == 0:
        raise PixportError("The EXR Huffman table is empty")
    L = int(lengths.max())
    # each code as an interval of L-bit windows; the intervals must not overlap
    lo = codes[used] << (L - lengths[used])
    hi = (codes[used] + 1) << (L - lengths[used])
    order = np.argsort(lo, kind="stable")
    lo, hi, sym_sorted = lo[order], hi[order], used[order]
    if np.any(hi[:-1] > lo[1:]):
        raise PixportError("The EXR Huffman table is damaged (codes overlap)")
    nbytes = (nbits + 7) // 8
    src = np.frombuffer(data, np.uint8, nbytes, pos)
    # the L-bit window at every bit position (L <= 58: nine bytes cover any start bit)
    padded = np.concatenate([src, np.zeros(9, np.uint8)]).astype(np.uint64)
    bpos = np.arange(nbits, dtype=np.int64)
    byte = bpos >> 3
    word = np.zeros(nbits, np.uint64)
    for k in range(8):
        word = (word << np.uint64(8)) | padded[byte + k]
    sh = (bpos & 7).astype(np.uint64)
    nxt9 = padded[byte + 8]
    word = (word << sh) | (nxt9 >> (np.uint64(8) - sh))
    win = (word >> np.uint64(64 - L)).astype(np.int64) if L < 64 else word.astype(np.int64)
    idx = np.searchsorted(lo, win, side="right") - 1
    ok = (idx >= 0) & (win < hi[np.maximum(idx, 0)])
    sym = np.where(ok, sym_sorted[np.maximum(idx, 0)], -1)
    ln = np.where(ok, lengths[np.maximum(sym, 0)], 0)
    step = ln + np.where(sym == rlc, 8, 0)
    step = np.where(ok, step, 0)
    nxt = (bpos + step).tolist()
    okl = ok.tolist()
    chain = []
    p = 0
    append = chain.append
    while p < nbits:
        if not okl[p]:
            raise PixportError("The EXR Huffman data is damaged (an unknown code)")
        append(p)
        p = nxt[p]
    if p != nbits:
        raise PixportError("The EXR Huffman data is damaged (a code runs past the end)")
    chain = np.asarray(chain, np.int64)
    syms = sym[chain]
    is_run = syms == rlc
    reps = np.ones(chain.size, np.int64)
    if is_run.any():
        rp = chain[is_run] + lengths[rlc]
        # the 8 count bits after each run code
        b0 = padded[rp >> 3].astype(np.int64)
        b1 = padded[(rp >> 3) + 1].astype(np.int64)
        reps[is_run] = ((b0 << 8 | b1) >> (8 - (rp & 7))) & 255
        values = np.where(is_run, -1, syms)
        last = np.maximum.accumulate(np.where(values >= 0, np.arange(values.size), -1))
        if is_run[0] or np.any(last < 0):
            raise PixportError("The EXR Huffman data is damaged (a run with nothing before it)")
        syms = syms[last]
    if int(reps.sum()) < nraw:
        raise PixportError("The EXR Huffman data is cut short")
    out = np.repeat(syms, reps)
    if out.size != nraw:
        raise PixportError("The EXR Huffman data is damaged (too much data)")
    return out.astype(np.uint16)


def _huf_lengths(freq: np.ndarray) -> np.ndarray:
    """Huffman code lengths for these frequencies (a two-queue construction)."""
    syms = np.flatnonzero(freq)
    n = syms.size
    lengths = np.zeros(freq.size, np.int64)
    if n == 1:
        lengths[syms[0]] = 1
        return lengths
    order = syms[np.argsort(freq[syms], kind="stable")]
    leaves = freq[order].tolist()
    parent = [0] * (2 * n - 1)
    weights = leaves + [0] * (n - 1)
    i, j, k = 0, n, n  # next leaf, next internal node, next free slot

    def take():
        nonlocal i, j
        if i < n and (j >= k or weights[i] <= weights[j]):
            i += 1
            return i - 1
        j += 1
        return j - 1
    while k < 2 * n - 1:
        a = take()
        b = take()
        weights[k] = weights[a] + weights[b]
        parent[a] = parent[b] = k
        k += 1
    depth = [0] * (2 * n - 1)
    for node in range(2 * n - 3, -1, -1):
        depth[node] = depth[parent[node]] + 1
    lengths[order] = depth[:n]
    return lengths


def huf_encode(values: np.ndarray) -> bytes:
    """OpenEXR's Huffman stream for 16-bit values (runs of 9 or more repeats use the run code)."""
    values = np.asarray(values, np.int64)
    if values.size == 0:
        return b""
    change = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.concatenate([[0], change])
    runs = np.diff(np.concatenate([starts, [values.size]]))
    vals = values[starts]
    # split runs into a symbol plus up to 255 repeats; short repeats are sent as symbols
    pieces_sym, pieces_rep = [], []
    full, rest = runs // 256, runs % 256
    reps_list = np.repeat(np.full(vals.size, 255), full)
    sym_list = np.repeat(vals, full)
    pieces_sym.append(sym_list)
    pieces_rep.append(reps_list)
    keep = rest > 0
    pieces_sym.append(vals[keep])
    pieces_rep.append(rest[keep] - 1)
    order_key = np.concatenate([np.repeat(starts, full) * 2, starts[keep] * 2 + 1])
    sym = np.concatenate(pieces_sym)
    rep = np.concatenate(pieces_rep)
    order = np.argsort(order_key, kind="stable")
    sym, rep = sym[order], rep[order]
    big = rep >= 9
    im = int(values.min())
    rlc = int(values.max()) + 1
    freq = np.zeros(rlc + 1, np.int64)
    small_sym = np.repeat(sym[~big], rep[~big] + 1)
    np.add.at(freq, small_sym, 1)
    np.add.at(freq, sym[big], 1)
    freq[rlc] += max(1, int(big.sum()))
    lengths = _huf_lengths(freq)
    if lengths.max() > 58:
        raise PixportError("Can't write this picture with PIZ (a Huffman code is too long)")
    codes = _canonical(lengths)
    # the token stream: (code, length) pairs, and 8-bit counts after run codes
    # small runs expand into repeated symbols
    expand = np.where(big, 1, rep + 1)
    s_exp = np.repeat(sym, expand)
    is_big_exp = np.repeat(big, expand)
    first_of = np.zeros(s_exp.size, bool)
    first_of[np.cumsum(expand) - expand] = True
    big_tok = is_big_exp & first_of
    # each expanded symbol gives one code; a big run adds the run code and an 8-bit count
    per = np.where(big_tok, 3, 1)
    total = int(per.sum())
    tok_code = np.empty(total, np.int64)
    tok_len = np.empty(total, np.int64)
    at = np.cumsum(per) - per
    tok_code[at] = codes[s_exp]
    tok_len[at] = lengths[s_exp]
    bi = at[big_tok]
    tok_code[bi + 1] = codes[rlc]
    tok_len[bi + 1] = lengths[rlc]
    tok_code[bi + 2] = rep[big]
    tok_len[bi + 2] = 8
    nbits = int(tok_len.sum())
    bits = _pack_bits(tok_code, tok_len, nbits)
    table = _huf_pack_table(lengths, im, rlc)
    head = struct.pack("<5I", im, rlc, len(table), nbits, 0)
    return head + table + bits


def _pack_bits(code: np.ndarray, length: np.ndarray, nbits: int) -> bytes:
    """Codes written most significant bit first, one after another."""
    nbytes = (nbits + 7) // 8
    if nbits == 0:
        return b""
    ends = np.cumsum(length)
    starts = ends - length
    bit_tok = np.repeat(np.arange(code.size), length)
    k = np.arange(nbits, dtype=np.int64) - np.repeat(starts, length)
    bit = (code[bit_tok] >> (length[bit_tok] - 1 - k)) & 1
    return np.packbits(bit.astype(np.uint8)).tobytes()[:nbytes]


def _huf_pack_table(lengths: np.ndarray, im: int, iM: int) -> bytes:
    vals: list[tuple[int, int]] = []
    i = im
    ls = lengths.tolist()
    while i <= iM:
        ln = ls[i]
        if ln == 0:
            run = 1
            while i + run <= iM and ls[i + run] == 0 and run < 261:
                run += 1
            if run >= 2:
                if run >= 6:
                    vals.append((63, 6))
                    vals.append((run - 6, 8))
                else:
                    vals.append((59 + run - 2, 6))
                i += run
                continue
        vals.append((ln, 6))
        i += 1
    code = np.array([v for v, _ in vals], np.int64)
    ln = np.array([n for _, n in vals], np.int64)
    return _pack_bits(code, ln, int(ln.sum()))


# ---------------------------------------------------------------- PIZ

def _wdec14(lo: np.ndarray, hi: np.ndarray):
    ls = lo.astype(np.int16).astype(np.int32)
    hs = hi.astype(np.int16).astype(np.int32)
    ai = ls + (hs & 1) + (hs >> 1)
    return (ai & 0xFFFF).astype(np.uint16), ((ai - hs) & 0xFFFF).astype(np.uint16)


def _wdec16(lo: np.ndarray, hi: np.ndarray):
    m = lo.astype(np.int32)
    d = hi.astype(np.int32)
    bb = (m - (d >> 1)) & 0xFFFF
    aa = (d + bb - 0x8000) & 0xFFFF
    return aa.astype(np.uint16), bb.astype(np.uint16)


def _wenc14(a: np.ndarray, b: np.ndarray):
    as_ = a.astype(np.int16).astype(np.int32)
    bs = b.astype(np.int16).astype(np.int32)
    return (((as_ + bs) >> 1) & 0xFFFF).astype(np.uint16), ((as_ - bs) & 0xFFFF).astype(np.uint16)


def _wenc16(a: np.ndarray, b: np.ndarray):
    ao = (a.astype(np.int32) + 0x8000) & 0xFFFF
    bi = b.astype(np.int32)
    m = (ao + bi) >> 1
    d = ao - bi
    m = np.where(d < 0, (m + 0x8000) & 0xFFFF, m)
    return m.astype(np.uint16), (d & 0xFFFF).astype(np.uint16)


def wav_decode(a: np.ndarray, maxval: int) -> None:
    """Undoes PIZ's 2D Haar wavelet in place (a: 2D uint16 view)."""
    ny, nx = a.shape
    wdec = _wdec14 if maxval < (1 << 14) else _wdec16
    n = min(nx, ny)
    p = 1
    while p <= n:
        p <<= 1
    p >>= 1
    p2 = p
    p >>= 1
    while p >= 1:
        nby, nbx = ny // p2, nx // p2
        ry0, ry1 = slice(0, nby * p2, p2), slice(p, nby * p2, p2)
        cx0, cx1 = slice(0, nbx * p2, p2), slice(p, nbx * p2, p2)
        if nby and nbx:
            i00, i10 = wdec(a[ry0, cx0], a[ry1, cx0])
            i01, i11 = wdec(a[ry0, cx1], a[ry1, cx1])
            a[ry0, cx0], a[ry0, cx1] = wdec(i00, i01)
            a[ry1, cx0], a[ry1, cx1] = wdec(i10, i11)
        if nx & p and nby:
            col = nbx * p2
            a[ry0, col], a[ry1, col] = wdec(a[ry0, col], a[ry1, col])
        if ny & p and nbx:
            row = nby * p2
            a[row, cx0], a[row, cx1] = wdec(a[row, cx0], a[row, cx1])
        p2 = p
        p >>= 1


def wav_encode(a: np.ndarray, maxval: int) -> None:
    ny, nx = a.shape
    wenc = _wenc14 if maxval < (1 << 14) else _wenc16
    n = min(nx, ny)
    p, p2 = 1, 2
    while p2 <= n:
        nby, nbx = ny // p2, nx // p2
        ry0, ry1 = slice(0, nby * p2, p2), slice(p, nby * p2, p2)
        cx0, cx1 = slice(0, nbx * p2, p2), slice(p, nbx * p2, p2)
        i00, i01 = wenc(a[ry0, cx0], a[ry0, cx1])
        i10, i11 = wenc(a[ry1, cx0], a[ry1, cx1])
        a[ry0, cx0], a[ry1, cx0] = wenc(i00, i10)
        a[ry0, cx1], a[ry1, cx1] = wenc(i01, i11)
        if nx & p:
            col = nbx * p2
            a[ry0, col], a[ry1, col] = wenc(a[ry0, col], a[ry1, col])
        if ny & p:
            row = nby * p2
            a[row, cx0], a[row, cx1] = wenc(a[row, cx0], a[row, cx1])
        p = p2
        p2 <<= 1


def _piz_geom(blk: _Block) -> list[tuple[int, int, int]]:
    return [(nx, ny, _SIZE[c.type] // 2) for c, nx, ny in blk.chans]


def piz_decode(raw: bytes, blk: _Block) -> list[np.ndarray]:
    nraw = blk.raw_size // 2
    if native is not None and hasattr(native, "exr_piz"):
        u16 = np.frombuffer(native.exr_piz(raw, _piz_geom(blk), nraw), "<u2")
        return _planar_u16(u16, blk)
    if len(raw) < 4:
        raise PixportError("The EXR PIZ data is cut short")
    mn, mx = struct.unpack_from("<HH", raw)
    pos = 4
    bitmap = np.zeros(8192, np.uint8)
    if mn <= mx:
        if mx >= 8192 or pos + mx - mn + 1 > len(raw):
            raise PixportError("The EXR PIZ data is damaged (bitmap)")
        bitmap[mn:mx + 1] = np.frombuffer(raw, np.uint8, mx - mn + 1, pos)
        pos += mx - mn + 1
    present = np.unpackbits(bitmap, bitorder="little").astype(bool)
    present[0] = True
    lut = np.zeros(65536, np.uint16)
    vals = np.flatnonzero(present).astype(np.uint16)
    lut[:vals.size] = vals
    maxval = vals.size - 1
    if pos + 4 > len(raw):
        raise PixportError("The EXR PIZ data is cut short")
    length = struct.unpack_from("<i", raw, pos)[0]
    pos += 4
    if length < 0 or pos + length > len(raw):
        raise PixportError("The EXR PIZ data is damaged (Huffman length)")
    u16 = huf_decode(raw[pos:pos + length], nraw).copy()
    off = 0
    for nx, ny, size in _piz_geom(blk):
        k = nx * ny * size
        arr = u16[off:off + k].reshape(ny, nx, size)
        for j in range(size):
            wav_decode(arr[:, :, j], maxval)
        off += k
    return _planar_u16(lut[u16], blk)


def piz_encode(planes: list[np.ndarray], blk: _Block) -> bytes:
    """PIZ data for per-channel arrays (each viewed as 16-bit words)."""
    u16 = np.concatenate([np.ascontiguousarray(p).view("<u2").ravel() for p in planes]).astype(np.uint16)
    if native is not None and hasattr(native, "exr_piz_encode"):
        return native.exr_piz_encode(u16.astype("<u2").tobytes(), _piz_geom(blk))
    present = np.zeros(65536, bool)
    present[u16] = True
    present[0] = False
    bitmap = np.packbits(present, bitorder="little")
    nz = np.flatnonzero(bitmap)
    present[0] = True
    fwd = np.cumsum(present) - 1
    maxval = int(present.sum()) - 1
    data = fwd[u16].astype(np.uint16)
    off = 0
    for nx, ny, size in _piz_geom(blk):
        k = nx * ny * size
        arr = data[off:off + k].reshape(ny, nx, size)
        for j in range(size):
            wav_encode(arr[:, :, j], maxval)
        off += k
    if nz.size:
        mn, mx = int(nz[0]), int(nz[-1])
        head = struct.pack("<HH", mn, mx) + bitmap[mn:mx + 1].tobytes()
    else:
        head = struct.pack("<HH", 8191, 0)
    huf = huf_encode(data)
    return head + struct.pack("<i", len(huf)) + huf


# ---------------------------------------------------------------- PXR24

def pxr24_decode(raw: bytes, blk: _Block) -> list[np.ndarray]:
    psize = {UINT: 4, HALF: 2, FLOAT: 3}
    size = sum(nx * ny * psize[c.type] for c, nx, ny in blk.chans)
    b = np.frombuffer(_inflate(raw, size), np.uint8)
    rows: list[list] = [[] for _ in blk.chans]
    if not _sub(blk):
        lines = blk.y1 - blk.y0 + 1
        grid = b.reshape(lines, -1)
        off = 0
        for i, (c, nx, _ny) in enumerate(blk.chans):
            k = nx * psize[c.type]
            rows[i] = [grid[:, off:off + k]]
            off += k
    else:
        off = 0
        for y in range(blk.y0, blk.y1 + 1):
            for i, (c, nx, _ny) in enumerate(blk.chans):
                if y % c.ys == 0:
                    k = nx * psize[c.type]
                    rows[i].append(b[off:off + k][None, :])
                    off += k
    out = []
    for (c, nx, ny), r in zip(blk.chans, rows):
        g = np.concatenate(r, axis=0) if r else np.zeros((0, nx * psize[c.type]), np.uint8)
        g = g.reshape(ny, psize[c.type], nx).astype(np.uint32)
        if c.type == HALF:
            d = (g[:, 0] << 8 | g[:, 1]).astype(np.uint16)
            out.append(np.cumsum(d, axis=1, dtype=np.uint16).view("<f2"))
        elif c.type == UINT:
            d = g[:, 0] << 24 | g[:, 1] << 16 | g[:, 2] << 8 | g[:, 3]
            out.append(np.cumsum(d, axis=1, dtype=np.uint32).view("<u4"))
        else:
            d = g[:, 0] << 16 | g[:, 1] << 8 | g[:, 2]
            v = (np.cumsum(d, axis=1, dtype=np.uint32) & 0xFFFFFF) << 8
            out.append(v.astype(np.uint32).view("<f4"))
    return out


# ---------------------------------------------------------------- B44

_EXP_TABLE = None


def _b44_exp_table() -> np.ndarray:
    """B44's table for perceptually linear (pLinear) channels, which the compressor stores as
    exp(x / 8): back to x = 8 ln(y). Zero, negative and non-finite codes become 0."""
    global _EXP_TABLE
    if _EXP_TABLE is None:
        h = np.arange(65536, dtype=np.uint32).astype(np.uint16).view(np.float16).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.where(np.isfinite(h) & (h > 0), 8.0 * np.log(np.where(h > 0, h, 1.0)), 0.0)
        _EXP_TABLE = v.astype(np.float16).view(np.uint16)
    return _EXP_TABLE


_EXP_LIST = None


def _b44_exp_list() -> list:
    global _EXP_LIST
    if _EXP_LIST is None:
        _EXP_LIST = _b44_exp_table().tolist()
    return _EXP_LIST


def _unpack14(b: np.ndarray) -> np.ndarray:
    b = b.astype(np.int64)
    s = np.empty((b.shape[0], 16), np.int64)
    shift = b[:, 2] >> 2
    bias = 0x20 << shift

    def d(v):
        return ((v & 0x3F) << shift) - bias
    s[:, 0] = b[:, 0] << 8 | b[:, 1]
    s[:, 4] = s[:, 0] + d(b[:, 2] << 4 | b[:, 3] >> 4)
    s[:, 8] = s[:, 4] + d(b[:, 3] << 2 | b[:, 4] >> 6)
    s[:, 12] = s[:, 8] + d(b[:, 4])
    s[:, 1] = s[:, 0] + d(b[:, 5] >> 2)
    s[:, 5] = s[:, 4] + d(b[:, 5] << 4 | b[:, 6] >> 4)
    s[:, 9] = s[:, 8] + d(b[:, 6] << 2 | b[:, 7] >> 6)
    s[:, 13] = s[:, 12] + d(b[:, 7])
    s[:, 2] = s[:, 1] + d(b[:, 8] >> 2)
    s[:, 6] = s[:, 5] + d(b[:, 8] << 4 | b[:, 9] >> 4)
    s[:, 10] = s[:, 9] + d(b[:, 9] << 2 | b[:, 10] >> 6)
    s[:, 14] = s[:, 13] + d(b[:, 10])
    s[:, 3] = s[:, 2] + d(b[:, 11] >> 2)
    s[:, 7] = s[:, 6] + d(b[:, 11] << 4 | b[:, 12] >> 4)
    s[:, 11] = s[:, 10] + d(b[:, 12] << 2 | b[:, 13] >> 6)
    s[:, 15] = s[:, 14] + d(b[:, 13])
    return s & 0xFFFF


def _b44_unsign(s: np.ndarray) -> np.ndarray:
    return np.where(s & 0x8000, s & 0x7FFF, ~s & 0xFFFF).astype(np.uint16)


def b44_decode(raw: bytes, blk: _Block) -> list[np.ndarray]:
    if native is not None and hasattr(native, "exr_b44"):
        geom = [(nx, ny, _SIZE[c.type], c.plinear and c.type == HALF) for c, nx, ny in blk.chans]
        table = _b44_exp_list() if any(g[3] for g in geom) else None
        return _planar_u16(np.frombuffer(native.exr_b44(raw, geom, table), "<u2"), blk)
    src = np.frombuffer(raw, np.uint8)
    n = len(raw)
    pos = 0
    out = []
    for c, nx, ny in blk.chans:
        if c.type != HALF:
            k = nx * ny * _SIZE[c.type]
            if pos + k > n:
                raise PixportError("The EXR B44 data is cut short")
            out.append(np.frombuffer(raw, _DTYPE[c.type], nx * ny, pos).reshape(ny, nx))
            pos += k
            continue
        bx, by = -(-nx // 4), -(-ny // 4)
        nb = bx * by
        starts = np.empty(nb, np.int64)
        flat = np.empty(nb, bool)
        for i in range(nb):
            if pos + 3 > n:
                raise PixportError("The EXR B44 data is cut short")
            starts[i] = pos
            if raw[pos + 2] >= 52:
                flat[i] = True
                pos += 3
            else:
                if pos + 14 > n:
                    raise PixportError("The EXR B44 data is cut short")
                flat[i] = False
                pos += 14
        s = np.empty((nb, 16), np.int64)
        if (~flat).any():
            st = starts[~flat]
            s[~flat] = _unpack14(src[st[:, None] + np.arange(14)])
        if flat.any():
            st = starts[flat]
            s[flat] = (src[st].astype(np.int64) << 8 | src[st + 1])[:, None]
        v = _b44_unsign(s)
        if c.plinear:
            v = _b44_exp_table()[v]
        grid = v.reshape(by, bx, 4, 4).transpose(0, 2, 1, 3).reshape(by * 4, bx * 4)
        out.append(np.ascontiguousarray(grid[:ny, :nx]).view("<f2"))
    return out


# ---------------------------------------------------------------- DWA

_DWA_LINEAR = None
_ZIGZAG = np.array([
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21,
    28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54,
    47, 55, 62, 63])
_LEGACY_RULES = [("r", 1, HALF, 0, True), ("red", 1, HALF, 0, True), ("g", 1, HALF, 1, True), ("grn", 1, HALF, 1, True),
                 ("green", 1, HALF, 1, True), ("b", 1, HALF, 2, True), ("blu", 1, HALF, 2, True),
                 ("blue", 1, HALF, 2, True), ("y", 1, HALF, -1, True), ("by", 1, HALF, -1, True),
                 ("ry", 1, HALF, -1, True), ("a", 2, UINT, -1, True), ("a", 2, HALF, -1, True),
                 ("a", 2, FLOAT, -1, True)]


def _dwa_to_linear() -> np.ndarray:
    """DWA's perceptual coding undone: y^2.2 up to 1, exp(2.2 (y - 1)) above; infinities and
    NaNs become 0; the sign is kept."""
    global _DWA_LINEAR
    if _DWA_LINEAR is None:
        bits = np.arange(65536, dtype=np.uint32).astype(np.uint16)
        h = bits.view(np.float16).astype(np.float32)
        a = np.abs(h)
        with np.errstate(over="ignore", invalid="ignore"):
            v = np.where(a <= 1, np.power(a, np.float32(2.2)), np.exp(np.float32(2.2) * (a - np.float32(1))))
            v = np.where(np.isfinite(h), np.where(h < 0, -v, v), np.float32(0)).astype(np.float16)
        _DWA_LINEAR = v.view(np.uint16)
    return _DWA_LINEAR


# the DCT constants as the reference decoder spells them: 0.5 cos(k pi / 16) to seven digits
_A, _B, _C, _D, _E, _F, _G = (np.float32(v) for v in (3.535536e-01, 4.903927e-01, 4.619398e-01, 4.157349e-01,
                                                     2.777855e-01, 1.913422e-01, 9.754573e-02))
_ROWS = [np.array(v, np.float32) for v in (
    [_A, _A, _A, _A], [_B, _D, _E, _G], [_C, _F, -_F, -_C], [_D, -_G, -_B, -_E], [_A, -_A, -_A, _A], [_E, -_B, _G, _D],
    [_F, -_C, _C, -_F], [_G, -_E, _D, -_B])]


def _idct_rows(x: np.ndarray) -> np.ndarray:
    """The row pass of the 8-point inverse DCT, as a matrix product: even and odd halves summed
    in order, then added and subtracted (float32, in the reference decoder's order)."""
    xs = [x[..., k:k + 1] for k in range(8)]
    ev = xs[0] * _ROWS[0]
    od = xs[1] * _ROWS[1]
    for k in (2, 4, 6):
        ev = ev + xs[k] * _ROWS[k]
    for k in (3, 5, 7):
        od = od + xs[k] * _ROWS[k]
    return np.concatenate([ev + od, (ev - od)[..., ::-1]], axis=-1)


def _idct_cols(x: np.ndarray) -> np.ndarray:
    """The column pass, factored (along the last axis)."""
    x0, x1, x2, x3, x4, x5, x6, x7 = (x[..., k] for k in range(8))
    al0, al1, al2, al3 = _C * x2, _F * x2, _C * x6, _F * x6
    be0 = (x1 * _B + x3 * _D) + (x5 * _E + x7 * _G)
    be1 = (x1 * _D - x3 * _G) - (x5 * _B + x7 * _E)
    be2 = (x1 * _E - x3 * _B) + (x5 * _G + x7 * _D)
    be3 = (x1 * _G - x3 * _E) + (x5 * _D - x7 * _B)
    th0 = _A * (x0 + x4)
    th3 = _A * (x0 - x4)
    th1 = al0 + al3
    th2 = al1 - al2
    ga0, ga1, ga2, ga3 = th0 + th1, th3 + th2, th3 - th2, th0 - th1
    return np.stack([ga0 + be0, ga1 + be1, ga2 + be2, ga3 + be3, ga3 - be3, ga2 - be2, ga1 - be1, ga0 - be0], axis=-1)


def idct8x8(blocks: np.ndarray, dc_only: np.ndarray) -> np.ndarray:
    """(n, 8, 8) float32 coefficients to samples; blocks with only a DC take the shortcut the
    reference decoder takes."""
    out = _idct_cols(_idct_rows(blocks).transpose(0, 2, 1)).transpose(0, 2, 1)
    if dc_only.any():
        out[dc_only] = (blocks[dc_only, 0, 0] * _A * _A)[:, None, None]
    return out


def _unrle_ac(ac: np.ndarray, nblocks: int, pos: int) -> tuple[np.ndarray, np.ndarray, int]:
    """nblocks blocks of zig-zag coefficients 1..63 from the AC stream at pos: 0xff00 ends a
    block, 0xffNN skips NN zeros, anything else is the next coefficient (a half)."""
    if native is not None and hasattr(native, "exr_dwa_ac"):
        coef, last, pos = native.exr_dwa_ac(ac.tobytes(), nblocks, pos)
        return np.frombuffer(coef, "<u2").reshape(nblocks, 64).copy(), np.frombuffer(last, np.uint8), pos
    coef = np.zeros((nblocks, 64), np.uint16)
    last = np.zeros(nblocks, np.uint8)
    vals = ac.tolist()
    n = len(vals)
    for blkno in range(nblocks):
        k = 1
        row = coef[blkno]
        lastnz = 0
        while k < 64:
            if pos >= n:
                raise PixportError("The EXR DWA data is cut short (AC values)")
            v = vals[pos]
            pos += 1
            if v == 0xFF00:
                break
            if v >> 8 == 0xFF:
                k += v & 0xFF
            else:
                if k > 63:
                    raise PixportError("The EXR DWA data is damaged (AC run)")
                row[k] = v
                lastnz = k
                k += 1
        last[blkno] = lastnz
    return coef, last, pos


def _dwa_lossy(chans: list, ac: np.ndarray, ac_pos: int, dc: np.ndarray, dc_pos: int, width: int, height: int,
               csc: bool) -> tuple[list[np.ndarray], int, int]:
    """Decodes one DCT group (one channel, or three in Y'CbCr) to linear halves (height, width)."""
    nbx, nby = -(-width // 8), -(-height // 8)
    nb = nbx * nby
    ncomp = len(chans)
    if dc_pos + ncomp * nb > dc.size:
        raise PixportError("The EXR DWA data is cut short (DC values)")
    coefs = []
    lasts = []
    # AC values are interleaved: for each block, each component in turn
    if ncomp == 1:
        cf, ls, ac_pos = _unrle_ac(ac, nb, ac_pos)
        coefs.append(cf)
        lasts.append(ls)
    else:
        allc, alll, ac_pos = _unrle_ac(ac, nb * ncomp, ac_pos)
        for k in range(ncomp):
            coefs.append(allc[k::ncomp])
            lasts.append(alll[k::ncomp])
    samples = []
    for k in range(ncomp):
        cf = coefs[k]
        cf[:, 0] = dc[dc_pos + k * nb: dc_pos + (k + 1) * nb]
        z = np.zeros((nb, 64), np.float32)
        z[:, _ZIGZAG] = cf.view(np.float16).astype(np.float32)
        samples.append(idct8x8(z.reshape(nb, 8, 8), lasts[k] == 0))
    dc_pos += ncomp * nb
    if csc:
        y, cb, cr = samples
        r = y + np.float32(1.5747) * cr
        g = y - np.float32(0.1873) * cb - np.float32(0.4682) * cr
        b = y + np.float32(1.8556) * cb
        samples = [r, g, b]
    lut = _dwa_to_linear()
    out = []
    for s in samples:
        grid = s.reshape(nby, nbx, 8, 8).transpose(0, 2, 1, 3).reshape(nby * 8, nbx * 8)[:height, :width]
        out.append(lut[grid.astype(np.float16).view(np.uint16)])
    return out, ac_pos, dc_pos


def dwa_decode(raw: bytes, blk: _Block) -> list[np.ndarray]:
    if len(raw) < 88:
        raise PixportError("The EXR DWA data is cut short")
    sizes = struct.unpack_from("<11Q", raw)
    version, unk_unc, unk_cmp, ac_cmp, dc_cmp, rle_cmp, rle_unc, rle_raw, ac_count, dc_count, ac_mode = sizes
    if version > 2:
        raise UnsupportedError(f"pixport doesn't read DWA version {version} data")
    pos = 88
    rules = []
    if version == 2:
        if pos + 2 > len(raw):
            raise PixportError("The EXR DWA data is cut short")
        rsize = struct.unpack_from("<H", raw, pos)[0]
        end = pos + rsize
        if rsize < 2 or end > len(raw):
            raise PixportError("The EXR DWA channel rules are damaged")
        pos += 2
        while pos < end:
            suffix, pos = _cstr(raw, pos)
            if pos + 2 > end:
                raise PixportError("The EXR DWA channel rules are damaged")
            value, typ = raw[pos], raw[pos + 1]
            pos += 2
            csc = (value >> 4) - 1
            scheme = (value >> 2) & 3
            if csc < -1 or csc >= 3 or scheme > 2:
                raise PixportError("The EXR DWA channel rules are damaged")
            rules.append((suffix, scheme, typ, csc, bool(value & 1)))
        pos = end
    else:
        rules = list(_LEGACY_RULES)
    total = unk_cmp + ac_cmp + dc_cmp + rle_cmp
    if pos + total > len(raw) or max(sizes) > 1 << 40:
        raise PixportError("The EXR DWA data is cut short")
    raw_max = blk.raw_size
    if unk_unc > raw_max or rle_raw > raw_max or rle_unc > 2 * raw_max + 64 or ac_count > raw_max or dc_count > raw_max:
        raise PixportError("The EXR DWA data is damaged (sizes)")
    unk = _inflate(raw[pos:pos + unk_cmp], unk_unc) if unk_cmp else b""
    pos += unk_cmp
    ac_raw = raw[pos:pos + ac_cmp]
    pos += ac_cmp
    if ac_count == 0:
        ac = np.zeros(0, np.uint16)
    elif ac_mode == 0:
        ac = huf_decode(ac_raw, ac_count)
    elif ac_mode == 1:
        ac = np.frombuffer(_inflate(ac_raw, ac_count * 2), "<u2")
    else:
        raise PixportError("The EXR DWA data is damaged (AC coding)")
    dc = np.frombuffer(_unpredict(_inflate(raw[pos:pos + dc_cmp], dc_count * 2)), "<u2") if dc_cmp else \
        np.zeros(0, np.uint16)
    pos += dc_cmp
    rle = rle_unpack(_inflate(raw[pos:pos + rle_cmp], rle_unc), rle_raw) if rle_cmp else b""
    # classify the channels
    schemes = []
    groups: dict = {}
    for i, (c, nx, ny) in enumerate(blk.chans):
        prefix, _, suffix = c.name.rpartition(".")
        scheme = 0
        for rs, rscheme, rtype, rcsc, nocase in rules:
            if rtype == c.type and (rs == suffix or (nocase and rs.lower() == suffix.lower())):
                scheme = rscheme
                if rcsc >= 0:
                    groups.setdefault(prefix, [-1, -1, -1])[rcsc] = i
                break
        schemes.append(scheme)
    csc_sets = [g for g in groups.values() if min(g) >= 0]
    in_csc = {i for g in csc_sets for i in g}
    out: list = [None] * len(blk.chans)
    ac_pos = dc_pos = 0
    for g in csc_sets:
        _c, nx, ny = blk.chans[g[0]]
        if any(blk.chans[i][1:] != (nx, ny) for i in g):
            raise PixportError("The EXR DWA colour channels differ in size")
        res, ac_pos, dc_pos = _dwa_lossy([blk.chans[i] for i in g], ac, ac_pos, dc, dc_pos, nx, ny, True)
        for i, r in zip(g, res):
            out[i] = r
    for i, (c, nx, ny) in enumerate(blk.chans):
        if schemes[i] == 1 and i not in in_csc:
            res, ac_pos, dc_pos = _dwa_lossy([blk.chans[i]], ac, ac_pos, dc, dc_pos, nx, ny, False)
            out[i] = res[0]
    # the lossy channels as their own types; the others from the RLE and raw buffers
    unk_pos = rle_pos = 0
    for i, (c, nx, ny) in enumerate(blk.chans):
        size = _SIZE[c.type]
        k = nx * ny
        if out[i] is not None:
            h = np.ascontiguousarray(out[i]).view("<f2")
            if c.type == FLOAT:
                out[i] = h.astype("<f4")
            elif c.type == HALF:
                out[i] = h
            else:
                raise PixportError("The EXR DWA data is damaged (a lossy integer channel)")
        elif schemes[i] == 2:
            if rle_pos + k * size > len(rle):
                raise PixportError("The EXR DWA data is cut short (RLE)")
            planes = np.frombuffer(rle, np.uint8, k * size, rle_pos).reshape(size, k)
            out[i] = np.ascontiguousarray(planes.T).view(_DTYPE[c.type]).reshape(ny, nx)
            rle_pos += k * size
        else:
            if unk_pos + k * size > len(unk):
                raise PixportError("The EXR DWA data is cut short (raw channels)")
            out[i] = np.frombuffer(unk, _DTYPE[c.type], k, unk_pos).reshape(ny, nx)
            unk_pos += k * size
    return out


# ---------------------------------------------------------------- chunks

def _decode(part: Part, raw: bytes, blk: _Block) -> list[np.ndarray]:
    expected = blk.raw_size
    comp = part.compression
    if comp == NONE or len(raw) == expected:
        if len(raw) != expected:
            raise PixportError("The EXR data is damaged (an uncompressed block of the wrong size)")
        return _split(raw, blk)
    if comp in (ZIP, ZIPS):
        return _split(_unpredict(_inflate(raw, expected)), blk)
    if comp == RLE:
        return _split(_unpredict(rle_unpack(raw, expected)), blk)
    if comp == PIZ:
        return piz_decode(raw, blk)
    if comp == PXR24:
        return pxr24_decode(raw, blk)
    if comp in (B44, B44A):
        return b44_decode(raw, blk)
    if comp in (DWAA, DWAB):
        return dwa_decode(raw, blk)
    name = COMPRESSION_NAMES[comp] if comp < len(COMPRESSION_NAMES) else str(comp)
    raise UnsupportedError(f"pixport doesn't read EXR files compressed with {name.upper()}")


BATCH = 32  # blocks each thread decodes at a time


def read_part(data: bytes, part: Part, multi: bool, index: int = 0, only: set[str] | None = None) -> dict[str, np.ndarray]:
    """Every channel of a (flat) part, or those named by ``only``, as stored, over its data window:
    {name: (ny, nx) array}."""
    if part.kind.startswith("deep"):
        raise UnsupportedError("This EXR part holds deep data (several samples per pixel), which pixport doesn't read")
    if part.compression not in _LINES:
        name = COMPRESSION_NAMES[part.compression] if part.compression < len(COMPRESSION_NAMES) else str(part.compression)
        raise UnsupportedError(f"pixport doesn't read EXR files compressed with {name.upper()}")
    x0, y0, x1, y1 = part.dw
    planes = {}
    for c in part.channels:
        if only is None or c.name in only:
            nx, ny = _count(x0, x1, c.xs), _count(y0, y1, c.ys)
            planes[c.name] = np.zeros((ny, nx), _DTYPE[c.type])
    n = len(data)
    head = 4 if multi else 0
    if part.tile is None:
        lines = _LINES[part.compression]
        nblocks = -(-(y1 - y0 + 1) // lines)
        if part.offsets.size < nblocks:
            raise PixportError("The EXR offset table is too short")

        def line_block(i: int) -> None:
            off = int(part.offsets[i])
            if off < 8 or off + head + 8 > n:
                raise PixportError("The EXR file is damaged (a chunk offset points outside the file)")
            if multi and struct.unpack_from("<i", data, off)[0] != index:
                raise PixportError("The EXR file is damaged (a chunk of another part)")
            y, size = struct.unpack_from("<ii", data, off + head)
            if size < 0 or off + head + 8 + size > n:
                raise PixportError("The EXR file is damaged (a chunk runs past the end)")
            if (y - y0) % lines or not y0 <= y <= y1:
                raise PixportError("The EXR file is damaged (a chunk for a line outside the picture)")
            if y != y0 + i * lines:  # the table lists the chunks top to bottom, whatever their order in the file
                raise PixportError("The EXR file is damaged (a chunk isn't the one its offset table names)")
            by1 = min(y + lines - 1, y1)
            blk = _block(part, x0, y, x1, by1)
            res = _decode(part, data[off + head + 8: off + head + 8 + size], blk)
            for (c, _nx, ny), arr in zip(blk.chans, res):
                if ny and c.name in planes:
                    r0 = _count(y0, y - 1, c.ys)
                    planes[c.name][r0:r0 + ny] = arr

        each(lambda start: [line_block(i) for i in range(start, min(start + BATCH, nblocks))], range(0, nblocks, BATCH))
    else:
        tw, th = part.tile[0], part.tile[1]
        ntx, nty = -(-(x1 - x0 + 1) // tw), -(-(y1 - y0 + 1) // th)
        if part.offsets.size < ntx * nty:
            raise PixportError("The EXR offset table is too short")
        if any(c.xs != 1 or c.ys != 1 for c in part.channels):
            raise PixportError("The EXR tiled part has subsampled channels")

        def tile_block(i: int) -> None:
            off = int(part.offsets[i])
            if off < 8 or off + head + 20 > n:
                raise PixportError("The EXR file is damaged (a tile offset points outside the file)")
            if multi and struct.unpack_from("<i", data, off)[0] != index:
                raise PixportError("The EXR file is damaged (a tile of another part)")
            tx, ty, lx, ly, size = struct.unpack_from("<5i", data, off + head)
            if lx or ly or (tx, ty) != (i % ntx, i // ntx):
                raise PixportError("The EXR file is damaged (a tile out of place)")
            if size < 0 or off + head + 20 + size > n:
                raise PixportError("The EXR file is damaged (a tile runs past the end)")
            bx0, by0 = x0 + tx * tw, y0 + ty * th
            bx1, by1 = min(bx0 + tw - 1, x1), min(by0 + th - 1, y1)
            blk = _block(part, bx0, by0, bx1, by1)
            res = _decode(part, data[off + head + 20: off + head + 20 + size], blk)
            for (c, nx, ny), arr in zip(blk.chans, res):
                if c.name in planes:
                    planes[c.name][by0 - y0: by0 - y0 + ny, bx0 - x0: bx0 - x0 + nx] = arr

        each(lambda start: [tile_block(i) for i in range(start, min(start + BATCH, ntx * nty))], range(0, ntx * nty, BATCH))
    return planes


def open_file(path) -> tuple[bytes, int, list[Part]]:
    data = Path(path).read_bytes()
    flags, parts, pos = parse(data)
    _offsets(data, parts, pos)
    return data, flags, parts


def read_channels(path, part: int | None = None) -> tuple[dict[str, np.ndarray], Part]:
    """The stored channels of one part (the main picture's part unless part is given)."""
    data, flags, parts = open_file(path)
    index = _main_part(parts) if part is None else part
    if not 0 <= index < len(parts):
        raise PixportError(f"The EXR file has no part {index}")
    return read_part(data, parts[index], bool(flags & 0x1000), index), parts[index]


# ---------------------------------------------------------------- the main picture

_BEAUTY = ("beauty", "rgba", "rgb", "combined", "color", "colour", "main", "image", "default")


def _layers(names: list[str]) -> dict[str, dict[str, str]]:
    """{layer prefix: {R/G/B/A/Y/RY/BY: channel name}} for the layers that hold a picture."""
    out: dict = {}
    for n in names:
        prefix, _, suffix = n.rpartition(".")
        key = {"r": "R", "red": "R", "g": "G", "green": "G", "b": "B", "blue": "B", "a": "A", "alpha": "A",
               "y": "Y", "ry": "RY", "by": "BY"}.get(suffix.lower())
        if key and (suffix in ("R", "G", "B", "A", "Y", "RY", "BY") or key not in out.get(prefix, {})):
            out.setdefault(prefix, {})[key] = n
    return {k: v for k, v in out.items() if set(v) & {"R", "G", "B", "Y"}}


def _main_layer(part: Part) -> tuple[str, dict[str, str]]:
    layers = _layers([c.name for c in part.channels])
    if not layers:
        return "", {}
    if "" in layers:
        return "", layers[""]
    views = part.attrs.get("multiView")
    if isinstance(views, list) and views and views[0] in layers:
        return views[0], layers[views[0]]
    for word in _BEAUTY:
        for k in sorted(layers):
            if word in k.lower():
                return k, layers[k]
    k = min(layers)
    return k, layers[k]


def _main_part(parts: list[Part]) -> int:
    flat = [i for i, p in enumerate(parts) if not p.kind.startswith("deep")]
    if not flat:
        return 0
    with_pic = [i for i in flat if _main_layer(parts[i])[1]]
    for word in _BEAUTY:
        for i in with_pic:
            if word in parts[i].name.lower():
                return i
    return with_pic[0] if with_pic else flat[0]


_CHROMA_TAPS = np.array([0.002128, -0.007540, 0.019597, -0.043159, 0.087929, -0.186077, 0.627123,
                         0.627123, -0.186077, 0.087929, -0.043159, 0.019597, -0.007540, 0.002128], np.float32)


def _upsample_chroma(c: np.ndarray, x_par: int, y_par: int, w: int, h: int) -> np.ndarray:
    """Full-size chroma from samples on every other row and column: the given samples stay, the
    others are filled by a 27-tap half-band filter (OpenEXR's luminance/chroma reconstruction),
    first along rows, then along columns."""
    def along(a, n, par, axis):
        a = np.moveaxis(a, axis, 0)
        full = np.zeros((n,) + a.shape[1:], np.float32)
        pos = np.arange(par, n, 2)
        k = min(len(pos), a.shape[0])
        full[pos[:k]] = a[:k]
        if k == 0:
            return np.moveaxis(full, 0, axis)
        # missing sample i takes the known samples at i-13, i-11, ..., i+13 (clamped to the edges)
        idx = np.arange(n)
        missing = idx[(idx - par) % 2 == 1]
        offs = np.arange(-13, 14, 2)
        src = missing[:, None] + offs[None, :]
        known_lo, known_hi = par, par + 2 * (k - 1)
        src = np.clip(src, known_lo, known_hi)
        src = known_lo + ((src - known_lo) // 2) * 2
        vals = full[src]  # (m, 14, ...)
        full[missing] = np.tensordot(vals, _CHROMA_TAPS, axes=([1], [0])) if vals.ndim == 2 else \
            np.einsum("mt...,t->m...", vals, _CHROMA_TAPS)
        return np.moveaxis(full, 0, axis)
    rows = along(c.astype(np.float32), w, x_par, 1)
    return along(rows, h, y_par, 0)


def _luminance_weights(prim) -> np.ndarray:
    m = rgb_to_xyz(prim if prim is not None else _REC709)
    return m[1].astype(np.float32)


def read(path: Path, fmt: str = "exr", part: int | None = None, **_options) -> Image:
    data, flags, parts = open_file(path)
    multi = bool(flags & 0x1000)
    if part is None:
        index = _main_part(parts)
    else:
        index = part
        if not 0 <= index < len(parts):
            raise PixportError(f"The EXR file has no part {index}")
    p = parts[index]
    if p.kind.startswith("deep"):
        raise UnsupportedError("This EXR file holds deep data (several samples per pixel), which pixport doesn't read")
    layer, chans = _main_layer(p)
    planes = read_part(data, p, multi, index, {*chans.values(), *(c.name for c in p.channels[:1])})
    x0, y0, x1, y1 = p.dw
    w, h = x1 - x0 + 1, y1 - y0 + 1
    prim = p.attrs.get("chromaticities")
    prim = prim if isinstance(prim, tuple) and len(prim) == 8 else None

    def full(name):
        a = planes[name]
        c = next(ch for ch in p.channels if ch.name == name)
        a = a.astype(np.float32)
        if c.xs == 1 and c.ys == 1:
            return a
        if c.xs == 2 and c.ys == 2 and name in (chans.get("RY"), chans.get("BY")):
            return _upsample_chroma(a, (-x0) % 2, (-y0) % 2, w, h)
        ix = (np.arange(w) + x0) // c.xs - (-(-x0 // c.xs))
        iy = (np.arange(h) + y0) // c.ys - (-(-y0 // c.ys))
        return a[np.clip(iy, 0, a.shape[0] - 1)][:, np.clip(ix, 0, a.shape[1] - 1)]

    alpha = full(chans["A"]) if "A" in chans else None
    if {"R", "G", "B"} & set(chans):
        zero = np.zeros((h, w), np.float32)
        rgb = np.stack([full(chans[k]) if k in chans else zero for k in "RGB"], axis=2)
        grey = False
    elif "Y" in chans:
        y = full(chans["Y"])
        if "RY" in chans and "BY" in chans:
            ry, by = full(chans["RY"]), full(chans["BY"])
            yw = _luminance_weights(prim)
            r = (ry + 1) * y
            b = (by + 1) * y
            g = (y - r * yw[0] - b * yw[2]) / yw[1]
            flat = (ry == 0) & (by == 0)
            rgb = np.stack([np.where(flat, y, r), np.where(flat, y, g), np.where(flat, y, b)], axis=2)
            grey = False
        else:
            rgb = y[:, :, None]
            grey = True
    else:
        first = p.channels[0].name if p.channels else None
        if first is None:
            raise PixportError("The EXR part has no channels")
        rgb = full(first)[:, :, None]
        grey = True
    if not grey and prim is not None:
        m = primaries_matrix(prim)
        if m is not None:
            rgb = np.einsum("ij,hwj->hwi", m.astype(np.float32), rgb).astype(np.float32)
    alpha_note = None
    if alpha is not None:
        # colour where alpha is 0 (glows, glass, a junk alpha): the alpha isn't coverage, so the
        # picture is kept as EXR viewers show it, over black, without alpha
        unseen = alpha <= 0
        if np.count_nonzero(np.abs(rgb[unseen]).max(axis=1) > 1e-6) > 0.01 * alpha.size:
            alpha = None
            alpha_note = "dropped: the colour reaches where alpha is 0"
    if alpha is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            straight = np.where(alpha[:, :, None] > 0, rgb / alpha[:, :, None], rgb)
        rgb = np.concatenate([straight, alpha[:, :, None]], axis=2).astype(np.float32)
    # place the data window in the display window
    dx0, dy0, dx1, dy1 = p.disp
    dwid, dhgt = dx1 - dx0 + 1, dy1 - dy0 + 1
    if dwid < 1 or dhgt < 1 or dwid > MAX_SIDE or dhgt > MAX_SIDE:
        raise PixportError(f"The EXR display window ({dwid} x {dhgt}) is out of range")
    cropped = None
    if dwid * dhgt > PAD_LIMIT and dwid * dhgt > 64 * rgb.shape[0] * rgb.shape[1]:
        # a display window far larger than the pixels the file holds (a few bytes can claim one of
        # gigapixels): the picture is the data window instead
        cropped = p.disp
        dx0, dy0, dx1, dy1 = x0, y0, x1, y1
    else:
        limits.check(dwid, dhgt, "EXR display window")
    if (dx0, dy0, dx1, dy1) != (x0, y0, x1, y1):
        canvas = np.zeros((dhgt, dwid, rgb.shape[2]), np.float32)
        sx0, sy0 = max(x0, dx0), max(y0, dy0)
        sx1, sy1 = min(x1, dx1), min(y1, dy1)
        if sx0 <= sx1 and sy0 <= sy1:
            canvas[sy0 - dy0:sy1 - dy0 + 1, sx0 - dx0:sx1 - dx0 + 1] = rgb[sy0 - y0:sy1 - y0 + 1, sx0 - x0:sx1 - x0 + 1]
        rgb = canvas
    meta: dict = {
        "exr_compression": COMPRESSION_NAMES[p.compression] if p.compression < len(COMPRESSION_NAMES) else p.compression,
        "exr_channels": [c.name for c in p.channels],
        "exr_layer": layer,
        "exr_data_window": p.dw,
        "exr_display_window": p.disp,
    }
    if len(parts) > 1:
        meta["exr_parts"] = [q.name or str(i) for i, q in enumerate(parts)]
        meta["exr_part"] = index
    if prim is not None:
        meta["chromaticities"] = prim
    if alpha_note:
        meta["exr_alpha"] = alpha_note
    if cropped is not None:
        meta["exr_cropped"] = "the display window is far larger than the data the file holds: the picture is its data window"
    for key, mkey in (("whiteLuminance", "white_luminance"), ("pixelAspectRatio", "pixel_aspect"),
                      ("owner", "owner"), ("comments", "comments"), ("capDate", "date"), ("view", "view"),
                      ("multiView", "views"), ("xDensity", "dpi")):
        v = p.attrs.get(key)
        if isinstance(v, (int, float, str, list)):
            meta[mkey] = v
    return Image(np.ascontiguousarray(rgb, np.float32), "linear", meta, "exr")


# ---------------------------------------------------------------- writing

def _deflate(data: bytes, level: int) -> bytes:
    """zlib data for ZIP blocks: the smaller of a filtered and a run-length deflate (the predicted
    bytes suit both), which beats the default strategy at the same level."""
    out = None
    for strategy in (zlib.Z_FILTERED, zlib.Z_RLE):
        c = zlib.compressobj(level, zlib.DEFLATED, 15, 9, strategy)
        z = c.compress(data) + c.flush()
        if out is None or len(z) < len(out):
            out = z
    return out


def _attr_bytes(name: str, typ: str, value: bytes) -> bytes:
    return name.encode() + b"\0" + typ.encode() + b"\0" + struct.pack("<i", len(value)) + value


def write(img: Image, path: Path, compression: str = "zip", pixel_type: str = "half", level: int = 6, **_options) -> None:
    """Writes img as a single-part scanline EXR: ``compression`` "zip" (default), "piz", "zips" or
    "none"; ``pixel_type`` "half" (default) or "float"; ``level`` is zlib's effort for ZIP. sRGB
    pictures are turned linear; colour is stored premultiplied by alpha, as EXR expects."""
    comp = {"none": NONE, "zip": ZIP, "zips": ZIPS, "piz": PIZ}.get(str(compression).lower())
    if comp is None:
        raise PixportError(f"pixport writes EXR with ZIP, ZIPS, PIZ or no compression, not {compression}")
    ptype = {"half": HALF, "float": FLOAT}.get(str(pixel_type).lower())
    if ptype is None:
        raise PixportError("pixel_type must be 'half' or 'float'")
    px = np.asarray(img.pixels, np.float32)
    if px.ndim == 2:
        px = px[:, :, None]
    h, w, nc = px.shape
    if h < 1 or w < 1:
        raise PixportError("Can't write an empty picture")
    if img.transfer != "linear":
        colour = linearise(px[:, :, :nc - (1 if nc in (2, 4) else 0)])
    else:
        colour = px[:, :, :nc - (1 if nc in (2, 4) else 0)]
    alpha = px[:, :, -1] if nc in (2, 4) else None
    if alpha is not None:
        colour = colour * np.clip(alpha, 0.0, None)[:, :, None]
    names = ["Y"] if colour.shape[2] == 1 else ["R", "G", "B"]
    planes = {n: colour[:, :, i] for i, n in enumerate(names)}
    if alpha is not None:
        planes["A"] = alpha
    order = sorted(planes)
    dt = _DTYPE[ptype]
    if ptype == HALF:  # brighter than a half can hold (the sun in an HDR): the brightest it holds, not infinity
        planes = {n: np.where(np.isfinite(p), np.clip(p, -65504.0, 65504.0), p) for n, p in planes.items()}
    arrays = [np.ascontiguousarray(planes[n].astype(dt)) for n in order]
    chans = [Channel(n, ptype) for n in order]
    chlist = b"".join(n.encode() + b"\0" + struct.pack("<iB3xii", ptype, 0, 1, 1) for n in order) + b"\0"
    box = struct.pack("<4i", 0, 0, w - 1, h - 1)
    header = (MAGIC + struct.pack("<I", 2)
              + _attr_bytes("channels", "chlist", chlist)
              + _attr_bytes("compression", "compression", bytes([comp]))
              + _attr_bytes("dataWindow", "box2i", box)
              + _attr_bytes("displayWindow", "box2i", box)
              + _attr_bytes("lineOrder", "lineOrder", b"\0")
              + _attr_bytes("pixelAspectRatio", "float", struct.pack("<f", 1.0))
              + _attr_bytes("screenWindowCenter", "v2f", struct.pack("<2f", 0.0, 0.0))
              + _attr_bytes("screenWindowWidth", "float", struct.pack("<f", 1.0))
              + b"\0")
    lines = _LINES[comp]
    part = Part({}, chans, (0, 0, w - 1, h - 1), (0, 0, w - 1, h - 1), comp, "scanlineimage")

    def chunk(y: int) -> bytes:
        y1 = min(y + lines, h)
        blk = _block(part, 0, y, w - 1, y1 - 1)
        sub = [a[y:y1] for a in arrays]
        rawb = np.concatenate([s.view(np.uint8).reshape(y1 - y, -1) for s in sub], axis=1).tobytes()
        if comp == NONE:
            body = rawb
        elif comp in (ZIP, ZIPS):
            body = _deflate(_predict(rawb), level)
        else:
            body = piz_encode(sub, blk)
        if len(body) >= len(rawb):
            body = rawb
        return struct.pack("<ii", y, len(body)) + body
    starts = range(0, h, lines)
    if comp == NONE or len(starts) < 4:
        chunks = [chunk(y) for y in starts]
    else:
        # zlib and the compiled PIZ coder let go of the interpreter: blocks compress side by side
        with ThreadPoolExecutor(max_workers=max(1, min(8, (os.cpu_count() or 2) // 2))) as pool:
            chunks = list(pool.map(chunk, starts))
    table_at = len(header)
    offsets = []
    at = table_at + 8 * len(chunks)
    for ch in chunks:
        offsets.append(at)
        at += len(ch)
    with open(path, "wb") as f:
        f.write(header)
        f.write(np.asarray(offsets, "<u8").tobytes())
        for ch in chunks:
            f.write(ch)
