"""Canon CR3: an ISO base media file (``ftyp crx``) holding the metadata as small TIFFs (CMT1 the
IFD0, CMT2 the EXIF IFD, CMT3 the maker note, CMT4 GPS, plus more maker-note TIFFs in the timed
metadata track: the colour data with white balance and black levels) and the sensor data as one
sample of a track whose ``CRAW`` sample entry carries a ``CMP1`` box describing its CRX coding.

CRX cuts the mosaic into tiles; each tile holds four planes (the R, G, G and B photosites, each at
half size), and each plane is coded either directly (lossless "RAW") or as an integer 5/3 wavelet
of up to three levels (C-RAW). Every subband has its own bitstream, coded line by line with
adaptive Golomb-Rice codes and JPEG-LS style run lengths; a plane's first subband predicts each
value from its neighbours (the median edge detector). C-RAW multiplies the coefficients back by
a quantisation step, one per subband, or on newer cameras a map of steps across the tile that
the tile's header data codes. The line decoders are bit-serial: pixport-native has them compiled
(``crx_band``, ``crx_qp``), with the same results."""

from __future__ import annotations

import os
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import _accel
from ..errors import PixportError, UnsupportedError
from . import RawData, preview
from .common import check_size, masked_black
from .tiff import Tiff, camera_metadata

CANON_UUID = bytes.fromhex("85c0b687820f11e08111f4ce462b6a48")
CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"dinf"}
MAX_TILES = 4096
MAX_ZEROS = 1 << 20     # leading zeros of a code before the data counts as damaged

# JPEG-LS run-length order (ITU-T T.87, table J): a 1 bit adds 1 << J[s] repeats
J = (0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15)
# a quantisation parameter q stands for the step QSTEP[q % 6] / 64 * 2 ** (q // 6)
QSTEP = (40, 45, 51, 57, 64, 72)


# ---------------------------------------------------------------------------------------------
# the container


def _boxes(d: bytes, start: int, end: int, clip: bool = False):
    """(type, start, body, end) for each box in d[start:end]."""
    pos = start
    while pos + 8 <= end:
        size, typ = struct.unpack_from(">I4s", d, pos)
        head = 8
        if size == 1:
            if pos + 16 > end:
                break
            size, head = struct.unpack_from(">Q", d, pos + 8)[0], 16
        elif size == 0:
            size = end - pos
        if size < head:
            raise ValueError(f"CR3 box {typ!r} has a bad size")
        if pos + size > end:
            if not clip:
                raise ValueError(f"CR3 box {typ!r} runs past its container")
            size = end - pos
        yield typ, pos, pos + head, pos + size
        pos += size


@dataclass
class Track:
    entry: bytes = b""          # sample entry type: CRAW, CTMD, ...
    entry_at: tuple = (0, 0)    # its body
    offset: int = -1            # first sample
    size: int = 0


@dataclass
class Container:
    cmt: dict = field(default_factory=dict)      # "CMT1".. -> start of its TIFF
    thumb: tuple = (0, 0)                         # the THMB box's JPEG (start, end)
    tracks: list = field(default_factory=list)


def _track(d: bytes, body: int, end: int) -> Track:
    t = Track()

    def walk(s, e, depth):
        for typ, pos, b, stop in _boxes(d, s, e):
            if typ in CONTAINERS and depth < 6:
                walk(b, stop, depth + 1)
            elif typ == b"stsd" and stop - b >= 16:
                for etyp, _p, eb, estop in _boxes(d, b + 8, stop):
                    t.entry, t.entry_at = etyp, (eb, estop)
                    break
            elif typ == b"stsz" and stop - b >= 12:
                size, count = struct.unpack_from(">II", d, b + 4)
                if size == 0 and count and stop - b >= 16:
                    size = struct.unpack_from(">I", d, b + 12)[0]
                t.size = size
            elif typ == b"co64" and stop - b >= 16:
                t.offset = struct.unpack_from(">Q", d, b + 8)[0]
            elif typ == b"stco" and stop - b >= 12:
                t.offset = struct.unpack_from(">I", d, b + 8)[0]

    walk(body, end, 0)
    return t


def parse_container(d: bytes) -> Container:
    if d[4:12] != b"ftypcrx ":
        raise PixportError("This isn't a Canon CR3 file")
    c = Container()
    for typ, _pos, body, end in _boxes(d, 0, len(d), clip=True):
        if typ != b"moov":
            continue
        for t2, _p2, b2, e2 in _boxes(d, body, end):
            if t2 == b"uuid" and d[b2:b2 + 16] == CANON_UUID:
                for t3, _p3, b3, _e3 in _boxes(d, b2 + 16, e2):
                    name = t3.decode("latin-1")
                    if name in ("CMT1", "CMT2", "CMT3", "CMT4"):
                        c.cmt[name] = b3
                    elif name == "THMB":
                        j = d.find(b"\xff\xd8", b3, min(_e3, b3 + 32))
                        if j > 0:
                            c.thumb = (j, _e3)
            elif t2 == b"trak":
                c.tracks.append(_track(d, b2, e2))
    if not c.tracks:
        raise PixportError("This CR3 has no tracks")
    return c


# ---------------------------------------------------------------------------------------------
# CRX headers


@dataclass
class Band:
    size: int = 0             # bytes the subband takes
    data_size: int = 0        # of which coded data (0: the subband is all zero)
    q: int = 0                # quantisation parameter (old headers)
    q_base: int = 0           # new headers: step = q_base + (map * q_mult >> 3)
    q_mult: int = 0
    offset: int = 0


@dataclass
class Plane:
    size: int = 0
    predicted: bool = False   # the first subband uses the median predictor
    bands: list = field(default_factory=list)


@dataclass
class Tile:
    size: int = 0
    qp_size: int = 0          # the quantisation map's coded size (new headers)
    qp_pad: int = 0
    offset: int = 0
    planes: list = field(default_factory=list)


@dataclass
class Header:
    """The CMP1 box: picture size, tiling and coding."""

    version: int
    width: int
    height: int
    tile_w: int
    tile_h: int
    bits: int
    planes: int
    cfa: int
    enc: int
    levels: int
    hdr_size: int

    @classmethod
    def parse(cls, b: bytes) -> Header:
        if len(b) < 32:
            raise PixportError("This CR3's CMP1 box is too short")
        h = cls(version=struct.unpack_from(">H", b, 4)[0], width=struct.unpack_from(">I", b, 8)[0],
                height=struct.unpack_from(">I", b, 12)[0], tile_w=struct.unpack_from(">I", b, 16)[0],
                tile_h=struct.unpack_from(">I", b, 20)[0], bits=b[24], planes=b[25] >> 4, cfa=b[25] & 15,
                enc=b[26] >> 4, levels=b[26] & 15, hdr_size=struct.unpack_from(">I", b, 28)[0])
        if h.version not in (0x100, 0x200) or not h.hdr_size:
            raise UnsupportedError(f"CRX version {h.version:#x} isn't supported")
        if h.planes == 1:
            raise UnsupportedError("CRX single-plane pictures aren't supported")
        if h.enc != 0:
            raise UnsupportedError(f"CRX coding type {h.enc} (newer HDR/lossless variants) isn't supported yet")
        if (h.planes != 4 or h.width & 1 or h.height & 1 or h.tile_w & 1 or h.tile_h & 1 or h.cfa > 3
                or not 9 <= h.bits <= 14 or h.levels > 3 or not h.tile_w or not h.tile_h
                or h.tile_w > h.width or h.tile_h > h.height):
            raise PixportError("This CR3's CRX header is damaged")
        check_size(h.width, h.height)
        if -(-h.width // h.tile_w) * -(-h.height // h.tile_h) > MAX_TILES:
            raise PixportError("This CR3 has too many tiles")
        return h


def parse_tiles(hdr: Header, d: bytes, start: int, size: int) -> list[list[Tile]]:
    """The tile, plane and subband headers at the start of the sensor data, with where each
    subband's bitstream is."""
    nbands = 3 * hdr.levels + 1
    rows, cols = -(-hdr.height // hdr.tile_h), -(-hdr.width // hdr.tile_w)
    hs = hdr.hdr_size
    if hs > size:
        raise ValueError("CRX headers run past the sensor data")
    h = d[start:start + hs]
    pos, data_at = 0, start + hs
    out = []

    def word(at, n=2):
        if at + n > len(h):
            raise ValueError("CRX headers are cut short")
        return int.from_bytes(h[at:at + n], "big")

    for r in range(rows):
        line = []
        for c in range(cols):
            sig, hsz = word(pos), word(pos + 2)
            if (sig, hsz) not in ((0xFF01, 8), (0xFF11, 16)):
                raise ValueError("CRX tile header is damaged")
            tile = Tile(size=word(pos + 4, 4), offset=data_at)
            if word(pos + 8) != r * cols + c:
                raise ValueError("CRX tiles are out of order")
            if sig == 0xFF11:
                tile.qp_size, tile.qp_pad = word(pos + 12, 4), word(pos + 16)
            pos += 4 + hsz
            at = data_at + tile.qp_size + tile.qp_pad
            for p in range(hdr.planes):
                sig, hsz = word(pos), word(pos + 2)
                if (sig, hsz) not in ((0xFF02, 8), (0xFF12, 8)) or h[pos + 8] >> 4 != p:
                    raise ValueError("CRX plane header is damaged")
                flags = h[pos + 8]
                if (flags >> 1) & 3:
                    raise UnsupportedError("CRX rounded-bits coding isn't supported")
                plane = Plane(size=word(pos + 4, 4), predicted=bool(flags & 8))
                pos += 4 + hsz
                band_at = at
                for b in range(nbands):
                    sig, hsz = word(pos), word(pos + 2)
                    if (sig, hsz) not in ((0xFF03, 8), (0xFF13, 16)) or h[pos + 8] >> 4 != b:
                        raise ValueError("CRX subband header is damaged")
                    band = Band(size=word(pos + 4, 4), offset=band_at)
                    if sig == 0xFF03:
                        bits = word(pos + 8, 4)
                        if bits & 0x8000000:
                            raise UnsupportedError("CRX per-line quantisation isn't supported")
                        band.q = (bits >> 19) & 0xFF
                        band.data_size = band.size - (bits & 0x7FFFF)
                    else:
                        if word(pos + 8) & 0xFFF or word(pos + 18):
                            raise UnsupportedError("This CRX subband header isn't supported")
                        band.q_mult, band.q_base = word(pos + 10), word(pos + 12, 4)
                        band.data_size = band.size - word(pos + 16)
                    if band.data_size < 0 or band_at + band.size > at + plane.size:
                        raise ValueError("CRX subbands don't fit their plane")
                    band_at += band.size
                    plane.bands.append(band)
                    pos += 4 + hsz
                at += plane.size
                tile.planes.append(plane)
            if at > tile.offset + tile.size or tile.offset + tile.size > start + size:
                raise ValueError("CRX tile data runs past the sensor data")
            if sig == 0xFF13 and not tile.qp_size:
                raise UnsupportedError("CRX tiles without a quantisation map aren't supported")
            data_at = tile.offset + tile.size
            line.append(tile)
        out.append(line)
    return out


# ---------------------------------------------------------------------------------------------
# the bitstream and the line decoders (pure Python; pixport-native has them compiled)


class Bits:
    """MSB-first bit reader over d[pos:end], zeros past the end (where a code soon has too
    many leading zeros to be real). ``acc`` holds the next ``n`` bits in its low bits; anything
    above them is stale."""

    __slots__ = ("acc", "d", "end", "n", "p")

    def __init__(self, d: bytes, pos: int, end: int):
        self.d, self.p, self.end, self.acc, self.n = d, pos, end, 0, 0

    def _fill(self) -> None:
        p = self.p
        chunk = self.d[p:min(p + 8, self.end)]
        if len(chunk) < 8:
            chunk = chunk + bytes(8 - len(chunk))
        self.acc = ((self.acc & ((1 << self.n) - 1)) << 64) | int.from_bytes(chunk, "big")
        self.n += 64
        self.p = p + 8

    def bit(self) -> int:
        if not self.n:
            self._fill()
        self.n -= 1
        return (self.acc >> self.n) & 1

    def bits(self, k: int) -> int:
        while self.n < k:
            self._fill()
        self.n -= k
        return (self.acc >> self.n) & ((1 << k) - 1)

    def zeros(self) -> int:
        """Zero bits before the next 1 (which is used up)."""
        z = 0
        while True:
            acc = self.acc & ((1 << self.n) - 1)
            if acc:
                bl = acc.bit_length()
                z += self.n - bl
                self.n = bl - 1
                if z > MAX_ZEROS:
                    raise ValueError("CRX data is damaged (no end to a code)")
                return z
            z += self.n
            self.n = 0
            if z > MAX_ZEROS:
                raise ValueError("CRX data is damaged (no end to a code)")
            self._fill()


def _code(br: Bits, k: int) -> int:
    z = br.zeros()
    if z >= 41:
        return br.bits(21)
    return (z << k) | br.bits(k) if k else z


def _knext(k: int, c: int, cap: int = 15) -> int:
    k = k - (c < ((1 << k) >> 1)) + ((c >> k) > 2) + ((c >> k) > 5)
    return k if not cap or k < cap else cap


def _run(br: Bits, s: int, room: int) -> tuple[int, int]:
    """A run's length after its leading 1 bit (at most ``room``), and the new run index."""
    n = 1
    while br.bit():
        n += 1 << J[s]
        if n > room:
            n = room
            break
        if s < 31:
            s += 1
        if n == room:
            break
    if n < room:
        if J[s]:
            n += br.bits(J[s])
        if s > 0:
            s -= 1
        if n > room:
            raise ValueError("CRX run is longer than its line")
    return n, s


def _zz(c: int) -> int:
    return (c >> 1) ^ -(c & 1)


def _med(a: int, b: int, c: int) -> int:
    """The median edge detector: the median of left, top and left + top - top-left."""
    g = a + b - c
    lo, hi = (a, b) if a < b else (b, a)
    return lo if g < lo else min(g, hi)


def _band_predicted(br: Bits, w: int, h: int) -> list[list[int]]:
    """A subband whose values are predicted from their neighbours (the first of each plane)."""
    k = s = 0
    rows = []
    # top line: predicted from the left; runs of zeros
    cur = [0] * (w + 2)
    i, left = 1, w
    while left > 1:
        if cur[i - 1]:
            pred = cur[i - 1]
        else:
            if br.bit():
                n, s = _run(br, s, left)
                i += n  # the run repeats the zero on the left
                left -= n
                if left <= 0:
                    break
            pred = 0
        c = _code(br, k)
        cur[i] = pred + _zz(c)
        k = _knext(k, c)
        i += 1
        left -= 1
    if left == 1:
        c = _code(br, k)
        cur[i] = cur[i - 1] + _zz(c)
        k = _knext(k, c)
    cur[w + 1] = cur[w] + 1
    rows.append(cur)
    prev = cur
    for _ in range(1, h):
        cur = [0] * (w + 2)
        cur[0] = prev[1]
        i, left = 1, w
        while left > 1:
            a = cur[i - 1]
            if a != prev[i] or a != prev[i + 1]:
                c = _code(br, k)
                cur[i] = _med(a, prev[i], prev[i - 1]) + _zz(c)
                d = prev[i + 1] - prev[i]
                k = _knext(k, (c + 2 * (d if d >= 0 else -d)) >> 1)
                i += 1
                left -= 1
                continue
            if br.bit():
                n, s = _run(br, s, left)
                cur[i:i + n] = [a] * n
                i += n
                left -= n
            if left > 0:
                c = _code(br, k)
                cur[i] = prev[i] + _zz(c)
                if left > 1:
                    d = prev[i + 1] - prev[i]
                    c = (c + 2 * (d if d >= 0 else -d)) >> 1
                k = _knext(k, c)
                i += 1
                left -= 1
        if left == 1:
            c = _code(br, k)
            cur[i] = _med(cur[i - 1], prev[i], prev[i - 1]) + _zz(c)
            k = _knext(k, c)
        cur[w + 1] = cur[w] + 1
        rows.append(cur)
        prev = cur
    return rows


def _band_plain(br: Bits, w: int, h: int) -> list[list[int]]:
    """A subband coded value by value (wavelet detail bands): runs of zeros, and each code's
    size predicted also from the line above."""
    k = s = 0
    rows = []
    kline = [0] * (w + 1)
    # top line
    cur = [0] * (w + 2)
    i, left = 1, w
    while left > 1:
        if cur[i - 1]:
            c = _code(br, k)
            cur[i] = _zz(c)
            k = _knext(k, c)
        else:
            if br.bit():
                n, s = _run(br, s, left)
                for j in range(i - 1, i - 1 + n):
                    kline[j] = 0
                i += n
                left -= n
                if left <= 0:
                    break
            c = _code(br, k)
            cur[i] = _zz(c + 1)  # after a run of zeros the value isn't zero
            k = _knext(k, c)
        kline[i - 1] = k
        i += 1
        left -= 1
    if left == 1:
        c = _code(br, k)
        cur[i] = _zz(c)
        k = _knext(k, c)
        kline[i - 1] = k
    rows.append(cur)
    prev = cur
    for _ in range(1, h):
        cur = [0] * (w + 2)
        i = 0
        while i < w - 1:
            if prev[i + 2] or prev[i + 1] or cur[i]:
                c = _code(br, k)
                cur[i + 1] = _zz(c)
                k = _knext(k, c, 0)
                if kline[i + 1] - k > 1:
                    k += 1
                elif k > 15:
                    k = 15
            else:
                if br.bit():
                    n, s = _run(br, s, w - i)
                    for j in range(i, i + n):
                        kline[j] = 0
                    i += n
                if i >= w - 1:
                    if i == w - 1:
                        c = _code(br, k)
                        cur[i + 1] = _zz(c + 1)
                        k = _knext(k, c)
                        kline[i] = k
                    i += 1
                    continue
                c = _code(br, k)
                cur[i + 1] = _zz(c + 1)
                k = _knext(k, c, 0)
                if kline[i + 1] - k > 1:
                    k += 1
                elif k > 15:
                    k = 15
            kline[i] = k
            i += 1
        if i == w - 1:
            c = _code(br, k)
            cur[i + 1] = _zz(c)
            k = _knext(k, c)
            kline[i] = k
        rows.append(cur)
        prev = cur
    return rows


def decode_band(d: bytes, pos: int, size: int, w: int, h: int, predicted: bool) -> np.ndarray:
    """One subband's values, (h, w) int32."""
    if w <= 0 or h <= 0:
        return np.zeros((max(h, 0), max(w, 0)), np.int32)
    if not size:
        return np.zeros((h, w), np.int32)
    if _accel.has("crx_band"):
        out = _accel.native.crx_band(d, pos, size, w, h, predicted)
        return np.frombuffer(out, "<i4").reshape(h, w)
    br = Bits(d, pos, pos + size)
    rows = (_band_predicted if predicted else _band_plain)(br, w, h)
    return _int32([r[1:w + 1] for r in rows])


def _int32(rows: list) -> np.ndarray:
    a = np.array(rows, dtype=np.int64)
    if a.size and (a.min() < -(1 << 31) or a.max() >= 1 << 31):
        raise ValueError("CRX values are out of range")
    return a.astype(np.int32)


def decode_qp(d: bytes, pos: int, size: int, w: int, h: int) -> np.ndarray:
    """A tile's quantisation map, (h, w) int32: Golomb-Rice coded differences from the median
    edge detector's prediction (from the left on the first line)."""
    if _accel.has("crx_qp"):
        return np.frombuffer(_accel.native.crx_qp(d, pos, size, w, h), "<i4").reshape(h, w)
    br = Bits(d, pos, pos + size)
    k = 0

    def code() -> int:
        z = br.zeros()
        if z >= 23:
            return br.bits(8)
        return (z << k) | br.bits(k) if k else z

    out = []
    prev = None
    for _ in range(h):
        cur = [0] * (w + 2)
        if prev is None:
            for x in range(w):
                c = code()
                cur[x + 1] = cur[x] + _zz(c)
                k = _knext(k, c, 7)
        else:
            cur[0] = prev[1]
            for x in range(w):
                c = code()
                cur[x + 1] = _med(cur[x], prev[x + 1], prev[x]) + _zz(c)
                if x < w - 1:
                    dh = prev[x + 2] - prev[x + 1]
                    k = _knext(k, (c + 2 * (dh if dh >= 0 else -dh)) >> 1, 7)
                else:
                    k = _knext(k, c, 7)
        cur[w + 1] = cur[w] + 1
        out.append(cur[1:w + 1])
        prev = cur
    return _int32(out).reshape(h, w)


# ---------------------------------------------------------------------------------------------
# geometry, quantisation and the wavelet


def _lengths(t: int, lo: bool, hi: bool, levels: int) -> list[tuple[int, int, int, int, int]]:
    """Along one axis of a tile of length t, per level (finest first): (output length, low-pass
    length, high-pass length, own low, own high). Next to another tile a subband carries the
    extra coefficients the inverse transform needs there: one high-pass value before the start
    (``lo``) and what the last output needs past the end (``hi``); there Canon's encoder also
    makes the low-pass length of every level but the coarsest odd."""
    out, own = t, t
    res = []
    for level in range(levels):
        ne, no = (out + 1) // 2, out // 2
        if hi:
            nl = nh = ne + (out % 2 == 0)
            if level < levels - 1 and nl % 2 == 0:
                nl += 1
        else:
            nl, nh = ne, no
        res.append((out, nl, nh + lo, (own + 1) // 2, own // 2))
        out, own = nl, (own + 1) // 2
    return res


def _qstep(q: np.ndarray | int):
    q = np.asarray(q, dtype=np.int64)
    if np.any(q < 0):
        raise ValueError("CRX quantisation parameter is negative")
    base = np.asarray(QSTEP, np.int64)[q % 6]
    sh = q // 6
    return np.where(sh < 6, base >> np.clip(6 - sh, 0, 63), base << np.clip(sh - 6, 0, 40))


def _qp_maps(qp: np.ndarray, levels: int) -> list[np.ndarray]:
    """Steps per level, coarsest first, from the tile's map (one value per 8 x 2 of a plane)."""
    h = qp.shape[0]
    maps = []
    if levels >= 3:
        rows = np.minimum(np.arange(0, -(-h // 4) * 4).reshape(-1, 4), h - 1)
        s = qp[rows].sum(axis=1)
        maps.append(_qstep(np.where(s < 0, -((-s) >> 2), s >> 2)))
    if levels >= 2:
        rows = np.minimum(np.arange(0, -(-h // 2) * 2).reshape(-1, 2), h - 1)
        s = qp[rows].sum(axis=1)
        maps.append(_qstep(np.where(s < 0, -((-s) >> 1), s >> 1)))
    maps.append(_qstep(qp))
    return maps


def _dequant_map(coef: np.ndarray, band: Band, qmap: np.ndarray, shift: int, x0: int, y0: int, ownw: int):
    """Scale a subband by its step from the tile's map: column x of the band (past the ``x0``
    extra ones) uses map column x >> shift, clamped to the band's own width; row y uses row
    y - y0, clamped to the map."""
    h, w = coef.shape
    rows = np.clip(np.arange(h) - y0, 0, qmap.shape[0] - 1)
    last = max(0, (ownw - 1) >> shift)
    cols = np.clip((np.arange(w) - x0) >> shift, 0, min(last, qmap.shape[1] - 1))
    cols[:x0] = 0
    step = band.q_base + ((qmap[rows][:, cols] * band.q_mult) >> 3)
    return coef * np.clip(step, 1, 0x168000).astype(np.int32)


def synth(low: np.ndarray, high: np.ndarray, n: int, lo: bool, hi: bool, axis: int) -> np.ndarray:
    """The inverse 5/3 integer lifting (as JPEG 2000's reversible wavelet) along ``axis``: n
    samples from the low- and high-pass halves. Ends without a neighbouring tile are mirrored;
    ``lo``/``hi`` mean the halves carry the neighbour's coefficients instead."""
    if axis == 1:
        return synth(low.T, high.T, n, lo, hi, 0).T
    # int32 is plenty: coefficients of 14-bit data stay far inside it
    low = low.astype(np.int32, copy=False)
    high = high.astype(np.int32, copy=False)
    if n <= 1:
        return low[:1].copy()
    ne, no = (n + 1) // 2, n // 2
    hc = high[1:] if lo else high          # high-pass values from this tile's start
    before = high[:1]                      # the one before the start (mirrored: the first)
    need = ne + (1 if hi and n % 2 == 0 else 0)
    if len(low) < need or len(hc) < (need if hi else no):
        raise ValueError("CRX subbands are smaller than the picture")
    hx = hc[:need]
    if len(hx) < need:                      # mirror the end
        hx = np.concatenate([hx, hc[no - 1:no][: need - len(hx)]])
    left = np.concatenate([before, hx[:-1]])
    even = low[:need] - ((left + hx + 2) >> 2)
    nxt = even[1:no + 1] if len(even) > no else np.concatenate([even[1:], even[-1:]])
    odd = hc[:no] + ((even[:no] + nxt) >> 1)
    out = np.empty((n,) + low.shape[1:], np.int32)
    out[0::2] = even[:ne]
    out[1::2] = odd
    return out


def decode_tile(d: bytes, hdr: Header, tile: Tile, plane: Plane, tw: int, th: int, flags: tuple,
                qmaps: list | None) -> np.ndarray:
    """One plane of one tile, (th, tw) int32 values around the middle grey."""
    left_t, right_t, top_t, bottom_t = flags
    if not hdr.levels:
        b = plane.bands[0]
        return decode_band(d, b.offset, b.data_size, tw, th, plane.predicted)
    xs = _lengths(tw, left_t, right_t, hdr.levels)
    ys = _lengths(th, top_t, bottom_t, hdr.levels)
    nlev = hdr.levels

    def band(idx: int, w: int, h: int, level: int, kind: str) -> np.ndarray:
        b = plane.bands[idx]
        v = decode_band(d, b.offset, b.data_size, w, h, plane.predicted and idx == 0)
        if qmaps is None:
            st = int(_qstep(b.q))
            return v * np.int32(st) if st != 1 else v
        # level: 1 finest .. nlev coarsest; the coarsest maps come first
        shift = 3 - level
        x0 = 1 if (kind[0] == "H" and left_t) else 0
        y0 = 1 if (kind[1] == "H" and top_t) else 0
        ownw = xs[level - 1][3] if kind[0] == "L" else xs[level - 1][4]
        return _dequant_map(v, b, qmaps[nlev - level], shift, x0, y0, ownw)

    lx, ly = xs[-1][1], ys[-1][1]
    img = band(0, lx, ly, nlev, "LL")
    idx = 1
    for level in range(nlev, 0, -1):
        ox, nlx, nhx = xs[level - 1][:3]
        oy, nly, nhy = ys[level - 1][:3]
        hl = band(idx, nhx, nly, level, "HL")
        lh = band(idx + 1, nlx, nhy, level, "LH")
        hh = band(idx + 2, nhx, nhy, level, "HH")
        idx += 3
        a = synth(img, hl, ox, left_t, right_t, axis=1)
        b = synth(lh, hh, ox, left_t, right_t, axis=1)
        img = synth(a, b, oy, top_t, bottom_t, axis=0)
    return img


def decode_crx(d: bytes, hdr: Header, start: int, size: int) -> np.ndarray:
    """The mosaic, (height, width) uint16."""
    tiles = parse_tiles(hdr, d, start, size)
    pw, ph = hdr.width // 2, hdr.height // 2
    tw0, th0 = hdr.tile_w // 2, hdr.tile_h // 2
    rows, cols = len(tiles), len(tiles[0])
    jobs = []
    for r, line in enumerate(tiles):
        for c, tile in enumerate(line):
            tw = tw0 if c < cols - 1 else pw - tw0 * (cols - 1)
            th = th0 if r < rows - 1 else ph - th0 * (rows - 1)
            flags = (c > 0, c < cols - 1, r > 0, r < rows - 1)
            jobs.append((r, c, tile, tw, th, flags))
    mid, top = 1 << (hdr.bits - 1), (1 << hdr.bits) - 1
    out = np.empty((hdr.height, hdr.width), np.uint16)
    spots = SPOTS[hdr.cfa]

    def qmaps_of(tile: Tile, tw: int, th: int):
        if not tile.qp_size:
            return None
        if not hdr.levels:
            raise UnsupportedError("CRX quantisation maps without a wavelet aren't supported")
        return _qp_maps(decode_qp(d, tile.offset, tile.qp_size, -(-tw // 8), -(-th // 2)) + 4, hdr.levels)

    def run_job(job, p):
        r, c, tile, tw, th, flags, qmaps = job
        v = decode_tile(d, hdr, tile, tile.planes[p], tw, th, flags, qmaps)
        y, x = spots[p]
        y0, x0 = 2 * r * th0 + y, 2 * c * tw0 + x
        out[y0:y0 + 2 * th:2, x0:x0 + 2 * tw:2] = np.clip(v + mid, 0, top)

    jobs = [j + (qmaps_of(j[2], j[3], j[4]),) for j in jobs]
    work = [(j, p) for j in jobs for p in range(hdr.planes)]
    if _accel.has("crx_band") and len(work) > 1:
        # the compiled decoder lets go of the GIL: tiles and planes decode side by side
        with ThreadPoolExecutor(max_workers=min(len(work), os.cpu_count() or 1, 8)) as pool:
            list(pool.map(lambda w: run_job(*w), work))
    else:
        for w in work:
            run_job(*w)
    return out


# where each plane (R, G1, G2, B) sits in the 2 x 2 repeat, per CMP1 layout, and the pattern
SPOTS = {0: ((0, 0), (0, 1), (1, 0), (1, 1)), 1: ((0, 1), (0, 0), (1, 1), (1, 0)),
         2: ((1, 0), (1, 1), (0, 0), (0, 1)), 3: ((1, 1), (1, 0), (0, 1), (0, 0))}
PATTERNS = {0: [[0, 1], [1, 2]], 1: [[1, 0], [2, 1]], 2: [[1, 2], [0, 1]], 3: [[2, 1], [1, 0]]}


# ---------------------------------------------------------------------------------------------
# metadata and decode()


def _ctmd_notes(d: bytes, track: Track) -> list:
    """Maker-note IFDs from the timed metadata track's TIFF records (types 7, 8 and 9)."""
    out = []
    pos, end = track.offset, track.offset + track.size
    if pos < 0 or end > len(d):
        return out
    while pos + 8 <= end and len(out) < 16:
        n, typ = struct.unpack_from("<IH", d, pos)
        if n < 8 or pos + n > end:
            break
        if typ in (7, 8, 9):
            for at in range(pos + 8, min(pos + 40, pos + n - 8)):
                if d[at:at + 4] in (b"II*\0", b"MM\0*"):
                    try:
                        t = Tiff(d[at:pos + n])
                    except PixportError:
                        break
                    if t.ifds:
                        out.append(t.ifds[0])
                    break
        pos += n
    return out


def _colour_data(notes: list) -> np.ndarray | None:
    for ifd in notes:
        e = ifd.get(0x4001)
        if e is not None and e.type == 3 and e.count >= 0x47:
            return e.values().astype(np.int64)
    return None


def decode(path: Path, fmt: str = "cr3") -> RawData:
    d = path.read_bytes()
    c = parse_container(d)
    best = None
    for t in c.tracks:
        if t.entry != b"CRAW":
            continue
        b0, b1 = t.entry_at
        at = d.find(b"CMP1", b0, b1)
        if at < 4:
            continue
        (blen,) = struct.unpack_from(">I", d, at - 4)
        hdr_bytes = d[at + 4:min(at - 4 + blen, b1)]
        if len(hdr_bytes) < 32:
            continue
        w, h = struct.unpack_from(">II", hdr_bytes, 8)
        if best is None or w * h > best[0]:
            best = (w * h, t, hdr_bytes)
    if best is None:
        raise PixportError("This CR3 has no sensor data")
    _, track, hdr_bytes = best
    hdr = Header.parse(hdr_bytes)
    if track.offset < 0 or track.size <= 0 or track.offset + track.size > len(d):
        raise PixportError("This CR3's sensor data runs past the end of the file")
    cfa = decode_crx(d, hdr, track.offset, track.size)

    tiffs = {}
    for name, at in c.cmt.items():
        try:
            tiffs[name] = Tiff(d, at)
        except PixportError:
            pass
    ifd0 = tiffs["CMT1"].ifds[0] if "CMT1" in tiffs and tiffs["CMT1"].ifds else None
    md = _metadata(tiffs, ifd0)
    mn = tiffs["CMT3"].ifds[0] if "CMT3" in tiffs and tiffs["CMT3"].ifds else None
    notes = _ctmd_notes(d, next((t for t in c.tracks if t.entry == b"CTMD"), Track()))
    cd = _colour_data(notes)

    pattern = np.array(PATTERNS[hdr.cfa], dtype=np.uint8)
    raw = RawData(cfa=cfa, pattern=pattern, metadata=md)
    raw.orientation = ifd0.int(274, 1) if ifd0 is not None else 1
    if raw.orientation not in range(1, 9):
        raw.orientation = 1
    raw.white = float((1 << hdr.bits) - 1)
    wb, black = _levels(cd)
    if wb is not None:
        raw.wb = wb
    info = mn.ints(0xE0) if mn is not None else None
    if info and len(info) >= 9 and info[1] == hdr.width and info[2] == hdr.height:
        left, top, right, bottom = info[5], info[6], info[7], info[8]
        if 0 <= left < right < hdr.width and 0 <= top < bottom < hdr.height:
            raw.crop = (top, left, bottom - top + 1, right - left + 1)
    if raw.crop is None:
        raw.crop = (0, 0, hdr.height, hdr.width)
    if black is not None:
        # the colour data's levels are R, G, G, B: put each where its plane sits
        spots = SPOTS[hdr.cfa]
        raw.black = np.zeros((2, 2))
        for p, (y, x) in enumerate(spots):
            raw.black[y, x] = black[p]
    else:
        raw.black = masked_black(cfa, pattern, left=raw.crop[1], top=raw.crop[0])
    # CR3 has no colour matrix: fit one from the camera's thumbnail (the THMB box)
    preview.attach(raw, d[c.thumb[0]:c.thumb[1]] if c.thumb[1] > c.thumb[0] else None)
    md["bits"] = hdr.bits
    return raw


def _levels(cd: np.ndarray | None) -> tuple:
    """(as-shot white balance, black per R, G, G, B) from the colour data (maker note 0x4001),
    whose layout goes by its version (the first value); either is None when unknown."""
    if cd is None or len(cd) < 0x60:
        return None, None
    ver = int(cd[0])
    at_wb, at_black = {16: (0x47, 0x149), 17: (0x47, 0x149), 18: (0x47, 0x149), 19: (0x47, 0x149),
                       32: (0x55, 0x157), 33: (0x55, 0x157), 34: (0x55, 0x157)}.get(ver, (None, None))
    if at_wb is None:
        # another layout: the as-shot levels are the first R, G, G, B with equal greens near 1024
        at_wb = next((i for i in range(0x3F, 0x60) if cd[i + 1] == cd[i + 2] and 512 <= cd[i + 1] <= 2048
                      and 64 <= cd[i] <= 16384 and 64 <= cd[i + 3] <= 16384), None)
    wb = None
    if at_wb is not None:
        r, g1, g2, b = (int(x) for x in cd[at_wb:at_wb + 4])
        g = (g1 + g2) / 2
        if min(r, g, b) > 0:
            wb = (r / g, 1.0, b / g)
    black = None
    if at_black is not None and at_black + 4 <= len(cd):
        lv = [int(x) for x in cd[at_black:at_black + 4]]
        if all(0 < v < 4096 for v in lv):
            black = lv
    return wb, black


def _metadata(tiffs: dict, ifd0) -> dict:
    """camera_metadata from IFD0 (CMT1) with the EXIF (CMT2) and GPS (CMT4) IFDs attached."""
    if ifd0 is None:
        return {}
    t = tiffs["CMT1"]
    if "CMT2" in tiffs and tiffs["CMT2"].ifds:
        t._exif = tiffs["CMT2"].ifds[0]
    if "CMT4" in tiffs and tiffs["CMT4"].ifds:
        t._gps = tiffs["CMT4"].ifds[0]
    return camera_metadata(t, ifd0)
