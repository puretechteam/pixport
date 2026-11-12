"""Fujifilm RAF: a big-endian header pointing at an embedded JPEG (whose EXIF has the camera
settings and orientation), a directory of RAF tags (sensor size, the crop the camera shows, the
X-Trans layout) and the sensor data, which on cameras since 2011 starts with a small TIFF whose
IFD 0xF000 gives its size, bit depth, black levels and white balance.

The sensor data is stored plain (16-bit words, or packed 12 or 14 bits), or in Fujifilm's
lossless compression: vertical strips 768 photosites wide, each coded on its own six rows at a
time. Each colour's values on those rows are gathered into lines; every value is predicted from
the lines above (and, for every second one, its decoded neighbours on the line), and the error
is coded with an adaptive Golomb-Rice code whose size follows one of 41 contexts of the local
gradients, as in JPEG-LS. The decoder is bit-serial: pixport-native has it compiled
(``fuji_strip``), with the same results."""

from __future__ import annotations

import os
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .. import _accel
from ..errors import PixportError, UnsupportedError
from . import RawData, preview
from .common import check_size, need, plateau_white
from .tiff import Tiff, camera_metadata

MAGIC = b"FUJIFILMCCD-RAW "
BLOCK = 768                 # photosites across one compressed strip
MAX_ZEROS = 1 << 20         # leading zeros of a code before the data counts as damaged

# line buffers of a strip: two kept from the six rows before, then the current ones
R0, R1, R2, R3, R4 = range(5)
G0, G1, G2, G3, G4, G5, G6, G7 = range(5, 13)
B0, B1, B2, B3, B4 = range(13, 18)
CURRENT = (R2, R3, R4, G2, G3, G4, G5, G6, G7, B2, B3, B4)


# ---------------------------------------------------------------------------------------------
# the container


def _tags(d: bytes, off: int, size: int) -> dict[int, bytes]:
    """The RAF directory: a count, then (tag, size, bytes) records."""
    out: dict[int, bytes] = {}
    if off < 0 or off + 4 > len(d):
        return out
    end = min(len(d), off + size)
    n = struct.unpack_from(">I", d, off)[0]
    pos = off + 4
    for _ in range(min(n, 1000)):
        if pos + 4 > end:
            break
        tag, sz = struct.unpack_from(">HH", d, pos)
        if pos + 4 + sz > end:
            break
        out[tag] = d[pos + 4:pos + 4 + sz]
        pos += 4 + sz
    return out


def _exif(d: bytes, off: int, size: int) -> Tiff | None:
    """The EXIF TIFF inside the embedded JPEG's APP1 segment."""
    end = min(len(d), off + size)
    if off < 0 or d[off:off + 2] != b"\xff\xd8":
        return None
    pos = off + 2
    for _ in range(64):
        if pos + 4 > end or d[pos] != 0xFF:
            return None
        marker, length = d[pos + 1], struct.unpack_from(">H", d, pos + 2)[0]
        if marker == 0xE1 and d[pos + 4:pos + 10] == b"Exif\0\0":
            try:
                return Tiff(d[:min(end, pos + 2 + length)], pos + 10)
            except PixportError:
                return None
        if marker in (0xDA, 0xD9):
            return None
        pos += 2 + length
    return None


def _thumbnail(d: bytes, tiff: Tiff | None) -> bytes | None:
    """The small JPEG in the EXIF's IFD1."""
    if tiff is None or len(tiff.ifds) < 2:
        return None
    off, size = tiff.ifds[1].offset(513), tiff.ifds[1].int(514)
    if off is None or not size or off < 0 or off + size > len(d):
        return None
    return d[off:off + size]


def xtrans_layout(tag: bytes) -> np.ndarray:
    """The 6 x 6 X-Trans layout from RAF tag 0x131 (stored last photosite first)."""
    if len(tag) < 36:
        raise PixportError("This RAF's X-Trans layout is damaged")
    a = np.frombuffer(tag[:36], np.uint8)[::-1].reshape(6, 6) & 3
    if a.max() > 2:
        raise PixportError("This RAF's X-Trans layout is damaged")
    return a.astype(np.uint8)


# ---------------------------------------------------------------------------------------------
# the compressed format


class Header:
    """The 16 bytes before compressed sensor data."""

    def __init__(self, b: bytes):
        if len(b) < 16 or b[:2] != b"IS":
            raise PixportError("This RAF's compressed data has no header")
        self.lossless, self.kind, self.bits = b[2], b[3], b[4]
        self.height, self.rounded, self.width, self.block = struct.unpack_from(">HHHH", b, 5)
        self.blocks, self.lines = b[13], struct.unpack_from(">H", b, 14)[0]
        if self.lossless != 1:
            raise UnsupportedError("Fujifilm's lossy compressed RAF isn't supported yet")
        if (self.kind not in (0, 16) or self.bits not in (12, 14, 16) or self.block != BLOCK
                or not 6 <= self.height <= 0x4002 or self.height % 6 or self.lines != self.height // 6
                or not BLOCK <= self.width <= 0x4200 or self.width % 24 or self.rounded % BLOCK
                or not 0 < self.blocks <= 16 or self.blocks != self.rounded // BLOCK
                or not 0 <= self.rounded - self.width < BLOCK):
            raise PixportError("This RAF's compressed data header is damaged")
        self.xtrans = self.kind == 16
        self.line_width = BLOCK * 2 // 3 if self.xtrans else BLOCK // 2


class Strip:
    """MSB-first bits of one strip, zeros past its end."""

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

    def bits(self, k: int) -> int:
        while self.n < k:
            self._fill()
        self.n -= k
        return (self.acc >> self.n) & ((1 << k) - 1)

    def zeros(self) -> int:
        z = 0
        while True:
            acc = self.acc & ((1 << self.n) - 1)
            if acc:
                bl = acc.bit_length()
                z += self.n - bl
                self.n = bl - 1
                if z > MAX_ZEROS:
                    raise ValueError("RAF compressed data is damaged (no end to a code)")
                return z
            z += self.n
            self.n = 0
            if z > MAX_ZEROS:
                raise ValueError("RAF compressed data is damaged (no end to a code)")
            self._fill()


def _quantiser(bits: int) -> list[int]:
    """Gradient buckets -4..4 (JPEG-LS thresholds 18, 67, 276), indexed by difference + top."""
    top = (1 << bits) - 1
    v = np.arange(-top, top + 1)
    a = np.abs(v)
    q = np.where(a == 0, 0, np.where(a < 18, 1, np.where(a < 67, 2, np.where(a < 276, 3, 4))))
    return (np.sign(v) * q).tolist()


def decode_strip(d: bytes, pos: int, size: int, hdr: Header) -> np.ndarray:
    """One strip's lines: (lines, 12, line_width) uint16, the current lines of every six rows in
    CURRENT order."""
    lw = hdr.line_width
    if _accel.has("fuji_strip"):
        out = _accel.native.fuji_strip(d, pos, size, lw, hdr.lines, hdr.bits, hdr.xtrans)
        return np.frombuffer(out, "<u2").reshape(hdr.lines, 12, lw)
    bits = hdr.bits
    top = (1 << bits) - 1
    total = 1 << bits
    limit = 4 * bits - bits - 1          # codes with at least this many zeros are sent whole
    qt = _quantiser(bits)
    br = Strip(d, pos, pos + size)
    lines = [[0] * (lw + 2) for _ in range(18)]
    first = max(2, (total + 32) >> 6)
    even = [[[first, 1] for _ in range(41)] for _ in range(3)]
    odd = [[[first, 1] for _ in range(41)] for _ in range(3)]
    out = np.empty((hdr.lines, 12, lw), np.uint16)

    def code(g: list) -> int:
        z = br.zeros()
        if z < limit:
            v1, v2 = g
            k = 0
            if v2 < v1:
                while k <= 14:
                    k += 1
                    if (v2 << k) >= v1:
                        break
            c = (z << k) | br.bits(k) if k else z
        else:
            c = br.bits(bits) + 1
        if c >= total:
            raise ValueError("RAF compressed data is damaged")
        c = -1 - (c >> 1) if c & 1 else c >> 1
        g[0] += -c if c < 0 else c
        if g[1] == 64:
            g[0] >>= 1
            g[1] >>= 1
        g[1] += 1
        return c

    def store(v: int) -> int:
        if v < 0:
            v += total
        elif v > top:
            v -= total
        return min(v, top) if v >= 0 else 0

    def even_sample(li: int, p: int, grads: list) -> None:
        cur, prev, prev2 = lines[li], lines[li - 1], lines[li - 2]
        rb, rc, rd, rf = prev[p + 1], prev[p], prev[p + 2], prev2[p + 1]
        grad = qt[rb - rf + top] * 9 + qt[rc - rb + top]
        dc, df, dd = abs(rc - rb), abs(rf - rb), abs(rd - rb)
        if dc > df and dc > dd:
            v = rf + rd + 2 * rb
        elif dd > dc and dd > df:
            v = rf + rc + 2 * rb
        else:
            v = rd + rc + 2 * rb
        c = code(grads[abs(grad)])
        v >>= 2
        cur[p + 1] = store(v - c if grad < 0 else v + c)

    def even_interp(li: int, p: int) -> None:
        cur, prev, prev2 = lines[li], lines[li - 1], lines[li - 2]
        rb, rc, rd, rf = prev[p + 1], prev[p], prev[p + 2], prev2[p + 1]
        dc, df, dd = abs(rc - rb), abs(rf - rb), abs(rd - rb)
        if dc > df and dc > dd:
            cur[p + 1] = (rf + rd + 2 * rb) >> 2
        elif dd > dc and dd > df:
            cur[p + 1] = (rf + rc + 2 * rb) >> 2
        else:
            cur[p + 1] = (rd + rc + 2 * rb) >> 2

    def odd_sample(li: int, p: int, grads: list) -> None:
        cur, prev = lines[li], lines[li - 1]
        ra, rb, rc, rd, rg = cur[p], prev[p + 1], prev[p], prev[p + 2], cur[p + 2]
        grad = qt[rb - rc + top] * 9 + qt[rc - ra + top]
        if (rb > rc and rb > rd) or (rb < rc and rb < rd):
            v = (rg + ra + 2 * rb) >> 2
        else:
            v = (ra + rg) >> 1
        c = code(grads[abs(grad)])
        cur[p + 1] = store(v - c if grad < 0 else v + c)

    def extend(a: int, b: int) -> None:
        for i in range(a, b + 1):
            lines[i][0] = lines[i - 1][1]
            lines[i][lw + 1] = lines[i - 1][lw]

    # each pass: (first line, second line, grads set, how the first line's even values are made,
    # how the second line's are made): "d" decoded, "i" interpolated (the colour has no
    # photosite there), "i2" interpolated where pos % 4 == 2, "d0" decoded where pos % 4 != 0.
    # The X-Trans layout puts four greens on rows 0, 1, 3 and 4 of every six and two on the others
    if hdr.xtrans:
        passes = ((R2, G2, 0, "i", "d"), (G3, B2, 1, "d", "i"), (R3, G4, 2, "d0", "i"),
                  (G5, B3, 0, "d", "i2"), (R4, G6, 1, "i2", "d"), (G7, B4, 2, "i", "d0"))
    else:
        passes = ((R2, G2, 0, "d", "d"), (G3, B2, 1, "d", "d"), (R3, G4, 2, "d", "d"),
                  (G5, B3, 0, "d", "d"), (R4, G6, 1, "d", "d"), (G7, B4, 2, "d", "d"))
    ranges = {R2: (R2, R4), R3: (R2, R4), R4: (R2, R4), B2: (B2, B4), B3: (B2, B4), B4: (B2, B4)}

    def even_one(li: int, p: int, how: str, grads: list) -> None:
        if how == "d" or (how == "d0" and p & 3) or (how == "i2" and (p & 3) != 2):
            even_sample(li, p, grads)
        else:
            even_interp(li, p)

    for row in range(hdr.lines):
        for a, b, gs, how_a, how_b in passes:
            ge, go = even[gs], odd[gs]
            pe, po = 0, 1
            while pe < lw or po < lw:
                if pe < lw:
                    even_one(a, pe, how_a, ge)
                    even_one(b, pe, how_b, ge)
                    pe += 2
                if pe > 8:
                    odd_sample(a, po, go)
                    odd_sample(b, po, go)
                    po += 2
            for li in (a, b):
                extend(*ranges.get(li, (G2, G7)))
        out[row] = [lines[i][1:lw + 1] for i in CURRENT]
        for dst, src in ((R0, R3), (R1, R4), (G0, G6), (G1, G7), (B0, B3), (B1, B4)):
            lines[dst][:] = lines[src]
        for a, n in ((R2, 3), (G2, 6), (B2, 3)):
            for i in range(a, a + n):
                lines[i][:] = [0] * (lw + 2)
            lines[a][0] = lines[a - 1][1]
            lines[a][lw + 1] = lines[a - 1][lw]
    return out


def _gather(xtrans: bool, pattern: np.ndarray, lw: int) -> tuple[np.ndarray, np.ndarray]:
    """Where each photosite of six rows of a strip comes from: (index into CURRENT, position),
    each (6, BLOCK)."""
    p = np.arange(BLOCK)
    src = np.zeros((6, BLOCK), np.intp)
    pos = np.zeros((6, BLOCK), np.intp)
    for r in range(6):
        colour = pattern[r % pattern.shape[0], p % pattern.shape[1]]
        line = np.where(colour == 0, CURRENT.index(R2 + r // 2),
                        np.where(colour == 2, CURRENT.index(B2 + r // 2), CURRENT.index(G2 + r)))
        src[r] = line
        if xtrans:
            pos[r] = (((p * 2 // 3) & ~1) | ((p % 3) & 1)) + ((p % 3) >> 1)
        else:
            pos[r] = p >> 1
    return src, np.minimum(pos, lw - 1)


def decode_compressed(d: bytes, start: int, size: int, pattern: np.ndarray) -> np.ndarray:
    need(d, start, 16)
    hdr = Header(d[start:start + 16])
    check_size(hdr.width, hdr.height)
    n = hdr.blocks
    table = start + 16
    need(d, table, 4 * n)
    sizes = struct.unpack_from(f">{n}I", d, table)
    pos = table + 4 * n
    if (4 * n) & 0xC:
        pos += 0x10 - ((4 * n) & 0xC)
    offsets = []
    for s in sizes:
        if pos + s > len(d):
            raise PixportError("This RAF's compressed data runs past the end of the file")
        offsets.append((pos, s))
        pos += s
    out = np.empty((hdr.height, hdr.width), np.uint16)
    src, at = _gather(hdr.xtrans, pattern, hdr.line_width)

    def one(i: int) -> None:
        off, s = offsets[i]
        strip = decode_strip(d, off, s, hdr)                    # (lines, 12, lw)
        w = min(BLOCK, hdr.width - i * BLOCK)
        # rows of six: photosite (r, x) is strip[line, src[r, x], at[r, x]]
        block = strip[:, src[:, :w], at[:, :w]]                 # (lines, 6, w)
        out[:, i * BLOCK:i * BLOCK + w] = block.reshape(hdr.lines * 6, w)

    if _accel.has("fuji_strip") and n > 1:
        with ThreadPoolExecutor(max_workers=min(n, os.cpu_count() or 1, 8)) as pool:
            list(pool.map(one, range(n)))
    else:
        for i in range(n):
            one(i)
    return out


# ---------------------------------------------------------------------------------------------
# decode()


def _plain(d: bytes, off: int, size: int, w: int, h: int) -> np.ndarray:
    """Uncompressed sensor data: 16-bit little-endian words, or 12 bits packed low bits first,
    or 14 bits packed high bits first in little-endian 32-bit words."""
    n = w * h
    if size >= n * 2:
        need(d, off, n * 2)
        return np.frombuffer(d, "<u2", n, off).reshape(h, w).copy()
    if size == n * 3 // 2 and w % 2 == 0:
        need(d, off, size)
        t = np.frombuffer(d, np.uint8, size, off).reshape(-1, 3).astype(np.uint16)
        out = np.empty((t.shape[0], 2), np.uint16)
        out[:, 0] = t[:, 0] | ((t[:, 1] & 15) << 8)
        out[:, 1] = (t[:, 1] >> 4) | (t[:, 2] << 4)
        return out.reshape(h, w)
    if size == n * 7 // 4 and (w * 7) % 16 == 0:
        need(d, off, size)
        # each 32-bit word's bytes reversed gives an MSB-first stream: 7 bytes, 4 values
        b = np.frombuffer(d, np.uint8, size, off).reshape(-1, 4)[:, ::-1].reshape(-1, 7).astype(np.uint16)
        out = np.empty((b.shape[0], 4), np.uint16)
        out[:, 0] = (b[:, 0] << 6) | (b[:, 1] >> 2)
        out[:, 1] = ((b[:, 1] & 3) << 12) | (b[:, 2] << 4) | (b[:, 3] >> 4)
        out[:, 2] = ((b[:, 3] & 15) << 10) | (b[:, 4] << 2) | (b[:, 5] >> 6)
        out[:, 3] = ((b[:, 5] & 63) << 8) | b[:, 6]
        return out.reshape(h, w)
    raise UnsupportedError("This RAF's sensor data packing isn't supported")


def decode(path: Path, fmt: str = "raf") -> RawData:
    d = path.read_bytes()
    if d[:16] != MAGIC or len(d) < 108:
        raise PixportError("This isn't a Fujifilm RAF file")
    jpeg_off, jpeg_len, meta_off, meta_len, cfa_off, cfa_len = struct.unpack_from(">6I", d, 84)
    tags = _tags(d, meta_off, meta_len)
    if 0x130 in tags and len(tags[0x130]) >= 2 and not tags[0x130][1] & 8:
        raise UnsupportedError("Fujifilm's 45-degree SuperCCD sensor layout isn't supported")
    pattern = xtrans_layout(tags[0x131]) if 0x131 in tags else np.array([[0, 1], [1, 2]], np.uint8)

    black: float | np.ndarray = 0.0
    wb = None
    need(d, cfa_off, 8, "sensor data")
    if d[cfa_off:cfa_off + 4] in (b"II*\0", b"MM\0*"):
        t = Tiff(d, cfa_off)
        sub = None
        for ifd in t.ifds:
            for s in t.children(ifd, 0xF000):
                sub = s
        if sub is None:
            raise PixportError("This RAF's sensor data directory is missing")
        w, h, bits = sub.int(0xF001, 0), sub.int(0xF002, 0), sub.int(0xF003, 14)
        off, size = sub.int(0xF007), sub.int(0xF008)
        if off is None or size is None:
            raise PixportError("This RAF's sensor data is missing")
        off += cfa_off
        lv = sub.ints(0xF00A)
        if lv:
            ph, pw = pattern.shape
            if len(lv) >= ph * pw:
                black = np.array(lv[:ph * pw], np.float64).reshape(ph, pw)
            elif len(lv) == 4 and pattern.shape == (6, 6):
                black = float(np.mean(lv))
            else:
                black = float(lv[0])
        g = sub.ints(0xF00E) or sub.ints(0xF00D)
        if g and len(g) >= 3 and min(g[:3]) > 0:
            wb = (g[1] / g[0], 1.0, g[2] / g[0])
    else:
        # cameras before 2011: the sensor data follows directly, its size in tag 0x100
        size_tag = tags.get(0x100, b"")
        if len(size_tag) < 4:
            raise PixportError("This RAF's sensor size is missing")
        h, w = struct.unpack(">HH", size_tag[:4])
        bits, off, size = 14, cfa_off, cfa_len
    check_size(w, h)
    if not 8 <= bits <= 16:
        raise PixportError("This RAF's bit depth is damaged")
    need(d, off, min(size, 16))
    if size * 8 < w * h * bits and d[off:off + 2] == b"IS":
        cfa = decode_compressed(d, off, size, pattern)
    else:
        cfa = _plain(d, off, size, w, h)
    h, w = cfa.shape

    if wb is None and 0x2FF0 in tags and len(tags[0x2FF0]) >= 8:
        g1, r, _g2, b = struct.unpack(">4H", tags[0x2FF0][:8])
        if min(g1, r, b) > 0:
            wb = (r / g1, 1.0, b / g1)
    tiff = _exif(d, jpeg_off, jpeg_len)
    md = camera_metadata(tiff) if tiff is not None else {}
    orient = tiff.ifds[0].int(274, 1) if tiff is not None and tiff.ifds else 1
    raw = RawData(cfa=cfa, pattern=pattern, black=black, white=float((1 << bits) - 1), metadata=md)
    raw.wb = wb or (1.0, 1.0, 1.0)
    raw.orientation = orient if orient in range(1, 9) else 1
    tl, sz = tags.get(0x110, b""), tags.get(0x111, b"")
    if len(tl) >= 4 and len(sz) >= 4:
        top, left = struct.unpack(">HH", tl[:4])
        ch, cw = struct.unpack(">HH", sz[:4])
        if ch and cw and top + ch <= h and left + cw <= w:
            raw.crop = (top, left, ch, cw)
    if raw.crop is None:
        raw.crop = (0, 0, h, w)
    raw.white = plateau_white(raw)
    # RAF has no colour matrix: fit one from the camera's thumbnail (the EXIF's IFD1)
    preview.attach(raw, _thumbnail(d, tiff))
    md["bits"] = bits
    return raw
