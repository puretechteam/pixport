"""Panasonic RW2: a TIFF-like file (magic 0x55) whose IFD0 carries Panasonic's own tags: sensor
size (0x02, 0x03), the camera's crop as borders (0x04-0x07), CFA layout (0x09), bit depth (0x0A),
black levels (0x1C-0x1E), white balance (0x24-0x26), the raw format (0x2D) and where the data is
(0x118).

Raw formats:
- 4 (most cameras to 2017): blocks of 0x4000 bytes, rotated by 0x2008, read 16 bytes at a time
  from the top bit down; each 14 photosites start fresh, with a 2-bit shift every third one and
  8-bit values that are either a new start or a scaled difference.
- 5 (G9, GH5...): the same rotated blocks, then 16-byte groups of ten 12-bit or nine 14-bit
  values, little-endian packed.
- 6 (S1...): 16-byte blocks of eleven photosites, 14-bit starts then 10-bit scaled values."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import _accel
from ..errors import PixportError, UnsupportedError
from . import RawData, colour
from .common import check_size, enough, need
from .tiff import Tiff, camera_metadata

# tag 0x09 -> 2x2 pattern (0=R 1=G 2=B)
CFA = {1: [[0, 1], [1, 2]], 2: [[1, 0], [2, 1]], 3: [[1, 2], [0, 1]], 4: [[2, 1], [1, 0]]}


def _v4(data: bytes, off: int, width: int, height: int) -> np.ndarray:
    enough(data, off, width * height, 8, "RW2 data")
    if _accel.has("panasonic_v4"):
        return np.asarray(_accel.native.panasonic_v4(data, off, width, height)).reshape(height, width)
    out = np.zeros(height * width, np.uint16)
    buf = bytearray(0x4002)
    vbits = 0
    pos = off
    n = len(data)

    def bits(nb: int) -> int:
        nonlocal vbits, pos
        if vbits == 0:
            a = data[pos:pos + 0x1FF8]
            b = data[pos + 0x1FF8:pos + 0x4000]
            buf[0x2008:0x4000] = a.ljust(0x1FF8, b"\0")
            buf[0:0x2008] = b.ljust(0x2008, b"\0")
            pos += 0x4000
        vbits = (vbits - nb) & 0x1FFFF
        byte = (vbits >> 3) ^ 0x3FF0
        return ((buf[byte] | (buf[byte + 1] << 8)) >> (vbits & 7)) & ((1 << nb) - 1)

    if off >= n:
        raise PixportError("The RW2 data is outside the file")
    sh = 0
    k = 0
    for _row in range(height):
        pred = [0, 0]
        nonz = [0, 0]
        for col in range(width):
            i = col % 14
            if i == 0:
                pred = [0, 0]
                nonz = [0, 0]
            if i % 3 == 2:
                sh = 4 >> (3 - bits(2))
            p = i & 1
            if nonz[p]:
                j = bits(8)
                if j:
                    pred[p] -= 0x80 << sh
                    if pred[p] < 0 or sh == 4:
                        pred[p] &= (1 << sh) - 1
                    pred[p] += j << sh
            else:
                nonz[p] = bits(8)
                if nonz[p] or i > 11:
                    pred[p] = (nonz[p] << 4) | bits(4)
            out[k] = pred[col & 1] & 0xFFFF
            k += 1
    return out.reshape(height, width)


def _rotated(data: bytes, off: int, nbytes: int) -> np.ndarray:
    """The stream as the decoders see it: each 0x4000-byte block starts at its byte 0x1FF8 and
    wraps round (the file's tail is padded with zeros)."""
    nblk = -(-nbytes // 0x4000)
    raw = np.frombuffer(data[off:off + nblk * 0x4000].ljust(nblk * 0x4000, b"\0"), np.uint8).reshape(nblk, 0x4000)
    return np.concatenate([raw[:, 0x1FF8:], raw[:, :0x1FF8]], axis=1).reshape(-1)[:nbytes]


def _v5(data: bytes, off: int, width: int, height: int, bits: int) -> np.ndarray:
    per = 9 if bits == 14 else 10
    blocks = -(-width // per)
    nbytes = blocks * 16 * height
    need(data, off, min(nbytes, 16), "RW2 data")
    if len(data) - off < nbytes // 2:
        raise PixportError("The RW2 data is cut short")
    a = _rotated(data, off, nbytes).reshape(height * blocks, 16)
    bitsarr = np.unpackbits(a, axis=1, bitorder="little")
    w = (1 << np.arange(bits)).astype(np.uint32)
    vals = np.stack([bitsarr[:, k * bits:(k + 1) * bits].astype(np.uint32) @ w for k in range(per)], axis=1)
    return vals.reshape(height, blocks * per)[:, :width].astype(np.uint16)


def _v6(data: bytes, off: int, width: int, height: int) -> np.ndarray:
    blocks = width // 11
    need(data, off, blocks * 16 * height, "RW2 data")
    a = np.frombuffer(data, np.uint8, blocks * 16 * height, off).reshape(-1, 16).astype(np.int64)
    # 128 bits from the top: 14, 14, then three times (2, 10, 10, 10)
    big = [a[:, 15 - k] for k in range(16)]          # big[0] is the most significant byte
    bitw = np.zeros((a.shape[0], 128), np.uint8)
    for k in range(16):
        for b in range(8):
            bitw[:, k * 8 + b] = (big[k] >> (7 - b)) & 1
    widths = [14, 14, 2, 10, 10, 10, 2, 10, 10, 10, 2, 10, 10, 10]
    vals, p = [], 0
    for wd in widths:
        weights = (1 << np.arange(wd - 1, -1, -1)).astype(np.int64)
        vals.append(bitw[:, p:p + wd].astype(np.int64) @ weights)
        p += wd
    nblk = a.shape[0]
    out = np.empty((nblk, 11), np.int64)
    oddeven = [np.zeros(nblk, np.int64), np.zeros(nblk, np.int64)]
    nonzero = [np.zeros(nblk, np.int64), np.zeros(nblk, np.int64)]
    pmul = np.zeros(nblk, np.int64)
    base = np.zeros(nblk, np.int64)
    v = 0
    for pix in range(11):
        if pix % 3 == 2:
            b = vals[v]
            v += 1
            b = np.where(b == 3, 4, b)
            base = 0x200 << b
            pmul = 1 << b
        e = vals[v].copy()
        v += 1
        q = pix % 2
        odd = oddeven[q] != 0
        e_odd = e * pmul
        add = (base < 0x2000) & (nonzero[q] > base)
        e_odd = e_odd + np.where(add, nonzero[q] - base, 0)
        e_even = np.where(e != 0, e, nonzero[q])
        new_nonzero = np.where(odd, e_odd, np.where(e != 0, e, nonzero[q]))
        oddeven[q] = np.where(odd, oddeven[q], e)
        nonzero[q] = new_nonzero
        ep = np.where(odd, e_odd, e_even)
        sp = ep - 15
        out[:, pix] = np.where((sp >= 0) & (sp <= 0xFFFF), sp, np.where(sp < 0, 0, 0x3FFF))
    res = np.zeros((height, width), np.uint16)
    res[:, :blocks * 11] = (out.reshape(height, blocks * 11) & 0xFFFF).astype(np.uint16)
    return res


def decode(path: Path, fmt: str = "rw2") -> RawData:
    data = path.read_bytes()
    t = Tiff(data)
    ifd = t.ifds[0] if t.ifds else None
    if ifd is None:
        raise PixportError("This RW2 has no IFD")
    width, height = ifd.int(0x02), ifd.int(0x03)
    if not width or not height:
        raise PixportError("This RW2 doesn't give its sensor size")
    check_size(width, height)
    off = ifd.int(0x118)
    if off is None:
        raise PixportError("This RW2's sensor data is missing")
    need(data, off, 1)
    bits = ifd.int(0x0A, 12)
    form = ifd.int(0x2D, 4)
    if form == 4 or form < 4:
        cfa = _v4(data, off, width, height)
    elif form == 5:
        cfa = _v5(data, off, width, height, bits)
    elif form == 6:
        cfa = _v6(data, off, width, height)
    else:
        raise UnsupportedError(f"Panasonic raw format {form} isn't supported yet")
    md = camera_metadata(t)
    raw = RawData(cfa=cfa, metadata=md)
    raw.pattern = np.array(CFA.get(ifd.int(0x09, 1), CFA[1]), np.uint8)
    raw.orientation = ifd.int(274, 1) or 1
    raw.white = float((1 << bits) - 1)
    bl = [ifd.int(t_) for t_ in (0x1C, 0x1D, 0x1E)]
    if all(v is not None for v in bl):
        # format 4 stores values 15 above the black levels it records
        extra = 15 if form <= 4 else 0
        r, g, b = (v + extra for v in bl)
        per = {0: r, 1: g, 2: b}
        raw.black = np.vectorize(lambda c: per[int(c)])(raw.pattern).astype(np.float64)
    wb = [ifd.int(t_) for t_ in (0x24, 0x25, 0x26)]
    if all(wb) and len(wb) == 3:
        raw.wb = (wb[0] / wb[1], 1.0, wb[2] / wb[1])
    top, left, bottom, right = (ifd.int(t_) for t_ in (0x04, 0x05, 0x06, 0x07))
    if None not in (top, left, bottom, right) and 0 <= top < bottom <= height and 0 <= left < right <= width:
        raw.crop = (top, left, bottom - top, right - left)
    raw.metadata["bits"] = bits
    jpg = ifd.get(0x2E)
    thumb = colour.exif_thumbnail(data, jpg.pos) if jpg is not None else None
    colour.attach(raw, t, [], [thumb] if thumb else None)
    return raw
