"""Lossless JPEG (ITU T.81 process 14, "LJ92"), the compression inside CR2, most DNGs and Sony's
lossless ARW, and the Huffman helpers the makers' own codecs share.

A frame has 1-4 components; each sample is predicted from its decoded neighbours (one of the
seven predictors of T.81 table H.1) and only the difference is Huffman coded. Canon's sRAW/mRAW
subsample the chroma; their MCU (e.g. Y Y Cb Cr) is decoded as one row of "slots", each slot
predicted from the same slot in the MCU before it, which is how Canon's encoder works.

:func:`decode` returns the samples as (rows, mcus_per_row * slots) uint16. The Huffman stage is
bit-serial; pixport-native has it compiled (``ljpeg_decode``), with the same results."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

from .. import _accel
from ..errors import PixportError, UnsupportedError

MAX_SAMPLES = 1 << 30


def build_lut(counts, symbols, bits: int = 16) -> np.ndarray:
    """A lookup table for a canonical Huffman code (JPEG's BITS/HUFFVAL): indexed by the next
    ``bits`` bits of the stream, each entry is ``length << 8 | symbol`` (0 where no code starts)."""
    lut = np.zeros(1 << bits, dtype=np.uint16)
    code, k = 0, 0
    syms = list(symbols)
    for length in range(1, 17):
        n = int(counts[length - 1]) if length - 1 < len(counts) else 0
        for _ in range(n):
            if k >= len(syms):
                raise ValueError("Huffman table has fewer symbols than codes")
            if length <= bits:
                if code >= 1 << length:
                    raise ValueError("Huffman table is over-full")
                lo = code << (bits - length)
                lut[lo:lo + (1 << (bits - length))] = length << 8 | (syms[k] & 0xFF)
            code += 1
            k += 1
        code <<= 1
    return lut


@dataclass
class Frame:
    """What the JPEG headers say: sample precision, size, components (id, h, v), the scan's
    Huffman table per component, predictor, point transform and restart interval."""

    precision: int = 0
    height: int = 0
    width: int = 0
    comps: list = field(default_factory=list)
    tables: dict = field(default_factory=dict)
    scan_tables: list = field(default_factory=list)
    predictor: int = 1
    pt: int = 0
    restart: int = 0
    scan_start: int = 0

    @property
    def slots(self) -> list[int]:
        """The component index of each sample in one MCU (Y Y Cb Cr for 4:2:2)."""
        if len(self.comps) == 1:
            return [0]
        out = []
        for i, (_cid, h, v) in enumerate(self.comps):
            out += [i] * (h * v)
        return out

    @property
    def mcus(self) -> int:
        hmax = max(h for _c, h, _v in self.comps)
        return (self.width + hmax - 1) // hmax if len(self.comps) > 1 else self.width

    @property
    def rows(self) -> int:
        vmax = max(v for _c, _h, v in self.comps)
        return (self.height + vmax - 1) // vmax if len(self.comps) > 1 else self.height


def parse(data: bytes, pos: int = 0, end: int | None = None) -> Frame:
    """Read the markers from SOI up to the start of the (first) scan's entropy-coded data."""
    end = len(data) if end is None else min(end, len(data))
    if data[pos:pos + 2] != b"\xff\xd8":
        raise PixportError("The lossless JPEG data doesn't start with SOI")
    f = Frame()
    p = pos + 2
    while p + 4 <= end:
        if data[p] != 0xFF:
            p += 1
            continue
        marker = data[p + 1]
        if marker in (0xFF, 0x01) or 0xD0 <= marker <= 0xD7:
            p += 1 if marker == 0xFF else 2
            continue
        length = struct.unpack_from(">H", data, p + 2)[0]
        seg = data[p + 4:p + 2 + length]
        if length < 2 or p + 2 + length > end:
            raise PixportError("A lossless JPEG marker runs past the data")
        if marker == 0xC4:
            q = 0
            while q + 17 <= len(seg):
                tc_th = seg[q]
                counts = list(seg[q + 1:q + 17])
                total = sum(counts)
                syms = seg[q + 17:q + 17 + total]
                if len(syms) < total:
                    raise PixportError("A Huffman table is cut short")
                try:
                    f.tables[tc_th & 0x0F] = build_lut(counts, syms)
                except ValueError as e:
                    raise PixportError(f"Bad Huffman table: {e}") from None
                q += 17 + total
        elif marker in (0xC0, 0xC1, 0xC2, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            raise UnsupportedError("This JPEG data isn't lossless (process 14)")
        elif marker == 0xC3:
            if len(seg) < 6:
                raise PixportError("The lossless JPEG frame header is cut short")
            f.precision, f.height, f.width, nf = struct.unpack_from(">BHHB", seg, 0)
            if not 1 <= nf <= 4 or len(seg) < 6 + 3 * nf:
                raise PixportError("The lossless JPEG frame has a bad component count")
            f.comps = [(seg[6 + 3 * i], seg[7 + 3 * i] >> 4, seg[7 + 3 * i] & 15) for i in range(nf)]
            if any(not 1 <= h <= 4 or not 1 <= v <= 4 for _c, h, v in f.comps):
                raise PixportError("The lossless JPEG frame has bad sampling factors")
            if not 2 <= f.precision <= 16:
                raise PixportError(f"Lossless JPEG precision {f.precision} isn't valid")
        elif marker == 0xDD:
            f.restart = struct.unpack_from(">H", seg, 0)[0] if len(seg) >= 2 else 0
        elif marker == 0xDA:
            ns = seg[0] if seg else 0
            if not f.comps or ns != len(f.comps) or len(seg) < 1 + 2 * ns + 3:
                raise UnsupportedError("Lossless JPEG with one scan per component isn't supported")
            ids = [c[0] for c in f.comps]
            tabs = {}
            for i in range(ns):
                cid, td = seg[1 + 2 * i], seg[2 + 2 * i] >> 4
                if cid not in ids:
                    raise PixportError("A lossless JPEG scan names an unknown component")
                tabs[ids.index(cid)] = td
            f.scan_tables = [tabs.get(i, 0) for i in range(len(f.comps))]
            f.predictor = seg[1 + 2 * ns]
            f.pt = seg[3 + 2 * ns] & 15
            f.scan_start = p + 2 + length
            if not 1 <= f.predictor <= 7:
                raise PixportError(f"Lossless JPEG predictor {f.predictor} isn't valid")
            if f.pt >= f.precision:
                raise PixportError("The lossless JPEG point transform isn't valid for its precision")
            for t in f.scan_tables:
                if t not in f.tables:
                    raise PixportError("A lossless JPEG scan uses a missing Huffman table")
            if f.width < 1 or f.height < 1 or f.rows * f.mcus * len(f.slots) > MAX_SAMPLES:
                raise PixportError("The lossless JPEG frame size isn't valid")
            return f
        p += 2 + length
    raise PixportError("The lossless JPEG data has no scan")


def segments(data: bytes, start: int, end: int) -> list[bytes]:
    """The entropy-coded data from ``start``, split at restart markers, with 0xFF00 stuffing
    removed; it ends at the first other marker (or ``end``)."""
    out, p = [], start
    seg_start = start
    while True:
        q = data.find(b"\xff", p, end)
        if q < 0 or q + 1 >= end:
            out.append(data[seg_start:end])
            break
        m = data[q + 1]
        if m == 0x00 or m == 0xFF:
            p = q + 1 if m == 0xFF else q + 2
            continue
        out.append(data[seg_start:q])
        if 0xD0 <= m <= 0xD7:
            seg_start = p = q + 2
            continue
        break
    return [s.replace(b"\xff\x00", b"\xff") for s in out]


def _diffs(seg: bytes, n: int, luts: list, slot_tab: list[int]) -> np.ndarray:
    """``n`` Huffman-coded differences from one stuffing-free segment (T.81 F.2.2.1 with
    SSSS 16 = 32768); bits past the end read as zeros."""
    buf = seg + b"\0" * 8
    nbits = len(seg) * 8
    out = np.empty(n, dtype=np.int32)
    tabs = [t.tolist() for t in luts]
    per = [tabs[t] for t in slot_tab]
    stride = len(per)
    pos = 0
    frm = int.from_bytes
    for i in range(n):
        lut = per[i % stride] if stride > 1 else per[0]
        b = pos >> 3
        sh = pos & 7
        w = frm(buf[b:b + 5], "big")
        e = lut[(w >> (24 - sh)) & 0xFFFF]
        if not e:
            if pos >= nbits:
                raise PixportError("The lossless JPEG data ends early")
            raise PixportError("The lossless JPEG data has a bad Huffman code")
        ln = e >> 8
        s = e & 0xFF
        if s == 0:
            out[i] = 0
            pos += ln
        elif s >= 16:
            out[i] = 32768
            pos += ln
        else:
            v = (w >> (40 - sh - ln - s)) & ((1 << s) - 1)
            out[i] = v if v >> (s - 1) else v - (1 << s) + 1
            pos += ln + s
    if pos > nbits + 64:
        raise PixportError("The lossless JPEG data ends early")
    return out


def predict(diffs: np.ndarray, rows: int, width: int, stride: int, predictor: int, precision: int, pt: int,
            restart_rows: int = 0, chain: int = 0, chain_rows: int = 1) -> np.ndarray:
    """Undo the prediction: ``diffs`` (rows x width, int) to samples (uint16). ``stride`` is the
    number of slots per MCU; each slot is predicted from its own earlier samples, except the first
    ``chain`` (luma) slots: split into ``chain_rows`` groups, each group is one sequence along the
    row, the first group starting from its start one MCU row up and the others from the first
    group's start (Canon sRAW: one group; Sony's YCbCr ARW: one per luma row)."""
    d = diffs.reshape(rows, width).astype(np.int64)
    if chain > 1:
        if predictor != 1 or restart_rows:
            raise UnsupportedError("This Canon sRAW coding isn't supported")
        out = np.empty((rows, width), dtype=np.int64)
        init = 1 << (precision - pt - 1)
        per = chain // chain_rows
        lumas = [np.array([j for j in range(width) if g * per <= j % stride < (g + 1) * per]) for g in range(chain_rows)]
        start0 = None
        for k, cols in enumerate(lumas + [np.arange(s_, width, stride) for s_ in range(chain, stride)]):
            part = d[:, cols]
            if k == 0 or k >= chain_rows:
                start = init + np.cumsum(part[:, 0])
            else:
                start = start0 + part[:, 0]
            if k == 0:
                start0 = start
            run = np.zeros_like(part)
            run[:, 1:] = np.cumsum(part[:, 1:], axis=1)
            out[:, cols] = start[:, None] + run
        out &= 0xFFFF
        if pt:
            out = (out << pt) & 0xFFFF
        return out.astype(np.uint16)
    out = np.empty((rows, width), dtype=np.int64)
    mask = 0xFFFF
    init = 1 << (precision - pt - 1)
    period = restart_rows or rows
    for r0 in range(0, rows, period):
        r1 = min(rows, r0 + period)
        blk, ob = d[r0:r1], out[r0:r1]
        # first row of the block: left neighbour, the first sample of each slot from init
        first = blk[0].copy()
        first[:stride] += init
        for s in range(min(stride, width)):
            ob[0, s::stride] = np.cumsum(first[s::stride])
        ob[0] &= mask
        if r1 - r0 == 1:
            continue
        # later rows: their first MCU from the sample above
        ob[1:, :stride] = (ob[0, :stride] + np.cumsum(blk[1:, :stride], axis=0)) & mask
        if predictor == 1:
            for s in range(min(stride, width)):
                col = ob[1:, s:s + 1]
                ob[1:, s + stride::stride] = (col + np.cumsum(blk[1:, s + stride::stride], axis=1)) & mask
            continue
        for r in range(1, r1 - r0):
            up = ob[r - 1]
            row = ob[r]
            dr = blk[r]
            if predictor == 2:
                row[stride:] = (up[stride:] + dr[stride:]) & mask
            elif predictor == 3:
                row[stride:] = (up[:-stride] + dr[stride:]) & mask
            elif predictor in (4, 5):
                b, c = up[stride:], up[:-stride]
                inc = dr[stride:] + ((b - c) if predictor == 4 else ((b - c) >> 1))
                for s in range(min(stride, width - stride)):
                    row[stride + s::stride] = (row[s] + np.cumsum(inc[s::stride])) & mask
            else:
                for j in range(stride, width):
                    a, b, c = int(row[j - stride]), int(up[j]), int(up[j - stride])
                    p = b + ((a - c) >> 1) if predictor == 6 else (a + b) >> 1
                    row[j] = (p + int(dr[j])) & mask
    if pt:
        out = (out << pt) & mask
    return out.astype(np.uint16)


def decode(data: bytes, pos: int = 0, end: int | None = None, frame: Frame | None = None,
           chain: bool = False, chain_rows: int = 1) -> tuple[np.ndarray, Frame]:
    """Decode the lossless JPEG at ``data[pos:end]``: (samples as rows x mcus*slots uint16, frame).
    ``chain``: the first component's samples are predicted as one sequence along each row (Canon
    sRAW)."""
    end = len(data) if end is None else min(end, len(data))
    f = frame or parse(data, pos, end)
    slots = f.slots
    stride = len(slots)
    nchain = slots.count(0) if chain and len(f.comps) > 1 else 0
    if nchain and (chain_rows < 1 or nchain % chain_rows):
        raise UnsupportedError("This lossless JPEG luma layout isn't supported")
    rows, width = f.rows, f.mcus * stride
    if (end - f.scan_start) * 8 < rows * width:
        raise PixportError("The lossless JPEG data is too short for its frame")
    if _accel.has("ljpeg_decode"):
        luts = [f.tables[f.scan_tables[c]] for c in range(len(f.comps))]
        tabs = np.stack(luts)
        slot_tab = np.array(slots, dtype=np.uint8)
        out = _accel.native.ljpeg_decode(data, f.scan_start, end, tabs, slot_tab, rows, width, f.predictor,
                                         f.precision, f.pt, f.restart, nchain, chain_rows)
        return np.asarray(out).reshape(rows, width), f
    segs = segments(data, f.scan_start, end)
    luts = [f.tables[f.scan_tables[c]] for c in range(len(f.comps))]
    total = rows * width
    if f.restart and len(segs) > 1:
        per = f.restart * stride
        if (f.restart * stride) % width:
            raise UnsupportedError("Lossless JPEG restart intervals that split a row aren't supported")
        parts, left = [], total
        for s in segs:
            if left <= 0:
                break
            k = min(per, left)
            parts.append(_diffs(s, k, luts, slots))
            left -= k
        if left > 0:
            raise PixportError("The lossless JPEG data ends early")
        diffs = np.concatenate(parts)
        restart_rows = per // width
    else:
        diffs = _diffs(segs[0], total, luts, slots)
        restart_rows = 0
    return predict(diffs, rows, width, stride, f.predictor, f.precision, f.pt, restart_rows, nchain, chain_rows), f


def unslice(samples: np.ndarray, height: int, width: int, slices: tuple[int, int, int] | None) -> np.ndarray:
    """Canon's CR2 layout: the decoded samples, in order, fill ``slices[0]`` vertical strips
    ``slices[1]`` wide and then one ``slices[2]`` wide, each top to bottom."""
    flat = samples.reshape(-1)
    if flat.size < height * width:
        raise PixportError("The CR2 image data is smaller than its size")
    if not slices or not slices[0]:
        return flat[:height * width].reshape(height, width)
    n, w1, w2 = slices
    widths = [w1] * n + [w2]
    if sum(widths) != width or min(widths) < 0:
        raise PixportError("The CR2 slice layout doesn't match its width")
    out = np.empty((height, width), dtype=samples.dtype)
    p, x = 0, 0
    for w in widths:
        if w:
            out[:, x:x + w] = flat[p:p + height * w].reshape(height, w)
        p += height * w
        x += w
    return out
