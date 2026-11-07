"""Fujifilm RAF: synthetic files built here (an encoder for the lossless compressed format, the
plain packings, the RAF header, tags and TIFF directory) decode back to the mosaic that went in,
through the Python decoder and pixport-native's alike; damaged files end in PixportError."""

from __future__ import annotations

import random
import struct

import numpy as np
import pytest

import pixport
from pixport import _accel
from pixport.errors import PixportError, UnsupportedError
from pixport.raw import fuji

NATIVE = _accel.native if _accel.has("fuji_strip") else None
# an X-Trans layout as cameras store it in tag 0x131 (last photosite first)
XTRANS = np.array([[1, 1, 0, 1, 1, 2], [1, 1, 2, 1, 1, 0], [2, 0, 1, 0, 2, 1],
                   [1, 1, 2, 1, 1, 0], [1, 1, 0, 1, 1, 2], [0, 2, 1, 2, 0, 1]], np.uint8)
BAYER = np.array([[0, 1], [1, 2]], np.uint8)
R0, R1, R2, R3, R4, G0, G1, G2, G3, G4, G5, G6, G7, B0, B1, B2, B3, B4 = range(18)
CURRENT = (R2, R3, R4, G2, G3, G4, G5, G6, G7, B2, B3, B4)


class Writer:
    def __init__(self):
        self.out: list[int] = []

    def put(self, v: int, n: int) -> None:
        self.out.extend((v >> i) & 1 for i in range(n - 1, -1, -1))

    def bytes(self) -> bytes:
        bits = self.out + [0] * (-len(self.out) % 8)
        return bytes(int("".join(map(str, bits[i:i + 8])), 2) for i in range(0, len(bits), 8))


def _line_of(colour: int, r: int) -> int:
    return R2 + r // 2 if colour == 0 else B2 + r // 2 if colour == 2 else G2 + r


def _pos(p: int, xtrans: bool) -> int:
    if not xtrans:
        return p >> 1
    return (((p * 2 // 3) & ~1) | ((p % 3) & 1)) + ((p % 3) >> 1)


def encode_strip(mosaic: np.ndarray, xtrans: bool, bits: int = 14) -> bytes:
    """One 768-wide strip (mosaic rows, a multiple of 6) in Fujifilm's lossless code."""
    rows, width = mosaic.shape
    lw = 512 if xtrans else 384
    pattern = XTRANS if xtrans else BAYER
    total, limit = 1 << bits, 3 * bits - 1
    first = max(2, (total + 32) >> 6)
    even = [[[first, 1] for _ in range(41)] for _ in range(3)]
    odd = [[[first, 1] for _ in range(41)] for _ in range(3)]
    lines = [[0] * (lw + 2) for _ in range(18)]
    want = [[0] * lw for _ in range(18)]
    wr = Writer()

    def q(d: int) -> int:
        a = abs(d)
        b = 0 if a == 0 else 1 if a < 18 else 2 if a < 67 else 3 if a < 276 else 4
        return b if d >= 0 else -b

    def emit(g: list, pred: int, x: int, neg: bool) -> None:
        c = (pred - x) if neg else (x - pred)
        c = (c + total // 2) % total - total // 2
        u = 2 * c if c >= 0 else -2 * c - 1
        v1, v2 = g
        k = 0
        if v2 < v1:
            while k <= 14:
                k += 1
                if (v2 << k) >= v1:
                    break
        if u >> k < limit:
            wr.out.extend([0] * (u >> k) + [1])
            wr.put(u & ((1 << k) - 1), k)
        else:
            wr.out.extend([0] * limit + [1])
            wr.put(u - 1, bits)
        g[0] += abs(c)
        if g[1] == 64:
            g[0] >>= 1
            g[1] >>= 1
        g[1] += 1

    def even_one(li: int, p: int, how: str, grads: list) -> None:
        cur, prev, prev2 = lines[li], lines[li - 1], lines[li - 2]
        rb, rc, rd, rf = prev[p + 1], prev[p], prev[p + 2], prev2[p + 1]
        dc, df, dd = abs(rc - rb), abs(rf - rb), abs(rd - rb)
        v = rf + rd + 2 * rb if dc > df and dc > dd else rf + rc + 2 * rb if dd > dc and dd > df else rd + rc + 2 * rb
        if how == "d" or (how == "d0" and p & 3) or (how == "i2" and (p & 3) != 2):
            grad = q(rb - rf) * 9 + q(rc - rb)
            emit(grads[abs(grad)], v >> 2, want[li][p], grad < 0)
            cur[p + 1] = want[li][p]
        else:
            cur[p + 1] = v >> 2

    def odd_one(li: int, p: int, grads: list) -> None:
        cur, prev = lines[li], lines[li - 1]
        ra, rb, rc, rd, rg = cur[p], prev[p + 1], prev[p], prev[p + 2], cur[p + 2]
        grad = q(rb - rc) * 9 + q(rc - ra)
        v = (rg + ra + 2 * rb) >> 2 if (rb > rc and rb > rd) or (rb < rc and rb < rd) else (ra + rg) >> 1
        emit(grads[abs(grad)], v, want[li][p], grad < 0)
        cur[p + 1] = want[li][p]

    def extend(a: int, b: int) -> None:
        for i in range(a, b + 1):
            lines[i][0] = lines[i - 1][1]
            lines[i][lw + 1] = lines[i - 1][lw]

    if xtrans:
        passes = ((R2, G2, 0, "i", "d"), (G3, B2, 1, "d", "i"), (R3, G4, 2, "d0", "i"),
                  (G5, B3, 0, "d", "i2"), (R4, G6, 1, "i2", "d"), (G7, B4, 2, "i", "d0"))
    else:
        passes = tuple((a, b, g, "d", "d") for a, b, g in ((R2, G2, 0), (G3, B2, 1), (R3, G4, 2),
                                                           (G5, B3, 0), (R4, G6, 1), (G7, B4, 2)))
    span = {R2: (R2, R4), R3: (R2, R4), R4: (R2, R4), B2: (B2, B4), B3: (B2, B4), B4: (B2, B4)}
    for row in range(rows // 6):
        for li in CURRENT:
            want[li] = [0] * lw
        for r in range(6):
            for p in range(width):
                colour = int(pattern[r % pattern.shape[0], p % pattern.shape[1]])
                want[_line_of(colour, r)][_pos(p, xtrans)] = int(mosaic[row * 6 + r, p])
        for a, b, gs, how_a, how_b in passes:
            pe, po = 0, 1
            while pe < lw or po < lw:
                if pe < lw:
                    even_one(a, pe, how_a, even[gs])
                    even_one(b, pe, how_b, even[gs])
                    pe += 2
                if pe > 8:
                    odd_one(a, po, odd[gs])
                    odd_one(b, po, odd[gs])
                    po += 2
            for li in (a, b):
                extend(*span.get(li, (G2, G7)))
        for dst, src in ((R0, R3), (R1, R4), (G0, G6), (G1, G7), (B0, B3), (B1, B4)):
            lines[dst] = list(lines[src])
        for a, n in ((R2, 3), (G2, 6), (B2, 3)):
            for i in range(a, a + n):
                lines[i] = [0] * (lw + 2)
            lines[a][0] = lines[a - 1][1]
            lines[a][lw + 1] = lines[a - 1][lw]
    return wr.bytes()


def compressed(mosaic: np.ndarray, xtrans: bool, bits: int = 14) -> bytes:
    h, w = mosaic.shape
    n = -(-w // 768)
    blocks = []
    for i in range(n):
        part = np.zeros((h, 768), np.int64)
        seg = mosaic[:, i * 768:(i + 1) * 768]
        part[:, :seg.shape[1]] = seg
        blocks.append(encode_strip(part[:, :seg.shape[1]], xtrans, bits))
    head = b"IS" + bytes([1, 16 if xtrans else 0, bits]) + struct.pack(">HHHHBH", h, n * 768, w, 768, n, h // 6)
    table = struct.pack(f">{n}I", *(len(b) for b in blocks))
    table += bytes(-len(table) % 16 if (4 * n) & 0xC else 0)
    return head + table + b"".join(blocks)


def tiff(entries: list[tuple[int, int, list]]) -> bytes:
    fmt = {1: "B", 2: "B", 3: "H", 4: "I"}
    data_at = 8 + 2 + 12 * len(entries) + 4
    table, extra = b"", b""
    for tag, typ, vals in sorted(entries):
        raw = bytes(vals) + b"\0" if typ == 2 else b"".join(struct.pack("<" + fmt[typ], v) for v in vals)
        count = len(raw) if typ == 2 else len(vals)
        if len(raw) <= 4:
            table += struct.pack("<HHI", tag, typ, count) + raw.ljust(4, b"\0")
        else:
            table += struct.pack("<HHII", tag, typ, count, data_at + len(extra))
            extra += raw + b"\0" * (len(raw) & 1)
    return b"II*\0" + struct.pack("<IH", 8, len(entries)) + table + b"\0\0\0\0" + extra


def build_raf(mosaic: np.ndarray, data: bytes, xtrans: bool, bits: int = 14, orientation: int = 1,
              crop=(6, 12, 24, 600)) -> bytes:
    h, w = mosaic.shape
    exif = tiff([(271, 2, b"FUJIFILM"), (272, 2, b"X-Test"), (274, 3, [orientation])])
    jpeg = b"\xff\xd8\xff\xe1" + struct.pack(">H", len(exif) + 8) + b"Exif\0\0" + exif + b"\xff\xd9"
    recs = [(0x100, struct.pack(">HH", h, w)), (0x110, struct.pack(">HH", crop[0], crop[1])),
            (0x111, struct.pack(">HH", crop[2], crop[3])), (0x130, b"\x0c\x0c\x0c\x0c")]
    if xtrans:
        recs.append((0x131, XTRANS.reshape(-1)[::-1].tobytes()))
    meta = struct.pack(">I", len(recs)) + b"".join(struct.pack(">HH", t, len(v)) + v for t, v in recs)
    blk = [64 + i for i in range(36 if xtrans else 4)]

    # lay out the TIFF by hand: header, IFD0 (one entry: 0xF000 -> sub IFD at 26), sub IFD, data
    ifd0 = b"II*\0" + struct.pack("<IH", 8, 1) + struct.pack("<HHII", 0xF000, 4, 1, 26) + b"\0\0\0\0"
    entries = [(0xF001, [w]), (0xF002, [h]), (0xF003, [bits]), (0xF007, [0]), (0xF008, [len(data)]),
               (0xF00A, blk), (0xF00E, [300, 600, 450])]
    sub_len = 2 + 12 * len(entries) + 4
    extra_at = 26 + sub_len
    extra, table = b"", b""
    data_at = extra_at + 4 * (len(blk) + 3) + 64
    for tag, vals in entries:
        if tag == 0xF007:
            vals = [data_at]
        raw = b"".join(struct.pack("<I", v) for v in vals)
        if len(raw) <= 4:
            table += struct.pack("<HHI", tag, 4, len(vals)) + raw
        else:
            table += struct.pack("<HHII", tag, 4, len(vals), extra_at + len(extra))
            extra += raw
    cfa = ifd0 + struct.pack("<H", len(entries)) + table + b"\0\0\0\0" + extra
    cfa = cfa.ljust(data_at, b"\0") + data
    head = fuji.MAGIC + b"0201" + b"FF129502" + b"X-Test".ljust(32, b"\0") + b"0100" + bytes(20)
    jpeg_at = 148
    meta_at = jpeg_at + len(jpeg)
    cfa_at = meta_at + len(meta)
    head += struct.pack(">6I", jpeg_at, len(jpeg), meta_at, len(meta), cfa_at, len(cfa))
    return head.ljust(jpeg_at, b"\0") + jpeg + meta + cfa


def mosaic_for_test(h: int, w: int, bits: int = 14, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    top = (1 << bits) - 1
    y, x = np.mgrid[0:h, 0:w]
    m = top // 8 + 3 * x + 40 * y + rng.integers(-40, 40, (h, w))
    m[:, :50] = 64
    m[3, 100:140] = top
    m[7, 300:330] = rng.integers(0, top + 1, 30)
    return np.clip(m, 0, top).astype(np.uint16)


def decoders():
    yield "python", None
    if NATIVE is not None:
        yield "native", NATIVE


def _strip(impl, data: bytes, lines: int, xtrans: bool, bits: int = 14) -> np.ndarray:
    lw = 512 if xtrans else 384
    if impl is None:
        saved = _accel.native
        _accel.native = None
        try:
            hdr = fuji.Header(b"IS\x01" + bytes([16 if xtrans else 0, bits]) + struct.pack(">HHHHBH", lines * 6, 768, 768, 768, 1, lines))
            return fuji.decode_strip(data, 0, len(data), hdr)
        finally:
            _accel.native = saved
    return np.frombuffer(impl.fuji_strip(data, 0, len(data), lw, lines, bits, xtrans), "<u2").reshape(lines, 12, lw)


@pytest.mark.parametrize("name,impl", list(decoders()))
@pytest.mark.parametrize("xtrans", [True, False])
def test_strip_round_trip(name, impl, xtrans):
    m = mosaic_for_test(12, 768, seed=3)
    data = encode_strip(m, xtrans)
    strip = _strip(impl, data, 2, xtrans)
    src, at = fuji._gather(xtrans, XTRANS if xtrans else BAYER, strip.shape[2])
    got = strip[:, src, at].reshape(12, 768)
    assert np.array_equal(got, m)


def test_native_matches_python_on_noise():
    if NATIVE is None:
        pytest.skip("pixport-native isn't installed")
    rnd = random.Random(5)
    for i in range(12):
        data = bytes(rnd.getrandbits(8) for _ in range(rnd.choice((16, 400, 3000))))
        xtrans = bool(i & 1)
        outs = []
        for impl in (None, NATIVE):
            try:
                outs.append(_strip(impl, data, 1, xtrans))
            except (ValueError, PixportError):
                outs.append("error")
        assert (isinstance(outs[0], str) and isinstance(outs[1], str)) or np.array_equal(outs[0], outs[1])


def _write(tmp_path, data: bytes, name: str = "t.raf"):
    p = tmp_path / name
    p.write_bytes(data)
    return p


@pytest.mark.parametrize("xtrans", [True, False])
def test_compressed_file(tmp_path, xtrans):
    m = mosaic_for_test(18, 792)                        # two strips, the second 24 wide
    p = _write(tmp_path, build_raf(m, compressed(m, xtrans), xtrans, crop=(6, 12, 12, 600)))
    raw = fuji.decode(p)
    assert np.array_equal(raw.cfa, m)
    assert raw.pattern.tolist() == (XTRANS if xtrans else BAYER).tolist()
    assert raw.crop == (6, 12, 12, 600)
    assert np.allclose(raw.wb, (2.0, 1.0, 1.5))
    assert np.asarray(raw.black).reshape(-1)[:4].tolist() == [64, 65, 66, 67]
    assert raw.white == 16383
    assert raw.metadata["model"] == "X-Test"
    assert pixport.detect(p) == "raf"


def _pack12(m: np.ndarray) -> bytes:
    v = m.reshape(-1, 2).astype(np.uint32)
    b = np.empty((v.shape[0], 3), np.uint8)
    b[:, 0] = v[:, 0] & 0xFF
    b[:, 1] = (v[:, 0] >> 8) | ((v[:, 1] & 15) << 4)
    b[:, 2] = v[:, 1] >> 4
    return b.tobytes()


def _pack14(m: np.ndarray) -> bytes:
    v = m.reshape(-1, 4).astype(np.uint64)
    word = (v[:, 0] << 42) | (v[:, 1] << 28) | (v[:, 2] << 14) | v[:, 3]
    b = np.stack([(word >> np.uint64(8 * (6 - i))) & np.uint64(0xFF) for i in range(7)], axis=1).astype(np.uint8)
    return b.reshape(-1, 4)[:, ::-1].tobytes()


@pytest.mark.parametrize("bits,pack", [(14, "16"), (12, "12"), (14, "14")])
def test_plain_files(tmp_path, bits, pack):
    m = mosaic_for_test(30, 640, bits=bits, seed=bits)
    data = {"16": m.astype("<u2").tobytes(), "12": _pack12(m), "14": _pack14(m)}[pack]
    p = _write(tmp_path, build_raf(m, data, True, bits=bits, crop=(6, 12, 24, 600)))
    raw = fuji.decode(p)
    assert np.array_equal(raw.cfa, m)
    assert raw.crop == (6, 12, 24, 600)
    assert raw.white == (1 << bits) - 1


def test_develops_upright(tmp_path):
    m = mosaic_for_test(30, 640)
    img = pixport.read(_write(tmp_path, build_raf(m, m.astype("<u2").tobytes(), False, orientation=8)), half=True)
    assert img.pixels.shape[:2] == (300, 12)
    assert img.format == "raf"


def test_lossy_is_unsupported(tmp_path):
    m = mosaic_for_test(18, 792)
    data = bytearray(compressed(m, True))
    data[2] = 0
    with pytest.raises(UnsupportedError):
        pixport.read(_write(tmp_path, build_raf(m, bytes(data), True)))


def test_damaged_files(tmp_path):
    m = mosaic_for_test(18, 792)
    good = build_raf(m, compressed(m, True), True)
    cfa_at = struct.unpack_from(">I", good, 100)[0]
    cases = [good[:120], good[:-500], good[:cfa_at] + b"XX" + good[cfa_at + 2:]]
    bad = bytearray(good)
    data_at = good.find(b"IS\x01\x10")
    bad[data_at + 5:data_at + 7] = b"\xff\xff"            # height
    cases.append(bytes(bad))
    junk = bytearray(good)
    junk[data_at + 80:data_at + 4000] = bytes(3920)
    cases.append(bytes(junk))
    for i, data in enumerate(cases):
        try:
            pixport.read(_write(tmp_path, data, f"bad{i}.raf"))
        except PixportError:
            continue
