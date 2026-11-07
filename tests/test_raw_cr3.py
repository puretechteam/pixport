"""Canon CR3: synthetic files built here (an encoder for CRX's line codes, the forward 5/3
wavelet, the ISO boxes and TIFF metadata) decode back to the mosaic that went in, through the
Python decoders and pixport-native's alike; damaged files end in PixportError."""

from __future__ import annotations

import random
import struct

import numpy as np
import pytest

import pixport
from pixport import _accel
from pixport.errors import PixportError, UnsupportedError
from pixport.raw import cr3

NATIVE = _accel.native if _accel.has("crx_band") else None


# ---------------------------------------------------------------------------------------------
# an encoder for the CRX line codes, step for step the decoder's state machine


class Writer:
    def __init__(self):
        self.out: list[int] = []

    def put(self, v: int, n: int) -> None:
        self.out.extend((v >> i) & 1 for i in range(n - 1, -1, -1))

    def code(self, c: int, k: int, esc: int = 41, esc_bits: int = 21) -> None:
        assert 0 <= c < 1 << esc_bits
        z = c >> k
        if z >= esc:
            self.out.extend([0] * esc + [1])
            self.put(c, esc_bits)
        else:
            self.out.extend([0] * z + [1])
            self.put(c & ((1 << k) - 1), k)

    def run(self, n: int, room: int, s: int) -> int:
        """A run of n (after its leading 1), as cr3._run reads it."""
        cnt = 1
        while True:
            step = 1 << cr3.J[s]
            if cnt == room:
                self.out.append(0)
                break
            if n - cnt >= step:
                self.out.append(1)
                cnt += step
                s = min(s + 1, 31)
                if cnt == room:
                    return s
            elif n == room:
                self.out.append(1)
                return s
            else:
                self.out.append(0)
                break
        if cnt < room:
            if cr3.J[s]:
                self.put(n - cnt, cr3.J[s])
            s = max(s - 1, 0)
        return s

    def bytes(self) -> bytes:
        bits = self.out + [0] * (-len(self.out) % 8)
        return bytes(int("".join(map(str, bits[i:i + 8])), 2) for i in range(0, len(bits), 8)) + bytes(8)


def uz(v: int) -> int:
    return 2 * v if v >= 0 else -2 * v - 1


def _same_run(vals, start: int, value: int, room: int) -> int:
    n = 0
    while n < room and vals[start + n] == value:
        n += 1
    return n


def encode_predicted(a: np.ndarray) -> bytes:
    h, w = a.shape
    wr, k, s = Writer(), 0, 0
    rows = [list(map(int, r)) for r in a]
    v = rows[0]
    cur = [0] * (w + 2)
    i, left = 1, w
    while left > 1:
        if cur[i - 1]:
            pred = cur[i - 1]
        else:
            n = _same_run(v, i - 1, 0, left)
            wr.out.append(1 if n else 0)
            if n:
                s = wr.run(n, left, s)
                i += n
                left -= n
                if left <= 0:
                    break
            pred = 0
        c = uz(v[i - 1] - pred)
        wr.code(c, k)
        cur[i] = v[i - 1]
        k = cr3._knext(k, c)
        i += 1
        left -= 1
    if left == 1:
        c = uz(v[i - 1] - cur[i - 1])
        wr.code(c, k)
        cur[i] = v[i - 1]
        k = cr3._knext(k, c)
    cur[w + 1] = cur[w] + 1
    prev = cur
    for y in range(1, h):
        v = rows[y]
        cur = [0] * (w + 2)
        cur[0] = prev[1]
        i, left = 1, w
        while left > 1:
            a0 = cur[i - 1]
            if a0 != prev[i] or a0 != prev[i + 1]:
                c = uz(v[i - 1] - cr3._med(a0, prev[i], prev[i - 1]))
                wr.code(c, k)
                cur[i] = v[i - 1]
                k = cr3._knext(k, (c + 2 * abs(prev[i + 1] - prev[i])) >> 1)
                i += 1
                left -= 1
                continue
            n = _same_run(v, i - 1, a0, left)
            wr.out.append(1 if n else 0)
            if n:
                s = wr.run(n, left, s)
                cur[i:i + n] = [a0] * n
                i += n
                left -= n
            if left > 0:
                c = uz(v[i - 1] - prev[i])
                wr.code(c, k)
                cur[i] = v[i - 1]
                cc = (c + 2 * abs(prev[i + 1] - prev[i])) >> 1 if left > 1 else c
                k = cr3._knext(k, cc)
                i += 1
                left -= 1
        if left == 1:
            c = uz(v[i - 1] - cr3._med(cur[i - 1], prev[i], prev[i - 1]))
            wr.code(c, k)
            cur[i] = v[i - 1]
            k = cr3._knext(k, c)
        cur[w + 1] = cur[w] + 1
        prev = cur
    return wr.bytes()


def encode_plain(a: np.ndarray) -> bytes:
    h, w = a.shape
    wr, k, s = Writer(), 0, 0
    rows = [list(map(int, r)) for r in a]
    kline = [0] * (w + 1)

    def adjust(k: int, i: int) -> int:
        if kline[i + 1] - k > 1:
            return k + 1
        return min(k, 15)

    v = rows[0]
    cur = [0] * (w + 2)
    i, left = 1, w
    while left > 1:
        if cur[i - 1]:
            c = uz(v[i - 1])
        else:
            n = _same_run(v, i - 1, 0, left)
            wr.out.append(1 if n else 0)
            if n:
                s = wr.run(n, left, s)
                for j in range(i - 1, i - 1 + n):
                    kline[j] = 0
                i += n
                left -= n
                if left <= 0:
                    break
            c = uz(v[i - 1]) - 1
        wr.code(c, k)
        k = cr3._knext(k, c)
        cur[i] = v[i - 1]
        kline[i - 1] = k
        i += 1
        left -= 1
    if left == 1:
        c = uz(v[i - 1])
        wr.code(c, k)
        k = cr3._knext(k, c)
        cur[i] = v[i - 1]
        kline[i - 1] = k
    prev = cur
    for y in range(1, h):
        v = rows[y]
        cur = [0] * (w + 2)
        i = 0
        while i < w - 1:
            if prev[i + 2] or prev[i + 1] or cur[i]:
                c = uz(v[i])
                wr.code(c, k)
                k = adjust(cr3._knext(k, c, 0), i)
                cur[i + 1] = v[i]
            else:
                n = _same_run(v, i, 0, w - i)
                wr.out.append(1 if n else 0)
                if n:
                    s = wr.run(n, w - i, s)
                    for j in range(i, i + n):
                        kline[j] = 0
                    i += n
                if i >= w - 1:
                    if i == w - 1:
                        c = uz(v[i]) - 1
                        wr.code(c, k)
                        k = cr3._knext(k, c)
                        kline[i] = k
                        cur[i + 1] = v[i]
                    i += 1
                    continue
                c = uz(v[i]) - 1
                wr.code(c, k)
                k = adjust(cr3._knext(k, c, 0), i)
                cur[i + 1] = v[i]
            kline[i] = k
            i += 1
        if i == w - 1:
            c = uz(v[i])
            wr.code(c, k)
            k = cr3._knext(k, c)
            kline[i] = k
            cur[i + 1] = v[i]
        prev = cur
    return wr.bytes()


def encode_qp(a: np.ndarray) -> bytes:
    h, w = a.shape
    wr, k = Writer(), 0
    prev = None
    for y in range(h):
        v = list(map(int, a[y]))
        cur = [0] * (w + 2)
        if prev is not None:
            cur[0] = prev[1]
        for x in range(w):
            pred = cur[x] if prev is None else cr3._med(cur[x], prev[x + 1], prev[x])
            c = uz(v[x] - pred)
            wr.code(c, k, 23, 8)
            cur[x + 1] = v[x]
            if prev is not None and x < w - 1:
                k = cr3._knext(k, (c + 2 * abs(prev[x + 2] - prev[x + 1])) >> 1, 7)
            else:
                k = cr3._knext(k, c, 7)
        cur[w + 1] = cur[w] + 1
        prev = cur
    return wr.bytes()


# ---------------------------------------------------------------------------------------------
# the forward 5/3 wavelet (JPEG 2000's reversible lifting, mirrored ends)


def fwd(x: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    if axis == 1:
        lo, hi = fwd(x.T, 0)
        return lo.T, hi.T
    x = x.astype(np.int64)
    n = len(x)
    even, odd = x[0::2], x[1::2]
    no = len(odd)
    nxt = even[1:no + 1] if len(even) > no else np.concatenate([even[1:], even[-1:]])
    high = odd - ((even[:no] + nxt) >> 1)
    if n == 1:
        return even, high
    hp = np.concatenate([high[:1], high])               # high[k - 1], mirrored before the start
    hn = high if len(high) == len(even) else np.concatenate([high, high[-1:]])
    low = even + ((hp[:len(even)] + hn + 2) >> 2)
    return low, high


def crx_bands(plane: np.ndarray, tiles_x: int, levels: int) -> list[list[np.ndarray]]:
    """Each tile's subbands (LL first, then coarsest to finest HL, LH, HH) for a plane cut into
    tiles_x equal tiles across: every tile is transformed with its neighbours' values around it,
    as Canon's encoder does, keeping the extra coefficients the decoder needs at tile edges."""
    w = plane.shape[1]
    tw = w // tiles_x
    out = []
    for c in range(tiles_x):
        a, b = c * tw, (c + 1) * tw if c < tiles_x - 1 else w
        lo, hi = c > 0, c < tiles_x - 1
        m = a - (a % (1 << levels)) if lo else 0           # margin on the left, keeps parity
        m = min(m, 64 - 64 % (1 << levels)) if lo else 0
        seg = plane[:, a - m:]
        xs = cr3._lengths(b - a, lo, hi, levels)
        bands = []
        off = m
        for level in range(1, levels + 1):
            low_rows, high_rows = fwd(seg, 0)
            ll, hl = fwd(low_rows, 1)
            lh, hh = fwd(high_rows, 1)
            _out, nl, nh = xs[level - 1][:3]
            o = off // 2
            bands.append((hl[:, o - lo:o - lo + nh], lh[:, o:o + nl], hh[:, o - lo:o - lo + nh]))
            seg, off = ll, o
        _out, nl, _nh = xs[-1][:3]
        tile = [seg[:, off:off + nl]]
        for hl, lh, hh in reversed(bands):
            tile += [hl, lh, hh]
        out.append(tile)
    return out


# ---------------------------------------------------------------------------------------------
# the container


def box(typ: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + typ + payload


def tiff(entries: list[tuple[int, int, list]]) -> bytes:
    """A little-endian TIFF with one IFD: (tag, type, values); type 2 takes bytes."""
    fmt = {1: "B", 2: "B", 3: "H", 4: "I", 5: "II", 7: "B"}
    data_at = 8 + 2 + 12 * len(entries) + 4
    table, extra = b"", b""
    for tag, typ, vals in sorted(entries):
        if typ == 2:
            raw = bytes(vals) + b"\0"
            count = len(raw)
        else:
            raw = b"".join(struct.pack("<" + fmt[typ], *(v if isinstance(v, tuple) else (v,))) for v in vals)
            count = len(vals)
        if len(raw) <= 4:
            table += struct.pack("<HHI", tag, typ, count) + raw.ljust(4, b"\0")
        else:
            table += struct.pack("<HHII", tag, typ, count, data_at + len(extra))
            extra += raw + b"\0" * (len(raw) & 1)
    return b"II*\0" + struct.pack("<IH", 8, len(entries)) + table + b"\0\0\0\0" + extra


def crx_data(bands: list[list[list[np.ndarray]]], planes_pred: bool = True, qp: list[bytes] | None = None,
             q_mult: int = 0) -> tuple[bytes, int]:
    """The sensor sample: headers then the data. bands[tile][plane] = [band arrays]; qp[tile]
    switches to the newer headers (with a quantisation map)."""
    new = qp is not None
    head, body = b"", b""
    for t, tile in enumerate(bands):
        planes_blob, plane_heads = b"", b""
        for p, blist in enumerate(tile):
            pblob, bheads = b"", b""
            for i, arr in enumerate(blist):
                enc = (encode_predicted if planes_pred and i == 0 else encode_plain)(arr)
                if new:
                    bheads += struct.pack(">HHIHHIHH", 0xFF13, 16, len(enc), i << 12, q_mult if i else 0,
                                          0 if i else 1, 0, 0)
                else:
                    bheads += struct.pack(">HHII", 0xFF03, 8, len(enc), (i << 28) | (4 << 19))
                pblob += enc
            plane_heads += struct.pack(">HHIB3x", 0xFF12 if new else 0xFF02, 8, len(pblob), (p << 4) | (8 if planes_pred else 0))
            plane_heads += bheads
            planes_blob += pblob
        lead = b""
        if new:
            q = qp[t]
            pad = -len(q) % 8
            lead = q + bytes(pad)
            th = struct.pack(">HHIHHIHH", 0xFF11, 16, len(lead) + len(planes_blob), t, 0x4000, len(q), pad, 0)
        else:
            th = struct.pack(">HHIHH", 0xFF01, 8, len(planes_blob), t, 0)
        head += th + plane_heads
        body += lead + planes_blob
    return head + body, len(head)


def cmp1(w: int, h: int, tw: int, th: int, levels: int, hdr_size: int, version: int = 0x100, enc: int = 0,
         tiles_x: int = 1) -> bytes:
    b = struct.pack(">HHHHIIII", 0xFF00, 0x30, version, 0, w, h, tw, th)
    b += bytes([14, (4 << 4) | 0, (enc << 4) | levels, 0x80 if tiles_x > 1 else 0]) + struct.pack(">I", hdr_size)
    return box(b"CMP1", b + bytes(24))


def craw_entry(w: int, h: int, cmp: bytes) -> bytes:
    body = bytearray(0x5A - 8)
    struct.pack_into(">HH", body, 0x18, w, h)
    return box(b"CRAW", bytes(body) + cmp)


def trak(entry: bytes, size: int, offset: int) -> bytes:
    stsd = box(b"stsd", struct.pack(">II", 0, 1) + entry)
    stsz = box(b"stsz", struct.pack(">III", 0, size, 1))
    co64 = box(b"co64", struct.pack(">IIQ", 0, 1, offset))
    return box(b"trak", box(b"mdia", box(b"minf", box(b"stbl", stsd + stsz + co64))))


def colour_data(wb=(2000, 1024, 1024, 1500), black=(511, 512, 513, 514)) -> bytes:
    cd = [0] * 0x200
    cd[0] = 16
    cd[0x47:0x4B] = wb
    cd[0x149:0x14D] = black
    return tiff([(0x4001, 3, cd)])


def build_cr3(mosaic: np.ndarray, tile_w: int, levels: int, orientation: int = 1, qp: bool = False,
              crop=(4, 6, 20, 30)) -> bytes:
    """A CR3 whose sensor data is ``mosaic`` (RGGB), coded as RAW (levels 0) or C-RAW."""
    h, w = mosaic.shape
    tiles_x = w // tile_w
    planes = [mosaic[y::2, x::2].astype(np.int64) - 8192 for y, x in ((0, 0), (0, 1), (1, 0), (1, 1))]
    if levels:
        per_plane = [crx_bands(p, tiles_x, levels) for p in planes]
        bands = [[per_plane[p][t] for p in range(4)] for t in range(tiles_x)]
    else:
        tw = tile_w // 2
        bands = [[[p[:, t * tw:(t + 1) * tw if t < tiles_x - 1 else None]] for p in planes] for t in range(tiles_x)]
    qps = None
    if qp:
        qw, qh = -(-(tile_w // 2) // 8), -(-(h // 2) // 2)
        rng = np.random.default_rng(5)
        qps = [encode_qp(rng.integers(-4, 3, (qh, qw))) for _ in range(tiles_x)]   # +4: steps of 0..6, all 1
    sample, hsz = crx_data(bands, qp=qps)
    left, top = crop[1], crop[0]
    info = [34, w, h, 1, 1, left, top, left + crop[3] - 1, top + crop[2] - 1, 0, 0, 0, 0]
    cmts = (box(b"CMT1", tiff([(271, 2, b"Canon"), (272, 2, b"Canon EOS Test"), (274, 3, [orientation])]))
            + box(b"CMT2", tiff([(33434, 5, [(1, 125)]), (34855, 3, [200])]))
            + box(b"CMT3", tiff([(0xE0, 3, info)]))
            + box(b"CMT4", tiff([(0, 1, [2, 3, 0, 0])])))
    uuid = box(b"uuid", cr3.CANON_UUID + box(b"CNCV", b"CanonCR3_001/00.09.00/00.00.00") + cmts)
    note = colour_data()
    ctmd = struct.pack("<IH", 20 + len(note), 8) + b"\x01\x01\x00\x01\xff\xff" + bytes(8) + note
    version = 0x200 if qp else 0x100

    def moov(off_raw: int, off_ctmd: int) -> bytes:
        entry = craw_entry(w, h, cmp1(w, h, tile_w, h, levels, hsz, version, tiles_x=tiles_x))
        return box(b"moov", uuid + trak(entry, len(sample), off_raw) + trak(box(b"CTMD", bytes(16)), len(ctmd), off_ctmd))

    ftyp = box(b"ftyp", b"crx \x00\x00\x00\x01crx isom")
    at = len(ftyp) + len(moov(0, 0)) + 8
    return ftyp + moov(at, at + len(sample)) + box(b"mdat", sample + ctmd)


def mosaic_for_test(h: int, w: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    m = 2000 + 40 * x + 25 * y + rng.integers(-30, 30, (h, w))
    m[: h // 4, : w // 3] = 512                          # flat: runs
    m[-3:, -5:] = 16383                                   # clipped
    m[h // 2, :] = rng.integers(0, 16384, w)              # wild values: escapes
    return np.clip(m, 0, 16383).astype(np.uint16)


# ---------------------------------------------------------------------------------------------
# the tests


def decoders():
    yield "python", None
    if NATIVE is not None:
        yield "native", NATIVE


def _decode_band(impl, data: bytes, w: int, h: int, predicted: bool) -> np.ndarray:
    if impl is None:
        br = cr3.Bits(data, 0, len(data))
        rows = (cr3._band_predicted if predicted else cr3._band_plain)(br, w, h)
        return cr3._int32([r[1:w + 1] for r in rows])
    return np.frombuffer(impl.crx_band(data, 0, len(data), w, h, predicted), "<i4").reshape(h, w)


def _band_cases():
    rng = np.random.default_rng(7)
    smooth = (np.add.outer(np.arange(12), np.arange(37)) * 3).astype(np.int64)
    sparse = rng.integers(-3, 4, (11, 29)) * (rng.random((11, 29)) < 0.25)
    wild = rng.integers(-(1 << 18), 1 << 18, (5, 17))      # differences still fit the 21-bit escape
    runs = np.zeros((9, 300), np.int64)
    runs[3, 150] = 7
    runs[5, -1] = -2
    runs[7, 0] = 1
    return [smooth, sparse, wild, runs, np.zeros((1, 1), np.int64), np.array([[5, -5]]), smooth[:, :1], sparse[:1]]


@pytest.mark.parametrize("name,impl", list(decoders()))
@pytest.mark.parametrize("predicted", [True, False])
def test_band_round_trip(name, impl, predicted):
    for a in _band_cases():
        data = (encode_predicted if predicted else encode_plain)(a)
        got = _decode_band(impl, data, a.shape[1], a.shape[0], predicted)
        assert np.array_equal(got, a), (name, a.shape)


@pytest.mark.parametrize("name,impl", list(decoders()))
def test_qp_round_trip(name, impl):
    a = np.random.default_rng(3).integers(-6, 9, (13, 21))
    data = encode_qp(a)
    if impl is None:
        got = _python_qp(data, 21, 13)
    else:
        got = np.frombuffer(impl.crx_qp(data, 0, len(data), 21, 13), "<i4").reshape(13, 21)
    assert np.array_equal(got, a)


def _python_qp(data, w, h):
    saved = _accel.native
    _accel.native = None
    try:
        return cr3.decode_qp(data, 0, len(data), w, h)
    finally:
        _accel.native = saved


def test_native_matches_python_on_noise():
    """Random bytes decode to the same values (or the same refusal) both ways."""
    if NATIVE is None:
        pytest.skip("pixport-native isn't installed")
    rnd = random.Random(11)
    for i in range(60):
        data = bytes(rnd.getrandbits(8) for _ in range(rnd.choice((8, 64, 400))))
        w, h, pred = rnd.randint(1, 40), rnd.randint(1, 6), bool(i & 1)
        try:
            py = _decode_band(None, data, w, h, pred)
        except ValueError:
            py = "error"
        try:
            nat = _decode_band(NATIVE, data, w, h, pred)
        except PixportError:
            nat = "error"
        assert (isinstance(py, str) and isinstance(nat, str)) or np.array_equal(py, nat)


def test_wavelet_inverts_forward():
    rng = np.random.default_rng(2)
    for n in (1, 2, 3, 4, 7, 8, 33):
        x = rng.integers(-9000, 9000, (n, 3))
        low, high = fwd(x, 0)
        assert np.array_equal(cr3.synth(low, high, n, False, False, 0), x)


def test_tile_lengths():
    # a tile with a neighbour on the right carries extra coefficients, low-pass lengths made odd
    assert [x[:3] for x in cr3._lengths(1572, False, True, 3)] == [(1572, 787, 787), (787, 395, 394), (395, 198, 198)]
    assert [x[:3] for x in cr3._lengths(1572, True, False, 3)] == [(1572, 786, 787), (786, 393, 394), (393, 197, 197)]
    assert [x[:3] for x in cr3._lengths(10, False, False, 2)] == [(10, 5, 5), (5, 3, 2)]


def test_qp_maps_and_steps():
    qp = np.array([[4, 10, 16], [5, 11, 40], [6, 12, 18]])
    l1, = cr3._qp_maps(qp, 1)
    assert l1.tolist() == [[1, 2, 4], [1, 2, 64], [1, 2, 5]]       # about 2 ** ((q - 4) / 6), floored
    maps = cr3._qp_maps(qp, 3)
    assert [m.shape for m in maps] == [(1, 3), (2, 3), (3, 3)]
    assert maps[1][0].tolist() == cr3._qstep(np.array([4, 10, 28])).tolist()  # rows averaged in pairs
    band = cr3.Band(q_base=1, q_mult=8)
    got = cr3._dequant_map(np.ones((2, 5), np.int32), band, np.array([[8, 16]]), shift=1, x0=0, y0=0, ownw=4)
    assert got.tolist() == [[9, 9, 17, 17, 17]] * 2


def _write(tmp_path, data: bytes, name="t.cr3"):
    p = tmp_path / name
    p.write_bytes(data)
    return p


def test_lossless_two_tiles(tmp_path):
    m = mosaic_for_test(32, 48)
    p = _write(tmp_path, build_cr3(m, 24, 0))
    raw = cr3.decode(p)
    assert np.array_equal(raw.cfa, m)
    assert raw.pattern.tolist() == [[0, 1], [1, 2]]
    assert raw.crop == (4, 6, 20, 30)
    assert np.allclose(raw.wb, (2000 / 1024, 1.0, 1500 / 1024))
    assert raw.black.tolist() == [[511, 512], [513, 514]]
    assert raw.white == 16383
    assert raw.metadata["model"] == "Canon EOS Test" and raw.metadata["iso"] == 200
    assert pixport.detect(p) == "cr3"


@pytest.mark.parametrize("levels", [1, 2, 3])
def test_craw_one_tile(tmp_path, levels):
    m = mosaic_for_test(40, 56, seed=levels)
    p = _write(tmp_path, build_cr3(m, 56, levels))
    assert np.array_equal(cr3.decode(p).cfa, m)


@pytest.mark.parametrize("width,tile", [(160, 80), (164, 82), (204, 68)])
def test_craw_tiles_side_by_side(tmp_path, width, tile):
    m = mosaic_for_test(48, width, seed=width)
    p = _write(tmp_path, build_cr3(m, tile, 3))
    assert np.array_equal(cr3.decode(p).cfa, m)


def test_craw_with_quantisation_map(tmp_path):
    m = mosaic_for_test(48, 80, seed=9)
    p = _write(tmp_path, build_cr3(m, 80, 3, qp=True))
    assert np.array_equal(cr3.decode(p).cfa, m)


def test_develops_upright(tmp_path):
    m = mosaic_for_test(32, 48)
    img = pixport.read(_write(tmp_path, build_cr3(m, 48, 0, orientation=6)), half=True)
    assert img.pixels.shape[:2] == (15, 10)              # crop 20 x 30, half size, turned
    assert img.format == "cr3"


def test_unsupported_coding(tmp_path):
    data = bytearray(build_cr3(mosaic_for_test(16, 16), 16, 0))
    at = data.find(b"CMP1") + 4
    data[at + 26] = 0x30                                  # coding type 3
    with pytest.raises(UnsupportedError):
        pixport.read(_write(tmp_path, bytes(data)))


def test_damaged_files(tmp_path):
    good = build_cr3(mosaic_for_test(32, 48), 24, 2)
    cases = [good[:200], good[:-300], good[:30] + b"\xff" * 4 + good[34:]]
    hdr = good.find(b"\xff\x01\x00\x08")
    cases.append(good[:hdr] + b"\xff\x05" + good[hdr + 2:])
    cmp = good.find(b"CMP1") + 4
    big = bytearray(good)
    struct.pack_into(">II", big, cmp + 8, 60000, 60000)
    cases.append(bytes(big))
    zeros = bytearray(good)
    start = good.find(b"mdat") + 4 + 300
    zeros[start:start + 2000] = bytes(2000)
    cases.append(bytes(zeros))
    for i, data in enumerate(cases):
        p = _write(tmp_path, data, f"bad{i}.cr3")
        try:
            pixport.read(p)
        except PixportError:
            continue
        # a damaged band may still decode to something; it must never crash
