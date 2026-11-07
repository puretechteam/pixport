"""RAW decoders on files built here: lossless JPEG with every predictor and 1-4 components (and
Canon's sRAW luma runs), CR2 slices, DNGs (strips, tiles, lossless JPEG, linear, levels, crop and
colour matrices); the makers' bit-serial codecs give the same values through Python and
pixport-native on arbitrary input; damaged and hostile files end in PixportError."""

from __future__ import annotations

import random
import struct

import numpy as np
import pytest
from raw_writer import Tiff, ljpeg_encode
from raw_writer import dng as make_dng

import pixport
from pixport import _accel
from pixport.errors import PixportError
from pixport.raw import common, dng, ljpeg, nikon, olympus, panasonic, pentax, sony

NATIVE = _accel.native if _accel.has("ljpeg_decode") else None


@pytest.fixture
def pure(monkeypatch):
    """Run the Python decoders even when pixport-native is installed."""
    monkeypatch.setattr(_accel, "native", None)


def _smooth(rows: int, width: int, top: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:rows, 0:width]
    base = (np.sin(x / 7.0) + np.cos(y / 5.0) + 2) * top / 4
    return np.clip(base + rng.normal(0, top / 50, base.shape), 0, top).astype(np.uint16)


@pytest.mark.parametrize("predictor", range(1, 8))
@pytest.mark.parametrize("ncomp", [1, 2, 3, 4])
def test_ljpeg_round_trip(predictor, ncomp):
    prec = 14 if ncomp % 2 else 16
    samples = _smooth(9, 6 * ncomp, (1 << prec) - 1, predictor * 10 + ncomp)
    data = ljpeg_encode(samples, ncomp, predictor, prec)
    out, f = ljpeg.decode(data)
    assert f.predictor == predictor and len(f.comps) == ncomp
    assert np.array_equal(out, samples)


def test_ljpeg_python_matches_native(pure):
    samples = _smooth(16, 40, 4095, 3)
    data = ljpeg_encode(samples, 2, 1, 12)
    out_py, _ = ljpeg.decode(data)
    assert np.array_equal(out_py, samples)
    if NATIVE is not None:
        f = ljpeg.parse(data)
        tabs = np.stack([f.tables[f.scan_tables[c]] for c in range(2)])
        nat = np.asarray(NATIVE.ljpeg_decode(data, f.scan_start, len(data), tabs, np.array([0, 1], np.uint8), 16, 40, 1, 12, 0, 0))
        assert np.array_equal(nat.reshape(16, 40), samples)


@pytest.mark.parametrize("sampling,chain", [([(2, 1), (1, 1), (1, 1)], 2), ([(2, 2), (1, 1), (1, 1)], 4)])
def test_canon_sraw_luma_runs(sampling, chain):
    slots = chain + 2
    s = _smooth(6, 5 * slots, 16383, chain)
    s[:, chain::slots] = 16384 + (s[:, chain::slots] % 300)
    s[:, chain + 1::slots] = 16384 - (s[:, chain + 1::slots] % 300)
    data = ljpeg_encode(s, 3, 1, 15, sampling=sampling, chain=chain)
    out, f = ljpeg.decode(data, chain=True)
    assert f.slots == [0] * chain + [1, 2]
    assert np.array_equal(out, s)


def test_ljpeg_damage():
    samples = _smooth(8, 16, 4095, 1)
    data = ljpeg_encode(samples, 2, 1, 12)
    with pytest.raises(PixportError):
        ljpeg.decode(data[:len(data) // 2] + b"\xff\xd9")
    with pytest.raises(PixportError):
        ljpeg.decode(b"\xff\xd8\xff\xc3\x00\x0b\x10\xff\xff\xff\xff\x01\x01\x11\x00")
    with pytest.raises(PixportError):
        ljpeg.decode(b"not a jpeg")


def test_cr2_slices():
    flat = np.arange(4 * 10, dtype=np.uint16)
    out = ljpeg.unslice(flat, 4, 10, (2, 3, 4))
    # slice 0: columns 0-2 top to bottom, slice 1: 3-5, last: 6-9
    assert out[:, 0:3].ravel().tolist() == list(range(12))
    assert out[:, 3:6].ravel().tolist() == list(range(12, 24))
    assert out[:, 6:10].ravel().tolist() == list(range(24, 40))
    with pytest.raises(PixportError):
        ljpeg.unslice(flat, 4, 11, (2, 3, 4))


@pytest.mark.parametrize("bits", [8, 10, 12, 14, 16])
@pytest.mark.parametrize("order", ["msb", "lsb"])
def test_unpack_bits(bits, order):
    rng = np.random.default_rng(bits)
    vals = rng.integers(0, 1 << bits, 64)
    s = "".join(format(int(v), f"0{bits}b") if order == "msb" else format(int(v), f"0{bits}b")[::-1] for v in vals)
    s += "0" * (-len(s) % 8)
    chunks = [s[i:i + 8] for i in range(0, len(s), 8)]
    buf = bytes(int(c, 2) if order == "msb" else int(c[::-1], 2) for c in chunks)
    assert common.unpack_bits(buf, 64, bits, order).tolist() == vals.tolist()


# ---------------------------------------------------------------------------------------------
# DNG


DNG_TAGS0 = [(50721, 10, [(10000, 10000), (-3000, 10000), (-500, 10000), (-4000, 10000), (12000, 10000), (2000, 10000),
                          (-500, 10000), (1500, 10000), (6000, 10000)]),
             (50778, 3, [21]), (50728, 5, [(5, 10), (1, 1), (7, 10)])]


def _cfa(h=24, w=32, seed=0):
    return _smooth(h, w, 4000, seed) + 100


@pytest.mark.parametrize("order", ["<", ">"])
@pytest.mark.parametrize("compression,tile", [(1, None), (1, 16), (7, None), (7, 16)])
def test_dng_mosaic(tmp_path, order, compression, tile):
    cfa = _cfa()
    extra = [(50714, 3, [100]), (50717, 3, [4095]), (50829, 4, [2, 2, 22, 30]), (50719, 4, [2, 2]), (50720, 4, [24, 16])]
    data = make_dng(cfa, order, compression, tile, extra_raw=extra, extra0=DNG_TAGS0)
    p = tmp_path / "t.dng"
    p.write_bytes(data)
    raw = dng.decode(p)
    assert np.array_equal(raw.cfa, cfa[2:22, 2:30])         # the active area
    assert raw.crop == (2, 2, 16, 24)
    assert float(np.max(raw.black)) == 100 and raw.white == 4095
    assert raw.wb == pytest.approx((2.0, 1.0, 1 / 0.7))
    assert raw.cam_xyz is not None and raw.metadata["colour"].startswith("matrix")
    img = pixport.read(p)
    assert img.pixels.shape == (16, 24, 3) and img.metadata["make"] == "Pixtest"
    assert np.isfinite(img.pixels).all()


def test_dng_linear(tmp_path):
    rgb = np.stack([_cfa(16, 16, s) for s in range(3)], axis=2)
    data = make_dng(rgb, "<", 7, None, extra_raw=[(50717, 3, [4095])], extra0=DNG_TAGS0, linear=True)
    p = tmp_path / "lin.dng"
    p.write_bytes(data)
    raw = dng.decode(p)
    assert raw.cfa is None and np.array_equal(raw.linear, rgb)
    assert pixport.read(p).pixels.shape == (16, 16, 3)


def test_dng_linearization_and_levels(tmp_path):
    cfa = (np.arange(16 * 16, dtype=np.uint16).reshape(16, 16) % 256)
    table = [min(65535, v * 3) for v in range(256)]
    extra = [(50712, 3, table), (50713, 3, [2, 2]), (50714, 3, [1, 2, 3, 4]), (50716, 11, [0.5] * 16)]
    p = tmp_path / "lin.dng"
    p.write_bytes(make_dng(cfa, "<", 1, None, bits=16, extra_raw=extra))
    raw = dng.decode(p)
    assert np.array_equal(raw.cfa, cfa * 3)
    assert raw.black.tolist() == [[1, 2], [3, 4]]
    assert raw.black_rows is not None and raw.black_rows.tolist() == [0.5] * 16


def test_dng_matrix_interpolation():
    from pixport.raw.tiff import Tiff as Reader
    a = [(10000, 10000), (0, 1), (0, 1), (0, 1), (10000, 10000), (0, 1), (0, 1), (0, 1), (10000, 10000)]
    b = [(20000, 10000), (0, 1), (0, 1), (0, 1), (20000, 10000), (0, 1), (0, 1), (0, 1), (20000, 10000)]
    t = Reader(Tiff("<").build([{"entries": [(50721, 10, a), (50722, 10, b), (50778, 3, [17]), (50779, 3, [21])]}]))
    neutral = np.array([1.0, 1.0, 1.0])
    m = dng.colour_matrix(t.ifds[0], neutral)
    # a neutral camera white reads as illuminant E (~5455 K): between StdA (2856 K) and D65
    w = (1 / 5455 - 1 / 6504) / (1 / 2856 - 1 / 6504)
    assert m[0, 0] == pytest.approx(w * 1 + (1 - w) * 2, abs=0.02)


def test_dng_damage(tmp_path):
    data = make_dng(_cfa(), "<", 7, 16, extra0=DNG_TAGS0)
    p = tmp_path / "bad.dng"
    for cut in (100, len(data) // 2, len(data) - 30):
        p.write_bytes(data[:cut])
        with pytest.raises(PixportError):
            pixport.read(p)
    rng = random.Random(5)
    for _ in range(30):
        bad = bytearray(data)
        for _ in range(8):
            bad[rng.randrange(8, len(bad))] = rng.randrange(256)
        p.write_bytes(bytes(bad))
        try:
            pixport.read(p)
        except PixportError:
            pass


@pytest.mark.parametrize("ext", ["dng", "cr2", "nef", "arw", "orf", "rw2", "pef"])
def test_garbage_is_an_error(tmp_path, ext):
    rng = random.Random(ext)
    p = tmp_path / f"x.{ext}"
    for head in (b"II*\0\x08\0\0\0", b"MM\0*\0\0\0\x08", b"IIRO\x08\0\0\0", b"IIU\0\x08\0\0\0", b"junk"):
        p.write_bytes(head + bytes(rng.randrange(256) for _ in range(3000)))
        with pytest.raises(PixportError):
            pixport.read(p)


# ---------------------------------------------------------------------------------------------
# the makers' codecs: Python and pixport-native agree value for value on any input


def _random(n: int, seed: int) -> bytes:
    rng = random.Random(seed)
    return bytes(rng.randrange(256) for _ in range(n))


def _both(fn_py, fn_nat):
    try:
        a = fn_py()
    except PixportError as e:
        a = e
    try:
        b = fn_nat()
    except PixportError as e:
        b = e
    if isinstance(a, Exception) or isinstance(b, Exception):
        assert type(a) is type(b), (a, b)
    else:
        assert np.array_equal(np.asarray(a), np.asarray(b))
    return a


needs_native = pytest.mark.skipif(not _accel.has("olympus_decode"), reason="pixport-native isn't installed")


@needs_native
@pytest.mark.parametrize("seed", range(4))
def test_twins_olympus(monkeypatch, seed):
    data = _random(4000, seed)
    nat = _accel.native

    def py():
        monkeypatch.setattr(_accel, "native", None)
        try:
            return olympus.decode_compressed(data, 0, 40, 24)
        finally:
            monkeypatch.setattr(_accel, "native", nat)

    _both(py, lambda: olympus.decode_compressed(data, 0, 40, 24))


@needs_native
@pytest.mark.parametrize("seed", range(4))
def test_twins_panasonic(monkeypatch, seed):
    data = _random(0x4000 * 2, seed)
    nat = _accel.native

    def py():
        monkeypatch.setattr(_accel, "native", None)
        try:
            return panasonic._v4(data, 0, 56, 30)
        finally:
            monkeypatch.setattr(_accel, "native", nat)

    _both(py, lambda: panasonic._v4(data, 0, 56, 30))


@needs_native
@pytest.mark.parametrize("seed", range(4))
def test_twins_pentax_nikon_sony(monkeypatch, seed):
    nat = _accel.native
    data = _random(6000, seed)

    def run(fn):
        monkeypatch.setattr(_accel, "native", None)
        try:
            return fn()
        finally:
            monkeypatch.setattr(_accel, "native", nat)

    lut = pentax.table(None, ">")
    _both(lambda: run(lambda: pentax._decode(data, 0, 32, 20, lut)), lambda: pentax._decode(data, 0, 32, 20, lut))
    meta = bytes([0x44, 0x20]) + struct.pack(">4h", 100, 200, 300, 400) + struct.pack(">H", 257) \
        + b"".join(struct.pack(">H", i * 16) for i in range(257)) + b"\0" * 400 + struct.pack(">H", 5)
    meta = meta[:562] + struct.pack(">H", 5) + meta[564:]
    _both(lambda: run(lambda: nikon.decode_compressed(data, 0, 32, 20, meta, ">", 12)[0]),
          lambda: nikon.decode_compressed(data, 0, 32, 20, meta, ">", 12)[0])
    curve = sony.sony_curve([8000, 10400, 12900, 14100])
    _both(lambda: run(lambda: sony.arw2(data, 0, 96, 20, curve)), lambda: sony.arw2(data, 0, 96, 20, curve))


def test_sony_decrypt_is_an_involution():
    plain = _random(400, 9)
    assert sony.decrypt(sony.decrypt(plain, 0x12345678), 0x12345678) == plain
    assert sony.decrypt(plain, 1) != plain


def test_padded_columns_and_masked_edges():
    img = _smooth(200, 300, 4000, 2) + 500
    img[:, -6:] = np.tile(img[:, -8:-6], 3)
    assert common.padded_columns(img) == 6
    img2 = np.full((400, 400), 1000, np.uint16)
    img2[:, :8] = 64
    assert common.masked_edges(img2, 64, 4095) == (0, 8, 400, 392)
