"""16-bit PNG and TIFF writing: decoded again (by a small reader here, and by Pillow for 8 bits), the
pixels come back exactly."""

import struct
import zlib

import numpy as np
import pytest

import pixport
from pixport import png, tiffwrite


def unfilter(raw: bytes, h: int, stride: int, bpp: int) -> np.ndarray:
    """PNG rows back from their filtered form (all five filters, as the PNG specification gives
    them), written apart from pixport's own code."""
    rows = np.frombuffer(raw, np.uint8).reshape(h, stride + 1)
    out = np.zeros((h, stride), np.int64)
    for y in range(h):
        kind, f = rows[y, 0], rows[y, 1:].astype(np.int64)
        up = out[y - 1] if y else np.zeros(stride, np.int64)
        cur = out[y]
        for x in range(stride):
            a = cur[x - bpp] if x >= bpp else 0
            b, c = up[x], (up[x - bpp] if x >= bpp else 0)
            if kind == 0:
                pred = 0
            elif kind == 1:
                pred = a
            elif kind == 2:
                pred = b
            elif kind == 3:
                pred = (a + b) // 2
            else:
                p_ = a + b - c
                pa, pb, pc = abs(p_ - a), abs(p_ - b), abs(p_ - c)
                pred = a if pa <= pb and pa <= pc else b if pb <= pc else c
            cur[x] = (f[x] + pred) & 0xFF
    return out.astype(np.uint8)


def read_png(data: bytes) -> tuple[np.ndarray, dict]:
    """The picture in a PNG as (h, w, channels) uint8/uint16: palette pictures are expanded to RGB or
    RGBA, the way any reader shows them."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, chunks = 8, b"", {}
    while pos < len(data):
        n, kind = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + n]
        assert zlib.crc32(kind + body) & 0xFFFFFFFF == struct.unpack(">I", data[pos + 8 + n:pos + 12 + n])[0]
        if kind == b"IHDR":
            w, h, depth, ctype = struct.unpack(">IIBB", body[:10])
        elif kind == b"IDAT":
            idat += body
        else:
            chunks[kind] = body
        pos += 12 + n
    c = {0: 1, 4: 2, 2: 3, 6: 4, 3: 1}[ctype]
    stride = (w * c * depth + 7) // 8
    rows = unfilter(zlib.decompress(idat), h, stride, max(1, c * depth // 8))
    if ctype == 3:
        bits = np.unpackbits(rows, axis=1).reshape(h, -1, depth)[:, :w]
        index = (bits * (1 << np.arange(depth - 1, -1, -1))).sum(axis=2)
        pal = np.frombuffer(chunks[b"PLTE"], np.uint8).reshape(-1, 3)
        alpha = np.full(len(pal), 255, np.uint8)
        trns = chunks.get(b"tRNS", b"")
        alpha[:len(trns)] = np.frombuffer(trns, np.uint8)
        rgba = np.concatenate([pal, alpha[:, None]], axis=1)[index]
        return (rgba if b"tRNS" in chunks else rgba[:, :, :3]), chunks
    px = rows.view(">u2").astype(np.uint16) if depth == 16 else rows
    return px.reshape(h, w, c), chunks


def same_picture(got: np.ndarray, want: np.ndarray) -> bool:
    """got (as read back) shows want: 16 bits stored as 8 when they held only 8, grey stored as one
    channel, an opaque alpha left out."""
    want = want if want.ndim == 3 else want[:, :, None]
    if want.dtype == np.uint16 and got.dtype == np.uint8:
        got = got.astype(np.uint16) * 257
    top = 65535 if want.dtype == np.uint16 else 255
    if want.shape[2] in (2, 4) and got.shape[2] in (1, 3):
        got = np.concatenate([got, np.full(got.shape[:2] + (1,), top, want.dtype)], axis=2)
    if want.shape[2] in (3, 4) and got.shape[2] in (1, 2):
        got = np.concatenate([np.repeat(got[:, :, :1], 3, axis=2), got[:, :, 1:]], axis=2)
    return got.shape == want.shape and np.array_equal(got.astype(np.int64), want.astype(np.int64))


def read_tiff(data: bytes) -> np.ndarray:
    assert data[:4] == b"II*\0"
    ifd = struct.unpack("<I", data[4:8])[0]
    n = struct.unpack("<H", data[ifd:ifd + 2])[0]
    tags = {}
    for i in range(n):
        tag, typ, count, val = struct.unpack("<HHII", data[ifd + 2 + 12 * i:ifd + 14 + 12 * i])
        size = {2: 1, 3: 2, 4: 4, 7: 1}[typ] * count
        raw = data[ifd + 10 + 12 * i:ifd + 14 + 12 * i] if size <= 4 else data[val:val + size]
        fmt = {3: "H", 4: "I"}.get(typ)
        tags[tag] = list(struct.unpack(f"<{count}{fmt}", raw[:size])) if fmt else raw[:size]
    w, h, c, bits = tags[256][0], tags[257][0], tags[277][0], tags[258][0]
    dt = np.dtype("<u2") if bits == 16 else np.uint8
    body = b"".join(zlib.decompress(data[o:o + n]) for o, n in zip(tags[273], tags[279]))
    flat = np.frombuffer(body, dt).reshape(h, w * c).copy()
    for x in range(c, w * c):  # undo the horizontal predictor
        flat[:, x] = flat[:, x] + flat[:, x - c]
    return flat.reshape(h, w, c), tags


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_png_round_trip(channels, dtype):
    rng = np.random.default_rng(channels)
    px = rng.integers(0, np.iinfo(dtype).max + 1, size=(37, 53, channels), dtype=dtype)
    got, chunks = read_png(png.encode(px, exif=b"MM\0*\0\0\0\x08\0\0", icc=b"fake-icc"))
    assert np.array_equal(got, px)
    assert chunks[b"eXIf"].startswith(b"MM\0*")
    assert zlib.decompress(chunks[b"iCCP"].split(b"\0\0", 1)[1]) == b"fake-icc"


def test_8_bit_png_opens_in_pillow(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    px = np.random.default_rng(1).integers(0, 256, size=(20, 30, 4), dtype=np.uint8)
    (tmp_path / "a.png").write_bytes(png.encode(px))
    assert np.array_equal(np.asarray(Image.open(tmp_path / "a.png")), px)


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_tiff_round_trip(channels, dtype):
    rng = np.random.default_rng(7)
    px = rng.integers(0, np.iinfo(dtype).max + 1, size=(150, 41, channels), dtype=dtype)  # several strips
    got, tags = read_tiff(tiffwrite.encode(px, icc=b"icc!", text={271: "Canon", 272: "EOS R5"}))
    assert np.array_equal(got, px)
    assert tags[271] == b"Canon\0" and tags[34675] == b"icc!" and tags[274] == [1]


def test_write_picks_16_bits_for_high_bit_pictures(tmp_path):
    img = pixport.Image(np.full((4, 4, 3), 0.5, np.float32), transfer="linear")
    pixport.write(img, tmp_path / "a.png")
    px, _ = read_png((tmp_path / "a.png").read_bytes())
    assert px.dtype == np.uint16 and px[0, 0, 0] == round(pixport.image.linear_to_srgb(np.float32(0.5)) * 65535)
    pixport.write(pixport.Image(np.zeros((4, 4, 3), np.float32)), tmp_path / "b.tif")
    assert read_tiff((tmp_path / "b.tif").read_bytes())[0].dtype == np.uint8


def test_png_filters_are_the_same_both_ways():
    """pixport-native's Paeth filter gives the bytes NumPy's does; every filter decodes back."""
    from pixport import _accel
    rng = np.random.default_rng(7)
    rows = rng.integers(0, 256, (37, 6 * 41), dtype=np.uint8)
    rows[:, 60:120] = 9  # flat stretches too
    prev = rng.integers(0, 256, rows.shape[1], dtype=np.uint8)
    native = _accel.native
    try:
        got = []
        for mode in (native, None):
            _accel.native = mode
            got.append(png._apply(rows, prev, 6, 4))
        assert np.array_equal(got[0], got[1])
    finally:
        _accel.native = native
    for kind in (0, 1, 2, 4, None):
        data = png.filtered(rows, 6, kind)
        assert np.array_equal(unfilter(data, *rows.shape, 6), rows)


def test_every_row_can_take_its_own_filter():
    rng = np.random.default_rng(3)
    rows = rng.integers(0, 256, size=(40, 90), dtype=np.uint8)
    rows[10:20] = rows[9]  # Up suits these
    rows[20:30] = np.arange(90, dtype=np.uint8) * 3  # Sub suits these, Up after the first
    data = png.filtered(rows, 3, png.OWN)
    assert len(set(data[:: 91])) > 1
    assert np.array_equal(unfilter(data, 40, 90, 3), rows)


@pytest.mark.skipif(not png._accel.has("png_deflate"), reason="needs pixport-native")
@pytest.mark.parametrize("level", [1, 6, 10, 12])
@pytest.mark.parametrize("size", [0, 1, 100, (1 << 16) - 1, (1 << 20), (1 << 20) + 1, (3 << 20) + 5])
def test_libdeflate_parts_join_into_one_good_stream(level, size):
    rng = np.random.default_rng(size + level)
    kinds = {
        "noise": rng.integers(0, 256, size=size, dtype=np.uint8).tobytes(),  # stored blocks
        "flat": bytes(size),  # one symbol again and again
        "ramp": (np.arange(size, dtype=np.uint32) // 7 % 251).astype(np.uint8).tobytes(),
        "mixed": (rng.integers(0, 4, size=size, dtype=np.uint8) * (np.arange(size) // 4096 % 2).astype(np.uint8)).tobytes(),
    }
    for name, raw in kinds.items():
        for part in (4096, 1 << 20):
            for threads in (1, 4):
                packed = png._accel.native.png_deflate(raw, level, threads, part)
                assert zlib.decompress(packed) == raw, (name, part, threads)


def test_libdeflate_is_used_at_the_usual_level_only(monkeypatch):
    raw = bytes(range(256)) * 16
    calls = []
    if png._accel.has("png_deflate"):
        real = png._accel.native.png_deflate
        monkeypatch.setattr(png._accel.native, "png_deflate", lambda *a: calls.append(a[1:]) or real(*a))
    assert zlib.decompress(png.deflate(raw, 3)) == raw
    assert not calls
    assert zlib.decompress(png.deflate(raw)) == raw
    assert calls == ([(png.PACK, png.threads(), png.PART)] if png._accel.has("png_deflate") else [])


@pytest.mark.skipif(not png._accel.has("png_deflate"), reason="needs pixport-native")
def test_libdeflate_level_suits_the_picture(monkeypatch):
    rng = np.random.default_rng(5)
    photo = rng.normal(128, 40, size=1 << 21).clip(0, 255).astype(np.uint8).tobytes()
    blank = (bytes(1000) + b"pattern of some sort" * 20) * 2000
    assert png._pack_level(photo[:1000]) == png.PACK  # small, so it can take its time
    assert png._pack_level(blank[:1000]) == png.PACK
    monkeypatch.setattr(png, "SMALL", 1000)
    assert png._pack_level(blank) == 6
    assert png._pack_level(photo) == png.PACK  # between SMALL and PHOTO
    monkeypatch.setattr(png, "PHOTO", 1000)
    assert png._pack_level(photo) == 1
    assert png._pack_level(blank) == 6
    for raw in (photo, blank):
        assert zlib.decompress(png.deflate(raw)) == raw


def test_photo_sized_pictures_do_not_weigh_each_rows_own_filter(monkeypatch):
    rows = np.random.default_rng(4).integers(0, 256, size=(40, 90), dtype=np.uint8)
    calls = []
    real = png._by_row
    monkeypatch.setattr(png, "_by_row", lambda *a: calls.append(1) or real(*a))
    png.filtered(rows, 3)
    assert calls  # small: it is weighed
    calls.clear()
    monkeypatch.setattr(png, "PHOTO", rows.size)
    data = png.filtered(rows, 3)
    assert not calls
    assert np.array_equal(unfilter(data, 40, 90, 3), rows)


@pytest.mark.parametrize("size", [(3 << 20) + 5, (9 << 20) + 1])
def test_packing_in_parts_is_one_good_stream(size):
    raw = (np.random.default_rng(size).integers(0, 9, size, dtype=np.uint8)).tobytes()
    packed = png.deflate(raw)
    assert zlib.decompress(packed) == raw  # checks the joined Adler-32 too
    whole = zlib.compress(raw, png.LEVEL)
    assert len(packed) < 1.002 * len(whole)  # each part can refer back into the one before


def test_small_ramps_pack_as_well_as_a_row_at_a_time_can():
    ramp = np.tile(np.arange(400, dtype=np.uint16) * 160, (400, 1))[:, :, None]  # a sideways ramp, 16 bits
    assert np.array_equal(read_png(png.encode(ramp))[0], ramp)
    assert len(png.encode(ramp)) < 4000


def test_a_blank_band_stays_blank_and_small():
    px = np.zeros((2000, 1000, 3), np.uint16)
    px[1500:1510, 10:20] = (40000, 20000, 10000)
    data = png.encode(px)
    assert np.array_equal(read_png(data)[0], px)
    assert len(data) < 25000


@pytest.mark.parametrize("name", ["plain", "not_a_number"])
def test_threaded_brightness_curve_matches_one_thread(monkeypatch, name):
    rng = np.random.default_rng(5)
    px = rng.random((500, 40, 3), np.float32) * 4
    if name == "not_a_number":
        px[100, 3] = np.nan
        px[300, 5] = np.inf
    px[200:300] = 0  # a blank band among them
    img = pixport.Image(px, "linear", {})
    many = img.to_srgb(bits=16)
    monkeypatch.setenv("PIXPORT_THREADS", "1")
    assert np.array_equal(many, img.to_srgb(bits=16))
    assert not many[200:300].any() and many[:64].any()


@pytest.mark.parametrize("name", ["two colours", "palette 200", "grey in rgb", "opaque rgba", "16 holding 8", "gradient"])
def test_png_is_stored_small_and_reads_back_the_same(name):
    """Reductions a picture allows (palette, grey, no alpha, 8 bits) never change a pixel."""
    rng = np.random.default_rng(3)
    y, x = np.mgrid[0:40, 0:53]
    a = {
        "two colours": np.where(rng.random((40, 53, 1)) > .5, [10, 20, 30, 255], [200, 100, 0, 0]),
        "palette 200": np.dstack([rng.integers(0, 200, (40, 53)), np.full((40, 53), 3), np.full((40, 53), 4)]),
        "grey in rgb": np.repeat(rng.integers(0, 256, (40, 53, 1)), 3, axis=2),
        "opaque rgba": np.dstack([rng.integers(0, 256, (40, 53, 3)), np.full((40, 53), 255)]),
        "16 holding 8": rng.integers(0, 256, (40, 53, 3)) * 257,
        "gradient": np.dstack([y * 1000, x * 700, (x * y) % 65536]),
    }[name]
    a = a.astype(np.uint16 if a.max() > 255 else np.uint8)
    got, chunks = read_png(png.encode(a))
    assert same_picture(got, a)
    if name in ("two colours", "palette 200"):
        assert b"PLTE" in chunks


def test_pixels_that_are_not_numbers_become_black_or_white():
    """OpenEXR files can hold NaN and infinity; the suite turns any NumPy warning into a failure."""
    px = np.array([[[np.nan, np.inf, -np.inf], [0.5, 2.0, -1.0]]], dtype=np.float32)
    for transfer, exposure in (("linear", 0), ("linear", 1), ("srgb", 0)):
        out = pixport.Image(px, transfer, {}).to_srgb(bits=8, **({"exposure": exposure} if exposure else {}))
        assert out[0, 0].tolist() == [0, 255, 0]
    assert pixport.Image(px, "linear", {}).to_srgb(bits=8, tonemap="reinhard")[0, 0].tolist() == [0, 255, 0]


def test_exr_half_keeps_a_very_bright_pixel_bright(tmp_path):
    """Above 65504 a half would round to infinity (and NumPy warns, which fails a test here)."""
    px = np.array([[[1e6, 2.0, np.inf]]], dtype=np.float32)
    pixport.write(pixport.Image(px, "linear", {}), tmp_path / "a.exr", "exr")
    back = pixport.read(tmp_path / "a.exr").pixels[0, 0]
    assert back[0] == 65504.0 and back[1] == 2.0 and np.isinf(back[2])
