"""OpenEXR: every compression (against what the OpenEXR library decodes), the file layouts
(scanline, tiled, multi-part, windows, subsampling), the main-picture rules, writing, damage."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

import pixport
from pixport import _accel, exr
from pixport.errors import PixportError, UnsupportedError
from pixport.image import Image

DATA = Path(__file__).parent / "data" / "exr"
CODECS = ["rle", "zips", "zip", "piz", "pxr24", "b44", "b44a", "dwaa", "dwab"]


# ---------------------------------------------------------------- a small EXR writer for test inputs

def _attr(name: str, typ: str, value: bytes) -> bytes:
    return name.encode() + b"\0" + typ.encode() + b"\0" + struct.pack("<i", len(value)) + value


def _chlist(chans) -> bytes:
    return b"".join(n.encode() + b"\0" + struct.pack("<iB3xii", t, plin, xs, ys) for n, t, plin, xs, ys in chans) + b"\0"


def _rle(buf: bytes) -> bytes:
    out = bytearray()
    i = 0
    while i < len(buf):
        j = i + 1
        while j < len(buf) and buf[j] == buf[i] and j - i < 128:
            j += 1
        if j - i >= 3:
            out += bytes((j - i - 1, buf[i]))
            i = j
            continue
        k = min(127, len(buf) - i)
        out += bytes((256 - k,)) + buf[i:i + k]
        i += k
    return bytes(out)


def _pxr24(lines: list[list[tuple[int, np.ndarray]]]) -> bytes:
    out = bytearray()
    for line in lines:
        for t, row in line:
            if t == exr.HALF:
                v = row.view(np.uint16).astype(np.int64)
                d = (v - np.concatenate([[0], v[:-1]])) & 0xFFFF
                out += (d >> 8).astype(np.uint8).tobytes() + (d & 255).astype(np.uint8).tobytes()
            elif t == exr.UINT:
                v = row.view(np.uint32).astype(np.int64)
                d = (v - np.concatenate([[0], v[:-1]])) & 0xFFFFFFFF
                out += b"".join(((d >> s) & 255).astype(np.uint8).tobytes() for s in (24, 16, 8, 0))
            else:
                v = row.view(np.uint32).astype(np.int64) >> 8  # truncated to 24 bits
                d = (v - np.concatenate([[0], v[:-1]])) & 0xFFFFFF
                out += b"".join(((d >> s) & 255).astype(np.uint8).tobytes() for s in (16, 8, 0))
    return zlib.compress(bytes(out))


def make_exr(channels: dict, compression: int = exr.NONE, dw=None, disp=None, attrs: bytes = b"",
             line_order: int = 0, tile=None, levels: int = 1) -> bytes:
    """channels: {name: array (ny, nx) or (array, xs, ys, plinear)}; arrays are the stored samples."""
    items = []
    for name in sorted(channels):
        v = channels[name]
        arr, xs, ys, plin = (v, 1, 1, 0) if isinstance(v, np.ndarray) else v
        t = {np.dtype("float16"): exr.HALF, np.dtype("float32"): exr.FLOAT, np.dtype("uint32"): exr.UINT}[arr.dtype]
        items.append((name, t, plin, xs, ys, arr))
    first = items[0][5]
    if dw is None:
        dw = (0, 0, first.shape[1] * items[0][3] - 1, first.shape[0] * items[0][4] - 1)
    disp = disp or dw
    x0, y0, x1, y1 = dw
    head = (_attr("channels", "chlist", _chlist([i[:5] for i in items]))
            + _attr("compression", "compression", bytes([compression]))
            + _attr("dataWindow", "box2i", struct.pack("<4i", *dw))
            + _attr("displayWindow", "box2i", struct.pack("<4i", *disp))
            + _attr("lineOrder", "lineOrder", bytes([line_order]))
            + _attr("pixelAspectRatio", "float", struct.pack("<f", 1.0))
            + _attr("screenWindowCenter", "v2f", struct.pack("<2f", 0, 0))
            + _attr("screenWindowWidth", "float", struct.pack("<f", 1.0)) + attrs)
    flags = 2
    chunks = []
    if tile is None:
        lines = {exr.NONE: 1, exr.RLE: 1, exr.ZIPS: 1, exr.ZIP: 16, exr.PIZ: 32, exr.PXR24: 16}[compression]
        for by in range(y0, y1 + 1, lines):
            by1 = min(by + lines - 1, y1)
            raw = bytearray()
            per_line = []
            for y in range(by, by1 + 1):
                line = []
                for name, t, plin, xs, ys, arr in items:
                    if y % ys == 0:
                        row = arr[exr._count(y0, y - 1, ys)]
                        raw += row.tobytes()
                        line.append((t, row))
                per_line.append(line)
            raw = bytes(raw)
            if compression == exr.NONE:
                body = raw
            elif compression in (exr.ZIP, exr.ZIPS):
                body = zlib.compress(exr._predict(raw))
            elif compression == exr.RLE:
                body = _rle(exr._predict(raw))
            elif compression == exr.PXR24:
                body = _pxr24(per_line)
            else:
                part = exr.Part({}, [exr.Channel(i[0], i[1], False, i[3], i[4]) for i in items], dw, disp, compression,
                                "scanlineimage")
                blk = exr._block(part, x0, by, x1, by1)
                planes = [arr[exr._count(y0, by - 1, ys):exr._count(y0, by1, ys)] for _n, _t, _p, _xs, ys, arr in items]
                body = exr.piz_encode(planes, blk)
            if len(body) >= len(raw):
                body = raw
            chunks.append(struct.pack("<ii", by, len(body)) + body)
        if line_order == 1:
            order = list(range(len(chunks)))[::-1]
        else:
            order = list(range(len(chunks)))
    else:
        flags |= 0x200
        tw, th = tile
        mode = 0 if levels == 1 else 1
        head += _attr("tiles", "tiledesc", struct.pack("<IIB", tw, th, mode))
        ntx, nty = -(-(x1 - x0 + 1) // tw), -(-(y1 - y0 + 1) // th)
        for ty in range(nty):
            for tx in range(ntx):
                ax0, ay0 = tx * tw, ty * th
                rows = []
                for yy in range(ay0, min(ay0 + th, y1 - y0 + 1)):
                    for *_r, arr in items:
                        rows.append(arr[yy, ax0:ax0 + tw].tobytes())
                raw = b"".join(rows)
                body = zlib.compress(exr._predict(raw)) if compression == exr.ZIP else raw
                if len(body) >= len(raw):
                    body = raw
                chunks.append(struct.pack("<5i", tx, ty, 0, 0, len(body)) + body)
        # lower mipmap levels: one dummy 1x1 tile each, never read
        for lv in range(1, levels):
            raw = b"".join(arr[:1, :1].tobytes() for *_r, arr in items)
            chunks.append(struct.pack("<5i", 0, 0, lv, lv, len(raw)) + raw)
        order = list(range(len(chunks)))
    head = b"\x76\x2f\x31\x01" + struct.pack("<I", flags) + head + b"\0"
    at = len(head) + 8 * len(chunks)
    offsets = [0] * len(chunks)
    body = b""
    for i in order:
        offsets[i] = at + len(body)
        body += chunks[i]
    return head + struct.pack(f"<{len(chunks)}Q", *offsets) + body


def make_multipart(parts: list[tuple[str, dict]]) -> bytes:
    """Single-line NONE-compressed parts: [(name, {channel: array})]."""
    heads, all_chunks = [], []
    for name, chans in parts:
        items = sorted(chans.items())
        h, w = items[0][1].shape
        head = (_attr("channels", "chlist", _chlist([(n, {2: exr.HALF, 4: exr.FLOAT}[a.dtype.itemsize], 0, 1, 1)
                                                     for n, a in items]))
                + _attr("compression", "compression", b"\0")
                + _attr("dataWindow", "box2i", struct.pack("<4i", 0, 0, w - 1, h - 1))
                + _attr("displayWindow", "box2i", struct.pack("<4i", 0, 0, w - 1, h - 1))
                + _attr("lineOrder", "lineOrder", b"\0")
                + _attr("pixelAspectRatio", "float", struct.pack("<f", 1.0))
                + _attr("screenWindowCenter", "v2f", struct.pack("<2f", 0, 0))
                + _attr("screenWindowWidth", "float", struct.pack("<f", 1.0))
                + _attr("name", "string", name.encode()) + _attr("type", "string", b"scanlineimage")
                + _attr("chunkCount", "int", struct.pack("<i", h)) + b"\0")
        heads.append(head)
        all_chunks.append([b"".join(a[y].tobytes() for _n, a in items) for y in range(h)])
    head = b"\x76\x2f\x31\x01" + struct.pack("<I", 2 | 0x1000) + b"".join(heads) + b"\0"
    n = sum(len(c) for c in all_chunks)
    at = len(head) + 8 * n
    offsets, body = [], b""
    for pi, chunks in enumerate(all_chunks):
        for y, raw in enumerate(chunks):
            offsets.append(at + len(body))
            body += struct.pack("<iii", pi, y, len(raw)) + raw
    return head + struct.pack(f"<{n}Q", *offsets) + body


def _pic(h=21, w=35, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    return {
        "R": (xx / w * 2).astype(np.float16), "G": (yy / h).astype(np.float16),
        "B": rng.random((h, w)).astype(np.float16), "Z": (xx * 3.5 + yy).astype(np.float32),
        "id": (xx * 100 + yy).astype(np.uint32),
    }


def _bits(a):
    return np.ascontiguousarray(a).view(np.uint16 if a.dtype.itemsize == 2 else np.uint32)


# ---------------------------------------------------------------- compressions

@pytest.mark.parametrize("codec", CODECS + ["tiled_piz"])
def test_codecs_match_openexr(codec):
    ref = np.load(DATA / "exr_expected.npz")
    name = codec if codec == "tiled_piz" else f"codec_{codec}"
    ours, _part = exr.read_channels(DATA / f"{name}.exr")
    keys = [k for k in ref.files if k.startswith(codec + "/")]
    assert keys
    for k in keys:
        ch = k.split("/", 1)[1]
        assert np.array_equal(_bits(ours[ch]), ref[k]), k


@pytest.mark.parametrize("comp", [exr.NONE, exr.RLE, exr.ZIPS, exr.ZIP, exr.PIZ, exr.PXR24])
def test_hand_built_compressions(tmp_path, comp):
    pic = _pic()
    p = tmp_path / "c.exr"
    p.write_bytes(make_exr(pic, comp))
    ours, _part = exr.read_channels(p)
    for k, v in pic.items():
        want = v
        if comp == exr.PXR24 and v.dtype == np.float32:
            want = (v.view(np.uint32) & 0xFFFFFF00).view(np.float32)
        assert np.array_equal(_bits(ours[k]), _bits(want)), k


def test_huffman_round_trip():
    rng = np.random.default_rng(4)
    cases = [
        np.zeros(1, np.uint16), np.full(5000, 7, np.uint16), np.array([1, 2] * 300 + [9] * 2000, np.uint16),
        rng.integers(0, 65536, 3000).astype(np.uint16), (rng.geometric(0.05, 20000) % 400).astype(np.uint16),
    ]
    for v in cases:
        enc = exr.huf_encode(v)
        assert np.array_equal(exr.huf_decode(enc, v.size), v)


@pytest.mark.parametrize("maxval", [100, 40000])
def test_wavelet_round_trip(maxval):
    rng = np.random.default_rng(5)
    for shape in [(1, 1), (1, 7), (5, 1), (13, 29), (32, 32), (33, 64)]:
        a = rng.integers(0, maxval + 1, shape).astype(np.uint16)
        b = a.copy()
        exr.wav_encode(b, maxval)
        exr.wav_decode(b, maxval)
        assert np.array_equal(a, b), shape


# ---------------------------------------------------------------- layouts and the main picture

def test_windows_and_line_order(tmp_path):
    pic = {"R": np.full((4, 6), 1.0, np.float16), "G": np.full((4, 6), 0.5, np.float16),
           "B": np.full((4, 6), 0.25, np.float16)}
    p = tmp_path / "w.exr"
    p.write_bytes(make_exr(pic, exr.ZIPS, dw=(-2, 3, 3, 6), disp=(0, 0, 9, 7), line_order=1))
    img = pixport.read(p)
    assert img.pixels.shape == (8, 10, 3)
    assert np.all(img.pixels[3:7, 0:4] == np.array([1.0, 0.5, 0.25], np.float32))
    assert np.all(img.pixels[:3] == 0) and np.all(img.pixels[:, 4:] == 0)
    assert img.metadata["exr_data_window"] == (-2, 3, 3, 6)


def test_tiled_with_mipmaps_reads_the_top_level(tmp_path):
    pic = _pic(37, 50)
    p = tmp_path / "t.exr"
    p.write_bytes(make_exr(pic, exr.ZIP, tile=(16, 8), levels=6))
    ours, _ = exr.read_channels(p)
    for k, v in pic.items():
        assert np.array_equal(_bits(ours[k]), _bits(v)), k


def test_luminance_chroma(tmp_path):
    h, w = 12, 16
    y = np.full((h, w), 0.5, np.float16)
    ry = np.full((h // 2, w // 2), 0.2, np.float16)
    by = np.full((h // 2, w // 2), -0.1, np.float16)
    p = tmp_path / "yc.exr"
    p.write_bytes(make_exr({"Y": y, "RY": (ry, 2, 2, 0), "BY": (by, 2, 2, 0)}, exr.ZIP))
    img = pixport.read(p)
    assert img.pixels.shape == (h, w, 3)
    r, g, b = 0.6, None, 0.45
    g = (0.5 - r * 0.2126 - b * 0.0722) / 0.7152
    assert np.allclose(img.pixels, np.array([r, g, b], np.float32), atol=2e-3)
    p.write_bytes(make_exr({"Y": y}, exr.ZIP))
    grey = pixport.read(p)
    assert grey.pixels.shape == (h, w, 1) and np.allclose(grey.pixels, 0.5)


def test_chromaticities_convert_to_srgb(tmp_path):
    pic = {"R": np.full((2, 3), 0.2, np.float16), "G": np.full((2, 3), 0.4, np.float16),
           "B": np.full((2, 3), 0.1, np.float16)}
    m709 = exr.rgb_to_xyz((0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290))
    xyz = m709 @ np.array([0.2, 0.4, 0.1])
    xyz_pic = {k: np.full((2, 3), v, np.float32) for k, v in zip("RGB", xyz)}
    chroma = _attr("chromaticities", "chromaticities", struct.pack("<8f", 1, 0, 0, 1, 0, 0, 1 / 3, 1 / 3))
    p = tmp_path / "xyz.exr"
    p.write_bytes(make_exr(xyz_pic, exr.NONE, attrs=chroma))
    img = pixport.read(p)
    assert np.allclose(img.pixels[0, 0], [0.2, 0.4, 0.1], atol=1e-5)
    p.write_bytes(make_exr(pic, exr.NONE, attrs=_attr("chromaticities", "chromaticities",
                                                      struct.pack("<8f", 0.64, 0.33, 0.3, 0.6, 0.15, 0.06, 0.3127, 0.329))))
    assert np.allclose(pixport.read(p).pixels[0, 0], [0.2, 0.4, 0.1], atol=1e-3)


def test_layers_and_views(tmp_path):
    a = np.full((2, 2), 0.1, np.float16)
    b = np.full((2, 2), 0.9, np.float16)
    p = tmp_path / "l.exr"
    # no unprefixed channels: the first view of multiView wins over other layers
    views = _attr("multiView", "stringvector", struct.pack("<i", 4) + b"left" + struct.pack("<i", 5) + b"right")
    p.write_bytes(make_exr({"right.R": b, "right.G": b, "right.B": b, "left.R": a, "left.G": a, "left.B": a}, attrs=views))
    img = pixport.read(p)
    assert img.metadata["exr_layer"] == "left" and np.allclose(img.pixels, 0.1, atol=1e-3)
    # unprefixed channels are the main picture
    p.write_bytes(make_exr({"R": b, "G": b, "B": b, "diffuse.R": a, "diffuse.G": a, "diffuse.B": a}))
    assert np.allclose(pixport.read(p).pixels, 0.9, atol=1e-3)
    # a beauty-named layer
    p.write_bytes(make_exr({"aov.R": a, "beauty.R": b, "beauty.G": b, "beauty.B": b}))
    assert pixport.read(p).metadata["exr_layer"] == "beauty"
    # nothing picture-like: the first channel as grey
    p.write_bytes(make_exr({"depth": np.full((2, 2), 3.0, np.float32)}))
    assert pixport.read(p).pixels.shape == (2, 2, 1)


def test_multipart_picks_the_picture(tmp_path):
    z = np.full((3, 4), 7.0, np.float32)
    c = np.full((3, 4), 0.5, np.float16)
    p = tmp_path / "m.exr"
    p.write_bytes(make_multipart([("depth", {"Z": z}), ("rgba", {"R": c, "G": c, "B": c, "A": np.ones_like(c)})]))
    img = pixport.read(p)
    assert img.metadata["exr_part"] == 1 and img.metadata["exr_parts"] == ["depth", "rgba"]
    assert img.pixels.shape == (3, 4, 4)
    ours, _part = exr.read_channels(p, 0)
    assert np.array_equal(ours["Z"], z)
    assert pixport.read(p, part=0).pixels.shape == (3, 4, 1)


def test_alpha_is_made_straight_or_dropped(tmp_path):
    a = np.full((4, 4), 0.5, np.float16)
    c = np.full((4, 4), 0.25, np.float16)
    p = tmp_path / "a.exr"
    p.write_bytes(make_exr({"R": c, "G": c, "B": c, "A": a}))
    img = pixport.read(p)
    assert np.allclose(img.pixels[0, 0], [0.5, 0.5, 0.5, 0.5])
    p.write_bytes(make_exr({"R": c, "G": c, "B": c, "A": np.zeros_like(a)}))
    img = pixport.read(p)
    assert img.pixels.shape == (4, 4, 3) and "exr_alpha" in img.metadata


def test_deep_and_unknown_compressions_are_unsupported(tmp_path):
    p = tmp_path / "d.exr"
    pic = _pic()
    data = bytearray(make_exr(pic, exr.NONE))
    i = data.index(b"compression\0compression\0") + len(b"compression\0compression\0") + 4
    data[i] = 10  # ZSTD
    p.write_bytes(bytes(data))
    with pytest.raises(UnsupportedError):
        pixport.read(p)
    head = make_exr(pic, exr.NONE)
    deep = head[:4] + struct.pack("<I", 2 | 0x800) + head[8:]
    deep = deep.replace(_attr("screenWindowWidth", "float", struct.pack("<f", 1.0)),
                        _attr("screenWindowWidth", "float", struct.pack("<f", 1.0))
                        + _attr("type", "string", b"deepscanline"), 1)
    p.write_bytes(deep)
    with pytest.raises(UnsupportedError, match="deep"):
        pixport.read(p)


# ---------------------------------------------------------------- writing

@pytest.mark.parametrize("comp", ["zip", "zips", "piz", "none"])
@pytest.mark.parametrize("ptype", ["half", "float"])
def test_write_round_trip(tmp_path, comp, ptype):
    rng = np.random.default_rng(8)
    px = (rng.random((45, 70, 4)) * [4, 2, 1, 1]).astype(np.float32)
    px[..., 3] = np.clip(px[..., 3], 0.05, 1)
    p = tmp_path / "w.exr"
    pixport.write(Image(px, "linear"), p, compression=comp, pixel_type=ptype)
    ours, part = exr.read_channels(p)
    assert part.compression == {"zip": exr.ZIP, "zips": exr.ZIPS, "piz": exr.PIZ, "none": exr.NONE}[comp]
    dt = np.float16 if ptype == "half" else np.float32
    premult = px[..., :3] * px[..., 3:]
    for i, k in enumerate("RGB"):
        assert np.array_equal(ours[k], premult[..., i].astype(dt))
    assert np.array_equal(ours["A"], px[..., 3].astype(dt))
    back = pixport.read(p).pixels
    assert np.allclose(back, px, rtol=2e-3, atol=1e-4)


def test_write_grey_and_srgb(tmp_path):
    p = tmp_path / "g.exr"
    pixport.write(Image(np.full((5, 6, 1), 0.5, np.float32), "srgb"), p)
    ours, _ = exr.read_channels(p)
    assert list(ours) == ["Y"]
    assert abs(float(ours["Y"][0, 0]) - 0.214) < 1e-3
    with pytest.raises(PixportError):
        pixport.write(Image(np.zeros((2, 2, 3), np.float32)), p, compression="b44")


# ---------------------------------------------------------------- damage

def _assert_only_pixport_errors(p: Path, good: bytes, seed: int):
    for cut in sorted(set(np.linspace(0, len(good) - 1, 60).astype(int).tolist())):
        p.write_bytes(good[:cut])
        with pytest.raises(PixportError):
            pixport.read(p)
    rng = np.random.default_rng(seed)
    for _ in range(80):
        b = bytearray(good)
        for i in rng.integers(8, len(b), 4):
            b[int(i)] = int(rng.integers(0, 256))
        p.write_bytes(bytes(b))
        try:
            pixport.read(p)
        except PixportError:
            pass


@pytest.mark.parametrize("codec", CODECS)
def test_damaged_codec_files(tmp_path, codec):
    _assert_only_pixport_errors(tmp_path / "x.exr", (DATA / f"codec_{codec}.exr").read_bytes(), CODECS.index(codec))


def test_hostile_headers(tmp_path):
    pic = _pic(4, 4)
    p = tmp_path / "h.exr"
    huge = make_exr(pic, exr.NONE).replace(struct.pack("<4i", 0, 0, 3, 3), struct.pack("<4i", 0, 0, 2000000, 2000000))
    p.write_bytes(huge)
    with pytest.raises(PixportError):
        pixport.read(p)
    p.write_bytes(b"\x76\x2f\x31\x01" + struct.pack("<I", 3))
    with pytest.raises(PixportError):
        pixport.read(p)


# ---------------------------------------------------------------- native and NumPy agree

@pytest.mark.skipif(not _accel.has("exr_piz"), reason="pixport-native isn't installed")
@pytest.mark.parametrize("codec", ["rle", "piz", "b44", "b44a", "dwaa", "dwab"])
def test_native_matches_numpy(monkeypatch, codec):
    fast, _ = exr.read_channels(DATA / f"codec_{codec}.exr")
    monkeypatch.setattr(exr, "native", None)
    slow, _ = exr.read_channels(DATA / f"codec_{codec}.exr")
    for k in fast:
        assert np.array_equal(_bits(fast[k]), _bits(slow[k])), k


@pytest.mark.skipif(not _accel.has("exr_piz_encode"), reason="pixport-native isn't installed")
def test_native_piz_writer_matches_numpy(monkeypatch):
    pic = _pic(37, 61, 3)
    part = exr.Part({}, [exr.Channel(k, {2: exr.HALF, 4: exr.FLOAT}[v.dtype.itemsize] if v.dtype.kind == "f" else exr.UINT)
                         for k, v in sorted(pic.items())], (0, 0, 60, 36), (0, 0, 60, 36), exr.PIZ, "scanlineimage")
    blk = exr._block(part, 0, 0, 60, 31)
    planes = [v[:32] for _k, v in sorted(pic.items())]
    fast = exr.piz_encode(planes, blk)
    monkeypatch.setattr(exr, "native", None)
    assert fast == exr.piz_encode(planes, blk)
    flat = [np.zeros((32, 61), np.float16)] * 3
    blk3 = exr._block(exr.Part({}, [exr.Channel(k, exr.HALF) for k in "BGR"], (0, 0, 60, 31), (0, 0, 60, 31), exr.PIZ,
                               "scanlineimage"), 0, 0, 60, 31)
    slow = exr.piz_encode(flat, blk3)
    monkeypatch.setattr(exr, "native", _accel.native)
    assert exr.piz_encode(flat, blk3) == slow
