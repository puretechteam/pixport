"""GIMP XCF: precisions, tile compressions, colour modes, layer properties, groups and masks, and
the layer modes against reference values worked out here from GIMP's definitions."""

from __future__ import annotations

import struct

import numpy as np
import pytest
from xcf_writer import PRECISION_DTYPE, L, XcfWriter, u8

import pixport
from pixport import _accel, xcf
from pixport.errors import PixportError


def lin(v):
    v = np.asarray(v, np.float64)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def srgb(v):
    v = np.asarray(v, np.float64)
    a = np.abs(v)
    return np.sign(v) * np.where(a <= 0.0031308, a * 12.92, 1.055 * a ** (1 / 2.4) - 0.055)


def solid(h, w, rgba) -> np.ndarray:
    return np.tile(u8(rgba), (h, w, 1))


def read(tmp_path, data: bytes, name="t.xcf"):
    p = tmp_path / name
    p.write_bytes(data)
    return pixport.read(p)


# ---------------------------------------------------------------- pixels

@pytest.mark.parametrize("comp", [0, 1, 2])
def test_single_layer_every_compression(tmp_path, comp):
    rng = np.random.default_rng(comp)
    px = rng.integers(0, 256, (70, 133, 4), dtype=np.uint8)
    px[:30, :40] = (10, 20, 30, 255)
    img = read(tmp_path, XcfWriter(133, 70, compression=comp).build([L("a", px)]))
    assert img.transfer == "srgb" and img.pixels.shape == (70, 133, 4)
    out = img.to_srgb(8)
    seen = px[..., 3] > 0  # fully transparent pixels keep no colour
    assert np.array_equal(out[seen], px[seen]) and np.array_equal(out[..., 3], px[..., 3])
    assert img.metadata["bits"] == 8


@pytest.mark.parametrize("prec", [100, 150, 200, 250, 300, 350, 500, 550, 600, 650, 700, 750])
def test_precisions(tmp_path, prec):
    rng = np.random.default_rng(prec)
    v = rng.random((9, 70, 4))
    v[..., 3] = 1
    dt = np.dtype(PRECISION_DTYPE[prec])
    raw = (np.rint(v * (2 ** (8 * dt.itemsize) - 1)).astype(dt) if dt.kind == "u" else v.astype(dt))
    img = read(tmp_path, XcfWriter(70, 9, precision=prec).build([L("a", raw)]))
    linear = prec in (100, 200, 300, 500, 600, 700)
    assert img.transfer == ("linear" if linear else "srgb")
    want = raw.astype(np.float64) / (2 ** (8 * dt.itemsize) - 1) if dt.kind == "u" else raw.astype(np.float64)
    assert np.allclose(img.pixels, want.astype(np.float32), rtol=0, atol=1e-6)
    assert img.metadata["bits"] == {1: 8, 2: 16}.get(dt.itemsize, 32)


def test_old_versions_32_bit_pointers(tmp_path):
    px = solid(10, 10, (0.2, 0.4, 0.6, 1))
    for version in (0, 1, 3, 10):
        img = read(tmp_path, XcfWriter(10, 10, version=version).build([L("a", px)]))
        assert np.array_equal(img.to_srgb(8), px)


def test_grey_and_indexed(tmp_path):
    g = np.zeros((5, 6, 2), np.uint8)
    g[..., 0] = np.arange(6) * 40
    g[..., 1] = 255
    img = read(tmp_path, XcfWriter(6, 5, base=1).build([L("g", g)]))
    assert img.pixels.shape == (5, 6, 2)
    assert np.array_equal(img.to_srgb(8)[0, :, 0], np.arange(6) * 40)
    cmap = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], np.uint8)
    idx = np.zeros((4, 3, 2), np.uint8)
    idx[..., 0] = [0, 1, 2]
    idx[..., 1] = [[255, 128, 127]] * 4  # GIMP: alpha below 128 is transparent in indexed images
    img = read(tmp_path, XcfWriter(3, 4, base=2, colormap=cmap).build([L("i", idx)]))
    out = img.to_srgb(8)
    assert out[0, 0].tolist() == [255, 0, 0, 255] and out[0, 1].tolist() == [0, 255, 0, 255]
    assert out[0, 2, 3] == 0


def test_offsets_visibility_opacity(tmp_path):
    bg = solid(20, 30, (1, 1, 1, 1))
    red = solid(10, 10, (1, 0, 0, 1))
    hidden = solid(20, 30, (0, 0, 0, 1))
    layers = [L("hidden", hidden, visible=False), L("red", red, x=25, y=-5, opacity=1.0), L("bg", bg)]
    out = read(tmp_path, XcfWriter(30, 20).build(layers)).to_srgb(8)
    assert out.shape == (20, 30, 4)
    assert out[0, 27].tolist() == [255, 0, 0, 255] and out[4, 29].tolist() == [255, 0, 0, 255]
    assert out[5, 27].tolist() == [255, 255, 255, 255] and out[0, 24].tolist() == [255, 255, 255, 255]


# ---------------------------------------------------------------- compositing

C1, C2 = np.array([0.8, 0.5, 0.2]), np.array([0.3, 0.6, 0.9])


def _two(tmp_path, mode, top_alpha=1.0, opacity=1.0, **spaces):
    bottom = solid(4, 4, (*C1, 1))
    top = solid(4, 4, (*C2, top_alpha))
    img = read(tmp_path, XcfWriter(4, 4).build([L("top", top, mode=mode, opacity=opacity, **spaces), L("bg", bottom)]))
    return img.pixels[0, 0].astype(np.float64)


def test_normal_composites_in_linear_light(tmp_path):
    a = 100 / 255
    c1, c2 = u8(C1) / 255, u8(C2) / 255
    got = _two(tmp_path, 28, top_alpha=a, composite_space=1, composite_mode=1)
    want = srgb(lin(c1) * (1 - a) + lin(c2) * a)
    assert np.allclose(got[:3], want, atol=1e-5) and got[3] == 1
    got = _two(tmp_path, 28, top_alpha=a, composite_space=2, composite_mode=1)
    assert np.allclose(got[:3], c1 * (1 - a) + c2 * a, atol=1e-5)


def test_legacy_normal_and_opacity(tmp_path):
    c1, c2 = u8(C1) / 255, u8(C2) / 255
    got = _two(tmp_path, 0, opacity=0.25)
    assert np.allclose(got[:3], c1 * 0.75 + c2 * 0.25, atol=1e-5)


@pytest.mark.parametrize("mode,fn", [
    (3, lambda a, b: a * b),
    (4, lambda a, b: 1 - (1 - a) * (1 - b)),
    (6, lambda a, b: np.abs(a - b)),
    (7, lambda a, b: np.minimum(a + b, 1)),
    (8, lambda a, b: np.maximum(a - b, 0)),
    (9, np.minimum),
    (10, np.maximum),
    (18, lambda a, b: np.where(b > 0.5, 1 - (1 - a) * (1 - (b - 0.5) * 2), 2 * a * b)),
    (19, lambda a, b: (1 - a) * a * b + a * (1 - (1 - a) * (1 - b))),
    (20, lambda a, b: np.clip(a - b + 128 / 255, 0, 1)),
    (21, lambda a, b: np.clip(a + b - 128 / 255, 0, 1)),
])
def test_legacy_modes_blend_on_perceptual_values(tmp_path, mode, fn):
    c1, c2 = u8(C1) / 255, u8(C2) / 255
    got = _two(tmp_path, mode)
    assert np.allclose(got[:3], fn(c1, c2), atol=1e-5), mode


def test_legacy_modes_never_raise_alpha(tmp_path):
    top = solid(4, 4, (*C2, 1))
    half = solid(4, 4, (*C1, 0.5))
    img = read(tmp_path, XcfWriter(4, 4).build([L("top", top, mode=3), L("bg", half)]))
    assert abs(img.pixels[0, 0, 3] - 128 / 255) < 1e-6


@pytest.mark.parametrize("mode,space,fn", [
    (30, 1, lambda a, b: a * b),  # multiply
    (31, 2, lambda a, b: 1 - (1 - a) * (1 - b)),  # screen
    (32, 1, lambda a, b: np.abs(a - b)),  # difference
    (33, 1, lambda a, b: a + b),  # addition
    (34, 1, lambda a, b: a - b),  # subtract
    (23, 2, lambda a, b: np.where(a < 0.5, 2 * a * b, 1 - 2 * (1 - a) * (1 - b))),  # overlay
    (46, 2, lambda a, b: a - b + 0.5),  # grain extract
    (47, 2, lambda a, b: a + b - 0.5),  # grain merge
    (50, 2, lambda a, b: a + 2 * b - 1),  # linear light
    (52, 2, lambda a, b: 0.5 - 2 * (a - 0.5) * (b - 0.5)),  # exclusion
    (53, 2, lambda a, b: a + b - 1),  # linear burn
    (51, 2, lambda a, b: np.where(a + b < 1, 0.0, 1.0)),  # hard mix
])
def test_modes_blend_in_their_space(tmp_path, mode, space, fn):
    c1, c2 = u8(C1) / 255, u8(C2) / 255
    got = _two(tmp_path, mode, blend_space=space, composite_space=1, composite_mode=2)
    if space == 1:
        want = srgb(fn(lin(c1), lin(c2)))
    else:
        want = fn(c1, c2)
    assert np.allclose(got[:3], want, atol=1e-4), mode


def test_clip_to_backdrop_with_partial_alpha(tmp_path):
    a = 0.5
    c1, c2 = u8(C1) / 255, u8(C2) / 255
    got = _two(tmp_path, 30, top_alpha=128 / 255, blend_space=1, composite_space=1, composite_mode=2)
    a = 128 / 255
    want = srgb(lin(c1) + (lin(c1) * lin(c2) - lin(c1)) * a)
    assert np.allclose(got[:3], want, atol=1e-5) and abs(got[3] - 1) < 1e-6


def test_hsv_and_lch_modes(tmp_path):
    c1, c2 = u8(C1) / 255, u8(C2) / 255
    got = _two(tmp_path, 40, blend_space=2, composite_space=1, composite_mode=2)  # value (HSV)
    assert np.allclose(got[:3], c1 * (c2.max() / c1.max()), atol=1e-4)
    got = _two(tmp_path, 27, blend_space=3, composite_space=1, composite_mode=2)  # lightness (LCH)
    lab_l = xcf._lab(lin(c2)[None].astype(np.float32))[0, 0]
    assert abs(xcf._lab(lin(got[:3])[None].astype(np.float32))[0, 0] - lab_l) < 0.05


def test_bottom_layer_shows_whatever_its_mode(tmp_path):
    img = read(tmp_path, XcfWriter(4, 4).build([L("only", solid(4, 4, (*C2, 1)), mode=30, composite_mode=2)]))
    assert np.allclose(img.pixels[0, 0, :3], u8(C2) / 255, atol=1e-6)


def test_masks(tmp_path):
    bg = solid(4, 8, (0, 0, 0, 1))
    top = solid(4, 8, (1, 1, 1, 1))
    mask = np.zeros((4, 8), np.uint8)
    mask[:, 4:] = 255
    out = read(tmp_path, XcfWriter(8, 4).build([L("top", top, mask=mask, mode=0), L("bg", bg)])).to_srgb(8)
    assert out[0, 0, 0] == 0 and out[0, 7, 0] == 255
    out = read(tmp_path, XcfWriter(8, 4).build([L("top", top, mask=mask, apply_mask=False, mode=0), L("bg", bg)])).to_srgb(8)
    assert out[0, 0, 0] == 255


def test_groups_and_pass_through(tmp_path):
    white = solid(4, 4, (1, 1, 1, 1))
    grey = solid(4, 4, (0.5, 0.5, 0.5, 1))
    child = L("multiply", grey, mode=30, blend_space=1, composite_space=1, composite_mode=2)
    # a normal group renders its children alone: the multiply layer is its bottom layer, so plain grey
    out = read(tmp_path, XcfWriter(4, 4).build([L("g", None, mode=28, children=[child]), L("bg", white)]))
    assert np.allclose(out.pixels[0, 0, :3], 128 / 255, atol=1e-6)
    # a pass-through group puts them on the backdrop: white times grey
    out = read(tmp_path, XcfWriter(4, 4).build([L("g", None, mode=61, children=[child]), L("bg", solid(4, 4, (0.8, 0.8, 0.8, 1)))]))
    want = srgb(lin(204 / 255) * lin(128 / 255))
    assert np.allclose(out.pixels[0, 0, :3], want, atol=1e-5)
    # a group's mask
    mask = np.zeros((4, 4), np.uint8)
    mask[:2] = 255
    black = solid(4, 4, (0, 0, 0, 1))
    out = read(tmp_path, XcfWriter(4, 4).build([L("g", None, mask=mask, children=[L("w", white)]), L("bg", black)]))
    assert out.to_srgb(8)[0, 0, 0] == 255 and out.to_srgb(8)[3, 0, 0] == 0


def test_nested_group_order(tmp_path):
    red, green, blue = (solid(2, 2, (*c, 1)) for c in ((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    tree = [L("outer", None, children=[L("inner", None, children=[L("r", red), L("g", green)]), L("b", blue)])]
    out = read(tmp_path, XcfWriter(2, 2).build(tree)).to_srgb(8)
    assert out[0, 0].tolist() == [255, 0, 0, 255]


def test_metadata(tmp_path):
    icc = b"fake icc profile bytes"
    img = read(tmp_path, XcfWriter(2, 2, icc=icc, resolution=(300.0, 300.0)).build([L("a", solid(2, 2, (0, 0, 0, 1)))]))
    assert img.metadata["icc"] == icc and img.metadata["dpi"] == (300.0, 300.0)
    assert img.metadata["xcf_layers"] == ["a"]


# ---------------------------------------------------------------- damage

@pytest.mark.parametrize("bad", [
    b"gimp xcf v011\0",
    b"gimp xcf v011\0" + struct.pack(">III", 0, 0, 0),
    b"gimp xcf v011\0" + struct.pack(">III", 100000, 100000, 0) + struct.pack(">I", 150),
    b"gimp xcf vX11\0" + bytes(20),
    b"gimp xcf v011\0" + struct.pack(">III", 4, 4, 7) + struct.pack(">I", 150),
    b"gimp xcf v011\0" + struct.pack(">III", 4, 4, 0) + struct.pack(">I", 999),
])
def test_bad_headers(tmp_path, bad):
    p = tmp_path / "b.xcf"
    p.write_bytes(bad)
    with pytest.raises(PixportError):
        pixport.read(p)


@pytest.mark.parametrize("comp", [1, 2])
def test_damage_only_raises_pixport_error(tmp_path, comp):
    rng = np.random.default_rng(comp)
    px = rng.integers(0, 256, (70, 90, 4), dtype=np.uint8)
    good = XcfWriter(90, 70, compression=comp).build(
        [L("a", px, mask=np.full((70, 90), 200, np.uint8)), L("g", None, children=[L("b", px)]), L("c", px)])
    p = tmp_path / "d.xcf"
    for cut in np.linspace(14, len(good) - 1, 50).astype(int).tolist():
        p.write_bytes(good[:cut])
        with pytest.raises(PixportError):
            pixport.read(p)
    for _ in range(60):
        b = bytearray(good)
        for i in rng.integers(14, len(b), 4):
            b[int(i)] = int(rng.integers(0, 256))
        p.write_bytes(bytes(b))
        try:
            pixport.read(p)
        except PixportError:
            pass


@pytest.mark.skipif(not _accel.has("xcf_rle_level"), reason="pixport-native isn't installed")
def test_native_matches_numpy(tmp_path, monkeypatch):
    rng = np.random.default_rng(9)
    px = rng.integers(0, 4, (150, 140, 4), dtype=np.uint8) * 60
    p = tmp_path / "n.xcf"
    p.write_bytes(XcfWriter(140, 150).build([L("a", px, x=-7, y=3)]))
    fast = pixport.read(p).pixels
    monkeypatch.setattr(xcf, "native", None)
    slow = pixport.read(p).pixels
    assert np.array_equal(fast, slow)
    data = p.read_bytes()
    for cut in range(len(data) - 400, len(data), 37):
        p.write_bytes(data[:cut])
        with pytest.raises(PixportError):
            pixport.read(p)
