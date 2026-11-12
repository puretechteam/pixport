"""Radiance HDR: reading every scanline form and orientation, writing, round trips, damage."""

from __future__ import annotations

import numpy as np
import pytest

import pixport
from pixport import _accel, hdr
from pixport.errors import PixportError
from pixport.image import Image


def _rgbe(h: int, w: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    px = rng.integers(0, 256, (h, w, 4), dtype=np.uint8)
    px[..., 3] = rng.integers(120, 140, (h, w))
    px[..., 0] |= 128  # normalised: the largest mantissa has its top bit set, as writers leave it
    px[: h // 2, : w // 3] = (200, 20, 30, 128)  # runs
    return px


def _file(head: bytes, res: str, body: bytes) -> bytes:
    return b"#?RADIANCE\n" + head + b"FORMAT=32-bit_rle_rgbe\n\n" + res.encode() + b"\n" + body


def test_round_trip_is_exact_on_rgbe_values(tmp_path):
    rgbe = _rgbe(37, 53)
    rgb = hdr.rgbe_to_float(rgbe)
    p = tmp_path / "a.hdr"
    pixport.write(Image(rgb, "linear"), p)
    img = pixport.read(p)
    assert img.transfer == "linear" and img.pixels.shape == (37, 53, 3)
    assert np.array_equal(img.pixels, rgb)
    assert np.array_equal(hdr.float_to_rgbe(img.pixels), np.where(rgbe[..., 3:] == 0, 0, rgbe))


def test_float_round_trip_within_rgbe_precision(tmp_path):
    rng = np.random.default_rng(3)
    rgb = (rng.random((20, 30, 3)) * np.logspace(-3, 3, 30)[None, :, None]).astype(np.float32)
    p = tmp_path / "b.hdr"
    pixport.write(Image(rgb, "linear"), p)
    back = pixport.read(p).pixels
    big = rgb.max(axis=2, keepdims=True)
    assert np.all(np.abs(back - rgb) <= big / 128)


def test_srgb_input_is_linearised_and_alpha_dropped(tmp_path):
    px = np.zeros((4, 9, 4), np.float32)
    px[..., :3] = 0.5
    px[..., 3] = 0.25
    p = tmp_path / "c.hdr"
    pixport.write(Image(px, "srgb"), p)
    back = pixport.read(p).pixels
    assert back.shape == (4, 9, 3)
    assert abs(float(back[0, 0, 0]) - 0.21404) < 0.002


def test_grey_input(tmp_path):
    p = tmp_path / "g.hdr"
    pixport.write(Image(np.full((10, 12, 1), 2.0, np.float32), "linear"), p)
    back = pixport.read(p).pixels
    assert np.allclose(back, 2.0, rtol=1 / 128)


def test_writer_runs_and_literals():
    row = np.array([5] * 300 + list(range(100)) + [7] * 3 + [9] * 4, np.uint8)
    out = bytearray()
    hdr._encode_component(row, out)
    # decode it back by hand
    vals, pos = [], 0
    while pos < len(out):
        c = out[pos]
        if c > 128:
            vals += [out[pos + 1]] * (c - 128)
            pos += 2
        else:
            vals += list(out[pos + 1:pos + 1 + c])
            pos += 1 + c
    assert vals == row.tolist()


@pytest.mark.parametrize("res,flip", [
    ("-Y 3 +X 4", lambda a: a),
    ("+Y 3 +X 4", lambda a: a[::-1]),
    ("-Y 3 -X 4", lambda a: a[:, ::-1]),
    ("+Y 3 -X 4", lambda a: a[::-1, ::-1]),
    ("+X 4 -Y 3", lambda a: a.transpose(1, 0, 2)),
    ("+X 4 +Y 3", lambda a: a.transpose(1, 0, 2)[::-1]),
    ("-X 4 -Y 3", lambda a: a.transpose(1, 0, 2)[:, ::-1]),
    ("-X 4 +Y 3", lambda a: a.transpose(1, 0, 2)[::-1, ::-1]),
])
def test_orientations(tmp_path, res, flip):
    data = np.arange(12 * 4, dtype=np.uint8).reshape(-1, 4) % 200 + 1
    data[:, 3] = 128
    first = int(res.split()[1])
    second = int(res.split()[3])
    stored = data.reshape(first, second, 4)
    p = tmp_path / "o.hdr"
    p.write_bytes(_file(b"", res, stored.tobytes()))
    img = pixport.read(p)
    # the stored scanlines, put upright: undo the layout the resolution line describes
    want = hdr.rgbe_to_float(stored)
    if res.split()[0][1] == "Y":
        expect = want[::-1] if res.startswith("+") else want
        if res.split()[2] == "-X":
            expect = expect[:, ::-1]
    else:
        expect = want.transpose(1, 0, 2)
        if res.startswith("-"):
            expect = expect[:, ::-1]
        if res.split()[2] == "+Y":
            expect = expect[::-1]
    assert img.pixels.shape == expect.shape
    assert np.array_equal(img.pixels, expect)
    assert flip is not None


def test_old_run_length_form(tmp_path):
    px = [bytes((10, 20, 30, 130)), bytes((1, 1, 1, 5)), bytes((40, 50, 60, 130)), bytes((1, 1, 1, 2)),
          bytes((1, 1, 1, 0))]
    body = b"".join(px)
    p = tmp_path / "old.hdr"
    p.write_bytes(_file(b"", "-Y 1 +X 9", body))
    img = pixport.read(p)
    e1 = hdr.rgbe_to_float(np.array([[10, 20, 30, 130]], np.uint8))[0]
    e2 = hdr.rgbe_to_float(np.array([[40, 50, 60, 130]], np.uint8))[0]
    row = img.pixels[0]
    assert np.array_equal(row[:6], np.repeat(e1[None], 6, 0))
    assert np.array_equal(row[6:9], np.repeat(e2[None], 3, 0))


def test_header_fields(tmp_path):
    body = _rgbe(2, 10).tobytes()
    p = tmp_path / "h.hdr"
    p.write_bytes(_file(b"# a comment\nEXPOSURE=2\nEXPOSURE=1.5\nSOFTWARE=test\n", "-Y 2 +X 10", body))
    img = pixport.read(p)
    assert img.metadata["hdr_exposure"] == 3.0
    assert img.metadata["hdr_software"] == "test"


def test_primaries_and_xyz(tmp_path):
    body = np.array([[200, 100, 50, 128]] * 10, np.uint8).tobytes()
    base = hdr.rgbe_to_float(np.array([[200, 100, 50, 128]], np.uint8))[0]
    p = tmp_path / "p.hdr"
    p.write_bytes(_file(b"PRIMARIES=0.640 0.330 0.300 0.600 0.150 0.060 0.3127 0.3290\n", "-Y 1 +X 10", body))
    assert np.allclose(pixport.read(p).pixels[0, 0], base)  # sRGB's own primaries: untouched
    p.write_bytes(_file(b"PRIMARIES=0.7347 0.2653 0.0 1.0 0.0001 -0.077 0.32168 0.33767\n", "-Y 1 +X 10", body))
    wide = pixport.read(p).pixels[0, 0]
    assert wide[0] > base[0] and wide[1] < base[1]  # wide-gamut red is redder than sRGB red
    x = b"#?RADIANCE\nFORMAT=32-bit_rle_xyze\n\n-Y 1 +X 10\n" + body
    p.write_bytes(x)
    img = pixport.read(p)
    assert img.metadata["hdr_primaries"] == "XYZ"
    m = np.linalg.inv(np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                                [0.0193339, 0.1191920, 0.9503041]]))
    bfd = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])
    d65 = np.array([0.3127 / 0.3290, 1.0, (1 - 0.3127 - 0.3290) / 0.3290])
    adapt = np.linalg.inv(bfd) @ np.diag((bfd @ d65) / (bfd @ np.ones(3))) @ bfd
    assert np.allclose(img.pixels[0, 0], m @ adapt @ base, rtol=1e-3, atol=1e-6)


def test_detect_and_formats(tmp_path):
    p = tmp_path / "noext"
    p.write_bytes(_file(b"", "-Y 1 +X 2", bytes(8)))
    assert pixport.detect(p) == "hdr"
    assert pixport.formats()["hdr"]["write"]


@pytest.mark.parametrize("bad", [
    b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n",
    b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 2 +Y 3\n",
    b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 0 +X 3\n",
    b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 99999999 +X 99999999\n",
    b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 2 +X 3\n" + bytes(10),
    b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 3\n" + bytes((1, 1, 1, 9)),
    b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 16\n" + bytes((2, 2, 0, 16, 200, 1)),
    b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 16\n" + bytes((2, 2, 0, 15)) + bytes(80),
    b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 16\n" + bytes((2, 2, 0, 16, 0, 1)),
])
def test_damage_is_pixport_error(tmp_path, bad):
    p = tmp_path / "bad.hdr"
    p.write_bytes(bad)
    with pytest.raises(PixportError):
        pixport.read(p)


def test_unknown_format_is_unsupported(tmp_path):
    p = tmp_path / "u.hdr"
    p.write_bytes(b"#?RADIANCE\nFORMAT=48-bit_something\n\n-Y 1 +X 1\n" + bytes(4))
    with pytest.raises(pixport.UnsupportedError):
        pixport.read(p)


def test_truncations_never_escape(tmp_path):
    rgbe = _rgbe(9, 40, 5)
    good = _file(b"", "-Y 9 +X 40", hdr.encode_rgbe(rgbe))
    p = tmp_path / "t.hdr"
    for cut in range(0, len(good), 7):
        p.write_bytes(good[:cut])
        with pytest.raises(PixportError):
            pixport.read(p)
    rng = np.random.default_rng(11)
    for _ in range(60):
        b = bytearray(good)
        for i in rng.integers(40, len(b), 3):
            b[i] = int(rng.integers(0, 256))
        p.write_bytes(bytes(b))
        try:
            pixport.read(p)
        except PixportError:
            pass


@pytest.mark.skipif(not _accel.has("hdr_decode"), reason="pixport-native isn't installed")
def test_native_matches_numpy(monkeypatch):
    rgbe = _rgbe(13, 77, 9)
    enc_native = hdr.encode_rgbe(rgbe)
    dec_native = hdr.decode_rgbe(enc_native, 0, 13, 77)
    monkeypatch.setattr(hdr, "native", None)
    enc_pure = hdr.encode_rgbe(rgbe)
    assert enc_native == enc_pure
    assert np.array_equal(hdr.decode_rgbe(enc_pure, 0, 13, 77), dec_native)
    assert np.array_equal(dec_native, rgbe)


@pytest.mark.skipif(not _accel.has("hdr_rgbe"), reason="pixport-native isn't installed")
def test_native_rgbe_conversion_matches_numpy(monkeypatch):
    rng = np.random.default_rng(12)
    rgb = (rng.standard_normal((50, 70, 3)) * np.logspace(-40, 38, 70)[None, :, None]).astype(np.float32)
    rgb[0, :4] = [[np.nan, 1, 2], [np.inf, 0, 0], [-np.inf, 5, 5], [1e-33, 0, 0]]
    fast = hdr.float_to_rgbe(rgb)
    monkeypatch.setattr(hdr, "native", None)
    assert np.array_equal(fast, hdr.float_to_rgbe(rgb))


def test_xyze_white_is_neutral_in_srgb(tmp_path):
    # equal XYZ is Radiance's equal-energy white: it was (1.205, 0.948, 0.909) times the value
    grey = bytes((128, 128, 128, 129))
    p = tmp_path / "w.hdr"
    p.write_bytes(b"#?RADIANCE\nFORMAT=32-bit_rle_xyze\n\n-Y 1 +X 4\n" + grey * 4)
    px = pixport.read(p).pixels[0, 0]
    assert np.ptp(px) < 1e-3 * px[0]
    assert abs(px[0] - 1.0) < 1e-2


def test_colorcorr_and_exposure_are_recorded_not_applied(tmp_path):
    body = np.array([[200, 100, 50, 128]] * 10, np.uint8).tobytes()
    base = hdr.rgbe_to_float(np.array([[200, 100, 50, 128]], np.uint8))[0]
    p = tmp_path / "c.hdr"
    p.write_bytes(_file(b"EXPOSURE=4\nCOLORCORR=2 0.5 1\n", "-Y 1 +X 10", body))
    img = pixport.read(p)
    assert np.array_equal(img.pixels[0, 0], base)
    assert img.metadata["hdr_exposure"] == 4.0
    assert img.metadata["hdr_colorcorr"] == (2.0, 0.5, 1.0)
