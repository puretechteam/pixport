"""Developing RawData: levels, white balance, highlight clipping, demosaicing (Bayer and X-Trans,
Python and pixport-native alike), colour matrix, crop, orientation, half size and the tone
choices."""

from __future__ import annotations

import numpy as np
import pytest

from pixport import _accel
from pixport.errors import PixportError
from pixport.raw import RawData
from pixport.raw.demosaic import demosaic
from pixport.raw.develop import bt709, daylight_wb, develop, rgb_cam

XTRANS = np.array([[1, 1, 0, 1, 1, 2], [1, 1, 2, 1, 1, 0], [2, 0, 1, 0, 2, 1],
                   [1, 1, 2, 1, 1, 0], [1, 1, 0, 1, 1, 2], [0, 2, 1, 2, 0, 1]], np.uint8)


def _scene(h: int, w: int) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w] / max(h, w)
    return np.stack([0.4 + 0.3 * np.sin(6 * x), 0.5 + 0.2 * np.cos(5 * y), 0.3 + 0.2 * np.sin(4 * (x + y))], 2).astype(np.float32)


def _mosaic(rgb: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    ph, pw = pattern.shape
    idx = np.tile(pattern, (-(-h // ph), -(-w // pw)))[:h, :w]
    return np.take_along_axis(rgb, idx[:, :, None].astype(np.int64), axis=2)[:, :, 0]


def _psnr(a, b) -> float:
    return float(10 * np.log10(1 / np.mean((a - b) ** 2)))


@pytest.mark.parametrize("pattern", [np.array([[0, 1], [1, 2]]), np.array([[1, 2], [0, 1]]), XTRANS])
def test_flat_colour_is_exact(pattern):
    rgb = np.empty((48, 60, 3), np.float32)
    rgb[:] = (0.25, 0.5, 0.75)
    out = demosaic(_mosaic(rgb, pattern), pattern)
    assert np.abs(out - rgb).max() < 1e-5


@pytest.mark.parametrize("pattern", [np.array([[0, 1], [1, 2]]), np.array([[2, 1], [1, 0]]), XTRANS])
def test_demosaic_quality(pattern):
    truth = _scene(96, 128)
    out = demosaic(_mosaic(truth, pattern), pattern)
    assert out.shape == truth.shape
    assert _psnr(out[6:-6, 6:-6], truth[6:-6, 6:-6]) > 40


@pytest.mark.skipif(not _accel.has("demosaic_bayer"), reason="pixport-native isn't installed")
@pytest.mark.parametrize("pattern", [np.array([[1, 0], [2, 1]]), XTRANS])
def test_demosaic_python_matches_native(monkeypatch, pattern):
    rng = np.random.default_rng(1)
    cfa = (rng.random((300, 70)) ** 2).astype(np.float32)     # a strip boundary and plenty of edges
    nat = demosaic(cfa, pattern)
    monkeypatch.setattr(_accel, "native", None)
    py = demosaic(cfa, pattern)
    assert np.array_equal(nat, py)


def _raw(**kw) -> RawData:
    truth = _scene(40, 64)
    wb = np.array([2.0, 1.0, 1.5], np.float32)
    cfa = _mosaic(truth / wb * 0.8, np.array([[0, 1], [1, 2]])) * 4000 + 200
    base = {"cfa": cfa.astype(np.uint16), "black": 200.0, "white": 4200.0, "wb": (2.0, 1.0, 1.5)}
    base.update(kw)
    return RawData(**base)


def test_levels_and_white_balance_linear():
    img = develop(_raw(), tone="linear")
    assert img.transfer == "linear" and img.pixels.shape == (40, 64, 3)
    truth = _scene(40, 64) * 0.8
    assert np.abs(img.pixels[4:-4, 4:-4] - truth[4:-4, 4:-4]).mean() < 0.01


def test_highlights_clip_to_white():
    cfa = np.full((32, 32), 4200, np.uint16)          # every photosite saturated
    img = develop(RawData(cfa=cfa, black=200.0, white=4200.0, wb=(2.0, 1.0, 1.5)), tone="linear")
    assert np.allclose(img.pixels, 1.0, atol=1e-5)    # neutral white, not pink


@pytest.mark.parametrize("o,expect", [(1, (40, 64)), (3, (40, 64)), (6, (64, 40)), (8, (64, 40)), (5, (64, 40))])
def test_orientation(o, expect):
    up = develop(_raw(), tone="linear").pixels
    img = develop(_raw(orientation=o), tone="linear").pixels
    assert img.shape[:2] == expect
    if o == 3:
        assert np.array_equal(img, up[::-1, ::-1])
    if o == 6:
        assert np.array_equal(img, up.transpose(1, 0, 2)[:, ::-1])
    if o == 8:
        assert np.array_equal(img, up.transpose(1, 0, 2)[::-1])


def test_crop_half_and_tones():
    raw = _raw(crop=(2, 4, 30, 50))
    assert develop(raw).pixels.shape == (30, 50, 3)
    assert develop(raw, half=True).pixels.shape == (15, 25, 3)
    magick = develop(raw)
    assert magick.transfer == "srgb"
    assert 0.9 < float(np.percentile(magick.pixels, 99.5)) <= 1.0
    auto = develop(raw, tone="auto")
    assert auto.transfer == "linear"
    assert np.allclose(bt709(np.clip(auto.pixels, 0, 1)), magick.pixels, atol=2e-4)
    with pytest.raises(PixportError):
        develop(raw, tone="film")
    with pytest.raises(PixportError):
        develop(raw, white_balance="tungsten")


def test_colour_matrix_and_white_balance_choices():
    cam_xyz = np.array([[0.9, -0.3, -0.1], [-0.4, 1.2, 0.2], [-0.05, 0.15, 0.6]])
    m = rgb_cam(cam_xyz)
    assert np.allclose(m.sum(axis=1), 1.0)              # white stays white
    assert rgb_cam(None).tolist() == np.eye(3).tolist()
    wb = daylight_wb(cam_xyz)
    assert wb[1] == 1.0 and wb[0] > 0 and wb[2] > 0
    raw = _raw(cam_xyz=cam_xyz)
    for choice in ("camera", "daylight", "auto"):
        img = develop(raw, white_balance=choice)
        assert np.isfinite(img.pixels).all()
    assert "colour" in develop(_raw()).metadata


def test_linear_input():
    rgb = (_scene(20, 30) * 3000 + 100).astype(np.float32)
    img = develop(RawData(linear=rgb, black=100.0, white=3100.0), tone="linear")
    assert np.abs(img.pixels - _scene(20, 30)).max() < 1e-4


def test_empty_raw_is_an_error():
    with pytest.raises(PixportError):
        develop(RawData())
    with pytest.raises(PixportError):
        develop(_raw(crop=(50, 70, 10, 10)))
