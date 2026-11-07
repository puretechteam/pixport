"""The preview colour fit: the baseline JPEG reader matches Pillow's, and a matrix is recovered
from a synthetic sensor mosaic and the "camera's" rendering of it."""

from __future__ import annotations

import io

import numpy as np
import pytest

from pixport.image import linear_to_srgb
from pixport.raw import RawData, preview

PIL = pytest.importorskip("PIL.Image")


def _jpeg(rgb: np.ndarray, **kw) -> bytes:
    buf = io.BytesIO()
    PIL.fromarray(rgb).save(buf, "JPEG", **kw)
    return buf.getvalue()


def _picture(h: int = 120, w: int = 160) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    rgb = np.stack([128 + 100 * np.sin(x / 9.0), 128 + 90 * np.cos(y / 7.0), (x + y) % 256], axis=2)
    return np.clip(rgb, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("sub", [0, 1, 2])
def test_jpeg_matches_pillow(sub):
    rgb = _picture()
    data = _jpeg(rgb, quality=90, subsampling=sub)
    ours = preview.decode_jpeg(data).astype(int)
    theirs = np.asarray(PIL.open(io.BytesIO(data)).convert("RGB")).astype(int)
    assert ours.shape == theirs.shape
    # Pillow smooths chroma when upsampling; the reader repeats it
    assert np.abs(ours - theirs).mean() < (1.0 if sub == 0 else 4.0)


def test_jpeg_grey_and_restarts():
    grey = _picture()[:, :, 0]
    buf = io.BytesIO()
    PIL.fromarray(grey).save(buf, "JPEG", quality=95)
    assert np.abs(preview.decode_jpeg(buf.getvalue())[:, :, 1].astype(int) - grey).mean() < 2


def test_jpeg_damage_is_valueerror():
    data = _jpeg(_picture())
    for bad in (data[:300], data[:20] + b"\xff\xc2" + data[22:], b"not a jpeg"):
        with pytest.raises((ValueError, IndexError, KeyError)):
            preview.decode_jpeg(bad)


def test_fit_recovers_the_matrix():
    rng = np.random.default_rng(4)
    m_true = np.array([[1.8, -0.6, -0.2], [-0.3, 1.6, -0.3], [0.05, -0.55, 1.5]])
    gh, gw = 60, 90
    # a scene of colour patches, in white-balanced camera RGB
    patches = rng.uniform(0.05, 0.6, (6, 9, 3))
    cam = np.repeat(np.repeat(patches, 10, axis=0), 10, axis=1)
    wb = (2.0, 1.0, 1.5)
    # the sensor: an RGGB mosaic, 8 x 8 photosites per preview pixel, black 100, white 4095
    big = np.repeat(np.repeat(cam, 8, axis=0), 8, axis=1)
    raw_rgb = big / np.array(wb) * (4095 - 100) + 100
    cfa = np.empty(big.shape[:2], np.uint16)
    cfa[0::2, 0::2] = raw_rgb[0::2, 0::2, 0]
    cfa[0::2, 1::2] = raw_rgb[0::2, 1::2, 1]
    cfa[1::2, 0::2] = raw_rgb[1::2, 0::2, 1]
    cfa[1::2, 1::2] = raw_rgb[1::2, 1::2, 2]
    raw = RawData(cfa=cfa, black=100.0, white=4095.0, wb=wb, crop=(0, 0) + cfa.shape)
    shown = np.clip(cam @ m_true.T, 0, 1)
    jpg = (linear_to_srgb(shown.astype(np.float32)) * 255).round().astype(np.uint8)
    assert jpg.shape[:2] == (gh, gw)
    cams, clipped = preview._block_means(cfa, raw.pattern, raw.black, raw.white, wb, raw.crop, gh, gw)
    assert not clipped.any()
    assert np.allclose(cams, cam, atol=2e-3)
    m, _ = preview.solve(cams.reshape(-1, 3), (shown + 1e-6).reshape(-1, 3), prior=0.0)
    assert np.abs(m - m_true / m_true.sum(axis=1, keepdims=True)).max() < 0.02
    # through the whole path: preview JPEG -> matrix pulled a little toward the typical one
    m2 = preview.fit_matrix(raw, jpg)
    assert m2 is not None and np.abs(m2 - m_true).max() < 0.25
    preview.attach(raw, _jpeg(jpg, quality=95))
    assert raw.metadata["colour"] == preview.FITTED and raw.cam_xyz.shape == (3, 3)


def test_fit_works_on_full_colour_data():
    """sRAW, mRAW and Sony's YCbCr RAW are already full colour: the same fit, without a mosaic."""
    rng = np.random.default_rng(5)
    m_true = np.array([[1.7, -0.5, -0.2], [-0.25, 1.5, -0.25], [0.0, -0.5, 1.5]])
    patches = rng.uniform(0.05, 0.6, (6, 9, 3))
    cam = np.repeat(np.repeat(patches, 10, axis=0), 10, axis=1)
    wb = (2.0, 1.0, 1.5)
    big = np.repeat(np.repeat(cam, 4, axis=0), 4, axis=1)
    lin = (big / np.array(wb) * (16383 - 1024) + 1024).astype(np.float32)
    raw = RawData(linear=lin, black=1024.0, white=16383.0, wb=wb)
    cams, clipped = preview._linear_means(lin, raw.black, raw.white, wb, (0, 0) + lin.shape[:2], 60, 90)
    assert not clipped.any() and np.allclose(cams, cam, atol=1e-4)
    shown = np.clip(cam @ m_true.T, 0, 1)
    jpg = (linear_to_srgb(shown.astype(np.float32)) * 255).round().astype(np.uint8)
    m = preview.fit_matrix(raw, jpg)
    assert m is not None and np.abs(m - m_true).max() < 0.25


def test_no_preview_gives_the_typical_matrix():
    raw = RawData(cfa=np.zeros((64, 64), np.uint16), white=4095.0, crop=(0, 0, 64, 64))
    preview.attach(raw, None)
    assert raw.metadata["colour"] == preview.TYPICAL
    preview.attach(raw, b"\xff\xd8garbage")
    assert raw.metadata["colour"] == preview.TYPICAL
