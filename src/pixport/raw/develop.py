"""Developing a :class:`RawData` into a picture: black and white levels, white balance, highlight
clipping, demosaicing, camera colour to sRGB primaries, the camera's crop and orientation, and an
exposure and tone curve.

The default (``tone="magick"``) renders as ImageMagick does through LibRaw's defaults, so a
converted RAW looks as it did before: the brightest 1% of the picture is let clip (auto-bright)
and the BT.709 curve (power 0.45, toe slope 4.5) is applied, giving display-ready values (an
Image with transfer "srgb"). ``tone="linear"`` keeps scene-linear values (transfer "linear", 1.0
= the sensor's white after white balance); ``tone="auto"`` applies only the auto-bright (also
linear)."""

from __future__ import annotations

import numpy as np

from ..errors import PixportError
from ..image import Image
from . import RawData
from .demosaic import demosaic

# linear sRGB (D65) -> XYZ, IEC 61966-2-1
XYZ_RGB = np.array([[0.4124564, 0.3575761, 0.1804375],
                    [0.2126729, 0.7151522, 0.0721750],
                    [0.0193339, 0.1191920, 0.9503041]])
D65 = np.array([0.95047, 1.0, 1.08883])


def rgb_cam(cam_xyz: np.ndarray | None) -> np.ndarray:
    """The 3x3 camera RGB (white balanced) -> linear sRGB matrix for a XYZ -> camera matrix: each
    row of camera-from-sRGB is scaled to sum 1 (so white stays white), then inverted."""
    if cam_xyz is None:
        return np.eye(3)
    cam_rgb = np.asarray(cam_xyz, dtype=np.float64)[:3, :3] @ XYZ_RGB
    sums = cam_rgb.sum(axis=1, keepdims=True)
    if np.any(np.abs(sums) < 1e-9):
        return np.eye(3)
    cam_rgb = cam_rgb / sums
    try:
        return np.linalg.inv(cam_rgb)
    except np.linalg.LinAlgError:
        return np.eye(3)


def daylight_wb(cam_xyz: np.ndarray | None) -> tuple[float, float, float] | None:
    """Multipliers that make D65 white neutral for this camera (from its matrix)."""
    if cam_xyz is None:
        return None
    resp = np.asarray(cam_xyz, dtype=np.float64)[:3, :3] @ D65
    if np.any(resp <= 0):
        return None
    m = 1.0 / resp
    return (float(m[0] / m[1]), 1.0, float(m[2] / m[1]))


def _bt709_params(power: float = 0.45, slope: float = 4.5) -> tuple[float, float]:
    """(threshold, offset) of the curve y = slope*x below the threshold, (1+offset)*x^power -
    offset above, joined smoothly (same value and slope)."""
    lo, hi = 1e-9, 1.0
    for _ in range(200):
        t = (lo + hi) / 2
        a = slope * t * (1 / power - 1)
        lhs = 1 + a
        rhs = slope * t ** (1 - power) / power
        if lhs > rhs:
            lo = t
        else:
            hi = t
    t = (lo + hi) / 2
    return t, slope * t * (1 / power - 1)


_T709, _A709 = _bt709_params()


def bt709(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return np.where(x < _T709, x * 4.5, (1 + _A709) * np.power(x, 0.45) - _A709).astype(np.float32)


_CURVE = bt709(np.arange(65536) / 65535.0)


def _matrix(rgb: np.ndarray, m: np.ndarray) -> np.ndarray:
    """``m`` (3x3) applied to every pixel, as float32."""
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    out = np.empty(rgb.shape, np.float32)
    for c in range(3):
        o = out[:, :, c]
        np.multiply(r, m[c, 0], out=o)
        o += g * m[c, 1]
        o += b * m[c, 2]
    return out


def _levels(raw: RawData, shape, oy: int, ox: int) -> np.ndarray:
    """The black level per photosite of a (rows x cols) window starting at (oy, ox)."""
    blk = np.asarray(raw.black, dtype=np.float32)
    if blk.ndim == 0:
        out = np.full(shape, blk, np.float32)
    else:
        ph, pw = blk.shape
        rolled = np.roll(blk, (-(oy % ph), -(ox % pw)), axis=(0, 1))
        out = np.tile(rolled, (-(-shape[0] // ph), -(-shape[1] // pw)))[: shape[0], : shape[1]].astype(np.float32)
    if raw.black_rows is not None:
        out = out + np.asarray(raw.black_rows, np.float32)[oy:oy + shape[0], None]
    if raw.black_cols is not None:
        out = out + np.asarray(raw.black_cols, np.float32)[None, ox:ox + shape[1]]
    return out


def _orient(px: np.ndarray, o: int) -> np.ndarray:
    if o == 2:
        return px[:, ::-1]
    if o == 3:
        return px[::-1, ::-1]
    if o == 4:
        return px[::-1]
    if o == 5:
        return px.transpose(1, 0, 2)
    if o == 6:
        return px.transpose(1, 0, 2)[:, ::-1]
    if o == 7:
        return px.transpose(1, 0, 2)[::-1, ::-1]
    if o == 8:
        return px.transpose(1, 0, 2)[::-1]
    return px


def _auto_wb(mosaic: np.ndarray, colour: np.ndarray, white: float) -> tuple[float, float, float]:
    """Grey world over the unclipped photosites."""
    sub = mosaic[::4, ::4]
    col = colour[::4, ::4]
    ok = sub < 0.95 * white
    means = []
    for c in range(3):
        sel = sub[(col == c) & ok]
        means.append(float(sel.mean()) if sel.size else 1.0)
    g = max(means[1], 1e-9)
    return tuple(g / max(m, 1e-9) for m in means)


def develop(raw: RawData, half: bool = False, white_balance: str = "camera", tone: str = "magick",
            bright: float = 1.0) -> Image:
    """The picture in ``raw``, as a linear-light :class:`Image` in sRGB primaries.

    ``half``: a half-size picture without demosaicing (each 2x2 block becomes one pixel), much
    quicker. ``white_balance``: ``"camera"`` (as shot), ``"daylight"`` or ``"auto"`` (grey
    world). ``tone``: ``"magick"``, ``"auto"`` or ``"linear"`` (see the module notes);
    ``bright`` scales the exposure."""
    if tone not in ("magick", "auto", "linear"):
        raise PixportError(f"Unknown tone {tone!r}")
    if raw.cfa is None and raw.linear is None:
        raise PixportError("The RAW file has no picture data")
    src = raw.cfa if raw.cfa is not None else raw.linear
    full_h, full_w = src.shape[:2]
    top, left, h, w = raw.crop or (0, 0, full_h, full_w)
    top, left = max(0, int(top)), max(0, int(left))
    h, w = min(int(h), full_h - top), min(int(w), full_w - left)
    if h < 1 or w < 1:
        raise PixportError("The RAW file's crop is empty")
    white = float(raw.white)
    md = dict(raw.metadata)
    wb = raw.wb
    if white_balance == "daylight":
        wb = daylight_wb(raw.cam_xyz) or wb
    elif white_balance not in ("camera", "auto"):
        raise PixportError(f"Unknown white balance {white_balance!r}")

    if raw.cfa is not None:
        pat = np.asarray(raw.pattern, dtype=np.uint8)
        ph, pw = pat.shape
        win = raw.cfa[top:top + h, left:left + w]
        pat_here = np.roll(pat, (-(top % ph), -(left % pw)), axis=(0, 1))
        colour = np.tile(pat_here, (-(-h // ph), -(-w // pw)))[:h, :w]
        blk = _levels(raw, (h, w), top, left)
        scale = np.float32(1.0) / np.maximum(np.float32(white) - blk, np.float32(1.0))
        x = (win.astype(np.float32) - blk) * scale
        if white_balance == "auto":
            wb = _auto_wb(x, colour, 1.0)
        mul = np.array(wb, dtype=np.float32)
        mul = mul / mul.min() if mul.min() > 0 else np.ones(3, np.float32)
        x *= mul[colour]
        np.clip(x, 0.0, 1.0, out=x)
        if half and pat.shape == (2, 2):
            hh, hw = h // 2, w // 2
            q = [x[dy:dy + 2 * hh:2, dx:dx + 2 * hw:2] for dy in (0, 1) for dx in (0, 1)]
            cq = [int(pat_here[dy, dx]) for dy in (0, 1) for dx in (0, 1)]
            rgb = np.zeros((hh, hw, 3), np.float32)
            for plane, c in zip(q, cq):
                rgb[:, :, c] += plane
            rgb[:, :, 1] *= np.float32(0.5)
        else:
            rgb = demosaic(x, pat_here)
            if half:
                rgb = rgb[: h & ~1, : w & ~1].reshape(h // 2, 2, w // 2, 2, 3).mean(axis=(1, 3)).astype(np.float32)
    else:
        lin = raw.linear[top:top + h, left:left + w].astype(np.float32)
        blk = np.asarray(raw.black, dtype=np.float32)
        blk = blk.reshape(-1)[:3] if blk.size >= 3 else np.full(3, blk.reshape(-1)[0] if blk.size else 0, np.float32)
        rgb = (lin - blk) / np.maximum(np.float32(white) - blk, np.float32(1.0))
        if white_balance == "auto":
            means = [float(rgb[::4, ::4, c].mean()) for c in range(3)]
            wb = tuple(means[1] / max(m, 1e-9) for m in means)
        mul = np.array(wb, dtype=np.float32)
        mul = mul / mul.min() if mul.min() > 0 else np.ones(3, np.float32)
        rgb = np.clip(rgb * mul, 0.0, 1.0)
        if half:
            rgb = rgb[: h & ~1, : w & ~1].reshape(h // 2, 2, w // 2, 2, 3).mean(axis=(1, 3)).astype(np.float32)

    m = rgb_cam(raw.cam_xyz).astype(np.float32)
    if raw.cam_xyz is None:
        md.setdefault("colour", "camera colour (no colour matrix found)")
    transfer = "linear"
    if tone == "linear":
        rgb = _matrix(rgb, m * np.float32(bright))
    else:
        # auto-bright: the level the brightest 1% of each channel reaches becomes white
        sample = _matrix(rgb[::4, ::4] if rgb.shape[0] * rgb.shape[1] > 2_000_000 else rgb, m)
        lvl = max(float(np.percentile(np.maximum(sample[:, :, c], 0), 99.0)) for c in range(3))
        lvl = max(lvl, 256 / 65535)
        rgb = _matrix(rgb, m * np.float32(bright / lvl))
        if tone == "magick":
            # the BT.709 curve through a 16-bit table; the values are then display-ready sRGB
            q = np.empty(rgb.shape, np.uint16)
            np.clip(rgb * np.float32(65535), 0, 65535, out=rgb)
            np.rint(rgb, out=rgb)
            q[...] = rgb
            rgb = _CURVE[q]
            transfer = "srgb"
    rgb = _orient(rgb, int(raw.orientation or 1))
    md.setdefault("wb", tuple(float(v) for v in wb))
    return Image(np.ascontiguousarray(rgb, dtype=np.float32), transfer=transfer, metadata=md)
