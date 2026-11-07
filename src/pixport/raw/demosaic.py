"""Demosaicing: a full-colour picture from a colour filter mosaic.

Bayer (2x2) mosaics use ratio-corrected directional interpolation, after L. Sanz Rodriguez's RCD
(2017): green is estimated along each axis from its neighbours, each estimate scaled by the local
low-pass ratio, and the axes are blended by how strongly the picture changes along each; red and
blue then follow as colour differences, diagonally at the opposite colour's photosites and along
the axes at green ones. Other repeats (Fujifilm's 6x6 X-Trans) use edge-weighted interpolation of
green, then of the colour differences, over the 5x5 neighbourhood.

Input: float32 mosaic, 0..1 (black subtracted, white balanced); output height x width x 3.
pixport-native has both compiled (``demosaic_bayer``, ``demosaic_any``) with the same results."""

from __future__ import annotations

import numpy as np

from .. import _accel

EPS = np.float32(1e-5)
PAD = 24        # border added around each strip (a multiple of the 2x2 and 6x6 repeats)
STRIP = 256     # rows per strip, to keep the temporaries small


def demosaic(cfa: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """RGB (float32, height x width x 3) from a mosaic and its repeat (0=R, 1=G, 2=B), the
    pattern aligned with ``cfa[0, 0]``."""
    cfa = np.ascontiguousarray(cfa, dtype=np.float32)
    pattern = np.asarray(pattern, dtype=np.uint8)
    h, w = cfa.shape
    if h < 2 or w < 2:
        out = np.zeros((h, w, 3), np.float32)
        for y in range(h):
            for x in range(w):
                out[y, x, :] = cfa[y, x]
        return out
    bayer = pattern.shape == (2, 2) and sorted(pattern.ravel().tolist()) == [0, 1, 1, 2] and \
        (pattern[0, 0] == 1) == (pattern[1, 1] == 1)
    if bayer and _accel.has("demosaic_bayer"):
        return np.asarray(_accel.native.demosaic_bayer(cfa, pattern)).reshape(h, w, 3)
    if not bayer and _accel.has("demosaic_any"):
        return np.asarray(_accel.native.demosaic_any(cfa, np.ascontiguousarray(pattern))).reshape(h, w, 3)
    ph, pw = pattern.shape
    if PAD % ph or PAD % pw or h < ph or w < pw:
        raise ValueError(f"A {ph} x {pw} colour filter repeat isn't supported on a {w} x {h} picture")
    fn = _rcd if bayer else _generic
    padded = cfa[np.ix_(pad_index(h, PAD, ph, bayer), pad_index(w, PAD, pw, bayer))]
    out = np.empty((h, w, 3), np.float32)
    for y0 in range(0, h, STRIP):
        y1 = min(h, y0 + STRIP)
        part = padded[y0:y1 + 2 * PAD]
        out[y0:y1] = fn(part, np.roll(pattern, (-(y0 % pattern.shape[0]), 0), axis=(0, 1)))[PAD:PAD + y1 - y0, PAD:PAD + w]
    return out


def pad_index(n: int, pad: int, period: int, mirror: bool) -> np.ndarray:
    """Source indices for a border of ``pad`` around ``n`` samples that keeps the colour filter's
    phase: mirrored about the edge sample for a 2x2 repeat, else the nearest whole repeat."""
    i = np.arange(-pad, n + pad)
    if mirror and n > pad:
        i = np.abs(i)
        i = np.where(i >= n, 2 * (n - 1) - i, i)
        return i
    lo = i % period
    hi = (n - period) + (i - (n - period)) % period
    return np.where(i < 0, lo, np.where(i >= n, hi, i))


def _masks(shape, pattern):
    ph, pw = pattern.shape
    reps = (-(-shape[0] // ph), -(-shape[1] // pw))
    # the padded grid starts PAD before the pattern's origin (PAD is a multiple of the repeat)
    idx = np.tile(pattern, reps)[: shape[0], : shape[1]]
    return idx


def _rcd(c: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """One padded strip (rows x cols, PAD border all round) demosaiced: rows x cols x 3."""
    hp, wp = c.shape
    idx = _masks(c.shape, pattern)
    is_g = idx == 1
    r = 4
    ih, iw = hp - 2 * r, wp - 2 * r

    def s(a, dy, dx):
        return a[r + dy:r + dy + ih, r + dx:r + dx + iw]

    def full(inner, fill=0.0):
        out = np.full((hp, wp), np.float32(fill), np.float32)
        out[r:r + ih, r:r + iw] = inner
        return out

    half = np.float32(0.5)
    # 1. how much the picture changes vertically and horizontally
    hv = full(np.abs(s(c, -1, 0) - s(c, 1, 0)) + half * np.abs(2 * s(c, 0, 0) - s(c, -2, 0) - s(c, 2, 0)))
    hh = full(np.abs(s(c, 0, -1) - s(c, 0, 1)) + half * np.abs(2 * s(c, 0, 0) - s(c, 0, -2) - s(c, 0, 2)))
    sv = EPS + s(hv, -1, 0) * s(hv, -1, 0) + s(hv, 0, 0) * s(hv, 0, 0) + s(hv, 1, 0) * s(hv, 1, 0)
    sh = EPS + s(hh, 0, -1) * s(hh, 0, -1) + s(hh, 0, 0) * s(hh, 0, 0) + s(hh, 0, 1) * s(hh, 0, 1)
    vh = full(sv / (sv + sh), 0.5)
    # 2. low-pass (all three colours of the 3x3 neighbourhood)
    q, e = np.float32(0.25), np.float32(0.125)
    lp = full(q * s(c, 0, 0) + e * (s(c, -1, 0) + s(c, 1, 0) + s(c, 0, -1) + s(c, 0, 1))
              + np.float32(0.0625) * (s(c, -1, -1) + s(c, -1, 1) + s(c, 1, -1) + s(c, 1, 1)))
    # 3. green at red and blue photosites
    g_n = EPS + np.abs(s(c, -1, 0) - s(c, 1, 0)) + np.abs(s(c, 0, 0) - s(c, -2, 0)) + np.abs(s(c, -1, 0) - s(c, -3, 0)) \
        + np.abs(s(c, -2, 0) - s(c, -4, 0))
    g_s = EPS + np.abs(s(c, 1, 0) - s(c, -1, 0)) + np.abs(s(c, 0, 0) - s(c, 2, 0)) + np.abs(s(c, 1, 0) - s(c, 3, 0)) \
        + np.abs(s(c, 2, 0) - s(c, 4, 0))
    g_w = EPS + np.abs(s(c, 0, -1) - s(c, 0, 1)) + np.abs(s(c, 0, 0) - s(c, 0, -2)) + np.abs(s(c, 0, -1) - s(c, 0, -3)) \
        + np.abs(s(c, 0, -2) - s(c, 0, -4))
    g_e = EPS + np.abs(s(c, 0, 1) - s(c, 0, -1)) + np.abs(s(c, 0, 0) - s(c, 0, 2)) + np.abs(s(c, 0, 1) - s(c, 0, 3)) \
        + np.abs(s(c, 0, 2) - s(c, 0, 4))
    l0 = s(lp, 0, 0)
    two = np.float32(2)
    n_est = s(c, -1, 0) * (two * l0 / (EPS + l0 + s(lp, -2, 0)))
    s_est = s(c, 1, 0) * (two * l0 / (EPS + l0 + s(lp, 2, 0)))
    w_est = s(c, 0, -1) * (two * l0 / (EPS + l0 + s(lp, 0, -2)))
    e_est = s(c, 0, 1) * (two * l0 / (EPS + l0 + s(lp, 0, 2)))
    v_est = (g_s * n_est + g_n * s_est) / (g_n + g_s)
    h_est = (g_e * w_est + g_w * e_est) / (g_e + g_w)
    vh_c = s(vh, 0, 0)
    vh_n = np.float32(0.25) * (s(vh, -1, -1) + s(vh, -1, 1) + s(vh, 1, -1) + s(vh, 1, 1))
    vh_d = np.where(np.abs(half - vh_c) < np.abs(half - vh_n), vh_n, vh_c)
    green = np.clip(v_est * (1 - vh_d) + h_est * vh_d, 0, 1)
    g = full(np.where(s(is_g, 0, 0), s(c, 0, 0), green))
    # 4. red at blue photosites and blue at red ones, along the diagonals
    hp_ = full(np.abs(s(c, -1, -1) - s(c, 1, 1)) + half * np.abs(2 * s(c, 0, 0) - s(c, -2, -2) - s(c, 2, 2)))
    hq_ = full(np.abs(s(c, -1, 1) - s(c, 1, -1)) + half * np.abs(2 * s(c, 0, 0) - s(c, -2, 2) - s(c, 2, -2)))
    sp = EPS + s(hp_, -1, -1) * s(hp_, -1, -1) + s(hp_, 0, 0) * s(hp_, 0, 0) + s(hp_, 1, 1) * s(hp_, 1, 1)
    sq = EPS + s(hq_, -1, 1) * s(hq_, -1, 1) + s(hq_, 0, 0) * s(hq_, 0, 0) + s(hq_, 1, -1) * s(hq_, 1, -1)
    pq = full(sp / (sp + sq), 0.5)
    pq_c = s(pq, 0, 0)
    pq_n = np.float32(0.25) * (s(pq, -1, 0) + s(pq, 1, 0) + s(pq, 0, -1) + s(pq, 0, 1))
    pq_d = np.where(np.abs(half - pq_c) < np.abs(half - pq_n), pq_n, pq_c)
    d = c - g    # colour difference, meaningful at red and blue photosites
    nw_g = EPS + np.abs(s(c, -1, -1) - s(c, 1, 1)) + np.abs(s(c, -1, -1) - s(c, -3, -3)) + np.abs(s(g, 0, 0) - s(g, -2, -2))
    se_g = EPS + np.abs(s(c, 1, 1) - s(c, -1, -1)) + np.abs(s(c, 1, 1) - s(c, 3, 3)) + np.abs(s(g, 0, 0) - s(g, 2, 2))
    ne_g = EPS + np.abs(s(c, -1, 1) - s(c, 1, -1)) + np.abs(s(c, -1, 1) - s(c, -3, 3)) + np.abs(s(g, 0, 0) - s(g, -2, 2))
    sw_g = EPS + np.abs(s(c, 1, -1) - s(c, -1, 1)) + np.abs(s(c, 1, -1) - s(c, 3, -3)) + np.abs(s(g, 0, 0) - s(g, 2, -2))
    p_est = (se_g * s(d, -1, -1) + nw_g * s(d, 1, 1)) / (nw_g + se_g)
    q_est = (sw_g * s(d, -1, 1) + ne_g * s(d, 1, -1)) / (ne_g + sw_g)
    diag = p_est * (1 - pq_d) + q_est * pq_d
    # at a red photosite this is blue minus green (and the other way round)
    rb = full(diag)
    is_r, is_b = idx == 0, idx == 2
    dr = np.where(is_r, d, np.where(is_b, rb, 0)).astype(np.float32)    # R - G at red and blue photosites
    db = np.where(is_b, d, np.where(is_r, rb, 0)).astype(np.float32)    # B - G at red and blue photosites
    # 5. red and blue at green photosites, along the axes (differences of the four neighbours)
    gn = EPS + np.abs(s(g, 0, 0) - s(g, -2, 0)) + np.abs(s(g, -1, 0) - s(g, 1, 0))
    gs = EPS + np.abs(s(g, 0, 0) - s(g, 2, 0)) + np.abs(s(g, 1, 0) - s(g, -1, 0))
    gw = EPS + np.abs(s(g, 0, 0) - s(g, 0, -2)) + np.abs(s(g, 0, -1) - s(g, 0, 1))
    ge = EPS + np.abs(s(g, 0, 0) - s(g, 0, 2)) + np.abs(s(g, 0, 1) - s(g, 0, -1))
    vh_ng = np.float32(0.25) * (s(vh, -1, 0) + s(vh, 1, 0) + s(vh, 0, -1) + s(vh, 0, 1))
    vh_g = np.where(np.abs(half - vh_c) < np.abs(half - vh_ng), vh_ng, vh_c)
    out = np.empty((hp, wp, 3), np.float32)
    g0 = s(g, 0, 0)
    for ch, dd in ((0, dr), (2, db)):
        v = (gs * s(dd, -1, 0) + gn * s(dd, 1, 0)) / (gn + gs)
        hz = (ge * s(dd, 0, -1) + gw * s(dd, 0, 1)) / (gw + ge)
        at_g = v * (1 - vh_g) + hz * vh_g
        val = g0 + np.where(s(is_g, 0, 0), at_g, s(dd, 0, 0))
        out[r:r + ih, r:r + iw, ch] = np.clip(val, 0, 1)
    out[:, :, 1] = g
    return out


def _generic(c: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """Any repeat (X-Trans): green by edge-weighted averaging of the green photosites in the
    5x5 neighbourhood, then red and blue as weighted colour differences."""
    hp, wp = c.shape
    idx = _masks(c.shape, pattern)
    r = 2
    ih, iw = hp - 2 * r, wp - 2 * r
    offs = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3) if (dy, dx) != (0, 0)]

    def s(a, dy, dx):
        return a[r + dy:r + dy + ih, r + dx:r + dx + iw]

    def interp(vals: np.ndarray, known: np.ndarray, guide: np.ndarray) -> np.ndarray:
        num = np.zeros((ih, iw), np.float32)
        den = np.zeros((ih, iw), np.float32)
        g0 = s(guide, 0, 0)
        for dy, dx in offs:
            k = s(known, dy, dx)
            wgt = k / ((np.float32(dy * dy + dx * dx)) * (EPS + np.abs(s(guide, dy, dx) - g0)))
            num += wgt * s(vals, dy, dx)
            den += wgt
        return np.where(den > 0, num / np.maximum(den, np.float32(1e-20)), 0).astype(np.float32)

    known_g = (idx == 1).astype(np.float32)
    green = interp(c * known_g, known_g, c)
    g = np.array(c)
    g[r:r + ih, r:r + iw] = np.where(s(idx, 0, 0) == 1, s(c, 0, 0), green)
    out = np.empty((hp, wp, 3), np.float32)
    out[:, :, 1] = g
    for ch in (0, 2):
        known = (idx == ch).astype(np.float32)
        diff = (c - g) * known
        est = interp(diff, known, g)
        val = s(g, 0, 0) + np.where(s(idx, 0, 0) == ch, s(c, 0, 0) - s(g, 0, 0), est)
        out[r:r + ih, r:r + iw, ch] = np.clip(val, 0, 1)
    return out
