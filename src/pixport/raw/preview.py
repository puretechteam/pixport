"""A colour matrix for cameras whose RAW files don't carry one, fitted from the camera's own
small preview: the thumbnail JPEG (baseline, decoded here) shows the same scene rendered by the
camera, so the 3x3 matrix that turns the white-balanced sensor colours into the preview's
colours can be estimated from the two.

Only hue and saturation are matched (each pair of colours is compared by direction, not
brightness), since the camera's tone curve and the developer's auto-brightness differ anyway;
neutral stays neutral (every row of the matrix sums to 1). A picture of few colours can't pin a
matrix down, so the fit is pulled toward GENERIC, a typical camera's matrix."""

from __future__ import annotations

import struct

import numpy as np

from ..image import srgb_to_linear

# linear sRGB (D65) -> XYZ, IEC 61966-2-1
XYZ_RGB = np.array([[0.4124564, 0.3575761, 0.1804375],
                    [0.2126729, 0.7151522, 0.0721750],
                    [0.0193339, 0.1191920, 0.9503041]])
MAX_PREVIEW = 1 << 22      # pixels a preview may have to be decoded here
# A typical camera's white-balanced camera RGB -> linear sRGB (rows sum to 1): the mean of the
# D65 matrices that Leica (M10, Q Typ 116), Pentax (K-1, K-3) and DJI (FC220) write into their
# own DNG files (ColorMatrix tags of raw.pixls.us samples), each turned camera -> sRGB.
GENERIC = np.array([[1.5504, -0.4194, -0.1310],
                    [-0.2154, 1.5111, -0.2957],
                    [0.0335, -0.4948, 1.4613]])
ZIGZAG = np.array([0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
                   13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
                   52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63])


# ---------------------------------------------------------------------------------------------
# baseline JPEG (ITU T.81, sequential Huffman, 8-bit), enough for camera thumbnails


def _huffman(counts: bytes, symbols: bytes) -> dict:
    """{(length, code): symbol} for a canonical Huffman table."""
    table, code, k = {}, 0, 0
    for length in range(1, 17):
        for _ in range(counts[length - 1]):
            if k >= len(symbols) or code >= 1 << length:
                raise ValueError("JPEG Huffman table is damaged")
            table[(length, code)] = symbols[k]
            code += 1
            k += 1
        code <<= 1
    return table


def _unstuff(d: bytes, pos: int) -> tuple[list[bytes], int]:
    """The entropy-coded data from pos: its restart intervals (stuffing removed), and where the
    next marker is."""
    parts, cur = [], bytearray()
    n = len(d)
    while pos < n:
        b = d[pos]
        if b != 0xFF:
            cur.append(b)
            pos += 1
            continue
        nxt = d[pos + 1] if pos + 1 < n else 0xD9
        if nxt == 0:
            cur.append(0xFF)
            pos += 2
        elif 0xD0 <= nxt <= 0xD7:
            parts.append(bytes(cur))
            cur = bytearray()
            pos += 2
        elif nxt == 0xFF:
            pos += 1
        else:
            break
    parts.append(bytes(cur))
    return parts, pos


class _Bits:
    __slots__ = ("d", "n", "p", "v")

    def __init__(self, d: bytes):
        self.d, self.p, self.v, self.n = d, 0, 0, 0

    def bit(self) -> int:
        if not self.n:
            self.v = self.d[self.p] if self.p < len(self.d) else 0
            self.p += 1
            self.n = 8
            if self.p > len(self.d) + 64:
                raise ValueError("JPEG data ends early")
        self.n -= 1
        return (self.v >> self.n) & 1

    def bits(self, k: int) -> int:
        v = 0
        for _ in range(k):
            v = (v << 1) | self.bit()
        return v

    def symbol(self, table: dict) -> int:
        code = 0
        for length in range(1, 17):
            code = (code << 1) | self.bit()
            s = table.get((length, code))
            if s is not None:
                return s
        raise ValueError("JPEG data is damaged (no such code)")


def _extend(v: int, size: int) -> int:
    return v - (1 << size) + 1 if size and v < 1 << (size - 1) else v


def decode_jpeg(d: bytes) -> np.ndarray:
    """A baseline JPEG as (height, width, 3) uint8 RGB (grey pictures repeated in all three)."""
    if d[:2] != b"\xff\xd8":
        raise ValueError("Not a JPEG")
    pos, qt, dc_t, ac_t, frame, restart = 2, {}, {}, {}, None, 0
    comps: list = []
    while pos + 4 <= len(d):
        if d[pos] != 0xFF:
            pos += 1
            continue
        marker = d[pos + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7 or marker == 0xFF:
            pos += 2 if marker != 0xFF else 1
            continue
        if marker == 0xD9:
            break
        seg_len = struct.unpack_from(">H", d, pos + 2)[0]
        seg = d[pos + 4:pos + 2 + seg_len]
        if len(seg) != seg_len - 2:
            raise ValueError("JPEG segment runs past the end")
        if marker == 0xDB:
            i = 0
            while i < len(seg):
                pq, tq = seg[i] >> 4, seg[i] & 15
                if pq:
                    qt[tq] = np.frombuffer(seg[i + 1:i + 129], ">u2").astype(np.float64)
                    i += 129
                else:
                    qt[tq] = np.frombuffer(seg[i + 1:i + 65], np.uint8).astype(np.float64)
                    i += 65
        elif marker == 0xC4:
            i = 0
            while i + 17 <= len(seg):
                tc, th = seg[i] >> 4, seg[i] & 15
                counts = seg[i + 1:i + 17]
                n = sum(counts)
                (dc_t if tc == 0 else ac_t)[th] = _huffman(counts, seg[i + 17:i + 17 + n])
                i += 17 + n
        elif marker in (0xC0, 0xC1):
            prec, h, w, nc = struct.unpack_from(">BHHB", seg, 0)
            if prec != 8 or nc not in (1, 3) or not h or not w or h * w > MAX_PREVIEW:
                raise ValueError("This JPEG isn't one the preview reader takes")
            comps = [(seg[6 + 3 * k], seg[7 + 3 * k] >> 4, seg[7 + 3 * k] & 15, seg[8 + 3 * k]) for k in range(nc)]
            if nc == 1:
                comps = [(comps[0][0], 1, 1, comps[0][3])]   # one component: a block per MCU
            if any(not 1 <= c[1] <= 2 or not 1 <= c[2] <= 2 for c in comps):
                raise ValueError("JPEG sampling isn't supported")
            frame = (h, w)
        elif 0xC2 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            raise ValueError("Only baseline JPEG previews are read")
        elif marker == 0xDD:
            restart = struct.unpack_from(">H", seg, 0)[0]
        elif marker == 0xDA:
            if frame is None:
                raise ValueError("JPEG scan before its frame")
            ns = seg[0]
            if ns != len(comps):
                raise ValueError("Only single-scan JPEG previews are read")
            sel = {seg[1 + 2 * k]: (seg[2 + 2 * k] >> 4, seg[2 + 2 * k] & 15) for k in range(ns)}
            parts, pos = _unstuff(d, pos + 2 + seg_len)
            return _scan(parts, frame, comps, sel, qt, dc_t, ac_t, restart)
        pos += 2 + seg_len
    raise ValueError("JPEG has no picture")


def _idct_matrix() -> np.ndarray:
    k = np.arange(8)
    c = np.where(k == 0, np.sqrt(1 / 8), np.sqrt(2 / 8))
    return c[:, None] * np.cos((2 * k[None, :] + 1) * k[:, None] * np.pi / 16)


def _scan(parts, frame, comps, sel, qt, dc_t, ac_t, restart) -> np.ndarray:
    h, w = frame
    hmax, vmax = max(c[1] for c in comps), max(c[2] for c in comps)
    mx, my = -(-w // (8 * hmax)), -(-h // (8 * vmax))
    blocks = {c[0]: np.zeros((my * c[2], mx * c[1], 64)) for c in comps}
    total = mx * my
    per = restart or total
    for c in comps:
        if c[0] not in sel or c[3] not in qt or sel[c[0]][0] not in dc_t or sel[c[0]][1] not in ac_t:
            raise ValueError("JPEG tables are missing")
    m = 0
    for part in parts:
        if m >= total:
            break
        br = _Bits(part)
        pred = {c[0]: 0 for c in comps}
        for _ in range(min(per, total - m)):
            by, bx = divmod(m, mx)
            for cid, hs, vs, _tq in comps:
                dct, act = dc_t[sel[cid][0]], ac_t[sel[cid][1]]
                for v in range(vs):
                    for u in range(hs):
                        coef = blocks[cid][by * vs + v, bx * hs + u]
                        s = br.symbol(dct)
                        pred[cid] += _extend(br.bits(s), s)
                        coef[0] = pred[cid]
                        k = 1
                        while k < 64:
                            rs = br.symbol(act)
                            r, s = rs >> 4, rs & 15
                            if not s:
                                if r != 15:
                                    break
                                k += 16
                                continue
                            k += r
                            if k > 63:
                                raise ValueError("JPEG block is damaged")
                            coef[ZIGZAG[k]] = _extend(br.bits(s), s)
                            k += 1
            m += 1
    t = _idct_matrix()
    planes = []
    for cid, hs, vs, tq in comps:
        b = blocks[cid] * qt[tq][np.argsort(ZIGZAG)] if len(qt[tq]) == 64 else blocks[cid]
        b = b.reshape(b.shape[0], b.shape[1], 8, 8)
        px = np.einsum("ki,abkl,lj->abij", t, b, t) + 128
        img = px.transpose(0, 2, 1, 3).reshape(b.shape[0] * 8, b.shape[1] * 8)
        img = np.repeat(np.repeat(img, vmax // vs, axis=0), hmax // hs, axis=1)
        planes.append(img[:h, :w])
    if len(planes) == 1:
        rgb = np.stack(planes * 3, axis=2)
    else:
        y, cb, cr = planes[0], planes[1] - 128, planes[2] - 128
        rgb = np.stack([y + 1.402 * cr, y - 0.344136 * cb - 0.714136 * cr, y + 1.772 * cb], axis=2)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------------------------
# the fit


def _block_means(cfa, pattern, black, white, wb, crop, gh: int, gw: int):
    """The white-balanced camera colour of each of gh x gw areas of the crop (linear, 1.0 the
    sensor's white), and which areas have a clipped photosite. The crop is seen as whole
    repeats of the colour pattern, so one pass over the data sums every colour."""
    top, left, h, w = crop
    ph, pw = pattern.shape
    y0, x0 = -(-top // ph), -(-left // pw)              # first whole repeat inside the crop
    ny, nx = (top + h) // ph - y0, (left + w) // pw - x0
    if ny < gh or nx < gw:
        # a preview finer than the sensor: nothing to compare
        return np.zeros((gh, gw, 3)), np.ones((gh, gw), bool)
    view = cfa[y0 * ph:(y0 + ny) * ph, x0 * pw:(x0 + nx) * pw].reshape(ny, ph, nx, pw)
    r0 = (np.arange(gh) * ny) // gh
    c0 = (np.arange(gw) * nx) // gw
    s = np.add.reduceat(np.add.reduceat(view, r0, axis=0, dtype=np.int64), c0, axis=2)   # gh, ph, gw, pw
    peak = np.maximum.reduceat(np.maximum.reduceat(view, r0, axis=0), c0, axis=2).max(axis=(1, 3))
    n = np.outer(np.diff(np.append(r0, ny)), np.diff(np.append(c0, nx)))                 # repeats per area
    blk = np.broadcast_to(np.asarray(black, np.float64), pattern.shape) if np.ndim(black) else np.full(pattern.shape, float(black))
    pat = pattern  # the view starts on a whole repeat, so the pattern is as it is
    sums = np.zeros((gh, gw, 3))
    counts = np.zeros(3)
    for y in range(ph):
        for x in range(pw):
            c = int(pat[y, x])
            sums[:, :, c] += s[:, y, :, x] - n * blk[y, x]
            counts[c] += 1
    mean = sums / (n[:, :, None] * np.maximum(counts, 1)) / max(float(white) - float(np.max(blk)), 1.0)
    cam = mean * np.asarray(wb, np.float64) / max(min(wb), 1e-6)
    return cam, peak >= 0.97 * float(white)


def _linear_means(linear, black, white, wb, crop, gh: int, gw: int):
    """:func:`_block_means` for full-colour data (camera RGB before white balance, as RawData's
    ``linear``)."""
    top, left, h, w = crop
    if h < gh or w < gw:
        return np.zeros((gh, gw, 3)), np.ones((gh, gw), bool)
    view = linear[top:top + h, left:left + w]
    r0 = (np.arange(gh) * h) // gh
    c0 = (np.arange(gw) * w) // gw
    s = np.add.reduceat(np.add.reduceat(view, r0, axis=0, dtype=np.float64), c0, axis=1)
    peak = np.maximum.reduceat(np.maximum.reduceat(view, r0, axis=0), c0, axis=1).max(axis=2)
    n = np.outer(np.diff(np.append(r0, h)), np.diff(np.append(c0, w)))
    blk = float(np.max(black))
    mean = (s / n[:, :, None] - blk) / max(float(white) - blk, 1.0)
    cam = mean * np.asarray(wb, np.float64) / max(min(wb), 1e-6)
    return cam, peak >= 0.97 * float(white)


def _trim(img: np.ndarray) -> np.ndarray:
    """The preview without the black bands a 4:3 thumbnail puts around a 3:2 picture (dark on
    average: the JPEG's ringing lifts a few of their pixels)."""
    bright = img.max(axis=2).astype(np.float64)
    lit_rows = np.nonzero(bright.mean(axis=1) >= 24)[0]
    lit_cols = np.nonzero(bright.mean(axis=0) >= 24)[0]
    if not len(lit_rows) or not len(lit_cols):
        return img[:0, :0]
    img = img[lit_rows[0]:lit_rows[-1] + 1, lit_cols[0]:lit_cols[-1] + 1]
    # the band edge is often a blend: drop one more line all round
    return img[1:-1, 1:-1] if min(img.shape[:2]) > 8 else img


def solve(cam: np.ndarray, ref: np.ndarray, prior: float = 0.5, toward: np.ndarray = GENERIC) -> tuple[np.ndarray, float] | None:
    """The 3x3 M (rows summing to 1) that best turns camera colours ``cam`` (n x 3, white
    balanced) into the directions of ``ref`` (n x 3, linear sRGB): M @ cam parallel to ref, by
    least squares on the cross products, pulled toward ``toward`` by ``prior`` (a share of the
    data's weight) and reweighted to shed outliers. With the residual left."""
    n = len(cam)
    if n < 50:
        return None
    x = cam / np.linalg.norm(cam, axis=1, keepdims=True)
    t = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    z = np.zeros_like(x)
    # (M x)_r t_g - (M x)_g t_r = 0 and (M x)_b t_g - (M x)_g t_b = 0; unknowns [m_r, m_g, m_b]
    a = np.vstack([np.hstack([x * t[:, 1:2], -x * t[:, 0:1], z]), np.hstack([z, -x * t[:, 2:3], x * t[:, 1:2]])])
    c = np.kron(np.eye(3), np.ones((1, 3)))
    t0 = np.asarray(toward, np.float64).reshape(-1)
    wgt = np.ones(len(a))
    theta = t0
    for _ in range(8):
        aw = a * wgt[:, None]
        lam = prior * float((aw * aw).sum()) / 9
        kkt = np.block([[aw.T @ aw + lam * np.eye(9), c.T], [c, np.zeros((3, 3))]])
        try:
            theta = np.linalg.solve(kkt, np.concatenate([lam * t0, np.ones(3)]))[:9]
        except np.linalg.LinAlgError:
            return None
        r = np.abs(a @ theta)
        scale = max(float(np.median(r)) * 3.0, 1e-9)
        wgt = 1.0 / np.maximum(1.0, r / scale)
    m = theta.reshape(3, 3)
    if not np.all(np.isfinite(m)) or abs(np.linalg.det(m)) < 1e-3:
        return None
    return m, float(np.median(np.abs(a @ theta)))


def fit_matrix(raw, preview_rgb: np.ndarray) -> np.ndarray | None:
    """Camera (white balanced) -> linear sRGB, 3x3, fitted from the camera's own rendering of
    the same picture: ``preview_rgb`` (h x w x 3 uint8 sRGB, any size; letterbox bands are cut
    off, and a preview turned upright is turned back). None when they can't be matched."""
    src = raw.cfa if raw.cfa is not None else raw.linear
    if src is None or preview_rgb is None or preview_rgb.ndim != 3 or preview_rgb.shape[2] < 3:
        return None
    full_h, full_w = src.shape[:2]
    crop = raw.crop or (0, 0, full_h, full_w)
    whole = np.asarray(preview_rgb)[:, :, :3]
    # the trimmed preview, or (a dark picture whose edges look like bands) the whole one
    shapes = [_trim(whole), whole]
    best = None
    for img, turn in ((s, t) for s in shapes for t in (0, 1, 3)):
        # many photosites per preview pixel, and not too many pixels to sum over
        while img.shape[0] * img.shape[1] > 40_000:
            hh, ww = img.shape[0] // 2 * 2, img.shape[1] // 2 * 2
            img = img[:hh, :ww].reshape(hh // 2, 2, ww // 2, 2, 3).mean(axis=(1, 3))
        cand = np.rot90(img, turn)
        gh, gw = cand.shape[:2]
        if gh < 16 or gw < 16 or abs(crop[3] / crop[2] - gw / gh) > 0.06 * crop[3] / crop[2]:
            continue
        if raw.cfa is not None:
            cam, clipped = _block_means(raw.cfa, np.asarray(raw.pattern), raw.black, raw.white, raw.wb, crop, gh, gw)
        else:
            cam, clipped = _linear_means(raw.linear, raw.black, raw.white, raw.wb, crop, gh, gw)
        ref8 = cand.astype(np.float64)
        ref = srgb_to_linear(ref8 / 255.0).astype(np.float64)
        # flat areas only: the preview's 3x3 neighbourhood barely changes
        pad = np.pad(ref8, ((1, 1), (1, 1), (0, 0)), mode="edge")
        nb = np.stack([pad[dy:dy + gh, dx:dx + gw] for dy in range(3) for dx in range(3)])
        flat = (nb.max(axis=0) - nb.min(axis=0)).max(axis=2) < 24
        ok = (flat & ~clipped & (ref8.max(axis=2) < 245) & (ref8.min(axis=2) > 10)
              & (cam.min(axis=2) > 0.003) & (cam.max(axis=2) < 0.9))
        got = solve(cam[ok], ref[ok])
        if got is not None and (best is None or got[1] < best[1]):
            best = got
    return None if best is None else best[0]


def cam_xyz(m: np.ndarray, wb) -> np.ndarray:
    """XYZ -> camera (RawData.cam_xyz, DNG's ColorMatrix sense) for a fitted camera -> sRGB
    matrix ``m`` and the white balance the camera colours were taken with."""
    wbv = np.asarray(wb, np.float64)
    return np.diag(wbv[1] / wbv) @ np.linalg.inv(m) @ np.linalg.inv(XYZ_RGB)


FITTED = "matrix: fitted from the camera's preview"
TYPICAL = "matrix: a typical camera's (the preview couldn't be fitted)"


def attach(raw, jpeg: bytes | None) -> None:
    """Give ``raw`` (which has no matrix of its own) one fitted from its preview JPEG, or the
    typical camera's, and say which in ``metadata["colour"]``."""
    m = None
    if jpeg:
        try:
            m = fit_matrix(raw, decode_jpeg(jpeg))
        except (ValueError, IndexError, KeyError, struct.error):
            m = None
    raw.cam_xyz = cam_xyz(GENERIC if m is None else m, raw.wb)
    raw.metadata["colour"] = TYPICAL if m is None else FITTED
