"""Pieces the RAW decoders share: size caps, bit unpacking, black from masked pixels and the
sensor's saturation point."""

from __future__ import annotations

import numpy as np

from ..errors import PixportError

MAX_SIDE = 50_000
MAX_PIXELS = 1 << 30


def check_size(width: int, height: int) -> None:
    if not isinstance(width, int) or not isinstance(height, int):
        raise PixportError("The picture's size is missing")
    if not (0 < width <= MAX_SIDE and 0 < height <= MAX_SIDE) or width * height > MAX_PIXELS:
        raise PixportError(f"A {width} x {height} picture is outside what pixport reads")


def enough(data: bytes, off: int, samples: int, bits_each: float, what: str = "sensor data") -> None:
    """A coded stream can't be shorter than its shortest codes: refuse one too small for the
    picture it claims (a hostile size would otherwise mean a long decode of nothing)."""
    if (len(data) - off) * 8 < samples * bits_each:
        raise PixportError(f"The {what} is too short for the picture's size")


def need(data: bytes, off: int, size: int, what: str = "sensor data") -> None:
    if off < 0 or size < 0 or off + size > len(data):
        raise PixportError(f"The {what} runs past the end of the file")


def unpack_bits(buf: bytes, count: int, bits: int, order: str = "msb") -> np.ndarray:
    """``count`` unsigned ``bits``-wide values packed back to back: ``"msb"`` (big-endian bit
    order, as most RAWs) or ``"lsb"`` (little-endian bit order)."""
    nbytes = (count * bits + 7) // 8
    if len(buf) < nbytes:
        buf = bytes(buf) + b"\0" * (nbytes - len(buf))
    a = np.frombuffer(buf, np.uint8, nbytes)
    if bits == 8:
        return a.astype(np.uint16)
    if bits == 16:
        return a.view(">u2" if order == "msb" else "<u2").astype(np.uint16)
    if bits == 12 and count % 2 == 0:
        t = a[: count // 2 * 3].reshape(-1, 3).astype(np.uint16)
        out = np.empty((t.shape[0], 2), np.uint16)
        if order == "msb":
            out[:, 0] = (t[:, 0] << 4) | (t[:, 1] >> 4)
            out[:, 1] = ((t[:, 1] & 15) << 8) | t[:, 2]
        else:
            out[:, 0] = t[:, 0] | ((t[:, 1] & 15) << 8)
            out[:, 1] = (t[:, 1] >> 4) | (t[:, 2] << 4)
        return out.reshape(-1)
    bitarr = np.unpackbits(a, bitorder="big" if order == "msb" else "little")[: count * bits].reshape(count, bits)
    w = (1 << np.arange(bits - 1, -1, -1)) if order == "msb" else (1 << np.arange(bits))
    return (bitarr.astype(np.uint32) @ w.astype(np.uint32)).astype(np.uint16)


def masked_black(img: np.ndarray, pattern: np.ndarray, left: int = 0, top: int = 0, mask=None) -> np.ndarray | float:
    """The black level per CFA position, from the optically masked pixels: ``mask`` (left, top,
    right, bottom, inclusive) when the camera names them, else the outer half of the columns left
    of the picture (or of the rows above it); 0 when the sensor has none. ``pattern`` is aligned
    with ``img[0, 0]``."""
    ph, pw = pattern.shape
    area, x0, y0 = None, 0, 0
    if mask is not None and len(mask) == 4 and mask[2] - mask[0] >= 3 and mask[3] - mask[1] >= 16:
        ml, mt, mr, mb = (int(v) for v in mask)
        if 0 <= ml and 0 <= mt and mr < img.shape[1] and mb < img.shape[0]:
            area, x0, y0 = img[mt:mb + 1, ml:mr + 1], ml, mt
    if area is None and left >= 8:
        area = img[:, :left // 2]
    elif area is None and top >= 8:
        area = img[:top // 2]
    if area is None or area.size < 64:
        return 0.0
    out = np.zeros((ph, pw))
    for y in range(ph):
        for x in range(pw):
            sub = area[(y - y0) % ph::ph, (x - x0) % pw::pw]
            out[y, x] = float(np.round(sub.mean())) if sub.size else 0.0
    return out


def saturation(img: np.ndarray, crop, nominal: float, black=0.0) -> float:
    """Where the sensor clips: the bit depth's maximum unless the picture shows a plateau below
    it (many photosites stuck at the same top value), which is then the white level."""
    top, left, h, w = crop
    vis = img[top:top + h, left:left + w]
    sample = vis[::4, ::4] if vis.size > 4_000_000 else vis
    mx = int(sample.max()) if sample.size else int(nominal)
    if mx >= nominal:
        return float(nominal)
    hi = int(np.count_nonzero(sample >= mx - max(1, mx // 400)))
    if hi >= max(16, sample.size // 20000) and mx > float(np.max(black)) + 64:
        return float(mx)
    return float(nominal)


def _blocks(a: np.ndarray, h: int, w: int) -> np.ndarray:
    """``a`` averaged down to h x w (trimming the remainder)."""
    by, bx = a.shape[0] // h, a.shape[1] // w
    return a[:by * h, :bx * w].reshape((h, by, w, bx) + a.shape[2:]).mean(axis=(1, 3))


def guess_bayer(img: np.ndarray, crop, wb, black=0.0, reference: np.ndarray | None = None) -> np.ndarray:
    """A Bayer layout read from the picture itself, for files that don't record one: the two
    greens are the diagonal pair that collects the most light; red and blue are told apart by
    which assignment matches the colours of ``reference`` (the camera's own small RGB rendering of
    the same picture) or, without one, which the as-shot white balance makes most neutral.
    Aligned with ``img[0, 0]``."""
    top, left, h, w = crop
    top, left = top & ~1, left & ~1
    vis = img[top:top + (h & ~1), left:left + (w & ~1)].astype(np.float32)
    blk = np.broadcast_to(np.asarray(black, dtype=np.float64), (2, 2)) if np.ndim(black) else np.full((2, 2), float(black))
    planes = [[vis[y::2, x::2] - np.float32(blk[(top + y) & 1, (left + x) & 1]) for x in range(2)] for y in range(2)]
    m = np.array([[max(1e-3, float(planes[y][x][::4, ::4].mean())) for x in range(2)] for y in range(2)])
    if m[0, 0] + m[1, 1] >= m[0, 1] + m[1, 0]:
        greens, others = ((0, 0), (1, 1)), ((0, 1), (1, 0))
    else:
        greens, others = ((0, 1), (1, 0)), ((0, 0), (1, 1))
    first_red = None
    if reference is not None and reference.ndim == 3 and reference.shape[2] >= 3 and min(reference.shape[:2]) >= 16:
        rh, rw = min(64, reference.shape[0]), min(96, reference.shape[1])
        ref = _blocks(reference[:, :, :3].astype(np.float64), rh, rw) + 1.0
        g = _blocks((planes[greens[0][0]][greens[0][1]] + planes[greens[1][0]][greens[1][1]]).astype(np.float64), rh, rw) / 2
        a = _blocks(planes[others[0][0]][others[0][1]].astype(np.float64), rh, rw)
        b = _blocks(planes[others[1][0]][others[1][1]].astype(np.float64), rh, rw)
        g, a, b = (np.maximum(x, 1.0) for x in (g, a, b))
        cr, cb = np.log(ref[:, :, 0] / ref[:, :, 1]).ravel(), np.log(ref[:, :, 2] / ref[:, :, 1]).ravel()
        ca, cbb = np.log(a / g).ravel(), np.log(b / g).ravel()

        def corr(x, y):
            x, y = x - x.mean(), y - y.mean()
            d = float(np.sqrt((x * x).sum() * (y * y).sum()))
            return float((x * y).sum()) / d if d > 0 else 0.0

        keep = corr(ca, cr) + corr(cbb, cb)
        swap = corr(cbb, cr) + corr(ca, cb)
        if abs(keep - swap) > 0.2:
            first_red = keep > swap
    if first_red is None:
        g = (m[greens[0]] + m[greens[1]]) / 2
        wr, wbb = max(wb[0], 1e-3), max(wb[2], 1e-3)
        a, b = m[others[0]], m[others[1]]
        first_red = abs(np.log(a * wr / g)) + abs(np.log(b * wbb / g)) <= abs(np.log(b * wr / g)) + abs(np.log(a * wbb / g))
    pat = np.ones((2, 2), dtype=np.uint8)
    red, blue = (others[0], others[1]) if first_red else (others[1], others[0])
    pat[red], pat[blue] = 0, 2
    # the planes started at (top, left); shift to img[0, 0]
    return np.roll(pat, (top & 1, left & 1), axis=(0, 1))


def padded_columns(img: np.ndarray, period: int = 2, limit: int = 64) -> int:
    """How many columns at the right edge only repeat the column ``period`` to their left (the
    padding some cameras write to round the row length up)."""
    h, w = img.shape[:2]
    n = 0
    rows = img[:: max(1, h // 512)]
    while n < min(limit, w - 2 * period):
        x = w - 1 - n
        if not np.array_equal(rows[:, x], rows[:, x - period]) or not np.array_equal(img[:, x], img[:, x - period]):
            break
        n += 1
    return n


def masked_edges(img: np.ndarray, black=0.0, white: float | None = None, limit: int = 64):
    """(top, left, height, width) without the optically masked rows and columns along the edges
    (each no brighter anywhere than just above black, then a clear step up to the picture), or
    None when there are none."""
    h, w = img.shape
    if h < 4 * limit or w < 4 * limit:
        return None
    blk = float(np.max(black)) if np.size(black) else 0.0
    top_val = float(white) if white else float(img[::8, ::8].max())
    thr = blk + max(8.0, 0.004 * (top_val - blk))
    sub_r, sub_c = img[:, :: max(1, w // 2048)], img[:: max(1, h // 2048), :]

    def run(lines) -> int:
        n = 0
        for k in range(limit):
            line = lines(k)
            if np.percentile(line, 99) > thr:
                break
            n += 1
        if n == 0 or n >= limit:
            return 0
        return n + (n & 1)  # keep the colour filter phase

    t = run(lambda k: sub_r[k])
    b = run(lambda k: sub_r[h - 1 - k])
    lft = run(lambda k: sub_c[:, k])
    rgt = run(lambda k: sub_c[:, w - 1 - k])
    if not (t or b or lft or rgt):
        return None
    return (t, lft, h - t - b, w - lft - rgt)
