"""PNG writing, 8 or 16 bits a channel (Pillow can't write 16-bit colour), with the photo's EXIF and
ICC profile carried over (eXIf and iCCP chunks).

Pictures are stored as small as they losslessly go: 16 bits that only hold 8 bits' worth become 8,
an alpha channel that's opaque everywhere is dropped, grey stays one channel, and 256 colours or
fewer become a palette (1, 2, 4 or 8 bits a pixel). Each check looks at a sample first, so a
photo, which passes none of them, costs next to nothing."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

from . import _accel
from .image import Image, each, threads, upright_exif

COLOUR_TYPE = {1: 0, 2: 4, 3: 2, 4: 6}  # channels -> grey, grey+alpha, RGB, RGBA


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _apply(rows: np.ndarray, prev: np.ndarray, bpp: int, kind: int) -> np.ndarray:
    """rows (uint8, n x stride) under PNG filter ``kind`` (0 None, 1 Sub, 2 Up, 4 Paeth), ``prev``
    being the row above the first one (zeros at the top)."""
    if kind == 0:
        return rows
    left = np.zeros_like(rows)
    left[:, bpp:] = rows[:, :-bpp]
    if kind == 1:
        return rows - left
    if kind == 4 and _accel.has("png_paeth"):
        n, stride = rows.shape
        got = _accel.native.png_paeth(np.ascontiguousarray(rows).tobytes(), np.ascontiguousarray(prev).tobytes(), stride, bpp)
        return np.frombuffer(got, np.uint8).reshape(n, stride)
    up = np.concatenate([prev[None], rows[:-1]])
    if kind == 2:
        return rows - up
    ul = np.zeros_like(rows)
    ul[:, bpp:] = up[:, :-bpp]
    a, b, c = left.astype(np.int16), up.astype(np.int16), ul.astype(np.int16)
    p = a + b - c
    pa, pb, pc = np.abs(p - a), np.abs(p - b), np.abs(p - c)
    pred = np.where((pa <= pb) & (pa <= pc), left, np.where(pb <= pc, up, ul))
    return rows - pred


WINDOW = 16 << 10  # bytes of each sampled row that the choice looks at


def _window(stride: int, bpp: int) -> slice:
    """Columns from the middle of a row, whole pixels, about WINDOW bytes."""
    if stride <= WINDOW:
        return slice(0, stride)
    start = (stride - WINDOW) // 2 // bpp * bpp
    return slice(start, start + WINDOW // bpp * bpp)


def _packed(rows: np.ndarray, prev: np.ndarray, bpp: int, kind: int, strategy: int, cols: slice) -> int:
    data = _apply(rows[:, cols], prev[cols], bpp, kind).tobytes()
    z = zlib.compressobj(6, zlib.DEFLATED, 15, 9, strategy)
    return len(z.compress(data) + z.flush())


def _by_row(rows: np.ndarray, prev: np.ndarray, bpp: int) -> tuple[np.ndarray, np.ndarray]:
    """rows (uint8, n x stride) filtered and the filter of each: every row under the filter that
    leaves the smallest values, the rule most PNG writers go by."""
    tries = np.stack([_apply(rows, prev, bpp, k) for k in (0, 1, 2, 4)])
    kinds = np.argmin(np.minimum(tries, -tries).sum(axis=2, dtype=np.uint64), axis=0)  # bytes as signed values
    return tries[kinds, np.arange(len(rows))], np.array((0, 1, 2, 4), np.uint8)[kinds]


def _best_filter(rows: np.ndarray, prev: np.ndarray, bpp: int, strategy: int, own: bool = True) -> int:
    """What packs these rows smallest, judged by packing a few of them (a window from the middle)
    each way: one filter for them all (Sub, Up, Paeth or None), or, if ``own``, OWN, each row's own."""
    cols = _window(rows.shape[1], bpp)
    take = rows[:6]
    sizes = {k: _packed(take, prev, bpp, k, strategy, cols) for k in (1, 2, 4, 0)}
    if not own:
        return min(sizes, key=sizes.get)
    z = zlib.compressobj(6, zlib.DEFLATED, 15, 9, strategy)
    own = _by_row(take[:, cols], prev[cols], bpp)[0].tobytes()
    sizes[OWN] = len(z.compress(own) + z.flush()) + len(take)  # a filter byte each, hardly packed
    return min(sizes, key=sizes.get)


def _strategy(rows: np.ndarray, bpp: int) -> int:
    """zlib's default strategy or Z_FILTERED, whichever packs a few bands of this picture smaller
    (each under its best filter)."""
    h, stride = rows.shape
    cols = _window(stride, bpp)
    strategies = (zlib.Z_DEFAULT_STRATEGY, zlib.Z_FILTERED)

    def trial(y: int) -> list[int]:
        band, prev = rows[y:y + 6], rows[y - 1] if y else np.zeros(stride, np.uint8)
        return [min(_packed(band, prev, bpp, k, strategy, cols) for k in (1, 2, 4, 0)) for strategy in strategies]

    sizes = np.sum(each(trial, sorted({int(v) for v in np.linspace(0, max(0, h - 6), 4)})), axis=0)
    return strategies[int(np.argmin(sizes))]


def filtered(rows: np.ndarray, bpp: int, kind: int | None = None, strategy: int = zlib.Z_DEFAULT_STRATEGY) -> bytes:
    """Rows of bytes (height x stride, uint8), each behind its PNG filter-type byte: ``kind``, or,
    band by band, what :func:`_best_filter` finds packs smallest with ``strategy``. Bands are worked
    out on several threads (NumPy lets go of Python's lock)."""
    h, stride = rows.shape
    out = np.empty((h, stride + 1), dtype=np.uint8)
    band = max(8, min(ROWS, PART // max(stride, 1)))

    def one(y: int) -> None:
        r = rows[y:y + band]
        prev = rows[y - 1] if y else np.zeros(stride, np.uint8)
        k = kind
        if k is None:
            k = _best_filter(r, prev, bpp, strategy, rows.size < PHOTO) if r.any() else 0  # blank, as most of a rendered layer is
        if k == OWN:
            out[y:y + band, 1:], out[y:y + band, 0] = _by_row(r, prev, bpp)
            return
        out[y:y + band, 0] = k
        if k == 1:  # the common case, straight into place
            out[y:y + band, 1:bpp + 1] = r[:, :bpp]
            np.subtract(r[:, bpp:], r[:, :-bpp], out=out[y:y + band, bpp + 1:], dtype=np.uint8, casting="unsafe")
            return
        out[y:y + band, 1:] = _apply(r, prev, bpp, k)

    each(one, range(0, h, band))
    return out.tobytes()


def _sample(a: np.ndarray) -> np.ndarray:
    """Some pixels from all over (rows and columns at a stride), to rule a reduction out cheaply."""
    h, w = a.shape[:2]
    return a[::max(1, h // 61), ::max(1, w // 67)]


def reduce(px: np.ndarray):
    """The smallest lossless form of an (h, w, c) uint8/uint16 picture: (array, colour type, depth,
    palette bytes or None, tRNS bytes or None). The array is what the rows are made of."""
    if px.dtype == np.uint16 and not np.any(_sample(px) % 257) and not np.any(px % 257):
        px = (px // 257).astype(np.uint8)  # 8-bit values stored as 16 bits (v * 257)
    c = px.shape[2]
    top = 65535 if px.dtype == np.uint16 else 255
    if c in (2, 4) and np.all(_sample(px)[:, :, -1] == top) and np.all(px[:, :, -1] == top):
        px, c = px[:, :, :-1], c - 1
    if c in (3, 4):
        s = _sample(px)
        if np.all(s[:, :, 0] == s[:, :, 1]) and np.all(s[:, :, 1] == s[:, :, 2]) \
                and np.array_equal(px[:, :, 0], px[:, :, 1]) and np.array_equal(px[:, :, 1], px[:, :, 2]):
            px = px[:, :, [0, 3]] if c == 4 else px[:, :, :1]
            c -= 2
    if px.dtype == np.uint8 and c in (3, 4):
        def packed(a):
            a = a.astype(np.uint32)
            v = (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]
            return v | (a[..., 3] << 24) if c == 4 else v
        if len(np.unique(packed(_sample(px)))) <= 256:
            colours, index = np.unique(packed(px).reshape(-1), return_inverse=True)
            if len(colours) <= 256:
                pal = np.stack([(colours >> 16) & 255, (colours >> 8) & 255, colours & 255], 1).astype(np.uint8)
                trns = None
                if c == 4:
                    alpha = (colours >> 24).astype(np.uint8)
                    order = np.argsort(alpha == 255, kind="stable")  # see-through colours first: a short tRNS
                    rank = np.empty_like(order)
                    rank[order] = np.arange(len(order))
                    pal, alpha, index = pal[order], alpha[order], rank[index]
                    k = int(np.count_nonzero(alpha < 255))
                    trns = alpha[:k].tobytes() if k else None
                n = len(colours)
                depth = 1 if n <= 2 else 2 if n <= 4 else 4 if n <= 16 else 8
                return index.reshape(px.shape[:2]).astype(np.uint8), 3, depth, pal.tobytes(), trns
    return px, COLOUR_TYPE[c], 16 if px.dtype == np.uint16 else 8, None, None


def _pack_bits(index: np.ndarray, depth: int) -> np.ndarray:
    """Palette indices (h x w) packed depth bits a pixel, rows padded to whole bytes."""
    if depth == 8:
        return index
    per = 8 // depth
    h, w = index.shape
    pad = (-w) % per
    a = np.pad(index, ((0, 0), (0, pad))).reshape(h, -1, per).astype(np.uint8)
    out = np.zeros(a.shape[:2], np.uint8)
    for k in range(per):
        out |= a[:, :, k] << (8 - depth * (k + 1))
    return out


PART = 1 << 20  # bytes each thread packs at a time
ROWS = 64  # rows that take one filter choice, at most
LEVEL = 6  # zlib's: 7 took EXRs 1.7 times as long for 0.4% less
PACK = 10  # libdeflate's, with pixport-native, below PHOTO: 6 to 9% smaller than zlib's 6 on HDR pictures
BLANK = 8  # a part that packs to under an eighth of its size is mostly blank
SMALL = 2 << 20  # bytes of rows under which a whole picture is also packed with each row's own filter
OWN = -1  # filtered(): every row under its own filter
PHOTO = 32 << 20  # bytes of rows from which bands don't weigh OWN: 10% of the time for 0.1% of the size


def _adler_join(first: int, second: int, size: int) -> int:
    """The Adler-32 of two runs of bytes, from each one's own and the second's length."""
    base = 65521
    a = ((first & 0xFFFF) + (second & 0xFFFF) - 1) % base
    b = ((first >> 16) + (second >> 16) + (size % base) * ((first & 0xFFFF) - 1)) % base
    return b << 16 | a


def _pack_level(raw: bytes) -> int:
    """libdeflate's level for these rows, a part from the middle deciding. Small pictures can take their
    time. Mostly blank ones (renders with a little in them) save few bytes for the time trying harder
    costs. Photos gain nothing from it: its 1 is about 4 times as fast as zlib's 6 and as small."""
    if len(raw) < SMALL:
        return PACK
    middle = raw[len(raw) // 2:len(raw) // 2 + PART]
    fast, usual = (len(_accel.native.png_deflate(middle, n, 1, PART)) for n in (1, 6))
    if usual * BLANK < len(middle):
        return 6
    if len(raw) < PHOTO:
        return PACK
    return 1 if fast <= usual * 1.01 else 6


def deflate(raw: bytes, level: int = LEVEL, strategy: int = zlib.Z_DEFAULT_STRATEGY) -> bytes:
    """A zlib stream of raw, packed on several threads at once (zlib lets go of Python's lock while
    it works): each part is its own deflate run, ended on a byte boundary, and the parts join into
    one stream any PNG reader takes, as pigz does. A part starts with the 32 KB before it to refer
    back to, so splitting costs next to nothing in size. With pixport-native, at the usual level,
    libdeflate packs instead: smaller, and its threads don't wait on each other for memory."""
    if level == LEVEL and _accel.has("png_deflate"):
        return _accel.native.png_deflate(raw, _pack_level(raw), threads(), PART)
    if threads() == 1 or len(raw) <= 2 * PART:
        z = zlib.compressobj(level, zlib.DEFLATED, 15, 9, strategy)
        return z.compress(raw) + z.flush()
    view = memoryview(raw)

    def pack(start: int) -> tuple[bytes, int, int]:
        z = zlib.compressobj(level, zlib.DEFLATED, -15, 9, strategy, bytes(view[start - (32 << 10):start])) if start \
            else zlib.compressobj(level, zlib.DEFLATED, -15, 9, strategy)
        last = start + PART >= len(raw)
        part = view[start:start + PART]
        return z.compress(part) + z.flush(zlib.Z_FINISH if last else zlib.Z_SYNC_FLUSH), zlib.adler32(part), len(part)

    parts = each(pack, range(0, len(raw), PART))
    check = 1
    for _, adler, size in parts:
        check = _adler_join(check, adler, size)
    return b"\x78\x9c" + b"".join(p[0] for p in parts) + struct.pack(">I", check)


def encode(pixels: np.ndarray, *, exif: bytes | None = None, icc: bytes | None = None, level: int = LEVEL) -> bytes:
    """A PNG of an (h, w, c) uint8 or uint16 array, in its smallest lossless form (:func:`reduce`)."""
    if pixels.ndim == 2:
        pixels = pixels[:, :, None]
    h, w, c = pixels.shape
    if c not in COLOUR_TYPE or pixels.dtype not in (np.uint8, np.uint16) or not (0 < w < 2**31 and 0 < h < 2**31):
        raise ValueError("PNG takes 1 to 4 channels of 8 or 16 bits")
    px, colour, depth, palette, trns = reduce(pixels)
    out = [b"\x89PNG\r\n\x1a\n", _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, colour, 0, 0, 0))]
    if icc:
        out.append(_chunk(b"iCCP", b"ICC Profile\x00\x00" + zlib.compress(icc)))
    if exif:
        out.append(_chunk(b"eXIf", exif))
    if palette is not None:
        out.append(_chunk(b"PLTE", palette))
        if trns:
            out.append(_chunk(b"tRNS", trns))
        data = np.ascontiguousarray(_pack_bits(px, depth))
        body = deflate(filtered(data, 1, 0), level)  # palette pictures pack best unfiltered
    else:
        nc = px.shape[2]
        data = np.ascontiguousarray(px.astype(">u2") if depth == 16 else px).view(np.uint8).reshape(h, -1)
        strategy = _strategy(data, nc * depth // 8)
        body = deflate(filtered(data, nc * depth // 8, None, strategy), level, strategy)
        if data.size <= SMALL and len(body) * 32 < data.size:  # test patterns and flat drawings: a few rows can mislead the choice here
            body = min(body, deflate(filtered(data, nc * depth // 8, OWN), level, strategy), key=len)
    for i in range(0, len(body), 1 << 26):  # IDAT chunks of up to 64 MB
        out.append(_chunk(b"IDAT", body[i:i + (1 << 26)]))
    out.append(_chunk(b"IEND", b""))
    return b"".join(out)


def write(img: Image, path: Path, bits: int | None = None, **opts) -> None:
    """bits: 8 or 16; by default 16 for pictures with more than 8 bits of their own (RAW, EXR, HDR,
    high-bit XCF), 8 otherwise."""
    if bits is None:
        bits = 16 if img.transfer == "linear" or img.metadata.get("bits", 8) > 8 else 8
    px = img.to_srgb(bits=bits, exposure=opts.get("exposure", 0.0), tonemap=opts.get("tonemap", "clip"))
    Path(path).write_bytes(encode(px, exif=upright_exif(img.metadata.get("exif")), icc=img.display_icc(),
                                  level=opts.get("level", LEVEL)))
