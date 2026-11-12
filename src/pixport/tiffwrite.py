"""TIFF writing, 8 or 16 bits a channel (Pillow can't write 16-bit colour): Deflate-compressed strips
with the horizontal predictor, the ICC profile, and the camera's make and model and the date from the
photo (as TIFF tags; the rest of the EXIF isn't carried)."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

from .image import Image

ROWS_PER_STRIP = 64
ASCII, SHORT, LONG, UNDEFINED = 2, 3, 4, 7


def _predict(block: np.ndarray, c: int) -> np.ndarray:
    """TIFF predictor 2: each sample minus the same channel one pixel to the left (wrapping)."""
    out = block.copy()
    out[:, c:] = block[:, c:] - block[:, :-c]
    return out


def encode(pixels: np.ndarray, *, icc: bytes | None = None, text: dict[int, str] | None = None, level: int = 6) -> bytes:
    if pixels.ndim == 2:
        pixels = pixels[:, :, None]
    h, w, c = pixels.shape
    if c not in (1, 2, 3, 4) or pixels.dtype not in (np.uint8, np.uint16):
        raise ValueError("TIFF takes 1 to 4 channels of 8 or 16 bits")
    bits = 16 if pixels.dtype == np.uint16 else 8
    flat = np.ascontiguousarray(pixels).reshape(h, w * c)
    strips = []
    for y in range(0, h, ROWS_PER_STRIP):
        block = _predict(flat[y:y + ROWS_PER_STRIP], c)
        strips.append(zlib.compress(block.astype("<u2" if bits == 16 else np.uint8).tobytes(), level))
    entries: list[tuple[int, int, int, bytes]] = []  # tag, type, count, value bytes

    def add(tag, typ, values):
        if typ in (ASCII, UNDEFINED):
            data = values if isinstance(values, bytes) else values.encode("latin-1", "replace") + b"\0"
            entries.append((tag, typ, len(data), data))
        else:
            fmt = {SHORT: "<H", LONG: "<I"}[typ]
            entries.append((tag, typ, len(values), b"".join(struct.pack(fmt, v) for v in values)))

    photometric = 2 if c >= 3 else 1
    add(256, LONG, [w])
    add(257, LONG, [h])
    add(258, SHORT, [bits] * c)
    add(259, SHORT, [8])  # Deflate
    add(262, SHORT, [photometric])
    for tag, value in sorted((text or {}).items()):
        if value:
            add(tag, ASCII, value)
    add(273, LONG, [0] * len(strips))
    add(274, SHORT, [1])  # upright
    add(277, SHORT, [c])
    add(278, LONG, [ROWS_PER_STRIP])
    add(279, LONG, [len(s) for s in strips])
    add(284, SHORT, [1])
    add(317, SHORT, [2])  # horizontal predictor
    if c in (2, 4):
        add(338, SHORT, [2])  # unassociated alpha
    if icc:
        add(34675, UNDEFINED, icc)
    entries.sort(key=lambda e: e[0])
    ifd_size = 2 + 12 * len(entries) + 4
    pos = 8 + ifd_size
    extra = []
    for _tag, _typ, _count, data in entries:  # values over 4 bytes go after the IFD
        if len(data) > 4:
            extra.append(data + (b"\0" if len(data) % 2 else b""))
    data_start = pos + sum(len(x) for x in extra)
    offsets, at = [], data_start
    for s in strips:
        offsets.append(at)
        at += len(s)
    i = next(i for i, e in enumerate(entries) if e[0] == 273)
    entries[i] = (273, LONG, len(offsets), b"".join(struct.pack("<I", o) for o in offsets))
    ifd = [struct.pack("<H", len(entries))]
    blobs, at = [], pos
    for tag, typ, count, data in entries:
        if len(data) <= 4:
            ifd.append(struct.pack("<HHI", tag, typ, count) + data.ljust(4, b"\0"))
        else:
            ifd.append(struct.pack("<HHII", tag, typ, count, at))
            padded = data + (b"\0" if len(data) % 2 else b"")
            blobs.append(padded)
            at += len(padded)
    ifd.append(struct.pack("<I", 0))
    return b"II*\0" + struct.pack("<I", 8) + b"".join(ifd) + b"".join(blobs) + b"".join(strips)


def write(img: Image, path: Path, bits: int | None = None, **opts) -> None:
    if bits is None:
        bits = 16 if img.transfer == "linear" or img.metadata.get("bits", 8) > 8 else 8
    px = img.to_srgb(bits=bits, exposure=opts.get("exposure", 0.0), tonemap=opts.get("tonemap", "clip"))
    md = img.metadata
    text = {271: str(md.get("make", "") or ""), 272: str(md.get("model", "") or ""),
            306: str(md.get("datetime", "") or ""), 305: "pixport"}
    Path(path).write_bytes(encode(px, icc=img.display_icc(), text=text, level=opts.get("level", 6)))
