"""The picture every reader returns and every writer takes."""

from __future__ import annotations

import os
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

# sRGB's transfer curve (IEC 61966-2-1): linear light to the values a display expects
_A = 0.055
BAND = 64  # rows worked on at a time


def threads() -> int:
    """One per core (at most 8), or fewer when PIXPORT_THREADS says so."""
    try:
        cap = int(os.environ.get("PIXPORT_THREADS", "0"))
    except ValueError:
        cap = 0
    n = min(os.cpu_count() or 1, 8)
    return max(1, min(n, cap) if cap > 0 else n)


def each(work, items) -> list:
    """work(item) for every item, on several threads when there's more than one to do and more than
    one thread (zlib and NumPy let go of Python's lock while they work). Results keep their order."""
    items = list(items)
    if threads() > 1 and len(items) > 1:
        with ThreadPoolExecutor(threads()) as pool:
            return list(pool.map(work, items))
    return [work(item) for item in items]


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, (1 + _A) * np.power(x, 1 / 2.4, dtype=np.float32) - _A).astype(np.float32)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + _A) / (1 + _A), 2.4, dtype=np.float32)).astype(np.float32)


def upright_exif(exif: bytes | None) -> bytes | None:
    """exif (TIFF-structured) with its Orientation set to 1: the pixels are turned upright already,
    so a viewer mustn't turn them again. Anything it can't read is left as it is."""
    if not exif or len(exif) < 8 or exif[:2] not in (b"II", b"MM"):
        return exif
    e = "<" if exif[:2] == b"II" else ">"
    try:
        ifd = struct.unpack(e + "I", exif[4:8])[0]
        n = struct.unpack(e + "H", exif[ifd:ifd + 2])[0]
        for i in range(min(n, 1000)):
            at = ifd + 2 + 12 * i
            tag, typ = struct.unpack(e + "HH", exif[at:at + 4])
            if tag == 0x0112 and typ == 3:
                return exif[:at + 8] + struct.pack(e + "H", 1) + exif[at + 10:]
    except struct.error:
        pass
    return exif


@dataclass
class Image:
    """A decoded picture.

    ``pixels`` is (height, width, channels): 1 (grey), 2 (grey + alpha), 3 (RGB) or 4 (RGBA),
    already turned upright. ``transfer`` says what the numbers mean:

    - ``"linear"``: float32 scene-linear light in sRGB/Rec.709 primaries, 1.0 = diffuse white
      (EXR, HDR, RAW after colour conversion); may go above 1.
    - ``"srgb"``: display-referred values with the sRGB curve, float32 0..1 (XCF, JPEG XL, ...).

    ``metadata`` carries what a writer may copy on: ``exif`` (bytes, TIFF-structured, without the
    "Exif\\0\\0" prefix), ``icc`` (bytes), ``xmp`` (bytes), ``make``, ``model``, ``iso``,
    ``exposure``, ``dpi`` and anything format-specific."""

    pixels: np.ndarray
    transfer: str = "srgb"
    metadata: dict = field(default_factory=dict)
    format: str = ""

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def channels(self) -> int:
        return int(self.pixels.shape[2]) if self.pixels.ndim == 3 else 1

    @property
    def has_alpha(self) -> bool:
        return self.channels in (2, 4)

    def to_srgb(self, bits: int = 8, exposure: float = 0.0, tonemap: str = "clip") -> np.ndarray:
        """Display-ready sRGB as uint8 (bits=8) or uint16 (bits=16), alpha kept.

        For linear pictures ``exposure`` is in stops, and ``tonemap`` is ``"clip"`` (values over
        1 become white, as most viewers show an EXR) or ``"reinhard"`` (highlights rolled off)."""
        px = np.asarray(self.pixels, dtype=np.float32)
        if px.ndim == 2:
            px = px[:, :, None]
        top = 255 if bits == 8 else 65535
        linear = self.transfer == "linear"
        result = np.empty(px.shape, np.uint8 if bits == 8 else np.uint16)

        def band(y: int) -> None:
            part = px[y:y + BAND]
            with np.errstate(invalid="ignore"):  # not-a-number counts as something there
                blank = not part.any()
            if blank:  # as most of a rendered layer is: black stays black
                result[y:y + BAND] = 0
                return
            if not np.isfinite(part).all():  # an EXR can hold them: not-a-number is black, infinity as bright as it gets
                part = np.nan_to_num(part, nan=0.0, posinf=1e30, neginf=0.0)
            colour, alpha = (part[:, :, :-1], part[:, :, -1:]) if self.has_alpha else (part, None)
            if linear:
                if exposure:
                    colour = colour * np.float32(2.0 ** exposure)
                if tonemap == "reinhard":
                    colour = np.maximum(colour, 0.0)  # at -1 this would divide by zero
                    colour = colour / (1.0 + colour)
                colour = linear_to_srgb(colour)
            colour = np.clip(colour, 0.0, 1.0)
            out = colour if alpha is None else np.concatenate([colour, np.clip(alpha, 0.0, 1.0)], axis=2)
            result[y:y + BAND] = np.rint(out * top)

        each(band, range(0, px.shape[0], BAND))
        return result

    def to_pil(self, bits: int = 8, **kw):
        """A Pillow image (needs Pillow), 8 bits per channel, or 16 for grey (Pillow's I;16)."""
        from PIL import Image as PILImage
        arr = self.to_srgb(bits=8 if bits == 8 or self.channels != 1 else 16, **kw)
        mode = {1: "L", 2: "LA", 3: "RGB", 4: "RGBA"}[arr.shape[2]]
        if arr.dtype == np.uint16:
            return PILImage.fromarray(arr[:, :, 0], "I;16")
        img = PILImage.fromarray(arr if arr.shape[2] > 1 else arr[:, :, 0], mode)
        if self.metadata.get("icc"):
            img.info["icc_profile"] = self.metadata["icc"]
        return img
