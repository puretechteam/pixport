"""Camera RAW: each maker's container and compression gives a :class:`RawData` (the sensor's
numbers and what is needed to turn them into a picture); :func:`develop` (``raw.develop``) makes
the picture: black and white levels, white balance, demosaicing, camera colour to sRGB, the
crop the camera itself applies, and the picture turned upright."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..image import Image

# fmt -> module in this package that decodes it
DECODERS = {"dng": "dng", "cr2": "canon", "cr3": "cr3", "nef": "nikon", "arw": "sony", "orf": "olympus",
            "rw2": "panasonic", "raf": "fuji", "pef": "pentax"}


@dataclass
class RawData:
    """What a RAW file holds, before developing.

    Exactly one of ``cfa`` (a mosaic: one value per photosite, uint16, height x width) or
    ``linear`` (already full colour, height x width x 3, as in sRAW/mRAW and linear DNGs) is set.

    - ``pattern``: the colour filter repeat as a 2D array of 0=R, 1=G, 2=B (2x2 for Bayer,
      6x6 for X-Trans), aligned with ``cfa[0, 0]``.
    - ``black``: black level, a scalar or an array the shape of ``pattern`` (per photosite in the
      repeat); ``black_rows``/``black_cols`` optional per-row/column extra offsets.
    - ``white``: the saturation level (after black subtraction it maps to 1.0).
    - ``wb``: as-shot white balance multipliers (R, G, B) normalised so G == 1.
    - ``cam_xyz``: 3x3 matrix XYZ(D65) -> camera RGB (DNG's ColorMatrix sense); ``None`` when
      unknown (then the picture is developed with an identity matrix and that is noted).
    - ``crop``: (top, left, height, width) of the picture the camera shows, inside ``cfa``.
    - ``orientation``: EXIF orientation 1-8.
    - ``metadata``: exif (bytes), make, model, iso, exposure, lens..., copied to the Image."""

    cfa: np.ndarray | None = None
    linear: np.ndarray | None = None
    pattern: np.ndarray = field(default_factory=lambda: np.array([[0, 1], [1, 2]], dtype=np.uint8))
    black: float | np.ndarray = 0.0
    black_rows: np.ndarray | None = None
    black_cols: np.ndarray | None = None
    white: float = 65535.0
    wb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    cam_xyz: np.ndarray | None = None
    crop: tuple[int, int, int, int] | None = None
    orientation: int = 1
    metadata: dict = field(default_factory=dict)


def decode(path: Path, fmt: str) -> RawData:
    mod = importlib.import_module(f"{__name__}.{DECODERS[fmt]}")
    return mod.decode(path, fmt)


def read(path: Path, fmt: str = "dng", **options) -> Image:
    from .develop import develop
    return develop(decode(path, fmt), **options)
