"""Which formats pixport reads and writes, telling them apart by content, and read()/write()."""

from __future__ import annotations

import importlib
import os
import struct
import zlib
from pathlib import Path

from .errors import PixportError
from .image import Image

# what a damaged file makes the readers raise; read() turns these into PixportError
DAMAGE = (struct.error, LookupError, ValueError, ArithmeticError, EOFError, zlib.error, MemoryError, RecursionError,
          NotImplementedError, OverflowError)

# fmt: (module, name, readable, writable)
FORMATS: dict[str, tuple[str, str, bool, bool]] = {
    "dng": ("raw", "Adobe Digital Negative (DNG)", True, False),
    "cr2": ("raw", "Canon RAW 2 (CR2)", True, False),
    "cr3": ("raw", "Canon RAW 3 (CR3)", True, False),
    "nef": ("raw", "Nikon RAW (NEF)", True, False),
    "arw": ("raw", "Sony RAW (ARW)", True, False),
    "orf": ("raw", "Olympus / OM System RAW (ORF)", True, False),
    "rw2": ("raw", "Panasonic RAW (RW2)", True, False),
    "raf": ("raw", "Fujifilm RAW (RAF)", True, False),
    "pef": ("raw", "Pentax RAW (PEF)", True, False),
    "xcf": ("xcf", "GIMP image (XCF)", True, False),
    "exr": ("exr", "OpenEXR", True, True),
    "hdr": ("hdr", "Radiance HDR (RGBE)", True, True),
    "jxl": ("jxl", "JPEG XL", True, True),
    "png": ("png", "PNG (8 or 16 bits)", False, True),
    "tiff": ("tiffwrite", "TIFF (8 or 16 bits)", False, True),
    "tif": ("tiffwrite", "TIFF (8 or 16 bits)", False, True),
}
RAW = {k for k, v in FORMATS.items() if v[0] == "raw"}


def formats() -> dict[str, dict]:
    """{format: {"name", "read", "write"}}."""
    return {k: {"name": n, "read": r, "write": w} for k, (_m, n, r, w) in FORMATS.items()}


def _module(name: str):
    return importlib.import_module(f"pixport.{name}")


def detect(path: str | os.PathLike) -> str:
    """The format of a file, from its content first and its extension second."""
    p = Path(path)
    try:
        with open(p, "rb") as f:
            head = f.read(64)
    except OSError as e:
        raise PixportError(f"Can't open {p}: {e}") from None
    ext = p.suffix.lower().lstrip(".")
    if head[:4] == b"\x76\x2f\x31\x01":
        return "exr"
    if head[:10] == b"#?RADIANCE" or head[:6] == b"#?RGBE":
        return "hdr"
    if head[:9] == b"gimp xcf ":
        return "xcf"
    if head[:2] == b"\xff\x0a" or head[:12] == b"\x00\x00\x00\x0cJXL \r\n\x87\n":
        return "jxl"
    if head[4:12] == b"ftypcrx ":
        return "cr3"
    if head[:16] == b"FUJIFILMCCD-RAW ":
        return "raf"
    if head[:4] in (b"IIRO", b"IIRS", b"MMOR"):
        return "orf"
    if head[:4] == b"IIU\x00":
        return "rw2"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        if ext in RAW:
            return ext
        if head[8:10] == b"CR":
            return "cr2"
        return "dng" if ext not in RAW else ext
    if ext in FORMATS:
        return ext
    raise PixportError(f"{p.name} isn't a picture pixport reads")


def read(path: str | os.PathLike, fmt: str | None = None, **options) -> Image:
    """Decode a picture. ``options`` go to the format's reader (for RAW: ``half=True`` for a
    quick half-size picture, ``white_balance="camera"|"auto"|"daylight"``)."""
    fmt = (fmt or detect(path)).lower()
    entry = FORMATS.get(fmt)
    if entry is None or not entry[2]:
        raise PixportError(f"pixport doesn't read {fmt}")
    try:
        img = _module(entry[0]).read(Path(path), fmt=fmt, **options)
    except PixportError:
        raise
    except OSError as e:
        raise PixportError(f"Can't read {Path(path).name}: {e}") from None
    except DAMAGE as e:
        raise PixportError(f"{Path(path).name} looks damaged ({type(e).__name__}: {e})") from None
    img.format = img.format or fmt
    return img


def write(img: Image, path: str | os.PathLike, fmt: str | None = None, **options) -> None:
    """Encode a picture as EXR, HDR or JPEG XL (from the extension unless ``fmt`` is given)."""
    fmt = (fmt or Path(path).suffix.lstrip(".")).lower()
    entry = FORMATS.get(fmt)
    if entry is None or not entry[3]:
        raise PixportError(f"pixport doesn't write {fmt}")
    _module(entry[0]).write(img, Path(path), **options)
