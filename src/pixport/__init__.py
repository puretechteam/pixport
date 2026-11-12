"""pixport: camera RAW, GIMP XCF, OpenEXR, Radiance HDR and JPEG XL pictures in Python with NumPy.

    import pixport
    img = pixport.read("photo.cr3")          # an Image: pixels as a NumPy array, plus metadata
    rgb = img.to_srgb(bits=8)                # display-ready sRGB, (height, width, 3) uint8
    pixport.write(img, "photo.exr")          # EXR, HDR and JPEG XL can be written

Formats (``pixport.formats()``): RAW from Canon (CR2, CR3), Nikon (NEF), Sony (ARW), Olympus/OM
(ORF), Panasonic (RW2), Fujifilm (RAF, Bayer and X-Trans), Pentax (PEF) and any DNG; GIMP's XCF
(layers composited as GIMP shows them); OpenEXR; Radiance HDR; JPEG XL.
"""

from __future__ import annotations

__version__ = "0.2.0"

from .errors import PixportError, UnsupportedError
from .image import Image
from .registry import detect, formats, read, write

__all__ = ["Image", "PixportError", "UnsupportedError", "__version__", "detect", "formats", "read", "write"]
