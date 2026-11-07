"""pixport on the command line: ``pixport photo.cr3 photo.png`` (JPEG and the formats pixport
doesn't write itself need Pillow)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import PixportError, __version__, formats, read, write


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pixport", description="Convert camera RAW, XCF, EXR, HDR and JPEG XL pictures.")
    ap.add_argument("src", nargs="?")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--bits", type=int, choices=(8, 16), default=8, help="bits per channel for PNG/TIFF (default 8)")
    ap.add_argument("--exposure", type=float, default=0.0, help="stops brighter (+) or darker (-) for EXR/HDR")
    ap.add_argument("--formats", action="store_true", help="list the formats and exit")
    ap.add_argument("--version", action="version", version=f"pixport {__version__}")
    a = ap.parse_args(argv)
    if a.formats:
        for k, v in formats().items():
            print(f"{k:5} {'read' if v['read'] else '    '} {'write' if v['write'] else '     '}  {v['name']}")
        return 0
    if not a.src or not a.dst:
        ap.error("give a picture to read and a file to write")
    try:
        img = read(a.src)
        dst = Path(a.dst)
        if dst.suffix.lower().lstrip(".") in {k for k, v in formats().items() if v["write"]}:
            write(img, dst)
        else:
            pil = img.to_pil(bits=a.bits, exposure=a.exposure)
            pil.save(dst)
    except PixportError as e:
        print(f"pixport: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
