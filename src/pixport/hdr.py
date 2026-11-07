"""Radiance HDR (RGBE), written from Greg Ward's description of the format in "Graphics Gems II"
(Real Pixels) and the Radiance file-format notes.

A text header ("#?RADIANCE", then ``NAME=value`` lines, ended by a blank line), a resolution line
("-Y 512 +X 768": the slow axis first; the sign gives the direction), then the pixels: four bytes
each, an 8-bit mantissa per channel sharing one exponent byte. Scanlines are flat, in the old
run-length form (a (1, 1, 1, n) pixel repeats the one before), or in the newer form where each of
the four components is run-length coded on its own after a (2, 2, length) marker.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ._accel import native
from .errors import PixportError, UnsupportedError
from .image import Image, srgb_to_linear

MAX_SIDE = 50_000
MAX_PIXELS = 1_000_000_000
_RES = re.compile(rb"^([-+])([XY])\s+(\d+)\s+([-+])([XY])\s+(\d+)\s*$")
_SRGB_PRIMARIES = (0.640, 0.330, 0.300, 0.600, 0.150, 0.060, 0.3127, 0.3290)


def _header(data: bytes) -> tuple[dict, int, tuple]:
    """The header lines, where the pixels start and the resolution line's parts."""
    if not (data.startswith(b"#?")):
        raise PixportError("Not a Radiance HDR file (no #? signature)")
    info: dict = {"exposure": 1.0}
    pos = 0
    end = min(len(data), 65536)
    while True:
        nl = data.find(b"\n", pos, end)
        if nl < 0:
            raise PixportError("The HDR header is cut short")
        line = data[pos:nl].rstrip(b"\r")
        pos = nl + 1
        if not line:
            break
        if line.startswith(b"#"):
            continue
        key, eq, value = line.partition(b"=")
        if not eq:
            continue
        key = key.strip().upper()
        text = value.strip().decode("latin-1")
        if key == b"FORMAT":
            info["format"] = text
        elif key == b"EXPOSURE":
            try:
                info["exposure"] *= float(text)
            except ValueError:
                pass
        elif key == b"PRIMARIES":
            try:
                p = tuple(float(v) for v in text.split())
            except ValueError:
                p = ()
            if len(p) == 8:
                info["primaries"] = p
        elif key == b"COLORCORR":
            try:
                c = tuple(float(v) for v in text.split())
            except ValueError:
                c = ()
            if len(c) == 3:
                info["colorcorr"] = c
        elif key in (b"SOFTWARE", b"GAMMA", b"PIXASPECT", b"VIEW"):
            info[key.decode().lower()] = text
    nl = data.find(b"\n", pos, pos + 256)
    if nl < 0:
        raise PixportError("The HDR resolution line is missing")
    m = _RES.match(data[pos:nl].rstrip(b"\r"))
    if not m or m.group(2) == m.group(5):
        raise PixportError("The HDR resolution line is damaged")
    return info, nl + 1, m.groups()


def _ops_new(data: bytes, pos: int, width: int, starts: list, lengths: list, steps: list) -> int:
    """Collects one new-style scanline's four run-length coded components: where each run or
    literal comes from in data and how long it is. Returns where the next scanline starts."""
    n = len(data)
    for _ in range(4):
        left = width
        while left > 0:
            if pos >= n:
                raise PixportError("The HDR pixel data is cut short")
            c = data[pos]
            if c > 128:
                c -= 128
                if c > left or pos + 1 >= n:
                    raise PixportError("The HDR pixel data is damaged (a run overflows its scanline)")
                starts.append(pos + 1)
                steps.append(0)
                pos += 2
            else:
                if c == 0 or c > left or pos + 1 + c > n:
                    raise PixportError("The HDR pixel data is damaged (a bad literal run)")
                starts.append(pos + 1)
                steps.append(1)
                pos += 1 + c
            lengths.append(c)
            left -= c
    return pos


def _expand(data: bytes, starts: list, lengths: list, steps: list, total: int) -> np.ndarray:
    """The bytes the (start, length, step) list describes, all at once."""
    src = np.frombuffer(data, np.uint8)
    st = np.asarray(starts, np.int64)
    ln = np.asarray(lengths, np.int64)
    sp = np.asarray(steps, np.int64)
    if int(ln.sum()) != total:
        raise PixportError("The HDR pixel data is damaged")
    first = np.repeat(np.cumsum(ln) - ln, ln)
    ramp = np.arange(total, dtype=np.int64) - first
    return src[np.repeat(st, ln) + ramp * np.repeat(sp, ln)]


def _decode_old(data: bytes, pos: int, width: int) -> tuple[np.ndarray, int]:
    """One scanline in the flat or old run-length form (4-byte pixels, (1,1,1,n) repeats)."""
    out = np.empty((width, 4), np.uint8)
    n = len(data)
    i = 0
    shift = 0
    while i < width:
        if pos + 4 > n:
            raise PixportError("The HDR pixel data is cut short")
        px = data[pos:pos + 4]
        pos += 4
        if px[0] == 1 and px[1] == 1 and px[2] == 1:
            if i == 0:
                raise PixportError("The HDR pixel data is damaged (a repeat with nothing before it)")
            run = px[3] << shift
            if i + run > width:
                raise PixportError("The HDR pixel data is damaged (a repeat overflows its scanline)")
            out[i:i + run] = out[i - 1]
            i += run
            shift += 8
            if shift > 24:
                raise PixportError("The HDR pixel data is damaged (a repeat that never ends)")
        else:
            out[i] = np.frombuffer(px, np.uint8)
            i += 1
            shift = 0
    return out, pos


def decode_rgbe(data: bytes, pos: int, lines: int, width: int) -> np.ndarray:
    """(lines, width, 4) RGBE bytes from the pixel data at pos."""
    if native is not None and hasattr(native, "hdr_decode"):
        return np.frombuffer(native.hdr_decode(data, pos, lines, width), np.uint8).reshape(lines, width, 4)
    rows = []
    starts: list = []
    lengths: list = []
    steps: list = []
    new_rows: list = []
    n = len(data)
    for y in range(lines):
        if 8 <= width <= 0x7FFF and pos + 4 <= n and data[pos] == 2 and data[pos + 1] == 2 and data[pos + 2] < 128:
            if (data[pos + 2] << 8 | data[pos + 3]) != width:
                raise PixportError("The HDR pixel data is damaged (a scanline of the wrong length)")
            pos = _ops_new(data, pos + 4, width, starts, lengths, steps)
            new_rows.append(y)
            rows.append(None)
        else:
            row, pos = _decode_old(data, pos, width)
            rows.append(row)
    out = np.empty((lines, width, 4), np.uint8)
    if new_rows:
        planes = _expand(data, starts, lengths, steps, len(new_rows) * 4 * width)
        out[new_rows] = planes.reshape(len(new_rows), 4, width).transpose(0, 2, 1)
    for y, row in enumerate(rows):
        if row is not None:
            out[y] = row
    return out


def rgbe_to_float(rgbe: np.ndarray) -> np.ndarray:
    """Radiance's own conversion: (mantissa + 0.5) * 2^(exponent - 136); exponent 0 is black."""
    e = rgbe[..., 3].astype(np.int32)
    scale = np.where(e == 0, 0.0, np.ldexp(np.float32(1.0), e - 136)).astype(np.float32)
    out = (rgbe[..., :3].astype(np.float32) + np.float32(0.5)) * scale[..., None]
    return out


def float_to_rgbe(rgb: np.ndarray) -> np.ndarray:
    """The inverse of rgbe_to_float (values under 1e-32 and below zero become black)."""
    if native is not None and hasattr(native, "hdr_rgbe"):
        flat = np.ascontiguousarray(rgb, "<f4")
        return np.frombuffer(native.hdr_rgbe(flat.tobytes()), np.uint8).reshape(flat.shape[:-1] + (4,))
    rgb = np.nan_to_num(np.asarray(rgb, np.float32), nan=0.0, posinf=1e30, neginf=0.0)
    np.clip(rgb, 0.0, 1e38, out=rgb)
    v = rgb.max(axis=-1)
    _mant, exp = np.frexp(v)
    small = v < 1e-32
    exp = np.where(small, 0, np.clip(exp, -127, 127))
    # v = mant * 2^exp, so the mantissas are rgb * 256 / 2^exp: an exact power-of-two scaling
    scale = np.ldexp(np.float32(1), (8 - exp).astype(np.int32)).astype(np.float32)
    out = np.empty(rgb.shape[:-1] + (4,), np.uint8)
    out[..., :3] = np.minimum(np.floor(rgb * scale[..., None]), 255)
    out[..., :3][small] = 0
    out[..., 3] = np.where(small, 0, exp + 128)
    return out


_LIN8 = None


def linearise(x: np.ndarray) -> np.ndarray:
    """sRGB-curved values to linear light; 8-bit-exact inputs (k / 255) go through a table."""
    global _LIN8
    x = np.asarray(x, np.float32)
    q = np.rint(x * np.float32(255))
    if x.size and np.array_equal(q, x * np.float32(255)) and q.min() >= 0 and q.max() <= 255:
        if _LIN8 is None:
            _LIN8 = srgb_to_linear(np.arange(256, dtype=np.float32) / np.float32(255))
        return _LIN8[q.astype(np.uint8)]
    return srgb_to_linear(x)


def _to_srgb_primaries(rgb: np.ndarray, prim: tuple) -> np.ndarray:
    from .exr import primaries_matrix
    m = primaries_matrix(prim)
    if m is None:
        return rgb
    return np.einsum("ij,hwj->hwi", m.astype(np.float32), rgb).astype(np.float32)


def read(path: Path, fmt: str = "hdr", **_options) -> Image:
    data = Path(path).read_bytes()
    info, pos, res = _header(data)
    s1, a1, n1, s2, _a2, n2 = res
    n1, n2 = int(n1), int(n2)
    if n1 < 1 or n2 < 1 or n1 > MAX_SIDE or n2 > MAX_SIDE or n1 * n2 > MAX_PIXELS:
        raise PixportError(f"The HDR picture's size ({n2} x {n1}) is out of range")
    fmt_name = info.get("format", "32-bit_rle_rgbe")
    if fmt_name not in ("32-bit_rle_rgbe", "32-bit_rle_xyze"):
        raise UnsupportedError(f"pixport doesn't read HDR pixels in the {fmt_name} format")
    rgb = rgbe_to_float(decode_rgbe(data, pos, n1, n2))
    if a1 == b"Y":
        if s1 == b"+":
            rgb = rgb[::-1]
        if s2 == b"-":
            rgb = rgb[:, ::-1]
    else:
        rgb = rgb.transpose(1, 0, 2)
        if s1 == b"-":
            rgb = rgb[:, ::-1]
        if s2 == b"+":
            rgb = rgb[::-1]
    rgb = np.ascontiguousarray(rgb)
    if "colorcorr" in info:
        rgb = rgb / np.asarray(info["colorcorr"], np.float32)
    meta = {"hdr_exposure": info["exposure"]}
    for k in ("software", "gamma", "pixaspect", "view"):
        if k in info:
            meta["hdr_" + k] = info[k]
    if fmt_name == "32-bit_rle_xyze":
        from .exr import XYZ_TO_SRGB
        rgb = np.einsum("ij,hwj->hwi", XYZ_TO_SRGB.astype(np.float32), rgb).astype(np.float32)
        meta["hdr_primaries"] = "XYZ"
    elif "primaries" in info:
        meta["hdr_primaries"] = info["primaries"]
        rgb = _to_srgb_primaries(rgb, info["primaries"])
    return Image(rgb.astype(np.float32, copy=False), "linear", meta, "hdr")


def _encode_component(row: np.ndarray, out: bytearray) -> None:
    """One component of a scanline in the new run-length form: runs of 3 or more as runs (a run
    of 3 costs no more than the literal bytes it replaces, and usually saves a count byte)."""
    n = row.size
    if n == 0:
        return
    change = np.flatnonzero(row[1:] != row[:-1]) + 1
    bounds = np.concatenate(([0], change, [n]))
    lit = 0  # where the pending literal stretch starts
    for a, b in zip(bounds[:-1].tolist(), bounds[1:].tolist()):
        if b - a < 3:
            continue
        while lit < a:
            k = min(128, a - lit)
            out.append(k)
            out += row[lit:lit + k].tobytes()
            lit += k
        v = int(row[a])
        while b - a >= 3:
            k = min(127, b - a)
            out.append(128 + k)
            out.append(v)
            a += k
        lit = a
    while lit < n:
        k = min(128, n - lit)
        out.append(k)
        out += row[lit:lit + k].tobytes()
        lit += k


def encode_rgbe(rgbe: np.ndarray) -> bytes:
    """New-style run-length scanlines for (height, width, 4) RGBE bytes."""
    h, w = rgbe.shape[:2]
    if native is not None and hasattr(native, "hdr_encode"):
        return bytes(native.hdr_encode(np.ascontiguousarray(rgbe).tobytes(), h, w))
    if not 8 <= w <= 0x7FFF:
        return np.ascontiguousarray(rgbe).tobytes()
    out = bytearray()
    planes = np.ascontiguousarray(rgbe.transpose(0, 2, 1))
    head = bytes((2, 2, w >> 8, w & 255))
    for y in range(h):
        out += head
        for c in range(4):
            _encode_component(planes[y, c], out)
    return bytes(out)


def write(img: Image, path: Path, **_options) -> None:
    """Writes img as run-length coded RGBE (top row first); alpha is dropped."""
    px = np.asarray(img.pixels, np.float32)
    if px.ndim == 2:
        px = px[:, :, None]
    c = px.shape[2]
    colour = px[:, :, :3] if c >= 3 else np.repeat(px[:, :, :1], 3, axis=2)
    if img.transfer != "linear":
        colour = linearise(colour)
    h, w = colour.shape[:2]
    if h < 1 or w < 1:
        raise PixportError("Can't write an empty picture")
    data = encode_rgbe(float_to_rgbe(colour))
    head = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\nSOFTWARE=pixport\n\n" + f"-Y {h} +X {w}\n".encode()
    Path(path).write_bytes(head + data)
