"""Olympus / OM System ORF: a TIFF (magic "RO"/"RS") whose IFD0 strip holds the sensor data,
packed 12-bit or 16-bit, or in Olympus' adaptive coding; the maker note's ImageProcessing IFD
(0x2040) has the white balance (0x0100), the camera's own colour matrix (0x0200), the black
levels (0x0600) and the crop (0x0612-0x0615).

The coding works photosite by photosite, keeping three numbers per column parity (last
magnitude, a running average, a run of small values) that set how many low bits are sent
plainly; the high part is a unary count of zeros. The value is a difference from a prediction
made of the same-colour neighbours two to the left, above and above-left (a median-like choice
that switches to a plane fit across edges)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import _accel
from ..errors import PixportError
from . import RawData, colour
from .common import check_size, enough, need
from .tiff import Tiff, camera_metadata


def decode_compressed(data: bytes, off: int, width: int, height: int) -> np.ndarray:
    enough(data, off, width * height, 4, "ORF data")
    if _accel.has("olympus_decode"):
        return np.asarray(_accel.native.olympus_decode(data, off, width, height)).reshape(height, width)
    buf = data[off + 7:] + b"\0" * 16
    nbytes_bits = len(buf) * 8
    out = np.zeros((height, width), dtype=np.int64)
    pos = 0
    frm = int.from_bytes
    for row in range(height):
        acarry = [[0, 0, 0], [0, 0, 0]]
        r = out[row]
        up = out[row - 2] if row >= 2 else None
        line = [0] * width
        upl = up.tolist() if up is not None else None
        for col in range(width):
            carry = acarry[col & 1]
            i = 2 if carry[2] < 3 else 0
            nbits = 2 + i
            c0 = carry[0] & 0xFFFF
            while c0 >> (nbits + i):
                nbits += 1
            b = pos >> 3
            if b + 8 > len(buf):
                w = frm((buf[b:] + b"\0" * 8)[:8], "big")
            else:
                w = frm(buf[b:b + 8], "big")
            w = (w << (pos & 7)) & 0xFFFFFFFFFFFFFFFF
            b3 = w >> 61
            low = b3 & 3
            sign = -1 if b3 & 4 else 0
            w = (w << 3) & 0xFFFFFFFFFFFFFFFF
            pos += 3
            top12 = w >> 52
            if top12 == 0:
                pos += 12
                w = (w << 12) & 0xFFFFFFFFFFFFFFFF
                k = 16 - nbits
                high = (w >> (64 - k)) >> 1 if k > 0 else 0
                pos += max(k, 0)
                w = (w << max(k, 0)) & 0xFFFFFFFFFFFFFFFF
            else:
                high = 12 - top12.bit_length()
                pos += high + 1
                w = (w << (high + 1)) & 0xFFFFFFFFFFFFFFFF
            if nbits > 0:
                if nbits + (pos & 7) > 64:
                    # refill (rare: very long codes)
                    b = pos >> 3
                    w = (frm((buf[b:b + 8] + b"\0" * 8)[:8], "big") << (pos & 7)) & 0xFFFFFFFFFFFFFFFF
                lowbits = w >> (64 - nbits)
                pos += nbits
            else:
                lowbits = 0
            carry[0] = (high << nbits) | lowbits
            diff = (carry[0] ^ sign) + carry[1]
            carry[1] = (diff * 3 + carry[1]) >> 5
            carry[2] = 0 if carry[0] > 16 else carry[2] + 1
            if row < 2 and col < 2:
                pred = 0
            elif row < 2:
                pred = line[col - 2]
            elif col < 2:
                pred = upl[col]
            else:
                wv, nv, nw = line[col - 2], upl[col], upl[col - 2]
                if (wv < nw < nv) or (nv < nw < wv):
                    if abs(wv - nw) > 32 or abs(nv - nw) > 32:
                        pred = wv + nv - nw
                    else:
                        pred = (wv + nv) >> 1
                else:
                    pred = wv if abs(wv - nw) > abs(nv - nw) else nv
            line[col] = (pred + ((diff << 2) | low)) & 0xFFFF
        r[:] = line
        if pos > nbytes_bits + 64:
            raise PixportError("The ORF data ends early")
    return out.astype(np.uint16)


def decode(path: Path, fmt: str = "orf") -> RawData:
    data = path.read_bytes()
    t = Tiff(data)
    ifd = t.ifds[0] if t.ifds else None
    if ifd is None or not ifd.int(256):
        raise PixportError("This ORF has no sensor data")
    width, height = ifd.int(256), ifd.int(257)
    check_size(width, height)
    offs, counts = ifd.ints(273) or [], ifd.ints(279) or []
    if not offs:
        raise PixportError("This ORF's sensor data is missing")
    off, size = offs[0], sum(counts)
    need(data, off, 1)
    size = min(size, len(data) - off)
    n = width * height
    if size >= n * 2:
        cfa = np.frombuffer(data, "<u2" if t.order == "<" else ">u2", n, off).reshape(height, width).astype(np.uint16)
    elif size >= n * 3 // 2:
        # 12-bit packed, little-endian nibble order
        a = np.frombuffer(data, np.uint8, n * 3 // 2, off).reshape(-1, 3).astype(np.uint16)
        out = np.empty((a.shape[0], 2), np.uint16)
        out[:, 0] = a[:, 0] | ((a[:, 1] & 15) << 8)
        out[:, 1] = (a[:, 1] >> 4) | (a[:, 2] << 4)
        cfa = out.reshape(height, width)
    else:
        cfa = decode_compressed(data, off, width, height)
    md = camera_metadata(t)
    raw = RawData(cfa=cfa, white=4095.0, metadata=md)
    raw.orientation = ifd.int(274, 1) or 1
    mk = t.makernote()
    if mk and mk[0] == "olympus":
        mn = mk[1]
        ip = None
        if 0x2040 in mn:
            ip = t.ifd(mn.int(0x2040) + mn.base, mn.base, mn.order)
        if ip is not None:
            wb = ip.ints(0x0100)
            if wb and len(wb) >= 2 and wb[0] > 0 and wb[1] > 0:
                raw.wb = (wb[0] / 256.0, 1.0, wb[1] / 256.0)
            cm = ip.get(0x0200)
            if cm is not None and cm.count >= 9:
                m = (cm.values()[:9].astype(np.int64) & 0xFFFF)
                m = np.where(m >= 0x8000, m - 0x10000, m).reshape(3, 3) / 256.0
                raw.cam_xyz = colour.from_rgb_cam(m)
            bl = ip.ints(0x0600)
            if bl and len(bl) >= 4:
                # per colour (R, G, G, B) -> per position of the RGGB mosaic
                raw.black = np.array([[bl[0], bl[1]], [bl[2], bl[3]]], dtype=np.float64)
            vb = ip.ints(0x0611)
            if vb and 8 < vb[0] <= 16:
                raw.white = float((1 << vb[0]) - 1)
            crop = [ip.int(k) for k in (0x0612, 0x0613, 0x0614, 0x0615)]
            if all(v is not None for v in crop):
                cl, ct, cw, ch = crop
                if cw > 0 and ch > 0 and cl + cw <= width and ct + ch <= height:
                    raw.crop = (ct, cl, ch, cw)
    raw.metadata["bits"] = 12
    colour.attach(raw, t)
    return raw
