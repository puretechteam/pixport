"""Sony ARW: a TIFF with the mosaic in a SubIFD, stored
uncompressed (14-bit in 16-bit words, or 12-bit packed), as Sony's "compressed RAW" (ARW2:
blocks of 16 same-colour photosites as an 11-bit max and min, their positions and 7-bit deltas,
then a tone curve from tag 0x7010), or as lossless JPEG tiles (ARW 4, compression 7).

White balance, black and white levels are in the SR2SubIFD, which is encrypted: the key from
SR2Private (found through DNGPrivateData, 0xC634) seeds a 127-word pad generator (a linear
congruential start, then a shift-register recurrence), and the keystream is XORed in."""

from __future__ import annotations

import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .. import _accel
from ..errors import PixportError, UnsupportedError
from . import RawData, colour, ljpeg
from .common import check_size, need, padded_columns, unpack_bits
from .tiff import Tiff, camera_metadata, read_ifd


def sony_curve(points) -> np.ndarray:
    """The 4096-entry curve of tag 0x7010: slope 1, 2, 4, 8, 16 between its knots."""
    knots = [0] + [(int(p) >> 2) & 0xFFF for p in points[:4]] + [4095]
    curve = np.arange(0x4001, dtype=np.int64)
    for i in range(5):
        for j in range(knots[i] + 1, min(knots[i + 1], 0x4000) + 1):
            curve[j] = curve[j - 1] + (1 << i)
    return curve


def decrypt(buf: bytes, key: int) -> bytes:
    """SR2SubIFD decryption (whole 32-bit words; a trailing partial word is left as is)."""
    pad = [0] * 128
    k = key & 0xFFFFFFFF
    for p in range(4):
        k = (k * 48828125 + 1) & 0xFFFFFFFF
        pad[p] = k
    pad[3] = ((pad[3] << 1) | ((pad[0] ^ pad[2]) >> 31)) & 0xFFFFFFFF
    for p in range(4, 127):
        pad[p] = (((pad[p - 4] ^ pad[p - 2]) << 1) | ((pad[p - 3] ^ pad[p - 1]) >> 31)) & 0xFFFFFFFF
    n = len(buf) // 4
    words = np.frombuffer(buf, ">u4", n).copy()
    stream = np.empty(n, np.uint32)
    p = 127
    for i in range(n):
        v = pad[(p + 1) & 127] ^ pad[(p + 65) & 127]
        pad[p & 127] = v
        stream[i] = v
        p += 1
    return (words ^ stream).astype(">u4").tobytes() + buf[n * 4:]


def arw2(data: bytes, off: int, width: int, height: int, curve: np.ndarray) -> np.ndarray:
    """Sony's compressed RAW: each row is ``width`` bytes of 16-byte blocks, each block 16
    photosites of one colour (every other column)."""
    need(data, off, width * height, "ARW data")
    if _accel.has("sony_arw2"):
        vals = np.asarray(_accel.native.sony_arw2(data, off, width, height)).reshape(height, width)
    else:
        nblk = min(width // 16, 2 * -(-(width - 30) // 32)) if width > 30 else 0
        rows = np.frombuffer(data, np.uint8, width * height, off).reshape(height, width)
        blocks = rows[:, :nblk * 16].reshape(height, nblk, 16)
        b = blocks.astype(np.uint32)
        head = b[:, :, 0] | (b[:, :, 1] << 8) | (b[:, :, 2] << 16) | (b[:, :, 3] << 24)
        mx = head & 0x7FF
        mn = (head >> 11) & 0x7FF
        imax = (head >> 22) & 0xF
        imin = (head >> 26) & 0xF
        rng = mx.astype(np.int64) - mn.astype(np.int64)
        sh = np.zeros_like(rng)
        for s in range(4):
            sh += (0x80 << s) <= rng
        # the deltas: 7 bits each from bit 30 of the block (little-endian); a 15th (when the max
        # and the min share a position) runs on into the next block's bytes
        padded = np.zeros((height, width + 16), np.uint8)
        padded[:, :width] = rows
        win = np.stack([padded[:, j * 16:j * 16 + 18] for j in range(nblk)], axis=1)
        bits = np.unpackbits(win, axis=2, bitorder="little")
        out = np.empty((height, nblk, 16), np.int64)
        k = np.zeros((height, nblk), np.int64)     # which delta is next
        weights = (1 << np.arange(7)).astype(np.int64)
        deltas = np.stack([bits[:, :, 30 + 7 * j:37 + 7 * j].astype(np.int64) @ weights for j in range(15)], axis=2)
        for i in range(16):
            is_max = imax == i
            is_min = (imin == i) & ~is_max
            d = np.take_along_axis(deltas, np.minimum(k, 14)[:, :, None], axis=2)[:, :, 0]
            v = np.minimum((d << sh) + mn, 0x7FF)
            out[:, :, i] = np.where(is_max, mx, np.where(is_min, mn, v))
            k += ~(is_max | is_min)
        vals = np.zeros((height, width), np.int64)
        # block j of a pair covers even columns, the next odd ones
        for j in range(nblk):
            base = (j // 2) * 32 + (j & 1)
            cols = base + 2 * np.arange(16)
            ok = cols < width
            vals[:, cols[ok]] = out[:, j, :][:, ok]
        vals = vals.astype(np.uint16)
    return curve[np.minimum(vals.astype(np.int64) << 1, len(curve) - 1)].astype(np.uint16)


def _sr2(t: Tiff, ifd0) -> dict:
    """SR2SubIFD tags as {tag: values}, decrypted; {} when absent."""
    priv = ifd0.ints(0xC634) if 0xC634 in ifd0 else None
    if not priv:
        return {}
    e = ifd0[0xC634]
    raw = e.raw()
    off = struct.unpack(t.order + "I", raw[:4])[0] if len(raw) >= 4 else (priv[0] if priv else 0)
    sr2 = t.ifd(off)
    if sr2 is None:
        return {}
    so, sl, key = sr2.int(0x7200), sr2.int(0x7201), sr2.get(0x7221)
    if so is None or not sl or key is None or so + sl > len(t.data) or sl > 1 << 20:
        return {}
    kraw = key.raw()[:4]
    k = struct.unpack(t.order + "I", kraw)[0] if len(kraw) == 4 else key.value
    plain = decrypt(t.data[so:so + sl], k)
    # the decrypted block is an IFD whose offsets count from the file's start
    buf = bytearray(t.data)
    buf[so:so + sl] = plain
    try:
        sub = read_ifd(bytes(buf), so, 0, t.order)
    except (ValueError, struct.error):
        return {}
    return {tag: ent.values() for tag, ent in sub.items() if ent.type in (3, 4, 8, 9)}


def decode(path: Path, fmt: str = "arw") -> RawData:
    data = path.read_bytes()
    t = Tiff(data)
    ifd = None
    for cand in t.all_ifds():
        if cand.int(254, 0) == 0 and cand.int(262) in (32803, 6, 34892) and cand.int(256):
            ifd = cand
            break
    if ifd is None:
        raise PixportError("This Sony RAW has no sensor data")
    if ifd.int(262) == 6:
        return _ycbcr(data, t, ifd)
    if ifd.int(262) != 32803:
        raise UnsupportedError("This Sony RAW's photometric interpretation isn't supported")
    width, height, bits = ifd.int(256), ifd.int(257), ifd.int(258, 12)
    check_size(width, height)
    comp = ifd.int(259, 1)
    md = camera_metadata(t)
    n = width * height
    shift = 0     # the SR2 levels are 14-bit; 12-bit packed data needs them shifted down
    curve_top = None
    if comp == 7:
        tw, th = ifd.int(322), ifd.int(323)
        offs, sizes = ifd.ints(324) or [], ifd.ints(325) or []
        if not tw or not th or not offs:
            raise PixportError("This ARW's tiles are missing")
        across = -(-width // tw)
        cfa = np.zeros((height, width), np.uint16)

        def one(i):
            off, size = offs[i], sizes[i] if i < len(sizes) else len(data) - offs[i]
            samples, f = ljpeg.decode(data, off, min(len(data), off + size))
            return i, samples, f

        jobs = range(min(len(offs), across * -(-height // th)))
        with ThreadPoolExecutor(max_workers=8) as pool:
            res = list(pool.map(one, jobs)) if _accel.has("ljpeg_decode") else map(one, jobs)
        for i, s, f in res:
            top, left = (i // across) * th, (i % across) * tw
            tile = _sony_tile(s, f, th, tw)
            h = min(th, height - top)
            w = min(tw, width - left)
            cfa[top:top + h, left:left + w] = tile[:h, :w]
    elif comp == 32767 or comp == 1:
        off = ifd.int(273)
        if off is None:
            raise PixportError("This ARW's sensor data has no offset")
        size = ifd.int(279, 0)
        need(data, off, 1)
        size = min(size, len(data) - off)
        if size >= n * 2:
            cfa = np.frombuffer(data, "<u2" if t.order == "<" else ">u2", n, off).reshape(height, width).astype(np.uint16)
        elif size >= n * 3 // 2 and comp == 32767 and size < n * 2 and size != n:
            cfa = unpack_bits(data[off:off + n * 3 // 2], n, 12, "lsb").reshape(height, width)
            shift = 2
        elif size >= n:
            pts = ifd.ints(0x7010) or [8000, 10400, 12900, 14100]
            curve = sony_curve(pts)
            cfa = arw2(data, off, width, height, curve)
            curve_top = int(curve[min(0x7FF << 1, len(curve) - 1)])  # the top value an 11-bit delta reaches
        else:
            raise PixportError("This ARW's sensor data is smaller than its size")
    else:
        raise UnsupportedError(f"Sony RAW compression {comp} isn't supported")

    pat = ifd.ints(33422) or [0, 1, 1, 2]
    pattern = np.array(pat[:4], np.uint8).reshape(2, 2) if len(pat) >= 4 and max(pat[:4]) <= 2 else \
        np.array([[0, 1], [1, 2]], np.uint8)
    raw = RawData(cfa=cfa, pattern=pattern, metadata=md)
    raw.orientation = t.ifds[0].int(274, 1) or 1
    sr2 = _levels(t, ifd)
    raw.wb = _as_shot(sr2) or raw.wb
    cm = sr2.get(0x7800)
    blk = sr2.get(0x7310)
    if blk is not None and len(blk) >= 4:
        raw.black = np.array([[blk[0], blk[1]], [blk[2], blk[3]]], dtype=np.float64) / (1 << shift)
    wl = sr2.get(0x787F)
    tag_white = ifd.int(50717)
    if wl is not None and len(wl) >= 1:
        raw.white = float(max(wl[:3])) / (1 << shift)
    elif tag_white:
        raw.white = float(tag_white)
    elif curve_top is not None:
        raw.white = float(curve_top)
    else:
        raw.white = float((1 << bits) - 1)
    co, cs = ifd.ints(50719), ifd.ints(50720)
    if co and cs and len(co) >= 2 and len(cs) >= 2 and 0 < cs[0] and 0 < cs[1] \
            and co[0] + cs[0] <= width and co[1] + cs[1] <= height:
        raw.crop = (co[1], co[0], cs[1], cs[0])
    else:
        pad = padded_columns(cfa)
        if pad:
            raw.crop = (0, 0, height, width - pad)
    raw.metadata["bits"] = bits
    # Sony's own matrix (SR2SubIFD 0x7800) is the fallback: measured against ImageMagick (Adobe's
    # matrices), the fit to the camera's preview is closer (mean CIEDE2000 1.5-1.8 against 3.0-6.5)
    colour.attach(raw, t)
    if raw.cam_xyz is None and cm is not None and len(cm) >= 9:
        raw.cam_xyz = colour.from_rgb_cam(np.asarray(cm[:9], dtype=np.float64) / 1024.0)
        raw.metadata["colour"] = colour.FROM_FILE
    return raw


def _levels(t: Tiff, ifd) -> dict:
    """The SR2SubIFD's tags, with black, white balance and white levels the raw IFD has itself
    (some bodies keep them there)."""
    sr2 = _sr2(t, t.ifds[0])
    for tag in (0x7310, 0x7313, 0x787F):
        if tag in ifd and tag not in sr2:
            sr2[tag] = ifd[tag].values()
    return sr2


def _as_shot(sr2: dict) -> tuple[float, float, float] | None:
    """As-shot white balance: WB_RGGBLevels (0x7313), or the older WB_GRBGLevels (0x7303)."""
    wbv = sr2.get(0x7313)
    if wbv is not None and len(wbv) >= 4 and min(wbv[:4]) > 0:
        g = (float(wbv[1]) + float(wbv[2])) / 2
        return (float(wbv[0]) / g, 1.0, float(wbv[3]) / g)
    wbv = sr2.get(0x7303)
    if wbv is not None and len(wbv) >= 4 and min(wbv[:4]) > 0:
        g = (float(wbv[0]) + float(wbv[3])) / 2
        return (float(wbv[1]) / g, 1.0, float(wbv[2]) / g)
    return None


def _sony_tile(samples: np.ndarray, f: ljpeg.Frame, th: int, tw: int) -> np.ndarray:
    """A lossless JPEG tile as photosites: Sony codes each 2x2 Bayer block as the four
    components of one sample, so a th x tw tile is a (th/2) x (tw/2) four-component frame."""
    if len(f.comps) == 4 and samples.shape == (th // 2, tw * 2):
        s = samples.reshape(th // 2, tw // 2, 2, 2)
        return s.transpose(0, 2, 1, 3).reshape(th, tw)
    if samples.size >= th * tw:
        return samples.reshape(-1)[:th * tw].reshape(th, tw)
    raise PixportError("An ARW tile is smaller than the tile size")


def _ycbcr(data: bytes, t: Tiff, ifd) -> RawData:
    """Sony's medium and small lossless RAW: lossless JPEG tiles of YCbCr 4:2:0 (TIFF's
    YCbCrCoefficients and ReferenceBlackWhite give the conversion), already demosaiced."""
    from .canon import _interp
    width, height = ifd.int(256), ifd.int(257)
    check_size(width, height)
    if ifd.int(259) != 7:
        raise UnsupportedError("This Sony YCbCr RAW's compression isn't supported")
    tw, th = ifd.int(322), ifd.int(323)
    offs, sizes = ifd.ints(324) or [], ifd.ints(325) or []
    if not tw or not th or not offs or tw % 2 or th % 2:
        raise PixportError("This ARW's tiles are missing")
    kr, kg, kb = (ifd.floats(529) or [0.299, 0.587, 0.114])[:3]
    ref = ifd.floats(532) or [0, 16383, 16384, 16384, 16384, 16384]
    zero = ref[2] if len(ref) >= 3 else 16384
    across = -(-width // tw)
    rgb = np.zeros((height, width, 3), np.float32)

    def one(i):
        off = offs[i]
        size = sizes[i] if i < len(sizes) else len(data) - off
        return i, ljpeg.decode(data, off, min(len(data), off + size), chain=True, chain_rows=2)

    jobs = range(min(len(offs), across * -(-height // th)))
    with ThreadPoolExecutor(max_workers=8) as pool:
        res = list(pool.map(one, jobs)) if _accel.has("ljpeg_decode") else map(one, jobs)
    for i, (s_, f) in res:
        if f.slots != [0, 0, 0, 0, 1, 2]:
            raise UnsupportedError("This Sony YCbCr layout isn't supported")
        m = s_.reshape(s_.shape[0], -1, 6).astype(np.float32)
        y = np.empty((m.shape[0] * 2, m.shape[1] * 2), np.float32)
        for k in range(4):
            y[(k >> 1)::2, (k & 1)::2] = m[:, :, k]
        cb = _interp(_interp(m[:, :, 4] - np.float32(zero), 1), 0)
        cr = _interp(_interp(m[:, :, 5] - np.float32(zero), 1), 0)
        r = y + np.float32(2 - 2 * kr) * cr
        b = y + np.float32(2 - 2 * kb) * cb
        g = (y - np.float32(kr) * r - np.float32(kb) * b) / np.float32(kg)
        top, left = (i // across) * th, (i % across) * tw
        h, w = min(th, height - top), min(tw, width - left)
        rgb[top:top + h, left:left + w] = np.stack([r, g, b], axis=2)[:h, :w]
    np.maximum(rgb, 0, out=rgb)
    raw = RawData(linear=rgb, metadata=camera_metadata(t))
    raw.orientation = t.ifds[0].int(274, 1) or 1
    raw.white = float(ref[1]) if len(ref) >= 2 and ref[1] > 0 else 16383.0
    # already white balanced, but with the black level still in (measured against LibRaw and
    # the same scene's full RAW: a black of 1024 in each channel)
    sr2 = _levels(t, ifd)
    blk = sr2.get(0x7310)
    if blk is not None and len(blk) >= 4:  # on the 14-bit scale, the samples on BitsPerSample's
        raw.black = float(np.mean(blk[:4])) * (1 << max(0, (ifd.int(258) or 14) - 14))
    co, cs = ifd.ints(50719), ifd.ints(50720)
    if co and cs and len(co) >= 2 and len(cs) >= 2 and co[0] + cs[0] <= width and co[1] + cs[1] <= height:
        raw.crop = (co[1], co[0], cs[1], cs[0])
    raw.metadata["bits"] = 14
    colour.attach(raw, t)
    cm = sr2.get(0x7800)
    if raw.cam_xyz is None and cm is not None and len(cm) >= 9:
        raw.cam_xyz = colour.from_rgb_cam(np.asarray(cm[:9], dtype=np.float64) / 1024.0)
        raw.metadata["colour"] = colour.FROM_FILE
    return raw
