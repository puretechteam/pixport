"""Adobe DNG (spec 1.7): the raw IFD (NewSubfileType 0, CFA or LinearRaw) in IFD0 or a SubIFD,
stored in strips or tiles, uncompressed (8-32 bits), lossless JPEG (7) or Deflate (8, with the
floating-point predictors); with the linearization table, black levels (per CFA position, row
and column), white level, active area and default crop, CFA pattern, as-shot neutral and the
colour matrices, interpolated between their two calibration illuminants for the as-shot white."""

from __future__ import annotations

import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .. import _accel
from ..errors import PixportError, UnsupportedError
from . import RawData, colour, ljpeg
from .common import check_size, unpack_bits
from .tiff import Ifd, Tiff, camera_metadata

# CalibrationIlluminant (EXIF LightSource) -> correlated colour temperature (K)
ILLUMINANT_K = {1: 5500, 2: 4200, 3: 2850, 4: 5500, 9: 5500, 10: 6500, 11: 7500, 12: 6400, 13: 5000, 14: 4200,
                15: 3450, 17: 2856, 18: 4874, 19: 6774, 20: 5503, 21: 6504, 22: 7504, 23: 5003, 24: 3200}


def _raw_ifd(t: Tiff) -> Ifd:
    best = None
    for ifd in t.all_ifds():
        if ifd.int(254, 0) & 1:
            continue
        big = best is None or ifd.int(256, 0) * ifd.int(257, 1) > best.int(256) * best.int(257, 1)
        if ifd.int(262) in (32803, 34892) and ifd.int(256) and big:
            best = ifd
    if best is None:
        raise PixportError("This DNG has no raw image")
    return best


def _xy_to_xyz(x: float, y: float) -> np.ndarray:
    return np.array([x / y, 1.0, (1 - x - y) / y])


def _xyz_to_xy(v: np.ndarray) -> tuple[float, float]:
    s = float(v.sum())
    return (float(v[0] / s), float(v[1] / s)) if s > 0 else (0.3457, 0.3585)


def _cct(x: float, y: float) -> float:
    """Correlated colour temperature from xy (McCamy's approximation), clamped to 2000-50000 K."""
    n = (x - 0.3320) / (0.1858 - y) if abs(0.1858 - y) > 1e-9 else 0
    k = 449 * n ** 3 + 3525 * n ** 2 + 6823.3 * n + 5520.33
    return float(min(50000.0, max(2000.0, k)))


def colour_matrix(ifd0: Ifd, neutral: np.ndarray | None) -> np.ndarray | None:
    """XYZ -> camera for the as-shot white: ColorMatrix1/2 (with CameraCalibration and
    AnalogBalance) weighted by inverse colour temperature, found by iterating from the neutral."""
    mats = []
    for cm_tag, cc_tag, il_tag in ((50721, 50723, 50778), (50722, 50724, 50779)):
        cm = ifd0.floats(cm_tag)
        if not cm or len(cm) < 9:
            continue
        m = np.array(cm[:9]).reshape(3, 3)
        cc = ifd0.floats(cc_tag)
        if cc and len(cc) >= 9:
            m = np.array(cc[:9]).reshape(3, 3) @ m
        ab = ifd0.floats(50727)
        if ab and len(ab) >= 3:
            m = np.diag(ab[:3]) @ m
        mats.append((ILLUMINANT_K.get(ifd0.int(il_tag, 21), 6504), m))
    if not mats:
        return None
    if len(mats) == 1 or neutral is None:
        return mats[-1][1]
    (t1, m1), (t2, m2) = sorted(mats, key=lambda a: a[0])

    def blend(k: float) -> np.ndarray:
        if k <= t1:
            return m1
        if k >= t2:
            return m2
        w = (1 / k - 1 / t2) / (1 / t1 - 1 / t2)
        return w * m1 + (1 - w) * m2

    k = 5000.0
    for _ in range(20):
        m = blend(k)
        try:
            xyz = np.linalg.solve(m, neutral)
        except np.linalg.LinAlgError:
            break
        x, y = _xyz_to_xy(xyz)
        nk = _cct(x, y)
        if abs(nk - k) < 1:
            k = nk
            break
        k = nk
    return blend(k)


def _tiles(ifd: Ifd, width: int, height: int):
    """[(offset, size, top, left, rows, cols)] for strips or tiles."""
    tw, th = ifd.int(322), ifd.int(323)
    if tw and th:
        offs, sizes = ifd.ints(324) or [], ifd.ints(325) or []
        across = -(-width // tw)
        down = -(-height // th)
        if len(offs) < across * down or len(sizes) < across * down:
            raise PixportError("This DNG's tile table is cut short")
        return [(offs[i], sizes[i], (i // across) * th, (i % across) * tw, th, tw) for i in range(across * down)]
    offs, sizes = ifd.ints(273) or [], ifd.ints(279) or []
    rps = ifd.int(278, height) or height
    if not offs or len(sizes) < len(offs):
        raise PixportError("This DNG's strip table is missing")
    n = min(len(offs), -(-height // rps))
    return [(offs[i], sizes[i], i * rps, 0, min(rps, height - i * rps), width) for i in range(n)]  # the last strip may be short


def _unpredict(a: np.ndarray, pred: int, spp: int, bits: int) -> np.ndarray:
    """Undo TIFF/DNG horizontal differencing (2, 34892 by 2, 34893 by 4) on (rows, cols*spp)."""
    if pred in (2, 34892, 34893):
        step = spp * {2: 1, 34892: 2, 34893: 4}[pred]
        out = a.astype(np.int64)
        for s in range(step):
            out[:, s::step] = np.cumsum(out[:, s::step], axis=1)
        return (out & ((1 << bits) - 1)).astype(a.dtype)
    return a


def _float_unpredict(buf: bytes, rows: int, cols: int, spp: int, nbytes: int, pred: int) -> np.ndarray:
    """Floating-point predictor (3, 34894 by 2, 34895 by 4): bytes differenced along the row,
    then reordered from byte planes (most significant first) to big-endian samples."""
    n = cols * spp
    a = np.frombuffer(buf, np.uint8, rows * n * nbytes).reshape(rows, n * nbytes).astype(np.uint8)
    step = spp * {3: 1, 34894: 2, 34895: 4}[pred]
    out = a.copy()
    for s in range(step):
        out[:, s::step] = np.cumsum(a[:, s::step], axis=1, dtype=np.uint64).astype(np.uint8)
    if nbytes not in (2, 4):
        raise UnsupportedError(f"{nbytes * 8}-bit floating-point DNG data isn't supported")
    planes = out.reshape(rows, nbytes, n).transpose(0, 2, 1)
    be = np.ascontiguousarray(planes).view(">f2" if nbytes == 2 else ">f4")
    return be.reshape(rows, n).astype(np.float32)


def _decode_tile(data: bytes, off: int, size: int, rows: int, cols: int, spp: int, bits: int, comp: int, pred: int,
                 order: str, sample_format: int) -> np.ndarray:
    if off < 0 or off >= len(data) or size <= 0:
        raise PixportError("A DNG tile is outside the file")
    end = min(len(data), off + size)
    n = rows * cols * spp
    if comp == 7:
        samples, _f = ljpeg.decode(data, off, end)
        if samples.size < n:
            raise PixportError("A DNG lossless JPEG tile is smaller than the tile")
        return samples.reshape(-1)[:n].reshape(rows, cols * spp)
    if comp == 8:
        try:
            # never more than the tile needs: whole bytes a sample, a padded byte a row
            buf = zlib.decompressobj().decompress(data[off:end], n * -(-bits // 8) + rows + 64)
        except zlib.error as e:
            raise PixportError(f"A DNG Deflate tile is damaged: {e}") from None
        if sample_format == 3:
            nb = bits // 8
            if pred in (3, 34894, 34895):
                if len(buf) < n * nb:
                    raise PixportError("A DNG Deflate tile is cut short")
                return _float_unpredict(buf, rows, cols, spp, nb, pred)
            dt = {2: "f2", 4: "f4", 8: "f8"}.get(nb)
            if dt is None:
                raise UnsupportedError(f"{bits}-bit floating-point DNG data isn't supported")
            return np.frombuffer(buf, np.dtype(dt).newbyteorder(order), n).reshape(rows, cols * spp).astype(np.float32)
        a = _raw_samples(buf, 0, len(buf), rows, cols * spp, bits, order)
        return _unpredict(a, pred, spp, bits)
    if comp == 1:
        if sample_format == 3:
            nb = bits // 8
            dt = {2: "f2", 4: "f4", 8: "f8"}.get(nb)
            if dt is None:
                raise UnsupportedError(f"{bits}-bit floating-point DNG data isn't supported")
            if off + n * nb > len(data):
                raise PixportError("A DNG strip is cut short")
            return np.frombuffer(data, np.dtype(dt).newbyteorder(order), n, off).reshape(rows, cols * spp).astype(np.float32)
        return _raw_samples(data, off, end, rows, cols * spp, bits, order)
    if comp in (34892, 52546):
        raise UnsupportedError("Lossy DNG (JPEG or JPEG XL compressed raw data) isn't supported yet")
    raise UnsupportedError(f"DNG compression {comp} isn't supported")


def _raw_samples(data: bytes, off: int, end: int, rows: int, n: int, bits: int, order: str) -> np.ndarray:
    """Uncompressed samples: rows of ``n`` values, ``bits`` each, rows padded to whole bytes."""
    if bits == 16:
        need = rows * n * 2
        if off + need > len(data):
            raise PixportError("A DNG strip is cut short")
        return np.frombuffer(data, np.dtype("u2").newbyteorder(order), rows * n, off).reshape(rows, n).astype(np.uint16)
    if bits == 32:
        if off + rows * n * 4 > len(data):
            raise PixportError("A DNG strip is cut short")
        a = np.frombuffer(data, np.dtype("u4").newbyteorder(order), rows * n, off)
        return np.minimum(a, 0xFFFF).reshape(rows, n).astype(np.uint16)
    if bits == 8:
        if off + rows * n > len(data):
            raise PixportError("A DNG strip is cut short")
        return np.frombuffer(data, np.uint8, rows * n, off).reshape(rows, n).astype(np.uint16)
    if not 1 <= bits <= 16:
        raise UnsupportedError(f"{bits}-bit DNG data isn't supported")
    row_bytes = (n * bits + 7) // 8
    if off + row_bytes * rows > len(data):
        rows_have = max(0, (len(data) - off) // row_bytes)
        if rows_have == 0:
            raise PixportError("A DNG strip is cut short")
    buf = data[off:off + row_bytes * rows]
    if n * bits % 8 == 0:
        return unpack_bits(buf, rows * n, bits, "msb").reshape(rows, n)
    return np.stack([unpack_bits(buf[r * row_bytes:(r + 1) * row_bytes], n, bits, "msb") for r in range(rows)])


def decode(path: Path, fmt: str = "dng") -> RawData:
    data = path.read_bytes()
    t = Tiff(data)
    ifd0 = t.ifds[0]
    ifd = _raw_ifd(t)
    width, height = ifd.int(256), ifd.int(257)
    check_size(width, height)
    spp = ifd.int(277, 1)
    bits = (ifd.ints(258) or [16])[0]
    comp = ifd.int(259, 1)
    pred = ifd.int(317, 1)
    fmt_ = (ifd.ints(339) or [1])[0]
    linear = ifd.int(262) == 34892
    if not linear and spp != 1:
        raise UnsupportedError("CFA DNGs with more than one sample per photosite aren't supported")
    if spp not in (1, 3, 4):
        raise UnsupportedError(f"DNGs with {spp} samples per pixel aren't supported")
    if ifd.int(284, 1) != 1 and spp > 1:
        raise UnsupportedError("Planar DNG data isn't supported")
    pieces = _tiles(ifd, width, height)
    dtype = np.float32 if fmt_ == 3 else np.uint16
    img = np.zeros((height, width * spp), dtype=dtype)

    def one(p):
        off, size, _top, _left, rows, cols = p
        return p, _decode_tile(data, off, size, rows, cols, spp, bits, comp, pred, t.order, fmt_)

    jobs = len(pieces) > 1 and comp == 7 and _accel.has("ljpeg_decode")
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(one, pieces)) if jobs else map(one, pieces)
    for (off, size, top, left, rows, cols), tile in results:
        if top >= height or left >= width:
            continue
        h = min(rows, height - top, tile.shape[0])
        w = min(cols, width - left)
        img[top:top + h, left * spp:(left + w) * spp] = tile[:h, :w * spp]

    lin_table = ifd.ints(50712)
    if lin_table and fmt_ != 3:
        lut = np.array(lin_table + [lin_table[-1]] * max(0, 65536 - len(lin_table)), dtype=np.int64)[:65536]
        img = np.minimum(lut, 65535).astype(np.uint16)[img]

    # active area
    aa = ifd.ints(50829)
    top, left, bottom, right = (aa[:4] if aa and len(aa) >= 4 else [0, 0, height, width])
    if not (0 <= top < bottom <= height and 0 <= left < right <= width):
        top, left, bottom, right = 0, 0, height, width
    if linear:
        arr = img.reshape(height, width, spp)[top:bottom, left:right, :3]
    else:
        arr = img[top:bottom, left:right]
    ah, aw = bottom - top, right - left

    md = camera_metadata(t, ifd0)
    raw = RawData(metadata=md)
    if linear:
        raw.linear = np.ascontiguousarray(arr)
    else:
        raw.cfa = np.ascontiguousarray(arr)
    # levels
    nch = 3 if linear else 1
    rep = ifd.ints(50713) or [1, 1]
    rr, rc = (max(1, rep[0]), max(1, rep[1])) if len(rep) >= 2 else (1, 1)
    bl = ifd.floats(50714)
    if bl:
        need = rr * rc * nch
        vals = (bl + [bl[-1]] * need)[:need]
        b = np.array(vals).reshape(rr, rc, nch)
        raw.black = b[:, :, :3].mean(axis=(0, 1)) if linear else b[:, :, 0]
    dh, dv = ifd.floats(50715), ifd.floats(50716)
    if dh and len(dh) >= aw and not linear:
        raw.black_cols = np.array(dh[:aw], dtype=np.float32)
    if dv and len(dv) >= ah and not linear:
        raw.black_rows = np.array(dv[:ah], dtype=np.float32)
    wl = ifd.floats(50717)
    raw.white = float(max(wl[:3])) if wl else float((1 << bits) - 1 if fmt_ != 3 else 1.0)
    if not linear:
        dims = ifd.ints(33421) or [2, 2]
        pat = ifd.ints(33422) or [0, 1, 1, 2]
        ph, pw = (dims[0], dims[1]) if len(dims) >= 2 else (2, 2)
        plane = ifd0.ints(50710) or ifd.ints(50710) or [0, 1, 2]
        if ph * pw > 64 or len(pat) < ph * pw or ph < 1 or pw < 1:
            raise UnsupportedError("This DNG's colour filter pattern isn't supported")
        colours = [plane[c] if c < len(plane) else c for c in pat[:ph * pw]]
        if any(c > 2 for c in colours):
            raise UnsupportedError("DNGs with a four-colour filter (CMYG, RGBE) aren't supported")
        p = np.array(colours, dtype=np.uint8).reshape(ph, pw)
        # the pattern starts at the top-left of the active area
        raw.pattern = p
    # crop inside the active area
    co, cs = ifd.floats(50719), ifd.floats(50720)
    if co and cs and len(co) >= 2 and len(cs) >= 2:
        cl, ct, cw, ch = int(co[0]), int(co[1]), int(cs[0]), int(cs[1])
        if 0 <= ct and 0 <= cl and ch > 0 and cw > 0 and ct + ch <= ah and cl + cw <= aw:
            raw.crop = (ct, cl, ch, cw)
    raw.orientation = ifd0.int(274, 1) or 1
    # white balance and colour
    neutral = ifd0.floats(50728)
    nv = None
    if neutral and len(neutral) >= 3 and min(neutral[:3]) > 0:
        nv = np.array(neutral[:3])
    cm = colour_matrix(ifd0, nv)
    if nv is None:
        xy = ifd0.floats(50729)
        if xy and len(xy) >= 2 and cm is not None and xy[1] > 0:
            nv = cm @ _xy_to_xyz(xy[0], xy[1])
            nv = nv / nv[1] if nv[1] > 0 else None
    if nv is not None and np.all(nv > 0):
        raw.wb = (float(nv[1] / nv[0]), 1.0, float(nv[1] / nv[2]))
    raw.cam_xyz = cm
    raw.metadata["colour"] = colour.FROM_DNG if cm is not None else colour.NONE
    raw.metadata["bits"] = bits
    return raw
