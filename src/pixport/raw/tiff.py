"""TIFF structure as RAW files use it: image file directories (IFDs) in either byte order, their
SubIFDs, the EXIF and GPS IFDs, and each maker's note with its own header and offset base.

Everything is read from one ``bytes`` object and every offset is checked against its length: an
entry pointing outside the file is dropped, not followed. :func:`exif_block` packs the EXIF and
GPS tags into a small standalone TIFF (the bytes ``Image.metadata["exif"]`` carries)."""

from __future__ import annotations

import struct

import numpy as np

from ..errors import PixportError

# type: (NumPy code, size)
TYPES = {1: ("u1", 1), 2: ("u1", 1), 3: ("u2", 2), 4: ("u4", 4), 5: ("u4", 8), 6: ("i1", 1), 7: ("u1", 1),
         8: ("i2", 2), 9: ("i4", 4), 10: ("i4", 8), 11: ("f4", 4), 12: ("f8", 8), 13: ("u4", 4), 16: ("u8", 8),
         17: ("i8", 8), 18: ("u8", 8)}
MAX_ENTRIES = 2000      # entries in one IFD
MAX_IFDS = 256          # IFDs in one file, all chains together

# tags
SUBIFDS, EXIF_IFD, GPS_IFD, MAKERNOTE, INTEROP_IFD = 330, 34665, 34853, 37500, 40965


class Entry:
    """One IFD entry; ``pos`` is where its value is in the data (inline or at its offset)."""

    __slots__ = ("count", "data", "order", "pos", "tag", "type")

    def __init__(self, tag: int, typ: int, count: int, pos: int, order: str, data: bytes):
        self.tag, self.type, self.count, self.pos, self.order, self.data = tag, typ, count, pos, order, data

    @property
    def size(self) -> int:
        return TYPES[self.type][1] * self.count

    def raw(self) -> bytes:
        return self.data[self.pos:self.pos + self.size]

    def values(self) -> np.ndarray:
        """The values as an array (rationals as float64)."""
        code, _size = TYPES[self.type]
        dt = np.dtype(code).newbyteorder(self.order)
        if self.type in (5, 10):
            a = np.frombuffer(self.data, dt, self.count * 2, self.pos).astype(np.float64)
            num, den = a[0::2], a[1::2]
            return np.divide(num, den, out=np.zeros_like(num), where=den != 0)
        return np.frombuffer(self.data, dt, self.count, self.pos)

    @property
    def value(self):
        """The first value (an int or float)."""
        v = self.values()
        if not len(v):
            raise ValueError(f"TIFF tag {self.tag} is empty")
        x = v[0]
        return float(x) if self.type in (5, 10, 11, 12) else int(x)

    def text(self) -> str:
        return self.raw().split(b"\0", 1)[0].decode("latin-1").strip()


class Ifd(dict):
    """{tag: Entry} plus where the IFD was (``pos``), its value base and byte order."""

    def __init__(self, pos: int, base: int, order: str, data: bytes):
        super().__init__()
        self.pos, self.base, self.order, self.data, self.next = pos, base, order, data, 0
        self.dropped = 0  # entries whose values would be outside the file

    def int(self, tag: int, default=None):
        e = self.get(tag)
        return default if e is None or not e.count else int(e.values()[0])

    def ints(self, tag: int, default=None) -> list | None:
        e = self.get(tag)
        return default if e is None else [int(x) for x in e.values()]

    def floats(self, tag: int, default=None) -> list | None:
        e = self.get(tag)
        return default if e is None else [float(x) for x in e.values()]

    def text(self, tag: int, default: str = "") -> str:
        e = self.get(tag)
        return default if e is None else e.text()

    def offset(self, tag: int, default=None):
        """The value of an offset tag, made absolute with this IFD's base."""
        v = self.int(tag)
        return default if v is None else v + self.base

    def offsets(self, tag: int) -> list[int]:
        v = self.ints(tag)
        return [] if v is None else [x + self.base for x in v]


def read_ifd(data: bytes, pos: int, base: int, order: str) -> Ifd:
    """The IFD at ``pos``; offsets inside it count from ``base``."""
    n = len(data)
    if pos < 0 or pos + 2 > n:
        raise ValueError(f"TIFF directory at {pos} is outside the file")
    count = struct.unpack_from(order + "H", data, pos)[0]
    if count > MAX_ENTRIES:
        raise ValueError(f"TIFF directory at {pos} claims {count} entries")
    ifd = Ifd(pos, base, order, data)
    end = pos + 2 + 12 * count
    if end > n:
        count = (n - pos - 2) // 12
        end = pos + 2 + 12 * count
    for i in range(count):
        tag, typ, cnt = struct.unpack_from(order + "HHI", data, pos + 2 + 12 * i)
        if typ not in TYPES:
            continue
        size = TYPES[typ][1] * cnt
        vpos = pos + 2 + 12 * i + 8
        if size > 4:
            vpos = struct.unpack_from(order + "I", data, vpos)[0] + base
        if vpos < 0 or vpos + size > n:
            ifd.dropped += 1
            continue
        ifd[tag] = Entry(tag, typ, cnt, vpos, order, data)
    if end + 4 <= n:
        ifd.next = struct.unpack_from(order + "I", data, end)[0]
    return ifd


class Tiff:
    """A TIFF structure starting at ``base`` in ``data`` (a whole RAW file, or a maker note's
    embedded TIFF). ``ifds`` is the main chain (IFD0, IFD1, ...)."""

    def __init__(self, data: bytes, base: int = 0, magic: tuple[int, ...] = (42, 0x4F52, 0x5352, 0x55)):
        if len(data) < base + 8:
            raise PixportError("The file is too short to be a TIFF-based RAW")
        bo = data[base:base + 2]
        if bo not in (b"II", b"MM"):
            raise PixportError("The file isn't TIFF-structured")
        self.data, self.base = data, base
        self.order = "<" if bo == b"II" else ">"
        m, first = struct.unpack_from(self.order + "HI", data, base + 2)
        if m not in magic:
            raise PixportError("The file isn't TIFF-structured")
        self.magic = m
        self._parsed: dict[tuple, Ifd | None] = {}
        self.ifds: list[Ifd] = []
        off, chain = first, set()
        while off and len(self.ifds) < 64 and off + base not in chain:
            chain.add(off + base)
            ifd = self.ifd(off + base, base)
            if ifd is None:
                break
            self.ifds.append(ifd)
            off = ifd.next

    def ifd(self, pos: int, base: int | None = None, order: str | None = None) -> Ifd | None:
        """The IFD at absolute ``pos`` (each one parsed once), or None when it's outside the file
        or there are too many."""
        key = (pos, self.base if base is None else base, order or self.order)
        if key in self._parsed:
            return self._parsed[key]
        if len(self._parsed) >= MAX_IFDS:
            return None
        try:
            found = read_ifd(self.data, pos, key[1], key[2])
        except (ValueError, struct.error):
            found = None
        self._parsed[key] = found
        return found

    def children(self, ifd: Ifd, tag: int) -> list[Ifd]:
        """The IFDs an offset tag (SubIFDs, EXIF, GPS) points at."""
        out = []
        for off in ifd.offsets(tag):
            sub = self.ifd(off, ifd.base, ifd.order)
            if sub is not None:
                out.append(sub)
        return out

    def all_ifds(self) -> list[Ifd]:
        """The main chain and every SubIFD under it, depth first, each once (also when the file
        points at one twice, or in a loop)."""
        out: list[Ifd] = []
        seen: set[int] = set()

        def walk(ifd: Ifd, depth: int):
            if id(ifd) in seen:
                return
            seen.add(id(ifd))
            out.append(ifd)
            if depth < 4:
                for sub in self.children(ifd, SUBIFDS):
                    walk(sub, depth + 1)

        for ifd in self.ifds:
            walk(ifd, 0)
        return out

    def exif(self) -> Ifd | None:
        """The EXIF IFD (pointed at from IFD0, or from any IFD in the chain)."""
        if not hasattr(self, "_exif"):
            self._exif = None
            for ifd in self.ifds:
                subs = self.children(ifd, EXIF_IFD)
                if subs:
                    self._exif = subs[0]
                    break
        return self._exif

    def gps(self) -> Ifd | None:
        if not hasattr(self, "_gps"):
            self._gps = None
            for ifd in self.ifds:
                subs = self.children(ifd, GPS_IFD)
                if subs:
                    self._gps = subs[0]
                    break
        return self._gps

    def makernote(self) -> tuple[str, Ifd] | None:
        """(maker, IFD) for the maker note in the EXIF IFD, or None."""
        exif = self.exif()
        if exif is None or MAKERNOTE not in exif:
            return None
        return parse_makernote(self, exif[MAKERNOTE].pos, exif[MAKERNOTE].count)


def parse_makernote(tiff: Tiff, pos: int, size: int) -> tuple[str, Ifd] | None:
    """Each maker's note: its header, where its IFD is, and what its offsets count from."""
    d = tiff.data
    head = d[pos:pos + 18]
    order = tiff.order

    def bo(at: int) -> str:
        return {b"II": "<", b"MM": ">"}.get(d[at:at + 2], order)

    ifd = None
    if head.startswith(b"Nikon\0\x02"):
        try:
            inner = Tiff(d, pos + 10)
        except PixportError:
            return None
        return ("nikon", inner.ifds[0]) if inner.ifds else None
    if head.startswith(b"Nikon\0\x01"):
        maker, ifd = "nikon", tiff.ifd(pos + 8)
    elif head.startswith((b"OLYMPUS\x00", b"OM SYSTEM\x00")):
        skip = 8 if head.startswith(b"OLYMPUS\0") else 12
        maker, ifd = "olympus", tiff.ifd(pos + skip + 4, pos, bo(pos + skip))
    elif head.startswith((b"OLYMP\x00", b"EPSON\x00")):
        maker, ifd = "olympus", tiff.ifd(pos + 8)
    elif head.startswith(b"Panasonic\0"):
        maker, ifd = "panasonic", tiff.ifd(pos + 12)
    elif head.startswith((b"SONY", b"VHAB")):
        maker, ifd = "sony", tiff.ifd(pos + 12)
    elif head.startswith(b"AOC\0"):
        # Pentax: offsets from the note's start in PEF files, from the TIFF base in JPEGs and DNGs
        tries = [t for t in (tiff.ifd(pos + 6, base, bo(pos + 4)) for base in (pos, tiff.base)) if t]
        ifd = min(tries, key=lambda t: t.dropped) if tries else None  # the base that keeps its values in the file
        maker = "pentax"
    elif head.startswith(b"PENTAX \0"):
        maker, ifd = "pentax", tiff.ifd(pos + 10, pos, bo(pos + 8))
    elif head.startswith(b"FUJIFILM"):
        off = struct.unpack_from("<I", d, pos + 8)[0]
        maker, ifd = "fujifilm", tiff.ifd(pos + off, pos, "<")
    elif head.startswith(b"LEICA"):
        maker, ifd = "leica", tiff.ifd(pos + 8)
    else:
        # Canon (and some Sony): the IFD itself, offsets from the TIFF base
        maker, ifd = "canon", tiff.ifd(pos)
    return None if ifd is None else (maker, ifd)


# EXIF tags copied into exif_block's IFD0 (the rest of IFD0 describes the RAW data itself)
_IFD0_KEEP = {270, 271, 272, 305, 306, 315, 33432}


def _pack_ifd(entries: list[tuple[int, int, int, bytes]], start: int) -> bytes:
    """A little-endian IFD placed at offset ``start``: entries are (tag, type, count, value bytes)."""
    entries = sorted(entries)
    head = struct.pack("<H", len(entries))
    table, extra = b"", b""
    data_at = start + 2 + 12 * len(entries) + 4
    for tag, typ, count, val in entries:
        if len(val) <= 4:
            table += struct.pack("<HHI", tag, typ, count) + val.ljust(4, b"\0")
        else:
            if (data_at + len(extra)) & 1:
                extra += b"\0"
            table += struct.pack("<HHII", tag, typ, count, data_at + len(extra))
            extra += val
    return head + table + b"\0\0\0\0" + extra


def _le_value(e: Entry) -> bytes | None:
    """An entry's value re-encoded little-endian, or None when it can't be copied."""
    if e.type in (1, 2, 6, 7):
        return e.raw()
    if e.size > 1 << 16:
        return None
    code = TYPES[e.type][0]
    n = e.count * (2 if e.type in (5, 10) else 1)
    a = np.frombuffer(e.data, np.dtype(code).newbyteorder(e.order), n, e.pos)
    return a.astype(np.dtype(code).newbyteorder("<")).tobytes()


def exif_block(tiff: Tiff, ifd0: Ifd | None = None) -> bytes | None:
    """EXIF as standalone TIFF bytes (little-endian, no "Exif\\0\\0" prefix): camera, lens, date,
    exposure and GPS, with Orientation 1 (the picture is turned upright when developed). The maker
    note is left out: its internal offsets don't survive a move."""
    ifd0 = ifd0 if ifd0 is not None else (tiff.ifds[0] if tiff.ifds else None)
    if ifd0 is None:
        return None
    exif, gps = tiff.exif(), tiff.gps()

    def copy(ifd: Ifd | None, keep=None, drop=()) -> list:
        out = []
        for tag, e in (ifd or {}).items():
            if (keep is not None and tag not in keep) or tag in drop or e.type == 13:
                continue
            v = _le_value(e)
            if v is not None:
                out.append((tag, e.type, e.count, v))
        return out

    main = copy(ifd0, _IFD0_KEEP) + [(274, 3, 1, struct.pack("<H", 1))]
    ex = copy(exif, drop={MAKERNOTE, INTEROP_IFD, 0xA005, 0xEA1C}) if exif is not None else []
    gp = copy(gps) if gps is not None else []
    if not ex and not gp and len(main) == 1:
        return None
    ptrs = ([(EXIF_IFD, 4, 1, b"\0\0\0\0")] if ex else []) + ([(GPS_IFD, 4, 1, b"\0\0\0\0")] if gp else [])
    # lay out IFD0, then the EXIF IFD, then GPS; pointer values are filled in once sizes are known
    blob0 = _pack_ifd(main + ptrs, 8)
    exif_at = 8 + len(blob0) + (len(blob0) & 1)
    blob_e = _pack_ifd(ex, exif_at) if ex else b""
    gps_at = exif_at + len(blob_e) + (len(blob_e) & 1)
    final = [(t, ty, c, struct.pack("<I", exif_at if t == EXIF_IFD else gps_at) if t in (EXIF_IFD, GPS_IFD) else v)
             for t, ty, c, v in main + ptrs]
    blob0 = _pack_ifd(final, 8)
    blob_g = _pack_ifd(gp, gps_at) if gp else b""
    out = b"II*\0" + struct.pack("<I", 8) + blob0
    out += b"\0" * (exif_at - len(out)) + blob_e
    out += b"\0" * (gps_at - len(out)) + blob_g
    return out


def camera_metadata(tiff: Tiff, ifd0: Ifd | None = None) -> dict:
    """make, model, iso, exposure, aperture, focal length, lens, date and exif bytes."""
    ifd0 = ifd0 if ifd0 is not None else (tiff.ifds[0] if tiff.ifds else Ifd(0, 0, "<", b""))
    md: dict = {}
    for key, tag in (("make", 271), ("model", 272), ("software", 305), ("datetime", 306)):
        v = ifd0.text(tag)
        if v:
            md[key] = v
    ex = tiff.exif()
    if ex is not None:
        for key, tag, kind in (("exposure", 33434, float), ("aperture", 33437, float), ("iso", 34855, int),
                               ("focal_length", 37386, float), ("lens", 42036, str), ("datetime", 36867, str)):
            e = ex.get(tag)
            if e is None or not e.count:
                continue
            try:
                md[key] = e.text() if kind is str else kind(e.values()[0])
            except (ValueError, IndexError):
                pass
    try:
        blob = exif_block(tiff, ifd0)
    except (ValueError, struct.error):
        blob = None
    if blob:
        md["exif"] = blob
    return md
