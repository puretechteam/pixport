"""Small TIFF, DNG and lossless JPEG writers for the RAW tests: enough to build files that
exercise the readers' paths (both byte orders, SubIFDs, EXIF and GPS, maker notes, strips and
tiles, every lossless JPEG predictor)."""

from __future__ import annotations

import struct

import numpy as np

TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}


def _refs(vals) -> bool:
    return isinstance(vals, tuple) and bool(vals) and all(isinstance(v, str) and v.startswith("@") for v in vals)


class Tiff:
    """IFDs as lists of (tag, type, values); ``sub`` maps a tag in an IFD to child IFDs that the
    tag will point at. Values are a list of numbers, a bytes object or a str."""

    def __init__(self, order: str = "<"):
        self.order = order

    def _value(self, typ, vals) -> bytes:
        o = self.order
        if isinstance(vals, str):
            return vals.encode("latin-1") + b"\0"
        if isinstance(vals, (bytes, bytearray)):
            return bytes(vals)
        if typ in (5, 10):
            out = b""
            for v in vals:
                num, den = (v if isinstance(v, tuple) else (round(v * 10000), 10000))
                out += struct.pack(o + ("II" if typ == 5 else "ii"), num, den)
            return out
        code = {1: "B", 3: "H", 4: "I", 8: "h", 9: "i", 11: "f", 12: "d"}[typ]
        return struct.pack(o + code * len(vals), *vals)

    def build(self, ifds: list, blobs: dict | None = None, magic: int = 42) -> bytes:
        """``ifds``: the main chain, each a dict {"entries": [(tag, type, values)], "children":
        {tag: [ifd, ...]}}. ``blobs``: {name: bytes} placed after the IFDs; an entry value
        ("@name",) becomes that blob's offset (type 4)."""
        o = self.order
        out = bytearray((b"II" if o == "<" else b"MM") + struct.pack(o + "HI", magic, 8))
        blobs = blobs or {}
        blob_at = {}
        # lay out every IFD (depth first), then blobs; two passes to resolve offsets
        order_list = []

        def collect(ifd):
            order_list.append(ifd)
            for kids in ifd.get("children", {}).values():
                for k in kids:
                    collect(k)

        for ifd in ifds:
            collect(ifd)
        for _ in range(2):
            pos = 8
            places = {}
            for ifd in order_list:
                places[id(ifd)] = pos
                n = len(ifd["entries"]) + len(ifd.get("children", {}))
                pos += 2 + 12 * n + 4
                for tag, typ, vals in ifd["entries"]:
                    size = 4 * len(vals) if _refs(vals) else len(self._value(typ, vals))
                    if size > 4:
                        pos += size + (size & 1)
                for tag, kids in ifd.get("children", {}).items():
                    if len(kids) > 1:
                        pos += 4 * len(kids)
            for name, b in blobs.items():
                blob_at[name] = pos
                pos += len(b) + (len(b) & 1)
            self._places = places
        out = bytearray((b"II" if o == "<" else b"MM") + struct.pack(o + "HI", magic, 8))
        chain = {id(ifd): (ifds[i + 1] if i + 1 < len(ifds) else None) for i, ifd in enumerate(ifds)}
        for ifd in order_list:
            assert len(out) == places[id(ifd)], (len(out), places[id(ifd)])
            ents = []
            for tag, typ, vals in ifd["entries"]:
                if _refs(vals):
                    ents.append((tag, 4, len(vals), struct.pack(o + "I" * len(vals), *(blob_at[v[1:]] for v in vals))))
                else:
                    v = self._value(typ, vals)
                    ents.append((tag, typ, len(v) // TYPE_SIZE[typ], v))
            for tag, kids in ifd.get("children", {}).items():
                offs = [places[id(k)] for k in kids]
                ents.append((tag, 4, len(offs), struct.pack(o + "I" * len(offs), *offs)))
            ents.sort(key=lambda e: e[0])
            n = len(ents)
            data_at = len(out) + 2 + 12 * n + 4
            table, extra = b"", b""
            for tag, typ, count, v in ents:
                if len(v) <= 4:
                    table += struct.pack(o + "HHI", tag, typ, count) + v.ljust(4, b"\0")
                else:
                    table += struct.pack(o + "HHII", tag, typ, count, data_at + len(extra))
                    extra += v + b"\0" * (len(v) & 1)
            nxt = chain.get(id(ifd))
            out += struct.pack(o + "H", n) + table + struct.pack(o + "I", places[id(nxt)] if nxt is not None else 0)
            out += extra
        for name, b in blobs.items():
            assert len(out) == blob_at[name]
            out += b + b"\0" * (len(b) & 1)
        return bytes(out)


# ---------------------------------------------------------------------------------------------
# lossless JPEG


class Bits:
    def __init__(self):
        self.bits: list[int] = []

    def put(self, v: int, n: int) -> None:
        self.bits.extend((v >> i) & 1 for i in range(n - 1, -1, -1))

    def bytes(self, stuff: bool = True) -> bytes:
        b = self.bits + [1] * (-len(self.bits) % 8)
        raw = bytes(int("".join(map(str, b[i:i + 8])), 2) for i in range(0, len(b), 8))
        return raw.replace(b"\xff", b"\xff\x00") if stuff else raw


def huffman_code(counts, symbols) -> dict:
    """{symbol: (code, length)} for a canonical table."""
    table, code, k = {}, 0, 0
    for length in range(1, 17):
        for _ in range(counts[length - 1]):
            table.setdefault(symbols[k], (code, length))
            code += 1
            k += 1
        code <<= 1
    return table


# a table covering SSSS 0-16
LJ_COUNTS = [0, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
LJ_SYMBOLS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


def put_diff(w: Bits, table: dict, diff: int) -> None:
    diff = ((diff + 32768) & 0xFFFF) - 32768
    if diff == -32768:
        code, ln = table[16]
        w.put(code, ln)
        return
    s = abs(diff).bit_length()
    code, ln = table[s]
    w.put(code, ln)
    if s:
        w.put(diff if diff > 0 else diff + (1 << s) - 1, s)


def ljpeg_encode(samples: np.ndarray, ncomp: int, predictor: int = 1, precision: int = 16, sampling=None,
                 chain: int = 0) -> bytes:
    """``samples``: rows x (mcus * slots) as the decoder returns them; one Huffman table."""
    rows, width = samples.shape
    comps = sampling or [(1, 1)] * ncomp
    slots = len(comps) if sampling is None else sum(h * v for h, v in comps)
    x = samples.astype(np.int64)
    init = 1 << (precision - 1)
    table = huffman_code(LJ_COUNTS, LJ_SYMBOLS)
    w = Bits()
    for r in range(rows):
        last = {}
        for j in range(width):
            s = j % slots
            g = 0 if s < chain else s
            if chain:
                if g in last:
                    pred = last[g]
                else:
                    pred = init if r == 0 else x[r - 1, j if g else 0]
                last[g] = int(x[r, j])
            elif r == 0:
                pred = init if j < slots else x[r, j - slots]
            elif j < slots:
                pred = x[r - 1, j]
            else:
                a, b, c = int(x[r, j - slots]), int(x[r - 1, j]), int(x[r - 1, j - slots])
                pred = {1: a, 2: b, 3: c, 4: a + b - c, 5: a + ((b - c) >> 1), 6: b + ((a - c) >> 1),
                        7: (a + b) >> 1}[predictor]
            put_diff(w, table, int(x[r, j]) - int(pred))
    hdr = b"\xff\xd8"
    dht = bytes([0x00]) + bytes(LJ_COUNTS) + bytes(LJ_SYMBOLS)
    hdr += b"\xff\xc4" + struct.pack(">H", 2 + len(dht)) + dht
    nf = len(comps)
    hmax = max(h for h, _ in comps)
    vmax = max(v for _, v in comps)
    fw = (width // slots) * hmax
    fh = rows * vmax
    sof = struct.pack(">BHHB", precision, fh, fw, nf) + b"".join(bytes([i + 1, (h << 4) | v, 0]) for i, (h, v) in enumerate(comps))
    hdr += b"\xff\xc3" + struct.pack(">H", 2 + len(sof)) + sof
    sos = bytes([nf]) + b"".join(bytes([i + 1, 0]) for i in range(nf)) + bytes([predictor, 0, 0])
    hdr += b"\xff\xda" + struct.pack(">H", 2 + len(sos)) + sos
    return hdr + w.bytes() + b"\xff\xd9"


# ---------------------------------------------------------------------------------------------
# DNG


def dng(cfa: np.ndarray, order: str = "<", compression: int = 1, tile: int | None = None, bits: int = 16,
        extra_raw: list | None = None, extra0: list | None = None, linear: bool = False) -> bytes:
    """A DNG holding ``cfa`` (uint16, height x width, or height x width x 3 when ``linear``)."""
    t = Tiff(order)
    h, w = cfa.shape[:2]
    spp = 3 if linear else 1
    blobs = {}
    raw_entries = [(254, 4, [0]), (256, 4, [w]), (257, 4, [h]), (258, 3, [bits] * spp), (259, 3, [compression]),
                   (262, 3, [34892 if linear else 32803]), (277, 3, [spp]), (284, 3, [1])]
    if not linear:
        raw_entries += [(33421, 3, [2, 2]), (33422, 1, [0, 1, 1, 2])]
    raw_entries += extra_raw or []
    flat = cfa.reshape(h, w * spp)
    if tile:
        offs, sizes = [], []
        for ty in range(0, h, tile):
            for tx in range(0, w, tile):
                piece = np.zeros((tile, tile * spp), np.uint16)
                part = flat[ty:ty + tile, tx * spp:(tx + tile) * spp]
                piece[:part.shape[0], :part.shape[1]] = part
                if compression == 7:
                    b = ljpeg_encode(piece.reshape(tile, -1), 2 if not linear else 3, precision=bits)
                else:
                    b = piece.astype(order + "u2").tobytes()
                name = f"t{len(offs)}"
                blobs[name] = b
                offs.append(name)
                sizes.append(len(b))
        raw_entries += [(322, 4, [tile]), (323, 4, [tile])]
        raw_entries += [(325, 4, sizes)]
        tile_offsets = offs
    else:
        if compression == 7:
            b = ljpeg_encode(flat.reshape(h, -1), 2 if not linear else 3, precision=bits)
        else:
            b = flat.astype(order + "u2").tobytes()
        blobs["strip"] = b
        raw_entries += [(273, 4, ("@strip",)), (279, 4, [len(b)]), (278, 4, [h])]
        tile_offsets = None
    if tile_offsets:
        raw_entries.append((324, 4, tuple("@" + n for n in tile_offsets)))
    raw = {"entries": raw_entries}
    ifd0 = {"entries": [(254, 4, [1]), (256, 4, [8]), (257, 4, [8]), (271, 2, "Pixtest"), (272, 2, "Model One"),
                        (274, 3, [1]), (50706, 1, [1, 4, 0, 0])] + (extra0 or []),
            "children": {330: [raw]}}
    data = t.build([ifd0], blobs)
    return data
