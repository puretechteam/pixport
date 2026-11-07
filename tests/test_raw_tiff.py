"""The TIFF/EXIF reader under RAW files: both byte orders, SubIFDs, EXIF and GPS IFDs, maker notes
with their own headers and offset bases, and hostile structures (offsets past the end, loops,
huge counts) that are skipped rather than followed; and the standalone EXIF block for writers."""

from __future__ import annotations

import struct

import numpy as np
import pytest
from raw_writer import Tiff

from pixport.errors import PixportError
from pixport.raw import tiff


def _file(order: str = "<", makernote: bytes = b"") -> bytes:
    gps = {"entries": [(0, 1, [2, 3, 0, 0]), (1, 2, "N"), (2, 5, [(52, 1), (30, 1), (1234, 100)])]}
    exif_entries = [(33434, 5, [(1, 250)]), (34855, 3, [400]), (36867, 2, "2024:05:06 07:08:09"), (42036, 2, "50mm F1.8")]
    if makernote:
        exif_entries.append((37500, 7, makernote))
    exif = {"entries": exif_entries}
    sub = {"entries": [(254, 4, [0]), (256, 4, [64]), (257, 4, [48])]}
    ifd0 = {"entries": [(271, 2, "Maker"), (272, 2, "Model X"), (274, 3, [6]), (306, 2, "2024:05:06 07:08:09")],
            "children": {330: [sub], 34665: [exif], 34853: [gps]}}
    ifd1 = {"entries": [(256, 4, [16]), (257, 4, [12])]}
    return Tiff(order).build([ifd0, ifd1])


@pytest.mark.parametrize("order", ["<", ">"])
def test_ifds_subifds_exif_gps(order):
    t = tiff.Tiff(_file(order))
    assert t.order == order
    assert len(t.ifds) == 2
    assert t.ifds[0].text(271) == "Maker" and t.ifds[0].int(274) == 6
    assert [i.int(256) for i in t.all_ifds()] == [None, 64, 16]
    assert t.exif().int(34855) == 400
    assert t.exif()[33434].value == pytest.approx(0.004)
    assert t.gps().text(1) == "N"
    assert t.gps().floats(2) == pytest.approx([52, 30, 12.34])
    md = tiff.camera_metadata(t)
    assert md["make"] == "Maker" and md["model"] == "Model X" and md["iso"] == 400
    assert md["lens"] == "50mm F1.8" and md["datetime"] == "2024:05:06 07:08:09"


@pytest.mark.parametrize("order", ["<", ">"])
def test_exif_block_is_standalone(order):
    t = tiff.Tiff(_file(order))
    blob = tiff.exif_block(t)
    assert blob[:4] == b"II*\0"
    again = tiff.Tiff(blob)
    assert again.ifds[0].text(272) == "Model X"
    assert again.ifds[0].int(274) == 1          # the developed picture is upright
    assert again.exif().int(34855) == 400
    assert again.exif().text(36867) == "2024:05:06 07:08:09"
    assert again.gps().floats(2) == pytest.approx([52, 30, 12.34])
    assert 37500 not in again.exif()            # the maker note is left behind


def _nikon_note() -> bytes:
    inner = Tiff(">").build([{"entries": [(0x0C, 5, [(1945, 1024), (1408, 1024), (1, 1), (1, 1)]), (0x3D, 3, [400] * 4)]}])
    return b"Nikon\0\x02\x10\0\0" + inner


def test_nikon_makernote_is_its_own_tiff():
    t = tiff.Tiff(_file("<", _nikon_note()))
    maker, mn = t.makernote()
    assert maker == "nikon" and mn.order == ">"
    assert mn.floats(0x0C)[:2] == pytest.approx([1945 / 1024, 1408 / 1024])
    assert mn.ints(0x3D) == [400] * 4


def test_olympus_makernote_offsets_from_its_start():
    # entries with values over 4 bytes point from the note's own start
    body = struct.pack("<H", 1) + struct.pack("<HHII", 0x0100, 3, 4, 18 + 12) + b"\0\0\0\0" + struct.pack("<4H", 516, 480, 256, 256)
    note = b"OLYMPUS\0II\x03\0" + body
    t = tiff.Tiff(_file("<", note))
    maker, mn = t.makernote()
    assert maker == "olympus"
    assert mn.ints(0x0100) == [516, 480, 256, 256]


def test_bad_offsets_and_loops_are_skipped():
    d = bytearray(_file("<"))
    t = tiff.Tiff(bytes(d))
    # point IFD1's next at IFD0 (a loop) and an entry's offset far past the end
    ifd1 = t.ifds[1]
    end = ifd1.pos + 2 + 12 * len(ifd1)
    d[end:end + 4] = struct.pack("<I", t.ifds[0].pos)
    e = t.ifds[0][272]
    entry_at = next(t.ifds[0].pos + 2 + 12 * i for i in range(len(t.ifds[0]))
                    if struct.unpack_from("<H", d, t.ifds[0].pos + 2 + 12 * i)[0] == 272)
    struct.pack_into("<I", d, entry_at + 8, 10 ** 9)
    t2 = tiff.Tiff(bytes(d))
    assert len(t2.ifds) == 2                       # the loop ends the chain
    assert 272 not in t2.ifds[0]                   # the out-of-file value is dropped
    assert e.text() == "Model X"


def test_huge_counts_and_truncation():
    d = _file("<")
    # a count that would need gigabytes is dropped, not allocated
    t = tiff.Tiff(d)
    pos = t.ifds[0].pos
    bad = bytearray(d)
    struct.pack_into("<I", bad, pos + 2 + 4, 0x7FFFFFFF)
    t2 = tiff.Tiff(bytes(bad))
    assert len(t2.ifds[0]) == len(t.ifds[0]) - 1
    for n in (0, 4, 9, 20, len(d) // 2):
        try:
            tt = tiff.Tiff(d[:n])
            tiff.camera_metadata(tt)
        except PixportError:
            pass


def test_not_tiff():
    with pytest.raises(PixportError):
        tiff.Tiff(b"PK\3\4" + b"\0" * 20)
    with pytest.raises(PixportError):
        tiff.Tiff(b"II*\0")


def test_entry_values_types():
    t = Tiff("<").build([{"entries": [(1, 8, [-5, 7]), (2, 9, [-70000]), (3, 11, [1.5]), (4, 12, [2.25]),
                                      (5, 10, [(-1, 4)])]}])
    ifd = tiff.Tiff(t).ifds[0]
    assert ifd.ints(1) == [-5, 7] and ifd.ints(2) == [-70000]
    assert ifd.floats(3) == [1.5] and ifd.floats(4) == [2.25] and ifd.floats(5) == [-0.25]
    assert isinstance(ifd[1].values(), np.ndarray)
