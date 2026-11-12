"""Hostile sizes, damaged structures and corner cases: a pixel budget every reader keeps to, files
far too small for the picture they claim, and the cases a review of the readers and writers turned
up (each test names what used to go wrong)."""

from __future__ import annotations

import struct
import subprocess
import sys
import tracemalloc

import numpy as np
import pytest
import test_exr as E
import test_raw_cr3 as C
from raw_writer import Tiff
from raw_writer import dng as make_dng
from xcf_writer import L, XcfWriter

import pixport
from pixport import exr, hdr, jxl, limits
from pixport.errors import PixportError, UnsupportedError
from pixport.image import Image
from pixport.raw import RawData, common, cr3, dng, ljpeg, nikon, olympus, sony, tiff


def _empty_xcf(w: int, h: int) -> bytes:
    """An XCF with no layers: 54 bytes, any canvas size."""
    return (b"gimp xcf v011\0" + struct.pack(">III", w, h, 0) + struct.pack(">I", 150) + struct.pack(">II", 0, 0)
            + struct.pack(">QQ", 0, 0))


# ---------------------------------------------------------------------------------------------
# the pixel budget


def test_a_few_bytes_claiming_a_giant_picture_are_refused_before_anything_is_made(tmp_path):
    p = tmp_path / "giant.xcf"
    p.write_bytes(_empty_xcf(20000, 20000))  # 400 million pixels from 54 bytes (it used to read them)
    tracemalloc.start()
    with pytest.raises(limits.TooLargeError, match="max_pixels"):
        pixport.read(p)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < 5 << 20
    h = tmp_path / "giant.hdr"
    h.write_bytes(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 20000 +X 20000\n" + bytes(64))
    with pytest.raises(limits.TooLargeError):
        pixport.read(h)


def test_max_pixels_sets_the_budget_for_one_call(tmp_path):
    p = tmp_path / "small.xcf"
    p.write_bytes(_empty_xcf(64, 64))
    assert pixport.read(p).pixels.shape == (64, 64, 4)
    with pytest.raises(limits.TooLargeError):
        pixport.read(p, max_pixels=1000)
    assert pixport.read(p).pixels.shape == (64, 64, 4)  # back to the default afterwards
    big = tmp_path / "big.xcf"
    big.write_bytes(_empty_xcf(17000, 16000))  # 272 million pixels: over the default, allowed here
    with pytest.raises(limits.TooLargeError):
        pixport.read(big)
    with limits.budget(17000 * 16000):
        assert limits.max_pixels() == 17000 * 16000
    assert limits.max_pixels() == limits.MAX_PIXELS


def test_an_empty_xcf_canvas_costs_half_the_memory(tmp_path):
    p = tmp_path / "empty.xcf"
    p.write_bytes(_empty_xcf(1500, 1500))
    tracemalloc.start()
    img = pixport.read(p)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert img.pixels.shape == (1500, 1500, 4) and not img.pixels.any()
    assert peak / (1500 * 1500) < 40  # bytes a pixel (it took 71: the curve and the result were extra copies)


def test_a_cr3_with_no_picture_data_is_refused(tmp_path):
    w = h = 1024
    head = struct.pack(">HHIHH", 0xFF01, 8, 0, 0, 0)
    for p in range(4):
        head += struct.pack(">HHIB3x", 0xFF02, 8, 0, (p << 4) | 8)
        head += struct.pack(">HHII", 0xFF03, 8, 0, 4 << 19)
    entry = C.craw_entry(w, h, C.cmp1(w, h, w, h, 0, len(head)))
    ftyp = C.box(b"ftyp", b"crx \x00\x00\x00\x01crx isom")
    at = len(ftyp) + len(C.box(b"moov", C.trak(entry, len(head), 0))) + 8
    f = tmp_path / "empty.cr3"
    f.write_bytes(ftyp + C.box(b"moov", C.trak(entry, len(head), at)) + C.box(b"mdat", head))
    tracemalloc.start()
    with pytest.raises(PixportError, match="no picture data"):
        pixport.read(f)  # 394 bytes made a 4096 x 4096 picture (235 MB) before
    assert tracemalloc.get_traced_memory()[1] < 5 << 20
    tracemalloc.stop()


def test_an_exr_display_window_far_larger_than_its_data_is_not_filled_in(tmp_path):
    pic = {k: np.full((4, 4), 1.0, np.float16) for k in "RGB"}
    p = tmp_path / "huge.exr"
    p.write_bytes(E.make_exr(pic, dw=(0, 0, 3, 3), disp=(0, 0, 19999, 9999)))
    img = pixport.read(p)  # was a 200-million-pixel picture of black around 16 pixels
    assert img.pixels.shape == (4, 4, 3) and "exr_cropped" in img.metadata
    q = tmp_path / "framed.exr"
    q.write_bytes(E.make_exr(pic, dw=(2, 2, 5, 5), disp=(0, 0, 9, 9)))
    framed = pixport.read(q)  # an ordinary display window is still the picture
    assert framed.pixels.shape == (10, 10, 3) and "exr_cropped" not in framed.metadata


def test_an_hdr_with_far_too_little_data_is_refused(tmp_path):
    p = tmp_path / "short.hdr"
    p.write_bytes(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1000 +X 1000\n" + bytes(100))
    with pytest.raises(PixportError, match="too short"):
        pixport.read(p)


@pytest.mark.skipif(not jxl.available(), reason="pixport-native isn't installed")
def test_jpeg_xl_keeps_to_the_budget():
    data = jxl.encode(Image(np.zeros((64, 64, 3), np.float32), "srgb"), lossless=True)
    assert jxl.decode(data).pixels.shape == (64, 64, 3)
    with limits.budget(1000), pytest.raises(PixportError, match="limits"):
        jxl.decode(data)


# ---------------------------------------------------------------------------------------------
# damaged structures end in PixportError, never in another exception


def test_an_arw_with_no_data_offset_and_an_exr_with_float_windows_are_damage(tmp_path):
    raw = {"entries": [(254, 4, [0]), (256, 4, [8]), (257, 4, [8]), (258, 3, [14]), (259, 3, [1]), (262, 3, [32803]),
                       (279, 4, [128])]}
    p = tmp_path / "nostrip.arw"
    p.write_bytes(Tiff("<").build([{"entries": [(271, 2, "SONY")], "children": {330: [raw]}}]) + bytes(200))
    with pytest.raises(PixportError, match="offset"):
        pixport.read(p)  # was a bare TypeError
    pic = {k: np.ones((4, 4), np.float16) for k in "RGB"}
    q = tmp_path / "box2f.exr"
    q.write_bytes(E.make_exr(pic).replace(b"dataWindow\0box2i\0", b"dataWindow\0box2f\0"))
    with pytest.raises(PixportError, match="window"):
        pixport.read(q)


def test_type_and_attribute_errors_from_a_reader_are_damage(tmp_path, monkeypatch):
    p = tmp_path / "x.hdr"
    p.write_bytes(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 1\n" + bytes(4))
    for exc in (TypeError, AttributeError):
        def boom(*a, exc=exc, **k):
            raise exc("inside the reader")
        monkeypatch.setattr(hdr, "read", boom)
        with pytest.raises(PixportError, match="damaged"):
            pixport.read(p)


def test_exr_chunks_must_be_the_ones_the_offset_table_names(tmp_path):
    pic = {k: np.full((8, 4), 1.0, np.float16) for k in "RGB"}
    d = bytearray(E.make_exr(pic))
    _f, parts, pos = exr.parse(bytes(d))
    exr._offsets(bytes(d), parts, pos)
    first = int(parts[0].offsets[0])
    for i in range(8):
        struct.pack_into("<Q", d, pos + 8 * i, first)  # every slot points at the first chunk
    p = tmp_path / "dup.exr"
    p.write_bytes(bytes(d))
    with pytest.raises(PixportError, match="offset table"):
        pixport.read(p)  # read "fine" with 7 of 8 rows black before


# ---------------------------------------------------------------------------------------------
# RAW


def test_a_dng_whose_last_strip_is_short(tmp_path):
    cfa = (np.arange(64 * 50).reshape(50, 64) % 4000).astype(np.uint16)
    h, w, rps = 50, 64, 16
    strips = [cfa[i:i + rps].astype("<u2").tobytes() for i in range(0, h, rps)]
    raw = {"entries": [(254, 4, [0]), (256, 4, [w]), (257, 4, [h]), (258, 3, [16]), (259, 3, [1]), (262, 3, [32803]),
                       (277, 3, [1]), (33421, 3, [2, 2]), (33422, 1, [0, 1, 1, 2]),
                       (273, 4, tuple(f"@s{i}" for i in range(len(strips)))), (279, 4, [len(b) for b in strips]),
                       (278, 4, [rps])]}
    ifd0 = {"entries": [(254, 4, [1]), (256, 4, [8]), (257, 4, [8]), (274, 3, [1]), (50706, 1, [1, 4, 0, 0])],
            "children": {330: [raw]}}
    p = tmp_path / "rps.dng"
    p.write_bytes(Tiff("<").build([ifd0], {f"s{i}": b for i, b in enumerate(strips)}))
    assert np.array_equal(dng.decode(p).cfa, cfa)  # "A DNG strip is cut short" before


def test_a_deflate_dng_tile_inflates_no_further_than_it_needs(tmp_path, monkeypatch):
    import zlib
    cfa = np.zeros((32, 32), np.uint16)
    p = tmp_path / "z.dng"
    p.write_bytes(make_dng(cfa, compression=8, tile=32).replace(cfa.astype("<u2").tobytes(),
                                                                 zlib.compress(bytes(32 * 32 * 8))))
    asked = []
    real = zlib.decompressobj

    class Spy:
        def __init__(self):
            self.z = real()

        def decompress(self, data, max_length=0):
            asked.append(max_length)
            return self.z.decompress(data, max_length)
    monkeypatch.setattr(zlib, "decompressobj", Spy)
    try:
        dng.decode(p)
    except PixportError:
        pass
    assert asked and max(asked) <= 32 * 32 * 2 + 32 + 64  # 8 bytes a sample were let through before


def test_walking_the_ifds_again_finds_the_same_ones():
    t = tiff.Tiff(make_dng((np.arange(64 * 50).reshape(50, 64) % 4000).astype(np.uint16)))
    first = t.all_ifds()
    assert len(first) == 2 and len(t.all_ifds()) == 2  # the second walk lost every SubIFD
    assert all(a is b for a, b in zip(first, t.all_ifds()))


def test_a_pentax_note_in_a_dng_counts_from_the_files_start():
    # a maker note whose values count from the TIFF's start (as in DNGs and JPEGs), not the note's
    value = struct.pack("<4H", 600, 601, 602, 603)
    t = Tiff("<")
    body = struct.pack("<H", 2) + struct.pack("<HHIHH", 0x0005, 3, 1, 7, 0)  # a value that fits in its entry
    body += struct.pack("<HHII", 0x0200, 3, 4, 0) + b"\0\0\0\0"               # and one at an offset
    exif = {"entries": [(37500, 7, b"AOC\0II" + body)]}
    base = t.build([{"entries": [(271, 2, "PENTAX")], "children": {34665: [exif]}}], {"v": value})
    vat = base.rindex(value)
    data = bytearray(base)
    note = data.index(b"AOC\0II")
    struct.pack_into("<I", data, note + 6 + 2 + 12 + 8, vat)  # the second value's offset from the file's start
    assert vat + note > len(data)  # counted from the note's start it would be past the end
    maker, mn = tiff.Tiff(bytes(data)).makernote()
    assert maker == "pentax" and mn.int(0x0005) == 7
    assert mn.ints(0x0200) == [600, 601, 602, 603]  # lost before: the note's own base was always taken


def test_cr3_clipped_highlights_reach_white(tmp_path):
    m = np.minimum(C.mosaic_for_test(32, 48), 15000)
    m[8:16, 10:26] = 15000  # the sensor clips at 15000, below 14 bits' 16383
    p = tmp_path / "clip.cr3"
    p.write_bytes(C.build_cr3(m, 24, 0))
    assert cr3.decode(p).white == 15000  # was 16383: clipped areas came out short of white, then pink
    m[9, 12] = 16383
    p.write_bytes(C.build_cr3(m, 24, 0))
    assert cr3.decode(p).white == 16383  # a photosite at 16383: the bit depth's top it is


def test_orf_clipped_highlights_reach_white(tmp_path):
    h, w = 32, 48
    rng = np.random.default_rng(2)
    cfa = (rng.integers(300, 3000, (h, w))).astype(np.uint16)
    cfa[4:12, 4:20] = 3950  # clips below 12 bits' 4095
    ifd0 = {"entries": [(256, 4, [w]), (257, 4, [h]), (258, 3, [16]), (259, 3, [1]), (273, 4, ("@s",)),
                        (279, 4, [cfa.nbytes]), (271, 2, "OLYMPUS")]}
    p = tmp_path / "x.orf"
    p.write_bytes(Tiff("<").build([ifd0], {"s": cfa.astype("<u2").tobytes()}, magic=0x4F52))
    assert olympus.decode(p).white == 3950


def test_per_colour_black_levels_follow_the_mosaic():
    levels = [10, 20, 21, 30]  # R, G, G, B
    assert common.colour_black(levels, np.array([[0, 1], [1, 2]])).tolist() == [[10, 20], [21, 30]]
    assert common.colour_black(levels, np.array([[1, 0], [2, 1]])).tolist() == [[20, 10], [30, 21]]  # GRBG
    assert common.colour_black(levels, np.array([[2, 1], [1, 0]])).tolist() == [[30, 20], [21, 10]]  # BGGR


def _nef(note_tags) -> bytes:
    inner = Tiff(">").build([{"entries": note_tags}])
    note = b"Nikon\0\x02\x10\0\0" + inner
    raw = {"entries": [(254, 4, [0]), (256, 4, [64]), (257, 4, [32]), (258, 3, [14]), (259, 3, [34713]),
                       (262, 3, [32803]), (273, 4, ("@s",)), (279, 4, [600])]}
    ifd0 = {"entries": [(271, 2, "NIKON CORPORATION")], "children": {330: [raw], 34665: [{"entries": [(37500, 7, note)]}]}}
    return Tiff("<").build([ifd0], {"s": bytes(range(256)) * 3})


def test_nikon_high_efficiency_raw_says_so(tmp_path):
    p = tmp_path / "he.nef"
    for kind in (13, 14):
        p.write_bytes(_nef([(0x93, 3, [kind])]))
        with pytest.raises(UnsupportedError, match="High Efficiency"):
            pixport.read(p)  # "no linearization table" (or garbage through the Huffman decoder) before
    p.write_bytes(_nef([(0x93, 3, [8])]))
    with pytest.raises(UnsupportedError, match="small NEF"):
        pixport.read(p)


def test_hostile_curves_and_point_transforms_are_damage(monkeypatch):
    from pixport import _accel
    monkeypatch.setattr(_accel, "native", None)  # the Python decoder's own checks
    meta = bytes([0x44, 0x40]) + struct.pack(">4hH", 0, 0, 0, 0, 5000) + bytes(2 * 5000)
    with pytest.raises(PixportError, match="curve"):
        nikon._curve(meta, ">", 14)  # a divide by zero and a curve of zeros before
    head = bytearray(b"\xff\xd8")
    head += b"\xff\xc3" + struct.pack(">HBHHB", 11, 12, 2, 2, 1) + bytes([1, 0x11, 0])
    head += b"\xff\xc4" + struct.pack(">H", 3 + 16 + 1) + bytes([0]) + bytes([1] + [0] * 15) + bytes([0])
    head += b"\xff\xda" + struct.pack(">HB", 8, 1) + bytes([1, 0]) + bytes([1, 0, 12])
    with pytest.raises(PixportError, match="point transform"):
        ljpeg.decode(bytes(head) + bytes(16) + b"\xff\xd9", 0, len(head) + 18)


def test_cr3_steps_and_colour_data_from_damaged_files():
    assert cr3.plain_step(255) == cr3.MAX_STEP  # wrapped to 0 in int32 before
    assert (np.ones(4, np.int32) * np.int32(cr3.plain_step(255))).min() > 0
    cd = np.zeros(0x61, np.uint16)
    cd[0] = 99  # a layout pixport doesn't know: the levels are looked for
    assert cr3._levels(cd) == (None, None)  # an IndexError before


def test_sony_compressed_without_its_levels_whites_at_the_curves_top(tmp_path):
    w, h = 64, 4
    raw = {"entries": [(254, 4, [0]), (256, 4, [w]), (257, 4, [h]), (258, 3, [12]), (259, 3, [32767]),
                       (262, 3, [32803]), (273, 4, ("@s",)), (279, 4, [w * h])]}
    p = tmp_path / "x.arw"
    p.write_bytes(Tiff("<").build([{"entries": [(271, 2, "SONY")], "children": {330: [raw]}}],
                                  {"s": bytes(range(256)) * (w * h // 256)}))
    top = int(sony.sony_curve([8000, 10400, 12900, 14100])[0x7FF << 1])
    assert sony.decode(p).white == top  # 4095 before, with values up to the curve's 17204


def test_bit_unpacking_matches_bit_by_bit():
    rng = np.random.default_rng(4)
    for bits in (10, 11, 13, 14):
        for order in ("msb", "lsb"):
            count = 1003
            buf = rng.integers(0, 256, (count * bits + 7) // 8, dtype=np.uint8).tobytes()
            flat = np.unpackbits(np.frombuffer(buf, np.uint8), bitorder="big" if order == "msb" else "little")
            want = [int("".join(map(str, flat[i * bits:(i + 1) * bits][:: 1 if order == "msb" else -1])), 2)
                    for i in range(count)]
            assert common.unpack_bits(buf, count, bits, order).tolist() == want


def test_develop_with_levels_that_repeat_and_that_dont():
    rng = np.random.default_rng(6)
    cfa = rng.integers(300, 4000, (40, 60)).astype(np.uint16)
    a = RawData(cfa=cfa, black=np.array([[256.0, 257.0], [255.0, 258.0]]), white=4095.0, wb=(2.0, 1.0, 1.5))
    b = RawData(cfa=cfa, black=np.array([[256.0, 257.0], [255.0, 258.0]]), white=4095.0, wb=(2.0, 1.0, 1.5),
                black_rows=np.zeros(40))  # the general path
    from pixport.raw.develop import develop
    assert np.array_equal(develop(a).pixels, develop(b).pixels)
    assert np.array_equal(develop(a, white_balance="auto").pixels, develop(b, white_balance="auto").pixels)


# ---------------------------------------------------------------------------------------------
# XCF, HDR, writers and the command line


def test_a_normal_layer_keeps_its_composite_mode(tmp_path):
    top = np.full((4, 4, 4), 255, np.uint8)
    half = np.zeros((4, 4, 4), np.uint8)
    half[:, :2] = 255  # the left half opaque, the right half empty
    want = {None: [1, 1, 1, 1], 1: [1, 1, 1, 1], 2: [1, 1, 0, 0], 3: [1, 1, 1, 1], 4: [1, 1, 0, 0]}
    for mode, alpha in want.items():
        p = tmp_path / "c.xcf"
        p.write_bytes(XcfWriter(4, 4).build([L("top", top, composite_mode=mode), L("bottom", half)]))
        assert pixport.read(p).pixels[0, :, 3].tolist() == alpha, mode  # every mode came out as union


def test_flat_and_run_length_hdr_rows_give_the_same_pixels(monkeypatch):
    monkeypatch.setattr(hdr, "native", None)
    rng = np.random.default_rng(1)
    w = 40
    rows = rng.integers(2, 255, (3, w, 4), dtype=np.uint8)
    flat = b"".join(r.tobytes() for r in rows)
    out = hdr.decode_rgbe(flat, 0, 3, w)
    assert np.array_equal(out, rows)
    runs = rows[0, :1].tobytes() + bytes([1, 1, 1, w - 1])  # one pixel, then the old form's repeat
    out = hdr.decode_rgbe(runs + flat[4 * w:], 0, 3, w)
    assert (out[0] == rows[0, 0]).all() and np.array_equal(out[1:], rows[1:])


def test_to_pil_writes_16_bit_grey():
    pytest.importorskip("PIL")
    img = Image(np.linspace(0, 1, 12, dtype=np.float32).reshape(3, 4, 1), "srgb")
    pil = img.to_pil(bits=16)  # "Invalid mode for data type" on Pillow 13
    assert pil.mode == "I;16" and pil.getpixel((3, 2)) == 65535


def test_a_linear_pictures_profile_isnt_written_with_srgb_values(tmp_path):
    icc = b"\0" * 128 + b"a linear-light profile"
    lin = Image(np.full((4, 4, 3), 0.2, np.float32), "linear", {"icc": icc})
    srgb = Image(np.full((4, 4, 3), 0.2, np.float32), "srgb", {"icc": icc})
    for name in ("a.png", "a.tif"):
        pixport.write(lin, tmp_path / name)
        assert b"a linear-light profile" not in _unpacked(tmp_path / name)  # shown far too bright before
        pixport.write(srgb, tmp_path / name)
        assert b"a linear-light profile" in _unpacked(tmp_path / name)
    assert lin.display_icc() is None and srgb.display_icc() == icc


def _unpacked(p) -> bytes:
    import zlib
    data = p.read_bytes()
    if data[:4] == b"\x89PNG":
        at = data.find(b"iCCP")
        return zlib.decompress(data[data.index(b"\0\0", at + 4) + 2:]) if at > 0 else b""
    return data


@pytest.mark.skipif(not jxl.available(), reason="pixport-native isn't installed")
def test_jpeg_xl_stores_the_exif_upright():
    t = Tiff("<")
    exif = t.build([{"entries": [(0x0112, 3, [6]), (271, 2, "Maker")]}])
    img = Image(np.zeros((8, 8, 3), np.float32), "srgb", {"exif": exif})
    back = jxl.decode(jxl.encode(img, lossless=True))
    assert tiff.Tiff(back.metadata["exif"]).ifds[0].int(0x0112) == 1  # 6 before: viewers turned it again


def test_tiff_is_told_apart_from_dng(tmp_path):
    plain = tmp_path / "scan.tif"
    plain.write_bytes(Tiff("<").build([{"entries": [(256, 4, [4]), (257, 4, [4])]}]))
    assert pixport.detect(plain) == "tiff"  # "dng" before, and a misleading DNG error
    with pytest.raises(PixportError, match="doesn't read tiff"):
        pixport.read(plain)
    raw = tmp_path / "photo.tif"
    raw.write_bytes(make_dng(np.zeros((8, 8), np.uint16)))
    assert pixport.detect(raw) == "dng"


def test_write_errors_are_pixport_errors(tmp_path):
    img = Image(np.zeros((4, 4, 3), np.float32), "srgb")
    with pytest.raises(PixportError, match="Can't write"):
        pixport.write(img, tmp_path / "no" / "such" / "folder" / "a.png")


def test_the_command_line_takes_bits_and_exposure(tmp_path):
    pytest.importorskip("PIL")
    src = tmp_path / "a.hdr"
    pixport.write(Image(np.full((4, 4, 3), 0.25, np.float32), "linear"), src)
    cli = [sys.executable, "-m", "pixport"]
    run = lambda *a: subprocess.run(cli + [str(x) for x in a], capture_output=True, text=True, timeout=120, check=False)
    assert run(src, tmp_path / "o8.png", "--bits", "8").returncode == 0
    assert (tmp_path / "o8.png").read_bytes()[24] == 8  # the IHDR bit depth: 16 before
    run(src, tmp_path / "e0.png")
    run(src, tmp_path / "e3.png", "--exposure", "3")
    assert (tmp_path / "e0.png").read_bytes() != (tmp_path / "e3.png").read_bytes()  # the same bytes before
    for bad in (tmp_path / "o.xyz", tmp_path / "missing" / "o.png"):
        r = run(src, bad)
        assert r.returncode == 1 and "Traceback" not in r.stderr and "pixport:" in r.stderr


def test_a_budget_below_one_pixel_is_a_mistake_in_the_call_not_damage(tmp_path):
    p = tmp_path / "small.xcf"
    p.write_bytes(_empty_xcf(8, 8))
    with pytest.raises(ValueError, match="at least 1"):  # "looks damaged (ValueError ...)" before
        pixport.read(p, max_pixels=0)
