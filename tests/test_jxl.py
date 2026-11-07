"""JPEG XL through pixport-native (libjxl): lossless round trips, lossy quality, metadata both
ways, JPEG recompression, and damaged or hostile files ending in PixportError."""

from __future__ import annotations

import io

import numpy as np
import pytest

import pixport
from pixport import PixportError, UnsupportedError, jxl
from pixport.image import Image

needs_native = pytest.mark.skipif(not jxl.available(), reason="pixport-native isn't installed")

EXIF = b"II*\x00\x08\x00\x00\x00\x01\x00\x0f\x01\x02\x00\x06\x00\x00\x00\x1a\x00\x00\x00\x00\x00\x00\x00Pixel\x00"
XMP = b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF/></x:xmpmeta>'


def smooth(h: int = 96, w: int = 128, c: int = 3, seed: int = 1) -> np.ndarray:
    """A photo-like float picture: gradients plus gentle noise, 0..1."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    rng = np.random.default_rng(seed)
    planes = [0.5 + 0.4 * np.sin(x / (9 + 4 * i) + y / (13 + 3 * i)) for i in range(c)]
    px = np.stack(planes, axis=2) + rng.normal(0, 0.01, (h, w, c)).astype(np.float32)
    return np.clip(px, 0, 1).astype(np.float32)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(1.0 / mse)


def test_without_native_says_so(monkeypatch, tmp_path):
    monkeypatch.setattr(pixport._accel, "native", None)
    assert not jxl.available()
    f = tmp_path / "a.jxl"
    f.write_bytes(b"\xff\x0a" + bytes(30))
    with pytest.raises(UnsupportedError, match="JPEG XL needs pixport-native"):
        pixport.read(f)
    with pytest.raises(UnsupportedError, match="JPEG XL needs pixport-native"):
        pixport.write(Image(smooth(4, 4)), tmp_path / "b.jxl")


def test_quality_to_distance():
    assert jxl.distance_from_quality(90) == pytest.approx(1.0)
    assert jxl.distance_from_quality(100) == 0.0
    assert jxl.distance_from_quality(30) == pytest.approx(6.4)
    assert jxl.distance_from_quality(0) == pytest.approx(25.0)


@needs_native
@pytest.mark.parametrize("channels", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_lossless_integer_round_trip(tmp_path, channels, dtype):
    top = np.iinfo(dtype).max
    px = (np.random.default_rng(channels).random((37, 53, channels)) * top).astype(dtype)
    f = tmp_path / "x.jxl"
    pixport.write(Image(px, metadata={"bits": 16 if dtype == np.uint16 else 8}), f, lossless=True)
    img = pixport.read(f)
    assert img.format == "jxl" and img.transfer == "srgb" and img.pixels.dtype == np.float32
    assert img.pixels.shape == px.shape
    assert img.metadata["bits"] == (16 if dtype == np.uint16 else 8)
    back = np.rint(img.pixels * top).astype(dtype)
    assert np.array_equal(back, px)


@needs_native
def test_lossless_float_keeps_hdr(tmp_path):
    px = smooth(40, 30) * 6.0  # well over 1.0, as HDR is
    f = tmp_path / "hdr.jxl"
    pixport.write(Image(px, "linear"), f, lossless=True)
    img = pixport.read(f)
    assert img.transfer == "linear"
    assert np.array_equal(img.pixels, px)


@needs_native
def test_float_srgb_lossless(tmp_path):
    px = smooth(20, 24, 4)
    data = jxl.encode(Image(px, metadata={"jxl_float": True}), lossless=True)
    img = jxl.decode(data)
    assert img.transfer == "srgb" and img.metadata["jxl_float"]
    assert np.array_equal(img.pixels, px)


@needs_native
def test_lossy_default_is_visually_lossless(tmp_path):
    px = smooth(192, 256)
    f = tmp_path / "lossy.jxl"
    pixport.write(Image(px), f)
    lossless = jxl.encode(Image(px), lossless=True)
    assert f.stat().st_size < len(lossless) / 2
    back = pixport.read(f).pixels
    ref = np.rint(px * 255) / 255
    assert psnr(back, ref) > 33  # the fine noise is what distance 1.0 lets go


@needs_native
def test_quality_90_is_distance_1(tmp_path):
    img = Image(smooth(64, 64))
    assert jxl.encode(img) == jxl.encode(img, distance=1.0) == jxl.encode(img, quality=90)
    assert len(jxl.encode(img, quality=50)) < len(jxl.encode(img, quality=95))
    assert jxl.encode(img, quality=100) == jxl.encode(img, lossless=True)


@needs_native
def test_metadata_both_ways(tmp_path):
    from PIL import ImageCms
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    px = smooth(16, 16)
    f = tmp_path / "meta.jxl"
    pixport.write(Image(px, metadata={"exif": EXIF, "xmp": XMP, "icc": icc}), f, lossless=True)
    img = pixport.read(f)
    assert img.metadata["exif"] == EXIF
    assert img.metadata["xmp"] == XMP
    assert img.metadata.get("icc")  # the profile (libjxl may store it as the colour space it describes)
    lossy = jxl.decode(jxl.encode(Image(px, metadata={"exif": EXIF, "xmp": XMP})))
    assert lossy.metadata["exif"] == EXIF and lossy.metadata["xmp"] == XMP
    assert "icc" not in lossy.metadata  # plain sRGB needs no profile


def _p3_profile() -> bytes:
    """A small ICC v2 display profile: Display P3 primaries (D50-adapted), gamma 2.2."""
    import struct

    def s15(v: float) -> bytes:
        return struct.pack(">i", round(v * 65536))

    def xyz(x: float, y: float, z: float) -> bytes:
        return b"XYZ " + bytes(4) + s15(x) + s15(y) + s15(z)

    curve = b"curv" + bytes(4) + struct.pack(">IH", 1, round(2.2 * 256)) + bytes(2)
    desc = b"desc" + bytes(4) + struct.pack(">I", 3) + b"P3\0" + bytes(12) + bytes(67)
    tags = [(b"desc", desc), (b"wtpt", xyz(0.9642, 1.0, 0.8249)), (b"rXYZ", xyz(0.5151, 0.2412, -0.0011)),
            (b"gXYZ", xyz(0.2920, 0.6922, 0.0419)), (b"bXYZ", xyz(0.1571, 0.0666, 0.7841)),
            (b"rTRC", curve), (b"gTRC", curve), (b"bTRC", curve)]
    table, body = b"", b""
    offset = 128 + 4 + 12 * len(tags)
    for sig, data in tags:
        data += bytes(-len(data) % 4)
        table += sig + struct.pack(">II", offset + len(body), len(data))
        body += data
    size = offset + len(body)
    head = (struct.pack(">I", size) + bytes(4) + b"\x02\x10\x00\x00" + b"mntrRGB XYZ " + bytes(12) + b"acsp"
            + bytes(24) + bytes(4) + s15(0.9642) + s15(1.0) + s15(0.8249) + bytes(48))
    return head + struct.pack(">I", len(tags)) + table + body


@needs_native
@pytest.mark.parametrize("lossless", [True, False])
def test_icc_profile_kept_or_converted(lossless):
    px = smooth(32, 32)
    data = jxl.encode(Image(px, metadata={"icc": _p3_profile()}), lossless=lossless)
    own = jxl.decode(data)
    assert own.metadata.get("icc")  # the pixels stay in the P3 space, described by a profile
    assert psnr(own.pixels, px) > 35
    srgb = jxl.decode(data, srgb=True)
    assert "icc" not in srgb.metadata
    # P3 red is outside sRGB: converted, a saturated red clips and the picture changes
    assert np.abs(srgb.pixels - own.pixels).max() > 0.02


@needs_native
def test_jpeg_recompression_is_lossless_and_smaller(tmp_path):
    from PIL import Image as PILImage
    buf = io.BytesIO()
    PILImage.fromarray(np.rint(smooth(240, 320) * 255).astype(np.uint8)).save(buf, "JPEG", quality=90)
    jpeg = buf.getvalue()
    small = jxl.from_jpeg(jpeg)
    assert len(small) < len(jpeg)
    assert jxl.to_jpeg(small) == jpeg
    img = jxl.decode(small)
    assert img.metadata["jxl_jpeg"] is True
    ref = np.asarray(PILImage.open(io.BytesIO(jpeg)), dtype=np.float32) / 255
    assert psnr(img.pixels, ref) > 40  # the same DCT coefficients; only the IDCT's rounding differs
    with pytest.raises(PixportError, match="wasn't made from a JPEG"):
        jxl.to_jpeg(jxl.encode(Image(smooth(8, 8))))
    with pytest.raises(PixportError):
        jxl.from_jpeg(b"\xff\xd8\xff\xe0 not really a JPEG")


@needs_native
def test_detect_and_formats(tmp_path):
    f = tmp_path / "noext"
    f.write_bytes(jxl.encode(Image(smooth(8, 8))))
    assert pixport.detect(f) == "jxl"
    f.write_bytes(jxl.encode(Image(smooth(8, 8), metadata={"exif": EXIF})))  # the container form
    assert f.read_bytes()[:12] == b"\x00\x00\x00\x0cJXL \r\n\x87\n"
    assert pixport.detect(f) == "jxl"
    assert pixport.read(f, "jxl").metadata["exif"] == EXIF
    assert pixport.formats()["jxl"] == {"name": "JPEG XL", "read": True, "write": True}


@needs_native
def test_integer_pixels_accepted(tmp_path):
    px = (smooth(10, 12) * 255).astype(np.uint8)
    img = jxl.decode(jxl.encode(Image(px), lossless=True))
    assert np.array_equal(np.rint(img.pixels * 255).astype(np.uint8), px)


@needs_native
def test_bad_input_to_the_encoder():
    with pytest.raises(PixportError):
        jxl.encode(Image(np.zeros((4, 4, 5), np.float32)))
    with pytest.raises(PixportError):
        jxl.encode(Image(np.zeros((0, 4, 3), np.float32)))
    # effort outside 1..10 is held to it
    assert jxl.decode(jxl.encode(Image(smooth(8, 8)), effort=99)).pixels.shape == (8, 8, 3)


def _size_header(width: int, height: int) -> bytes:
    """A bare codestream whose size header claims width x height, metadata all default."""
    bits: list[int] = []

    def put(value: int, n: int) -> None:
        bits.extend((value >> i) & 1 for i in range(n))

    def u32(value: int) -> None:  # U32(1 + u(9), 1 + u(13), 1 + u(18), 1 + u(30))
        for sel, n in enumerate((9, 13, 18, 30)):
            if value - 1 < (1 << n):
                put(sel, 2)
                put(value - 1, n)
                return

    put(0, 1)  # not a multiple of 8
    u32(height)
    put(0, 3)  # no ratio: the width follows
    u32(width)
    put(1, 1)  # metadata all default
    put(0, 64)
    out = bytearray(b"\xff\x0a")
    for i in range(0, len(bits), 8):
        out.append(sum(b << k for k, b in enumerate(bits[i:i + 8])))
    return bytes(out)


@needs_native
@pytest.mark.parametrize("size", [(60_000, 1), (1, 60_000), (40_000, 40_000)])
def test_hostile_sizes_refused(tmp_path, size):
    f = tmp_path / "big.jxl"
    f.write_bytes(_size_header(*size) + bytes(64))
    with pytest.raises(PixportError, match="past pixport's limits"):
        pixport.read(f)


@needs_native
def test_damaged_files_end_in_pixport_error(tmp_path):
    good = jxl.encode(Image(smooth(48, 64, 4), metadata={"exif": EXIF}), distance=2.0)
    f = tmp_path / "d.jxl"
    for data in (b"", b"\xff\x0a", good[: len(good) // 2], good[:40], b"\xff\x0a" + b"\xff" * 300):
        f.write_bytes(data)
        with pytest.raises(PixportError):
            pixport.read(f, "jxl")
    rng = np.random.default_rng(7)
    decoded = 0
    for _ in range(150):
        bad = bytearray(good)
        for pos in rng.integers(0, len(bad), 3):
            bad[pos] ^= 1 << int(rng.integers(0, 8))
        f.write_bytes(bytes(bad))
        try:
            img = pixport.read(f, "jxl")
        except PixportError:
            continue
        decoded += 1
        assert img.pixels.shape[2] in (1, 2, 3, 4)
    assert decoded < 150
