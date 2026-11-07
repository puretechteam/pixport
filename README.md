# pixport

Read camera RAW photos (Canon CR2 and CR3, Nikon NEF, Sony ARW, Olympus and OM System ORF, Panasonic RW2, Fujifilm RAF with Bayer and X-Trans sensors, Pentax PEF, any DNG), GIMP's XCF, OpenEXR, Radiance HDR and JPEG XL, and write PNG and TIFF (16 bits a channel), EXR, HDR and JPEG XL, in Python with NumPy. No ImageMagick, no LibRaw.

```
pip install git+https://github.com/puretechteam/pixport
pip install https://github.com/puretechteam/pixport/releases/download/v0.1.0/pixport_native-0.1.0-cp310-abi3-win_amd64.whl
```

The second line is optional for everything but JPEG XL: it's the compiled add-on for 64-bit Windows (each [release](https://github.com/puretechteam/pixport/releases) has Linux and macOS wheels too). Without it pixport reads and writes the rest the same way with its NumPy code, more slowly: a big RAW photo takes up to a minute.

```python
import pixport

img = pixport.read("photo.cr3")             # an Image: float32 pixels (h, w, channels) and metadata
rgb = img.to_srgb(bits=8)                   # display-ready sRGB as a uint8 array
pixport.write(img, "photo.png")             # 16 bits a channel, with the photo's EXIF
pixport.write(img, "photo.exr")             # EXR, HDR and JPEG XL too
pixport.formats()                           # {"cr3": {"name": "Canon RAW 3 (CR3)", "read": True, "write": False}, ...}
```

or from a terminal:

```
pixport photo.nef photo.png
pixport render.exr render.jpg --exposure -1
pixport --formats
```

## Why pixport?

[ConvertAll](https://github.com/puretechteam/ConvertAll), a drag-and-drop file converter, used ImageMagick (about 60 MB to install) for these formats. pixport replaces it there: on the same files it is faster, its RAW PNGs are smaller, and its pictures match LibRaw's and OpenEXR's. It's a sibling of [meshport](https://github.com/puretechteam/meshport) (3D), [packport](https://github.com/puretechteam/packport) (archives) and [bookport](https://github.com/puretechteam/bookport) (ebooks and documents).

## What it reads and writes

| Format | Read | Write |
|---|---|---|
| Camera RAW: CR2 (with sRAW and mRAW), CR3 (with cRAW), NEF (Nikon's High Efficiency RAW excepted), ARW (with Sony's medium and small RAW), ORF, RW2, RAF (Bayer and X-Trans), PEF, DNG | ✓ | |
| GIMP XCF (8 to 32 bits, layers, masks, groups, every layer mode) | ✓ | |
| OpenEXR (scanline and tiled, multi-part, every compression: RLE, ZIPS, ZIP, PIZ, PXR24, B44, B44A, DWAA, DWAB) | ✓ | ✓ (ZIP or PIZ) |
| Radiance HDR (RGBE) | ✓ | ✓ |
| JPEG XL (with pixport-native) | ✓ | ✓ (lossy, lossless, and a JPEG repacked without loss) |
| PNG, TIFF | | ✓ (8 or 16 bits a channel) |

**Camera RAW** is developed the way LibRaw (and so ImageMagick) does by default: black and white levels, the camera's white balance, demosaicing, the camera's colour matrix, automatic brightness and the BT.709 curve, turned as the camera was held, with EXIF kept. The colour matrix comes from the file when it has one (DNG, Olympus, Sony) or is fitted to the camera's own preview of the same picture; `img.metadata["colour"]` says which. `pixport.raw.develop` takes `half=True` (a quick half-size picture), `white_balance="daylight"` and `tone="linear"` for scene-linear values.

**XCF** pictures are composited as GIMP 2.10 shows them, in linear light. **EXR** keeps every part, channel and layer the file has; the picture is its display window, as OpenEXR defines it. **JPEG XL** goes through libjxl, the reference library; `pixport.jxl.from_jpeg` repacks a JPEG into JPEG XL without changing a pixel, and `to_jpeg` gives the very same JPEG back.

**PNG** is written as small as it losslessly goes: 16 bits that only hold 8 become 8, an opaque alpha channel is dropped, grey stays grey, 256 colours or fewer become a palette, and the filter (one for the band, or each row's own) and zlib strategy are chosen per band of rows by packing a sample each way. It packs on several cores. With pixport-native the packing is libdeflate's: on photos about 4 times as fast as zlib at the same size, and 4 to 6% smaller on HDR, XCF and JPEG XL pictures.

## Speed and size

Measured on the same PC against ImageMagick 7.1.2, one conversion at a time, the two taking turns on each file, best of 3, with pixport-native installed and Python already running, as ConvertAll calls it (the `pixport` command adds about 0.25 s of start-up to each file):

| Files | ImageMagick | pixport |
|---|---|---|
| 12 RAW photos (Apple DNG, Canon CR2 and CR3, sRAW, mRAW) → 16-bit PNG, through ConvertAll | 91.7 s, 1,054 MB | 59.0 s, 1,032 MB (10 of 12 faster) |
| 71 EXR files → PNG | 25.7 s, 71 MB | 19.3 s, 92.5 MB; 26.5 MB at `bits=8` (48 of 71 faster, the others within 0.4 s) |
| 16 HDR files → PNG | 28.2 s, 39 MB | 12.4 s, 51 MB; 18 MB at `bits=8` (every one faster) |
| 28 XCF files → PNG (ImageMagick with `-flatten`) | 7.1 s, 3.6 MB | 4.5 s, 3.1 MB (22 of 28 faster) |
| 27 JPEG XL files → PNG | 18.3 s, 42.0 MB | 11.6 s, 39.6 MB (21 of 27 faster) |
| JPEG XL encoding, same settings | | the same bytes, 1.4 to 2.5 times faster |
| 22 JPEGs → JPEG XL | encoded again, with loss | 17% smaller (median, 12 to 28%), every pixel the same, and the JPEG rebuilds byte for byte |

HDR and EXR PNGs come out bigger than ImageMagick's at 16 bits a channel, for a reason. ImageMagick writes an HDR file's PNG as linear light marked `gAMA 1.0`, which most viewers (Windows Photos, Pillow, anything ignoring that chunk) show far too dark; pixport writes sRGB, which looks right everywhere and packs less tightly. For 16 of the EXR files ImageMagick keeps only the data window, pixport the whole display window, four times the pixels. `bits=8` makes them smaller than ImageMagick's. To compare the PNG writers alone, the 112 PNGs ImageMagick made in these runs were packed again by pixport: 8.8% smaller in all, 109 of the 112 smaller.

## How it's tested

The suite has 378 tests (with `PIXPORT_PURE=1`, 295 run in pure Python and 78 that need pixport-native skip); any warning fails one. Against these oracles:

- **LibRaw** (through rawpy): the raw sensor values of 70 of 76 RAW files from 11 makers (raw.pixls.us, CC0) are bit for bit the same; the other 6 store full colour instead of a mosaic (Canon sRAW and mRAW, Sony's medium and small ARW) and are checked by colour. Against ImageMagick's developed pictures the median colour difference is ΔE2000 2.08 (no file above 4.1) and the median PSNR 36.5 dB; the difference is the colour matrix, which ImageMagick takes from Adobe's tables.
- **OpenEXR 3**: 251 of 251 parts of the OpenEXR sample images decode to the same values, every compression included; ImageMagick fails on 14 mipmapped files and mixes up stereo views.
- **djxl** (libjxl): the JPEG XL conformance files decode to the same pixels.
- **XCF**: files built in the tests for every precision, compression and colour mode, and every layer mode checked against values worked out from GIMP's definitions.
- **Pillow** and **OpenCV** read back every PNG and TIFF written (8 and 16 bits, palettes, every filter), and a decoder written in the tests apart from pixport's code checks each filter.
- **pixport-native against NumPy**: the same results both ways for every decoder and the PNG filter.
- **Damaged files** (cut short, impossible sizes and counts) raise `pixport.PixportError`, never anything else, without huge allocations.

## Development

```
python -m venv .venv
.venv\Scripts\pip install -e .[dev]
.venv\Scripts\python -m pytest
```

pixport-native needs Rust, maturin, CMake and a C++17 compiler (it builds libjxl from source): `maturin develop --release -m native/Cargo.toml`. On Windows, with Visual Studio 2022's CMake on PATH, set `CMAKE_GENERATOR=Visual Studio 17 2022` if a newer Visual Studio is installed too. A Rust build keeps the paths of its sources; for a wheel you'll hand out, set `RUSTFLAGS=--remap-path-prefix=%USERPROFILE%=~` first (the release's wheels are built by GitHub Actions). libjxl's own paths are trimmed by `native/build.rs`.

Code layout (`src/pixport/`): `image.py` (the Image, sRGB curves), `registry.py` (formats, read, write), `png.py`, `tiffwrite.py`, `exr.py`, `hdr.py`, `xcf.py`, `jxl.py`; `raw/` with a TIFF reader, lossless JPEG, a module per maker, `develop.py` and the colour matrices (`colour.py`, `preview.py`); `_accel.py` loads pixport-native, whose Rust source is in `native/`.

## Credits

Written from the formats' specifications and public descriptions: Adobe's DNG specification, TIFF 6.0, ITU-T T.81 (lossless JPEG), the OpenEXR file layout, Greg Ward's RGBE format, GIMP's XCF documentation, and public notes on Canon's CR3 and the makers' RAW layouts. LibRaw, OpenEXR and djxl were used only as black boxes to check results; no GPL or LGPL code is in pixport. pixport-native contains libjxl (BSD-3-Clause) with Highway, Brotli and skcms; their licenses ship in the wheel (see `native/THIRD-PARTY.md`).

## License

MIT, see [LICENSE](LICENSE).
