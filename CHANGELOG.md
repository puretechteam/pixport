# Changelog

Newest first.

## 2026-11-12: pixport 0.2.0, a pixel budget, neutral HDR colours, faster RAW and a hardening pass

- **Greys stay grey in EXR and HDR with a non-D65 white**: the chromaticities are now adapted to D65 with the Bradford transform. ACES AP0 grey (D60) came out sRGB (1.047, 0.995, 0.915) and now comes out (1.000, 1.000, 1.000); XYZE white (1, 1, 1) came out (1.205, 0.948, 0.909) and now comes out (1.000, 1.000, 1.000). A D65 file is unchanged.
- **Radiance HDR shows the stored pixels as they are**: `COLORCORR` is no longer divided out (as other viewers do); it and `EXPOSURE` are kept in `metadata["hdr_colorcorr"]` and `metadata["hdr_exposure"]`.
- **A pixel budget for every reader**: a header can claim any size, so pictures over 2^28 pixels (268 million) are refused before anything is made for them, with `pixport.limits.TooLargeError`; `pixport.read(path, max_pixels=n)` allows more. A 54-byte XCF claiming 20000 x 20000 is now refused using under 5 MB, where it set out to build a 400-million-pixel picture. JPEG XL keeps to the same budget, and pixport-native now reports a picture it has no memory for as an error instead of ending the program.
- Fixed: files far too small for the picture they claim are refused: a 394-byte CR3 claiming 4096 x 4096 (it used 769 MB, now 4 MB), an HDR with fewer than 8 bytes a scanline, and an EXR whose display window is more than 64 times its data and over 16 million pixels (a few bytes could claim 20000 x 10000 of black around 16 pixels; the picture is now the data window, noted in `metadata["exr_cropped"]`). A Deflate DNG tile is inflated no further than the tile needs (8 bytes a sample were let through).
- Faster XCF with less memory: the result is made in place and the sRGB curve without full-size copies. An empty 3000 x 3000 canvas: 279 MB at the peak instead of 639 MB, 0.24 s instead of 0.62 s; a 4-layer 1200 x 1200 file 0.64 s instead of 0.71 s, the same pixels.
- Faster RAW developing: the colour matrix is one matrix product and the black levels, scale and white balance work per position in the mosaic's repeat. A 24-megapixel photo develops in 1.02 s instead of 2.10 s, 834 MB at the peak instead of 1,050 MB; 0.05% of 16-bit values differ, by at most 5 in 65535 (rounding).
- Faster packed 14-bit NEF and other unusual bit depths without pixport-native: a 24-megapixel 14-bit unpack takes 0.28 s and 144 MB instead of 0.46 s and 1,776 MB, the same values.
- Faster HDR without pixport-native: scanlines with no run-length coding are read whole, 1024 x 512 in 0.005 s instead of 0.35 s.
- Fixed: clipped highlights in CR3, Panasonic, Olympus, Pentax and Fujifilm photos stopped short of white and turned pink after white balance when the sensor clips below its bit depth. The white level is now where the picture shows the sensor clipping, as for CR2.
- Fixed: Sony compressed ARWs without their SR2 levels were developed with a white level of 4095 though their values go up to 17204, so they came out blown out.
- Fixed: Nikon's High Efficiency RAW (HE, HE*) and small NEFs were read as ordinary compressed NEFs (garbage or "bad Huffman code"); they now say plainly that they aren't supported.
- Fixed: a DNG whose rows per strip don't divide its height ("A DNG strip is cut short").
- Fixed: reading a RAW's IFDs a second time lost every SubIFD, so the colour matrix was never fitted from a SubIFD preview (Nikon, Sony).
- Fixed: a Pentax maker note in a DNG or JPEG (counting from the file's start, not the note's) was read with the wrong base and lost its levels.
- Fixed: Pentax and Olympus per-colour black levels were placed as if every sensor were RGGB.
- Fixed: an XCF Normal layer ignored its composite mode (clip to backdrop, clip to layer, intersection) and was always composited as union.
- Fixed: an EXR chunk wasn't checked against its slot in the offset table, so a damaged table read with 7 of 8 rows black; it is now an error, as in OpenEXR.
- Fixed: an ARW with no data offset and an EXR with a non-integer data window ended in a bare TypeError; damaged files now always raise `PixportError`, also from writers (a folder that isn't there, a TIFF over 4 GB).
- Fixed: hostile values in CR3 quantisation, CR3 colour data, NEF curves and lossless JPEG point transforms ended in wrong pixels or a confusing error instead of a clear one.
- Fixed: a linear picture's ICC profile (an XCF in linear precision, say) was attached to the sRGB values written to PNG, TIFF and Pillow, so viewers showed it far too bright.
- Fixed: JPEG XL files kept the source's EXIF orientation although their pixels are already upright, so viewers turned them a second time.
- Fixed: `Image.to_pil(bits=16)` on a grey picture failed on Pillow 13.
- Fixed: the `pixport` command ignored `--bits` and `--exposure` for PNG and TIFF, and an unknown extension or a missing folder ended in a traceback. `--bits` now defaults to 16 for RAW, EXR, HDR and deep XCF, as `pixport.write` does.
- `pixport.detect` tells an ordinary TIFF from a DNG (it said "dng" for any `.tif`, then gave a misleading DNG error).
- `pixport.read(path, max_pixels=0)` raises `ValueError` ("max_pixels must be at least 1"), not a "looks damaged" error naming the file.
- pixport-native: JPEG XL only tries a file again as sRGB when its own colours can't be given (any damage made it decode twice), and libjxl's memory budget counts what it already holds.

## 2026-11-05: pixport 0.1.0

First release: camera RAW, GIMP XCF, OpenEXR, Radiance HDR and JPEG XL in Python with NumPy, plus PNG and TIFF writing, written to replace ImageMagick in ConvertAll. Measured against ImageMagick 7.1.2 on the same PC:

- **Camera RAW** from Canon (CR2 with sRAW and mRAW, CR3 with cRAW), Nikon, Sony (with its medium and small RAW), Olympus and OM System, Panasonic, Fujifilm (Bayer and X-Trans), Pentax and any DNG. 12 photos to 16-bit PNG through ConvertAll: 59.0 s instead of 91.7 s, 1,032 MB instead of 1,054 MB. The sensor values of 70 of 76 files from 11 makers are bit for bit LibRaw's (the other 6 hold full colour, not a mosaic); against ImageMagick's pictures, the median colour difference is ΔE2000 2.08 (none above 4.1).
- **OpenEXR**, every compression, multi-part and tiled: 251 of 251 parts decode as OpenEXR does; 71 files to PNG in 19.3 s instead of 25.7 s (48 of them faster, the others within 0.4 s; 16 keep their whole picture, four times the pixels ImageMagick writes), and the 14 mipmapped ones ImageMagick can't read are read. Written as ZIP or PIZ, 0.1 to 6.4% smaller than ImageMagick's for the same pixels.
- **Radiance HDR**: 16 files to PNG in 12.4 s instead of 28.2 s; written 0.8% smaller.
- **GIMP XCF**, composited in linear light as GIMP 2.10 does: 28 files in 4.5 s instead of 7.1 s, PNGs 13% smaller.
- **JPEG XL** through libjxl, in the optional pixport-native: 27 files to PNG in 11.6 s instead of 18.3 s, encoded to the same bytes as ImageMagick 1.4 to 2.5 times faster, and a JPEG repacked without loss: 17% smaller (median of 22, 12 to 28%), the very same JPEG back with `jxl.to_jpeg`.
- **PNG and TIFF**, 8 or 16 bits a channel, with EXIF and ICC profiles. PNGs are stored as small as they losslessly go (palette, grey, no opaque alpha, 8 bits when 16 hold only 8; filter and zlib strategy chosen by packing samples) and packed on several cores, with libdeflate when pixport-native is installed: ImageMagick's own pictures packed again by pixport come out 8.8% smaller, 109 of 112 of them.
- **pixport-native** (optional, Rust): the RAW decoders, demosaicing, EXR, XCF and HDR codecs and PNG's Paeth filter compiled, with the same results as the NumPy code, JPEG XL, and PNG packing with libdeflate. Its wheels carry libjxl's, Highway's, Brotli's, skcms's and libdeflate's licenses.
