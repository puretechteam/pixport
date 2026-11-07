# Changelog

Newest first.

## 2026-11-05: pixport 0.1.0

First release: camera RAW, GIMP XCF, OpenEXR, Radiance HDR and JPEG XL in Python with NumPy, plus PNG and TIFF writing, written to replace ImageMagick in ConvertAll. Measured against ImageMagick 7.1.2 on the same PC:

- **Camera RAW** from Canon (CR2 with sRAW and mRAW, CR3 with cRAW), Nikon, Sony (with its medium and small RAW), Olympus and OM System, Panasonic, Fujifilm (Bayer and X-Trans), Pentax and any DNG. 12 photos to 16-bit PNG through ConvertAll: 59.0 s instead of 91.7 s, 1,032 MB instead of 1,054 MB. The sensor values of 70 of 76 files from 11 makers are bit for bit LibRaw's (the other 6 hold full colour, not a mosaic); against ImageMagick's pictures, the median colour difference is ΔE2000 2.08 (none above 4.1).
- **OpenEXR**, every compression, multi-part and tiled: 251 of 251 parts decode as OpenEXR does; 71 files to PNG in 19.3 s instead of 25.7 s (48 of them faster, the others within 0.4 s; 16 keep their whole picture, four times the pixels ImageMagick writes), and the 14 mipmapped ones ImageMagick can't read are read. Written as ZIP or PIZ, 0.1 to 6.4% smaller than ImageMagick's for the same pixels.
- **Radiance HDR**: 16 files to PNG in 12.4 s instead of 28.2 s; written 0.8% smaller.
- **GIMP XCF**, composited in linear light as GIMP 2.10 does: 28 files in 4.5 s instead of 7.1 s, PNGs 13% smaller.
- **JPEG XL** through libjxl, in the optional pixport-native: 27 files to PNG in 11.6 s instead of 18.3 s, encoded to the same bytes as ImageMagick 1.4 to 2.5 times faster, and a JPEG repacked without loss: 17% smaller (median of 22, 12 to 28%), the very same JPEG back with `jxl.to_jpeg`.
- **PNG and TIFF**, 8 or 16 bits a channel, with EXIF and ICC profiles. PNGs are stored as small as they losslessly go (palette, grey, no opaque alpha, 8 bits when 16 hold only 8; filter and zlib strategy chosen by packing samples) and packed on several cores, with libdeflate when pixport-native is installed: ImageMagick's own pictures packed again by pixport come out 8.8% smaller, 109 of 112 of them.
- **pixport-native** (optional, Rust): the RAW decoders, demosaicing, EXR, XCF and HDR codecs and PNG's Paeth filter compiled, with the same results as the NumPy code, JPEG XL, and PNG packing with libdeflate. Its wheels carry libjxl's, Highway's, Brotli's, skcms's and libdeflate's licenses.
