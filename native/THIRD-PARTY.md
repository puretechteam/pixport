# Third-party code in pixport-native

pixport-native is MIT-licensed (see LICENSE). Its wheels also contain the code below, compiled in
statically. Anyone shipping the wheel (or a program built around it) ships these licenses with it.
Every license file named here is inside the source package it comes from, at the version given. Copies of the C and C++ ones are
in `licenses/`, and the wheel carries them (with this file) in its `.dist-info/licenses` folder.

## C and C++: JPEG XL

JPEG XL is read and written with libjxl, built from source by `build.rs` from the tree in the
`jpegxl-src` 0.12.0 crate (libjxl v0.12.0 with its bundled dependencies). Only the library is
built: no tools, no PNG/JPEG/EXR plugins, no libjpeg-turbo, sjpeg or lcms.

| Component | Version | License | License files (in `jpegxl-src-0.12.0/libjxl/`) | Upstream copy |
|---|---|---|---|---|
| libjxl (encoder, decoder, threads, CMS glue) | 0.12.0 | BSD-3-Clause, plus a royalty-free patent grant | `LICENSE`, `PATENTS` (the authors are listed in `AUTHORS`) | https://github.com/libjxl/libjxl/blob/v0.12.0/LICENSE and `/PATENTS` |
| Highway (SIMD) | 1.2.0 (commit 457c891) | Apache-2.0 OR BSD-3-Clause; pixport-native takes it under BSD-3-Clause | `third_party/highway/LICENSE-BSD3` | https://github.com/google/highway/blob/1.2.0/LICENSE-BSD3 |
| Brotli (box and ICC compression) | 1.2.0 (commit 028fb5a) | MIT | `third_party/brotli/LICENSE` | https://github.com/google/brotli/blob/v1.2.0/LICENSE |
| skcms (colour management) | commit 96d9171 (2025-09-16) | BSD-3-Clause | `third_party/skcms/LICENSE` | https://skia.googlesource.com/skcms/+/96d9171c94b937a1b5f0293de7309ac16311b722/LICENSE |

The `jpegxl-src` crate's own build code is not used (pixport-native has its own `build.rs`), so
nothing of it ends up in the wheel; only the libjxl source it carries does.

## C: PNG packing

PNG data is packed with libdeflate, built from the source in the `libdeflate-sys` 1.26.1 crate by
that crate's build code. Its license is `licenses/libdeflate-COPYING` (the crate's
`libdeflate/COPYING`).

| Component | License | Upstream copy |
|---|---|---|
| libdeflate | MIT | https://github.com/ebiggers/libdeflate |

## Rust crates linked into the module

| Crate | Version | License | License files (in the crate package) |
|---|---|---|---|
| pyo3 | 0.29.3 | MIT OR Apache-2.0 | `LICENSE-MIT`, `LICENSE-APACHE` |
| libdeflater | 1.26.1 | Apache-2.0 | `LICENSE` (copied to `licenses/libdeflater-LICENSE`) |
| libdeflate-sys | 1.26.1 | Apache-2.0 | `LICENSE` (the same text) |
| pyo3-ffi | 0.29.3 | MIT OR Apache-2.0 | `LICENSE-MIT`, `LICENSE-APACHE` |
| numpy (rust-numpy) | 0.29.0 | BSD-2-Clause | `LICENSE` |
| ndarray | 0.17.2 | MIT OR Apache-2.0 | `LICENSE-MIT`, `LICENSE-APACHE` |
| num-complex | 0.4.6 | MIT OR Apache-2.0 | `LICENSE-MIT`, `LICENSE-APACHE` |
| num-integer | 0.1.47 | MIT OR Apache-2.0 | `LICENSE-MIT`, `LICENSE-APACHE` |
| num-traits | 0.2.19 | MIT OR Apache-2.0 | `LICENSE-MIT`, `LICENSE-APACHE` |
| matrixmultiply | 0.3.11 | MIT OR Apache-2.0 | `LICENSE-MIT`, `LICENSE-APACHE` |
| rawpointer | 0.2.1 | MIT OR Apache-2.0 | `LICENSE-MIT`, `LICENSE-APACHE` |
| rustc-hash | 2.1.3 | Apache-2.0 OR MIT | `LICENSE-MIT`, `LICENSE-APACHE` |
| libc | 0.2.191 | MIT OR Apache-2.0 | `LICENSE-MIT`, `LICENSE-APACHE` |
| once_cell | 1.21.4 | MIT OR Apache-2.0 | `LICENSE-MIT`, `LICENSE-APACHE` |
| Rust standard library | the toolchain's | MIT OR Apache-2.0 | `LICENSE-MIT`, `LICENSE-APACHE` in https://github.com/rust-lang/rust |

For the crates under "MIT OR Apache-2.0", pixport-native takes the MIT terms; ship `LICENSE-MIT`.
The crate packages are at `https://crates.io/api/v1/crates/<name>/<version>/download` (and in
`~/.cargo/registry/src/` after a build). `cargo tree -e normal` in `native/` lists them.

Used only while building, so not in the wheel: cmake, cc, jpegxl-src's build code, pyo3-build-config,
pyo3-macros, pyo3-macros-backend, syn, quote, proc-macro2, unicode-ident, heck, target-lexicon.

The Windows wheel uses the Microsoft C runtime DLLs (`vcruntime140.dll`, `msvcp140.dll`) that
Python itself installs; they are not bundled. The Linux wheels link the system `libstdc++`.
