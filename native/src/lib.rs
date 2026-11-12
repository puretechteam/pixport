//! pixport's hot loops in Rust: the same results as its NumPy code, faster, plus JPEG XL. pixport
//! uses them when this module is installed and falls back to NumPy when it isn't (or when
//! PIXPORT_PURE=1). Bad input is reported as pixport's own PixportError; nothing here panics on a
//! damaged file (every read is bounds-checked).
//!
//! Each format lives in its own file and registers itself in `pixport_native` below.

use pyo3::prelude::*;

mod exr;
mod hdr;
mod jxl;
mod png;
mod raw_cr3;
mod raw_demosaic;
mod raw_fuji;
mod raw_huff;
mod raw_maker;
mod xcf;

/// A pixport.PixportError carrying msg.
pub fn damaged(py: Python<'_>, msg: &str) -> PyErr {
    match py.import("pixport.errors").and_then(|m| m.getattr("PixportError")) {
        Ok(cls) => PyErr::from_value(cls.call1((msg,)).unwrap_or_else(|e| e.into_value(py).into_bound(py).into_any())),
        Err(e) => e,
    }
}

/// A decoding failure, carried out of code that runs without the GIL.
#[derive(Debug)]
pub struct Damage(pub String);

impl Damage {
    pub fn new(msg: impl Into<String>) -> Self {
        Damage(msg.into())
    }
}

pub type Res<T> = Result<T, Damage>;

pub fn raise(py: Python<'_>, d: Damage) -> PyErr {
    damaged(py, &d.0)
}

#[pymodule]
fn pixport_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    // registrations: one line per function, grouped by format (raw, exr, xcf, jxl)
    m.add_function(wrap_pyfunction!(png::png_paeth, m)?)?;
    m.add_function(wrap_pyfunction!(png::png_deflate, m)?)?;
    m.add_function(wrap_pyfunction!(raw_cr3::crx_band, m)?)?;
    m.add_function(wrap_pyfunction!(raw_cr3::crx_qp, m)?)?;
    m.add_function(wrap_pyfunction!(raw_fuji::fuji_strip, m)?)?;
    m.add_function(wrap_pyfunction!(raw_huff::ljpeg_decode, m)?)?;
    m.add_function(wrap_pyfunction!(raw_huff::nikon_decode, m)?)?;
    m.add_function(wrap_pyfunction!(raw_demosaic::demosaic_bayer, m)?)?;
    m.add_function(wrap_pyfunction!(raw_demosaic::demosaic_any, m)?)?;
    m.add_function(wrap_pyfunction!(raw_maker::olympus_decode, m)?)?;
    m.add_function(wrap_pyfunction!(raw_maker::sony_arw2, m)?)?;
    m.add_function(wrap_pyfunction!(raw_maker::pentax_decode, m)?)?;
    m.add_function(wrap_pyfunction!(raw_maker::panasonic_v4, m)?)?;
    m.add_function(wrap_pyfunction!(exr::exr_rle, m)?)?;
    m.add_function(wrap_pyfunction!(exr::exr_huf, m)?)?;
    m.add_function(wrap_pyfunction!(exr::exr_piz, m)?)?;
    m.add_function(wrap_pyfunction!(exr::exr_b44, m)?)?;
    m.add_function(wrap_pyfunction!(exr::exr_dwa_ac, m)?)?;
    m.add_function(wrap_pyfunction!(exr::exr_piz_encode, m)?)?;
    m.add_function(wrap_pyfunction!(hdr::hdr_decode, m)?)?;
    m.add_function(wrap_pyfunction!(hdr::hdr_encode, m)?)?;
    m.add_function(wrap_pyfunction!(hdr::hdr_rgbe, m)?)?;
    m.add_function(wrap_pyfunction!(xcf::xcf_rle_level, m)?)?;
    m.add_function(wrap_pyfunction!(jxl::jxl_decode, m)?)?;
    m.add_function(wrap_pyfunction!(jxl::jxl_to_jpeg, m)?)?;
    m.add_function(wrap_pyfunction!(jxl::jxl_encode, m)?)?;
    m.add_function(wrap_pyfunction!(jxl::jxl_from_jpeg, m)?)?;
    m.add_function(wrap_pyfunction!(jxl::jxl_version, m)?)?;
    Ok(())
}
