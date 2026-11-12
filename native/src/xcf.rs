//! GIMP XCF's run-length tiles, as xcf.py decodes them: each 64 x 64 tile holds one stream per
//! byte of the pixel; every stream is a series of short or long runs and literals.

use crate::{raise, Damage, Res};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::PyBytes;

/// One tile's streams, written pixel-interleaved into out (npix * bpp bytes). Returns where the
/// tile's data ended.
fn tile(d: &[u8], mut pos: usize, end: usize, bpp: usize, npix: usize, out: &mut [u8]) -> Res<usize> {
    let end = end.min(d.len());
    for b in 0..bpp {
        let mut x = 0usize;
        while x < npix {
            if pos >= end {
                return Err(Damage::new("The XCF tile data is cut short"));
            }
            let op = d[pos] as usize;
            if op < 128 {
                let (k, v, step) = if op < 127 {
                    if pos + 2 > end {
                        return Err(Damage::new("The XCF tile data is cut short"));
                    }
                    (op + 1, d[pos + 1], 2)
                } else {
                    if pos + 4 > end {
                        return Err(Damage::new("The XCF tile data is cut short"));
                    }
                    ((d[pos + 1] as usize) << 8 | d[pos + 2] as usize, d[pos + 3], 4)
                };
                if k > npix - x {
                    return Err(Damage::new("The XCF tile data is damaged (a run overflows its tile)"));
                }
                for i in 0..k {
                    out[(x + i) * bpp + b] = v;
                }
                x += k;
                pos += step;
            } else {
                let (k, start) = if op == 128 {
                    if pos + 3 > end {
                        return Err(Damage::new("The XCF tile data is cut short"));
                    }
                    ((d[pos + 1] as usize) << 8 | d[pos + 2] as usize, pos + 3)
                } else {
                    (256 - op, pos + 1)
                };
                if k > npix - x || start + k > end {
                    return Err(Damage::new("The XCF tile data is damaged (a literal overflows its tile)"));
                }
                for i in 0..k {
                    out[(x + i) * bpp + b] = d[start + i];
                }
                x += k;
                pos = start + k;
            }
        }
    }
    Ok(pos)
}

/// A whole level of RLE tiles (64 x 64, row by row) as (h, w, bpp) bytes.
fn level(d: &[u8], ptrs: &[u64], w: usize, h: usize, bpp: usize) -> Res<Vec<u8>> {
    let ntx = w.div_ceil(64);
    let nty = h.div_ceil(64);
    let nt = ntx * nty;
    if ptrs.len() < nt {
        return Err(Damage::new("The XCF level has too few tiles"));
    }
    let mut out = vec![0u8; w * h * bpp];
    let mut buf = vec![0u8; 64 * 64 * bpp];
    for t in 0..nt {
        let (ty, tx) = (t / ntx, t % ntx);
        let (x0, y0) = (tx * 64, ty * 64);
        let (tw, th) = ((w - x0).min(64), (h - y0).min(64));
        let start = ptrs[t] as usize;
        let next = ptrs.get(t + 1).copied().unwrap_or(0) as usize;
        let end = if next > start && next <= d.len() { next } else { d.len() };
        if start == 0 || start >= d.len() {
            return Err(Damage::new("The XCF file is damaged (a tile pointer outside the file)"));
        }
        tile(d, start, end, bpp, tw * th, &mut buf)?;
        for r in 0..th {
            let dst = ((y0 + r) * w + x0) * bpp;
            out[dst..dst + tw * bpp].copy_from_slice(&buf[r * tw * bpp..(r + 1) * tw * bpp]);
        }
    }
    Ok(out)
}

#[pyfunction]
pub fn xcf_rle_level<'py>(py: Python<'py>, data: PyBackedBytes, ptrs: Vec<u64>, w: usize, h: usize, bpp: usize)
                          -> PyResult<Bound<'py, PyBytes>> {
    if bpp == 0 || bpp > 64 || w.saturating_mul(h).saturating_mul(bpp) > 1 << 33 {
        return Err(raise(py, Damage::new("The XCF level size isn't valid")));
    }
    let out = py.detach(|| level(&data, &ptrs, w, h, bpp)).map_err(|e| raise(py, e))?;
    Ok(PyBytes::new(py, &out))
}
