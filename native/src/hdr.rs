//! Radiance HDR scanlines, as hdr.py reads and writes them: flat or old-style run-length RGBE
//! pixels, and the newer form with each component run-length coded on its own.

use crate::{raise, Damage, Res};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::PyBytes;

fn decode(data: &[u8], mut pos: usize, lines: usize, width: usize) -> Res<Vec<u8>> {
    let n = data.len();
    let mut out = vec![0u8; lines * width * 4];
    let mut comp = vec![0u8; width];
    for y in 0..lines {
        let row = &mut out[y * width * 4..(y + 1) * width * 4];
        if (8..=0x7FFF).contains(&width) && pos + 4 <= n && data[pos] == 2 && data[pos + 1] == 2 && data[pos + 2] < 128 {
            if ((data[pos + 2] as usize) << 8 | data[pos + 3] as usize) != width {
                return Err(Damage::new("The HDR pixel data is damaged (a scanline of the wrong length)"));
            }
            pos += 4;
            for c in 0..4 {
                let mut x = 0usize;
                while x < width {
                    if pos >= n {
                        return Err(Damage::new("The HDR pixel data is cut short"));
                    }
                    let k = data[pos] as usize;
                    if k > 128 {
                        let k = k - 128;
                        if k > width - x || pos + 1 >= n {
                            return Err(Damage::new("The HDR pixel data is damaged (a run overflows its scanline)"));
                        }
                        comp[x..x + k].fill(data[pos + 1]);
                        pos += 2;
                        x += k;
                    } else {
                        if k == 0 || k > width - x || pos + 1 + k > n {
                            return Err(Damage::new("The HDR pixel data is damaged (a bad literal run)"));
                        }
                        comp[x..x + k].copy_from_slice(&data[pos + 1..pos + 1 + k]);
                        pos += 1 + k;
                        x += k;
                    }
                }
                for (x, &v) in comp.iter().enumerate() {
                    row[x * 4 + c] = v;
                }
            }
        } else {
            let mut x = 0usize;
            let mut shift = 0u32;
            while x < width {
                if pos + 4 > n {
                    return Err(Damage::new("The HDR pixel data is cut short"));
                }
                let px = [data[pos], data[pos + 1], data[pos + 2], data[pos + 3]];
                pos += 4;
                if px[0] == 1 && px[1] == 1 && px[2] == 1 {
                    if x == 0 {
                        return Err(Damage::new("The HDR pixel data is damaged (a repeat with nothing before it)"));
                    }
                    let run = (px[3] as usize) << shift;
                    if x + run > width {
                        return Err(Damage::new("The HDR pixel data is damaged (a repeat overflows its scanline)"));
                    }
                    let prev = [row[x * 4 - 4], row[x * 4 - 3], row[x * 4 - 2], row[x * 4 - 1]];
                    for k in 0..run {
                        row[(x + k) * 4..(x + k) * 4 + 4].copy_from_slice(&prev);
                    }
                    x += run;
                    shift += 8;
                    if shift > 24 {
                        return Err(Damage::new("The HDR pixel data is damaged (a repeat that never ends)"));
                    }
                } else {
                    row[x * 4..x * 4 + 4].copy_from_slice(&px);
                    x += 1;
                    shift = 0;
                }
            }
        }
    }
    Ok(out)
}

#[pyfunction]
pub fn hdr_decode<'py>(py: Python<'py>, data: PyBackedBytes, pos: usize, lines: usize, width: usize) -> PyResult<Bound<'py, PyBytes>> {
    if lines.saturating_mul(width) > 1 << 30 {
        return Err(raise(py, Damage::new("The HDR picture is too large")));
    }
    let out = py.detach(|| decode(&data, pos, lines, width)).map_err(|e| raise(py, e))?;
    Ok(PyBytes::new(py, &out))
}

/// One component of a scanline: runs of 3 or more as runs, the rest as literals.
fn encode_component(row: &[u8], out: &mut Vec<u8>) {
    let n = row.len();
    let mut lit = 0usize;
    let mut a = 0usize;
    while a < n {
        let mut b = a + 1;
        while b < n && row[b] == row[a] {
            b += 1;
        }
        if b - a >= 3 {
            while lit < a {
                let k = (a - lit).min(128);
                out.push(k as u8);
                out.extend_from_slice(&row[lit..lit + k]);
                lit += k;
            }
            let mut s = a;
            while b - s >= 3 {
                let k = (b - s).min(127);
                out.push((128 + k) as u8);
                out.push(row[a]);
                s += k;
            }
            lit = s;
        }
        a = b;
    }
    while lit < n {
        let k = (n - lit).min(128);
        out.push(k as u8);
        out.extend_from_slice(&row[lit..lit + k]);
        lit += k;
    }
}

#[pyfunction]
pub fn hdr_encode<'py>(py: Python<'py>, rgbe: PyBackedBytes, h: usize, w: usize) -> PyResult<Bound<'py, PyBytes>> {
    if h.saturating_mul(w).saturating_mul(4) != rgbe.len() {
        return Err(raise(py, Damage::new("The HDR pixel buffer has the wrong size")));
    }
    let out = py.detach(|| {
        if !(8..=0x7FFF).contains(&w) {
            return rgbe.to_vec();
        }
        let mut out = Vec::with_capacity(rgbe.len());
        let mut comp = vec![0u8; w];
        for y in 0..h {
            out.extend_from_slice(&[2, 2, (w >> 8) as u8, (w & 255) as u8]);
            let row = &rgbe[y * w * 4..(y + 1) * w * 4];
            for c in 0..4 {
                for (x, v) in comp.iter_mut().enumerate() {
                    *v = row[x * 4 + c];
                }
                encode_component(&comp, &mut out);
            }
        }
        out
    });
    Ok(PyBytes::new(py, &out))
}

/// RGBE from float RGB, as hdr.py's float_to_rgbe: NaN and negatives black, infinity 1e30, the
/// shared exponent from the largest channel, mantissas by an exact power-of-two scaling.
fn to_rgbe(rgb: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; rgb.len() / 3 * 4];
    for (px, o) in rgb.as_chunks::<3>().0.iter().zip(out.as_chunks_mut::<4>().0.iter_mut()) {
        let c = px.map(|v| if v.is_nan() { 0.0 } else if v == f32::INFINITY { 1e30 } else { v.clamp(0.0, 1e38) });
        let v = c[0].max(c[1]).max(c[2]);
        if v < 1e-32 {
            continue;
        }
        let e = (((v.to_bits() >> 23) & 0xFF) as i32 - 126).clamp(-127, 127);
        let scale = f32::from_bits(((8 - e + 127) as u32) << 23);
        for (dst, &src) in o.iter_mut().zip(c.iter()) {
            *dst = (src * scale).floor().min(255.0) as u8;
        }
        o[3] = (e + 128) as u8;
    }
    out
}

#[pyfunction]
pub fn hdr_rgbe<'py>(py: Python<'py>, rgb: PyBackedBytes) -> PyResult<Bound<'py, PyBytes>> {
    if !rgb.len().is_multiple_of(12) {
        return Err(raise(py, Damage::new("The HDR pixel buffer has the wrong size")));
    }
    let out = py.detach(|| {
        let v: Vec<f32> = rgb.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
        to_rgbe(&v)
    });
    Ok(PyBytes::new(py, &out))
}
