//! Makers' own RAW codings, value for value the same as the Python: Olympus (pixport/raw/olympus.py
//! `decode_compressed`), Sony's compressed ARW (sony.py `arw2`), Panasonic (panasonic.py) and
//! Pentax (pentax.py, through the shared Huffman reader in raw_huff.rs).

use crate::raw_huff::Bits;
use crate::{raise, Damage, Res};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;

fn olympus(data: &[u8], off: usize, width: usize, height: usize) -> Res<Vec<u16>> {
    let start = (off + 7).min(data.len());
    let src = &data[start..];
    let total_bits = src.len() as u64 * 8 + 16 * 8;
    let mut br = Bits::new(src);
    let mut out = vec![0u16; width * height];
    for row in 0..height {
        let mut acarry = [[0i64; 3]; 2];
        for col in 0..width {
            let carry = &mut acarry[col & 1];
            let i: i64 = if carry[2] < 3 { 2 } else { 0 };
            let mut nbits: i64 = 2 + i;
            let c0 = carry[0] & 0xFFFF;
            while nbits + i < 64 && (c0 >> (nbits + i)) != 0 {
                nbits += 1;
            }
            let b3 = br.get(3) as i64;
            let low = b3 & 3;
            let sign: i64 = if b3 & 4 != 0 { -1 } else { 0 };
            let top12 = br.peek16() >> 4;
            let high: i64 = if top12 == 0 {
                br.skip(12);
                let k = 16 - nbits;
                if k > 0 {
                    (br.get(k as u32) as i64) >> 1
                } else {
                    0
                }
            } else {
                let h = 12 - (usize::BITS - top12.leading_zeros()) as i64;
                br.skip((h + 1) as u32);
                h
            };
            let lowbits = if nbits > 0 { br.get(nbits as u32) as i64 } else { 0 };
            carry[0] = (high << nbits) | lowbits;
            let diff = (carry[0] ^ sign) + carry[1];
            carry[1] = (diff * 3 + carry[1]) >> 5;
            carry[2] = if carry[0] > 16 { 0 } else { carry[2] + 1 };
            let pred: i64 = if row < 2 && col < 2 {
                0
            } else if row < 2 {
                out[row * width + col - 2] as i64
            } else if col < 2 {
                out[(row - 2) * width + col] as i64
            } else {
                let w = out[row * width + col - 2] as i64;
                let n = out[(row - 2) * width + col] as i64;
                let nw = out[(row - 2) * width + col - 2] as i64;
                if (w < nw && nw < n) || (n < nw && nw < w) {
                    if (w - nw).abs() > 32 || (n - nw).abs() > 32 {
                        w + n - nw
                    } else {
                        (w + n) >> 1
                    }
                } else if (w - nw).abs() > (n - nw).abs() {
                    w
                } else {
                    n
                }
            };
            out[row * width + col] = ((pred + ((diff << 2) | low)) & 0xFFFF) as u16;
        }
        if br.used > total_bits + 64 {
            return Err(Damage::new("The ORF data ends early"));
        }
    }
    Ok(out)
}

/// An Olympus-coded ORF's photosites (height * width uint16).
#[pyfunction]
pub fn olympus_decode<'py>(py: Python<'py>, data: PyBackedBytes, off: usize, width: usize, height: usize) -> PyResult<Bound<'py, PyArray1<u16>>> {
    if width.saturating_mul(height) > 1 << 30 {
        return Err(raise(py, Damage::new("The ORF size isn't valid")));
    }
    let out = py.detach(|| olympus(&data, off, width, height)).map_err(|e| raise(py, e))?;
    Ok(out.into_pyarray(py))
}

fn arw2(data: &[u8], off: usize, width: usize, height: usize) -> Vec<u16> {
    let mut out = vec![0u16; width * height];
    let nblk = if width > 30 { (width / 16).min(2 * (width - 30).div_ceil(32)) } else { 0 };
    for row in 0..height {
        let base_row = off + row * width;
        for j in 0..nblk {
            let b = &data[base_row + j * 16..base_row + j * 16 + 16];
            let line = &data[base_row..base_row + width];
            let at = |k: usize| -> u32 { *line.get(j * 16 + k).unwrap_or(&0) as u32 };
            let head = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            let mx = (head & 0x7FF) as i64;
            let mn = ((head >> 11) & 0x7FF) as i64;
            let imax = ((head >> 22) & 0xF) as usize;
            let imin = ((head >> 26) & 0xF) as usize;
            let mut sh = 0;
            while sh < 4 && (0x80i64 << sh) <= mx - mn {
                sh += 1;
            }
            let mut bit = 30usize;
            let base_col = (j / 2) * 32 + (j & 1);
            for i in 0..16 {
                let v = if i == imax {
                    mx
                } else if i == imin {
                    mn
                } else {
                    let byte = bit >> 3;
                    let word = at(byte) | at(byte + 1) << 8;
                    let d = ((word >> (bit & 7)) & 0x7F) as i64;
                    bit += 7;
                    ((d << sh) + mn).min(0x7FF)
                };
                let col = base_col + 2 * i;
                if col < width {
                    out[row * width + col] = v as u16;
                }
            }
        }
    }
    out
}

/// Sony's compressed ARW before the curve (height * width uint16, 11-bit values).
#[pyfunction]
pub fn sony_arw2<'py>(py: Python<'py>, data: PyBackedBytes, off: usize, width: usize, height: usize) -> PyResult<Bound<'py, PyArray1<u16>>> {
    if width.saturating_mul(height) > 1 << 30 || off.saturating_add(width * height) > data.len() {
        return Err(raise(py, Damage::new("The ARW data runs past the end of the file")));
    }
    let out = py.detach(|| arw2(&data, off, width, height));
    Ok(out.into_pyarray(py))
}

/// Pentax's Huffman coding (pentax.py `_decode`): the standard JPEG difference coding, MSB-first
/// without stuffing, two horizontal and four vertical predictors.
fn pentax(data: &[u8], off: usize, width: usize, height: usize, lut: &[u16]) -> Res<Vec<u16>> {
    let src = if off < data.len() { &data[off..] } else { &data[0..0] };
    let mut br = Bits::new(src);
    let mut out = vec![0u16; width * height];
    let mut vp = [0i64; 4];
    for row in 0..height {
        let mut hp = [0i64; 2];
        for col in 0..width {
            let e = lut[br.peek16()];
            if e == 0 {
                return Err(Damage::new("The PEF data has a bad Huffman code"));
            }
            br.skip((e >> 8) as u32);
            let s = (e & 0xFF) as u32;
            let diff: i64 = if s == 0 {
                0
            } else if s >= 16 {
                -32768
            } else {
                let v = br.get(s) as i64;
                if v >> (s - 1) != 0 {
                    v
                } else {
                    v - (1 << s) + 1
                }
            };
            if col < 2 {
                let k = (row & 1) * 2 + col;
                vp[k] += diff;
                hp[col] = vp[k];
            } else {
                hp[col & 1] += diff;
            }
            out[row * width + col] = (hp[col & 1] & 0xFFFF) as u16;
        }
    }
    Ok(out)
}

/// A Huffman-coded PEF's photosites (height * width uint16).
#[pyfunction]
pub fn pentax_decode<'py>(py: Python<'py>, data: PyBackedBytes, off: usize, width: usize, height: usize,
                          lut: PyReadonlyArray1<'py, u16>) -> PyResult<Bound<'py, PyArray1<u16>>> {
    let lut = lut.as_slice()?.to_vec();
    if lut.len() != 65536 || width.saturating_mul(height) > 1 << 30 {
        return Err(raise(py, Damage::new("Bad PEF decoder tables")));
    }
    let out = py.detach(|| pentax(&data, off, width, height, &lut)).map_err(|e| raise(py, e))?;
    Ok(out.into_pyarray(py))
}

/// Panasonic raw format 4 (panasonic.py `_v4`): the bit reader works on 0x4000-byte blocks,
/// rotated by 0x2008, from the top bit down.
fn panasonic4(data: &[u8], off: usize, width: usize, height: usize) -> Vec<u16> {
    let mut buf = vec![0u8; 0x4002];
    let mut vbits: u32 = 0;
    let mut pos = off;
    let mut out = vec![0u16; width * height];
    let mut bits = |nb: u32, buf: &mut Vec<u8>| -> i64 {
        if vbits == 0 {
            for k in 0..0x4000usize {
                let src = pos + k;
                let b = if src < data.len() { data[src] } else { 0 };
                let dst = if k < 0x1FF8 { 0x2008 + k } else { k - 0x1FF8 };
                buf[dst] = b;
            }
            pos += 0x4000;
        }
        vbits = vbits.wrapping_sub(nb) & 0x1FFFF;
        let byte = ((vbits >> 3) ^ 0x3FF0) as usize;
        let w = buf[byte] as u32 | (buf[byte + 1] as u32) << 8;
        ((w >> (vbits & 7)) & ((1u32 << nb) - 1)) as i64
    };
    let mut sh: i64 = 0;
    let mut k = 0usize;
    for _row in 0..height {
        let mut pred = [0i64; 2];
        let mut nonz = [0i64; 2];
        for col in 0..width {
            let i = col % 14;
            if i == 0 {
                pred = [0, 0];
                nonz = [0, 0];
            }
            if i % 3 == 2 {
                sh = 4 >> (3 - bits(2, &mut buf));
            }
            let p = i & 1;
            if nonz[p] != 0 {
                let j = bits(8, &mut buf);
                if j != 0 {
                    pred[p] -= 0x80 << sh;
                    if pred[p] < 0 || sh == 4 {
                        pred[p] &= (1 << sh) - 1;
                    }
                    pred[p] += j << sh;
                }
            } else {
                nonz[p] = bits(8, &mut buf);
                if nonz[p] != 0 || i > 11 {
                    pred[p] = (nonz[p] << 4) | bits(4, &mut buf);
                }
            }
            out[k] = (pred[col & 1] & 0xFFFF) as u16;
            k += 1;
        }
    }
    out
}

/// A format-4 RW2's photosites (height * width uint16).
#[pyfunction]
pub fn panasonic_v4<'py>(py: Python<'py>, data: PyBackedBytes, off: usize, width: usize, height: usize) -> PyResult<Bound<'py, PyArray1<u16>>> {
    if width.saturating_mul(height) > 1 << 30 || off >= data.len() {
        return Err(raise(py, Damage::new("The RW2 data is outside the file")));
    }
    let out = py.detach(|| panasonic4(&data, off, width, height));
    Ok(out.into_pyarray(py))
}
