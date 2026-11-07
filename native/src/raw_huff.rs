//! Huffman-coded RAW data: lossless JPEG (pixport/raw/ljpeg.py `segments`, `_diffs`, `predict`)
//! and Nikon's NEF coding (pixport/raw/nikon.py `_symbols`, `_predict`), value for value the same
//! as the Python. Lookup tables come from Python (`build_lut`): 65536 entries, length << 8 | symbol.

use crate::{raise, Damage, Res};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;

/// MSB-first reader over a buffer; past the end it reads zeros.
pub struct Bits<'a> {
    d: &'a [u8],
    pos: usize,
    acc: u64,
    n: u32,
    /// bits consumed so far
    pub used: u64,
}

impl<'a> Bits<'a> {
    pub fn new(d: &'a [u8]) -> Self {
        Bits { d, pos: 0, acc: 0, n: 0, used: 0 }
    }

    #[inline(always)]
    fn fill(&mut self) {
        while self.n <= 56 {
            let b = if self.pos < self.d.len() { self.d[self.pos] } else { 0 };
            self.acc |= (b as u64) << (56 - self.n);
            self.n += 8;
            self.pos += 1;
        }
    }

    #[inline(always)]
    pub fn peek16(&mut self) -> usize {
        if self.n < 32 {
            self.fill();
        }
        (self.acc >> 48) as usize
    }

    #[inline(always)]
    pub fn skip(&mut self, k: u32) {
        if k == 0 {
            return;
        }
        if self.n < k {
            self.fill();
        }
        self.acc <<= k;
        self.n -= k;
        self.used += k as u64;
    }

    #[inline(always)]
    pub fn get(&mut self, k: u32) -> u32 {
        if k == 0 {
            return 0;
        }
        if self.n < k {
            self.fill();
        }
        let v = (self.acc >> (64 - k)) as u32;
        self.acc <<= k;
        self.n -= k;
        self.used += k as u64;
        v
    }
}

/// The entropy-coded data from `start`, split at restart markers, stuffing removed; it ends at the
/// first other marker (ljpeg.segments).
fn segments(data: &[u8], start: usize, end: usize) -> Vec<Vec<u8>> {
    let end = end.min(data.len());
    let mut out = Vec::new();
    let mut cur = Vec::new();
    let mut p = start.min(end);
    while p < end {
        let b = data[p];
        if b != 0xFF {
            cur.push(b);
            p += 1;
            continue;
        }
        if p + 1 >= end {
            cur.push(b);
            break;
        }
        let m = data[p + 1];
        if m == 0x00 {
            cur.push(0xFF);
            p += 2;
        } else if m == 0xFF {
            cur.push(0xFF);
            p += 1;
        } else if (0xD0..=0xD7).contains(&m) {
            out.push(std::mem::take(&mut cur));
            p += 2;
        } else {
            break;
        }
    }
    out.push(cur);
    out
}

fn diffs(seg: &[u8], n: usize, luts: &[&[u16]], slot_tab: &[u8], out: &mut Vec<i32>) -> Res<()> {
    let nbits = seg.len() as u64 * 8;
    let mut br = Bits::new(seg);
    let stride = slot_tab.len();
    for i in 0..n {
        let lut = luts[slot_tab[i % stride] as usize];
        let e = lut[br.peek16()];
        if e == 0 {
            return Err(Damage::new(if br.used >= nbits {
                "The lossless JPEG data ends early"
            } else {
                "The lossless JPEG data has a bad Huffman code"
            }));
        }
        let ln = (e >> 8) as u32;
        let s = (e & 0xFF) as u32;
        br.skip(ln);
        let v = if s == 0 {
            0
        } else if s >= 16 {
            32768
        } else {
            let v = br.get(s) as i32;
            if v >> (s - 1) != 0 {
                v
            } else {
                v - (1 << s) + 1
            }
        };
        out.push(v);
    }
    if br.used > nbits + 64 {
        return Err(Damage::new("The lossless JPEG data ends early"));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn predict(d: &[i32], rows: usize, width: usize, stride: usize, predictor: u8, precision: u32, pt: u32, restart_rows: usize,
           chain: usize, chain_rows: usize) -> Res<Vec<u16>> {
    if pt + 1 > precision {
        return Err(Damage::new("The lossless JPEG point transform is larger than its precision"));
    }
    let init = 1i64 << (precision - pt - 1);
    let mut out = vec![0u16; rows * width];
    if chain > 1 {
        // Canon sRAW: the luma slots are one sequence along the row; chroma slots as usual
        if predictor != 1 || restart_rows > 0 {
            return Err(Damage::new("This Canon sRAW coding isn't supported"));
        }
        if chain_rows == 0 || !chain.is_multiple_of(chain_rows) {
            return Err(Damage::new("This lossless JPEG luma layout isn't supported"));
        }
        let per = chain / chain_rows;
        let mut first = vec![init; stride];
        for r in 0..rows {
            let mut cur = vec![0i64; stride];
            let mut seen = vec![false; stride];
            for j in 0..width {
                let s = j % stride;
                let g = if s < chain { (s / per) * per } else { s };
                let dv = d[r * width + j] as i64;
                let v = if seen[g] {
                    cur[g] + dv
                } else {
                    seen[g] = true;
                    if s < chain && g > 0 {
                        first[0] + dv
                    } else {
                        first[g] += dv;
                        first[g]
                    }
                };
                cur[g] = v;
                out[r * width + j] = (v & 0xFFFF) as u16;
            }
        }
        if pt > 0 {
            for v in out.iter_mut() {
                *v = ((*v as u32) << pt & 0xFFFF) as u16;
            }
        }
        return Ok(out);
    }
    let period = if restart_rows > 0 { restart_rows } else { rows };
    for r in 0..rows {
        let first = r % period == 0;
        for j in 0..width {
            let dv = d[r * width + j] as i64;
            let p: i64 = if first {
                if j < stride {
                    init
                } else {
                    out[r * width + j - stride] as i64
                }
            } else if j < stride {
                out[(r - 1) * width + j] as i64
            } else {
                let a = out[r * width + j - stride] as i64;
                let b = out[(r - 1) * width + j] as i64;
                let c = out[(r - 1) * width + j - stride] as i64;
                match predictor {
                    1 => a,
                    2 => b,
                    3 => c,
                    4 => a + b - c,
                    5 => a + ((b - c) >> 1),
                    6 => b + ((a - c) >> 1),
                    _ => (a + b) >> 1,
                }
            };
            out[r * width + j] = ((p + dv) & 0xFFFF) as u16;
        }
    }
    if pt > 0 {
        for v in out.iter_mut() {
            *v = ((*v as u32) << pt & 0xFFFF) as u16;
        }
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn ljpeg(data: &[u8], start: usize, end: usize, tabs: &[u16], slot_tab: &[u8], rows: usize, width: usize, predictor: u8,
         precision: u32, pt: u32, restart: usize, chain: usize, chain_rows: usize) -> Res<Vec<u16>> {
    let ntab = tabs.len() / 65536;
    if slot_tab.is_empty() || slot_tab.iter().any(|&t| t as usize >= ntab) || !width.is_multiple_of(slot_tab.len()) {
        return Err(Damage::new("Bad lossless JPEG tables"));
    }
    let luts: Vec<&[u16]> = (0..ntab).map(|i| &tabs[i * 65536..(i + 1) * 65536]).collect();
    let stride = slot_tab.len();
    let segs = segments(data, start, end);
    let total = rows * width;
    let mut d: Vec<i32> = Vec::with_capacity(total);
    let mut restart_rows = 0;
    if restart > 0 && segs.len() > 1 {
        let per = restart * stride;
        if !per.is_multiple_of(width) {
            return Err(Damage::new("Lossless JPEG restart intervals that split a row aren't supported"));
        }
        let mut left = total;
        for s in &segs {
            if left == 0 {
                break;
            }
            let k = per.min(left);
            diffs(s, k, &luts, slot_tab, &mut d)?;
            left -= k;
        }
        if left > 0 {
            return Err(Damage::new("The lossless JPEG data ends early"));
        }
        restart_rows = per / width;
    } else {
        diffs(&segs[0], total, &luts, slot_tab, &mut d)?;
    }
    predict(&d, rows, width, stride, predictor, precision, pt, restart_rows, chain, chain_rows)
}

/// Lossless JPEG samples (rows * width, uint16), as ljpeg.decode.
#[pyfunction]
#[pyo3(signature = (data, start, end, tabs, slot_tab, rows, width, predictor, precision, pt, restart, chain=0, chain_rows=1))]
#[allow(clippy::too_many_arguments)]
pub fn ljpeg_decode<'py>(py: Python<'py>, data: PyBackedBytes, start: usize, end: usize, tabs: PyReadonlyArray2<'py, u16>,
                         slot_tab: PyReadonlyArray1<'py, u8>, rows: usize, width: usize, predictor: u8, precision: u32,
                         pt: u32, restart: usize, chain: usize, chain_rows: usize) -> PyResult<Bound<'py, PyArray1<u16>>> {
    let tabs = tabs.as_slice()?.to_vec();
    let slots = slot_tab.as_slice()?.to_vec();
    if rows.saturating_mul(width) > 1 << 30 {
        return Err(raise(py, Damage::new("The lossless JPEG frame size isn't valid")));
    }
    let out = py
        .detach(|| ljpeg(&data, start, end, &tabs, &slots, rows, width, predictor, precision, pt, restart, chain, chain_rows))
        .map_err(|e| raise(py, e))?;
    Ok(out.into_pyarray(py))
}

#[allow(clippy::too_many_arguments)]
fn nikon(data: &[u8], off: usize, width: usize, height: usize, lut1: &[u16], lut2: &[u16], split: usize, vpred: &[i64]) -> Res<Vec<u16>> {
    let src = if off < data.len() { &data[off..] } else { &data[0..0] };
    let mut br = Bits::new(src);
    let mut out = vec![0u16; width * height];
    let mut vp = [vpred[0], vpred[1], vpred[2], vpred[3]];
    let mut lut = lut1;
    for row in 0..height {
        if split > 0 && split < height && row == split {
            lut = lut2;
        }
        let mut hp = [0i64; 2];
        for col in 0..width {
            let e = lut[br.peek16()];
            if e == 0 {
                return Err(Damage::new("The NEF data has a bad Huffman code"));
            }
            br.skip((e >> 8) as u32);
            let sym = (e & 0xFF) as i64;
            let (ln, shl) = (sym & 15, sym >> 4);
            let nb = ln - shl;
            let v = if nb > 0 { br.get(nb as u32) as i64 } else { 0 };
            let mut diff = (((v << 1) + 1) << shl) >> 1;
            if ln == 0 {
                diff = 0;
            } else if diff & (1 << (ln - 1)) == 0 {
                diff -= (1 << ln) - if shl != 0 { 0 } else { 1 };
            }
            if col < 2 {
                let k = (row & 1) * 2 + col;
                vp[k] += diff;
                hp[col] = vp[k];
            } else {
                hp[col & 1] += diff;
            }
            let wrapped = ((hp[col & 1] + 0x8000) & 0xFFFF) - 0x8000;
            out[row * width + col] = wrapped.clamp(0, 0x3FFF) as u16;
        }
    }
    Ok(out)
}

/// A Huffman-coded NEF's values before the curve (height * width uint16), as nikon._symbols and
/// nikon._predict.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn nikon_decode<'py>(py: Python<'py>, data: PyBackedBytes, off: usize, width: usize, height: usize,
                         lut1: PyReadonlyArray1<'py, u16>, lut2: PyReadonlyArray1<'py, u16>, split: usize,
                         vpred: PyReadonlyArray1<'py, i64>) -> PyResult<Bound<'py, PyArray1<u16>>> {
    let (l1, l2, vp) = (lut1.as_slice()?.to_vec(), lut2.as_slice()?.to_vec(), vpred.as_slice()?.to_vec());
    if l1.len() != 65536 || l2.len() != 65536 || vp.len() != 4 || width.saturating_mul(height) > 1 << 30 {
        return Err(raise(py, Damage::new("Bad NEF decoder tables")));
    }
    let out = py.detach(|| nikon(&data, off, width, height, &l1, &l2, split, &vp)).map_err(|e| raise(py, e))?;
    Ok(out.into_pyarray(py))
}
