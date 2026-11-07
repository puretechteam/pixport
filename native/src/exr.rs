//! OpenEXR's bit-serial parts, as exr.py decodes them: the run-length code (RLE compression and
//! DWA's alpha), the Huffman code (PIZ and DWA's AC values), PIZ as a whole (bitmap, Huffman,
//! wavelet, look-up table), B44/B44A blocks and DWA's run-length AC layout.

use crate::{raise, Damage, Res};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::PyBytes;

fn rle(data: &[u8], size: usize) -> Res<Vec<u8>> {
    let mut out = Vec::with_capacity(size);
    let mut pos = 0usize;
    let n = data.len();
    while pos < n && out.len() < size {
        let c = data[pos];
        if c >= 128 {
            let k = 256 - c as usize;
            if pos + 1 + k > n {
                return Err(Damage::new("The EXR run-length data is cut short"));
            }
            out.extend_from_slice(&data[pos + 1..pos + 1 + k]);
            pos += 1 + k;
        } else {
            let k = c as usize + 1;
            if pos + 1 >= n {
                return Err(Damage::new("The EXR run-length data is cut short"));
            }
            let v = data[pos + 1];
            out.extend(std::iter::repeat_n(v, k));
            pos += 2;
        }
    }
    if out.len() != size {
        return Err(Damage::new("The EXR run-length data is damaged (wrong size)"));
    }
    Ok(out)
}

#[pyfunction]
pub fn exr_rle<'py>(py: Python<'py>, data: PyBackedBytes, size: usize) -> PyResult<Bound<'py, PyBytes>> {
    if size > 1 << 31 {
        return Err(raise(py, Damage::new("The EXR block is too large")));
    }
    let out = py.detach(|| rle(&data, size)).map_err(|e| raise(py, e))?;
    Ok(PyBytes::new(py, &out))
}

// ---------------------------------------------------------------- Huffman

struct BitsIn<'a> {
    d: &'a [u8],
    pos: usize,
    acc: u64,
    have: u32,
}

impl<'a> BitsIn<'a> {
    fn new(d: &'a [u8]) -> Self {
        BitsIn { d, pos: 0, acc: 0, have: 0 }
    }
    fn take(&mut self, n: u32) -> Res<u64> {
        while self.have < n {
            if self.pos >= self.d.len() {
                return Err(Damage::new("The EXR Huffman table is cut short"));
            }
            self.acc = (self.acc << 8) | self.d[self.pos] as u64;
            self.pos += 1;
            self.have += 8;
        }
        self.have -= n;
        Ok((self.acc >> self.have) & ((1u64 << n) - 1))
    }
}

const LUT_BITS: u32 = 14;

fn huf(data: &[u8], nraw: usize) -> Res<Vec<u16>> {
    if nraw == 0 {
        return Ok(Vec::new());
    }
    if data.len() < 20 {
        return Err(Damage::new("The EXR Huffman data is cut short"));
    }
    let rd = |i: usize| u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]) as usize;
    let (im, i_m, nbits) = (rd(0), rd(4), rd(12));
    if im >= 65537 || i_m >= 65537 || im > i_m {
        return Err(Damage::new("The EXR Huffman data is damaged (symbol range)"));
    }
    // the code lengths
    let mut lengths = vec![0u8; i_m + 1];
    let mut br = BitsIn::new(&data[20..]);
    let mut i = im;
    while i <= i_m {
        let l = br.take(6)? as usize;
        if l == 63 {
            let run = br.take(8)? as usize + 6;
            if i + run > i_m + 1 {
                return Err(Damage::new("The EXR Huffman table is damaged"));
            }
            i += run;
        } else if l >= 59 {
            let run = l - 59 + 2;
            if i + run > i_m + 1 {
                return Err(Damage::new("The EXR Huffman table is damaged"));
            }
            i += run;
        } else {
            lengths[i] = l as u8;
            i += 1;
        }
    }
    let start = 20 + br.pos;
    if nbits > 8 * (data.len() - start) {
        return Err(Damage::new("The EXR Huffman data is cut short"));
    }
    let rlc = i_m;
    // canonical codes: longer codes take the smaller values
    let mut counts = [0u64; 59];
    for &l in &lengths {
        counts[l as usize] += 1;
    }
    let mut first = [0u64; 59];
    let mut c = 0u64;
    for l in (1..59).rev() {
        first[l] = c;
        c = (c + counts[l]) >> 1;
    }
    let mut next = first;
    let maxl = lengths.iter().copied().max().unwrap_or(0) as u32;
    if maxl == 0 {
        return Err(Damage::new("The EXR Huffman table is empty"));
    }
    // intervals of maxl-bit windows, for codes longer than the table
    let mut ivals: Vec<(u64, u64, u32, u8)> = Vec::new(); // (lo, hi, symbol, length)
    let tb = LUT_BITS.min(maxl);
    let mut table = vec![(0u32, 0u8); 1usize << tb]; // (symbol, length); length 0: look in ivals
    for (sym, &l) in lengths.iter().enumerate() {
        if l == 0 {
            continue;
        }
        let l = l as u32;
        let code = next[l as usize];
        next[l as usize] += 1;
        if code >> l != 0 {
            return Err(Damage::new("The EXR Huffman table is damaged (codes overflow)"));
        }
        ivals.push((code << (maxl - l), (code + 1) << (maxl - l), sym as u32, l as u8));
        if l <= tb {
            let lo = (code << (tb - l)) as usize;
            let hi = ((code + 1) << (tb - l)) as usize;
            for e in &mut table[lo..hi] {
                if e.1 != 0 {
                    return Err(Damage::new("The EXR Huffman table is damaged (codes overlap)"));
                }
                *e = (sym as u32, l as u8);
            }
        }
    }
    ivals.sort_unstable();
    for w in ivals.windows(2) {
        if w[0].1 > w[1].0 {
            return Err(Damage::new("The EXR Huffman table is damaged (codes overlap)"));
        }
    }
    let src = &data[start..];
    let byte = |k: usize| -> u64 { *src.get(k).unwrap_or(&0) as u64 };
    // the window of maxl bits at bit position p (maxl <= 58)
    let window = |p: usize| -> u64 {
        let b = p >> 3;
        let mut w = 0u64;
        for k in 0..8 {
            w = (w << 8) | byte(b + k);
        }
        let s = (p & 7) as u32;
        if s > 0 {
            w = (w << s) | (byte(b + 8) >> (8 - s));
        }
        w >> (64 - maxl)
    };
    let mut out: Vec<u16> = Vec::with_capacity(nraw);
    let mut p = 0usize;
    while p < nbits {
        let w = window(p);
        let (sym, l) = {
            let e = table[(w >> (maxl - tb)) as usize];
            if e.1 != 0 {
                (e.0 as usize, e.1 as usize)
            } else {
                let k = ivals.partition_point(|iv| iv.0 <= w);
                if k == 0 || w >= ivals[k - 1].1 {
                    return Err(Damage::new("The EXR Huffman data is damaged (an unknown code)"));
                }
                (ivals[k - 1].2 as usize, ivals[k - 1].3 as usize)
            }
        };
        p += l;
        if sym == rlc {
            if p + 8 > nbits {
                return Err(Damage::new("The EXR Huffman data is damaged (a code runs past the end)"));
            }
            let cnt = ((byte(p >> 3) << 8 | byte((p >> 3) + 1)) >> (8 - (p & 7))) as usize & 255;
            p += 8;
            let last = match out.last() {
                Some(&v) => v,
                None => return Err(Damage::new("The EXR Huffman data is damaged (a run with nothing before it)")),
            };
            if out.len() + cnt > nraw {
                return Err(Damage::new("The EXR Huffman data is damaged (too much data)"));
            }
            out.extend(std::iter::repeat_n(last, cnt));
        } else {
            if out.len() >= nraw {
                return Err(Damage::new("The EXR Huffman data is damaged (too much data)"));
            }
            out.push(sym as u16);
        }
    }
    if p != nbits {
        return Err(Damage::new("The EXR Huffman data is damaged (a code runs past the end)"));
    }
    if out.len() != nraw {
        return Err(Damage::new("The EXR Huffman data is cut short"));
    }
    Ok(out)
}

fn le16(v: &[u16]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 2);
    for &x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

#[pyfunction]
pub fn exr_huf<'py>(py: Python<'py>, data: PyBackedBytes, nraw: usize) -> PyResult<Bound<'py, PyBytes>> {
    if nraw > 1 << 30 {
        return Err(raise(py, Damage::new("The EXR block is too large")));
    }
    let out = py.detach(|| huf(&data, nraw).map(|v| le16(&v))).map_err(|e| raise(py, e))?;
    Ok(PyBytes::new(py, &out))
}

// ---------------------------------------------------------------- PIZ

#[inline]
fn wdec14(l: u16, h: u16) -> (u16, u16) {
    let ls = l as i16 as i32;
    let hs = h as i16 as i32;
    let ai = ls + (hs & 1) + (hs >> 1);
    (ai as u16, (ai - hs) as u16)
}

#[inline]
fn wdec16(l: u16, h: u16) -> (u16, u16) {
    let m = l as i32;
    let d = h as i32;
    let bb = (m - (d >> 1)) & 0xFFFF;
    let aa = (d + bb - 0x8000) & 0xFFFF;
    (aa as u16, bb as u16)
}

/// Undoes the wavelet on an ny x nx grid of elements at a[y * oy + x * ox].
fn wav_decode(a: &mut [u16], nx: usize, ox: usize, ny: usize, oy: usize, maxval: usize) {
    let w14 = maxval < (1 << 14);
    let dec = if w14 { wdec14 } else { wdec16 };
    let n = nx.min(ny);
    let mut p = 1usize;
    while p <= n {
        p <<= 1;
    }
    p >>= 1;
    let mut p2 = p;
    p >>= 1;
    while p >= 1 {
        let (oy1, oy2, ox1, ox2) = (oy * p, oy * p2, ox * p, ox * p2);
        let nby = ny / p2;
        let nbx = nx / p2;
        for by in 0..nby {
            let py = by * oy2;
            for bx in 0..nbx {
                let px = py + bx * ox2;
                let (p01, p10) = (px + ox1, px + oy1);
                let p11 = p10 + ox1;
                let (i00, i10) = dec(a[px], a[p10]);
                let (i01, i11) = dec(a[p01], a[p11]);
                let (x00, x01) = dec(i00, i01);
                let (x10, x11) = dec(i10, i11);
                a[px] = x00;
                a[p01] = x01;
                a[p10] = x10;
                a[p11] = x11;
            }
            if nx & p != 0 {
                let px = py + nbx * ox2;
                let p10 = px + oy1;
                let (i00, i10) = dec(a[px], a[p10]);
                a[px] = i00;
                a[p10] = i10;
            }
        }
        if ny & p != 0 {
            let py = nby * oy2;
            for bx in 0..nbx {
                let px = py + bx * ox2;
                let p01 = px + ox1;
                let (i00, i01) = dec(a[px], a[p01]);
                a[px] = i00;
                a[p01] = i01;
            }
        }
        p2 = p;
        p >>= 1;
    }
}

fn piz(raw: &[u8], geom: &[(usize, usize, usize)], nraw: usize) -> Res<Vec<u16>> {
    if raw.len() < 4 {
        return Err(Damage::new("The EXR PIZ data is cut short"));
    }
    let mn = u16::from_le_bytes([raw[0], raw[1]]) as usize;
    let mx = u16::from_le_bytes([raw[2], raw[3]]) as usize;
    let mut pos = 4;
    let mut bitmap = [0u8; 8192];
    if mn <= mx {
        if mx >= 8192 || pos + mx - mn + 1 > raw.len() {
            return Err(Damage::new("The EXR PIZ data is damaged (bitmap)"));
        }
        bitmap[mn..=mx].copy_from_slice(&raw[pos..pos + mx - mn + 1]);
        pos += mx - mn + 1;
    }
    let mut lut = vec![0u16; 65536];
    let mut k = 0usize;
    for i in 0..65536usize {
        if i == 0 || bitmap[i >> 3] & (1 << (i & 7)) != 0 {
            lut[k] = i as u16;
            k += 1;
        }
    }
    let maxval = k - 1;
    if pos + 4 > raw.len() {
        return Err(Damage::new("The EXR PIZ data is cut short"));
    }
    let length = i32::from_le_bytes([raw[pos], raw[pos + 1], raw[pos + 2], raw[pos + 3]]);
    pos += 4;
    if length < 0 || pos + length as usize > raw.len() {
        return Err(Damage::new("The EXR PIZ data is damaged (Huffman length)"));
    }
    let mut v = huf(&raw[pos..pos + length as usize], nraw)?;
    let mut off = 0usize;
    for &(nx, ny, size) in geom {
        let n = nx * ny * size;
        if off + n > v.len() {
            return Err(Damage::new("The EXR PIZ data is damaged (channel sizes)"));
        }
        for j in 0..size {
            wav_decode(&mut v[off + j..off + n], nx, size, ny, nx * size, maxval);
        }
        off += n;
    }
    for x in v.iter_mut() {
        *x = lut[*x as usize];
    }
    Ok(v)
}

#[pyfunction]
pub fn exr_piz<'py>(py: Python<'py>, raw: PyBackedBytes, geom: Vec<(usize, usize, usize)>, nraw: usize) -> PyResult<Bound<'py, PyBytes>> {
    let total: usize = geom.iter().map(|g| g.0.saturating_mul(g.1).saturating_mul(g.2)).sum();
    if nraw > 1 << 30 || total != nraw {
        return Err(raise(py, Damage::new("The EXR PIZ block sizes don't add up")));
    }
    let out = py.detach(|| piz(&raw, &geom, nraw).map(|v| le16(&v))).map_err(|e| raise(py, e))?;
    Ok(PyBytes::new(py, &out))
}

// ---------------------------------------------------------------- B44

fn unpack14(b: &[u8], s: &mut [u16; 16]) {
    let b: Vec<u32> = b.iter().map(|&x| x as u32).collect();
    let mut t = [0u32; 16];
    t[0] = (b[0] << 8) | b[1];
    let shift = b[2] >> 2;
    let bias = 0x20u32 << shift;
    let d = |v: u32| -> u32 { (v & 0x3f) << shift };
    t[4] = t[0].wrapping_add(d((b[2] << 4) | (b[3] >> 4))).wrapping_sub(bias);
    t[8] = t[4].wrapping_add(d((b[3] << 2) | (b[4] >> 6))).wrapping_sub(bias);
    t[12] = t[8].wrapping_add(d(b[4])).wrapping_sub(bias);
    t[1] = t[0].wrapping_add(d(b[5] >> 2)).wrapping_sub(bias);
    t[5] = t[4].wrapping_add(d((b[5] << 4) | (b[6] >> 4))).wrapping_sub(bias);
    t[9] = t[8].wrapping_add(d((b[6] << 2) | (b[7] >> 6))).wrapping_sub(bias);
    t[13] = t[12].wrapping_add(d(b[7])).wrapping_sub(bias);
    t[2] = t[1].wrapping_add(d(b[8] >> 2)).wrapping_sub(bias);
    t[6] = t[5].wrapping_add(d((b[8] << 4) | (b[9] >> 4))).wrapping_sub(bias);
    t[10] = t[9].wrapping_add(d((b[9] << 2) | (b[10] >> 6))).wrapping_sub(bias);
    t[14] = t[13].wrapping_add(d(b[10])).wrapping_sub(bias);
    t[3] = t[2].wrapping_add(d(b[11] >> 2)).wrapping_sub(bias);
    t[7] = t[6].wrapping_add(d((b[11] << 4) | (b[12] >> 4))).wrapping_sub(bias);
    t[11] = t[10].wrapping_add(d((b[12] << 2) | (b[13] >> 6))).wrapping_sub(bias);
    t[15] = t[14].wrapping_add(d(b[13])).wrapping_sub(bias);
    for (o, &v) in s.iter_mut().zip(t.iter()) {
        *o = unsign(v & 0xFFFF);
    }
}

#[inline]
fn unsign(v: u32) -> u16 {
    if v & 0x8000 != 0 {
        (v & 0x7FFF) as u16
    } else {
        (!v & 0xFFFF) as u16
    }
}

fn b44(raw: &[u8], geom: &[(usize, usize, usize, bool)], table: &[u16]) -> Res<Vec<u16>> {
    let n = raw.len();
    let mut pos = 0usize;
    let mut out: Vec<u16> = Vec::new();
    for &(nx, ny, size, plin) in geom {
        if size != 2 {
            let k = nx * ny * size;
            if pos + k > n {
                return Err(Damage::new("The EXR B44 data is cut short"));
            }
            out.extend(raw[pos..pos + k].as_chunks::<2>().0.iter().map(|c| u16::from_le_bytes(*c)));
            pos += k;
            continue;
        }
        let base = out.len();
        out.resize(base + nx * ny, 0);
        let mut s = [0u16; 16];
        for by in (0..ny).step_by(4) {
            for bx in (0..nx).step_by(4) {
                if pos + 3 > n {
                    return Err(Damage::new("The EXR B44 data is cut short"));
                }
                if raw[pos + 2] >= 52 {
                    let v = unsign(((raw[pos] as u32) << 8) | raw[pos + 1] as u32);
                    s = [v; 16];
                    pos += 3;
                } else {
                    if pos + 14 > n {
                        return Err(Damage::new("The EXR B44 data is cut short"));
                    }
                    unpack14(&raw[pos..pos + 14], &mut s);
                    pos += 14;
                }
                for yy in 0..4 {
                    let y = by + yy;
                    if y >= ny {
                        break;
                    }
                    for xx in 0..4 {
                        let x = bx + xx;
                        if x >= nx {
                            break;
                        }
                        let v = s[yy * 4 + xx];
                        out[base + y * nx + x] = if plin && table.len() == 65536 { table[v as usize] } else { v };
                    }
                }
            }
        }
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (raw, geom, table=None))]
pub fn exr_b44<'py>(py: Python<'py>, raw: PyBackedBytes, geom: Vec<(usize, usize, usize, bool)>,
                    table: Option<Vec<u16>>) -> PyResult<Bound<'py, PyBytes>> {
    let total: usize = geom.iter().map(|g| g.0.saturating_mul(g.1).saturating_mul(g.2)).sum();
    if total > 1 << 31 {
        return Err(raise(py, Damage::new("The EXR block is too large")));
    }
    let t = table.unwrap_or_default();
    let out = py.detach(|| b44(&raw, &geom, &t).map(|v| le16(&v))).map_err(|e| raise(py, e))?;
    Ok(PyBytes::new(py, &out))
}

// ---------------------------------------------------------------- DWA

fn dwa_ac(ac: &[u16], nblocks: usize, mut pos: usize) -> Res<(Vec<u16>, Vec<u8>, usize)> {
    let mut coef = vec![0u16; nblocks * 64];
    let mut last = vec![0u8; nblocks];
    let n = ac.len();
    for b in 0..nblocks {
        let row = &mut coef[b * 64..b * 64 + 64];
        let mut k = 1usize;
        let mut lastnz = 0usize;
        while k < 64 {
            if pos >= n {
                return Err(Damage::new("The EXR DWA data is cut short (AC values)"));
            }
            let v = ac[pos];
            pos += 1;
            if v == 0xFF00 {
                break;
            }
            if v >> 8 == 0xFF {
                k += (v & 0xFF) as usize;
            } else {
                if k > 63 {
                    return Err(Damage::new("The EXR DWA data is damaged (AC run)"));
                }
                row[k] = v;
                lastnz = k;
                k += 1;
            }
        }
        last[b] = lastnz as u8;
    }
    Ok((coef, last, pos))
}

#[pyfunction]
pub fn exr_dwa_ac<'py>(py: Python<'py>, ac: PyBackedBytes, nblocks: usize, pos: usize)
                       -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>, usize)> {
    if nblocks > 1 << 26 {
        return Err(raise(py, Damage::new("The EXR DWA block is too large")));
    }
    let res = py.detach(|| {
        let v: Vec<u16> = ac.as_chunks::<2>().0.iter().map(|c| u16::from_le_bytes(*c)).collect();
        dwa_ac(&v, nblocks, pos)
    });
    let (coef, last, pos) = res.map_err(|e| raise(py, e))?;
    Ok((PyBytes::new(py, &le16(&coef)), PyBytes::new(py, &last), pos))
}

// ---------------------------------------------------------------- PIZ writing (exr.py's piz_encode)

#[inline]
fn wenc14(a: u16, b: u16) -> (u16, u16) {
    let (a, b) = (a as i16 as i32, b as i16 as i32);
    (((a + b) >> 1) as u16, (a - b) as u16)
}

#[inline]
fn wenc16(a: u16, b: u16) -> (u16, u16) {
    let ao = (a as i32 + 0x8000) & 0xFFFF;
    let b = b as i32;
    let mut m = (ao + b) >> 1;
    let d = ao - b;
    if d < 0 {
        m = (m + 0x8000) & 0xFFFF;
    }
    (m as u16, (d & 0xFFFF) as u16)
}

fn wav_encode(a: &mut [u16], nx: usize, ox: usize, ny: usize, oy: usize, maxval: usize) {
    let enc = if maxval < (1 << 14) { wenc14 } else { wenc16 };
    let n = nx.min(ny);
    let (mut p, mut p2) = (1usize, 2usize);
    while p2 <= n {
        let (nby, nbx) = (ny / p2, nx / p2);
        for by in 0..nby {
            let r0 = by * p2 * oy;
            let r1 = r0 + p * oy;
            for bx in 0..nbx {
                let c0 = bx * p2 * ox;
                let c1 = c0 + p * ox;
                let (i00, i01) = enc(a[r0 + c0], a[r0 + c1]);
                let (i10, i11) = enc(a[r1 + c0], a[r1 + c1]);
                let (x00, x10) = enc(i00, i10);
                let (x01, x11) = enc(i01, i11);
                a[r0 + c0] = x00;
                a[r1 + c0] = x10;
                a[r0 + c1] = x01;
                a[r1 + c1] = x11;
            }
            if nx & p != 0 {
                let c = nbx * p2 * ox;
                let (x, y) = enc(a[r0 + c], a[r1 + c]);
                a[r0 + c] = x;
                a[r1 + c] = y;
            }
        }
        if ny & p != 0 {
            let r = nby * p2 * oy;
            for bx in 0..nbx {
                let c0 = bx * p2 * ox;
                let c1 = c0 + p * ox;
                let (x, y) = enc(a[r + c0], a[r + c1]);
                a[r + c0] = x;
                a[r + c1] = y;
            }
        }
        p = p2;
        p2 <<= 1;
    }
}

/// Huffman code lengths: two queues, leaves sorted by frequency then symbol, ties to leaves.
fn huf_lengths(freq: &[u64]) -> Vec<u32> {
    let mut lengths = vec![0u32; freq.len()];
    let mut order: Vec<usize> = (0..freq.len()).filter(|&s| freq[s] > 0).collect();
    let n = order.len();
    if n == 1 {
        lengths[order[0]] = 1;
        return lengths;
    }
    order.sort_by_key(|&s| freq[s]); // stable: ties stay in symbol order
    let mut w: Vec<u64> = order.iter().map(|&s| freq[s]).collect();
    w.resize(2 * n - 1, 0);
    let mut parent = vec![0usize; 2 * n - 1];
    let (mut i, mut j, mut k) = (0usize, n, n);
    while k < 2 * n - 1 {
        let mut two = [0usize; 2];
        for t in &mut two {
            if i < n && (j >= k || w[i] <= w[j]) {
                *t = i;
                i += 1;
            } else {
                *t = j;
                j += 1;
            }
        }
        w[k] = w[two[0]] + w[two[1]];
        parent[two[0]] = k;
        parent[two[1]] = k;
        k += 1;
    }
    let mut depth = vec![0u32; 2 * n - 1];
    for node in (0..2 * n - 2).rev() {
        depth[node] = depth[parent[node]] + 1;
    }
    for (x, &s) in order.iter().enumerate() {
        lengths[s] = depth[x];
    }
    lengths
}

struct BitsOut {
    out: Vec<u8>,
    acc: u64,
    have: u32,
    total: usize,
}

impl BitsOut {
    fn new() -> Self {
        BitsOut { out: Vec::new(), acc: 0, have: 0, total: 0 }
    }
    fn put(&mut self, code: u64, len: u32) {
        let mut len = len;
        self.total += len as usize;
        while len > 0 {
            let take = len.min(32);
            let part = (code >> (len - take)) & ((1u64 << take) - 1);
            self.acc = (self.acc << take) | part;
            self.have += take;
            len -= take;
            while self.have >= 8 {
                self.have -= 8;
                self.out.push((self.acc >> self.have) as u8);
            }
            self.acc &= (1u64 << self.have) - 1;
        }
    }
    fn finish(mut self) -> (Vec<u8>, usize) {
        if self.have > 0 {
            self.out.push((self.acc << (8 - self.have)) as u8);
        }
        (self.out, self.total)
    }
}

fn huf_encode(values: &[u16]) -> Res<Vec<u8>> {
    if values.is_empty() {
        return Ok(Vec::new());
    }
    // runs: a symbol and up to 255 repeats per piece; 9 or more repeats use the run code
    let mut pieces: Vec<(u16, u32)> = Vec::new();
    let mut s = 0usize;
    while s < values.len() {
        let mut e = s + 1;
        while e < values.len() && values[e] == values[s] {
            e += 1;
        }
        let run = e - s;
        for _ in 0..run / 256 {
            pieces.push((values[s], 255));
        }
        if !run.is_multiple_of(256) {
            pieces.push((values[s], (run % 256 - 1) as u32));
        }
        s = e;
    }
    let im = values.iter().copied().min().unwrap_or(0) as usize;
    let rlc = values.iter().copied().max().unwrap_or(0) as usize + 1;
    let mut freq = vec![0u64; rlc + 1];
    let mut nbig = 0u64;
    for &(v, rep) in &pieces {
        if rep >= 9 {
            freq[v as usize] += 1;
            nbig += 1;
        } else {
            freq[v as usize] += rep as u64 + 1;
        }
    }
    freq[rlc] += nbig.max(1);
    let lengths = huf_lengths(&freq);
    if lengths.iter().any(|&l| l > 58) {
        return Err(Damage::new("Can't write this picture with PIZ (a Huffman code is too long)"));
    }
    let mut counts = [0u64; 59];
    for &l in &lengths {
        counts[l as usize] += 1;
    }
    let mut start = [0u64; 59];
    let mut c = 0u64;
    for l in (1..59).rev() {
        start[l] = c;
        c = (c + counts[l]) >> 1;
    }
    let mut codes = vec![0u64; lengths.len()];
    for (sym, &l) in lengths.iter().enumerate() {
        if l > 0 {
            codes[sym] = start[l as usize];
            start[l as usize] += 1;
        }
    }
    let mut bits = BitsOut::new();
    for &(v, rep) in &pieces {
        let (code, len) = (codes[v as usize], lengths[v as usize]);
        if rep >= 9 {
            bits.put(code, len);
            bits.put(codes[rlc], lengths[rlc]);
            bits.put(rep as u64, 8);
        } else {
            for _ in 0..=rep {
                bits.put(code, len);
            }
        }
    }
    let (data, nbits) = bits.finish();
    // the code length table
    let mut tab = BitsOut::new();
    let mut i = im;
    while i <= rlc {
        let l = lengths[i];
        if l == 0 {
            let mut run = 1usize;
            while i + run <= rlc && lengths[i + run] == 0 && run < 261 {
                run += 1;
            }
            if run >= 2 {
                if run >= 6 {
                    tab.put(63, 6);
                    tab.put((run - 6) as u64, 8);
                } else {
                    tab.put((59 + run - 2) as u64, 6);
                }
                i += run;
                continue;
            }
        }
        tab.put(l as u64, 6);
        i += 1;
    }
    let (table, _) = tab.finish();
    let mut out = Vec::with_capacity(20 + table.len() + data.len());
    for v in [im as u32, rlc as u32, table.len() as u32, nbits as u32, 0] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&table);
    out.extend_from_slice(&data);
    Ok(out)
}

fn piz_encode(raw: &[u8], geom: &[(usize, usize, usize)]) -> Res<Vec<u8>> {
    let mut v: Vec<u16> = raw.as_chunks::<2>().0.iter().map(|c| u16::from_le_bytes(*c)).collect();
    let mut present = vec![false; 65536];
    for &x in &v {
        present[x as usize] = true;
    }
    present[0] = false;
    let mut bitmap = [0u8; 8192];
    for (i, &p) in present.iter().enumerate() {
        if p {
            bitmap[i >> 3] |= 1 << (i & 7);
        }
    }
    present[0] = true;
    let mut fwd = vec![0u16; 65536];
    let mut k = 0usize;
    for (i, &p) in present.iter().enumerate() {
        if p {
            fwd[i] = k as u16;
            k += 1;
        }
    }
    let maxval = k - 1;
    for x in v.iter_mut() {
        *x = fwd[*x as usize];
    }
    let mut off = 0usize;
    for &(nx, ny, size) in geom {
        let n = nx * ny * size;
        if off + n > v.len() {
            return Err(Damage::new("The PIZ block sizes don't add up"));
        }
        for j in 0..size {
            wav_encode(&mut v[off + j..off + n], nx, size, ny, nx * size, maxval);
        }
        off += n;
    }
    let mut out = Vec::new();
    match (bitmap.iter().position(|&b| b != 0), bitmap.iter().rposition(|&b| b != 0)) {
        (Some(mn), Some(mx)) => {
            out.extend_from_slice(&(mn as u16).to_le_bytes());
            out.extend_from_slice(&(mx as u16).to_le_bytes());
            out.extend_from_slice(&bitmap[mn..=mx]);
        }
        _ => {
            out.extend_from_slice(&8191u16.to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
        }
    }
    let huf = huf_encode(&v)?;
    out.extend_from_slice(&(huf.len() as i32).to_le_bytes());
    out.extend_from_slice(&huf);
    Ok(out)
}

#[pyfunction]
pub fn exr_piz_encode<'py>(py: Python<'py>, raw: PyBackedBytes, geom: Vec<(usize, usize, usize)>) -> PyResult<Bound<'py, PyBytes>> {
    let total: usize = geom.iter().map(|g| g.0.saturating_mul(g.1).saturating_mul(g.2)).sum();
    if total.saturating_mul(2) != raw.len() || total > 1 << 30 {
        return Err(raise(py, Damage::new("The PIZ block sizes don't add up")));
    }
    let out = py.detach(|| piz_encode(&raw, &geom)).map_err(|e| raise(py, e))?;
    Ok(PyBytes::new(py, &out))
}
