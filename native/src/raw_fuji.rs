//! Fujifilm's lossless compressed RAF (pixport/raw/fuji.py `decode_strip`): one 768-wide strip,
//! six rows at a time, each value predicted from the lines above and its error Golomb-Rice coded
//! in one of 41 gradient contexts. The same results as the Python, value for value: the current
//! lines of every six rows (R2-R4, G2-G7, B2-B4) as little-endian uint16.

use crate::{raise, Damage, Res};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::PyBytes;

const MAX_ZEROS: u64 = 1 << 20;

const R0: usize = 0;
const R1: usize = 1;
const R2: usize = 2;
const R3: usize = 3;
const R4: usize = 4;
const G0: usize = 5;
const G1: usize = 6;
const G2: usize = 7;
const G3: usize = 8;
const G4: usize = 9;
const G5: usize = 10;
const G6: usize = 11;
const G7: usize = 12;
const B0: usize = 13;
const B1: usize = 14;
const B2: usize = 15;
const B3: usize = 16;
const B4: usize = 17;
const CURRENT: [usize; 12] = [R2, R3, R4, G2, G3, G4, G5, G6, G7, B2, B3, B4];

#[derive(Clone, Copy, PartialEq)]
enum How {
    Decode,
    Interp,
    InterpAt2, // interpolated where pos % 4 == 2
    DecodeNot0, // decoded where pos % 4 != 0
}

struct Bits<'a> {
    d: &'a [u8],
    pos: usize,
    end: usize,
    acc: u64,
    n: u32,
}

impl<'a> Bits<'a> {
    #[inline(always)]
    fn fill(&mut self) {
        while self.n <= 56 {
            let b = if self.pos < self.end { self.d[self.pos] } else { 0 };
            self.acc |= (b as u64) << (56 - self.n);
            self.n += 8;
            self.pos += 1;
        }
    }

    #[inline(always)]
    fn bits(&mut self, k: u32) -> u64 {
        if k == 0 {
            return 0;
        }
        if self.n < k {
            self.fill();
        }
        let v = self.acc >> (64 - k);
        self.acc <<= k;
        self.n -= k;
        v
    }

    #[inline(always)]
    fn zeros(&mut self) -> Res<u64> {
        let mut z = 0u64;
        loop {
            if self.n == 0 {
                self.fill();
            }
            let lz = self.acc.leading_zeros();
            if lz < self.n {
                z += lz as u64;
                self.acc = (self.acc << lz) << 1;
                self.n -= lz + 1;
                if z > MAX_ZEROS {
                    return Err(Damage::new("RAF compressed data is damaged (no end to a code)"));
                }
                return Ok(z);
            }
            z += self.n as u64;
            self.acc = 0;
            self.n = 0;
            if z > MAX_ZEROS {
                return Err(Damage::new("RAF compressed data is damaged (no end to a code)"));
            }
        }
    }
}

struct Strip<'a> {
    br: Bits<'a>,
    lw: usize,
    stride: usize,
    lines: Vec<i32>, // 18 lines of lw + 2
    bits: u32,
    top: i32,
    total: i32,
    limit: u64,
    q: Vec<i8>, // gradient bucket by difference + top
}

impl<'a> Strip<'a> {
    #[inline(always)]
    fn at(&self, line: usize, i: usize) -> i32 {
        self.lines[line * self.stride + i]
    }

    #[inline(always)]
    fn set(&mut self, line: usize, i: usize, v: i32) {
        self.lines[line * self.stride + i] = v;
    }

    #[inline(always)]
    fn qt(&self, diff: i32) -> i32 {
        self.q[(diff + self.top) as usize] as i32
    }

    #[inline(always)]
    fn code(&mut self, g: &mut [i32; 2]) -> Res<i32> {
        let z = self.br.zeros()?;
        let mut c: i64;
        if z < self.limit {
            let (v1, v2) = (g[0] as i64, g[1] as i64);
            let mut k = 0u32;
            if v2 < v1 {
                while k <= 14 {
                    k += 1;
                    if (v2 << k) >= v1 {
                        break;
                    }
                }
            }
            c = ((z << k) | self.br.bits(k)) as i64;
        } else {
            c = self.br.bits(self.bits) as i64 + 1;
        }
        if c >= self.total as i64 {
            return Err(Damage::new("RAF compressed data is damaged"));
        }
        c = if c & 1 != 0 { -1 - (c >> 1) } else { c >> 1 };
        g[0] += c.unsigned_abs() as i32;
        if g[1] == 64 {
            g[0] >>= 1;
            g[1] >>= 1;
        }
        g[1] += 1;
        Ok(c as i32)
    }

    #[inline(always)]
    fn store(&self, mut v: i32) -> i32 {
        if v < 0 {
            v += self.total;
        } else if v > self.top {
            v -= self.total;
        }
        if v >= 0 {
            v.min(self.top)
        } else {
            0
        }
    }

    #[inline(always)]
    fn even_pred(&self, li: usize, p: usize) -> (i32, i32, i32, i32, i32) {
        let rb = self.at(li - 1, p + 1);
        let rc = self.at(li - 1, p);
        let rd = self.at(li - 1, p + 2);
        let rf = self.at(li - 2, p + 1);
        let (dc, df, dd) = ((rc - rb).abs(), (rf - rb).abs(), (rd - rb).abs());
        let v = if dc > df && dc > dd {
            rf + rd + 2 * rb
        } else if dd > dc && dd > df {
            rf + rc + 2 * rb
        } else {
            rd + rc + 2 * rb
        };
        (v, rb, rc, rd, rf)
    }

    fn even(&mut self, li: usize, p: usize, how: How, grads: &mut [[i32; 2]; 41]) -> Res<()> {
        let decode = match how {
            How::Decode => true,
            How::Interp => false,
            How::InterpAt2 => p & 3 != 2,
            How::DecodeNot0 => p & 3 != 0,
        };
        let (v, rb, rc, _rd, rf) = self.even_pred(li, p);
        if !decode {
            self.set(li, p + 1, v >> 2);
            return Ok(());
        }
        let grad = self.qt(rb - rf) * 9 + self.qt(rc - rb);
        let c = self.code(&mut grads[grad.unsigned_abs() as usize])?;
        let v = v >> 2;
        let s = self.store(if grad < 0 { v - c } else { v + c });
        self.set(li, p + 1, s);
        Ok(())
    }

    fn odd(&mut self, li: usize, p: usize, grads: &mut [[i32; 2]; 41]) -> Res<()> {
        let ra = self.at(li, p);
        let rg = self.at(li, p + 2);
        let rb = self.at(li - 1, p + 1);
        let rc = self.at(li - 1, p);
        let rd = self.at(li - 1, p + 2);
        let grad = self.qt(rb - rc) * 9 + self.qt(rc - ra);
        let v = if (rb > rc && rb > rd) || (rb < rc && rb < rd) { (rg + ra + 2 * rb) >> 2 } else { (ra + rg) >> 1 };
        let c = self.code(&mut grads[grad.unsigned_abs() as usize])?;
        let s = self.store(if grad < 0 { v - c } else { v + c });
        self.set(li, p + 1, s);
        Ok(())
    }

    fn extend(&mut self, a: usize, b: usize) {
        let lw = self.lw;
        for i in a..=b {
            let l = self.at(i - 1, 1);
            let r = self.at(i - 1, lw);
            self.set(i, 0, l);
            self.set(i, lw + 1, r);
        }
    }
}

fn range_of(li: usize) -> (usize, usize) {
    match li {
        R2..=R4 => (R2, R4),
        B2..=B4 => (B2, B4),
        _ => (G2, G7),
    }
}

fn strip(d: &[u8], pos: usize, size: usize, lw: usize, nlines: usize, bits: u32, xtrans: bool) -> Res<Vec<u8>> {
    if !(12..=16).contains(&bits) || !(16..=4096).contains(&lw) || !(1..=0x4000).contains(&nlines) {
        return Err(Damage::new("RAF compressed data header is damaged"));
    }
    if pos > d.len() || size > d.len() - pos {
        return Err(Damage::new("RAF compressed data runs past the end of the file"));
    }
    let top = (1i32 << bits) - 1;
    let total = 1i32 << bits;
    let mut q = vec![0i8; (2 * top + 1) as usize];
    for (i, v) in q.iter_mut().enumerate() {
        let diff = i as i32 - top;
        let a = diff.abs();
        let b = if a == 0 {
            0
        } else if a < 18 {
            1
        } else if a < 67 {
            2
        } else if a < 276 {
            3
        } else {
            4
        };
        *v = (diff.signum() * b) as i8;
    }
    let stride = lw + 2;
    let mut s = Strip {
        br: Bits { d, pos, end: pos + size, acc: 0, n: 0 },
        lw,
        stride,
        lines: vec![0; 18 * stride],
        bits,
        top,
        total,
        limit: (4 * bits - bits - 1) as u64,
        q,
    };
    let first = std::cmp::max(2, (total + 32) >> 6);
    let mut even_g = [[[first, 1i32]; 41]; 3];
    let mut odd_g = [[[first, 1i32]; 41]; 3];
    use How::*;
    let passes: [(usize, usize, usize, How, How); 6] = if xtrans {
        [
            (R2, G2, 0, Interp, Decode),
            (G3, B2, 1, Decode, Interp),
            (R3, G4, 2, DecodeNot0, Interp),
            (G5, B3, 0, Decode, InterpAt2),
            (R4, G6, 1, InterpAt2, Decode),
            (G7, B4, 2, Interp, DecodeNot0),
        ]
    } else {
        [
            (R2, G2, 0, Decode, Decode),
            (G3, B2, 1, Decode, Decode),
            (R3, G4, 2, Decode, Decode),
            (G5, B3, 0, Decode, Decode),
            (R4, G6, 1, Decode, Decode),
            (G7, B4, 2, Decode, Decode),
        ]
    };
    let mut out: Vec<u8> = Vec::with_capacity(nlines * 12 * lw * 2);
    for _row in 0..nlines {
        for &(a, b, gs, how_a, how_b) in passes.iter() {
            let (mut pe, mut po) = (0usize, 1usize);
            while pe < lw || po < lw {
                if pe < lw {
                    s.even(a, pe, how_a, &mut even_g[gs])?;
                    s.even(b, pe, how_b, &mut even_g[gs])?;
                    pe += 2;
                }
                if pe > 8 {
                    s.odd(a, po, &mut odd_g[gs])?;
                    s.odd(b, po, &mut odd_g[gs])?;
                    po += 2;
                }
            }
            for li in [a, b] {
                let (x, y) = range_of(li);
                s.extend(x, y);
            }
        }
        for &li in CURRENT.iter() {
            for i in 1..=lw {
                out.extend_from_slice(&(s.at(li, i) as u16).to_le_bytes());
            }
        }
        for (dst, src) in [(R0, R3), (R1, R4), (G0, G6), (G1, G7), (B0, B3), (B1, B4)] {
            s.lines.copy_within(src * stride..(src + 1) * stride, dst * stride);
        }
        for (a, n) in [(R2, 3usize), (G2, 6), (B2, 3)] {
            s.lines[a * stride..(a + n) * stride].iter_mut().for_each(|v| *v = 0);
            let l = s.at(a - 1, 1);
            let r = s.at(a - 1, lw);
            s.set(a, 0, l);
            s.set(a, lw + 1, r);
        }
    }
    Ok(out)
}

/// One strip of Fujifilm compressed RAF: lines x 12 x line_width little-endian uint16.
#[pyfunction]
#[pyo3(signature = (data, pos, size, line_width, lines, bits, xtrans))]
#[allow(clippy::too_many_arguments)] // the Python signature
pub fn fuji_strip<'py>(
    py: Python<'py>,
    data: PyBackedBytes,
    pos: usize,
    size: usize,
    line_width: usize,
    lines: usize,
    bits: u32,
    xtrans: bool,
) -> PyResult<Bound<'py, PyBytes>> {
    let out = py.detach(|| strip(&data, pos, size, line_width, lines, bits, xtrans)).map_err(|e| raise(py, e))?;
    Ok(PyBytes::new(py, &out))
}
