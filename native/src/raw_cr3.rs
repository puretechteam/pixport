//! Canon CR3's CRX line decoders (pixport/raw/cr3.py `_band_predicted`, `_band_plain` and
//! `decode_qp`): adaptive Golomb-Rice codes with JPEG-LS style runs, one subband at a time. The
//! same results as the Python, value for value; the values come back as little-endian int32.

use crate::{raise, Damage, Res};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::PyBytes;

const J: [u32; 32] = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15];
/// Leading zeros of a code before the data counts as damaged.
const MAX_ZEROS: u64 = 1 << 20;

/// MSB-first bits of d[pos..end], zeros past the end (where a code soon has too many leading
/// zeros to be real).
struct Bits<'a> {
    d: &'a [u8],
    pos: usize,
    end: usize,
    acc: u64, // next bits, left-aligned
    n: u32,   // how many of them are valid
}

impl<'a> Bits<'a> {
    fn new(d: &'a [u8], pos: usize, end: usize) -> Self {
        let end = end.min(d.len());
        Bits { d, pos, end, acc: 0, n: 0 }
    }

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
    fn bit(&mut self) -> Res<bool> {
        Ok(self.bits(1) != 0)
    }

    /// Zero bits before the next 1 (used up too).
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
                // lz + 1 can be 64 (a 1 in the last bit): shift in two steps
                self.acc = (self.acc << lz) << 1;
                self.n -= lz + 1;
                if z > MAX_ZEROS {
                    return Err(Damage::new("CRX data is damaged (no end to a code)"));
                }
                return Ok(z);
            }
            z += self.n as u64;
            self.acc = 0;
            self.n = 0;
            if z > MAX_ZEROS {
                return Err(Damage::new("CRX data is damaged (no end to a code)"));
            }
        }
    }

    #[inline(always)]
    fn code(&mut self, k: u32) -> Res<i64> {
        let z = self.zeros()?;
        if z >= 41 {
            return Ok(self.bits(21) as i64);
        }
        Ok(((z << k) | self.bits(k)) as i64)
    }
}

#[inline(always)]
fn knext(k: u32, c: i64, cap: u32) -> u32 {
    let k = k as i64;
    let nk = k - (c < ((1i64 << k) >> 1)) as i64 + ((c >> k) > 2) as i64 + ((c >> k) > 5) as i64;
    let nk = nk.max(0) as u32;
    if cap == 0 || nk < cap {
        nk
    } else {
        cap
    }
}

#[inline(always)]
fn zz(c: i64) -> i64 {
    (c >> 1) ^ -(c & 1)
}

#[inline(always)]
fn med(a: i64, b: i64, c: i64) -> i64 {
    let g = a + b - c;
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    if g < lo {
        lo
    } else if g > hi {
        hi
    } else {
        g
    }
}

/// A run's length after its leading 1 bit (at most room), updating the run index s.
#[inline(always)]
fn run(br: &mut Bits, s: &mut usize, room: usize) -> Res<usize> {
    let mut n = 1usize;
    while br.bit()? {
        n += 1usize << J[*s];
        if n > room {
            n = room;
            break;
        }
        if *s < 31 {
            *s += 1;
        }
        if n == room {
            break;
        }
    }
    if n < room {
        if J[*s] > 0 {
            n += br.bits(J[*s]) as usize;
        }
        if *s > 0 {
            *s -= 1;
        }
        if n > room {
            return Err(Damage::new("CRX run is longer than its line"));
        }
    }
    Ok(n)
}

fn band_predicted(br: &mut Bits, w: usize, h: usize, out: &mut [i64]) -> Res<()> {
    let (mut k, mut s) = (0u32, 0usize);
    let mut prev = vec![0i64; w + 2];
    let mut cur = vec![0i64; w + 2];
    // top line
    let (mut i, mut left) = (1usize, w as isize);
    while left > 1 {
        let pred = if cur[i - 1] != 0 {
            cur[i - 1]
        } else {
            if br.bit()? {
                let n = run(br, &mut s, left as usize)?;
                i += n;
                left -= n as isize;
                if left <= 0 {
                    break;
                }
            }
            0
        };
        let c = br.code(k)?;
        cur[i] = pred + zz(c);
        k = knext(k, c, 15);
        i += 1;
        left -= 1;
    }
    if left == 1 {
        let c = br.code(k)?;
        cur[i] = cur[i - 1] + zz(c);
        k = knext(k, c, 15);
    }
    cur[w + 1] = cur[w] + 1;
    out[..w].copy_from_slice(&cur[1..w + 1]);
    std::mem::swap(&mut prev, &mut cur);
    for y in 1..h {
        cur.iter_mut().for_each(|v| *v = 0);
        cur[0] = prev[1];
        let (mut i, mut left) = (1usize, w as isize);
        while left > 1 {
            let a = cur[i - 1];
            if a != prev[i] || a != prev[i + 1] {
                let c = br.code(k)?;
                cur[i] = med(a, prev[i], prev[i - 1]) + zz(c);
                let d = (prev[i + 1] - prev[i]).abs();
                k = knext(k, (c + 2 * d) >> 1, 15);
                i += 1;
                left -= 1;
                continue;
            }
            if br.bit()? {
                let n = run(br, &mut s, left as usize)?;
                for v in &mut cur[i..i + n] {
                    *v = a;
                }
                i += n;
                left -= n as isize;
            }
            if left > 0 {
                let mut c = br.code(k)?;
                cur[i] = prev[i] + zz(c);
                if left > 1 {
                    let d = (prev[i + 1] - prev[i]).abs();
                    c = (c + 2 * d) >> 1;
                }
                k = knext(k, c, 15);
                i += 1;
                left -= 1;
            }
        }
        if left == 1 {
            let c = br.code(k)?;
            cur[i] = med(cur[i - 1], prev[i], prev[i - 1]) + zz(c);
            k = knext(k, c, 15);
        }
        cur[w + 1] = cur[w] + 1;
        out[y * w..(y + 1) * w].copy_from_slice(&cur[1..w + 1]);
        std::mem::swap(&mut prev, &mut cur);
    }
    Ok(())
}

fn band_plain(br: &mut Bits, w: usize, h: usize, out: &mut [i64]) -> Res<()> {
    let (mut k, mut s) = (0u32, 0usize);
    let mut kline = vec![0u32; w + 1];
    let mut prev = vec![0i64; w + 2];
    let mut cur = vec![0i64; w + 2];
    // top line
    let (mut i, mut left) = (1usize, w as isize);
    while left > 1 {
        if cur[i - 1] != 0 {
            let c = br.code(k)?;
            cur[i] = zz(c);
            k = knext(k, c, 15);
        } else {
            if br.bit()? {
                let n = run(br, &mut s, left as usize)?;
                for v in &mut kline[i - 1..i - 1 + n] {
                    *v = 0;
                }
                i += n;
                left -= n as isize;
                if left <= 0 {
                    break;
                }
            }
            let c = br.code(k)?;
            cur[i] = zz(c + 1);
            k = knext(k, c, 15);
        }
        kline[i - 1] = k;
        i += 1;
        left -= 1;
    }
    if left == 1 {
        let c = br.code(k)?;
        cur[i] = zz(c);
        k = knext(k, c, 15);
        kline[i - 1] = k;
    }
    out[..w].copy_from_slice(&cur[1..w + 1]);
    std::mem::swap(&mut prev, &mut cur);
    for y in 1..h {
        cur.iter_mut().for_each(|v| *v = 0);
        let mut i = 0usize;
        while i + 1 < w {
            if prev[i + 2] != 0 || prev[i + 1] != 0 || cur[i] != 0 {
                let c = br.code(k)?;
                cur[i + 1] = zz(c);
                k = knext(k, c, 0);
                if kline[i + 1] as i64 - k as i64 > 1 {
                    k += 1;
                } else if k > 15 {
                    k = 15;
                }
            } else {
                if br.bit()? {
                    let n = run(br, &mut s, w - i)?;
                    for v in &mut kline[i..i + n] {
                        *v = 0;
                    }
                    i += n;
                }
                if i + 1 >= w {
                    if i + 1 == w {
                        let c = br.code(k)?;
                        cur[i + 1] = zz(c + 1);
                        k = knext(k, c, 15);
                        kline[i] = k;
                    }
                    i += 1;
                    continue;
                }
                let c = br.code(k)?;
                cur[i + 1] = zz(c + 1);
                k = knext(k, c, 0);
                if kline[i + 1] as i64 - k as i64 > 1 {
                    k += 1;
                } else if k > 15 {
                    k = 15;
                }
            }
            kline[i] = k;
            i += 1;
        }
        if i + 1 == w {
            let c = br.code(k)?;
            cur[i + 1] = zz(c);
            k = knext(k, c, 15);
            kline[i] = k;
        }
        out[y * w..(y + 1) * w].copy_from_slice(&cur[1..w + 1]);
        std::mem::swap(&mut prev, &mut cur);
    }
    Ok(())
}

fn qp_map(br: &mut Bits, w: usize, h: usize, out: &mut [i64]) -> Res<()> {
    let mut k = 0u32;
    let mut prev = vec![0i64; w + 2];
    let mut cur = vec![0i64; w + 2];
    let code = |br: &mut Bits, k: u32| -> Res<i64> {
        let z = br.zeros()?;
        if z >= 23 {
            return Ok(br.bits(8) as i64);
        }
        Ok(((z << k) | br.bits(k)) as i64)
    };
    for y in 0..h {
        cur.iter_mut().for_each(|v| *v = 0);
        if y == 0 {
            for x in 0..w {
                let c = code(br, k)?;
                cur[x + 1] = cur[x] + zz(c);
                k = knext(k, c, 7);
            }
        } else {
            cur[0] = prev[1];
            for x in 0..w {
                let c = code(br, k)?;
                cur[x + 1] = med(cur[x], prev[x + 1], prev[x]) + zz(c);
                if x + 1 < w {
                    let d = (prev[x + 2] - prev[x + 1]).abs();
                    k = knext(k, (c + 2 * d) >> 1, 7);
                } else {
                    k = knext(k, c, 7);
                }
            }
        }
        cur[w + 1] = cur[w] + 1;
        out[y * w..(y + 1) * w].copy_from_slice(&cur[1..w + 1]);
        std::mem::swap(&mut prev, &mut cur);
    }
    Ok(())
}

fn to_i32_bytes(v: &[i64]) -> Res<Vec<u8>> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for &x in v {
        if x < i32::MIN as i64 || x > i32::MAX as i64 {
            return Err(Damage::new("CRX values are out of range"));
        }
        out.extend_from_slice(&(x as i32).to_le_bytes());
    }
    Ok(out)
}

fn check_dims(d: &[u8], pos: usize, size: usize, w: usize, h: usize) -> Res<()> {
    if w == 0 || h == 0 || w.checked_mul(h).is_none_or(|n| n > 1 << 30) {
        return Err(Damage::new("CRX subband size is out of range"));
    }
    if pos > d.len() || size > d.len() - pos {
        return Err(Damage::new("CRX subband runs past the end of the file"));
    }
    Ok(())
}

fn band(d: &[u8], pos: usize, size: usize, w: usize, h: usize, predicted: bool) -> Res<Vec<u8>> {
    check_dims(d, pos, size, w, h)?;
    let mut out = vec![0i64; w * h];
    let mut br = Bits::new(d, pos, pos + size);
    if predicted {
        band_predicted(&mut br, w, h, &mut out)?;
    } else {
        band_plain(&mut br, w, h, &mut out)?;
    }
    to_i32_bytes(&out)
}

fn qp(d: &[u8], pos: usize, size: usize, w: usize, h: usize) -> Res<Vec<u8>> {
    check_dims(d, pos, size, w, h)?;
    let mut out = vec![0i64; w * h];
    let mut br = Bits::new(d, pos, pos + size);
    qp_map(&mut br, w, h, &mut out)?;
    to_i32_bytes(&out)
}

/// One CRX subband: width x height values as little-endian int32 bytes.
#[pyfunction]
#[pyo3(signature = (data, pos, size, width, height, predicted))]
pub fn crx_band<'py>(
    py: Python<'py>,
    data: PyBackedBytes,
    pos: usize,
    size: usize,
    width: usize,
    height: usize,
    predicted: bool,
) -> PyResult<Bound<'py, PyBytes>> {
    let out = py.detach(|| band(&data, pos, size, width, height, predicted)).map_err(|e| raise(py, e))?;
    Ok(PyBytes::new(py, &out))
}

/// A CRX tile's quantisation map (before the +4): width x height little-endian int32.
#[pyfunction]
pub fn crx_qp<'py>(py: Python<'py>, data: PyBackedBytes, pos: usize, size: usize, width: usize, height: usize) -> PyResult<Bound<'py, PyBytes>> {
    let out = py.detach(|| qp(&data, pos, size, width, height)).map_err(|e| raise(py, e))?;
    Ok(PyBytes::new(py, &out))
}
