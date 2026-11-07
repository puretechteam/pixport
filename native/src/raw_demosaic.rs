//! Demosaicing (pixport/raw/demosaic.py `_rcd` and `_generic`), the same float32 arithmetic in the
//! same order, so the results are identical. The picture is cut into strips (each with its own
//! border of real rows) that are worked on by up to 8 threads.

use crate::{raise, Damage};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

const PAD: usize = 24;
const STRIP: usize = 64;
const EPS: f32 = 1e-5;

/// demosaic.pad_index: source index for padded position i (0..n+2*pad).
fn pad_index(n: usize, pad: usize, period: usize, mirror: bool) -> Vec<usize> {
    let (n_i, pad_i, p) = (n as i64, pad as i64, period as i64);
    (-pad_i..n_i + pad_i)
        .map(|i| {
            if mirror && n > pad {
                let mut k = i.abs();
                if k >= n_i {
                    k = 2 * (n_i - 1) - k;
                }
                k as usize
            } else if i < 0 {
                i.rem_euclid(p) as usize
            } else if i >= n_i {
                ((n_i - p) + (i - (n_i - p)).rem_euclid(p)) as usize
            } else {
                i as usize
            }
        })
        .collect()
}

struct Grid {
    h: usize,
    w: usize,
}

impl Grid {
    #[inline(always)]
    fn at(&self, a: &[f32], y: usize, x: usize, dy: isize, dx: isize) -> f32 {
        a[((y as isize + dy) as usize) * self.w + (x as isize + dx) as usize]
    }
}

/// One padded strip (h x w) of the mosaic, `idx` its colour per photosite: RGB, h x w x 3.
fn rcd(c: &[f32], idx: &[u8], h: usize, w: usize) -> Vec<f32> {
    let g_ = Grid { h, w };
    let r = 4usize;
    let n = h * w;
    let half = 0.5f32;
    let inner = |f: &mut dyn FnMut(usize, usize)| {
        for y in r..h.saturating_sub(r) {
            for x in r..w.saturating_sub(r) {
                f(y, x);
            }
        }
    };
    let at = |a: &[f32], y: usize, x: usize, dy: isize, dx: isize| g_.at(a, y, x, dy, dx);
    let _ = g_.h;
    // 1.
    let mut hv = vec![0f32; n];
    let mut hh = vec![0f32; n];
    inner(&mut |y, x| {
        hv[y * w + x] = (at(c, y, x, -1, 0) - at(c, y, x, 1, 0)).abs()
            + half * (2.0 * at(c, y, x, 0, 0) - at(c, y, x, -2, 0) - at(c, y, x, 2, 0)).abs();
        hh[y * w + x] = (at(c, y, x, 0, -1) - at(c, y, x, 0, 1)).abs()
            + half * (2.0 * at(c, y, x, 0, 0) - at(c, y, x, 0, -2) - at(c, y, x, 0, 2)).abs();
    });
    let mut vh = vec![0.5f32; n];
    inner(&mut |y, x| {
        let (a, b, d) = (at(&hv, y, x, -1, 0), at(&hv, y, x, 0, 0), at(&hv, y, x, 1, 0));
        let sv = EPS + a * a + b * b + d * d;
        let (a, b, d) = (at(&hh, y, x, 0, -1), at(&hh, y, x, 0, 0), at(&hh, y, x, 0, 1));
        let sh = EPS + a * a + b * b + d * d;
        vh[y * w + x] = sv / (sv + sh);
    });
    // 2.
    let mut lp = vec![0f32; n];
    inner(&mut |y, x| {
        lp[y * w + x] = 0.25f32 * at(c, y, x, 0, 0)
            + 0.125f32 * (at(c, y, x, -1, 0) + at(c, y, x, 1, 0) + at(c, y, x, 0, -1) + at(c, y, x, 0, 1))
            + 0.0625f32 * (at(c, y, x, -1, -1) + at(c, y, x, -1, 1) + at(c, y, x, 1, -1) + at(c, y, x, 1, 1));
    });
    // 3.
    let mut g = vec![0f32; n];
    inner(&mut |y, x| {
        if idx[y * w + x] == 1 {
            g[y * w + x] = at(c, y, x, 0, 0);
            return;
        }
        let cc = |dy, dx| at(c, y, x, dy, dx);
        let g_n = EPS + (cc(-1, 0) - cc(1, 0)).abs() + (cc(0, 0) - cc(-2, 0)).abs() + (cc(-1, 0) - cc(-3, 0)).abs()
            + (cc(-2, 0) - cc(-4, 0)).abs();
        let g_s = EPS + (cc(1, 0) - cc(-1, 0)).abs() + (cc(0, 0) - cc(2, 0)).abs() + (cc(1, 0) - cc(3, 0)).abs()
            + (cc(2, 0) - cc(4, 0)).abs();
        let g_w = EPS + (cc(0, -1) - cc(0, 1)).abs() + (cc(0, 0) - cc(0, -2)).abs() + (cc(0, -1) - cc(0, -3)).abs()
            + (cc(0, -2) - cc(0, -4)).abs();
        let g_e = EPS + (cc(0, 1) - cc(0, -1)).abs() + (cc(0, 0) - cc(0, 2)).abs() + (cc(0, 1) - cc(0, 3)).abs()
            + (cc(0, 2) - cc(0, 4)).abs();
        let l0 = at(&lp, y, x, 0, 0);
        let n_est = cc(-1, 0) * (2.0f32 * l0 / (EPS + l0 + at(&lp, y, x, -2, 0)));
        let s_est = cc(1, 0) * (2.0f32 * l0 / (EPS + l0 + at(&lp, y, x, 2, 0)));
        let w_est = cc(0, -1) * (2.0f32 * l0 / (EPS + l0 + at(&lp, y, x, 0, -2)));
        let e_est = cc(0, 1) * (2.0f32 * l0 / (EPS + l0 + at(&lp, y, x, 0, 2)));
        let v_est = (g_s * n_est + g_n * s_est) / (g_n + g_s);
        let h_est = (g_e * w_est + g_w * e_est) / (g_e + g_w);
        let vh_c = at(&vh, y, x, 0, 0);
        let vh_n = 0.25f32 * (at(&vh, y, x, -1, -1) + at(&vh, y, x, -1, 1) + at(&vh, y, x, 1, -1) + at(&vh, y, x, 1, 1));
        let vh_d = if (half - vh_c).abs() < (half - vh_n).abs() { vh_n } else { vh_c };
        g[y * w + x] = (v_est * (1.0 - vh_d) + h_est * vh_d).clamp(0.0, 1.0);
    });
    // 4.
    let mut hp = vec![0f32; n];
    let mut hq = vec![0f32; n];
    inner(&mut |y, x| {
        hp[y * w + x] = (at(c, y, x, -1, -1) - at(c, y, x, 1, 1)).abs()
            + half * (2.0 * at(c, y, x, 0, 0) - at(c, y, x, -2, -2) - at(c, y, x, 2, 2)).abs();
        hq[y * w + x] = (at(c, y, x, -1, 1) - at(c, y, x, 1, -1)).abs()
            + half * (2.0 * at(c, y, x, 0, 0) - at(c, y, x, -2, 2) - at(c, y, x, 2, -2)).abs();
    });
    let mut pq = vec![0.5f32; n];
    inner(&mut |y, x| {
        let (a, b, d) = (at(&hp, y, x, -1, -1), at(&hp, y, x, 0, 0), at(&hp, y, x, 1, 1));
        let sp = EPS + a * a + b * b + d * d;
        let (a, b, d) = (at(&hq, y, x, -1, 1), at(&hq, y, x, 0, 0), at(&hq, y, x, 1, -1));
        let sq = EPS + a * a + b * b + d * d;
        pq[y * w + x] = sp / (sp + sq);
    });
    let d: Vec<f32> = c.iter().zip(g.iter()).map(|(a, b)| a - b).collect();
    let mut rb = vec![0f32; n];
    inner(&mut |y, x| {
        let cc = |dy, dx| at(c, y, x, dy, dx);
        let gg = |dy, dx| at(&g, y, x, dy, dx);
        let pq_c = at(&pq, y, x, 0, 0);
        let pq_n = 0.25f32 * (at(&pq, y, x, -1, 0) + at(&pq, y, x, 1, 0) + at(&pq, y, x, 0, -1) + at(&pq, y, x, 0, 1));
        let pq_d = if (half - pq_c).abs() < (half - pq_n).abs() { pq_n } else { pq_c };
        let nw_g = EPS + (cc(-1, -1) - cc(1, 1)).abs() + (cc(-1, -1) - cc(-3, -3)).abs() + (gg(0, 0) - gg(-2, -2)).abs();
        let se_g = EPS + (cc(1, 1) - cc(-1, -1)).abs() + (cc(1, 1) - cc(3, 3)).abs() + (gg(0, 0) - gg(2, 2)).abs();
        let ne_g = EPS + (cc(-1, 1) - cc(1, -1)).abs() + (cc(-1, 1) - cc(-3, 3)).abs() + (gg(0, 0) - gg(-2, 2)).abs();
        let sw_g = EPS + (cc(1, -1) - cc(-1, 1)).abs() + (cc(1, -1) - cc(3, -3)).abs() + (gg(0, 0) - gg(2, -2)).abs();
        let p_est = (se_g * at(&d, y, x, -1, -1) + nw_g * at(&d, y, x, 1, 1)) / (nw_g + se_g);
        let q_est = (sw_g * at(&d, y, x, -1, 1) + ne_g * at(&d, y, x, 1, -1)) / (ne_g + sw_g);
        rb[y * w + x] = p_est * (1.0 - pq_d) + q_est * pq_d;
    });
    let mut dr = vec![0f32; n];
    let mut db = vec![0f32; n];
    for i in 0..n {
        match idx[i] {
            0 => {
                dr[i] = d[i];
                db[i] = rb[i];
            }
            2 => {
                db[i] = d[i];
                dr[i] = rb[i];
            }
            _ => {}
        }
    }
    // 5.
    let mut out = vec![0f32; n * 3];
    for i in 0..n {
        out[i * 3 + 1] = g[i];
    }
    inner(&mut |y, x| {
        let gg = |dy, dx| at(&g, y, x, dy, dx);
        let gn = EPS + (gg(0, 0) - gg(-2, 0)).abs() + (gg(-1, 0) - gg(1, 0)).abs();
        let gs = EPS + (gg(0, 0) - gg(2, 0)).abs() + (gg(1, 0) - gg(-1, 0)).abs();
        let gw = EPS + (gg(0, 0) - gg(0, -2)).abs() + (gg(0, -1) - gg(0, 1)).abs();
        let ge = EPS + (gg(0, 0) - gg(0, 2)).abs() + (gg(0, 1) - gg(0, -1)).abs();
        let vh_c = at(&vh, y, x, 0, 0);
        let vh_ng = 0.25f32 * (at(&vh, y, x, -1, 0) + at(&vh, y, x, 1, 0) + at(&vh, y, x, 0, -1) + at(&vh, y, x, 0, 1));
        let vh_g = if (half - vh_c).abs() < (half - vh_ng).abs() { vh_ng } else { vh_c };
        let g0 = gg(0, 0);
        let is_g = idx[y * w + x] == 1;
        for (ch, dd) in [(0usize, &dr), (2usize, &db)] {
            let v = (gs * at(dd, y, x, -1, 0) + gn * at(dd, y, x, 1, 0)) / (gn + gs);
            let hz = (ge * at(dd, y, x, 0, -1) + gw * at(dd, y, x, 0, 1)) / (gw + ge);
            let at_g = v * (1.0 - vh_g) + hz * vh_g;
            let val = g0 + if is_g { at_g } else { at(dd, y, x, 0, 0) };
            out[(y * w + x) * 3 + ch] = val.clamp(0.0, 1.0);
        }
    });
    out
}

/// demosaic._generic on one padded strip.
fn generic(c: &[f32], idx: &[u8], h: usize, w: usize) -> Vec<f32> {
    let r = 2usize;
    let n = h * w;
    let mut offs = Vec::new();
    for dy in -2isize..=2 {
        for dx in -2isize..=2 {
            if dy != 0 || dx != 0 {
                offs.push((dy, dx));
            }
        }
    }
    let at = |a: &[f32], y: usize, x: usize, dy: isize, dx: isize| a[((y as isize + dy) as usize) * w + (x as isize + dx) as usize];
    let interp = |vals: &[f32], known: &[f32], guide: &[f32], y: usize, x: usize| -> f32 {
        let mut num = 0f32;
        let mut den = 0f32;
        let g0 = guide[y * w + x];
        for &(dy, dx) in &offs {
            let k = at(known, y, x, dy, dx);
            let wgt = k / (((dy * dy + dx * dx) as f32) * (EPS + (at(guide, y, x, dy, dx) - g0).abs()));
            num += wgt * at(vals, y, x, dy, dx);
            den += wgt;
        }
        if den > 0.0 {
            num / den.max(1e-20)
        } else {
            0.0
        }
    };
    let known_g: Vec<f32> = idx.iter().map(|&k| if k == 1 { 1.0 } else { 0.0 }).collect();
    let cg: Vec<f32> = c.iter().zip(known_g.iter()).map(|(a, b)| a * b).collect();
    let mut g = c.to_vec();
    for y in r..h.saturating_sub(r) {
        for x in r..w.saturating_sub(r) {
            if idx[y * w + x] != 1 {
                g[y * w + x] = interp(&cg, &known_g, c, y, x);
            }
        }
    }
    let mut out = vec![0f32; n * 3];
    for i in 0..n {
        out[i * 3 + 1] = g[i];
    }
    for ch in [0u8, 2u8] {
        let known: Vec<f32> = idx.iter().map(|&k| if k == ch { 1.0 } else { 0.0 }).collect();
        let diff: Vec<f32> = (0..n).map(|i| (c[i] - g[i]) * known[i]).collect();
        for y in r..h.saturating_sub(r) {
            for x in r..w.saturating_sub(r) {
                let i = y * w + x;
                let g0 = g[i];
                let val = g0 + if idx[i] == ch { c[i] - g0 } else { interp(&diff, &known, &g, y, x) };
                out[i * 3 + ch as usize] = val.clamp(0.0, 1.0);
            }
        }
    }
    out
}

fn run(cfa: &[f32], h: usize, w: usize, pattern: &[u8], ph: usize, pw: usize, bayer: bool) -> Vec<f32> {
    let rows = pad_index(h, PAD, ph, bayer);
    let cols = pad_index(w, PAD, pw, bayer);
    let wp = w + 2 * PAD;
    let out = Mutex::new(vec![0f32; h * w * 3]);
    let next = AtomicUsize::new(0);
    let nstrips = h.div_ceil(STRIP);
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).clamp(1, 8);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                if k >= nstrips {
                    break;
                }
                let y0 = k * STRIP;
                let y1 = (y0 + STRIP).min(h);
                let hp = y1 - y0 + 2 * PAD;
                let mut c = vec![0f32; hp * wp];
                let mut idx = vec![0u8; hp * wp];
                for yy in 0..hp {
                    let src = rows[y0 + yy] * w;
                    // padded row y0 + yy sits at picture row y0 + yy - PAD; PAD is a whole number of repeats
                    let prow = (y0 + yy) % ph;
                    for xx in 0..wp {
                        c[yy * wp + xx] = cfa[src + cols[xx]];
                        idx[yy * wp + xx] = pattern[prow * pw + xx % pw];
                    }
                }
                let res = if bayer { rcd(&c, &idx, hp, wp) } else { generic(&c, &idx, hp, wp) };
                let mut o = out.lock().unwrap();
                for yy in 0..(y1 - y0) {
                    let src = ((yy + PAD) * wp + PAD) * 3;
                    let dst = (y0 + yy) * w * 3;
                    o[dst..dst + w * 3].copy_from_slice(&res[src..src + w * 3]);
                }
            });
        }
    });
    out.into_inner().unwrap()
}

fn check(py: Python<'_>, h: usize, w: usize, ph: usize, pw: usize) -> PyResult<()> {
    if ph == 0 || pw == 0 || !PAD.is_multiple_of(ph) || !PAD.is_multiple_of(pw) || h < ph || w < pw || h.saturating_mul(w) > 1 << 30 {
        return Err(raise(py, Damage::new("This colour filter repeat isn't supported")));
    }
    Ok(())
}

/// RGB (h * w * 3 float32) from a Bayer mosaic, as demosaic._rcd.
#[pyfunction]
pub fn demosaic_bayer<'py>(py: Python<'py>, cfa: PyReadonlyArray2<'py, f32>, pattern: PyReadonlyArray2<'py, u8>) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let (h, w) = (cfa.shape()[0], cfa.shape()[1]);
    let (ph, pw) = (pattern.shape()[0], pattern.shape()[1]);
    check(py, h, w, ph, pw)?;
    let (c, p) = (cfa.as_slice()?, pattern.as_slice()?.to_vec());
    let out = py.detach(|| run(c, h, w, &p, ph, pw, true));
    Ok(out.into_pyarray(py))
}

/// RGB from any mosaic (X-Trans), as demosaic._generic.
#[pyfunction]
pub fn demosaic_any<'py>(py: Python<'py>, cfa: PyReadonlyArray2<'py, f32>, pattern: PyReadonlyArray2<'py, u8>) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let (h, w) = (cfa.shape()[0], cfa.shape()[1]);
    let (ph, pw) = (pattern.shape()[0], pattern.shape()[1]);
    check(py, h, w, ph, pw)?;
    let (c, p) = (cfa.as_slice()?, pattern.as_slice()?.to_vec());
    let out = py.detach(|| run(c, h, w, &p, ph, pw, false));
    Ok(out.into_pyarray(py))
}
