//! PNG's Paeth filter, the one NumPy is slow at (three comparisons a byte; the same bytes as
//! png.py's NumPy version), and packing with libdeflate on several threads.

use std::sync::atomic::{AtomicUsize, Ordering};

use libdeflater::{Adler32, CompressionLvl, Compressor};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::PyBytes;

use crate::{raise, Damage};

#[inline]
fn predict(a: u8, b: u8, c: u8) -> u8 {
    let p = a as i16 + b as i16 - c as i16;
    let (pa, pb, pc) = ((p - a as i16).abs(), (p - b as i16).abs(), (p - c as i16).abs());
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// Rows (each `stride` bytes) under the Paeth filter, `prev` being the row above the first.
#[pyfunction]
pub fn png_paeth<'py>(py: Python<'py>, rows: PyBackedBytes, prev: PyBackedBytes, stride: usize, bpp: usize) -> PyResult<Bound<'py, PyBytes>> {
    if stride == 0 || !(1..=8).contains(&bpp) || !rows.len().is_multiple_of(stride) || prev.len() != stride {
        return Err(raise(py, Damage::new("PNG rows don't fit their size")));
    }
    let out = py.detach(|| {
        let mut out = vec![0u8; rows.len()];
        let mut up: &[u8] = &prev;
        for (row, o) in rows.chunks_exact(stride).zip(out.chunks_exact_mut(stride)) {
            for x in 0..bpp.min(stride) {
                o[x] = row[x].wrapping_sub(predict(0, up[x], 0));
            }
            for x in bpp..stride {
                o[x] = row[x].wrapping_sub(predict(row[x - bpp], up[x], up[x - bpp]));
            }
            up = row;
        }
        out
    });
    Ok(PyBytes::new(py, &out))
}

// ---- packing

const LENGTH_EXTRA: [u8; 29] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const CODE_ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

/// A Huffman code as deflate writes it, read by looking up the next bits (no more than it needs).
struct Code {
    table: Vec<u16>, // symbol << 4 | length; 0 where no code lands
    bits: u32,
}

impl Code {
    fn new() -> Self {
        Code { table: Vec::new(), bits: 0 }
    }

    fn build(&mut self, lens: &[u8]) -> Option<()> {
        let bits = lens.iter().copied().max().unwrap_or(0) as u32;
        if bits > 15 {
            return None;
        }
        self.bits = bits;
        self.table.clear();
        self.table.resize(1 << bits, 0);
        let mut count = [0u32; 16];
        for &l in lens {
            count[l as usize] += 1;
        }
        count[0] = 0;
        let mut next = [0u32; 17];
        for l in 1..16 {
            next[l + 1] = (next[l] + count[l]) << 1;
        }
        for (sym, &l) in lens.iter().enumerate() {
            if l == 0 {
                continue;
            }
            let l = l as u32;
            let code = next[l as usize];
            next[l as usize] += 1;
            if code >= 1 << l {
                return None; // more codes of this length than fit
            }
            let entry = (sym as u16) << 4 | l as u16;
            let mut at = (code.reverse_bits() >> (32 - l)) as usize;
            while at < self.table.len() {
                self.table[at] = entry;
                at += 1 << l;
            }
        }
        Some(())
    }
}

struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
}

impl Bits<'_> {
    #[inline]
    fn peek(&self) -> u64 {
        let at = self.pos >> 3;
        let mut word = [0u8; 8];
        if at + 8 <= self.data.len() {
            word.copy_from_slice(&self.data[at..at + 8]);
        } else if at < self.data.len() {
            word[..self.data.len() - at].copy_from_slice(&self.data[at..]);
        }
        u64::from_le_bytes(word) >> (self.pos & 7)
    }

    #[inline]
    fn take(&mut self, n: u32) -> u32 {
        let v = self.peek() & ((1u64 << n) - 1);
        self.pos += n as usize;
        v as u32
    }

    #[inline]
    fn symbol(&mut self, code: &Code) -> Option<usize> {
        let e = *code.table.get(self.peek() as usize & ((1usize << code.bits) - 1))?;
        if e == 0 {
            return None;
        }
        self.pos += (e & 15) as usize;
        Some((e >> 4) as usize)
    }
}

/// Where the last block of a deflate stream starts and ends, in bits, found by walking its blocks.
/// libdeflate only writes whole streams; to join several into one the last block's "final" flag has
/// to go, and that bit could be anywhere.
fn last_block(data: &[u8], lit: &mut Code, dist: &mut Code, fixed: &(Vec<u8>, Vec<u8>)) -> Option<(usize, usize)> {
    let mut b = Bits { data, pos: 0 };
    let total = data.len() * 8;
    let mut lens = [0u8; 320];
    let mut code_code = Code::new();
    loop {
        let start = b.pos;
        if start + 3 > total {
            return None;
        }
        let is_final = b.take(1) == 1;
        match b.take(2) {
            0 => {
                b.pos = (b.pos + 7) & !7;
                let len = b.take(16) as usize;
                b.pos += 16 + len * 8;
            }
            kind @ (1 | 2) => {
                if kind == 1 {
                    lit.build(&fixed.0)?;
                    dist.build(&fixed.1)?;
                } else {
                    let nlit = b.take(5) as usize + 257;
                    let ndist = b.take(5) as usize + 1;
                    let ncode = b.take(4) as usize + 4;
                    let mut small = [0u8; 19];
                    for &at in CODE_ORDER.iter().take(ncode) {
                        small[at] = b.take(3) as u8;
                    }
                    code_code.build(&small)?;
                    let mut i = 0;
                    while i < nlit + ndist {
                        let s = b.symbol(&code_code)?;
                        let (value, repeat) = match s {
                            0..=15 => (s as u8, 1),
                            16 => (*lens.get(i.checked_sub(1)?)?, 3 + b.take(2) as usize),
                            17 => (0, 3 + b.take(3) as usize),
                            _ => (0, 11 + b.take(7) as usize),
                        };
                        lens.get_mut(i..i + repeat)?.fill(value);
                        i += repeat;
                    }
                    lit.build(&lens[..nlit])?;
                    dist.build(&lens[nlit..nlit + ndist])?;
                }
                loop {
                    if b.pos > total {
                        return None;
                    }
                    let s = b.symbol(lit)?;
                    if s < 256 {
                        continue;
                    }
                    if s == 256 {
                        break;
                    }
                    b.pos += *LENGTH_EXTRA.get(s - 257)? as usize;
                    let d = b.symbol(dist)?;
                    if d > 29 {
                        return None;
                    }
                    b.pos += if d < 4 { 0 } else { (d - 2) / 2 };
                }
            }
            _ => return None,
        }
        if b.pos > total {
            return None;
        }
        if is_final {
            return Some((start, b.pos));
        }
    }
}

/// One thread's packer: a compressor and the room its parts go in, both kept for all its parts.
/// Asking the system for fresh memory for each part is what keeps threads from running at once:
/// on Windows they queue up to have its pages mapped.
struct Packer {
    z: Compressor,
    out: Vec<u8>,
    lit: Code,
    dist: Code,
    fixed: (Vec<u8>, Vec<u8>),
    done: Vec<(usize, usize, usize)>, // part, where it starts in out, its length
}

impl Packer {
    fn new(level: CompressionLvl, room: usize) -> Self {
        let mut fixed_lit = vec![8u8; 288];
        fixed_lit[144..256].fill(9);
        fixed_lit[256..280].fill(7);
        Packer { z: Compressor::new(level), out: Vec::with_capacity(room), lit: Code::new(), dist: Code::new(), fixed: (fixed_lit, vec![5u8; 30]), done: Vec::new() }
    }

    fn stored(&mut self, part: &[u8], last: bool) {
        let mut blocks = part.chunks(65535).peekable();
        if blocks.peek().is_none() && last {
            self.out.extend_from_slice(&[1, 0, 0, 0xFF, 0xFF]);
        }
        while let Some(block) = blocks.next() {
            self.out.push((last && blocks.peek().is_none()) as u8);
            self.out.extend_from_slice(&(block.len() as u16).to_le_bytes());
            self.out.extend_from_slice(&(!(block.len() as u16)).to_le_bytes());
            self.out.extend_from_slice(block);
        }
    }

    fn pack(&mut self, index: usize, part: &[u8], last: bool) {
        let start = self.out.len();
        let bound = self.z.deflate_compress_bound(part.len());
        self.out.resize(start + bound, 0);
        let Ok(size) = self.z.deflate_compress(part, &mut self.out[start..]) else {
            self.out.truncate(start);
            self.stored(part, last);
            self.done.push((index, start, self.out.len() - start));
            return;
        };
        self.out.truncate(start + size);
        if !last {
            match last_block(&self.out[start..], &mut self.lit, &mut self.dist, &self.fixed) {
                Some((first, end)) => {
                    // no longer the last block, and an empty stored one after it to end on a whole byte
                    self.out[start + first / 8] &= !(1 << (first % 8));
                    self.out.truncate(start + end.div_ceil(8));
                    if end % 8 != 0 {
                        self.out[start + end / 8] &= (1u8 << (end % 8)) - 1;
                    }
                    self.out.resize(start + (end + 3).div_ceil(8), 0);
                    self.out.extend_from_slice(&[0, 0, 0xFF, 0xFF]);
                }
                None => {
                    self.out.truncate(start);
                    self.stored(part, false);
                }
            }
        }
        self.done.push((index, start, self.out.len() - start));
    }
}

/// A zlib stream of `raw`, packed by libdeflate at `level` (1 to 12) on up to `threads` threads, a
/// part of `part_size` bytes at a time. The parts join into one stream, as a PNG needs.
#[pyfunction]
pub fn png_deflate<'py>(py: Python<'py>, raw: PyBackedBytes, level: i32, threads: usize, part_size: usize) -> PyResult<Bound<'py, PyBytes>> {
    let Ok(lvl) = CompressionLvl::new(level) else {
        return Err(raise(py, Damage::new("no such packing level")));
    };
    if part_size == 0 {
        return Err(raise(py, Damage::new("no such part size")));
    }
    let (packers, check) = py.detach(|| {
        let parts: Vec<&[u8]> = if raw.is_empty() { vec![&raw[..]] } else { raw.chunks(part_size).collect() };
        let threads = threads.clamp(1, parts.len());
        let room = raw.len() / threads + (raw.len() / threads) / 8 + (1 << 16);
        let next = AtomicUsize::new(0);
        let work = || {
            let mut p = Packer::new(lvl, room);
            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(part) = parts.get(i) else { return p };
                p.pack(i, part, i + 1 == parts.len());
            }
        };
        let packers: Vec<Packer> = if threads == 1 {
            vec![work()]
        } else {
            std::thread::scope(|s| {
                let running: Vec<_> = (0..threads).map(|_| s.spawn(work)).collect();
                running.into_iter().map(|t| t.join().expect("a packing thread stopped")).collect()
            })
        };
        let mut adler = Adler32::new();
        adler.update(&raw);
        (packers, adler.sum())
    });
    let mut order: Vec<(usize, usize, usize, usize)> = Vec::new();
    for (who, p) in packers.iter().enumerate() {
        order.extend(p.done.iter().map(|&(i, start, len)| (i, who, start, len)));
    }
    order.sort_unstable();
    let total: usize = order.iter().map(|o| o.3).sum::<usize>() + 6;
    PyBytes::new_with(py, total, |out| {
        out[..2].copy_from_slice(&[0x78, 0x9c]);
        let mut at = 2;
        for &(_, who, start, len) in &order {
            out[at..at + len].copy_from_slice(&packers[who].out[start..start + len]);
            at += len;
        }
        out[at..].copy_from_slice(&check.to_be_bytes());
        Ok(())
    })
}
