use super::NC;
use wide::f32x8;

#[inline]
fn load8(s: &[f32]) -> f32x8 {
    f32x8::from(<[f32; 8]>::try_from(&s[..8]).expect("8 lanes"))
}

#[inline]
fn store8(s: &mut [f32], x: f32x8) {
    s[..8].copy_from_slice(&x.to_array());
}

/// Dot product of two equal-length slices whose length is a multiple of 8.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [f32x8::ZERO; 4];
    let mut ca = a.chunks_exact(32);
    let mut cb = b.chunks_exact(32);
    for (x, y) in (&mut ca).zip(&mut cb) {
        for t in 0..4 {
            acc[t] = load8(&x[t * 8..]).mul_add(load8(&y[t * 8..]), acc[t]);
        }
    }
    for (x, y) in ca
        .remainder()
        .chunks_exact(8)
        .zip(cb.remainder().chunks_exact(8))
    {
        acc[0] = load8(x).mul_add(load8(y), acc[0]);
    }
    ((acc[0] + acc[1]) + (acc[2] + acc[3])).reduce_add()
}

/// Two query rows against two key rows: every loaded vector feeds two products.
#[inline]
fn dot2x2(q0: &[f32], q1: &[f32], k0: &[f32], k1: &[f32]) -> [f32; 4] {
    let (mut a00, mut a01, mut a10, mut a11) = (f32x8::ZERO, f32x8::ZERO, f32x8::ZERO, f32x8::ZERO);
    for c in (0..q0.len()).step_by(8) {
        let (x0, x1) = (load8(&q0[c..]), load8(&q1[c..]));
        let (y0, y1) = (load8(&k0[c..]), load8(&k1[c..]));
        a00 = x0.mul_add(y0, a00);
        a01 = x0.mul_add(y1, a01);
        a10 = x1.mul_add(y0, a10);
        a11 = x1.mul_add(y1, a11);
    }
    [
        a00.reduce_add(),
        a01.reduce_add(),
        a10.reduce_add(),
        a11.reduce_add(),
    ]
}

pub fn dot_tile(q: &[f32], k: &[f32], d: usize, mr: usize, nk: usize, out: &mut [f32]) {
    let row = |r: usize| &q[r * d..(r + 1) * d];
    let key = |j: usize| &k[j * d..(j + 1) * d];
    let (mr2, nk2) = (mr & !1, nk & !1);
    for r in (0..mr2).step_by(2) {
        for j in (0..nk2).step_by(2) {
            let s = dot2x2(row(r), row(r + 1), key(j), key(j + 1));
            out[r * NC + j] = s[0];
            out[r * NC + j + 1] = s[1];
            out[(r + 1) * NC + j] = s[2];
            out[(r + 1) * NC + j + 1] = s[3];
        }
    }
    for r in 0..mr {
        let from = if r < mr2 { nk2 } else { 0 };
        for j in from..nk {
            out[r * NC + j] = dot(row(r), key(j));
        }
    }
}

pub fn axpy_rows(acc: &mut [f32], p: &[f32], v: &[f32], d: usize) {
    let mut c0 = 0;
    while c0 < d {
        let w = (d - c0).min(64);
        if w == 64 {
            let mut r = [f32x8::ZERO; 8];
            for (t, x) in r.iter_mut().enumerate() {
                *x = load8(&acc[c0 + t * 8..]);
            }
            for (j, &pj) in p.iter().enumerate() {
                let vj = &v[j * d + c0..];
                let ps = f32x8::splat(pj);
                for (t, x) in r.iter_mut().enumerate() {
                    *x = ps.mul_add(load8(&vj[t * 8..]), *x);
                }
            }
            for (t, x) in r.iter().enumerate() {
                store8(&mut acc[c0 + t * 8..], *x);
            }
        } else {
            for (j, &pj) in p.iter().enumerate() {
                let ps = f32x8::splat(pj);
                for t in (0..w).step_by(8) {
                    let o = load8(&acc[c0 + t..]);
                    store8(
                        &mut acc[c0 + t..],
                        ps.mul_add(load8(&v[j * d + c0 + t..]), o),
                    );
                }
            }
        }
        c0 += w;
    }
}

/// `x[i] = exp(x[i] - shift)` in place; returns the sum of the results.
pub fn exp_shifted_sum(x: &mut [f32], shift: f32) -> f32 {
    let sh = f32x8::splat(shift);
    let mut acc = f32x8::ZERO;
    let mut chunks = x.chunks_exact_mut(8);
    for c in &mut chunks {
        let e = (load8(c) - sh).exp();
        store8(c, e);
        acc += e;
    }
    let mut sum = acc.reduce_add();
    for v in chunks.into_remainder() {
        *v = (*v - shift).exp();
        sum += *v;
    }
    sum
}
