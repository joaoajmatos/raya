use super::NC;
use std::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vst1q_f32};

/// # Safety
/// The CPU must support NEON; slices must be at least `mr * d` and `nk * d` long.
#[target_feature(enable = "neon")]
pub unsafe fn dot_tile(q: &[f32], k: &[f32], d: usize, mr: usize, nk: usize, out: &mut [f32]) {
    assert!(q.len() >= mr * d && k.len() >= nk * d && out.len() >= (mr.max(1) - 1) * NC + nk);
    for j in 0..nk {
        let kj = k.as_ptr().add(j * d);
        for r in 0..mr {
            let qr = q.as_ptr().add(r * d);
            let mut a0 = vdupq_n_f32(0.0);
            let mut a1 = vdupq_n_f32(0.0);
            let mut c = 0;
            while c + 8 <= d {
                a0 = vfmaq_f32(a0, vld1q_f32(qr.add(c)), vld1q_f32(kj.add(c)));
                a1 = vfmaq_f32(a1, vld1q_f32(qr.add(c + 4)), vld1q_f32(kj.add(c + 4)));
                c += 8;
            }
            out[r * NC + j] = vaddvq_f32(vaddq_f32(a0, a1));
        }
    }
}

/// # Safety
/// The CPU must support NEON; `v` must hold `p.len()` rows of length `d`.
#[target_feature(enable = "neon")]
pub unsafe fn axpy_rows(acc: &mut [f32], p: &[f32], v: &[f32], d: usize) {
    assert!(acc.len() >= d && v.len() >= p.len() * d);
    let mut c0 = 0;
    while c0 < d {
        let w = (d - c0).min(32);
        let n = w / 4;
        let mut r = [vdupq_n_f32(0.0); 8];
        for (t, x) in r.iter_mut().enumerate().take(n) {
            *x = vld1q_f32(acc.as_ptr().add(c0 + t * 4));
        }
        for (j, &pj) in p.iter().enumerate() {
            let vj = v.as_ptr().add(j * d + c0);
            let ps = vdupq_n_f32(pj);
            for (t, x) in r.iter_mut().enumerate().take(n) {
                *x = vfmaq_f32(*x, ps, vld1q_f32(vj.add(t * 4)));
            }
        }
        for (t, x) in r.iter().enumerate().take(n) {
            vst1q_f32(acc.as_mut_ptr().add(c0 + t * 4), *x);
        }
        c0 += w;
    }
}
