use super::NC;
use std::arch::x86_64::{
    __m256, _mm256_add_ps, _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_fmadd_ps,
    _mm256_loadu_ps, _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps, _mm_add_ps, _mm_add_ss,
    _mm_cvtss_f32, _mm_movehdup_ps, _mm_movehl_ps,
};

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum(x: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(x);
    let hi = _mm256_extractf128_ps(x, 1);
    let s = _mm_add_ps(lo, hi);
    let sh = _mm_movehdup_ps(s);
    let s2 = _mm_add_ps(s, sh);
    let hi2 = _mm_movehl_ps(sh, s2);
    _mm_cvtss_f32(_mm_add_ss(s2, hi2))
}

/// # Safety
/// The CPU must support AVX2 and FMA; slices must be at least `d` (resp. `nk * d`) long.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_tile(q: &[f32], k: &[f32], d: usize, mr: usize, nk: usize, out: &mut [f32]) {
    assert!(q.len() >= mr * d && k.len() >= nk * d && out.len() >= (mr.max(1) - 1) * NC + nk);
    for j in 0..nk {
        let kj = k.as_ptr().add(j * d);
        for r in 0..mr {
            let qr = q.as_ptr().add(r * d);
            let mut a0 = _mm256_setzero_ps();
            let mut a1 = _mm256_setzero_ps();
            let mut c = 0;
            while c + 16 <= d {
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(qr.add(c)), _mm256_loadu_ps(kj.add(c)), a0);
                a1 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(qr.add(c + 8)),
                    _mm256_loadu_ps(kj.add(c + 8)),
                    a1,
                );
                c += 16;
            }
            if c < d {
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(qr.add(c)), _mm256_loadu_ps(kj.add(c)), a0);
            }
            out[r * NC + j] = hsum(_mm256_add_ps(a0, a1));
        }
    }
}

/// # Safety
/// The CPU must support AVX2 and FMA; `v` must hold `p.len()` rows of length `d`.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn axpy_rows(acc: &mut [f32], p: &[f32], v: &[f32], d: usize) {
    assert!(acc.len() >= d && v.len() >= p.len() * d);
    let mut c0 = 0;
    while c0 < d {
        let w = (d - c0).min(64);
        let n = w / 8;
        let mut r = [_mm256_setzero_ps(); 8];
        for (t, x) in r.iter_mut().enumerate().take(n) {
            *x = _mm256_loadu_ps(acc.as_ptr().add(c0 + t * 8));
        }
        for (j, &pj) in p.iter().enumerate() {
            let vj = v.as_ptr().add(j * d + c0);
            let ps = _mm256_set1_ps(pj);
            for (t, x) in r.iter_mut().enumerate().take(n) {
                *x = _mm256_fmadd_ps(ps, _mm256_loadu_ps(vj.add(t * 8)), *x);
            }
        }
        for (t, x) in r.iter().enumerate().take(n) {
            _mm256_storeu_ps(acc.as_mut_ptr().add(c0 + t * 8), *x);
        }
        c0 += w;
    }
}
