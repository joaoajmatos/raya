/// Dense matrix multiply behind a trait so a BLAS library can be chosen later.
pub trait GemmBackend: Send + Sync {
    fn name(&self) -> &'static str;

    /// `out[i][j] = sum_t a[i][t] * b[j][t]` for row-major `a: [m, k]`, `b: [n, k]` and
    /// `out: [m, n]`, which is `a * b^T` and the layout of a `Linear` weight (`[out, in]`).
    ///
    /// # Panics
    /// If a slice length does not match `m`, `n` and `k`.
    fn matmul_nt(&self, out: &mut [f32], a: &[f32], b: &[f32], m: usize, n: usize, k: usize);
}

/// Plain loops, sequential f32 accumulation, single-threaded: the reference for feature 006.
#[derive(Debug, Default, Clone, Copy)]
pub struct Reference;

impl GemmBackend for Reference {
    fn name(&self) -> &'static str {
        "reference-loops"
    }

    fn matmul_nt(&self, out: &mut [f32], a: &[f32], b: &[f32], m: usize, n: usize, k: usize) {
        assert_eq!(a.len(), m * k, "a must be [m, k]");
        assert_eq!(b.len(), n * k, "b must be [n, k]");
        assert_eq!(out.len(), m * n, "out must be [m, n]");
        for i in 0..m {
            let arow = &a[i * k..(i + 1) * k];
            for j in 0..n {
                let brow = &b[j * k..(j + 1) * k];
                let mut sum = 0.0f32;
                for (x, y) in arow.iter().zip(brow) {
                    sum += x * y;
                }
                out[i * n + j] = sum;
            }
        }
    }
}
