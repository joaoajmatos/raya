use crate::attention::{validate, AttentionKernel, AttnParams, AttnShape, KernelError};

/// Scalar oracle and speed baseline: single-threaded, one dot product per allowed score,
/// no tiling and no explicit SIMD.
#[derive(Debug, Default, Clone, Copy)]
pub struct Naive;

impl AttentionKernel for Naive {
    fn name(&self) -> &'static str {
        "naive"
    }

    fn attend(
        &self,
        shape: AttnShape,
        params: AttnParams,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &mut [f32],
    ) -> Result<(), KernelError> {
        validate(shape, &params, q, k, v, out)?;
        let d = shape.head_dim;
        let mut scores: Vec<f32> = Vec::new();
        for h in 0..shape.heads {
            let base = h * shape.seq * d;
            for i in 0..shape.seq {
                let orow = &mut out[base + i * d..base + (i + 1) * d];
                orow.fill(0.0);
                if i >= params.valid_len {
                    continue;
                }
                let range = params.mask.key_range(i, params.valid_len);
                if range.is_empty() {
                    continue;
                }
                let qrow = &q[base + i * d..base + (i + 1) * d];
                scores.clear();
                let mut max = f32::NEG_INFINITY;
                for j in range.clone() {
                    let krow = &k[base + j * d..base + (j + 1) * d];
                    let mut s = 0.0f32;
                    for (a, b) in qrow.iter().zip(krow) {
                        s += a * b;
                    }
                    let s = s * params.scale;
                    max = max.max(s);
                    scores.push(s);
                }
                let mut sum = 0.0f32;
                for s in &mut scores {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                for (p, j) in scores.iter().zip(range) {
                    let vrow = &v[base + j * d..base + (j + 1) * d];
                    let w = p / sum;
                    for (o, x) in orow.iter_mut().zip(vrow) {
                        *o += w * x;
                    }
                }
            }
        }
        Ok(())
    }
}
