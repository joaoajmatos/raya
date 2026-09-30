#![allow(dead_code)]

use raya_kernels::{AttnParams, AttnShape, Mask};

/// Deterministic xorshift values uniform in [-1, 1].
pub fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            #[allow(clippy::cast_precision_loss)]
            let u = (s >> 40) as f32 / (1u64 << 24) as f32;
            2.0 * u - 1.0
        })
        .collect()
}

pub fn qkv(shape: AttnShape, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = shape.numel();
    (
        rand_vec(n, seed),
        rand_vec(n, seed + 1),
        rand_vec(n, seed + 2),
    )
}

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Independent dense reference: full `[seq, seq]` scores, boolean mask (as `reference.py`),
/// softmax, zeros for rows without an allowed key. Only for small `seq`.
pub fn dense_reference(
    shape: AttnShape,
    params: &AttnParams,
    q: &[f32],
    k: &[f32],
    v: &[f32],
) -> Vec<f32> {
    let (s, d) = (shape.seq, shape.head_dim);
    let mut out = vec![0.0f32; shape.numel()];
    for h in 0..shape.heads {
        let base = h * s * d;
        for i in 0..s {
            let allowed = |j: usize| {
                j < params.valid_len
                    && i < params.valid_len
                    && match params.mask {
                        Mask::Block3 => (i / 128).abs_diff(j / 128) <= 1,
                        Mask::Window64 => i.abs_diff(j) <= 64,
                    }
            };
            let mut scores = vec![f32::NEG_INFINITY; s];
            for (j, sc) in scores.iter_mut().enumerate() {
                if allowed(j) {
                    let dot: f32 = (0..d)
                        .map(|c| q[base + i * d + c] * k[base + j * d + c])
                        .sum();
                    *sc = dot * params.scale;
                }
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if max == f32::NEG_INFINITY {
                continue;
            }
            let exps: Vec<f32> = scores.iter().map(|x| (x - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for (j, e) in exps.iter().enumerate() {
                for c in 0..d {
                    out[base + i * d + c] += e / sum * v[base + j * d + c];
                }
            }
        }
    }
    out
}
