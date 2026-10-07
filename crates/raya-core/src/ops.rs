//! Scalar building blocks of the forward pass: plain loops, fp32 storage.
use raya_kernels::GemmBackend;

/// Error function in f64 (Rust's std has none): Taylor series for small `|x|`, continued
/// fraction for `erfc` in the tail, saturated beyond 6.
#[must_use]
pub fn erf(x: f64) -> f64 {
    let ax = x.abs();
    let value = if ax >= 6.0 {
        1.0
    } else if ax < 2.5 {
        let (mut term, mut sum) = (ax, ax);
        let x2 = ax * ax;
        let mut n = 0u32;
        while term.abs() > 1e-18 * sum.abs().max(1e-300) {
            n += 1;
            term *= -x2 / f64::from(n);
            sum += term / f64::from(2 * n + 1);
        }
        sum * std::f64::consts::FRAC_2_SQRT_PI
    } else {
        // erfc(x) = exp(-x^2) / sqrt(pi) / (x + (1/2) / (x + (2/2) / (x + (3/2) / ...)))
        let mut t = ax;
        for n in (1..=120u32).rev() {
            t = ax + f64::from(n) / 2.0 / t;
        }
        1.0 - (-ax * ax).exp() / std::f64::consts::PI.sqrt() / t
    };
    value.copysign(x)
}

/// Exact GELU, `x * Phi(x)`, as `nn.GELU()` and `ACT2FN["gelu"]` compute it.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn gelu(x: f32) -> f32 {
    let x = f64::from(x);
    (0.5 * x * (1.0 + erf(x * std::f64::consts::FRAC_1_SQRT_2))) as f32
}

#[must_use]
pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Row-wise `LayerNorm` over the last axis (`d`): `(x - mean) / sqrt(var + eps) * weight + bias`
/// with the biased variance.
///
/// # Panics
/// If `x.len()` is not a multiple of `d` or `out` and the parameters do not fit.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub fn layer_norm(
    out: &mut [f32],
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    eps: f32,
    d: usize,
) {
    assert!(d > 0 && x.len().is_multiple_of(d) && out.len() == x.len());
    assert_eq!(weight.len(), d);
    for (row, dst) in x.chunks(d).zip(out.chunks_mut(d)) {
        let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / d as f64;
        let var = row
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / d as f64;
        let inv = 1.0 / (var + f64::from(eps)).sqrt();
        for (i, (o, &v)) in dst.iter_mut().zip(row).enumerate() {
            let y = ((f64::from(v) - mean) * inv) as f32 * weight[i];
            *o = bias.map_or(y, |b| y + b[i]);
        }
    }
}

/// Max-shifted softmax in place.
pub fn softmax_in_place(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

/// `out = x * w^T + bias` for `x: [rows, in_f]`, `w: [out_f, in_f]`, `out: [rows, out_f]`.
#[allow(clippy::too_many_arguments)]
pub fn linear(
    gemm: &dyn GemmBackend,
    out: &mut [f32],
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    rows: usize,
    in_f: usize,
    out_f: usize,
) {
    gemm.matmul_nt(out, x, w, rows, out_f, in_f);
    if let Some(b) = bias {
        for row in out.chunks_mut(out_f) {
            for (o, bv) in row.iter_mut().zip(b) {
                *o += bv;
            }
        }
    }
}

/// Rotary embedding on a head-major `[heads][seq][head_dim]` buffer, positions `0..seq`,
/// in the rotate-half form: `x * cos + rotate_half(x) * sin`. The angles are computed in f32 as
/// the reference does (`inv_freq = 1 / theta^(2i / head_dim)`, `angle = position * inv_freq`).
///
/// # Panics
/// If the buffer length is not `heads * seq * head_dim` or `head_dim` is odd.
#[allow(clippy::cast_precision_loss)]
pub fn rope_in_place(x: &mut [f32], heads: usize, seq: usize, head_dim: usize, theta: f32) {
    assert!(head_dim.is_multiple_of(2) && x.len() == heads * seq * head_dim);
    let half = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1.0 / theta.powf((2 * i) as f32 / head_dim as f32))
        .collect();
    let mut cos = vec![0.0f32; half];
    let mut sin = vec![0.0f32; half];
    for pos in 0..seq {
        for i in 0..half {
            let angle = pos as f32 * inv_freq[i];
            cos[i] = angle.cos();
            sin[i] = angle.sin();
        }
        for h in 0..heads {
            let row = &mut x[(h * seq + pos) * head_dim..(h * seq + pos + 1) * head_dim];
            for i in 0..half {
                let (a, b) = (row[i], row[i + half]);
                row[i] = a * cos[i] - b * sin[i];
                row[i + half] = b * cos[i] + a * sin[i];
            }
        }
    }
}
