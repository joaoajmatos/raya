#![allow(clippy::many_single_char_names, clippy::cast_precision_loss)]
//! `Full` / `Sliding` beyond the tiny cases of `masks_dense.rs`: the production window of 64,
//! `head_dim` 64, nothing valid, and one valid row.
use raya_kernels::{AttentionKernel, AttnParams, AttnShape, Mask, Naive};

fn fill(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect()
}

fn dense(shape: AttnShape, p: &AttnParams, q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
    let (s, d) = (shape.seq, shape.head_dim);
    let mut out = vec![0.0f32; shape.numel()];
    for h in 0..shape.heads {
        let base = h * s * d;
        for i in 0..p.valid_len {
            let mut scores = vec![f32::NEG_INFINITY; s];
            for (j, sc) in scores.iter_mut().enumerate() {
                if j < p.valid_len && p.mask.allows(i, j) {
                    let dot: f32 = (0..d)
                        .map(|c| q[base + i * d + c] * k[base + j * d + c])
                        .sum();
                    *sc = dot * p.scale;
                }
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<f32> = scores.iter().map(|x| (x - max).exp()).collect();
            let sum: f32 = exp.iter().sum();
            for c in 0..d {
                out[base + i * d + c] = (0..s).map(|j| exp[j] / sum * v[base + j * d + c]).sum();
            }
        }
    }
    out
}

fn check(mask: Mask, heads: usize, seq: usize, head_dim: usize, valid_len: usize) {
    let shape = AttnShape {
        heads,
        seq,
        head_dim,
    };
    let (q, k, v) = (
        fill(shape.numel(), 11),
        fill(shape.numel(), 12),
        fill(shape.numel(), 13),
    );
    let params = AttnParams::new(mask, valid_len, head_dim);
    let mut got = vec![9.0f32; shape.numel()];
    Naive.attend(shape, params, &q, &k, &v, &mut got).unwrap();
    let want = dense(shape, &params, &q, &k, &v);
    for (i, (a, b)) in got.iter().zip(&want).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "{mask:?} seq {seq} valid {valid_len} idx {i}: {a} vs {b}"
        );
    }
}

#[test]
fn the_production_sliding_window_matches_dense_softmax() {
    check(Mask::Sliding(64), 2, 200, 64, 200);
    check(Mask::Sliding(64), 2, 200, 64, 131);
    check(Mask::Sliding(64), 1, 64, 64, 64); // window as wide as the sequence
    check(Mask::Sliding(64), 1, 65, 32, 65);
    check(Mask::Full, 2, 150, 64, 150);
}

#[test]
fn nothing_valid_gives_all_zero_rows_for_full_and_sliding() {
    let shape = AttnShape {
        heads: 2,
        seq: 7,
        head_dim: 8,
    };
    let x = fill(shape.numel(), 21);
    for mask in [Mask::Full, Mask::Sliding(4), Mask::Sliding(0)] {
        let mut out = vec![1.0f32; shape.numel()];
        let params = AttnParams::new(mask, 0, 8);
        Naive.attend(shape, params, &x, &x, &x, &mut out).unwrap();
        assert!(out.iter().all(|&o| o == 0.0), "{mask:?}");
    }
}

#[test]
fn a_single_valid_row_attends_only_to_itself() {
    let shape = AttnShape {
        heads: 1,
        seq: 5,
        head_dim: 8,
    };
    let (q, k, v) = (fill(40, 31), fill(40, 32), fill(40, 33));
    for mask in [Mask::Full, Mask::Sliding(2)] {
        let mut out = vec![0.0f32; 40];
        let params = AttnParams::new(mask, 1, 8);
        Naive.attend(shape, params, &q, &k, &v, &mut out).unwrap();
        // softmax over one key is 1, so the output row is the value row
        for c in 0..8 {
            assert!((out[c] - v[c]).abs() < 1e-6, "{mask:?}");
        }
        assert!(out[8..].iter().all(|&o| o == 0.0));
    }
}
