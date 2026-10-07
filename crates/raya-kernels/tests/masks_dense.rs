#![allow(clippy::many_single_char_names, clippy::cast_precision_loss)]
//! `Full` and `Sliding` masks (feature 006) against a dense masked softmax written out here.
use raya_kernels::{AttentionKernel, AttnParams, AttnShape, KernelError, Mask, Naive, Tiled};

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

fn check(mask: Mask, seq: usize, valid_len: usize) {
    let shape = AttnShape {
        heads: 2,
        seq,
        head_dim: 16,
    };
    let (q, k, v) = (
        fill(shape.numel(), 1),
        fill(shape.numel(), 2),
        fill(shape.numel(), 3),
    );
    let params = AttnParams::new(mask, valid_len, shape.head_dim);
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
fn full_and_sliding_match_dense_softmax() {
    for mask in [
        Mask::Full,
        Mask::Sliding(8),
        Mask::Sliding(0),
        Mask::Sliding(500),
    ] {
        check(mask, 40, 40);
        check(mask, 40, 23);
        check(mask, 3, 1);
    }
}

#[test]
fn padded_rows_are_zero() {
    let shape = AttnShape {
        heads: 1,
        seq: 10,
        head_dim: 8,
    };
    let x = fill(shape.numel(), 7);
    let mut out = vec![1.0f32; shape.numel()];
    let params = AttnParams::new(Mask::Full, 6, 8);
    Naive.attend(shape, params, &x, &x, &x, &mut out).unwrap();
    assert!(out[6 * 8..].iter().all(|&o| o == 0.0));
}

#[test]
fn sliding_window_edges() {
    assert!(Mask::Sliding(8).allows(20, 12) && Mask::Sliding(8).allows(20, 28));
    assert!(!Mask::Sliding(8).allows(20, 11) && !Mask::Sliding(8).allows(20, 29));
    assert_eq!(Mask::Sliding(8).key_range(2, 40), 0..11);
    assert_eq!(Mask::Full.key_range(5, 7), 0..7);
}

#[test]
fn tiled_rejects_the_reference_only_masks() {
    let shape = AttnShape {
        heads: 1,
        seq: 8,
        head_dim: 8,
    };
    let x = fill(shape.numel(), 5);
    let mut out = vec![0.0f32; shape.numel()];
    for mask in [Mask::Full, Mask::Sliding(4)] {
        let params = AttnParams::new(mask, 8, 8);
        let err = Tiled::default()
            .attend(shape, params, &x, &x, &x, &mut out)
            .unwrap_err();
        assert!(matches!(err, KernelError::Unsupported(_)), "{err}");
    }
}
