mod common;

use common::{max_abs_diff, qkv, rand_vec};
use raya_kernels::{AttentionKernel, AttnParams, AttnShape, Backend, Mask, Naive, Tiled, TiledConfig};

fn kernels() -> Vec<Box<dyn AttentionKernel>> {
    vec![
        Box::new(Naive),
        Box::new(Tiled::new(TiledConfig { threads: Some(2), backend: Backend::detect() })),
    ]
}

/// Query `i` depends on key `j` exactly when the mask allows `(i, j)` and `j < valid_len`.
#[test]
fn masks_are_exact() {
    let shape = AttnShape { heads: 1, seq: 300, head_dim: 8 };
    let n = shape.seq;
    for mask in [Mask::Block3, Mask::Window64] {
        let params = AttnParams::new(mask, n, 8);
        let q = rand_vec(shape.numel(), 5);
        let k = rand_vec(shape.numel(), 6);
        let zeros = vec![0.0; shape.numel()];
        for kernel in kernels() {
            // Put weight only on the V row of key `j`: out[i][0] is then p(i, j) * 1.
            for j in (0..n).filter(|j| [0, 1, 63, 64, 65, 127, 128, 129, 191, 255, 256, 299].contains(j)) {
                let mut v = zeros.clone();
                v[j * 8] = 1.0;
                let mut out = vec![0.0; shape.numel()];
                kernel.attend(shape, params, &q, &k, &v, &mut out).unwrap();
                for i in 0..n {
                    let weight = out[i * 8];
                    if mask.allows(i, j) {
                        assert!(weight > 0.0, "{} {mask:?}: query {i} should see key {j}", kernel.name());
                    } else {
                        assert!(weight == 0.0, "{} {mask:?}: query {i} must not see key {j}", kernel.name());
                    }
                }
            }
        }
    }
}

#[test]
fn padding_and_empty_rows_are_exact_zeros() {
    let shape = AttnShape { heads: 2, seq: 200, head_dim: 16 };
    let (q, k, v) = qkv(shape, 9);
    for mask in [Mask::Block3, Mask::Window64] {
        for valid in [0usize, 1, 150] {
            let params = AttnParams::new(mask, valid, 16);
            for kernel in kernels() {
                let mut out = vec![1.0; shape.numel()];
                kernel.attend(shape, params, &q, &k, &v, &mut out).unwrap();
                for h in 0..2 {
                    for i in valid..shape.seq {
                        let row = &out[(h * shape.seq + i) * 16..(h * shape.seq + i + 1) * 16];
                        assert!(row.iter().all(|&x| x == 0.0), "{} {mask:?} valid {valid} row {i}", kernel.name());
                    }
                }
                assert!(out.iter().all(|x| x.is_finite()));
            }
        }
    }
}

#[test]
fn extreme_logits_are_stable_and_match_the_oracle() {
    let shape = AttnShape { heads: 2, seq: 300, head_dim: 64 };
    let (mut q, mut k, v) = qkv(shape, 21);
    for x in &mut q {
        *x *= 400.0;
    }
    for x in &mut k {
        *x *= 400.0;
    }
    for mask in [Mask::Block3, Mask::Window64] {
        let params = AttnParams::new(mask, 300, 64);
        let mut want = vec![0.0; shape.numel()];
        Naive.attend(shape, params, &q, &k, &v, &mut want).unwrap();
        assert!(want.iter().all(|x| x.is_finite()), "oracle produced non-finite values");
        let mut got = vec![0.0; shape.numel()];
        kernels()[1].attend(shape, params, &q, &k, &v, &mut got).unwrap();
        assert!(got.iter().all(|x| x.is_finite()), "{mask:?}: tiled produced non-finite values");
        let diff = max_abs_diff(&got, &want);
        assert!(diff < 1e-5, "{mask:?}: diff {diff}");
    }
}
