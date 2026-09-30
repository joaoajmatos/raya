mod common;

use common::{dense_reference, max_abs_diff, qkv};
use raya_kernels::{AttentionKernel, AttnParams, AttnShape, KernelError, Mask, Naive};

#[test]
fn naive_matches_dense_reference() {
    for mask in [Mask::Block3, Mask::Window64] {
        for seq in [1usize, 7, 127, 128, 129, 257] {
            for (heads, head_dim) in [(1, 8), (3, 64)] {
                for valid in [seq, seq.saturating_sub(37), seq / 2] {
                    let shape = AttnShape { heads, seq, head_dim };
                    let params = AttnParams::new(mask, valid, head_dim);
                    let (q, k, v) = qkv(shape, 11 + seq as u64);
                    let mut out = vec![9.0; shape.numel()];
                    Naive.attend(shape, params, &q, &k, &v, &mut out).unwrap();
                    let want = dense_reference(shape, &params, &q, &k, &v);
                    let diff = max_abs_diff(&out, &want);
                    assert!(diff < 1e-6, "{mask:?} seq {seq} valid {valid} h{heads} d{head_dim}: {diff}");
                }
            }
        }
    }
}

#[test]
fn bad_arguments_are_errors_not_panics() {
    let shape = AttnShape { heads: 1, seq: 4, head_dim: 8 };
    let ok = vec![0.0; shape.numel()];
    let mut out = vec![0.0; shape.numel()];
    let p = AttnParams::new(Mask::Block3, 4, 8);

    let short = vec![0.0; 3];
    assert!(matches!(Naive.attend(shape, p, &short, &ok, &ok, &mut out), Err(KernelError::Length { name: "q", .. })));
    let mut short_out = vec![0.0; 3];
    assert!(matches!(Naive.attend(shape, p, &ok, &ok, &ok, &mut short_out), Err(KernelError::Length { name: "out", .. })));

    let bad_dim = AttnShape { head_dim: 12, ..shape };
    let buf = vec![0.0; bad_dim.numel()];
    let mut o = vec![0.0; bad_dim.numel()];
    assert!(matches!(Naive.attend(bad_dim, p, &buf, &buf, &buf, &mut o), Err(KernelError::Shape(_))));

    let too_long = AttnParams { valid_len: 5, ..p };
    assert!(matches!(Naive.attend(shape, too_long, &ok, &ok, &ok, &mut out), Err(KernelError::Shape(_))));

    let bad_block = AttnParams { block: 64, ..p };
    assert!(matches!(Naive.attend(shape, bad_block, &ok, &ok, &ok, &mut out), Err(KernelError::Unsupported(_))));
}
