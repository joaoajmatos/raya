#![allow(
    clippy::float_cmp,
    clippy::excessive_precision,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
use raya_core::ops::{erf, gelu, layer_norm, relu, rope_in_place, softmax_in_place};

#[test]
fn erf_matches_reference_values() {
    for (x, want) in [
        (0.0, 0.0),
        (0.5, 0.520_499_877_813_046_5),
        (1.0, 0.842_700_792_949_714_9),
        (2.0, 0.995_322_265_018_952_7),
        (2.5, 0.999_593_047_982_555),
        (3.0, 0.999_977_909_503_001_4),
        (4.0, 0.999_999_984_582_742_1),
    ] {
        assert!((erf(x) - want).abs() < 1e-12, "erf({x}) = {}", erf(x));
        assert!((erf(-x) + want).abs() < 1e-12, "erf(-{x})");
    }
    assert_eq!(erf(7.0), 1.0);
    assert_eq!(erf(-7.0), -1.0);
}

#[test]
fn erf_is_continuous_across_the_series_and_tail_switch() {
    let (lo, hi) = (erf(2.5 - 1e-9), erf(2.5 + 1e-9));
    assert!((hi - lo).abs() < 1e-8, "{lo} {hi}");
}

#[test]
fn gelu_is_exact_not_tanh() {
    assert_eq!(gelu(0.0), 0.0);
    assert!((gelu(1.0) - 0.841_344_75).abs() < 1e-7);
    assert!((gelu(-1.0) + 0.158_655_25).abs() < 1e-7);
    // the tanh approximation gives 0.841192 at 1.0
    assert!((gelu(1.0) - 0.841_192).abs() > 1e-4);
    assert!(gelu(-10.0).abs() < 1e-12 && (gelu(10.0) - 10.0).abs() < 1e-6);
    assert_eq!(relu(-2.0), 0.0);
    assert_eq!(relu(3.0), 3.0);
}

#[test]
fn layer_norm_uses_biased_variance_and_affine() {
    let x = [1.0, 2.0, 3.0, 4.0, 10.0, 10.0, 10.0, 10.0];
    let mut out = [0.0f32; 8];
    layer_norm(&mut out, &x, &[1.0; 4], None, 0.0, 4);
    for (g, w) in out[..4]
        .iter()
        .zip([-1.341_640_8, -0.447_213_6, 0.447_213_6, 1.341_640_8])
    {
        assert!((g - w).abs() < 1e-6, "{g} {w}");
    }
    // a constant row has zero variance: with eps it stays finite and is the bias
    layer_norm(&mut out, &x, &[2.0; 4], Some(&[0.5; 4]), 1e-5, 4);
    assert!(out[4..].iter().all(|&v| (v - 0.5).abs() < 1e-6));
}

#[test]
fn softmax_is_shift_invariant_and_sums_to_one() {
    let mut a = [1.0f32, 2.0, 3.0];
    let mut b = [1001.0f32, 1002.0, 1003.0];
    softmax_in_place(&mut a);
    softmax_in_place(&mut b);
    assert!((a.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    for (x, y) in a.iter().zip(&b) {
        assert!((x - y).abs() < 1e-6);
    }
    let mut masked = [0.5f32, -1e4];
    softmax_in_place(&mut masked);
    assert_eq!(masked, [1.0, 0.0]);
}

#[test]
fn rope_rotates_by_position_in_rotate_half_form() {
    // one head, head_dim 4, positions 0 and 1
    let mut x = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    rope_in_place(&mut x, 1, 2, 4, 10_000.0);
    assert_eq!(&x[..4], &[1.0, 0.0, 0.0, 0.0], "position 0 is the identity");
    // dim pair (0, 2) rotates by 1 rad (inv_freq 1), pair (1, 3) by 0.01 rad
    assert!((x[4] - 1.0f32.cos()).abs() < 1e-6);
    assert!((x[6] - 1.0f32.sin()).abs() < 1e-6);
    assert!(x[5].abs() < 1e-7 && x[7].abs() < 1e-7);
    let mut y = [0.0f32, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    rope_in_place(&mut y, 1, 2, 4, 10_000.0);
    assert!((y[5] - 0.01f32.cos()).abs() < 1e-6 && (y[7] - 0.01f32.sin()).abs() < 1e-6);
    // rotation preserves the norm
    let n: f32 = x[4..].iter().map(|v| v * v).sum();
    assert!((n - 1.0).abs() < 1e-6);
}
