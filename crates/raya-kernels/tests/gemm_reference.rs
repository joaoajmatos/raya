#![allow(clippy::float_cmp)]
use raya_kernels::{GemmBackend, Reference};

#[test]
fn matmul_nt_matches_hand_computed_values() {
    // a [2,3] x b[2,3]^T
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [1.0, 0.0, -1.0, 2.0, 1.0, 0.5];
    let mut out = [0.0f32; 4];
    Reference.matmul_nt(&mut out, &a, &b, 2, 2, 3);
    assert_eq!(out, [-2.0, 5.5, -2.0, 16.0]);
}

#[test]
fn matmul_nt_overwrites_and_handles_empty_k() {
    let mut out = [5.0f32; 6];
    Reference.matmul_nt(&mut out, &[], &[], 2, 3, 0);
    assert_eq!(out, [0.0; 6]);
    assert_eq!(Reference.name(), "reference-loops");
}

#[test]
#[should_panic(expected = "a must be")]
fn matmul_nt_checks_lengths() {
    let mut out = [0.0f32; 1];
    Reference.matmul_nt(&mut out, &[1.0], &[1.0, 2.0], 1, 1, 2);
}
