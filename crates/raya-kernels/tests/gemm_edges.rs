#![allow(clippy::float_cmp)]
use raya_kernels::{GemmBackend, Reference};

#[test]
#[should_panic(expected = "b must be")]
fn matmul_nt_checks_the_b_length() {
    let mut out = [0.0f32; 1];
    Reference.matmul_nt(&mut out, &[1.0, 2.0], &[1.0], 1, 1, 2);
}

#[test]
#[should_panic(expected = "out must be")]
fn matmul_nt_checks_the_out_length() {
    let mut out = [0.0f32; 2];
    Reference.matmul_nt(&mut out, &[1.0, 2.0], &[1.0, 2.0], 1, 1, 2);
}

#[test]
fn matmul_nt_with_an_empty_m_or_n_touches_nothing() {
    let mut none: [f32; 0] = [];
    Reference.matmul_nt(&mut none, &[], &[1.0, 2.0], 0, 1, 2); // m = 0
    Reference.matmul_nt(&mut none, &[1.0, 2.0], &[], 1, 0, 2); // n = 0
    Reference.matmul_nt(&mut none, &[], &[], 0, 0, 0);
}

#[test]
fn matmul_nt_propagates_nan_and_infinity() {
    let mut out = [0.0f32; 2];
    Reference.matmul_nt(&mut out, &[f32::NAN, 1.0], &[1.0; 4], 1, 2, 2);
    assert!(out.iter().all(|v| v.is_nan()));
    Reference.matmul_nt(
        &mut out,
        &[f32::INFINITY, 1.0],
        &[1.0, 0.0, -1.0, 0.0],
        1,
        2,
        2,
    );
    assert_eq!(out, [f32::INFINITY, f32::NEG_INFINITY]);
    // inf * 0 is NaN
    Reference.matmul_nt(
        &mut out,
        &[f32::INFINITY, 1.0],
        &[0.0, 1.0, 1.0, 1.0],
        1,
        2,
        2,
    );
    assert!(out[0].is_nan());
    assert_eq!(out[1], f32::INFINITY);
}
