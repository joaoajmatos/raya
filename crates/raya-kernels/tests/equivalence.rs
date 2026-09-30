#![allow(clippy::many_single_char_names, clippy::similar_names)]
mod common;

use common::{max_abs_diff, qkv};
use raya_kernels::{
    AttentionKernel, AttnParams, AttnShape, Backend, Mask, Naive, Tiled, TiledConfig,
};

fn check(mask: Mask, shape: AttnShape, valid: usize, threads: usize) -> Vec<f32> {
    let params = AttnParams::new(mask, valid, shape.head_dim);
    let (q, k, v) = qkv(shape, 100 + shape.seq as u64);
    let mut want = vec![0.0; shape.numel()];
    Naive.attend(shape, params, &q, &k, &v, &mut want).unwrap();
    let tiled = Tiled::new(TiledConfig {
        threads: Some(threads),
        backend: Backend::detect(),
    });
    let mut got = vec![7.0; shape.numel()];
    tiled.attend(shape, params, &q, &k, &v, &mut got).unwrap();
    let diff = max_abs_diff(&got, &want);
    assert!(
        diff < 1e-5,
        "{mask:?} seq {} valid {valid} threads {threads}: diff {diff}",
        shape.seq
    );
    got
}

#[test]
fn tiled_matches_naive_at_edge_lengths() {
    for mask in [Mask::Block3, Mask::Window64] {
        for seq in [1usize, 127, 128, 129, 1000, 2048] {
            let heads = if seq >= 1000 { 2 } else { 3 };
            let shape = AttnShape {
                heads,
                seq,
                head_dim: 64,
            };
            check(mask, shape, seq, 1);
        }
    }
}

#[test]
fn tiled_matches_naive_with_padded_tail() {
    for mask in [Mask::Block3, Mask::Window64] {
        for (seq, valid) in [
            (300usize, 263usize),
            (1000, 963),
            (256, 1),
            (256, 0),
            (129, 128),
        ] {
            check(
                mask,
                AttnShape {
                    heads: 2,
                    seq,
                    head_dim: 64,
                },
                valid,
                1,
            );
        }
    }
}

#[test]
fn tiled_matches_naive_with_sixteen_heads_and_other_head_dims() {
    for mask in [Mask::Block3, Mask::Window64] {
        check(
            mask,
            AttnShape {
                heads: 16,
                seq: 512,
                head_dim: 64,
            },
            512,
            2,
        );
        check(
            mask,
            AttnShape {
                heads: 2,
                seq: 300,
                head_dim: 8,
            },
            300,
            1,
        );
        check(
            mask,
            AttnShape {
                heads: 2,
                seq: 300,
                head_dim: 40,
            },
            300,
            1,
        );
        check(
            mask,
            AttnShape {
                heads: 2,
                seq: 300,
                head_dim: 96,
            },
            300,
            1,
        );
    }
}

#[test]
fn thread_count_does_not_change_results() {
    for mask in [Mask::Block3, Mask::Window64] {
        let shape = AttnShape {
            heads: 4,
            seq: 700,
            head_dim: 64,
        };
        let one = check(mask, shape, 650, 1);
        let four = check(mask, shape, 650, 4);
        assert_eq!(
            one, four,
            "{mask:?}: results differ between 1 and 4 threads"
        );
    }
}

#[test]
#[ignore = "slow in debug builds: run with --release -- --ignored"]
fn tiled_matches_naive_at_long_lengths() {
    for mask in [Mask::Block3, Mask::Window64] {
        for seq in [4096usize, 8192] {
            check(
                mask,
                AttnShape {
                    heads: 1,
                    seq,
                    head_dim: 64,
                },
                seq,
                2,
            );
        }
    }
}
