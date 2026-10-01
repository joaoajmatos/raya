#![allow(clippy::many_single_char_names, clippy::similar_names)]
mod common;

use std::path::PathBuf;

use common::max_abs_diff;
use raya_kernels::{
    AttentionKernel, AttnParams, AttnShape, Backend, Mask, Naive, Tiled, TiledConfig,
};
use safetensors::SafeTensors;

/// Agreement bar from spec raya:007 SC-001 and the golden's own recorded tolerance.
const TOL: f32 = 1e-5;

fn golden_dir(case: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/data/kernels")
        .join(case)
}

fn tensor(st: &SafeTensors, name: &str) -> Vec<f32> {
    let view = st
        .tensor(name)
        .unwrap_or_else(|e| panic!("tensor `{name}`: {e}"));
    view.data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Laya's `block_local` kernel goldens (laya:004, vendored from laya-sparse): Q/K/V, the native
/// reference output and the `PyTorch` kernel output, shape `[1, 16, 256, 64]` (batch 1 is the
/// head-major `[heads][seq][head_dim]` layout used here). Only `Block3` is Laya's `block_local`;
/// `Window64` has no golden.
#[test]
fn block_local_goldens_match_naive_and_tiled() {
    let dir = golden_dir("block_local_L256");
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("meta.json")).expect("meta.json"))
            .expect("meta.json parses");
    assert_eq!(meta["pattern"], "block_local");
    assert_eq!(meta["scale"], "1/sqrt(head_dim)");
    let shape = AttnShape {
        heads: usize::try_from(meta["heads"].as_u64().unwrap()).unwrap(),
        seq: usize::try_from(meta["length"].as_u64().unwrap()).unwrap(),
        head_dim: usize::try_from(meta["head_dim"].as_u64().unwrap()).unwrap(),
    };
    assert_eq!(meta["block"].as_u64(), Some(128));

    let bytes = std::fs::read(dir.join("tensors.safetensors")).expect("tensors.safetensors");
    let st = SafeTensors::deserialize(&bytes).expect("valid safetensors");
    let (q, k, v) = (tensor(&st, "q"), tensor(&st, "k"), tensor(&st, "v"));
    let reference = tensor(&st, "reference");
    let kernel = tensor(&st, "kernel");
    for buf in [&q, &k, &v, &reference, &kernel] {
        assert_eq!(buf.len(), shape.numel());
    }

    let params = AttnParams::new(Mask::Block3, shape.seq, shape.head_dim);
    let mut naive = vec![0.0; shape.numel()];
    Naive.attend(shape, params, &q, &k, &v, &mut naive).unwrap();
    let tiled_kernel = Tiled::new(TiledConfig {
        threads: Some(2),
        backend: Backend::detect(),
    });
    let mut tiled = vec![0.0; shape.numel()];
    tiled_kernel
        .attend(shape, params, &q, &k, &v, &mut tiled)
        .unwrap();

    for (name, got) in [("naive", &naive), ("tiled", &tiled)] {
        for (golden, want) in [("reference", &reference), ("kernel", &kernel)] {
            let diff = max_abs_diff(got, want);
            assert!(
                diff < TOL,
                "{name} vs golden `{golden}`: max abs diff {diff}"
            );
        }
    }
}
