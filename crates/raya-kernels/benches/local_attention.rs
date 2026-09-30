//! One-layer block-local attention timings: naive and tiled, both masks, 16 heads, head dim 64.
//!
//! `RAYA_BENCH_SEQS` (comma separated) overrides the default lengths 2048,4096,8192.
//! Writes `results/kernel-bench/<timestamp>.json` per `specs/007-block-local-kernel/contracts/bench-output.md`.

#[path = "../tests/common/mod.rs"]
mod common;

use std::path::{Path, PathBuf};
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, SamplingMode};
use raya_kernels::meta::BenchMetadata;
use raya_kernels::threads::default_threads;
use raya_kernels::{AttentionKernel, AttnParams, AttnShape, Backend, Mask, Naive, Tiled, TiledConfig};
use serde_json::json;

const HEADS: usize = 16;
const HEAD_DIM: usize = 64;
const SAMPLES: usize = 10;

struct Entry {
    kernel: &'static str,
    mask: Mask,
    backend: &'static str,
    seq: usize,
    threads: usize,
    func: String,
    stages: Option<(f64, f64, f64)>,
}

fn seqs() -> Vec<usize> {
    std::env::var("RAYA_BENCH_SEQS")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .filter(|v: &Vec<usize>| !v.is_empty())
        .unwrap_or_else(|| vec![2048, 4096, 8192])
}

fn estimate(dir: &Path, func: &str, seq: usize) -> Option<(f64, f64)> {
    let path = dir.join("local_attention").join(func).join(seq.to_string()).join("new/estimates.json");
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(path).ok()?).ok()?;
    Some((v["mean"]["point_estimate"].as_f64()?, v["std_dev"]["point_estimate"].as_f64()?))
}

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let criterion_dir = root.join("target/criterion");
    let mut c = Criterion::default().output_directory(&criterion_dir).configure_from_args();
    let default_t = default_threads();
    let backend = Backend::detect();
    let mut entries: Vec<Entry> = Vec::new();

    let mut group = c.benchmark_group("local_attention");
    group
        .sampling_mode(SamplingMode::Flat)
        .sample_size(SAMPLES)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5));

    for seq in seqs() {
        let shape = AttnShape { heads: HEADS, seq, head_dim: HEAD_DIM };
        let (q, k, v) = common::qkv(shape, 42);
        let mut out = vec![0.0f32; shape.numel()];
        for mask in [Mask::Block3, Mask::Window64] {
            let params = AttnParams::new(mask, seq, HEAD_DIM);

            let func = format!("naive_{}_t1", mask.name());
            group.bench_with_input(BenchmarkId::new(&func, seq), &seq, |b, _| {
                b.iter(|| Naive.attend(shape, params, &q, &k, &v, &mut out).unwrap());
            });
            entries.push(Entry { kernel: "naive", mask, backend: "scalar", seq, threads: 1, func, stages: None });

            let mut thread_counts = vec![1];
            if default_t != 1 {
                thread_counts.push(default_t);
            }
            for threads in thread_counts {
                let tiled = Tiled::new(TiledConfig { threads: Some(threads), backend });
                let func = format!("tiled_{}_{}_t{threads}", backend.name(), mask.name());
                group.bench_with_input(BenchmarkId::new(&func, seq), &seq, |b, _| {
                    b.iter(|| tiled.attend(shape, params, &q, &k, &v, &mut out).unwrap());
                });
                let stages = tiled
                    .attend_with_stages(shape, params, &q, &k, &v, &mut out)
                    .unwrap()
                    .percentages();
                entries.push(Entry { kernel: "tiled", mask, backend: backend.name(), seq, threads, func, stages });
            }
        }
    }
    group.finish();
    c.final_summary();

    let results: Vec<_> = entries
        .iter()
        .filter_map(|e| {
            let (mean, sd) = estimate(&criterion_dir, &e.func, e.seq)?;
            let mut row = json!({
                "kernel": e.kernel, "mask": e.mask.name(), "backend": e.backend,
                "seq": e.seq, "heads": HEADS, "head_dim": HEAD_DIM, "threads": e.threads,
                "mean_ms": mean / 1e6, "stddev_ms": sd / 1e6, "samples": SAMPLES,
            });
            if let Some((qk, sm, pv)) = e.stages {
                row["stages"] = json!({"qk_pct": qk, "softmax_pct": sm, "pv_pct": pv});
            }
            Some(row)
        })
        .collect();
    if results.is_empty() {
        eprintln!("no Criterion estimates found (filtered run?); not writing JSON");
        return;
    }
    let meta = BenchMetadata::collect(default_t);
    let doc = json!({"label": meta.label, "metadata": meta, "results": results});
    let dir = root.join("results/kernel-bench");
    std::fs::create_dir_all(&dir).expect("create results dir");
    let path = dir.join(format!("{}.json", meta.timestamp.replace(':', "-")));
    std::fs::write(&path, serde_json::to_vec_pretty(&doc).expect("json")).expect("write results");
    println!("wrote {}", path.display());
}
