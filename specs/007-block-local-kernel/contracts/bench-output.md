# Contract: benchmark output

`cargo bench -p raya-kernels --bench local_attention` writes:

1. Criterion's standard report under `target/criterion/`.
2. `results/kernel-bench/<timestamp>.json`:

```json
{
  "label": "cloud-agent hardware, indicative only",
  "metadata": {"cpu_model": "", "logical_cpus": 0, "physical_cpus": 0, "threads": 1,
               "features": ["portable"], "rustflags": "", "rustc": "", "git_commit": "", "timestamp": ""},
  "results": [{"kernel": "naive|tiled", "mask": "block3|window64", "backend": "portable|avx2|neon", "seq": 2048, "heads": 16, "head_dim": 64, "threads": 1,
               "mean_ms": 0.0, "stddev_ms": 0.0, "samples": 0,
               "stages": {"qk_pct": 0.0, "softmax_pct": 0.0, "pv_pct": 0.0}}]
}
```

- `stages` is present only when built with `--features stage-timing` and is proportional, not absolute.
- Seq lengths 2048, 4096, 8192; thread counts 1 and the default; both kernels and both masks at each.
- The JSON never contains a comparison with 85 ms; the notes state that only `block3` is comparable with it, and only once run through feature 008 on the target machine.
