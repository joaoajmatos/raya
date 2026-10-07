# Contract: benchmark output

`cargo bench -p raya-kernels --bench local_attention` writes:

1. Criterion's standard report under `target/criterion/`.
2. `results/kernel-bench/<timestamp>.json`:

```json
{
  "label": "unlabeled machine, indicative only",  // or "target machine" / "cloud-agent hardware, indicative only", per RAYA_TARGET_MACHINE
  "metadata": {"cpu_model": "", "logical_cpus": 0, "physical_cpus": 0, "threads": 1,
               "features": ["portable"], "backend": "portable|avx2|neon", "rustflags": "", "rustc": "", "git_commit": "", "timestamp": ""},
  "results": [{"kernel": "naive|tiled", "mask": "block3|window64", "backend": "scalar|portable|avx2|neon", "seq": 2048, "heads": 16, "head_dim": 64, "threads": 1,
               "mean_ms": 0.0, "stddev_ms": 0.0, "samples": 0,
               "stages": {"qk_pct": 0.0, "softmax_pct": 0.0, "pv_pct": 0.0}}]
}
```

- `metadata.backend` is the micro-kernel backend the run used: the detected one, or the one forced with `RAYA_BACKEND` (portable if that one is unavailable); `tiled` result rows repeat it (`naive` rows say `scalar`). `features` lists what is compiled in (`avx2` on every x86_64 build), not what ran.
- `stages` is present only when built with `--features stage-timing` and is proportional, not absolute.
- Seq lengths 2048, 4096, 8192; thread counts 1 and the default; both kernels and both masks at each.
- The JSON never contains a comparison with 85 ms; the notes state that only `block3` is comparable with it, and only once run through feature 008 on the target machine.
