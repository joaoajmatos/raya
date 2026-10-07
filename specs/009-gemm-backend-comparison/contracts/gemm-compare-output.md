# Contract: `results/gemm-compare/<timestamp>.json`

Git-ignored. Produced by the `gemm_compare` bench (`raya-kernels`, feature `gemm-candidates`). Fields:

```json
{
  "label": "<BenchMetadata.label, never hand-set>",
  "metadata": {
    "cpu_model": "", "logical_cpus": 0, "physical_cpus": 0, "features": [], "backend": "",
    "rustflags": "", "rustc": "", "git_commit": "", "timestamp": ""
  },
  "candidates": [
    {"name": "reference-loops|matrixmultiply|gemm|faer", "crate_version": "", "path": "scalar|avx2|avx512|neon|portable|...",
     "thread_switch": "how the thread count was set", "gates": {"per_shape": "pass|fail", "golden": "pass|fail|not-run", "detail": ""}}
  ],
  "protocol": {"warmup_calls": 3, "warmup_ms": 200, "min_samples": 15, "min_samples_large": 5, "passes": 2, "order": ["..."]},
  "results": [
    {"candidate": "", "shape": "enc_mlp_wi", "m": 512, "n": 5248, "k": 1024, "threads_requested": 1, "threads_effective": 1,
     "p50_ms": 0.0, "p90_ms": 0.0, "min_ms": 0.0, "samples": 0, "gflops_p50": 0.0, "ratio_to_reference": 0.0, "flag": null}
  ],
  "budget": [{"candidate": "", "threads": 1, "m": 512, "seconds": 0.0}],
  "notes": ["deviations from the protocol, power plan, background load as stated by the person running it"]
}
```

Rules:

- `label` is copied from `BenchMetadata`; the bench has no label option. A report quoting a number quotes the label with it.
- A candidate with `gates.per_shape` or `gates.golden` equal to `fail` has no entries in `results`.
- `ratio_to_reference` is `p50(candidate) / p50(reference-loops, same shape, 1 thread, same session)`; below 1 is faster.
- `flag` is `drift` when the two passes differ by more than 10%, `few-samples` when the sample count is below the minimum, else null.
- `threads_effective` is read back from the candidate, not assumed.
