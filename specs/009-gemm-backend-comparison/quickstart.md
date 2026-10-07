# Quickstart: GEMM Backend Comparison

Nothing below can run until HD-1 is approved and tasks T003 to T011 are implemented. This file states how the comparison will be driven; at the time of writing no command here has been run.

```bash
# default build is unchanged and has no candidates
cargo build --workspace
cargo test --workspace

# correctness gates (after T010, T011): per shape and golden, any machine
cargo test -p raya-kernels --features gemm-candidates
cargo test -p raya-cli --features gemm-candidates

# timing (after T012 to T014); on the i7-8700 only the researcher sets the label
RAYA_TARGET_MACHINE=1 cargo bench -p raya-kernels --features gemm-candidates --bench gemm_compare
# result: results/gemm-compare/<timestamp>.json (git-ignored)
```

**Reading a result** (User Stories 2 and 3): open the JSON, check `label` and `metadata.cpu_model` against 6C/12T i7-8700, look for `flag` values, compare `ratio_to_reference` per shape, then read `docs/gemm-backends.md` for the budget table, the cost table and the decisions.

**Rules while specifying and reviewing**: no benchmark, timing or hardware run (human rule of 2026-10-07); no `Cargo.lock` change; the spec PR contains specification files only.
