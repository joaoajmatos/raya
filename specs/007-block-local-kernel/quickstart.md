# Quickstart: validate feature 007

Prerequisites: Rust 1.94.1 via `rust-toolchain.toml`.

```bash
cargo build --workspace
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p raya-kernels                       # US1: correctness, SC-001
cargo test -p raya-kernels --features avx2       # US3: only where AVX2 exists
```

**Correctness** (User Story 1): `cargo test -p raya-kernels --test equivalence` covers lengths 1, 127, 128, 129, 1000, 2048 and a padded tail; `--test masking` checks both masks exactly (`Block3` and `Window64`) and all-masked rows; `--test goldens` runs if 002 fixtures exist.

**Benchmarks** (User Story 2, SC-002):

```bash
cargo bench -p raya-kernels --bench local_attention
RAYA_THREADS=1 cargo bench -p raya-kernels --bench local_attention
cargo bench -p raya-kernels --bench local_attention --features stage-timing   # breakdown
```

Expected: `results/kernel-bench/*.json` with metadata and naive/tiled at 2048, 4096, 8192, each for `block3` and `window64`.

**Conclusion** (User Story 3, SC-003, SC-004): read `docs/kernel-notes.md`; it states whether tiled is at least 3x faster than naive at 8192 tokens, single-threaded, for each mask, and labels all numbers "cloud-agent hardware, indicative only".
