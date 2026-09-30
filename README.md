# Raya

Research runtime: Rust CPU inference for Laya-style encoder-only decision models (Laya shape, fp32,
inference only). Scope, measurements and open questions: [docs/feasibility.md](docs/feasibility.md).
Work is specified in [specs/](specs/README.md) (Spec Kit; rules in
[.specify/memory/constitution.md](.specify/memory/constitution.md)).

**Status:** the workspace, CI and configuration types exist ([specs/001](specs/001-workspace-config/spec.md)).
The block-local attention kernel spike ([specs/007](specs/007-block-local-kernel/spec.md)) is implemented in
`raya-kernels`; results and conclusion are in [docs/kernel-notes.md](docs/kernel-notes.md). There is no model
math yet; the `raya` subcommands are stubs.

## Scope limits

- Raya-L only: Laya's exact shape, fp32, CPU inference, loading Laya's safetensors.
- No compression, architecture variants, training, or int8/fp16/bf16 paths.
- No SIMD target is chosen: the portable path is the default and AVX2/NEON stay optional.
- Laya (`reference/laya-sparse/`, a read-only git submodule) is the specification: match it, do not improve it.

## Layout

| Crate | Purpose |
|---|---|
| `raya-core` | configuration types, errors, model shell |
| `raya-kernels` | attention and GEMM kernels (the only crate allowed `unsafe`) |
| `raya-weights` | safetensors loading |
| `raya-tokenizer` | sequence builder over the Hugging Face tokenizer |
| `raya-cli` | the `raya` binary |

## Build and test

Rust 1.94.1 is pinned by `rust-toolchain.toml` and installed by rustup on first use.

```bash
git submodule update --init      # only needed to read the Laya reference
cargo build --workspace
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Kernel benchmarks: `cargo bench -p raya-kernels --bench local_attention` (see `docs/kernel-notes.md` for options
and how to read the results; off-target numbers are indicative only).

CI runs formatting, clippy, tests on Linux x86_64 and macOS arm64, and `cargo audit`. Windows builds are
supported but not part of CI.
