# Implementation Plan: Rust Workspace, CI and Config Types

**Branch**: `001-workspace-config` | **Date**: 2026-09-30 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/001-workspace-config/spec.md`

## Summary

Create the Cargo workspace (five crates), CI, and the typed, validated model configuration. Config comes from
the two files a Laya checkpoint directory holds: `encoder/config.json` (a Hugging Face `ModernBertConfig`)
and `rl_agent_config.json` (Laya's decision-head and agent settings). They are merged into a `ModelConfig`
and converted into the future-facing `ModelShape` (segments, attention kind, compression), of which only
Laya's own shape is supported. No model math; CLI subcommands are stubs.

## Technical Context

**Language/Version**: Rust stable, edition 2021, pinned to 1.94.1 in `rust-toolchain.toml`

**Primary Dependencies**: serde, serde_json, thiserror (raya-core); safetensors (raya-weights); tokenizers (raya-tokenizer); clap, anyhow (raya-cli). No others (spec FR-007); criterion is deferred to feature 007 where benches first exist.

**Storage**: JSON config files on disk; checked-in fixture under `tests/data/fixture-tiny/` (now `fixture-h64/` and `fixture-h128/`, vendored from laya-sparse)

**Testing**: `cargo test` (unit tests in `raya-core`, process-level tests for the CLI stubs, a source-level check of the `forbid(unsafe_code)` rule)

**Target Platform**: Linux x86_64 and macOS arm64 in CI; Windows supported, not in CI

**Project Type**: Cargo workspace: four libraries and one binary

**Performance Goals**: N/A (no computation in this feature)

**Constraints**: portable (no SIMD target); `unsafe` only in `raya-kernels`; unsupported shapes return `Error::Unimplemented`

**Scale/Scope**: Config parsing for one architecture family (ModernBERT-style encoder plus Laya head); one real fixture config

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

| Principle | Status |
|---|---|
| I. Laya is the specification | Pass. Field names, the global-layer rule and the head sizing are taken from `reference/laya-sparse/laya/common.py` and a real `ModernBertConfig` dump (research R1, R4). |
| II. Parity via golden fixtures | N/A here; the fixture config is generated from the real library, not hand-written, so later parity features inherit real values. |
| III. Measure, do not assume | N/A (no measurements). |
| IV. Raya-L, fp32, inference only | Pass. Only Laya's shape is constructible; everything else is `Unimplemented`. |
| V. Portable first; `unsafe` only in kernels | Pass. `forbid(unsafe_code)` in four crates, checked by a test; no SIMD. |
| VI. Reference before kernel | Pass. Kernel crate holds the trait only. |
| VII. Small surface, explicit errors | Pass. Dependencies match FR-007; errors name the field. |

Post-design re-check: still pass. One deviation from the spec text is recorded in research R3 and in the
spec's Clarifications (an added `AttentionKind` variant).

## Project Structure

### Documentation (this feature)

```text
specs/001-workspace-config/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── config-files.md
│   └── cli.md
└── tasks.md             # /speckit-tasks
```

### Source Code (repository root)

```text
Cargo.toml                      # workspace, shared lints and package metadata
rust-toolchain.toml
.github/workflows/ci.yml
crates/
├── raya-core/
│   ├── src/{lib.rs, error.rs, model.rs}
│   ├── src/config/{mod.rs, encoder.rs, agent.rs}
│   ├── src/shape.rs            # Segment, AttentionKind, Compression, ModelShape
│   └── tests/{config.rs, forbid_unsafe.rs}
├── raya-kernels/src/{lib.rs, attention.rs, gemm.rs}    # traits only
├── raya-weights/src/lib.rs
├── raya-tokenizer/src/lib.rs
└── raya-cli/
    ├── src/main.rs
    └── tests/stubs.rs
tests/data/fixture-tiny/   # now fixture-h64/ and fixture-h128/ (laya-sparse 357ad53)
├── encoder/config.json         # real ModernBertConfig.save_pretrained output
└── rl_agent_config.json
```

**Structure Decision**: a `crates/` directory keeps the repo root (docs, specs, reference, tests/data) uncluttered.
Fixture data stays at the repo-root `tests/data/` because features 002-006 also vendor data there; tests
locate it through `CARGO_MANIFEST_DIR/../../tests/data`.

## Complexity Tracking

No constitution violations.
