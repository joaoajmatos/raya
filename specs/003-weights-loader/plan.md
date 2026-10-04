# Implementation Plan: Weights Loader

**Branch**: `003-weights-loader` | **Date**: 2026-10-01 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/003-weights-loader/spec.md`

## Summary

Load a Laya `model.safetensors` into a `WeightStore` of fp32 slices addressed by typed keys. One mapping table, built from `ModelConfig`, turns the expected parameter set into (Laya name, typed key, shape) entries; the loader checks the file against it and reports every mismatch in a single aggregate error. The file is read once into an owned, 4-byte-aligned buffer (no mmap, no `unsafe`); tensors are borrowed views of that buffer. `raya info` prints config, tensor count, computed parameter count and bytes. No model math, no packing, no dtype conversion.

## Technical Context

**Language/Version**: Rust, toolchain pinned by `rust-toolchain.toml` (workspace `rust-version = 1.94`), edition 2021

**Primary Dependencies**: existing `safetensors` 0.6 (header parsing and bounds checks), `thiserror`, `clap`, `anyhow`, `serde_json`; one new dependency, `bytemuck` (safe aligned byte/`f32` casts), named here per constitution VII; dev-only `tempfile`

**Storage**: a single `model.safetensors`; tests use the vendored `tests/data/weights-h64.safetensors` and `weights-h128.safetensors`, with mutated copies in a temp dir

**Testing**: `cargo test --workspace`; integration tests in `crates/raya-weights/tests/` and `crates/raya-cli/tests/`; offline

**Target Platform**: Linux x86_64, macOS arm64, Windows (little-endian only; a big-endian target fails at compile time)

**Project Type**: library crates plus CLI in an existing workspace

**Performance Goals**: none claimed (constitution III); load is one sequential read

**Constraints**: no `unsafe` outside `raya-kernels`; fp32 only; vendored fixtures and `reference/` never modified

**Scale/Scope**: fixtures are 0.85 MB (h64) and 3.1 MB (h128); real Laya is about 1.7 GB, so one buffer and no second copy matters

## Constitution Check

| Principle | Status |
|---|---|
| I Laya is the specification | Pass. Names and layout come from `inventory.json` and `laya/common.py`; nothing is "fixed". |
| II Parity by goldens | Pass. Inventory coverage test plus bit equality against the vendored file; negative tests localize by tensor name. |
| III Measure, do not assume | Pass. No perf claims; `raya info` computes the parameter count. The shape table in research R4 is a hypothesis verified by the coverage test. |
| IV Raya-L, fp32 | Pass. F16/BF16 are widened to fp32 on load (permitted by IV, approved 2026-10-04); other dtypes return `Unimplemented`. No half-precision kernels. |
| V Portable first | Pass. No SIMD or layout choice; no packing hook. |
| VI Correctness before speed | Pass. Nothing optimized. |
| VII Small surface, explicit errors | Pass with one justified dependency (`bytemuck`). Loader fails loudly naming missing, unused and mis-shaped tensors. |

Re-check after design: unchanged.

## Project Structure

### Documentation (this feature)

```text
specs/003-weights-loader/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── weights-api.md
│   └── info-cli.md
├── checklists/requirements.md
└── tasks.md           # /speckit-tasks
```

### Source Code

```text
crates/raya-core/src/error.rs        # + Error::Weights(LoadReport)
crates/raya-weights/
├── Cargo.toml                       # + bytemuck, serde_json/tempfile (dev)
├── src/
│   ├── lib.rs
│   ├── key.rs                       # ParamKey (typed key)
│   ├── mapping.rs                   # the one table: ModelConfig -> Vec<Entry>
│   ├── buffer.rs                    # aligned owned file buffer
│   └── store.rs                     # WeightStore, load, load_dir
└── tests/
    ├── mapping_coverage.rs          # SC-002 against tests/data/inventory.json
    ├── load_fixtures.rs             # US1: h64 + h128, bit equality
    └── load_negative.rs             # US2: missing/extra/shape/dtype/io
crates/raya-cli/
├── Cargo.toml                       # + raya-core, raya-weights
├── src/main.rs                      # Info { dir }
└── tests/info.rs                    # SC-003
```

**Structure Decision**: loader logic lives in `raya-weights` (already scaffolded); only the error type touches `raya-core`; `raya info` formatting stays in `raya-cli`.

## Complexity Tracking

| Item | Why | Simpler alternative rejected |
|---|---|---|
| New dependency `bytemuck` | Safe `&mut [f32]` to `&mut [u8]` for reading into an aligned buffer, and safe `&[u8]` to `&[f32]` views | `memmap2` needs `unsafe`; per-tensor `Vec<f32>` copies double peak memory on a 1.7 GB file |
