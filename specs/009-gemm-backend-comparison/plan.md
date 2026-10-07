# Implementation Plan: GEMM Backend Comparison for the i7-8700

**Branch**: `009-gemm-backend-comparison` | **Date**: 2026-10-07 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/009-gemm-backend-comparison/spec.md`

## Summary

Wrap `matrixmultiply`, `gemm` and `faer` as `GemmBackend` implementations in `raya-kernels` behind a non-default feature `gemm-candidates`, prove each against `Reference` at every shape of the Laya-shape model and against the four vendored goldens at the raya:006 tolerances, then measure them with a Criterion bench at 1, 6 and 12 threads and write labeled JSON. The deliverable is evidence and a cost table for the researcher's decision (HD-1 to HD-8), not an adoption. This planning step adds no dependency, changes no code and runs nothing.

## Technical Context

**Language/Version**: Rust stable 1.94.1 (pinned), edition 2021.

**Primary Dependencies**: none added by this spec. After HD-1 and the raya:001 FR-007 amendment (T002), optional measurement dependencies of `raya-kernels` only: `matrixmultiply`, `gemm`, `faer` (versions compatible with 1.94.1, chosen at T003); after the result, Phase 5 keeps one of them (or none) as a normal dependency. Existing: `wide`, `num_cpus`, `serde`, `serde_json`, `thiserror`, `criterion` (dev).

**Storage**: `results/gemm-compare/<timestamp>.json` (git-ignored).

**Testing**: `cargo test -p raya-kernels --features gemm-candidates` (per-shape gate), `cargo test -p raya-cli --features gemm-candidates` (golden gate, candidate selected as the backend). Default `cargo test --workspace` is unchanged.

**Target Platform**: measurement on the i7-8700 (Windows, per the researcher's machine); CI builds the default features only on ubuntu-latest and macos-14; the `gemm-candidates` build is checked on Linux x86_64, macOS arm64 and Windows at T009 (constitution V) before any adoption discussion.

**Project Type**: Library crate extension plus a Criterion bench target.

**Performance Goals**: none. The comparison reports ratios to `Reference`; whether a ratio is worth a dependency is HD-4.

**Constraints**: fp32 only; no `target-cpu` flags; `unsafe` only in `raya-kernels`; default build and `Cargo.lock` unchanged until HD-1; `Reference` stays the default.

**Scale/Scope**: the shape list of `contracts/shapes.md` (about 10 GEMM kinds x 5 token counts, 3 thread settings, 4 backends including `Reference`).

## Constitution Check

| Principle | Status |
|---|---|
| I. Laya is the specification | Pass. Shapes come from the Laya checkpoint; no model code changes. |
| II. Parity via golden fixtures | Pass. Golden gate per candidate; per-shape bound covers the shapes the fixtures cannot. |
| III. Measure, do not assume | Pass. Every cost cell in the spec is marked a hypothesis until measured; the label comes from `BenchMetadata` under whichever wording of III is in force (1.2.0 once PR #12 merges); nothing is compared across machines. |
| IV. Raya-L, fp32, inference only | Pass. fp32 only; int8 and other precisions out of scope. |
| V. Portable first, runtime-detected SIMD | Pass with check. No `target-cpu`; the portable `Reference` is always present and the default; candidates rely on their own runtime detection (to verify at T004 to T006) and the path that ran is recorded. Threading in a candidate is explicit and recorded. |
| VI. Reference before kernel | Pass with recorded deviation. `Reference` exists and is the oracle; correctness gates precede timing. The bench is kernel-level, not whole-forward: a deviation recorded in the report, the note and the PR description; T018 closes it through raya:008. |
| VII. Small surface, explicit errors | Pass conditional on HD-1. "Dependencies are limited to those a specification names": this spec names the three candidates; raya:001 FR-007 (closed list) and SC-003 are amended first (T002, as raya:007 did for `wide` and `num_cpus`); no crate is added before that, and only one stays after Phase 5. The researcher accepts a GEMM dependency in principle (quoted in spec Assumptions); the crate is chosen by evidence. Unsupported CPU/config returns an explicit error. |

Post-design re-check: still pass; HD-1 is the only gate on dependencies and the plan stops cleanly without it.

## Project Structure

### Documentation (this feature)

```text
specs/009-gemm-backend-comparison/
├── spec.md
├── plan.md
├── research.md
├── quickstart.md
├── tasks.md
└── contracts/
    ├── shapes.md
    └── gemm-compare-output.md
```

### Source Code (after HD-1; none of it exists yet)

```text
crates/raya-kernels/
├── Cargo.toml                 # optional deps + feature gemm-candidates (non-default)
├── src/gemm/                  # existing gemm.rs moved only if needed; adapters under cfg(feature)
│   ├── mod.rs
│   ├── matrixmultiply.rs      # adapter, unsafe confined here
│   ├── gemm_crate.rs          # adapter, unsafe confined here
│   └── faer.rs                # adapter
├── tests/gemm_candidates.rs   # per-shape gate
└── benches/gemm_compare.rs    # Criterion bench + JSON writer
crates/raya-cli/tests/golden_candidates.rs   # golden gate, feature-gated
docs/gemm-backends.md          # results, costs, decisions
```

**Structure Decision**: everything that touches a candidate sits behind `gemm-candidates` in `raya-kernels`, the one crate allowed `unsafe`. `raya-core` is selected through a backend parameter only if the golden gate needs it (T007 decides the smallest change, for example a constructor taking `&dyn GemmBackend`; if that is a behavior change to `raya-core` it is flagged in the PR, not slipped in).

## Complexity Tracking

| Item | Why needed | Alternative rejected |
|---|---|---|
| Optional dependencies in `raya-kernels` | The comparison needs the real crates. | Re-implementing their algorithms would measure our code, not theirs. |
| Kernel-level bench (deviation from VI) | Whole-forward needs raya:008, which is a draft and a human-run step. | Waiting for 008 delays the evidence; T018 closes the gap later. |
| Three thread settings | 1 is the like-for-like, 6 and 12 show scaling and Hyper-Threading. | 1 only hides what a threaded backend would give (still HD-2 decides if it counts). |
