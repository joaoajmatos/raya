# Tasks: GEMM Backend Comparison for the i7-8700

**Input**: Design documents from `/specs/009-gemm-backend-comparison/`

**Prerequisites**: plan.md, spec.md, research.md, contracts/

**Tests**: Required (constitution VI: correctness gates before timing; workflow: every change ships with tests).

**Status**: nothing below is implemented. Writing this file ran nothing, added no dependency and left `Cargo.lock` untouched. No benchmark or timing run is part of writing or reviewing the spec (FR-009); the timing tasks are planned only.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: can run in parallel (different files, no dependency)
- **[Story]**: US1 (correct before fast), US2 (timings), US3 (decision note)
- **[HD-n]**: blocked until that human decision (spec, "Human Decisions"); an agent does not decide it

## Phase 1: Setup and Gates

- [ ] T001 [US1] Read the real checkpoint's tensor inventory (the Laya checkpoint under `reference/laya-sparse/` or the path the researcher gives; if absent, stop and say so) and confirm or correct `contracts/shapes.md`: head feed-forward width, `act_head` input width, MLP intermediate 2624, hidden 1024. Record the source and the corrections in `research.md` R9. Read-only; no run.
- [ ] T002 [HD-1] Amend `specs/001-workspace-config/spec.md` FR-007 and SC-003 to name `matrixmultiply`, `gemm` and `faer` (raya-kernels only, optional, feature `gemm-candidates`, measurement) as raya:007 did for `wide` and `num_cpus`; cite constitution VII ("Dependencies are limited to those a specification names") in the PR description. In the same PR nothing else adds a crate.
- [ ] T003 [HD-1] After T002: add the three crates as optional dependencies of `raya-kernels` with the non-default feature `gemm-candidates`; pick versions whose `rust-version` is at most 1.94 and record them. Verify `cargo build --workspace` and `cargo test --workspace` without the feature are unchanged and that the default dependency tree (`cargo tree`) is identical before and after (SC-004); record the `Cargo.lock` diff (crates added) for T008. Do not run any benchmark.

## Phase 2: Adapters (feature-gated, `raya-kernels`)

- [ ] T004 [P] [US1] `matrixmultiply` adapter implementing `GemmBackend` (`matmul_nt` as `sgemm` with `b` read through swapped strides, overwrite semantics, same length panics as `Reference`); confirm the API, the transposed-b mapping, the thread control and the MSRV from the pinned version's docs and source (research R3 `[open]` items); `unsafe` confined to this file with a comment stating the invariant.
- [ ] T005 [P] [US1] `gemm` crate adapter (`gemm::gemm` with the `Parallelism` argument set explicitly); confirm signature, MSRV, parallelism type (research R4 `[open]`); `unsafe` confined to this file.
- [ ] T006 [P] [US1] `faer` adapter (matmul over views with an explicit `Par`); confirm whether the API is safe and whether a transposed `b` is a zero-cost view (research R5 `[open]`).
- [ ] T007 [US1] Smallest change that lets `Network` run with a chosen `GemmBackend` for the golden gate (today `gemm: Reference` is a field; for example a constructor taking `&'a dyn GemmBackend`). Default stays `Reference`; if this touches `raya-core` behavior it is flagged in the PR, and `raya-core` keeps `#![forbid(unsafe_code)]`.
- [ ] T008 [US3] Record per candidate: licence (crate metadata), crates added to `Cargo.lock` (from T003), `cargo audit` result, `unsafe` lines in the adapter, rust-version vs 1.94, runtime-detection mechanism and the path name that runs on the target CPU. Output goes to `docs/gemm-backends.md` as measured or read facts, not estimates (FR-011).
- [ ] T009 [US3] Build and run the gates of T010 and T011 on Linux x86_64 and macOS arm64 (CI job with the feature) and on Windows (the researcher's machine, or a Windows runner if one exists); record per candidate pass or the error. A candidate that does not build somewhere is reported, not forced.

## Phase 3: Correctness Gates (before any timing)

- [ ] T010 [US1] `crates/raya-kernels/tests/gemm_candidates.rs` (feature-gated): each candidate against `Reference` at every shape of `contracts/shapes.md` (seeded inputs; at the largest `m` use a row subset if memory requires, recorded) and at the odd tiny shapes; bound `2 * k * 2^-24 * sum |a*b|` per element; wrong slice lengths panic like `Reference`; `m`, `n` or `k` of zero behaves like `Reference`; overwrite semantics (output prefilled with garbage).
- [ ] T011 [US1] `crates/raya-cli/tests/golden_candidates.rs` (feature-gated): the four golden cases with each candidate as the backend (via T007), at the asserted raya:006 tolerances (1e-5 layer outputs, 1e-6 probabilities); report the first divergent layer and the candidate on failure; repeat each threaded configuration several times and report any run-to-run difference (HD-2, HD-5 inputs).

**Checkpoint**: both gates pass for a candidate, or its failure is recorded and it gets no timings.

## Phase 4: Timings and the Decision Note

- [ ] T012 [US2] `crates/raya-kernels/benches/gemm_compare.rs` (feature-gated Criterion bench): cells per `spec.md` protocol (warm-up, samples, interleaved order and reverse pass, same-session `Reference` baseline), 1 thread first; skips a candidate whose gate failed; label from `BenchMetadata` (no label option).
- [ ] T013 [P] [US2] JSON writer to `results/gemm-compare/<timestamp>.json` per `contracts/gemm-compare-output.md` and unit tests that need no timing (field presence, label copied from metadata, a failed-gate candidate has no `results`, ratio and flag computation on synthetic numbers, budget sum on synthetic numbers).
- [ ] T014 [US2] [HD-2] Extend T012 to 6 and 12 threads with each candidate's own switch set explicitly and the effective count read back; skip entirely if HD-2 says single-threaded only.
- [ ] T015 [US2] [HD-7] Target-machine run: the researcher runs the bench on the i7-8700 with `RAYA_TARGET_MACHINE=1`, AC power, high-performance plan, nothing else running, and states the background load. An agent does not run it and does not set the label. Any run elsewhere is labeled by `BenchMetadata` and is indicative only.
- [ ] T016 [US2] Analyze the JSON: per-shape ratios, flags, GEMM budget per length and candidate; flag drift between passes above 10% and rerun that cell.
- [ ] T017 [US3] `docs/gemm-backends.md`: results with their labels (or "no run yet"), the Selection Criteria table filled per candidate (eliminators and ranking evidence), the cost table, negative and inconclusive results stated plainly, the constitution VI deviation, and the decisions HD-4 and HD-5 for the researcher. Fix `docs/tensor-runtime.md`: add a pointer and correct its omission of the pure-Rust SIMD crates (one paragraph; its decision for raya:006 stands).
- [ ] T018 [US2] [blocked on raya:008 and HD-7] Whole-forward check: run `raya bench` (raya:008) with `Reference` and with the candidate(s) that survived, on the i7-8700, and compare the effect with the GEMM budget; closes the constitution VI deviation. Not started until 008 exists.

## Phase 5: Adoption (only after HD-4; its own PR)

- [ ] T019 [HD-4] Re-check the dependency rules against the chosen crate: constitution VII, raya:001 FR-007 and SC-003, the CI `cargo audit`, licence, MSRV, the three-platform builds. If any fails, stop and report; do not adopt.
- [ ] T020 [HD-4] Adopt exactly the chosen crate as a normal dependency of `raya-kernels`, remove the other candidates and the `gemm-candidates` feature (or keep the feature only if the researcher wants the comparison reproducible), keep `Reference` as the portable fallback and the oracle, select the backend at construction with the chosen backend's runtime detection, record the backend that ran in benchmark output, and let callers request `Reference` explicitly (constitution V). Update raya:001 FR-007 to the final crate list.
- [ ] T021 [HD-4] Re-run the golden gate (T011) on the default build and `cargo test --workspace`, `cargo clippy -D warnings`, `cargo fmt --check`; update `docs/tensor-runtime.md` status and `specs/README.md`.

## Dependencies and order

- T001 first (read-only). T002 then T003 (HD-1). T004 to T006 parallel after T003; T007 after at least one adapter; T008 and T009 after T003 and the adapters.
- T010 and T011 before T012 onward (no timing for a candidate without both gates).
- T012 then T013 (parallel with T012's body), T014 after T012 (HD-2), T015 is the researcher's (HD-7), T016 after T015, T017 after T016, T018 after raya:008 and HD-7.
- Phase 5 only after T017 and HD-4.
- Spec PRs: this spec's own PR contains specification files only.
