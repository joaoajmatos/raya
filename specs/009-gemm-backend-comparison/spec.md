# Feature Specification: GEMM Backend Comparison for the i7-8700

**Feature Branch**: `009-gemm-backend-comparison`

**Created**: 2026-10-07

**Status**: Draft (specify, clarify, plan and tasks only; nothing implemented, no dependency added, `Cargo.lock` untouched, no run taken)

**Origin**: Follow-up to `docs/tensor-runtime.md` (feature raya:006), which compared candle, ndarray plus BLAS and plain loops and omitted the pure-Rust SIMD GEMM crates. Source survey: scout report `cpu-inference-survey.md` (2026-10-07).

**Depends on**: raya:006 (merged: `GemmBackend`, `Reference`, goldens), raya:007 (merged: bench metadata, runtime AVX2 dispatch). **Related**: raya:008 (bench harness, draft) supplies the whole-forward measurement. **Blocks**: nothing; it feeds a human decision.

**Input**: User description: "Plan a GEMM backend comparison for the i7-8700 (AVX2, FMA, no AVX-512, no VNNI, 6C/12T), fp32: the current plain Rust loops, and the pure-Rust runtime-detected GEMMs `matrixmultiply`, `gemm` and `faer`; a native BLAS through FFI is documented but excluded. Define the shapes from the Laya checkpoint, the correctness criterion against the vendored goldens, the measurement protocol and hardware labeling, thread settings, what each candidate costs in dependencies and `unsafe`, and the decision the human must make."

## Clarifications

### Session 2026-10-07

Answered from this spec, the constitution, `docs/tensor-runtime.md`, the survey and the code at `50ebd3e`; decisions that belong to the researcher are listed in "Human Decisions" and are not answered here.

- Q: Does this spec add a dependency, and is a GEMM dependency allowed? -> A: This spec adds none. The researcher's position (2026-10-07, quoted under Assumptions) is that a GEMM dependency is acceptable in principle; that is neither a choice of crate nor permission to add one now. The comparison decides which candidate, by evidence on the i7-8700. The dependency rules that apply are: constitution VII, "Dependencies are limited to those a specification names"; raya:001 FR-007, a closed list (serde, serde_json, thiserror, safetensors, tokenizers, clap, criterion (dev), anyhow in raya-cli only, wide and num_cpus in raya-kernels only) with SC-003 "No dependency outside FR-007 appears in any `Cargo.toml`"; and the CI dependency audit (constitution, Technical Constraints). Nothing in them forbids a GEMM crate; they require that a specification name it and that raya:001 FR-007 be amended to include it, as raya:007 did for `wide` and `num_cpus`. This spec therefore names `matrixmultiply`, `gemm` and `faer` as the candidates (satisfying VII) and makes the FR-007 amendment the first gated task (T002), done in the implementation PR, with the crates added as optional measurement dependencies only after that task. The adoption of one crate into the default build is a second step (Phase 5) that happens only after the comparison result and a re-check of the same rules, and removes the losing candidates from `Cargo.toml`.
- Q: Which operation is compared? -> A: `GemmBackend::matmul_nt(out, a, b, m, n, k)`: `out[m,n] = a[m,k] * b[n,k]^T`, row-major fp32, `b` being a Laya weight `[out, in]`. Bias, GELU/GeGLU and layer norm stay outside the backend (as in `Linear::apply`), so a candidate is a drop-in implementation of the trait and the model code is untouched (raya:006 FR-004).
- Q: Which shapes? -> A: Those of the Laya-shape model (hidden 1024, 16 heads of 64, MLP 2624 gated, 28 encoder layers, 2-layer head over the full sequence) at `m` = 512, 1024, 2048, 4096, 8192 tokens, plus the single-row and few-row GEMMs (`act_head`, scorer). Exact `[n, k]` list in `contracts/shapes.md`. The head feed-forward width is taken as 4 x hidden, as in the vendored fixture (`linear1 [256, 64]` at hidden 64); task T001 confirms it against the real checkpoint inventory before any run, and the spec value is corrected if it differs.
- Q: What is the correctness criterion? -> A: Two gates, both before any timing (constitution VI). (1) Whole forward: with the candidate as the `GemmBackend`, the four vendored golden cases pass `forward_golden` at the raya:006 tolerances. The spec-level tolerances are 1e-4 (layer outputs) and 1e-5 (probabilities); the asserted, tightened constants in `crates/raya-cli/tests/forward_golden.rs` are 1e-5 and 1e-6 and are the gate. A candidate that passes only the looser pair is reported and the researcher decides (HD-5). (2) Per-shape: at every real-checkpoint shape, each output element differs from `Reference` by at most `2 * k * eps * sum_t |a_t * b_t|` (`eps = 2^-24`), the standard bound for reordered f32 accumulation; the goldens are at hidden 64/128 and cannot cover k = 2624 or 4096, so this bound does. Bit-equality with `Reference` is not required, and summation order may differ (stated in the report).
- Q: How is the machine labeled? -> A: By `BenchMetadata`, never by hand. The label text and the rule for when each label applies are those of constitution principle III in force when the run is made (1.2.0 after PR #12: `RAYA_TARGET_MACHINE=1` gives "target machine", `cloud-agent` gives "cloud-agent hardware, indicative only", anything else "unlabeled machine, indicative only"; before it, any value other than `1` gave the cloud-agent label). This spec does not depend on the wording: it requires only that the label come from the metadata of the run, that the CPU model string be recorded, and that a report state which label a number carries. Numbers without the "target machine" label are never compared with target-machine numbers, `docs/feasibility.md` or Laya results.
- Q: Threads? -> A: Three settings per candidate and shape, recorded in the JSON: 1 (the like-for-like comparison with `Reference`, which is single-threaded), 6 (physical cores) and 12 (logical cores, Hyper-Threading). Each candidate's own parallelism switch is set explicitly (not left to a default) and read back into the JSON. Whether multithreaded GEMM is acceptable at all is HD-2, because raya:006 specified single-threaded deterministic loops and a threaded backend can change run-to-run results.
- Q: Is a native BLAS measured? -> A: No. OpenBLAS, BLIS, oneDNN and oneMKL through FFI are documented in `research.md` with their costs and are excluded: a native toolchain on the Windows measurement machine, `unsafe` FFI, thread-count-dependent summation order, and (MKL) a licence. Reopening is HD-6.
- Q: Does a kernel microbenchmark satisfy constitution VI ("speed claims MUST account for the whole forward pass")? -> A: No, and this is recorded as a deviation, as raya:007 did. The harness reports per-shape times and a GEMM budget (sum over the model's GEMMs of count x time per length). The whole-forward effect needs raya:008's `raya bench` with a selectable backend; that step is task T018, blocked on raya:008 and on the researcher running it on the target machine.
- Q: Where does the harness live? -> A: As a Criterion bench target in `raya-kernels` (the only crate where `unsafe` is allowed, and `matrixmultiply::sgemm` and `gemm::gemm` are `unsafe fn`), with the candidates as optional dependencies behind a non-default cargo feature `gemm-candidates`. Default builds, CI and `Cargo.lock` consumers are unaffected except that `Cargo.lock` lists the optional crates once HD-1 is approved. The alternative (a separate workspace) is HD-3.
- Q: Does the unsafe policy change? -> A: No. Adapters that call an `unsafe fn` live in `raya-kernels` behind the feature; `raya-core` and every other crate keep `#![forbid(unsafe_code)]`.

## Selection Criteria

The comparison selects among `Reference` (the raya:006 loops, i.e. no dependency), `matrixmultiply`, `gemm` and `faer`, by this evidence, all recorded per candidate:

| Criterion | Kind | Evidence |
|---|---|---|
| Correctness against the raya:006 goldens (and the per-shape bound) | eliminator | User Story 1 gates |
| Licence compatible with the repository's (MIT or Apache-2.0 expected) | eliminator | crate metadata at the pinned version, T008 |
| `cargo audit` clean (no advisory on the candidate or its new transitive crates) | eliminator | T008, the same audit CI runs |
| MSRV not above the pinned toolchain (rust-version 1.94) | eliminator | T004 to T006, build with 1.94.1 |
| Builds and passes the gates on Linux x86_64, macOS arm64 and Windows (constitution V) | eliminator | T009 |
| Speed on the i7-8700 at 1 thread, and at 6 and 12 if HD-2 allows threading | ranking | per-shape ratios and GEMM budget, T012 to T016 |
| Threading behavior: explicit control, effective count, run-to-run variation | ranking | T014, golden gate repeated |
| Dependency cost: new crates in the lock diff | ranking | T008 |
| `unsafe` surface (adapter lines, confined to `raya-kernels`) | ranking | T004 to T006 |

Native BLAS through FFI (OpenBLAS, BLIS, oneDNN, oneMKL) stays documented but excluded: it adds a native toolchain and FFI `unsafe`, and nothing in the constitution forbids it explicitly, so the exclusion is this spec's scope decision, reopenable by HD-6.

A candidate that survives every eliminator and is not slower than `Reference` at any shape that matters is a selection candidate. If one candidate is best on speed at every length and not worse on the cost rows, the report says it dominates; otherwise the trade-off goes to the researcher (HD-4). The comparison never selects `Reference` by default: it is the outcome when no candidate survives or none is worth its cost, stated plainly (constitution III).

## User Scenarios & Testing *(mandatory)*

The user is the researcher, who has to decide whether the plain loops of raya:006 are replaced by a pure-Rust SIMD GEMM and wants the evidence on the target CPU.

### User Story 1 - Candidates Are Correct Before They Are Fast (Priority: P1)

Each candidate is wrapped as a `GemmBackend` and checked: against `Reference` at every shape of the Laya-shape model, and end to end on the four golden cases.

**Why this priority**: constitution VI, "a benchmark MUST NOT run against a model whose output parity has not been checked". A fast wrong GEMM is worthless.

**Independent Test**: `cargo test -p raya-kernels --features gemm-candidates` and `cargo test -p raya-cli --features gemm-candidates` (golden gate; only after T002 and T003).

**Acceptance Scenarios**:

1. **Given** each candidate and each shape in `contracts/shapes.md`, **When** run on seeded random inputs, **Then** every element is within `2 * k * eps * sum |a*b|` of `Reference`.
2. **Given** each candidate selected as the model's backend, **When** the four golden cases run, **Then** `forward_golden` passes at the raya:006 asserted tolerances (1e-5 layer outputs, 1e-6 probabilities) or the failure names the first divergent layer and the candidate.
3. **Given** a candidate that fails either gate, **When** the benchmark is requested, **Then** it is skipped with the gate failure printed and recorded in the JSON, not timed silently.
4. **Given** the machine lacks AVX2 or FMA, **When** a candidate that needs it is selected, **Then** it falls back or returns an explicit error (`Error::Unimplemented`), and the JSON records which path ran.

---

### User Story 2 - Timings per Shape and Thread Setting (Priority: P2)

A bench measures each candidate and `Reference` on each shape at 1, 6 and 12 threads and writes JSON with the machine label and build metadata.

**Why this priority**: it is the evidence the decision needs.

**Independent Test**: run the bench on any machine for the fixture-sized shapes; check the JSON fields. The i7-8700 run is the researcher's (HD-7).

**Acceptance Scenarios**:

1. **Given** the shape list, **When** benchmarked, **Then** the JSON has p50, p90, min, samples, GFLOP/s and the fraction of the `Reference` time for every candidate, shape and thread count.
2. **Given** a run, **When** written, **Then** `BenchMetadata` fields are present (label, CPU model, logical and physical CPUs, threads, features, backend, rustc, git commit, timestamp) plus candidate crate versions and the candidate's thread switch.
3. **Given** `RAYA_TARGET_MACHINE` is unset, **When** run, **Then** the label is whatever `BenchMetadata` emits for an unverified machine and the report says the numbers are indicative only.
4. **Given** all runs, **When** summarized, **Then** the GEMM budget per length and candidate is computed from the per-shape medians and the layer counts.

---

### User Story 3 - A Decision Note With Costs (Priority: P3)

`docs/gemm-backends.md` states results (or that none exist yet), the cost of each candidate in dependencies, `unsafe`, build and audit, and the decisions for the researcher; `docs/tensor-runtime.md` gets a pointer and its omission of the pure-Rust crates is corrected.

**Why this priority**: the researcher decides, so the note has to make the trade-off explicit.

**Independent Test**: read the note against `research.md`.

**Acceptance Scenarios**:

1. **Given** the note, **When** read, **Then** every performance statement is a recorded measurement with its label, or marked a hypothesis.
2. **Given** the note, **When** read, **Then** for each candidate it lists licence, new crates in `Cargo.lock` (counted from the lock diff), `unsafe` surface, runtime detection, Windows/macOS-arm64/Linux build status and `cargo audit` result.
3. **Given** a negative or inconclusive result (no candidate beats `Reference` by a useful margin, or none passes the gates), **When** reported, **Then** it says so plainly (constitution III).

## Edge Cases

- `m` of 1 to a few rows (`act_head`, scorer): packing overhead can make a blocked GEMM slower than loops; measured and reported, not excluded.
- Dimensions that are not multiples of the vector width: every real `n` and `k` here is a multiple of 8 (2624 = 8 x 328, 5248, 1024, 4096), but `act_0` has `k` = 1028 (a multiple of 4 only, to confirm) and the scorer output has `n` = 1; the correctness gate also covers odd `m`, `n`, `k` (`contracts/shapes.md`).
- Packing the static weights on every call is what `matmul_nt` implies; a pre-packed weight path is outside this spec (it would change the trait) and is named as a hypothesis in the note.
- Hyper-Threading: 12 threads may be slower than 6 for a compute-bound GEMM; that is a result, not an error.
- Windows power plan and turbo: recorded in the report text (not detectable by the harness); runs under a laptop-style power saver are rejected by the protocol.
- A candidate whose published version needs a newer rustc than 1.94.1 is reported as not buildable, not forced.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-000**: No dependency is added, and `Cargo.lock` is not changed, before HD-1 and the raya:001 FR-007 amendment (T002); the winning crate enters the default build only in Phase 5.
- **FR-001**: Each candidate implements `GemmBackend` unchanged (`matmul_nt(out, a, b, m, n, k)`, overwrite semantics, panics on wrong slice lengths as `Reference` does) inside `raya-kernels`, behind the non-default feature `gemm-candidates`; no other crate gains a dependency or `unsafe`.
- **FR-002**: `Reference` stays the default backend and the oracle. This spec changes no default and no model code.
- **FR-003**: The correctness gates of User Story 1 run before any timing; a candidate that fails is not timed.
- **FR-004**: Shapes come from `contracts/shapes.md`, derived from the Laya checkpoint's tensor shapes and the model config; the list is checked against the real checkpoint inventory (T001) before a run.
- **FR-005**: Measurement protocol (below) is followed and recorded; deviations are recorded in the JSON `notes`.
- **FR-006**: Results are written to `results/gemm-compare/<timestamp>.json` (git-ignored) per `contracts/gemm-compare-output.md`; the label comes from `BenchMetadata`.
- **FR-007**: Thread settings 1, 6 and 12 are explicit per candidate and recorded; the candidate's own switch is set, not defaulted.
- **FR-008**: No `target-cpu` or `target-feature` flags (constitution V); candidates with their own runtime detection are used as they ship; the backend that actually ran (path name) is recorded.
- **FR-009**: No timing, benchmark or hardware run is part of writing or reviewing this spec; the runs are tasks T012 to T016 and the target-machine run is the researcher's.
- **FR-010**: The deviation from constitution VI (kernel-level, not whole-forward) is stated in the report, the note and the PR description.
- **FR-011**: Candidate costs are recorded from the actual lock diff and `cargo audit`/`cargo tree` output, not from this spec's estimates.

### Measurement Protocol

- **Machine**: the i7-8700 (6C/12T, AVX2, FMA, no AVX-512, no VNNI) for the numbers the decision rests on; anything else is indicative only. The researcher confirms the machine and `RAYA_TARGET_MACHINE=1` for those runs (HD-7).
- **Conditions**: AC power, high-performance power plan, nothing else running, reported background load; CPU model, logical and physical core counts read from the machine and compared in the report with 6C/12T.
- **Build**: `cargo build --release` with default profile, toolchain 1.94.1, no `RUSTFLAGS` (recorded), features `gemm-candidates`.
- **Per cell** (candidate x shape x thread count): inputs generated once from fixed seeds; warm-up of at least 3 calls and 200 ms; at least 15 timed calls, each of which runs `matmul_nt` into a reused output buffer; the weights are not pre-packed. Report p50, p90, min, samples; the primary statistic is p50. Large shapes (8192 tokens) may use fewer samples (at least 5) with the number recorded.
- **Order**: candidates interleaved per shape in a fixed rotation, and a second pass in reverse order, to expose drift (thermal, boost); a result that changes by more than 10% between passes is flagged and rerun.
- **Same-session baseline**: `Reference` is measured in the same session on the same shapes at 1 thread; ratios are against that run only.
- **Threads**: 1, 6, 12 as in FR-007. Set via each candidate's own mechanism; the harness reads back and records the effective count.
- **Not measured**: whole forward (T018, raya:008), memory bandwidth effects of pre-packing, int8/fp16/bf16 (constitution IV), AVX-512 (absent on the target).

### Candidate Cost Summary (hypotheses until T008 and the Phase 3 gates record them)

| | `Reference` (raya:006) | `matrixmultiply` | `gemm` | `faer` | native BLAS (excluded) |
|---|---|---|---|---|---|
| New crates | 0 | to count (survey: pure Rust, f32/f64 gemm only) | to count (survey: `gemm-common`, `gemm-f32`, `raw-cpuid`, `pulp` and others) | to count (largest tree of the three) | native library plus `*-src` crates |
| `unsafe` | none | `sgemm` is `unsafe fn`: adapter in `raya-kernels` | `gemm::gemm` is `unsafe fn`: adapter in `raya-kernels` | safe matmul API expected; confirm at T004 | FFI, `unsafe` |
| Runtime AVX2 detection | n/a | yes (survey, since 0.2.0) | yes (survey) | yes (survey) | library-dependent |
| Native toolchain | no | no | no | no | yes |
| Threading | none | own `threading` feature (to confirm) | rayon-based parallelism (to confirm) | rayon-based parallelism (to confirm) | library thread pool |
| Licence | project | Apache-2.0 or MIT | MIT | MIT | BSD-3 / Apache-2.0 / ISSL (MKL) |

Every cell is `[documented]` by the survey or an open item resolved by the named task; none is a measurement.

## Human Decisions

None of these is decided by this spec. Implementation tasks that depend on one are marked blocked.

- **HD-1**: Approve the three candidate crates as measurement dependencies and the raya:001 FR-007 amendment that names them (the researcher's "i dont mind having a gemm dep" is read as acceptance in principle, not as this approval). Versions compatible with rustc 1.94.1 are chosen at T003. Unblocks T002 and T003; not approving leaves this spec as a plan.
- **HD-2**: Is a multithreaded GEMM acceptable for Raya at all, given raya:006 chose single-threaded deterministic loops, or are only 1-thread numbers of interest? Decides whether T014 (6 and 12 threads) runs.
- **HD-3**: Harness placement: a feature-gated bench in `raya-kernels` (this spec's choice; `Cargo.lock` lists the optional crates, `unsafe` allowed) or a separate workspace outside the root workspace (root lock untouched, but adapters calling `unsafe fn` then sit outside `raya-kernels` and need a constitution clarification).
- **HD-4**: After the results: which candidate to adopt into the default build (Phase 5), keep `Reference`, or defer. Phase 5 runs only after this decision, the re-check of constitution VII and raya:001 FR-007/SC-003, a clean `cargo audit`, and the CI builds; it adds exactly one crate to the default features and removes the others.
- **HD-5**: If a candidate passes the spec-level tolerances (1e-4 / 1e-5) but not the tightened asserted ones (1e-5 / 1e-6), is it acceptable (and are the asserted constants then relaxed for that backend)?
- **HD-6**: Keep native BLAS/oneDNN/oneMKL excluded (this spec's position), or add a native-BLAS candidate later.
- **HD-7**: Run the target-machine measurement (T015) on the i7-8700 and set `RAYA_TARGET_MACHINE=1` there. The researcher owns this run and the labeling; the target CPU and precision remain an open decision (constitution V), assumed to be the i7-8700 until decided otherwise.
- **HD-8**: Order versus raya:008: run the GEMM comparison first, after raya:008, or fold the whole-forward step into 008's harness.

## Success Criteria *(mandatory)*

- **SC-001**: Every candidate passes both correctness gates, or the report names the gate it failed and no timing exists for it.
- **SC-002**: For each (candidate, shape, thread count) the JSON holds p50, p90, min, samples and the `Reference` ratio, with the label and the other `BenchMetadata` fields (including `threads`).
- **SC-003**: `docs/gemm-backends.md` gives, per candidate, the measured crate count, `unsafe` surface, audit status and CI build status on Linux x86_64, macOS arm64 and Windows, plus the decision list; no cell is left as a hypothesis without that marking.
- **SC-004**: `Cargo.lock` and default builds are byte-identical before and after for default features (the candidates exist only under `gemm-candidates`, once HD-1 is approved).
- **SC-005**: No benchmark or timing run happened while specifying (FR-009), and no dependency was added by this spec.

## Assumptions

- Researcher's decision on dependencies, verbatim (2026-10-07): **"i dont mind having a gemm dep"**. Read as: adding a GEMM dependency is acceptable in principle. It does not choose a crate and does not authorize adding one now; the choice is by the evidence of this comparison and the adding happens in the implementation of this spec, after the result and a check of the dependency rules (constitution VII; raya:001 FR-007 and SC-003; the CI audit).
- The target machine is the i7-8700 until the researcher decides otherwise; its ISA facts (AVX2, FMA, no AVX-512, no VNNI) are as in the survey.
- fp32 only (constitution IV); low-precision candidates are out of scope.
- The vendored fixtures (hidden 64 and 128) are the only goldens; real-checkpoint shapes are covered by the per-shape bound, not by goldens.
- Survey facts about the crates are `[documented]` and must be re-verified from the pinned crate versions' own docs and source at T004, T005 and T006.
