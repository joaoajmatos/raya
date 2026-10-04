# Tasks: Fused Block-Local Attention Kernel (Spike)

**Input**: Design documents from `/specs/007-block-local-kernel/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Required (constitution VI: every optimized kernel is tested against the naive oracle; workflow: every change ships with tests).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: can run in parallel (different files, no dependency)
- **[Story]**: US1 (correct kernels), US2 (benchmarks and breakdown), US3 (SIMD features and honest notes)

## Phase 1: Setup

- [X] T001 Read `reference/laya-sparse/experiments/kernels/{local.py,reference.py}` (submodule is initialised; run `git submodule update --init reference/laya-sparse` if not) and append a short "Confirmed against reference" section to `specs/007-block-local-kernel/research.md` R1 listing: `block_local` = `|blk_i - blk_j| <= 1` with block 128, scale `1/sqrt(D)`, key validity `j < length`, empty rows return zeros. Any difference from R1 is recorded as a divergence, not adopted silently
- [X] T002 Add `wide`, `num_cpus` and `criterion` to `[workspace.dependencies]` in `Cargo.toml` (latest versions compatible with rustc 1.94.1; `criterion` with `default-features = false` and `html_reports` off if it avoids extra crates); add them, `thiserror`, `serde`, `serde_json` to `crates/raya-kernels/Cargo.toml` (`criterion` as dev-dependency), features `avx2 = []`, `neon = []`, `stage-timing = []`, and a `[[bench]] name = "local_attention" harness = false`
- [X] T003 [P] `crates/raya-kernels/build.rs`: emit `RAYA_RUSTC_VERSION` (`rustc -V`) and `RAYA_GIT_COMMIT` (`git rev-parse HEAD`, `unknown` on failure) via `cargo:rustc-env`, with `rerun-if-changed=../../.git/HEAD`
- [X] T004 [P] Extend `.github/workflows/ci.yml` only if needed so `cargo test --workspace` still passes with the new dependencies; add a `cargo test -p raya-kernels --features stage-timing` step to the ubuntu `test` job. Confirm `cargo audit` stays clean for `wide`, `num_cpus`, `criterion` (CI steps added; `cargo audit` is not installed in the authoring container, so the audit result comes from the CI `audit` job)

## Phase 2: Foundational

- [X] T005 `crates/raya-kernels/src/attention.rs`: replace the stub with `AttnShape { heads, seq, head_dim }`, `Mask { Block3, Window64 }` (`Block3`: `|i/128 - j/128| <= 1`; `Window64`: `|i-j| <= 64`), `AttnParams { mask, block (128), valid_len, scale }`, `KernelError { Shape, Length, Unsupported }` (thiserror), and the trait `attend(&self, shape, params, q, k, v, out) -> Result<(), KernelError>` per `contracts/kernel-api.md`. Validation helper: buffers each `heads*seq*head_dim`, `head_dim % 8 == 0`, `valid_len <= seq`, `block == 128`, all else `Unsupported`/`Shape`/`Length` (no panics). Keep `name()`
- [X] T006 `crates/raya-kernels/src/naive.rs`: `Naive` kernel: single-threaded scalar f32, per-score loop over only the keys the mask allows and `j < valid_len`, subtract row max, sum, divide; rows `i >= valid_len` and rows with no allowed key output zeros; head-major layout. Export from `lib.rs`. This is the correctness oracle and the SC-004 speed baseline: no tiling, no explicit SIMD, contiguous access
- [X] T007 [P] `crates/raya-kernels/tests/common/mod.rs`: deterministic xorshift generator for Q/K/V (uniform in [-1, 1], fixed seeds), helper `max_abs_diff`, and an independent dense reference (`[seq, seq]` scores with the mask from `reference.py`, for small `seq` only) used to validate `Naive` itself
- [X] T008 `crates/raya-kernels/tests/oracle.rs`: `Naive` versus the dense reference for both masks at lengths 1, 7, 127, 128, 129, 257, padded tail (`valid_len < seq`), heads 1 and 3, head dim 8 and 64; tolerance 1e-6. Also checks `Naive` returns the `KernelError` cases of T005 instead of panicking

**Checkpoint**: `cargo test -p raya-kernels --test oracle` passes; the oracle is verified before any optimised kernel exists (constitution VI).

## Phase 3: User Story 1 - Correct Kernels (P1)

**Goal**: a tiled kernel that agrees with the oracle for both masks within 1e-5.

**Independent Test**: `cargo test -p raya-kernels` green including every listed edge length.

- [X] T009 [US1] `crates/raya-kernels/src/simd/{mod.rs,portable.rs}`: the two micro-kernels per research R3: `dot_tile` (MR x NC scores, `wide::f32x8` over head dim, several accumulators) and `axpy_tile` (`acc[r] += p[r][j] * v[j]`), plus a `Backend` enum (`Portable`, `Avx2`, `Neon`) and `Backend::detect()` that returns `Portable` unless a feature-gated path is compiled in and the CPU supports it (feature code lands in US3)
- [X] T010 [US1] `crates/raya-kernels/src/tiled/tile.rs`: online-softmax tile routine for one (head, query micro-tile of MR = 8 rows): for each key tile of NC = 64 over the mask's key range (`Window64`: `[i0-64, i0+MR-1+64]`; `Block3`: `[b*128-128, b*128+256)`), compute scores, apply mask and `j < valid_len` by `-inf`, update running max `m` and sum `l`, rescale and accumulate the `[MR][head_dim]` output, finalize by dividing by `l`; skip tiles with no allowed entry; `i >= valid_len` rows write zeros; a row with `l == 0` writes zeros. No allocation larger than one micro-tile of scores
- [X] T011 [US1] `crates/raya-kernels/src/threads.rs` and `crates/raya-kernels/src/tiled/mod.rs`: `Tiled`/`TiledConfig { threads: Option<usize>, backend: Backend }`; default threads = `RAYA_THREADS` if set and valid, else `num_cpus::get_physical()`; work split with `std::thread::scope` and an atomic task counter over (head, query block) with disjoint `out` slices (no `unsafe`); `attend` validates via T005 and runs tasks
- [X] T012 [P] [US1] `crates/raya-kernels/tests/equivalence.rs`: `Tiled` versus `Naive` for both masks at lengths 1, 127, 128, 129, 1000, 2048 and a padded tail (`valid_len = seq - 37`), 16 heads is too slow for the naive at 2048 so use heads 2 there and head dim 64, plus a 16-head 512 case; absolute tolerance 1e-5; run with 1 thread and with 4 threads and require bitwise-identical results between thread counts
- [X] T013 [P] [US1] `crates/raya-kernels/tests/equivalence.rs`: add a long-sequence case, `Tiled` versus `Naive` for both masks at seq 4096 and 8192 with 1 head and head dim 64 (covers FR-004 lengths; marked `#[ignore]` if it takes more than ~30 s in debug, and run locally with `cargo test -p raya-kernels --release -- --ignored`)
- [X] T014 [P] [US1] `crates/raya-kernels/tests/masking.rs`: exactness of both masks through the public kernel: put a large value in one V row and zero elsewhere, then assert the output of query `i` is influenced by key `j` exactly when the mask allows `(i, j)` (check every `(i, j)` for `seq = 300`, including block boundaries 127/128 and window edges 64/65); all-masked rows (`valid_len = 0`, and padded query rows) return exact zeros; extreme logits (Q and K scaled so scores reach +/-1e4) produce no NaN or inf and match the oracle to 1e-5
- [X] T015 [P] [US1] `crates/raya-kernels/tests/goldens.rs`: if a PyTorch attention golden for the `block_local` pattern exists under `tests/data/` (feature 002), compare `Naive` and `Tiled` with it at 1e-5; otherwise print "goldens absent (feature 002 not merged); skipped" and pass
- [X] T016 [US1] Document design in `docs/kernel-notes.md` section "Design": layout, tile sizes, key ranges per mask, padding and empty-row handling, online softmax, threading, and the statement that `Block3` is a superset of `Window64` (spec US1 scenario 4); leave the results and conclusion sections as placeholders for T025

**Checkpoint**: `cargo test -p raya-kernels` green; US1 scenarios 1-4 verified.

## Phase 4: User Story 2 - Benchmarks and Breakdown (P2)

**Goal**: reproducible one-layer timings with metadata and a stage breakdown.

**Independent Test**: `cargo bench -p raya-kernels --bench local_attention` writes the JSON; `--features stage-timing` adds stage percentages.

- [X] T017 [P] [US2] `crates/raya-kernels/src/meta.rs`: `BenchMetadata` per `data-model.md`: CPU model (`/proc/cpuinfo` `model name`; macOS `sysctl -n machdep.cpu.brand_string`; Windows `PROCESSOR_IDENTIFIER`; `unknown` otherwise), `num_cpus::get()` and `get_physical()`, threads, enabled cargo features (from `cfg!`), `RUSTFLAGS` (env at build time via `option_env!`), `RAYA_RUSTC_VERSION`, `RAYA_GIT_COMMIT`, timestamp; label "cloud-agent hardware, indicative only" unless `RAYA_TARGET_MACHINE=1`. Unit test that serialisation contains every key
- [X] T018 [P] [US2] `crates/raya-kernels/src/tiled/stages.rs`: `StageTimes { qk_ns, softmax_ns, pv_ns, tiles }`; with feature `stage-timing`, `tile.rs` (T010) wraps the QK, softmax and PV sections in `Instant` accumulators per worker and `Tiled::attend_timed` returns the merged `StageTimes`; without the feature the code compiles to nothing (`cfg`), and a test asserts percentages sum to ~100
- [X] T019 [US2] `crates/raya-kernels/benches/local_attention.rs`: Criterion groups for `naive` and `tiled`, each mask (`block3`, `window64`), seq 2048, 4096, 8192, 16 heads, head dim 64, `valid_len = seq`, inputs from the T007 generator; tiled at 1 thread and at the default thread count; sample sizes reduced for `naive` at 8192 (document the choice); after the run write `results/kernel-bench/<timestamp>.json` per `contracts/bench-output.md` (mean, stddev, samples from Criterion's estimates, plus stage percentages when built with `stage-timing`). Output directory created if missing
- [X] T020 [US2] Verify reproducibility: run the bench twice with the documented command, confirm both JSON files parse, contain metadata and all 12 (kernel x mask x length) rows at default threads plus the 1-thread tiled rows; record the exact commands in `docs/kernel-notes.md` section "How to reproduce"

**Checkpoint**: US2 scenarios 1-3 verified.

## Phase 5: User Story 3 - Optional SIMD and Honest Notes (P3)

**Goal**: optional `avx2`/`neon` micro-kernels and a notes file with a clear conclusion.

**Independent Test**: build and test with and without each feature; read `docs/kernel-notes.md`.

- [X] T021 [P] [US3] `crates/raya-kernels/src/simd/avx2.rs` (`#[cfg(all(feature = "avx2", target_arch = "x86_64"))]`): `dot_tile` and `axpy_tile` with AVX2 + FMA using `core::arch::x86_64` and `#[target_feature(enable = "avx2,fma")]`, each `unsafe` call site commented with its safety argument; `Backend::detect()` (T009) selects it via `is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")`, else falls back to `Portable` at construction
- [X] T022 [P] [US3] `crates/raya-kernels/src/simd/neon.rs` (`#[cfg(all(feature = "neon", target_arch = "aarch64"))]`): the same two micro-kernels with `core::arch::aarch64` `float32x4_t` FMA; selected by `detect()` on aarch64 when the feature is on; everything else falls back to `Portable`. If no aarch64 hardware is available in the session, compile-check with `cargo check --target aarch64-unknown-linux-gnu --features neon` (skip with a note if the target is not installed)
- [X] T023 [US3] `crates/raya-kernels/tests/dispatch.rs`: default features use only `Portable` (`Backend::detect()` returns `Portable` when no feature is on); with a feature enabled on supporting hardware `Tiled` reports that backend in `name()` and the T012 equivalence cases pass for it; requesting an unavailable backend falls back to `Portable` at construction
- [X] T024 [US3] Add the `Backend` (and thread count) to the T019 bench ids and JSON so a feature run is recorded and comparable; run the bench with and without `--features avx2` where AVX2 exists
- [X] T025 [US3] Finish `docs/kernel-notes.md`: results table per mask and length (naive and tiled, 1 thread and default, stage percentages), every figure labeled "cloud-agent hardware, indicative only" and no 85 ms figure placed in or beside the results tables (constitution III); state that only `block3` is comparable with the PyTorch 85 ms figure, that this is invalid until run through feature 008 on the target machine, and that `window64` and `block3` do different amounts of work; the SC-004 verdict for each mask (3x at 8192, single-threaded, same machine) with an explicit negative result and reasons if missed; what was tried and what was slower than expected; state that the bench and the SC-004 verdict are kernel-only, not whole-forward, a recorded deviation from constitution VI (repeat it in the PR description)

**Checkpoint**: US3 scenarios 1-4 verified.

## Phase 6: Polish

- [X] T026 [P] Run `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings` (also with `--features avx2`, `--features stage-timing`), `cargo test --workspace`; fix findings
- [X] T027 [P] Update `README.md` (kernel spike status, how to run the bench, pointer to `docs/kernel-notes.md`) and `specs/README.md` status for 007; verify `raya-kernels` is the only crate without `forbid(unsafe_code)` (`crates/raya-core/tests/forbid_unsafe.rs` still passes) and that no dependency beyond spec 001 FR-007 (as amended) appears in any `Cargo.toml`
- [X] T028 Walk `quickstart.md` end to end and fix any command that does not work; confirm SC-001 to SC-004 each have evidence (test, JSON, notes)

## Phase 7: Follow-up - runtime-detected AVX2 by default (amendment 2026-10-04)

- [X] T031 Amend `.specify/memory/constitution.md` principle V to 1.1.0 as its own commit (human-approved; i7-8700 evidence cited)
- [X] T032 `crates/raya-kernels/src/simd/mod.rs`: compile the AVX2 module on every x86_64 build (no cargo feature), `Backend::detect()` via `is_x86_feature_detected!("avx2")` and `"fma"` with the portable fallback, testable `select` rule, `Backend::from_name`; `dot_tile`/`axpy_rows` re-check the CPU before the `unsafe` call (sound for a caller-named `Backend::Avx2`); NEON unchanged
- [X] T033 `Cargo.toml`: keep `avx2 = []` as a documented no-op; `src/meta.rs`: record the selected `backend` in the bench metadata; `benches/local_attention.rs`: `RAYA_BACKEND` override
- [X] T034 Tests: `tests/dispatch.rs` (default picks the best backend for this CPU, every available backend matches the oracle on several shapes and head dims), `tests/simd_kernels.rs` (micro-kernels vs portable for head dims 8..136, unavailable backend gives the portable result), unit tests in `simd/mod.rs`
- [X] T035 Update spec 007, plan, README, `docs/kernel-notes.md`; run build, fmt, clippy `-D warnings`, test (VS env, `CC=cl`); CI stays green on macos-14 (portable)

## Dependencies and order

- Phase 1 then Phase 2 (T005 to T008) block everything; the oracle is tested (T008) before the tiled kernel exists.
- US1 (T009 to T016): T009 and T010 before T011; T012 to T015 after T011 and in parallel with each other.
- US2 (T017 to T020): T017 and T018 in parallel; T019 after both and after T011; T020 last.
- US3 (T021 to T025): T021 and T022 in parallel after T009; T023 after them; T024 after T019; T025 after everything measured.
- Polish last.

## Parallel examples

- After T005: T006 (oracle) and T007 (test helpers).
- After T011: T012 to T015.
- US3: T021 and T022 touch different files and run together.

## Implementation strategy

MVP = Phases 1 to 3: a verified naive oracle and a tiled kernel that matches it for both masks. Then US2 to measure, then US3 to add SIMD features and write the conclusion. Stop and report after US2 if SC-004 is clearly missed: the negative result is still a valid outcome, and the notes (T025) carry it.
