# Implementation Plan: Fused Block-Local Attention Kernel (Spike)

**Branch**: `007-block-local-kernel` | **Date**: 2026-09-30 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/007-block-local-kernel/spec.md`

## Summary

Fill in the `AttentionKernel` trait in `raya-kernels` and add two implementations of block-local
attention, each under two masks (`Block3`, Laya's `block_local` and the kernel the 85 ms figure measures, and exact `Window64`): a scalar `naive` oracle (also the speed baseline) and a FlashAttention-style `tiled` kernel
(online softmax, no score matrix, multithreaded over (head, query block), portable SIMD through `wide`, optional
`avx2` and `neon` micro-kernels with runtime dispatch). Correctness tests compare tiled with naive at edge
lengths; Criterion benches time one layer at 2K, 4K and 8K tokens and write a metadata JSON; a stage-timing
feature gives the QK / softmax / PV breakdown. `docs/kernel-notes.md` records the conclusion against the 3x bar
of SC-004, negative or not.

## Technical Context

**Language/Version**: Rust stable 1.94.1, edition 2021 (`std::simd` excluded: nightly-only)

**Primary Dependencies**: `wide` (raya-kernels only, added by this feature; spec 001 FR-007 amended), `num_cpus` (raya-kernels only, added by this feature; spec 001 FR-007 amended), `thiserror` (raya-kernels, already allowed), `serde` and `serde_json` (bench metadata only, already allowed), `criterion` (dev, allowed). Nothing else: threads use `std::thread::scope`, CPU model (the core counts come from `num_cpus`) comes from `/proc/cpuinfo`, `sysctl` or `PROCESSOR_IDENTIFIER`, rustc version and git commit come from a `build.rs` running `rustc -V` and `git rev-parse`.

**Storage**: Bench output under `results/kernel-bench/` (git-ignored). Optional PyTorch goldens read from `tests/data/` if present.

**Testing**: `cargo test -p raya-kernels` (naive-vs-tiled equivalence, mask-exactness, NaN stability, backend dispatch); goldens test skips with a printed notice if feature 002 fixtures are absent.

**Target Platform**: Linux x86_64 and macOS arm64 in CI (portable path); `avx2`/`neon` features tested only where the hardware exists; Windows supported.

**Project Type**: Library crate plus a Criterion bench target.

**Performance Goals**: SC-004: for each mask, tiled >= 3x faster than naive at 8192 tokens, 16 heads, head dim 64, single-threaded, same machine. No absolute target in this feature.

**Constraints**: fp32 only; no full score matrix (working set per task is O(tile)); `unsafe` only inside `raya-kernels`, and only if a micro-kernel needs it; baseline target features in default builds.

**Scale/Scope**: Sequences up to 8192, 16 heads, head dim 64, block 128, masks `Block3` and `Window64` (64).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

| Principle | Status |
|---|---|
| I. Laya is the specification | Pass. `Block3` matches `reference.py`/`local.py` (read and confirmed, research R1); `Window64` is the model's own local window. The divergence between the two (feasibility open question) is reported, not hidden. |
| II. Parity via golden fixtures | Pass with condition. PyTorch goldens are used when feature 002 output exists; otherwise the naive oracle is the reference and the notes say so. |
| III. Measure, do not assume | Pass. Metadata JSON per run; cloud numbers labeled "cloud-agent hardware, indicative only"; the 85 ms comparison is stated invalid until run through 008. |
| IV. Raya-L, fp32, inference only | Pass. fp32 only, no other dtypes. |
| V. Portable first, SIMD optional | Pass. `wide` path is the default; `avx2`/`neon` are off by default with runtime dispatch and fallback. `unsafe` stays in `raya-kernels`. |
| VI. Reference before kernel | Pass. Naive oracle lands first (tasks order) and tiled is tested against it. This spike benchmarks the kernel in isolation, a recorded deviation (see Complexity Tracking), and the notes and the PR description state it. |
| VII. Small surface, explicit errors | Pass with note. `wide` and `num_cpus` are new dependencies, authorised by the spec clarifications and the spec 001 amendment. Shape or length errors return `KernelError`, not a panic. |

Post-design re-check: still pass. No spec deviations.

## Project Structure

### Documentation (this feature)

```text
specs/007-block-local-kernel/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── kernel-api.md
│   └── bench-output.md
└── tasks.md             # /speckit-tasks
```

### Source Code (repository root)

```text
Cargo.toml                          # add wide, criterion to [workspace.dependencies]
crates/raya-kernels/
├── Cargo.toml                      # features: avx2, neon, stage-timing; build.rs
├── build.rs                        # RUSTC_VERSION, GIT_COMMIT env for metadata
├── src/
│   ├── lib.rs
│   ├── attention.rs                # AttentionKernel, AttnShape, AttnParams, KernelError
│   ├── naive.rs                    # scalar oracle
│   ├── tiled/{mod.rs, tile.rs, stages.rs}   # driver, online-softmax tile, stage timers
│   ├── simd/{mod.rs, portable.rs, avx2.rs, neon.rs}   # micro-kernels and dispatch
│   ├── threads.rs                  # RAYA_THREADS, work split over (head, block)
│   └── meta.rs                     # BenchMetadata (CPU, cores, flags, rustc, commit)
├── tests/{equivalence.rs, masking.rs, dispatch.rs, goldens.rs}
└── benches/local_attention.rs
docs/kernel-notes.md
.gitignore                          # results/ already ignored
```

**Structure Decision**: all kernel code stays in `raya-kernels`; the bench lives in that crate so Criterion is a
dev-dependency of one crate. The bench metadata module is in the library (behind no feature) so feature 008 can
reuse it.

## Complexity Tracking

| Item | Why needed | Simpler alternative rejected |
|---|---|---|
| Kernel-only speed claim (deviation from constitution VI, "speed claims MUST account for the whole forward pass") | A spike must decide whether the kernel is worth pursuing before the whole forward exists (feature 006 is not merged); SC-004 is labeled a kernel-level result, and whole-forward claims wait for 008 | Blocking the spike on 006 and 008 would remove its purpose |
| New dependencies `wide`, `num_cpus` | Stable portable SIMD (clarified); physical core count for the thread default | `std::simd` needs nightly; auto-vectorisation was offered and declined |
| `build.rs` in raya-kernels | rustc and commit in metadata without a dependency | A `vergen`-style crate is a new dependency |
