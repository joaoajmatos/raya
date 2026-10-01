# Feature Specification: Fused Block-Local Attention Kernel (Spike)

**Feature Branch**: `p6-block-local-kernel` (specification directory only; no speckit branch created)

**Created**: 2026-09-30

**Status**: Implemented

**Verification**: local `cargo build`, `cargo fmt --check`, `cargo clippy -D warnings` (including `raya-kernels` with `--features avx2`) and `cargo test` (workspace) pass; CI is verified on GitHub only. The SC-004 speedup figures in `docs/kernel-notes.md` come from cloud-agent hardware (4 vCPU Xeon) and are indicative only, not i7-8700 results. US1 scenario 3 (PyTorch goldens) is met once the bump PR (vendoring the laya-sparse `cefb300` `block_local_L256` golden into `tests/data/kernels/`) merges: `tests/goldens.rs` then checks `Naive` and `Tiled` (`Block3`) against it at 1e-5. Until then the goldens are not on `main` and the test skips. `Window64` has no golden.

**Origin**: Prompt P6 of the former `docs/agent-prompts.md`.

**Depends on**: 001 (002 optional, for PyTorch reference outputs). **Blocks**: 008.

**Input**: User description: "Implement a naive and a tiled FlashAttention-style block-local attention kernel in raya-kernels, verify against the naive oracle, and benchmark one layer at 2K, 4K and 8K tokens. Negative results are welcome."

## Clarifications

### Session 2026-09-30

- Q: Which mask do the oracle and tiled kernel compute (exact +/-64 window or the full 3-block span)? → A: Initially exact +/-64; revised after reading `reference/laya-sparse/experiments/kernels/local.py`, see the next entry.
- Q: `local.py` (the 85 ms kernel) and `reference.py` define `block_local` as the 3-block span (`|blk_i - blk_j| <= 1`, block 128), not +/-64. Which semantics? → A: Both, selected by a `Mask` parameter: `Block3` (Laya's `block_local`, the only one comparable with the 85 ms kernel) and `Window64` (exact `|i-j| <= 64`, the real model's local window). The oracle and tiled kernel implement both, the benchmarks report both, and SC-004 applies to each mask separately.
- Q: How does the kernel handle a padded tail? → A: The `AttentionKernel` trait takes a per-sequence valid length; keys at or beyond it are masked, and rows with no valid keys output zeros (never NaN).
- Q: What outcome counts as a win for the spike? → A: Relative bar: tiled at least 3x faster than this naive oracle at 8192 tokens, single-threaded, same machine; the 85 ms comparison is deferred to feature 008 on the target CPU.
- Q: What CPU feature baseline does the default portable SIMD build assume? → A: Baseline target features only (no `target-cpu` flags); faster paths come only from the optional `avx2`/`neon` features with runtime dispatch. Bench metadata records the RUSTFLAGS used.
- Q: How is the portable SIMD default implemented on the pinned stable toolchain? → A: With the `wide` crate, added to `raya-kernels` only; spec 001 FR-007 is amended to allow it (`std::simd` is nightly-only).
- Q: What must the naive oracle look like so the 3x bar is meaningful? → A: A single-threaded scalar `f32` per-score loop over only the keys the mask allows (no tiling, no explicit SIMD, sensible memory access); it is both the correctness oracle and the speed baseline for SC-004.
- Q: What is the default thread count for the tiled kernel and benchmarks? → A: Physical core count (via `num_cpus`), overridable via `RAYA_THREADS`; benchmarks run at 1 thread (for SC-004) and at the default, and record the thread count in the JSON.

## User Scenarios & Testing *(mandatory)*

The user is the researcher deciding whether a Rust kernel can beat the PyTorch block-local kernel (85 ms per layer at 8K on the i7-8700, `docs/feasibility.md` section 2.2).

### User Story 1 - Correct Kernels (Priority: P1)

A naive per-score oracle and a tiled kernel implement block-local attention under two selectable masks and agree within 1e-5 for each: `Block3` (Laya's `block_local`: a query in 128-block *i* sees keys in blocks *i-1*, *i*, *i+1*) and `Window64` (exact `|i-j| <= 64`). Both are tiled by 128-blocks and differ only in the mask applied inside each tile.

**Why this priority**: A fast wrong kernel is worthless, and the oracle is reused by feature 006.

**Independent Test**: Tiled versus naive on random inputs at edge lengths; against PyTorch goldens if available.

**Acceptance Scenarios**:

1. **Given** lengths 1, 127, 128, 129, 1000, 2048 and a padded tail, **When** tiled and naive run, **Then** outputs agree within 1e-5.
2. **Given** an all-masked row or extreme logits, **When** run, **Then** softmax is stable and no NaN appears.
3. **Given** feature 002 goldens, **When** available, **Then** outputs also match the PyTorch kernel.
4. **Given** the design, **When** documented, **Then** it states exactly how edge blocks and padding are handled and verifies that each mask is computed exactly (no key outside it is attended, every key inside it is) and states that `Block3` is a superset of `Window64`.

---

### User Story 2 - Benchmarks and Breakdown (Priority: P2)

One-layer Criterion benchmarks at 2048, 4096 and 8192 tokens, with a per-stage breakdown (QK tile, softmax, PV tile), written to JSON with machine and build metadata.

**Why this priority**: Decides whether the spike is worth pursuing, but only after correctness.

**Independent Test**: Run the bench; inspect the JSON and breakdown.

**Acceptance Scenarios**:

1. **Given** 16 heads, head dim 64, **When** benchmarked, **Then** timings exist for naive and tiled at each length and for each mask.
2. **Given** a run, **When** the JSON is written, **Then** it records CPU model, core count, feature flags, rustc version and git commit.
3. **Given** thread count defaults to the physical core count and is overridable via `RAYA_THREADS`, **When** varied, **Then** results reflect it; benchmarks run at 1 thread and at the default, and the JSON records the thread count.

---

### User Story 3 - Optional SIMD and Honest Notes (Priority: P3)

Portable SIMD is used by default; `avx2` and `neon` micro-kernels are optional cargo features with runtime dispatch and a portable fallback. `docs/kernel-notes.md` records design, what was tried and what was slower than expected.

**Why this priority**: SIMD target is an open decision; the notes carry the learning even if the kernel loses.

**Independent Test**: Build with and without the features; read the notes.

**Acceptance Scenarios**:

1. **Given** default features, **When** built on any supported target, **Then** only the portable path is used and tests pass.
2. **Given** a feature enabled on supporting hardware, **When** run, **Then** dispatch selects it and results still match the oracle.
3. **Given** `docs/kernel-notes.md`, **When** read, **Then** numbers are labeled "cloud-agent hardware, indicative only", the 85 ms comparison is stated as valid only for `Block3` and only once run through feature 008 on the target machine, and `Window64` results are stated as not comparable with 85 ms.
4. **Given** the tiled kernel does not beat naive or cannot be made competitive, **When** reported, **Then** the notes say so and why.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `AttentionKernel` trait in `raya-kernels` (shared with feature 006; define it here if 006 has not merged). It takes a per-sequence valid length: keys at or beyond it are masked, and a row with no valid keys outputs zeros.
- **FR-002**: Both implementations take a `Mask` (`Block3` or `Window64`, as defined in User Story 1) and compute it exactly. `Block3` follows `reference/laya-sparse/experiments/kernels/reference.py` (`block_local`, scale `1/sqrt(head_dim)`, keys `j >= valid_len` masked, rows with no allowed key output zeros). Implementations: `naive` oracle (single-threaded scalar `f32`, per-score loop over only the keys the mask allows, no tiling or explicit SIMD; also the speed baseline for SC-004); `tiled` FlashAttention-style (Q tile, K/V tiles, micro-tiled dot-product kernels, online softmax, never a full score matrix), multithreaded across (head, block).
- **FR-003**: Portable SIMD via the `wide` crate by default (`raya-kernels` only; `std::simd` is nightly-only and excluded; spec 001 FR-007 amended to allow `wide`); `avx2` and `neon` optional features; neither is the default. The default build uses baseline target features only (no `-C target-cpu` flags); the bench metadata records the RUSTFLAGS in effect.
- **FR-004**: Supports sequences up to 8192, 16 heads, head dim 64, block 128.
- **FR-005**: Criterion benches per User Story 2, with metadata JSON.
- **FR-006**: `unsafe` allowed here only (constitution V).

## Success Criteria *(mandatory)*

- **SC-001**: Correctness tests pass at 1e-5 on every listed case.
- **SC-002**: Benchmarks and breakdown are reproducible from documented commands.
- **SC-003**: The notes state a clear conclusion, including a negative one if warranted.
- **SC-004**: The spike is a win for a mask if tiled is at least 3x faster than this naive oracle at 8192 tokens under that mask (16 heads, head dim 64, single-threaded, same machine). Missing the bar is reported as a negative result; no absolute comparison with 85 ms is made in this feature. This is a kernel-level result only, not a whole-forward-pass speed claim (a recorded deviation from constitution VI, acceptable for a spike).

## Assumptions

- The reference implementation is `experiments/kernels/local.py` in laya-sparse, with semantics defined by `experiments/kernels/reference.py` (the former prompt pointed at `laya/experiments/`, which does not exist). It computes the 3-block span, not +/-64; the spec's earlier "equivalent to the +/-64 window" claim was wrong and is replaced by the two-mask design.
- Cloud timings never count as evidence against the 85 ms reference.
