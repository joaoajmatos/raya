# Feature Specification: Fused Block-Local Attention Kernel (Spike)

**Feature Branch**: `p6-block-local-kernel` (specification directory only; no speckit branch created)

**Created**: 2026-09-30

**Status**: Draft

**Origin**: Prompt P6 of the former `docs/agent-prompts.md`.

**Depends on**: 001 (002 optional, for PyTorch reference outputs). **Blocks**: 008.

**Input**: User description: "Implement a naive and a tiled FlashAttention-style block-local attention kernel in raya-kernels, verify against the naive oracle, and benchmark one layer at 2K, 4K and 8K tokens. Negative results are welcome."

## User Scenarios & Testing *(mandatory)*

The user is the researcher deciding whether a Rust kernel can beat the PyTorch block-local kernel (85 ms per layer at 8K on the i7-8700, `docs/feasibility.md` section 2.2).

### User Story 1 - Correct Kernels (Priority: P1)

A naive per-score oracle and a tiled kernel implement block-local attention (block 128, each query block attends its own block plus one neighbour per side, equivalent to Laya's +/-64 window) and agree within 1e-5.

**Why this priority**: A fast wrong kernel is worthless, and the oracle is reused by feature 006.

**Independent Test**: Tiled versus naive on random inputs at edge lengths; against PyTorch goldens if available.

**Acceptance Scenarios**:

1. **Given** lengths 1, 127, 128, 129, 1000, 2048 and a padded tail, **When** tiled and naive run, **Then** outputs agree within 1e-5.
2. **Given** an all-masked row or extreme logits, **When** run, **Then** softmax is stable and no NaN appears.
3. **Given** feature 002 goldens, **When** available, **Then** outputs also match the PyTorch kernel.
4. **Given** the design, **When** documented, **Then** it states exactly how edge blocks and padding are handled and verifies semantic equivalence with the +/-64 window.

---

### User Story 2 - Benchmarks and Breakdown (Priority: P2)

One-layer Criterion benchmarks at 2048, 4096 and 8192 tokens, with a per-stage breakdown (QK tile, softmax, PV tile), written to JSON with machine and build metadata.

**Why this priority**: Decides whether the spike is worth pursuing, but only after correctness.

**Independent Test**: Run the bench; inspect the JSON and breakdown.

**Acceptance Scenarios**:

1. **Given** 16 heads, head dim 64, **When** benchmarked, **Then** timings exist for naive and tiled at each length.
2. **Given** a run, **When** the JSON is written, **Then** it records CPU model, core count, feature flags, rustc version and git commit.
3. **Given** thread count is configurable, **When** varied, **Then** results reflect it.

---

### User Story 3 - Optional SIMD and Honest Notes (Priority: P3)

Portable SIMD is used by default; `avx2` and `neon` micro-kernels are optional cargo features with runtime dispatch and a portable fallback. `docs/kernel-notes.md` records design, what was tried and what was slower than expected.

**Why this priority**: SIMD target is an open decision; the notes carry the learning even if the kernel loses.

**Independent Test**: Build with and without the features; read the notes.

**Acceptance Scenarios**:

1. **Given** default features, **When** built on any supported target, **Then** only the portable path is used and tests pass.
2. **Given** a feature enabled on supporting hardware, **When** run, **Then** dispatch selects it and results still match the oracle.
3. **Given** `docs/kernel-notes.md`, **When** read, **Then** numbers are labeled "cloud-agent hardware, indicative only" and the 85 ms comparison is stated as invalid until run through feature 008 on the target machine.
4. **Given** the tiled kernel does not beat naive or cannot be made competitive, **When** reported, **Then** the notes say so and why.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `AttentionKernel` trait in `raya-kernels` (shared with feature 006; define it here if 006 has not merged).
- **FR-002**: Implementations: `naive` oracle; `tiled` FlashAttention-style (Q tile, K/V tiles, register-blocked micro-GEMM, online softmax, never a full score matrix), multithreaded across (head, block).
- **FR-003**: Portable SIMD (`std::simd` or `wide`) by default; `avx2` and `neon` optional features; neither is the default.
- **FR-004**: Supports sequences up to 8192, 16 heads, head dim 64, block 128.
- **FR-005**: Criterion benches per User Story 2, with metadata JSON.
- **FR-006**: `unsafe` allowed here only (constitution V).

## Success Criteria *(mandatory)*

- **SC-001**: Correctness tests pass at 1e-5 on every listed case.
- **SC-002**: Benchmarks and breakdown are reproducible from documented commands.
- **SC-003**: The notes state a clear conclusion, including a negative one if warranted.

## Assumptions

- The reference implementation is `experiments/kernels/local.py` in laya-sparse (the former prompt pointed at `laya/experiments/`, which does not exist).
- Cloud timings never count as evidence against the 85 ms reference.
