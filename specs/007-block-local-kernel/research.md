# Research: Fused Block-Local Attention Kernel

## R1. Reference semantics

**Decision**: Implement `out[i] = softmax_j(q_i . k_j * scale) v_j` for keys `j` with `|i - j| <= 64` and `j < valid_len`, `scale = 1/sqrt(head_dim)` by default (a parameter). Query rows `i >= valid_len` output zeros.

**Rationale**: Clarified exact +/-64 window and valid-length padding. Any row `i < valid_len` always has itself as a valid key, so the only empty rows are padded queries; softmax never divides by zero for real rows.

**Open check**: `reference/laya-sparse` is empty in the authoring environment, so `experiments/kernels/local.py` and Laya's local attention mask could not be read. Task T-check (first in tasks) reads both and confirms (a) window inclusivity (`<= 64` versus `< 64`), (b) scale, (c) that the PyTorch block-local kernel itself computes the exact window rather than the 3-block span (feasibility row "exact +/-64 local kernel equivalent to native: open" suggests it may not). Any divergence is reported, per constitution I, not silently adopted.

## R2. Tiling geometry

**Decision**: A task is one (head, query block of 128 rows). Inside it, query rows are processed in micro-tiles of `MR = 8`; for each micro-tile the key range is `[i0 - 64, i0 + MR - 1 + 64]`, walked in key tiles of `NC = 64`. Scores for one micro-tile and key tile live in a stack array `[MR][NC]`; running max `m`, running sum `l` and the `[MR][64]` output accumulator are rescaled online. Tiles with no in-window entry are skipped; edge tiles apply the `|i-j| <= 64` and `j < valid_len` mask by setting scores to `-inf` before the max.

**Rationale**: For a 128-block the keys any row needs span at most 256 positions, so visiting whole neighbour blocks would do ~2x the needed work; micro-tile ranges cut that to ~(2*64 + MR) keys per row. Q block (128x64 f32 = 32 KiB) and a 256-key K/V span (2 x 64 KiB) fit L2; micro-tile state fits L1/registers.

**Alternatives considered**: whole-block pair tiles (simpler, more wasted work, rejected); MR = 4 or 16 (to be tuned, recorded in the notes).

## R3. SIMD strategy

**Decision**: Two micro-kernels only: (1) `dot_tile`: MR x NC scores from Q rows and K rows (K stored row-major, so each score is a 64-long dot product vectorised over head dim with `wide::f32x8`, several accumulators for ILP), and (2) `axpy_tile`: `acc[r] += p[r][j] * v[j]` over head dim. Portable versions use `wide`; `avx2` adds FMA, `neon` uses `f32x4` FMA. Dispatch: `is_x86_feature_detected!("avx2")` and `fma` (x86), always on (aarch64) when the feature is enabled, chosen once at kernel construction; `portable` otherwise. Default builds never use the feature-gated paths.

**Rationale**: keeps `unsafe` and platform code in two small functions; the driver and softmax are shared, so the oracle comparison covers every backend.

**Alternatives considered**: transposed K for outer-product GEMM tiles (faster with `MR x NR` register blocking but costs a transpose per block; recorded as a follow-up experiment in the notes if 3x is missed).

## R4. Threading

**Decision**: `std::thread::scope` with a shared atomic task counter over (head, block) tasks; each worker writes disjoint output rows (`out` split through `chunks_mut` or `split_at_mut` per head block, no `unsafe`). `RAYA_THREADS` overrides the count; a `threads` field on the kernel overrides both (used by benches and tests).

**Rationale**: no `rayon` (not in the allowed dependency list). 16 heads x 64 blocks = 1024 tasks at 8K balances well.

## R5. Stage breakdown

**Decision**: Cargo feature `stage-timing` compiles in per-thread `Instant`-based accumulators around the QK, softmax and PV sections of each micro-tile and exposes `StageTimes`. Off by default, never part of the Criterion headline numbers.

**Rationale**: timer overhead on ~1 us sections is significant, so the breakdown is indicative (proportions), not the headline figure; both are recorded and the notes say so.

**Alternatives considered**: external profilers (`perf`) are not reproducible across the three OSes; sampling inside the bench would need a dependency.

## R6. Thread default uses `num_cpus`

**Decision**: The default thread count is `num_cpus::get_physical()`, overridable by `RAYA_THREADS`. Metadata records both `num_cpus::get()` (logical) and `get_physical()`.

**Rationale**: matches the clarified spec (physical cores). `std::thread::available_parallelism` is logical only; on the i7-8700 (6C/12T) the difference matters. `num_cpus` is a small, widely used crate; spec 001 FR-007 is amended to allow it in `raya-kernels` only.

**Alternatives considered**: `available_parallelism` with OS-specific parsing (rejected: more code than the dependency).

## R7. Goldens

**Decision**: `tests/goldens.rs` looks for a PyTorch attention golden under `tests/data/` (written by feature 002, name fixed in that feature) and skips with a printed line if it is missing. Tolerance 1e-5 absolute, justified by fp32 reassociation in the online softmax.

## R8. Benchmark inputs

**Decision**: Deterministic pseudo-random Q/K/V from a fixed-seed xorshift generator, values ~N(0, 1)-ish (uniform in [-1, 1]) so logits are O(1); 16 heads, head dim 64, `valid_len = seq`. Layout head-major `[heads, seq, head_dim]`, row-major `f32`, contiguous.

**Rationale**: Head-major keeps each head's Q, K and V contiguous for tiling. Feature 006 adapts its `[seq, heads, dim]` activations with a transpose or strided view; the trait therefore documents strides explicitly (see contract), settled here so 006 adopts it.
