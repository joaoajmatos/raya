# Research: Fused Block-Local Attention Kernel

## R1. Reference semantics (read from `reference/laya-sparse`)

**Decision**: One formula, two masks: `out[i] = softmax_j(q_i . k_j * scale) v_j`, `scale = 1/sqrt(head_dim)`, over keys `j < valid_len` that the mask allows; rows `i >= valid_len` and rows with no allowed key output zeros. `Block3`: `|i/128 - j/128| <= 1`. `Window64`: `|i - j| <= 64`.

**Evidence**: `experiments/kernels/reference.py` defines `block_local` as `(blk_i - blk_j).abs() <= 1` with block 128, scale `1/sqrt(D)`, key validity `j < lengths[b]`, and the empty-row-is-zero convention. `local.py` (the 85 ms kernel) builds exactly that with `unfold` over 3 blocks. So the 85 ms number measures `Block3`, which allows up to 384 keys per query against 129 for `Window64`.

**Rationale**: Constitution I (match Laya, report the observation). Both masks are needed: `Block3` for any comparison with 85 ms, `Window64` for the model's actual local attention. `Block3` is a superset of `Window64`, so the two tile drivers share everything but the key range and mask.

**Open check**: Window64 inclusivity (`<= 64` versus `< 64`) and the ModernBERT sliding-window definition belong to the model layer (feature 006/002 goldens); this feature tests `<= 64` as specified and records it.

## R2. Tiling geometry

**Decision**: A task is one (head, query block of 128 rows). Inside it, query rows are processed in micro-tiles of `MR = 8`. The key range per micro-tile is `[i0 - 64, i0 + MR - 1 + 64]` for `Window64` and the whole 3-block span `[b*128 - 128, b*128 + 256)` for `Block3`, walked in key tiles of `NC = 64`. Scores for one micro-tile and key tile live in a stack array `[MR][NC]`; running max `m`, running sum `l` and the `[MR][64]` output accumulator are rescaled online. Tiles with no in-window entry are skipped; edge tiles apply the mask (`Block3` or `|i-j| <= 64`) and `j < valid_len` by setting scores to `-inf` before the max.

**Rationale**: Under `Window64` the keys any row needs span at most 256 positions, so visiting whole neighbour blocks would do ~2x the needed work; micro-tile ranges cut that to ~(2*64 + MR) keys per row. `Block3` needs the whole 384-key span, so its micro-tile range is the full span and its cost is about 3x `Window64` per query by construction; the two masks' speeds must not be compared with each other as if equal work. Q block (128x64 f32 = 32 KiB) and a 256-key K/V span (2 x 64 KiB) fit L2; micro-tile state fits L1/registers.

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

**Decision**: `tests/goldens.rs` looks for a PyTorch attention golden (the `Block3` case, from `local.py`/`reference.py`) under `tests/data/` (written by feature 002, name fixed in that feature) and skips with a printed line if it is missing. Tolerance 1e-5 absolute, justified by fp32 reassociation in the online softmax.

## R8. Benchmark inputs

**Decision**: Deterministic pseudo-random Q/K/V from a fixed-seed xorshift generator, values ~N(0, 1)-ish (uniform in [-1, 1]) so logits are O(1); 16 heads, head dim 64, `valid_len = seq`. Layout head-major `[heads, seq, head_dim]`, row-major `f32`, contiguous.

**Rationale**: Head-major keeps each head's Q, K and V contiguous for tiling. Feature 006 adapts its `[seq, heads, dim]` activations with a transpose or strided view; the trait therefore documents strides explicitly (see contract), settled here so 006 adopts it.
