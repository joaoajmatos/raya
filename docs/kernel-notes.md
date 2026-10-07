# Block-local attention kernel: notes (raya:007)

Spike on whether a Rust FlashAttention-style kernel is competitive for Laya's block-local attention.
Spec: [specs/007-block-local-kernel](../specs/007-block-local-kernel/spec.md).

**All numbers below are "cloud-agent hardware, indicative only"** (4 vCPU Intel Xeon @ 2.80 GHz, `num_cpus`
reports 4 physical and 4 logical; rustc 1.94.1; default `RUSTFLAGS`, no `target-cpu`). They are kernel-only
results, not whole-forward results, a recorded deviation from constitution VI. They are not compared with
`docs/feasibility.md` or any Laya result directory. That label is the author's statement for these figures
(the author knew the machine); the bench itself writes it only when run with `RAYA_TARGET_MACHINE=cloud-agent`
(constitution 1.2.0). `RAYA_TARGET_MACHINE=1` marks the i7-8700 and an unset variable gives
"unlabeled machine, indicative only"; the recorded `label` is never edited by hand.

## What was implemented

Two masks, one formula `out[i] = sum_j softmax_j(scale * q_i . k_j) v_j`, `scale = 1/sqrt(head_dim)`, over keys
`j < valid_len` that the mask allows; rows `i >= valid_len` and rows with no allowed key are exact zeros.

| Mask | Keys allowed for query `i` | Per-query keys (full rows) | Source |
|---|---|---|---|
| `Block3` | blocks `i/128 - 1 ..= i/128 + 1` | up to 384 | `reference/laya-sparse/experiments/kernels/{local,reference}.py` (`block_local`) |
| `Window64` | `abs(i - j) <= 64` | 129 | the model's exact local window |

`local.py` (the kernel behind the feasibility study's PyTorch block-local figure) computes `Block3`, not
`Window64`. `Block3` is a superset of `Window64` and does about 3x the work per query, so the two masks must not
be compared with each other as if equal work.

Kernels (`crates/raya-kernels`):

- `Naive`: single-threaded scalar f32, one dot product per allowed score, no tiling, no explicit SIMD. It is
  the correctness oracle and the speed baseline.
- `Tiled`: parallel over (head, query block of 128 rows) with `std::thread::scope`; per task, 8-row
  micro-tiles walk the keys in tiles of 64 with an online softmax (running max and sum, rescaled
  accumulator). Only one `8 x 64` score tile exists at a time; no score matrix. Key range per micro-tile is the
  union of its rows' allowed ranges; per-row masks are applied inside the tile, so edge tiles do no work for
  keys outside the mask (the `Window64` range is about 136 keys per 8 rows, `Block3` the 3-block span).
- Layout `[heads][seq][head_dim]`, row-major f32; padding through `valid_len` (`AttnParams`).
- SIMD: `wide::f32x8` portable micro-kernels by default (QK: 2 rows x 2 keys register blocking; PV: a row's
  64 output lanes held in registers across keys; exp: `wide` vectorised). AVX2 + FMA micro-kernels
  (compiled into every x86_64 build since constitution 1.1.0; the `avx2` cargo feature is now a no-op) and the
  optional `neon` feature replace the two micro-kernels, selected once at construction by runtime
  detection, with fallback to portable. Builds use only baseline target features (no `target-cpu` flags).
- Threads: `RAYA_THREADS`, else `num_cpus::get_physical()`.

## Correctness

`cargo test -p raya-kernels`: the oracle is checked against an independent dense reference (full score matrix,
boolean mask as in `reference.py`) at 1e-6; `Tiled` is checked against `Naive` at 1e-5 for both masks at lengths
1, 127, 128, 129, 1000, 2048, padded tails, 16 heads, head dims 8/40/64/96, and (ignored by default, run in
release) 4096 and 8192. Mask exactness is tested for every `(i, j)` pair at block and window edges; extreme
logits (scores around 1e4) stay finite and match; results are bit-identical between 1 and 4 threads. The AVX2
path passes the same tests. NEON compiles for `aarch64-unknown-linux-gnu` but has not been run on hardware.
PyTorch goldens (`laya:004`, vendored from laya-sparse `cefb300` into `tests/data/kernels/`) cover `block_local` at
length 256 only (16 heads, head dim 64); `tests/goldens.rs` checks `Naive` and `Tiled` (`Block3`) against both the
native reference and the PyTorch kernel output at 1e-5. `Window64` and longer lengths have no golden.

## How to reproduce

```bash
cargo test -p raya-kernels
cargo test -p raya-kernels --release -- --ignored            # 4096 and 8192 equivalence
cargo bench -p raya-kernels --bench local_attention           # writes results/kernel-bench/<ts>.json
RAYA_THREADS=1 cargo bench -p raya-kernels --bench local_attention
RAYA_BACKEND=portable cargo bench -p raya-kernels --bench local_attention   # force the portable path
cargo bench -p raya-kernels --bench local_attention --features stage-timing   # stage percentages
RAYA_BENCH_SEQS=512 cargo bench -p raya-kernels --bench local_attention       # quick run
```

The JSON records CPU model, logical and physical core counts, threads, cargo features, `RUSTFLAGS`, rustc
version and git commit. Criterion samples are 10 per configuration (flat sampling) because one naive 8K
iteration takes seconds.

## Results (16 heads, head dim 64, one layer, `valid_len = seq`)

Mean milliseconds, 10 samples. Speedup is naive divided by tiled. "t1" is one thread; "t4" is the default here
(4 physical cores).

Default build (portable `wide` path):

| Mask | Tokens | Naive | Tiled t1 | t1 speedup | Tiled t4 | t4 speedup |
|---|---|---|---|---|---|---|
| Block3 | 2048 | 829 | 269 | 3.08x | 69 | 12.1x |
| Block3 | 4096 | 1659 | 517 | 3.21x | 131 | 12.6x |
| Block3 | 8192 | 3537 | 994 | 3.56x | 267 | 13.3x |
| Window64 | 2048 | 277 | 86 | 3.20x | 25 | 10.9x |
| Window64 | 4096 | 554 | 179 | 3.10x | 48 | 11.6x |
| Window64 | 8192 | 1141 | 354 | 3.22x | 102 | 11.2x |

`--features avx2` (AVX2 + FMA micro-kernels; same machine):

| Mask | Tokens | Naive | Tiled t1 | t1 speedup | Tiled t4 | t4 speedup |
|---|---|---|---|---|---|---|
| Block3 | 2048 | 812 | 185 | 4.39x | 53 | 15.4x |
| Block3 | 4096 | 1658 | 364 | 4.56x | 101 | 16.5x |
| Block3 | 8192 | 3334 | 778 | 4.29x | 201 | 16.6x |
| Window64 | 2048 | 271 | 70 | 3.88x | 19 | 14.2x |
| Window64 | 4096 | 600 | 140 | 4.28x | 37 | 16.2x |
| Window64 | 8192 | 1175 | 303 | 3.87x | 76 | 15.4x |

Spread at 8192, one thread, default build: naive `Block3` 3537 +/- 199 ms, tiled 994 +/- 72; naive `Window64`
1141 +/- 57, tiled 354 +/- 18 (one standard deviation).

Stage breakdown at 2048 tokens, one thread, default build, from a `stage-timing` build (percentages only: the
per-tile timers add overhead, so absolute times from that build are not meaningful):

| Mask | QK | softmax | PV |
|---|---|---|---|
| Block3 | 38.8% | 29.8% | 31.4% |
| Window64 | 40.1% | 28.2% | 31.7% |

Work per query differs between masks: `Window64` is about one third of the work of `Block3`, and the tiled
kernel's time reflects that (2048: 86 ms versus 269 ms).

## Conclusion (SC-003, SC-004)

**SC-004 is met for both masks at 8192 tokens, one thread, on this machine:** 3.56x for `Block3` and 3.22x for
`Window64` with the default portable build; 4.29x and 3.87x with `avx2`. The `Window64` margin on the default
build is thin (the ratio is about 3.2x with a one-sigma spread of roughly 7%, so 3.0x to 3.5x), so on the default build that result
should be read as "about 3x", not as a clear win; the `avx2` build clears the bar with room. With
threads the tiled kernel scales close to linearly with the four cores (about 3.7x on four threads).

What this does and does not show:

- It shows a hand-written portable Rust kernel reaches about 3-3.5x a scalar per-score implementation and
  about 4-4.5x with AVX2 on this CPU, and that online-softmax tiling needs no score matrix.
- The baseline is a scalar loop (no auto-vectorisation of reductions, since floating-point addition is not
  reordered), so part of the speedup is plain SIMD rather than tiling. The spike did not try to separate them.
- It says nothing about the PyTorch block-local kernel's 85 ms per layer at 8K: that figure was measured on the
  target CPU with a different implementation, and any comparison is invalid until both are run through feature
  008 on that machine. Only `Block3` could ever be compared with it (same semantics); `Window64` does a third of
  the work and is not comparable.

## What was tried

| Change | Effect (512 tokens, one thread, default build, cloud machine) |
|---|---|
| First portable kernel (per-score dot, scalar `exp`) | Block3 61.6 ms, Window64 24.6 ms (2.8x, 2.7x versus naive) |
| `wide` vectorised `exp` for the softmax | Block3 58.1 ms, Window64 24.8 ms: little change; the softmax stage also holds the max, scale and rescale passes |
| QK 2x2 register blocking (each loaded vector feeds two products) | Block3 52.2 ms, Window64 22.6 ms (3.5x, 3.1x) |
| AVX2 + FMA micro-kernels (runtime-detected; was a feature) | about 1.2x to 1.5x over the portable path at 2K-8K (tiled, one thread) |

Slower or less useful than expected: the vectorised exp barely helped; softmax remains about 28-30% of the
time, so the online-softmax bookkeeping (max, scale, rescale, per-row bounds) matters as much as the
arithmetic. The AVX2 gain is modest; a likely reason is that the softmax stage (about 28-30%) is not
accelerated by the AVX2 micro-kernels, which is a hypothesis, not measured.

Not tried (follow-ups, in order of expected value): transposing K per block for an outer-product QK
(`MR x NR` register tile), AVX-512, fusing the scale into Q once, tuning `MR` and the key tile width, a
separate measurement of tiling versus SIMD contribution, and whole-forward measurements through raya:008.

## Limits of this spike

- Kernel-only benchmark, one layer, random inputs uniform in [-1, 1]; not a whole-forward measurement.
- Single machine, shared cloud vCPUs, 10 samples; differences of under about 10% are within noise.
- NEON untested on hardware; no macOS arm64 timings.
- PyTorch goldens exist only for `block_local` at length 256; longer lengths and `Window64` are checked against
  an independent dense reference and against `Naive` instead.
