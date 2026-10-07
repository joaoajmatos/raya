# Research: GEMM Backend Comparison

Evidence labels as in the survey: `[measured]` recorded in this repository, `[documented]` stated by a cited external source (survey of 2026-10-07), `[inference]` reading, `[open]` resolved by a named task. Nothing here comes from a run of any candidate.

## R1. What the current backend is and what it already proved

`Reference` (`crates/raya-kernels/src/gemm.rs`): plain loops, sequential f32 accumulation, single thread. Parity with Laya through it: worst intermediate difference 7.2e-7, worst probability difference 3.0e-8 over the four goldens (`specs/006-reference-forward/quickstart.md`) `[measured]`. `docs/tensor-runtime.md` lists its speed as a hypothesis ("clearly the slowest") and no run exists `[measured: none]`. So the question this spec answers has never been measured.

## R2. Where GEMM time matters (context, not a result)

From the survey of the PyTorch native fp32 run on the i7-8700 `[measured]` (`laya-sparse` Phase 1 `full` report): linear layers are about 76% of the clean p50 at 512 tokens and 52% at 2K, but attention including the head is 74% at 8K. `[inference]`: a faster GEMM helps short and medium inputs most and the 8K case least; the head GEMMs run over the full sequence, so they scale with `m` like the encoder ones. This only motivates the shape list; it is a different software stack and not a prediction for Raya.

## R3. Candidate: `matrixmultiply`

- Pure Rust, `f32`/`f64` gemm only; licence Apache-2.0 or MIT `[documented]`.
- Runtime x86 feature detection since 0.2.0, AVX2 packing since 0.3.5, AVX-512 kernels since 0.3.11 `[documented]` (changelog). On the i7-8700 only the AVX2 path applies.
- API: `sgemm` is an `unsafe fn` taking raw pointers and strides `[documented]`; the adapter needs `unsafe` and therefore lives in `raya-kernels`. A transposed `b` is expressible through strides (`b[n,k]` read as `k x n` with swapped strides) `[open: T004 confirms]`.
- Threading: an optional `threading` cargo feature exists `[documented, open: T004 confirms the name, the control (an environment variable, `MATMUL_NUM_THREADS`, is what I expect) and whether it can be set per call]`.
- Cost to record at T008: crate count in the lock diff, `cargo audit`, Windows/macOS-arm64 build.

## R4. Candidate: `gemm` (faer lineage)

- Pure Rust, f16/f32/f64, no-std capable; MIT; release 0.19 dated 2025-11-14 `[documented]`.
- Runtime CPU detection through `raw-cpuid` and `pulp`; dependencies `gemm-common`, `gemm-f32`, `gemm-f64` `[documented]`.
- API: `gemm::gemm` is an `unsafe fn` with a `Parallelism` argument (`None` or a Rayon variant) `[open: T005 confirms]`; the parallelism is per call, which makes the thread switch explicit.
- `candle`'s CPU matmul uses this stack `[documented]`, so a result here is also a result for candle's CPU path (not measured separately).
- MSRV against 1.94.1 and Windows/macOS-arm64 builds `[open: T005, T009]`.

## R5. Candidate: `faer`

- Pure Rust linear algebra; MIT; its home moved to Codeberg `[documented]`; uses SIMD explicitly and dispatches precompiled variants by detected CPU features `[documented]` (its paper).
- API: a safe `matmul` over matrix views with a `Par` argument (`Seq` or Rayon) is expected `[open: T006 confirms, including whether a transposed `b` is a zero-cost view]`; if it is safe, no `unsafe` is needed for this candidate.
- Largest dependency tree of the three `[inference]`; the lock diff at T008 decides.
- Both `gemm` and `faer` descend from the same author's work; results can be close, which is a finding rather than a duplicate.

## R6. Excluded: native BLAS and oneDNN/oneMKL through FFI

| Library | Licence | Rust path | Why excluded |
|---|---|---|---|
| OpenBLAS | BSD-3-Clause | `openblas-src` | native build, `DYNAMIC_ARCH` runtime kernels; FFI `unsafe`; summation order depends on thread count |
| BLIS | BSD-3-Clause | `blis-src`, `blis-sys2` | build configuration chosen at build time; same FFI and determinism points |
| oneDNN | Apache-2.0 | `onednn-sys`, `onednn-src` (0.0.x, low activity) | native CMake build; JIT per detected ISA; bindings immature |
| Intel oneMKL | ISSL | `intel-mkl-src` | strongest sgemm in class `[documented]`; proprietary licence and redistribution terms |

All `[documented]` in the survey. Exclusion reasons that hold without a run: a native toolchain on the Windows measurement machine and CI, `unsafe` FFI outside the allowed pattern, and the determinism row of `docs/tensor-runtime.md` (BLAS/oneDNN summation order varies with thread count and library version, which re-opens the raya:006 tolerances). Excluded, not rejected: HD-6.

## R7. Determinism and the tolerance consequence

Any candidate reorders accumulation, so results are not bit-equal to `Reference`. Per-element error versus an exact result is bounded by about `k * eps * sum |a*b|` for any ordering; the gate uses twice that bound against `Reference`, so it accepts any correct ordering and rejects a wrong kernel `[inference]`. A threaded candidate may also differ run to run; the golden gate is run more than once per candidate and thread setting, and a difference between runs is reported (HD-2).

## R8. Timing method decisions

- The weights are static, so packing `b` every call is a cost a real forward would not pay for repeated calls on the same weight; the trait forces it. A pre-packed variant would change the trait and is a hypothesis in the note, not part of this spec.
- Interleaved order and a reverse second pass guard against thermal and boost drift on a 65 W desktop part; the 10% drift flag is a choice, not a standard.
- p50 is the primary statistic, as in the Laya reports the researcher already reads.
- GFLOP/s is reported against `2*m*n*k`; it is a derived number, not a hardware-peak claim, and no peak is asserted for the i7-8700 here.

## R9. Open items resolved by tasks

| Item | Task |
|---|---|
| Real head feed-forward width and `act_head` input width | T001 |
| `matrixmultiply` transposed-b stride mapping, thread control | T004 |
| `gemm::gemm` signature, `Parallelism`, MSRV | T005 |
| `faer` matmul API, transposed view, `Par`, safe or not | T006 |
| New-crate counts, audit, build on three platforms | T008, T009 |
| Smallest change to select a backend in `raya-core` | T007 |
