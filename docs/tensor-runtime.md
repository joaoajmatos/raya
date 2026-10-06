# Tensor runtime for the reference forward pass

**Status:** decision for spec 006 (reference forward pass). **Decision:** plain Rust loops behind the existing `GemmBackend` trait in `raya-kernels`, no new dependency, no threading.

Scope: this compares ways to do the dense linear algebra and tensor plumbing of the Raya-L forward pass (fp32, CPU). It does not choose a target CPU or a precision; both remain human decisions (constitution, "Workflow"). Every performance statement below is a **hypothesis**: no run of any candidate exists in this repository, and no timing was taken for 006 (constitution III). The only measured facts are the parity numbers in `specs/006-reference-forward/quickstart.md`.

## What the forward pass needs

- `out = x * W^T + b` with `W` as stored by Laya (`[out, in]`, row-major): projections, MLP, scorer, `act_head`.
- LayerNorm, softmax, exact GELU, ReLU, RoPE: elementwise or row-wise, no library needed.
- Attention through Raya's own `AttentionKernel` trait (spec 007), so the runtime must not own attention.
- Weights are already fp32 slices from `raya-weights` (`WeightStore`), so no tensor type is required to hold them.

## Candidates

| | (a) `candle` | (b) `ndarray` + BLAS binding | (c) plain Rust loops (chosen) |
|---|---|---|---|
| **Build simplicity on Windows / macOS / Linux** | Pure-Rust CPU path builds with cargo alone; optional MKL / Accelerate backends add native-library setup. Adds a large dependency tree (hypothesis: noticeable compile time, to be measured if it is ever adopted). | The BLAS crates (`openblas-src`, `blas-src` with MKL or Accelerate) need a system or vendored BLAS; this is the usual source of Windows and CI friction. `ndarray` alone is pure Rust. | Nothing to install; identical on all three platforms and on the project's CI (ubuntu-latest, macos-14) and the Windows measurement machine. |
| **Large-GEMM performance** | Hypothesis: competitive for large matrices on its optimized backends, below a tuned BLAS; unverified. | Hypothesis: best of the three for large GEMM when linked to a tuned BLAS (MKL, OpenBLAS, Accelerate); unverified, and depends on the BLAS and the CPU. | Hypothesis: clearly the slowest (no blocking, no SIMD in this path); acceptable because 006 is a correctness oracle with no speed goal (spec FR-006). Not measured. |
| **Calling our own attention kernel** | Possible but requires converting between `Tensor` and the `&[f32]` buffers of `AttentionKernel`; the tensor type leaks into model code unless wrapped. | Natural: `ndarray` views expose `&[f32]` slices; BLAS only covers GEMM, so attention stays ours. | Natural: activations are already `&[f32]` / `Vec<f32>` row-major, which is exactly what the kernel trait takes. |
| **Dependencies (constitution VII)** | New large crate. | New crates plus a native library. | None. |
| **Threading (spec FR-006)** | Multi-threaded by default on some backends; "no threading beyond the GEMM library default" is then the library's choice. | BLAS libraries are usually multi-threaded by default; same remark. | Single-threaded, deterministic. |
| **Determinism and error localization** | Summation order is the backend's. | Summation order is the BLAS's and may vary with thread count. | Fixed sequential order; the measured gap to Laya is reproducible. |

## Decision and rationale

Use (c) for 006. Reasons that hold without a benchmark:

1. The reference must be easy to audit against Laya's formulas and must build everywhere, including the Windows machine used for measurements, with no native toolchain.
2. The model code only needs one primitive (`matmul_nt`); a trait in `raya-kernels` already isolates it (`GemmBackend`), which satisfies spec FR-004: a BLAS or `candle` backend can later implement the trait without touching model code.
3. Parity with Laya was reached with plain loops: the worst intermediate difference over the four golden cases is 7.2e-7 and the worst probability difference 3.0e-8 (see `quickstart.md`), so the loops are not a correctness limitation.

## When to revisit

- Feature 008 (benchmark harness) will run on the target CPU, which is a human-owned step. If whole-forward time there is dominated by the linear layers, implement `GemmBackend` with (b) or with Raya's own SIMD micro-kernels (`raya-kernels`, constitution V: runtime-detected AVX2, portable fallback) and compare with a recorded run. Until then "BLAS is faster than loops" stays a hypothesis for this code base.
- Precision (fp32 only today) and the target CPU are not affected by this decision.
