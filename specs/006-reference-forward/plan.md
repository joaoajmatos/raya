# Implementation Plan: Reference Forward Pass (Correctness Only)

**Branch**: `006-reference-forward` | **Date**: 2026-10-07 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/006-reference-forward/spec.md`

## Summary

A plain, unoptimized fp32 port of `DecisionModel.forward` (ModernBERT encoder, `type_emb`, pre-norm decision head, marker scorer, `act_head`) in `raya-core`, with a trace facility that captures every intermediate under the golden tensor names, a calibration step ported from `Agent._decode_answers`, a working `raya predict`, and `docs/tensor-runtime.md`. Attention goes through the existing `AttentionKernel` trait (spec 007), extended additively with `Full` and `Sliding` masks and served by the scalar `Naive` kernel. Dense linear algebra is plain Rust loops behind the existing `GemmBackend` trait. Verification is layer-by-layer comparison with the four vendored forward goldens, reporting the maximum absolute difference per tensor.

## Technical Context

**Language/Version**: Rust 1.94.1 (`rust-toolchain.toml`), edition 2021

**Primary Dependencies**: no new crate in `Cargo.lock`. `raya-core` gains a path dependency on `raya-kernels` (attention and GEMM traits). `raya-cli` already depends on `raya-core` and `raya-weights` and gains `raya-tokenizer` and `serde_json` (both already in the workspace).

**Storage**: checkpoint directory read by `raya_weights::load_dir`; `tokenizer.json` read by `SequenceBuilder::from_dir`. Tests read the vendored `tests/data/{short,medium,padded,fastpath_off}/`, `weights-h64|h128.safetensors`, `fixture-h64|h128/` and `tokenizer/tokenizer.json`.

**Testing**: `cargo test --workspace`, offline. Unit tests with hand-computed tiny cases in `raya-core`; golden and CLI integration tests in `raya-cli/tests/` (the only crate that sees core, weights and tokenizer together).

**Target Platform**: Linux x86_64, macOS arm64, Windows (portable path; the AVX2 micro-kernels of 007 are untouched and unused here).

**Project Type**: library crates plus the `raya` binary

**Performance Goals**: none (FR-006, constitution III). No timing is recorded or claimed.

**Constraints**: `#![forbid(unsafe_code)]` in `raya-core` and `raya-cli`; no threading; fp32 only; vendored fixtures and `reference/` untouched; Laya behavior matched, oddities reported.

**Scale/Scope**: fixture models (hidden 64 and 128, 3 layers, 1 head layer, sequences up to 225 tokens). The code is shape-generic, so the real 28-layer checkpoint would run (slowly); that is not tested here.

## Constitution Check

| Principle | Status |
|---|---|
| I Laya is the specification | Pass. Ports `ModernBertModel.forward` (transformers 5.18 as pinned by the goldens), `DecisionModel.forward`, `nn.TransformerEncoderLayer(norm_first=True)` and the calibration of `Agent._decode_answers`. Oddities are recorded in research.md, not fixed. |
| II Parity by goldens | Pass. All four cases, every golden tensor; failures name case, tensor and first divergent layer. Tolerances 1e-4 and 1e-5 per the spec; measured gaps reported. |
| III Measure, do not assume | Pass. No performance claim; the runtime comparison labels throughput statements as hypotheses. |
| IV Raya-L, fp32 | Pass. fp32 throughout; a non-`LayaInterleaved` shape or unsupported config returns `Error::Unimplemented`. |
| V Portable first | Pass. Plain loops; no `target-cpu` flags; `unsafe` stays confined to `raya-kernels` (the changes there are safe code). |
| VI Correctness before speed | Pass. This is the oracle later kernels and the benchmark harness compare against; attention is behind the single kernel trait. |
| VII Small surface, explicit errors | Pass. No new dependency. Unsupported inputs (non-prefix attention mask, ids outside the vocabulary, marker out of range, unsupported activation or layer type) are explicit errors. |

Re-check after design: unchanged. One cross-crate refactor (moving `ParamKey` and `HeadPart` into `raya-core`, re-exported from `raya-weights`) avoids a dependency cycle and changes no public path.

## Project Structure

### Documentation (this feature)

```text
specs/006-reference-forward/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/forward-api.md
└── tasks.md           # /speckit-tasks
```

### Source Code

```text
crates/raya-kernels/src/
├── attention.rs     # Mask gains Full and Sliding(half_window); key_range/allows/name extended
├── gemm.rs          # GemmBackend::matmul_nt + Reference (plain loops)
└── tiled/mod.rs     # new masks return KernelError::Unsupported
crates/raya-core/src/
├── params.rs        # ParamKey, HeadPart (moved from raya-weights), ParamSource trait
├── ops.rs           # layer_norm, gelu (exact, own erf), relu, softmax, RoPE
├── encoder.rs       # embeddings, ModernBERT layers, final norm
├── head.rs          # pre-norm head layers, scorer, act_head, features
├── trace.rs         # Trace: ordered (name, shape, data)
├── calibrate.rs     # temperature clamp, bucket, calibrated softmax
└── model.rs         # Model::bind, Network::forward(inputs, kernel, trace)
crates/raya-weights/src/   # re-exports ParamKey/HeadPart; impl ParamSource for WeightStore
crates/raya-cli/src/predict.rs
crates/raya-cli/tests/{forward_golden.rs, predict.rs}
crates/raya-core/tests/{ops.rs, forward_tiny.rs, calibrate.rs, stub_kernel.rs}
docs/tensor-runtime.md
```

**Structure Decision**: forward code lives in `raya-core`, as the spec requires. `raya-weights` already depends on `raya-core`, so `raya-core` cannot name `WeightStore`; it defines the `ParamSource` trait and `raya-weights` implements it. Golden tests sit in `raya-cli` because it is the one crate that depends on core, weights and tokenizer without a cycle (a dev-dependency cycle would duplicate core's types).

## Design decisions

1. **Layout**: row-major `[seq, d]` activations, one sequence per forward call. A tokenizer `Batch` is run row by row with each row's `valid_len`; padding never reaches valid rows, so this equals Laya's batched result. Padded query rows are zero (007 trait contract) and are not compared.
2. **Encoder** (per `ModernBertModel`): `tok_embeddings` then LayerNorm (no bias, `norm_eps`). Per layer: pre-norm attention (layer 0 has no attention norm), `Wqkv` viewed as `[seq, 3, heads, head_dim]`, rotate-half RoPE (global theta on global layers, local theta on local layers; `inv_freq = theta^(-2i/head_dim)`, angle `pos * inv_freq`), scale `head_dim^-0.5`, mask `Full` or `Sliding(local_attention / 2)`, `Wo`, residual; pre-norm gated MLP (`Wi` output halves `input` and `gate`, `gelu(input) * gate`, `Wo`), residual. Final LayerNorm. Dropout is the identity (eval).
3. **Head** (per `nn.TransformerEncoderLayer(norm_first=True)` with `_DynamicMultiheadAttention`): `x = x + out_proj(attn(norm1(x)))` with packed `in_proj`, scale `head_dim^-0.5`, key padding as `valid_len`; `x = x + linear2(relu(linear1(norm2(x))))`; LayerNorm eps 1e-5. The fast-path equivalence (5e-7) is Laya's own claim and is exercised by the `fastpath_off` golden.
4. **Scoring**: gather marker rows (`marker_pos` clamped at 0), scorer `LayerNorm -> Linear -> GELU(exact) -> Linear`, invalid markers set to `-1e4`, softmax, features `[top1, top1 - top2, entropy / ln(max(k, 2)), k / 255]` (a single marker gives top2 = 0), `act_head` over `[h[0], features]`.
5. **Attention kernel**: `Mask::Full` and `Mask::Sliding(w)` are added to the 007 enum; `Naive` serves them (it visits only allowed keys, which equals the dense masked softmax). `Tiled` returns `Unsupported` for them. A stub-kernel test swaps in a second implementation without touching model code (US3).
6. **GEMM**: `GemmBackend::matmul_nt(out, a, b, m, n, k)` computes `a [m,k] x b[n,k]^T`; `Reference` is plain loops with sequential f32 accumulation, single-threaded.
7. **GELU**: exact (`0.5 x (1 + erf(x / sqrt 2))`), because torch's `nn.GELU()` default and `ACT2FN["gelu"]` are exact. Rust std has no `erf`, so `ops.rs` carries its own f64 implementation (Taylor series for small `|x|`, continued fraction for the tail), unit-tested against known values; no new dependency.
8. **Calibration**: `calibrate(logits[..k], qtype, k, agent_config)` per research R7. `raya predict` prints calibrated probabilities and the act probability as JSON, unrounded.
9. **Trace**: `Trace` stores tensors in execution order under the golden names (`embedding_output`, `encoder_layer_NN`, `encoder_final_norm`, `type_emb_add`, `head_input`, `head_layer_NN`, `marker_states`, `logits`, `probabilities`, `act_features`, `pooled`, `act_logits`; `encoder_output` is the same tensor as `encoder_final_norm`). Opt-in via `Option<&mut Trace>`.
10. **Golden comparison**: per tensor, the max absolute difference over valid rows; the test prints a table and asserts `<= 1e-4` for intermediates and logits and `<= 1e-5` for `probabilities` (research R9). A failure names case, tensor and, for layer tensors, the first divergent layer index.
