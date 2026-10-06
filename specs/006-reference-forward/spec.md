# Feature Specification: Reference Forward Pass (Correctness Only)

**Feature Branch**: `006-reference-forward`

**Created**: 2026-09-30

**Status**: Implemented

**Origin**: Prompt P5 of the former `docs/agent-prompts.md`.

**Depends on**: 002, 003, 005. **Blocks**: 008.

**Input**: User description: "A correct, unoptimized Raya-L forward pass in raya-core, verified layer by layer against Laya's golden fixtures, with a tensor-runtime recommendation, a trace facility, an AttentionKernel trait, and a working `raya predict`."

## Clarifications

### Session 2026-10-07

Answered from this spec, the constitution and the pinned Laya reference (`357ad53`); none needed the researcher.

- Q: What does `raya predict --question` take, given that a Laya question has a type, instructions and options? -> A: A Laya question definition as JSON (`{"type": "choice", "instructions": ..., "criteria": ...}`), passed inline or as a path to a JSON file; `--document` is a text file used as the state string. The model directory holds `model.safetensors`, `rl_agent_config.json`, `encoder/config.json` and `tokenizer.json`; `--tokenizer` overrides the tokenizer path (the vendored `fixture-h*` directories carry no tokenizer).
- Q: Where does temperature calibration happen, in the forward or after it? -> A: After. Laya's `DecisionModel.forward` returns raw marker logits and `act_logits`; calibration is `Agent._decode_answers`: `z = logits[:k] / T`, with `T = temperature_by_options[temp_bucket(qtype, k)]` falling back to `temperature[qtype]`, each clamped to [0.5, 5.0] (non-finite becomes 1.0), then a max-shifted softmax. Raya ports that as a separate calibration step. The golden `probabilities` tensor is `softmax(logits)` without temperature (the fixture's temperatures are all 1.0), and `act_probability` is `softmax(act_logits)[0]`.
- Q: How does the existing `AttentionKernel` (spec 007, merged) express the encoder's global and local layers and the head's key-padding mask? -> A: Adopt the 007 trait and extend `Mask` additively with `Full` and `Sliding(half_window)` (`|i - j| <= half_window`, where `half_window = local_attention / 2`, 8 for the fixture and 64 for the real checkpoint). A dense reference kernel implements them. The tiled kernel returns `KernelError::Unsupported` for the new variants. Padding is the trait's `valid_len` prefix; an attention mask that is not a prefix of ones is `Error::Unimplemented`.
- Q: Which positions are compared in the padded case, and may the tolerances change? -> A: Rows `0..valid_len` only (the 007 trait zeroes padded query rows, whereas Laya computes them; they never influence valid rows). Tolerances start at 1e-4 absolute for layer outputs and 1e-5 for probabilities and are tightened to the measured error only with the measurement recorded; any gap is reported, not hidden.
- Q: Which tensor runtime does 006 use? -> A: Plain Rust loops in `raya-kernels` behind the existing `GemmBackend` trait, no new dependency, no threading. `docs/tensor-runtime.md` compares candle, ndarray plus BLAS and plain loops on build simplicity and on calling our own attention kernel; large-GEMM performance claims are labeled hypotheses because no run exists. Precision (fp32) and target CPU are untouched.

## User Scenarios & Testing *(mandatory)*

The user is the researcher, who needs a Rust model that provably computes what Laya computes.

### User Story 1 - Layer-by-Layer Parity (Priority: P1)

For every golden case, Raya's embedding output, each encoder layer output, each decision-head layer output and final probabilities match Laya within tolerance. A failing layer is reported by index.

**Why this priority**: This is the correctness baseline for every optimization.

**Independent Test**: Run the golden comparison suite on all cases.

**Acceptance Scenarios**:

1. **Given** each golden case, **When** the forward runs with tracing, **Then** embedding and every encoder layer output match within 1e-4 absolute (tightened if achievable, with justification).
2. **Given** the decision head, **When** run, **Then** head layer outputs match and probabilities match within 1e-5.
3. **Given** a divergence, **When** a test fails, **Then** the message names the first divergent layer index.
4. **Given** the padded case, **When** run, **Then** padding does not change valid-token outputs relative to the golden.

---

### User Story 2 - Runnable Prediction (Priority: P2)

`raya predict --model <dir> --document <file> --question <text>` tokenizes, runs the forward and prints calibrated probabilities.

**Why this priority**: Gives an end-to-end path that feature 008 benchmarks.

**Independent Test**: Run on the fixture model and compare with the golden probabilities.

**Acceptance Scenarios**:

1. **Given** a model dir, document file and question, **When** `raya predict` runs, **Then** it prints probabilities equal to Laya's for the same input within tolerance.

---

### User Story 3 - Pluggable Attention and Documented Runtime Choice (Priority: P3)

Attention is invoked through an `AttentionKernel` trait that the dense reference implements, so later kernels plug in unchanged. `docs/tensor-runtime.md` compares candidate tensor runtimes and recommends one.

**Why this priority**: Decouples feature 007 and protects model code from kernel churn.

**Independent Test**: Swap a stub kernel in a test without touching model code; read the doc.

**Acceptance Scenarios**:

1. **Given** the trait, **When** a second implementation is supplied, **Then** the model runs with it unchanged.
2. **Given** `docs/tensor-runtime.md`, **When** read, **Then** it compares (a) `candle`, (b) ndarray plus a BLAS binding, (c) plain Rust loops on Windows/macOS/Linux build simplicity, large-GEMM performance and ability to call our own attention kernel, and recommends one.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Implement embeddings (token embedding then bias-free LayerNorm, `norm_eps`), RoPE (separate theta for global and local layers, rotate-half form), norms as in Laya (no attention norm in layer 0, final norm), gated MLP (exact-erf GELU on the first half times the second half), dense attention with the global pattern and the local +/-window mask (mask on dense scores is acceptable here), the configured layer pattern, the 2-layer decision head over the full sequence, marker-token scoring, and temperature-calibrated probabilities (calibration is a separate step after the forward, see Clarifications). The decision head follows `nn.TransformerEncoderLayer(norm_first=True)`: LayerNorm eps 1e-5, ReLU feed-forward, `max(1, hidden / 64)` heads.
- **FR-002**: Reproduce the whole `DecisionModel.forward` output: `type_emb` addition, the head layers with key-padding mask, marker gathering, `scorer` logits masked to marker validity, and the `act_head` logits from pooled `h[:, 0]` plus top-1, top-1 minus top-2, entropy and option-count features (single-option case included).
- **FR-003**: Forward trace facility capturing every intermediate activation.
- **FR-004**: Tensor use sits behind a small internal trait so the runtime can be swapped.
- **FR-005**: `AttentionKernel` trait shared with feature 007 (merged): adopt its definition and extend `Mask` additively with `Full` and `Sliding(half_window)`; the tiled kernel rejects those variants.
- **FR-006**: No optimization and no threading beyond the GEMM library default.
- **FR-007**: fp32 only; the decision head uses standard (non-fast-path) attention semantics, which Laya shows equal to the fast path within 5e-7.

## Success Criteria *(mandatory)*

- **SC-001**: All golden cases pass at the stated tolerances.
- **SC-002**: `raya predict` output matches golden probabilities within 1e-5.
- **SC-003**: A failing comparison names the case, the tensor and the first divergent layer index.

## Assumptions

- Tolerances start at 1e-4 (layers) and 1e-5 (probabilities) and are tightened only if the measured error allows.
- `act_head` logits are in scope even though the former prompt listed only marker scoring, because Laya's forward returns them.
