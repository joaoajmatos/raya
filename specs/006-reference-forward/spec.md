# Feature Specification: Reference Forward Pass (Correctness Only)

**Feature Branch**: `p5-forward` (specification directory only; no speckit branch created)

**Created**: 2026-09-30

**Status**: Draft

**Origin**: Prompt P5 of the former `docs/agent-prompts.md`.

**Depends on**: 002, 003, 005. **Blocks**: 008.

**Input**: User description: "A correct, unoptimized Raya-L forward pass in raya-core, verified layer by layer against Laya's golden fixtures, with a tensor-runtime recommendation, a trace facility, an AttentionKernel trait, and a working `raya predict`."

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

- **FR-001**: Implement embeddings, RoPE (separate theta for global and local layers), norms as in Laya, gated MLP, dense attention with the global pattern and the local +/-window mask (mask on dense scores is acceptable here), the configured layer pattern, the 2-layer decision head over the full sequence, marker-token scoring, and temperature-calibrated probabilities.
- **FR-002**: Reproduce the whole `DecisionModel.forward` output: `type_emb` addition, the head layers with key-padding mask, marker gathering, `scorer` logits masked to marker validity, and the `act_head` logits from pooled `h[:, 0]` plus top-1, top-1 minus top-2, entropy and option-count features (single-option case included).
- **FR-003**: Forward trace facility capturing every intermediate activation.
- **FR-004**: Tensor use sits behind a small internal trait so the runtime can be swapped.
- **FR-005**: `AttentionKernel` trait shared with feature 007 (if 007 merged first, adopt its definition).
- **FR-006**: No optimization and no threading beyond the GEMM library default.
- **FR-007**: fp32 only; the decision head uses standard (non-fast-path) attention semantics, which Laya shows equal to the fast path within 5e-7.

## Success Criteria *(mandatory)*

- **SC-001**: All golden cases pass at the stated tolerances.
- **SC-002**: `raya predict` output matches golden probabilities within 1e-5.

## Assumptions

- Tolerances start at 1e-4 (layers) and 1e-5 (probabilities) and are tightened only if the measured error allows.
- `act_head` logits are in scope even though the former prompt listed only marker scoring, because Laya's forward returns them.
