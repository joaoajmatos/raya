# Feature Specification: Golden Fixtures from Laya

**Feature Branch**: `raya/p2-golden-fixtures` **in laya-sparse** (specification directory only; no speckit branch created)

**Created**: 2026-09-30

**Status**: Draft

**Origin**: Prompt P2 of the former `docs/agent-prompts.md`.

**Runs in**: `laya-sparse` (Python). Output is copied into Raya's `tests/data/` by hand, with the laya-sparse SHA in each `meta.json`.

**Depends on**: nothing. **Blocks**: 003, 006, and optionally 007.

**Input**: User description: "Export, from Laya's tiny fixture model, the contract Raya's Rust tests verify against: state_dict inventory, per-layer activations, final probabilities, and kernel reference outputs. Offline and deterministic."

## User Scenarios & Testing *(mandatory)*

The user is the Raya implementer, who needs ground truth that localizes a divergence to a layer, a tensor or a kernel.

### User Story 1 - Weight Inventory (Priority: P1)

The implementer gets a JSON listing every state_dict key with shape and dtype, for the fixture model (and, behind a flag, a real local checkpoint).

**Why this priority**: The Rust key mapping and loader validation are driven by it.

**Independent Test**: Run the inventory command; compare to the model's `state_dict()`.

**Acceptance Scenarios**:

1. **Given** the fixture model, **When** `golden weights-inventory` runs, **Then** `inventory.json` lists every key with shape and dtype.
2. **Given** no flag, **When** tests run, **Then** no real checkpoint is touched.

---

### User Story 2 - Forward-Pass Goldens (Priority: P1)

For each case (short: fits one local window; medium: several blocks crossing global/local layer boundaries; padded batch-of-one; decision head with PyTorch fast path forced off) the implementer gets safetensors files with input ids, attention inputs, embedding output, every encoder layer output, final encoder output, decision-head per-layer outputs, final logits and probabilities, the fixture weights, and a `meta.json` (config, seed, torch version, git commit).

**Why this priority**: These are the contract for the reference forward pass.

**Independent Test**: Export to a temp dir; check file presence and shapes; re-run and compare bytes.

**Acceptance Scenarios**:

1. **Given** the fixture model, **When** `golden export` runs, **Then** each case directory holds all tensors above and `meta.json`.
2. **Given** the fast-path-off case, **When** exported, **Then** outputs equal the fast-path-on outputs within 1e-6 or export fails.
3. **Given** a repeated run, **When** compared, **Then** output is byte-identical (fixed seeds, fp32, eval mode, single thread).

---

### User Story 3 - Kernel Goldens (Priority: P2)

The implementer gets random q/k/v inputs with the native-layer reference output and the existing PyTorch kernel output, for the block-local (block 128, +/-64) and gather-attend-scatter kernels at lengths 256, 1000, 2048, 16 heads, head dim 64.

**Why this priority**: Feature 007 can test against PyTorch output, but has a naive oracle without it.

**Independent Test**: Run `golden kernels`; check the recorded max abs diff.

**Acceptance Scenarios**:

1. **Given** each length, **When** exported, **Then** the record holds inputs, native reference, kernel output and max abs diff.
2. **Given** a diff above 1e-5, **When** exporting, **Then** the command fails.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: New module `experiments/golden/`, invoked as `python -m experiments golden ...` (`experiments/` is at the laya-sparse repo root, not under `laya/`).
- **FR-002**: Subcommands `weights-inventory`, `export`, `kernels`.
- **FR-003**: Everything runs offline on the existing tiny fixture model from `tests/experiments/`.
- **FR-004**: Output goes to `artifacts/raya-golden/`; only tiny-model cases (under 5 MB total) are committable, everything else git-ignored.
- **FR-005**: Offline pytest in `tests/experiments/` covers file presence, shapes and byte-identical re-runs.
- **FR-006**: Decision-head goldens include the `type_emb` addition, marker gathering, `scorer` logits, and the `act_head` logits, since Laya's `DecisionModel.forward` returns both.

## Success Criteria *(mandatory)*

- **SC-001**: Pytest passes offline within the repo's normal test budget.
- **SC-002**: Two exports produce identical bytes.
- **SC-003**: Committed fixtures total under 5 MB.

## Assumptions

- `DecisionModel.forward` in `laya/common.py` is the source of truth for what "decision-head outputs" means.
- The former prompt referenced `docs/raya-feasibility.md`; the file is `docs/feasibility.md` in Raya.
