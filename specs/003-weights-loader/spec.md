# Feature Specification: Weights Loader

**Feature Branch**: `p3-weights` (specification directory only; no speckit branch created)

**Created**: 2026-09-30

**Status**: Draft

**Origin**: Prompt P3 of the former `docs/agent-prompts.md`.

**Depends on**: 001, 002. **Blocks**: 006.

**Input**: User description: "Memory-mapped safetensors loading with a single, loud key mapping from Laya's state_dict names to Raya's, shape validation, a zero-copy WeightStore, and `raya info`."

## User Scenarios & Testing *(mandatory)*

The user is the Raya implementer and the researcher inspecting a checkpoint.

### User Story 1 - Load Laya Weights Exactly (Priority: P1)

A Laya checkpoint (single file or sharded) loads into a `WeightStore` exposing tensors by Raya's internal name as fp32 slices with shapes. bf16/fp16 tensors convert to fp32 on load and are never kept.

**Why this priority**: No forward pass is possible without weights.

**Independent Test**: Load the fixture model weights from `tests/data/golden/*/` and compare tensors bit for bit with the file.

**Acceptance Scenarios**:

1. **Given** a single-file or sharded checkpoint, **When** loaded, **Then** all tensors are available by internal name, fp32 tensors without copying.
2. **Given** a bf16/fp16 checkpoint, **When** loaded, **Then** tensors are fp32 and no low-precision copy persists.
3. **Given** a few tensors, **When** compared with the source file, **Then** values are bit-identical.

---

### User Story 2 - Fail Loudly on Mismatch (Priority: P1)

Any missing, unused or wrongly shaped tensor aborts loading with a report.

**Why this priority**: Silent weight mismatches would invalidate every parity result.

**Independent Test**: Negative tests: missing key, wrong shape, extra key.

**Acceptance Scenarios**:

1. **Given** `inventory.json`, **When** the mapping table is checked, **Then** every inventory key is mapped exactly once and every Raya parameter is covered.
2. **Given** a checkpoint with missing or extra tensors, **When** loaded, **Then** the error lists both the missing and the unused names.
3. **Given** a tensor whose shape differs from the config-derived shape, **When** loaded, **Then** the error names the tensor and both shapes.

---

### User Story 3 - Inspect a Checkpoint (Priority: P2)

`raya info <checkpoint-dir>` prints the config, the parameter count computed from loaded tensors, and total bytes.

**Why this priority**: The feasibility doc warns that parameter counts were quoted without computation.

**Independent Test**: Run on the fixture checkpoint; compare the count with the sum of tensor sizes.

**Acceptance Scenarios**:

1. **Given** a checkpoint dir, **When** `raya info` runs, **Then** it prints config, computed parameter count and total bytes, never a number quoted from docs.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Memory-mapped loading, fp32 only in memory.
- **FR-002**: One key-mapping table in code, driven by `tests/data/inventory.json`.
- **FR-003**: Shape validation against `raya-core` config with tensor-named errors.
- **FR-004**: `WeightStore` returns `&[f32]` plus shape; a pluggable GEMM packing function runs once at load, default row-major no-op.
- **FR-005**: `raya info` per User Story 3.
- **FR-006**: No model math.

## Success Criteria *(mandatory)*

- **SC-001**: Positive and negative loader tests pass offline.
- **SC-002**: Mapping coverage test proves no inventory key is unmapped or double-mapped.

## Assumptions

- Laya's decision-head parameters (`type_emb`, `scorer`, `act_head`, `temperature`) are part of the inventory and must be mapped.
