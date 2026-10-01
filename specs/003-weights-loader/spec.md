# Feature Specification: Weights Loader

**Feature Branch**: `003-weights-loader`

**Created**: 2026-09-30 (draft); updated 2026-10-01 against `main` at `99687aa`

**Status**: Draft

**Origin**: Prompt P3 of the former `docs/agent-prompts.md`.

**Depends on**: [001](../001-workspace-config/spec.md) (merged: `raya-core` config types and `Error`), [002](../002-golden-fixtures/spec.md) (merged: vendored `tests/data/weights-h64.safetensors`, `weights-h128.safetensors`, `inventory.json` from laya-sparse `cefb300`). **Blocks**: [006](../006-reference-forward/spec.md).

**Input**: User description: "Load a Laya safetensors checkpoint for Raya-L (Laya shape, fp32): one key-mapping table that matches the vendored inventory exactly, shape validation against the config, loud failure on any mismatch, a `WeightStore` of zero-copy fp32 slices, and `raya info`. No model math, no compression, no training."

## Context: what exists on `main`

- `raya-core` provides `ModelConfig` (`from_dir`, `from_json`), `ModelShape`, `HeadConfig` and `Error { Config, Io, Json, Unimplemented }`. `raya-weights` is an empty crate that re-exports `safetensors::SafeTensors`. `raya info` is a stub that exits 2.
- A Laya checkpoint directory holds `model.safetensors`, `rl_agent_config.json` and `encoder/config.json` (`laya/agent.py` in the reference submodule). The vendored fixtures are flat instead: weights in `tests/data/weights-h64.safetensors` (config in `tests/data/fixture-tiny/`) and `weights-h128.safetensors` (config embedded in `tests/data/fastpath_off/meta.json`).
- `tests/data/inventory.json` (h64 fixture, 3 encoder layers, 1 decision-head layer) lists 44 keys, all `float32`, 210822 parameters. Its groups: `encoder.embeddings.{tok_embeddings,norm}`, `encoder.layers.N.{attn.Wqkv,attn.Wo,mlp.Wi,mlp.Wo,mlp_norm}`, `encoder.layers.N.attn_norm` for N >= 1 only (layer 0 has none), `encoder.final_norm`, `head.layers.N.*` (self-attention in/out projections, two linears, two norms with weight and bias), `type_emb`, `scorer.{0,1,3}`, `act_head.{0,2}`, `temperature`. There is no decoder or classifier tensor (embeddings are tied and unused).
- Tensor widths scale with `hidden_size` (64 and 128 in the fixtures), so shapes must be derived from the config, not copied from the inventory.

## Clarifications

### Session 2026-10-01

Resolved by the implementing agent with defaults (no user questions asked).

- Q: Memory-map the file or read it once? → A: Read once into a single owned, 4-byte-aligned buffer; tensors are borrowed views of it. `memmap2` needs `unsafe`, which is reserved for `raya-kernels` (constitution, FR-008). Memory-mapping is deferred to a later feature that can place it in `raya-kernels` or amend the rule.
- Q: What happens to bf16/fp16 tensors? → A: Rejected with `Error::Unimplemented` naming tensor and dtype. Constitution IV allows conversion on load but this feature stays strictly fp32; see Human-owned items.
- Q: What does the loader take as input? → A: A weights file path plus an already-built `ModelConfig` (fixtures are flat and h128's config lives in `meta.json`), and a convenience `load_dir` for the `model.safetensors` + `rl_agent_config.json` + `encoder/config.json` layout used by `raya info`.
- Q: How are tensors addressed? → A: By a typed key (parameter kind plus layer index), not free strings; the mapping table is the only place Laya string names appear.
- Q: How are loader failures reported? → A: One new `Error` variant carrying all mismatches (missing, unused, shape, dtype, duplicate), so a caller sees every problem in one run; I/O and parse failures carry the path.
- Q: Sharded checkpoints? → A: Out of scope; Laya writes a single `model.safetensors`.
- Q: Pluggable GEMM packing hook? → A: Dropped; no consumer, SIMD target undecided.
- Q: Config variants (e.g. zero head layers)? → A: The mapping follows `ModelConfig`/`ModelShape`; whatever `ModelShape::ensure_supported` rejects is rejected before any file is read.

### Human-owned items (not decided here)

- Whether bf16/fp16 checkpoints must load via fp32 conversion (precision scope beyond fp32; Constitution IV permits it, the brief for this feature excludes it).
- Target CPU and any SIMD-oriented weight layout/packing (open; the packing hook is dropped until decided).

## User Scenarios & Testing *(mandatory)*

The user is the Raya implementer (feature 006 consumes the store) and the researcher inspecting a checkpoint.

### User Story 1 - Load Laya Weights Exactly (Priority: P1)

A Laya `model.safetensors` plus its config loads into a `WeightStore`. Every tensor is available by Raya's internal name as a fp32 slice with its shape, and values are bit-identical to the file.

**Why this priority**: No forward pass is possible without weights.

**Independent Test**: Load `weights-h64.safetensors` with the `fixture-tiny` config and `weights-h128.safetensors` with its h128 config; compare every tensor bit for bit with the file.

**Acceptance Scenarios**:

1. **Given** the h64 and h128 fixture weights and their configs, **When** loaded, **Then** all 44 tensors are available by internal name with the config-derived shapes.
2. **Given** a loaded store, **When** any tensor is compared with the source file, **Then** values are bit-identical.
3. **Given** a loaded store, **When** tensors are read, **Then** no per-tensor copy is made (slices borrow from the store's single buffer).
4. **Given** a checkpoint whose tensors are not fp32, **When** loaded, **Then** loading fails with `Error::Unimplemented` naming the tensor and dtype (no conversion path in this feature).

---

### User Story 2 - Fail Loudly on Mismatch (Priority: P1)

Any missing, unused, wrongly shaped or wrongly typed tensor aborts loading with a report that names every offender.

**Why this priority**: Silent weight mismatches would invalidate every parity result.

**Independent Test**: Negative tests built from the fixture (drop a key, add a key, change a shape, change a dtype); the vendored files are never modified, mutated copies are made in a temp dir.

**Acceptance Scenarios**:

1. **Given** `inventory.json`, **When** the mapping table is checked for the h64 config, **Then** every inventory key is mapped exactly once and every Raya parameter is covered.
2. **Given** a checkpoint with missing or extra tensors, **When** loaded, **Then** the error lists both the missing and the unused names.
3. **Given** a tensor whose shape differs from the config-derived shape, **When** loaded, **Then** the error names the tensor, the expected shape and the found shape.
4. **Given** a missing or unreadable file, or a malformed safetensors header, **When** loaded, **Then** the error names the path.
5. **Given** a config the model shape rejects (unsupported variant), **When** loaded, **Then** the shape's own error is returned and no file is read.

---

### User Story 3 - Inspect a Checkpoint (Priority: P2)

`raya info <checkpoint-dir>` prints the config summary, the number of tensors, the parameter count computed from the loaded tensors, and the total bytes.

**Why this priority**: The feasibility doc warns that parameter counts were quoted without computation.

**Independent Test**: Run on a directory assembled from the h64 fixture; the parameter count must equal 210822 (the inventory's `n_parameters`) and the sum of tensor sizes.

**Acceptance Scenarios**:

1. **Given** a checkpoint dir, **When** `raya info` runs, **Then** it prints config, tensor count, computed parameter count and total bytes, never a number quoted from docs.
2. **Given** a bad checkpoint, **When** `raya info` runs, **Then** it prints the loader's report and exits non-zero.

### Edge Cases

- Layer 0 has no `attn_norm`; layers 1 and up do. The mapping must follow the config, not a fixed list of 44.
- The decision head's attention uses `in_proj_weight` / `in_proj_bias` (one fused tensor), unlike the encoder's `Wqkv`; shapes come from `HeadConfig`.
- `temperature` has 3 entries and `type_emb` has 3 rows in the fixtures; both are part of the inventory and must be loaded.
- A zero-length tensor or a duplicate key in the file is an error, not a skip.
- Extra metadata in the safetensors header (`__metadata__`) is ignored, not reported as an unused tensor.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Load a single `model.safetensors` (weights path plus a `ModelConfig`, and a directory convenience) for Raya-L; only fp32 tensors are accepted. Any other dtype returns `Error::Unimplemented`. Sharded checkpoints are out of scope (Laya writes one file).
- **FR-002**: One key-mapping table in code from Laya `state_dict` names to Raya typed keys (parameter kind plus layer index), parameterized by the config (layer count, layer 0 without `attn_norm`, head layer count). It is the only place Laya names appear.
- **FR-003**: Shape validation against the `raya-core` config with tensor-named errors; the expected shape of each tensor is derived from the config, not from the inventory.
- **FR-004**: Errors use the shared `raya_core::Error`; one added variant carries the full report (missing, unused, shape, dtype, duplicate). Loading is all-or-nothing and reports every mismatch in one error.
- **FR-005**: `WeightStore` exposes `&[f32]` plus shape by internal name, with no per-tensor copy. Memory is released when the store drops.
- **FR-006**: `raya info <dir>` per User Story 3, reading the directory layout `model.safetensors` + `rl_agent_config.json` + `encoder/config.json`.
- **FR-007**: No model math, no packing/repacking of weights, no tied or derived tensors synthesized. `reference/` and vendored fixtures are never modified; tests use temp copies.
- **FR-008**: No `unsafe` in `raya-weights` or `raya-cli` (constitution: `unsafe` only in `raya-kernels`).

### Key Entities

- **Key mapping table**: ordered list of (Laya name, Raya internal name, shape rule) built from the config.
- **WeightStore**: owns the file's data; maps internal name to fp32 slice and shape.
- **Load report**: the aggregate error listing missing, unused, mis-shaped and mis-typed tensors.

## Success Criteria *(mandatory)*

- **SC-001**: Positive (h64, h128) and negative loader tests pass offline with no network and no real checkpoint.
- **SC-002**: A coverage test proves no inventory key is unmapped or double-mapped for the h64 config, and that the set of mapped names equals the inventory's 44 keys.
- **SC-003**: `raya info` on the h64 fixture prints a parameter count of 210822, equal to both the inventory's `n_parameters` and the sum of loaded tensor sizes.
- **SC-004**: Every negative scenario in User Story 2 produces an error that names the offending tensor or path.
- **SC-005**: `cargo fmt --check`, `cargo clippy -D warnings` and `cargo test --workspace` pass; vendored files are byte-unchanged (`git diff` on `tests/data/` and `reference/` is empty).

## Assumptions

- Laya's decision-head parameters (`type_emb`, `scorer`, `act_head`, `temperature`) are part of the inventory and must be mapped.
- Reading the file into memory once is acceptable. True memory-mapping needs `unsafe`, which the constitution reserves for `raya-kernels`, so it is deferred (see Clarifications).
- The pluggable GEMM-packing hook from the earlier draft is dropped: no kernel consumes it yet and the SIMD target is an open human decision. Feature 006 or a kernel feature may add it.
- bf16/fp16 conversion on load (allowed by constitution principle IV) is not part of this feature; adding it is a scope decision for the human.
- The h128 config is taken from the vendored `tests/data/fastpath_off/meta.json` `encoder_config`/`agent_config`; no new fixture is needed.
