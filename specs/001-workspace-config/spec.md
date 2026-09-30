# Feature Specification: Rust Workspace, CI and Config Types

**Feature Branch**: `p1-workspace` (specification directory only; no speckit branch created)

**Created**: 2026-09-30

**Status**: Draft

**Origin**: Prompt P1 of the former `docs/agent-prompts.md`.

**Depends on**: nothing. **Blocks**: 003, 005, 007.

**Input**: User description: "Create a Cargo workspace with raya-core, raya-kernels, raya-weights, raya-tokenizer and raya-cli, CI on x86_64 and arm64, and config types that parse Laya's model config and express the future segment/attention shape without implementing anything beyond Laya's shape."

## User Scenarios & Testing *(mandatory)*

The user is the researcher (and later agents) who needs a stable skeleton so every later feature lands in a known crate with enforced lint, safety and CI rules.

### User Story 1 - Buildable, Gated Workspace (Priority: P1)

A contributor clones the repo, runs the documented build and test commands, and gets a green build on Linux and macOS. CI enforces formatting, lints, tests and a dependency audit on every PR.

**Why this priority**: Every other feature depends on a workspace that builds and on CI catching regressions.

**Independent Test**: Fresh clone, build and test; push a branch and observe CI on both platforms.

**Acceptance Scenarios**:

1. **Given** a fresh clone, **When** the documented build and test commands run, **Then** all five crates build and tests pass.
2. **Given** a PR, **When** CI runs, **Then** fmt, `clippy -D warnings`, tests (ubuntu-latest x86_64, macos-14 arm64) and the dependency audit all run, with dependency caching.
3. **Given** any crate other than `raya-kernels`, **When** it contains `unsafe`, **Then** the build fails.
4. **Given** `raya info`, `raya predict`, `raya bench`, **When** invoked, **Then** each exists as a stubbed subcommand that reports it is not implemented.

---

### User Story 2 - Laya Config Parses into Raya Types (Priority: P2)

The researcher loads a Laya model config and gets typed fields: layer count, hidden size, heads, intermediate size, global-attention layer pattern, local window, RoPE thetas (global and local), norm epsilon, decision-head layers and marker tokens. Laya's shape converts to a single segment with no compression.

**Why this priority**: The model struct, weights validation and forward pass all read these types.

**Independent Test**: Parse a checked-in config JSON and assert every field; convert to the segment form and assert one segment, no compression.

**Acceptance Scenarios**:

1. **Given** a Laya config JSON under `tests/data/`, **When** parsed, **Then** all fields listed above are populated and validated.
2. **Given** a config with a missing or inconsistent field, **When** parsed, **Then** the error names the field.
3. **Given** Laya's shape, **When** converted to `Segment { layers, attention: AttentionKind }`, **Then** it yields one segment and `Compression::None`.
4. **Given** an `AttentionKind` other than the supported ones, or any other `Compression`, **When** requested, **Then** `Error::Unimplemented` is returned.

## Clarifications

### Session 2026-09-30

- Q: Which files does "Laya's model config" mean? -> A: `encoder/config.json` (Hugging Face `ModernBertConfig`) plus `rl_agent_config.json`, read from a checkpoint directory (plan research R1).
- Q: Where do "marker tokens" come from? -> A: the `[MASK]` marker id is a tokenizer property, not in either config file. The config records the special-token ids the encoder config has (pad, cls, sep, bos, eos); feature 005 resolves the marker.
- Q: Laya interleaves global and local layers, but `Segment` holds one `AttentionKind`. -> A: add `AttentionKind::LayaInterleaved { global_every, window }` as the one supported kind. `Full`, `BlockLocal`, `Gas`, `Hybrid` stay defined and are `Unimplemented` (research R3). This adds one variant to FR-004.
- Q: Do architecture fields have defaults? -> A: no; missing required fields are errors naming the JSON key (research R5).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Workspace at repo root with crates `raya-core` (config, errors, model shell), `raya-kernels` (empty modules and traits), `raya-weights`, `raya-tokenizer` (wraps the Hugging Face `tokenizers` crate), `raya-cli` (binary `raya`).
- **FR-002**: Rust stable, edition 2021, pinned `rust-toolchain.toml`, workspace-level lints; `#![forbid(unsafe_code)]` in every crate except `raya-kernels`.
- **FR-003**: Config types cover the fields named in User Story 2, derived from `reference/laya-sparse/laya/` and, once feature 002 exists, its exported config in `tests/data/`.
- **FR-004**: Future-facing types from `docs/feasibility.md` section 4.2: `Segment`, `AttentionKind::{Full, BlockLocal{w}, Gas{s}, Hybrid}` plus `LayaInterleaved{global_every, window}` (see Clarifications), and `Compression` with the single variant `None`.
- **FR-005**: CI per User Story 1 scenario 2, with caching.
- **FR-006**: `README.md` states what Raya is, the scope limits of the constitution, and build/test commands.
- **FR-007**: Dependencies limited to serde, serde_json, thiserror, safetensors, tokenizers, clap, criterion (dev) and anyhow (raya-cli only).
- **FR-008**: No model math.

### Key Entities

- **ModelConfig**: Laya's architecture description.
- **Segment / AttentionKind / Compression**: the configurable shape; only the Laya instance is implemented.

## Success Criteria *(mandatory)*

- **SC-001**: Clean clone builds and tests green on both CI platforms.
- **SC-002**: Config tests cover every field and every error path above.
- **SC-003**: No dependency outside FR-007 appears in any `Cargo.toml`.

## Assumptions

- The config JSON checked in for tests is a minimal Laya-shaped sample until feature 002's export exists.
- Windows builds are supported but not part of CI (see feature 008).
