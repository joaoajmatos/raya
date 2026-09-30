# Feature Specification: Tokenizer and Sequence Builder

**Feature Branch**: `p4-tokenizer` (specification directory only; no speckit branch created)

**Created**: 2026-09-30

**Status**: Draft

**Origin**: Prompt P4 of the former `docs/agent-prompts.md`.

**Depends on**: 001, 002, 004 (vendored goldens). **Blocks**: 006.

**Input**: User description: "A Rust port of Laya's sequence builder in raya-tokenizer that matches Laya's marker layout, state handling, question types, truncation and padding exactly, verified against vendored goldens."

## User Scenarios & Testing *(mandatory)*

The user is the Raya implementer, who needs model inputs identical to Laya's for the same request.

### User Story 1 - Primary Workload Parity (Priority: P1)

One document plus one question produces the same input_ids, position ids and marker positions as Laya.

**Why this priority**: This is the workload every measurement and decision uses.

**Independent Test**: Compare ids and marker positions with the golden files for the single-question records, exactly.

**Acceptance Scenarios**:

1. **Given** `tokenizer.json`, **When** loaded through the `tokenizers` crate, **Then** tokenization of a fixed text matches the golden ids.
2. **Given** a document and one question, **When** the sequence is built, **Then** ids, positions and marker positions equal the golden record exactly.
3. **Given** a document longer than max_len or exactly max_len, **When** built, **Then** truncation and truncation flags match the golden record.

---

### User Story 2 - Multiple Questions, State, Edge Inputs (Priority: P2)

Several questions over one document, state handling, empty documents, unicode, and text containing marker-like strings all match Laya.

**Why this priority**: Needed for full `predict()` parity, but not for the first benchmark.

**Independent Test**: Golden comparison for each remaining record.

**Acceptance Scenarios**:

1. **Given** several questions of types choice, score and noul, **When** built, **Then** layouts match the goldens.
2. **Given** state, an empty document, unicode or marker-like strings, **When** built, **Then** output matches the goldens exactly.

---

### User Story 3 - Report Accidental Behavior (Priority: P3)

Places where Laya's behavior looks accidental or underspecified are listed for the researcher, not fixed.

**Why this priority**: Parity is the goal; the observations feed later specs.

**Independent Test**: Review the PR description for the list.

**Acceptance Scenarios**:

1. **Given** the port is complete, **When** the PR is opened, **Then** it lists each suspected accident and confirms Raya still matches Laya.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Port the layout `[CLS] <type> question: instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]`, including the 48-token option cap, the head budget and per-option cap, `max_len` cut, and marker filtering, as in `build_sequence` (`laya/common.py`).
- **FR-002**: Match state serialization, question-type rendering and option rendering from `laya/common.py` and `laya/agent.py`, including mask-token replacement in user text and left or right truncation of state.
- **FR-003**: Produce padding and attention inputs as Laya's collation does.
- **FR-004**: Tests compare ids and marker positions against `tests/data/tokenizer/` exactly.
- **FR-005**: MUST NOT "fix" Laya behavior.
- **FR-006**: Support reusing one tokenized state across several questions, as Laya does.

## Success Criteria *(mandatory)*

- **SC-001**: Every golden record matches exactly.
- **SC-002**: Tests run offline.

## Assumptions

- The vendored goldens record the laya-sparse commit they were generated from.
