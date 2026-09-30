# Feature Specification: Tokenizer Goldens from Laya

**Feature Branch**: `raya/p4b-tokenizer-goldens` **in laya-sparse** (specification directory only; no speckit branch created)

**Created**: 2026-09-30

**Status**: Draft

**Origin**: Prompt P4b of the former `docs/agent-prompts.md`.

**Runs in**: `laya-sparse` (Python). Output is vendored into Raya's `tests/data/tokenizer/`.

**Depends on**: 002 (`experiments/golden/` module). **Blocks**: 005.

**Input**: User description: "Extend the golden module with a tokenizer command that calls Laya's real sequence builder on a fixed list of inputs and saves ids, positions, marker positions and truncation flags."

## User Scenarios & Testing *(mandatory)*

The user is the Raya implementer porting the sequence builder, who needs Laya's exact outputs, edge cases included.

### User Story 1 - Sequence-Builder Goldens (Priority: P1)

Running `python -m experiments golden tokenizer --out artifacts/raya-golden/tokenizer/` writes one JSON record per fixed input: document text, questions (types choice/score/noul), state, max_len, resulting input_ids, position ids, marker positions and truncation flags.

**Why this priority**: The Rust port is verified against these records.

**Independent Test**: Run offline with the tiny fixture tokenizer; check records exist for every listed input; re-run and compare bytes.

**Acceptance Scenarios**:

1. **Given** the fixed input list, **When** the command runs, **Then** records exist for: empty document; one document plus one question (primary workload); one document plus several questions; with state; unicode; text containing marker-like strings; a document far longer than max_len; a document exactly max_len.
2. **Given** a repeated run, **When** outputs are compared, **Then** they are byte-identical.
3. **Given** the run, **When** it finishes, **Then** the `tokenizer.json` used and a `meta.json` with the laya-sparse commit are saved beside the records.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Call Laya's real builder (`laya/common.py` `build_sequence`, with `laya/agent.py` for state handling and multi-question flow); MUST NOT reimplement it.
- **FR-002**: Use the tiny fixture model's tokenizer if it has one; otherwise the real `tokenizer.json`, only when a local checkpoint is given by flag and never in offline tests.
- **FR-003**: Do not change any Laya behavior. Behavior that looks accidental is listed in the PR description.
- **FR-004**: Offline pytest asserts byte-identical re-runs.
- **FR-005**: Records also capture the builder's stats (options, distinct options, tokens per option) and `truncate_left` and `head_max_len` settings where exercised.

## Success Criteria *(mandatory)*

- **SC-001**: All listed inputs are covered and the test passes offline.
- **SC-002**: No file under `laya/` is modified.

## Assumptions

- The former prompt named `laya/agent.py` and `laya/common.py` as the sources; the builder itself lives in `common.py`.
