# Tasks: Tokenizer and Sequence Builder

**Input**: Design documents from `/specs/005-tokenizer-sequence-builder/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Required (constitution II; spec FR-004, SC-001 to SC-004).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: can run in parallel (different files, no dependency)
- **[Story]**: US1 (primary workload parity), US2 (multi-question, state, edge inputs), US3 (collation), US4 (report accidental behavior)

## Phase 1: Setup

- [ ] T001 `crates/raya-tokenizer/Cargo.toml`: add `serde` and `serde_json` (workspace) as dependencies and `safetensors` (workspace) as a dev-dependency; no new crate may appear in `Cargo.lock` (check with `git diff Cargo.lock`)
- [ ] T002 [P] `crates/raya-tokenizer/src/lib.rs`: declare `mod json; mod question; mod builder; mod collate;`, public re-exports per `contracts/sequence-api.md`, keep `#![forbid(unsafe_code)]` and `pub use tokenizers::Tokenizer`

## Phase 2: Foundational (blocks all stories)

- [ ] T003 `crates/raya-tokenizer/src/json.rs`: `JsonValue` (order-preserving `Deserialize`, `parse`, duplicate keys keep last value at first position) and `dumps()` reproducing Python `json.dumps(ensure_ascii=False)` per `research.md` R3: separators `", "` and `": "`, the listed escapes, lowercase `\u00xx`, Python float `repr` (positional with `.0`, scientific `1e-07` / `1e+16` when exponent < -4 or >= 16), integers in decimal
- [ ] T004 [P] `crates/raya-tokenizer/tests/json_text.rs`: expected strings from Python semantics: key order kept, nested arrays/objects, escapes (`"`, `\`, newline, tab, U+0001, DEL, non-ASCII literal), floats (`7.0`, `0.5`, `1e-07`, `1e+16`, `123456789012345.0`, `-0.0`), big ints, duplicate keys; and `parse(s).dumps() == s` for each `state_serialized` in the 11 records
- [ ] T005 `crates/raya-tokenizer/src/question.rs`: `QType` (choice 0, score 1, noul 2), `QuestionDef`, `Crit`, `Question::from_def` implementing `_to_internal` plus the validation in the spec Clarifications (errors `Error::Config` with field `questions.<id>.<key>`; list label that is not string, integer or bool gives `Error::Unimplemented`; exact duplicate list labels give `Error::Config`; non-string `instructions` become `dumps()` text) and `Question::options()` implementing `render_options`, `render_criterion` and `_resolve_noul_labels` (labels `.strip()`ped, defaults, `str(label)` with `True`/`False`)
- [ ] T006 [P] `crates/raya-tokenizer/tests/render_options.rs`: option text for choice (dict with and without descriptions, list form, int/bool labels, structured description as compact JSON), score (`level i: ...`), noul (defaults, custom `labels`, partial `criteria`, empty description falls back to default); error cases for `labels` on non-noul, bad noul keys, empty criteria, equal noul labels
- [ ] T007 [P] Test helpers in `crates/raya-tokenizer/tests/common/mod.rs`: paths under `tests/data/tokenizer/`, `meta.json` special ids, record loader (UTF-8 read), builder from `tokenizer.json`, helper turning a record's `state`/`state_serialized` into a `State` (parse `state_serialized` for dict and array states, plain text otherwise) and its `questions` into `(id, QuestionDef)` pairs. Never write under `tests/data/`

**Checkpoint**: JSON text, question rendering and test helpers exist; no builder logic yet.

## Phase 3: User Story 1 - Primary Workload Parity (Priority: P1) MVP

**Goal**: one document plus one question matches Laya exactly, including at the `max_len` boundary.

**Independent Test**: `cargo test -p raya-tokenizer --test goldens -- primary empty long exact max_len_plus_one`.

- [ ] T008 [US1] `crates/raya-tokenizer/src/builder.rs`: `SequenceBuilder::from_file` / `from_dir` (read `[CLS]`, `[SEP]`, `[MASK]`, `[PAD]` ids with `token_to_id`; missing file or token gives `Error::Config` naming the path or token; never enable tokenizer truncation or padding), `State`, `state_ids` (serialize, replace `[MASK]` with a space, encode without special tokens), `Item`, `Stats`, `Trace`, `Item::position_ids`
- [ ] T009 [US1] `builder.rs`: `build_sequence` transcribed from `research.md` R6 (48-token option cap, head budget `< 16` per-option cap with signed floor division, head cut `max(8, budget)`, `room`, left or right state cut, `max_len` cut, marker filter, stats on capped option lists, derived `Trace` fields)
- [ ] T010 [US1] `crates/raya-tokenizer/tests/goldens.rs`: for every record and every item compare `input_ids`, `markers`, `length`, `head_length`, `qtype`, `options_stats`, `state_budget`, `state_tokens_total`, `state_tokens_kept`, `state_truncated`, `sequence_clipped`, `position_ids`; failure messages name record, question id and first differing index; also assert special-token ids equal `meta.json` `special_tokens` and that fixed text encodes to the expected ids. Start with the US1 records (`primary_one_question`, `empty_document`, `long_document`, `exact_max_len`, `max_len_plus_one`)

**Checkpoint**: US1 green (MVP).

## Phase 4: User Story 2 - Multiple Questions, State, Edge Inputs (Priority: P2)

**Goal**: several questions, dict and list states, unicode, marker-like strings and option-budget pressure match Laya.

**Independent Test**: `cargo test -p raya-tokenizer --test goldens` (all 11 records) and `--test errors`.

- [ ] T011 [US2] `builder.rs`: `encode_state` mirroring `Agent._encode_state` (state tokenized once, `truncate_left` for array states, one item per question in order, lost-marker guard returning `Error::Config { field: "head_max_len", .. }` naming the question)
- [ ] T012 [US2] `goldens.rs`: extend to all 11 records (`several_questions`, `state_dict`, `state_conversation_list`, `unicode`, `marker_like_strings`, `options_over_head_budget`); `record` is built through `encode_state`, and for the single-question primary record also through `build_sequence` directly
- [ ] T013 [US2] `crates/raya-tokenizer/tests/errors.rs`: missing tokenizer file names the path; tokenizer lacking a special token names it; unknown question type; lost markers (tiny `head_max_len`) names question and limit; `max_len` smaller than the head does not panic; room 0 with `truncate_left` keeps none of the state; empty state and empty instructions build

**Checkpoint**: US1 and US2 green.

## Phase 5: User Story 3 - Collation (Priority: P2)

**Goal**: `collate_items` semantics and agreement with the forward goldens.

**Independent Test**: `cargo test -p raya-tokenizer --test collate`.

- [ ] T014 [US3] `crates/raya-tokenizer/src/collate.rs`: `Batch` and `collate(groups, pad_id, min_len) -> Option<Batch>` per `research.md` R7 (flatten groups in order; `min_len` raises the padded length for the forward-golden `pad_to` case only)
- [ ] T015 [US3] `crates/raya-tokenizer/tests/collate.rs`: unit tests (mixed lengths and marker counts, zero markers, pad values, empty input gives `None`); forward goldens `short`, `medium`, `padded`: build the `input` block from `meta.json` (question with `type`/`instructions`/`criteria`, state, `max_len`, `head_max_len`), compare with `input_ids`, `attention_mask`, `marker_pos`, `marker_mask`, `qtype` read from `tensors.safetensors` (int64/bool), `padded` with `min_len = pad_to`; keep the test-side construction of the question from `meta.json` explicit about the sorted-key caveat (R5)

**Checkpoint**: US1 to US3 green.

## Phase 6: User Story 4 - Report Accidental Behavior (Priority: P3)

- [ ] T016 [US4] Collect findings for the PR description: only `[MASK]` sanitized, `[PAD]`/`[CLS]`/`[SEP]` in user text become real ids with an all-ones attention mask, silent truncation, `unicode` record mostly `[UNK]`, the `sort_keys` order loss in the goldens, `option_order` never used by `_encode_state`, plus anything the goldens revealed during implementation (including any `tokenizers` 0.22 versus 0.23 divergence). No code change

## Phase 7: Polish & Cross-Cutting

- [ ] T017 Run `cargo build --workspace`, `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace` (VS env, `CC=cl`); confirm `git diff --stat main -- tests/data reference` is empty (SC-004) and `Cargo.lock` has no new crate; record output for the PR
- [ ] T018 Update `specs/005-tokenizer-sequence-builder/spec.md` Status to Implemented only after T017 passes; tick the quickstart steps; update `specs/README.md` only if it tracks status (it does not)

## Dependencies & Execution Order

- Phase 1, then Phase 2. T004, T006, T007 are parallel after T003/T005 (T004 needs T003, T006 needs T005, T007 needs T003 and T005).
- US1 needs Phase 2; T010 can be written before T009 (fails until then).
- US2 extends `builder.rs` and `goldens.rs` from US1 (T011 after T009).
- US3 needs `Item` from T008 only; it can run in parallel with US2 after T009 since it touches `collate.rs` and `tests/collate.rs`.
- Polish last.

## Implementation Strategy

1. MVP = Phases 1 to 3 (primary workload parity).
2. Add multi-question, state and the guard (US2) before 006 consumes the crate.
3. Collation (US3) last of the code; it is a thin consumer of `Item`.

## Notes

- Risk: `tokenizers` 0.22.2 may differ from the Python 0.23.2 that wrote the goldens; the goldens are the check, any difference is a finding.
- Risk: goldens are key-sorted, so dict order is verified only through `state_serialized` (R5).
- Risk: Python float `repr` and `str(label)` parity beyond the goldens rests on unit tests with expected strings derived from Python semantics.
- Human-owned, not decided: golden regeneration with ordered keys, depth of Laya's question validation, exotic label types (spec open questions 1 to 3).
