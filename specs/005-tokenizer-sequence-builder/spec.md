# Feature Specification: Tokenizer and Sequence Builder

**Feature Branch**: `005-sequence-builder`

**Created**: 2026-09-30 (draft); updated 2026-10-03 against `main` at `01de6c8`

**Status**: Draft

**Origin**: Prompt P4 of the former `docs/agent-prompts.md`.

**Depends on**: [001](../001-workspace-config/spec.md) (merged: `raya-core` config and `Error`), [002](../002-golden-fixtures/spec.md) (merged: forward goldens whose `input` blocks and `input_ids`/`marker_pos` tensors double as collation checks), [004](../004-tokenizer-goldens/spec.md) (merged: `tests/data/tokenizer/` from laya-sparse `cefb300`, implemented as `laya:005`). **Blocks**: [006](../006-reference-forward/spec.md).

**Input**: User description: "A Rust port of Laya's tokenizer use and sequence builder in `raya-tokenizer` that matches Laya's marker layout, state handling, question types, truncation and padding exactly, verified against the vendored goldens."

## Context: what exists on `main`

- `raya-tokenizer` is a stub: `pub use tokenizers::Tokenizer;`. The `tokenizers` crate (0.22, `onig`, no default features) is already a dependency; `raya-core` provides `Error { Config, Io, Json, Unimplemented, Weights }`.
- `tests/data/tokenizer/` holds `tokenizer.json` (WordPiece, BERT lowercase normalizer, vocab 231; added special tokens `[PAD]`=0, `[UNK]`=1, `[CLS]`=2, `[SEP]`=3, `[MASK]`=4), `meta.json` (laya-sparse `cefb300`, `option_token_cap` 48) and 11 records: `empty_document`, `exact_max_len`, `long_document`, `marker_like_strings`, `max_len_plus_one`, `options_over_head_budget`, `primary_one_question`, `several_questions`, `state_conversation_list`, `state_dict`, `unicode`.
- A record holds the caller-facing input (`questions` in the public form `type`/`instructions`/`criteria`/`labels`, `state`, `max_len`, `head_max_len`, `truncate_left`), the normalized form Laya builds from it (`internal_questions`: `t`/`ins`/`crit`/`labels`), `state_serialized`, and one item per question with `input_ids`, `markers`, `position_ids` (derived `range(len)`), `length`, `head_length`, `qtype`, `options_stats` (`options`, `options_distinct`, `tokens_per_option`), `state_tokens_total/kept`, `state_truncated`, `sequence_clipped`.
- The goldens are written with `sort_keys=True`. Key order of a dict `state` and of a dict `criteria` is therefore **not** preserved in the record's `state`/`questions`, although it changes the serialized text and the option order. `state_serialized` keeps the real order (e.g. `{"task": ..., "customer": ..., ...}` for `state_dict`).
- Forward goldens (`tests/data/{short,medium,padded,fastpath_off}/meta.json`) carry an `input` block (question, state, `max_len`, `head_max_len`) and the resulting `input_ids`, `attention_mask`, `marker_pos`, `marker_mask`, `qtype`, `position_ids` tensors; `padded` is padded to 160 from an unpadded length of 82.
- Laya's behavior to port is `build_sequence`, `serialize_state`, `render_criterion`, `render_options`, `_resolve_noul_labels` and `collate_items` in `laya/common.py`, and `Agent._to_internal`, `Agent._encode_state` (and the question validation next to them) in `laya/agent.py`, all in the read-only submodule at `cefb300`. `reference/laya-sparse/specs/005-tokenizer-goldens/findings.md` lists builder behavior that looks accidental.

## Clarifications

### Session 2026-10-03

Resolved by the implementing agent with defaults (no user questions asked); items needing the human are listed below.

- Q: Which Laya entry points are in scope? → A: The sequence-building path as `Agent._encode_state` drives it: normalize a public question definition (`_to_internal`), render options, build one sequence per question over a shared, once-tokenized state, check the `len(markers) == len(options)` guard, and collate items into a padded batch. The forward pass, probabilities, answer decoding and `predict()` stay in 006.
- Q: How much of Laya's question validation is ported? → A: Only what the builder cannot render without: question type must be `choice`, `score` or `noul`; criteria shape per type (choice: non-empty dict or list; score: non-empty list; noul: absent or dict keyed `true`/`false`); `labels` only on `noul` with distinct non-empty `false`/`true` strings; `instructions` non-string is JSON-serialized as in `_to_internal`. Violations return `Error::Config` naming the question. Laya's remaining checks (duplicate/unhashable/null choice labels, per-request limits, HTTP-facing messages) belong with `predict()` in 006 and are not ported here.
- Q: What is the input type? → A: The public question form as `serde_json::Value`/typed equivalent (`type`, `instructions`, `criteria`, `labels`), plus the state as string, JSON object or JSON array. Dict key order is preserved on input (an order-preserving JSON map), because Laya relies on Python dict order for both option order and state text.
- Q: How is Python's JSON text reproduced? → A: State and structured criteria are serialized as Python's `json.dumps(..., ensure_ascii=False)` does: separators `", "` and `": "` (the criterion path passes them explicitly, the state path gets them as Python's defaults), keys in insertion order, non-ASCII kept as is. Numbers and escapes outside what the goldens exercise are covered by unit tests with expected strings taken from Python semantics, not from the goldens.
- Q: How are non-string choice labels rendered? → A: `str(label)` as in Python for strings, integers and booleans (`True`/`False`). Floats and other types return `Error::Unimplemented` naming the question, because matching Python's float `repr` is not required by any golden. See open question 3.
- Q: How are `max_len`, `head_max_len` and the 48-token option cap applied? → A: Exactly as `build_sequence`: option text is tokenized without special tokens and cut to its first 48 tokens (Rust: encode, then truncate; same result as the tokenizer-side `truncation=True, max_length=48`), each option gets a leading `[MASK]`; if the head budget `head_max_len - sum(option lengths) < 16`, each option is cut to `max(4, (head_max_len - 16) // n_options)`; the question head is cut to `max(8, remaining budget)`; `room = max(0, max_len - len(ids) - 1)`; state ids are `state[:room]`, or `state[len-room:]` when `truncate_left` (list states); the result is cut to `max_len` and markers `>= max_len` are dropped. The `[-0:]` pitfall Laya already guards is reproduced by the same expression, not "fixed".
- Q: Position ids? → A: The builder returns none; Raya exposes `position_ids = 0..len` as a derived helper, matching the golden field and what ModernBERT derives. Padded positions are not specified by Laya's collation (it returns no position ids); the forward pass (006) decides.
- Q: Is the tokenizer file loaded by path only? → A: Yes: `Tokenizer::from_file` on a `tokenizer.json`, special-token ids read from the tokenizer (`[CLS]`, `[SEP]`, `[MASK]`, `[PAD]`) and checked against `meta.json` in tests. A checkpoint-directory convenience mirrors `load_dir` in 003 (`tokenizer.json` beside `model.safetensors`; the exact directory layout is confirmed in planning).
- Q: Thread safety? → A: Laya serializes encoding with a lock because Python's fast tokenizer mutates on truncation/padding. The Rust builder never enables tokenizer-side truncation or padding, so a `&Tokenizer` is shared read-only and no lock is needed.

- Q: Question-type ids? → A: `choice`=0, `score`=1, `noul`=2 (`QTYPES` in `laya/common.py`), matching the golden `qtype`.
- Q: What do `options_distinct` and the other stats measure? → A: As in `build_sequence(return_stats=True)`: counted on the capped option id lists before assembly (distinct token spans), `tokens_per_option` is the per-option cap when the head budget forced one, else none. The derived truncation fields (`state_tokens_total/kept`, `state_truncated`, `sequence_clipped`, `head_length`) are computed by the Rust builder from the same quantities the golden generator used and compared with the record.
- Q: Where does the lost-marker guard live? → A: In the batch-facing entry point that mirrors `Agent._encode_state`; the single-sequence builder returns the (possibly short) marker list like `build_sequence`, so both behaviors stay testable.
- Q: Several states in one call? → A: Each state is encoded once against all questions; collation accepts a list of per-state item groups and flattens them in order, as `collate_items` does.
- Q: `target`, `label` and `meta` collation fields? → A: Out of scope (training and bookkeeping); `qtype` is the only per-row field carried.

### Open questions for the human (defaults above stand until answered)

1. **Key order in the goldens.** `sort_keys=True` loses the order of dict `state` and dict `criteria`. Default: tests parse `state_serialized` with an order-preserving map and require our serializer to reproduce it exactly (a round-trip), and criteria order is taken from the golden's option order. Better: ask laya-sparse (`laya:005`) to also record `state` and `questions` as ordered pair lists, then re-vendor. Which?
2. **Validation depth.** The default ports only the checks needed to render a question (above). Should the full `_validate_questions` rule set move into Raya now (larger surface, mostly error text), or wait for 006/`predict()`?
3. **Exotic choice labels.** Floats, large ints and nested JSON as list labels (`[1.0, 2]`) need Python `str()`/`repr` parity. Default: `Unimplemented` for floats; strings, ints and bools only. Acceptable?

## User Scenarios & Testing *(mandatory)*

The user is the Raya implementer, who needs model inputs identical to Laya's for the same request, so that the reference forward (006) can be compared with Laya's goldens end to end.

### User Story 1 - Primary Workload Parity (Priority: P1)

One document plus one question produces the same `input_ids`, derived position ids and marker positions as Laya, including at the `max_len` boundary.

**Why this priority**: This is the workload every measurement and decision uses.

**Independent Test**: `cargo test -p raya-tokenizer --test goldens` on `primary_one_question`, `empty_document`, `long_document`, `exact_max_len`, `max_len_plus_one`: ids, markers, `length`, `head_length`, `state_tokens_*`, `state_truncated`, `sequence_clipped` equal the record exactly. Tokenization of a fixed text matches the golden ids.

**Acceptance Scenarios**:

1. **Given** `tokenizer.json` loaded through the `tokenizers` crate, **When** a fixed text is encoded without special tokens, **Then** the ids equal the golden ids.
2. **Given** a document and one question, **When** the sequence is built, **Then** ids, positions and marker positions equal the golden record exactly.
3. **Given** a document longer than `max_len`, exactly `max_len`, and `max_len + 1`, **When** built, **Then** the output and the truncation flags match each record.
4. **Given** an empty document, **When** built, **Then** the sequence is the head plus `[SEP]` (length 67 in the record) and matches.

---

### User Story 2 - Multiple Questions, State, Edge Inputs (Priority: P2)

Several questions over one document, state handling (string, dict, conversation list), unicode, text containing marker-like strings, and option-budget pressure all match Laya.

**Why this priority**: Needed for full `predict()` parity, but not for the first benchmark.

**Independent Test**: Golden comparison for `several_questions`, `state_dict`, `state_conversation_list`, `unicode`, `marker_like_strings`, `options_over_head_budget`.

**Acceptance Scenarios**:

1. **Given** questions of types choice, score and noul (including `labels` and omitted criteria), **When** built, **Then** every item's layout and `qtype` match the record; the state is tokenized once and reused.
2. **Given** a dict state, **When** serialized, **Then** the text equals `state_serialized`; **Given** a list state, **Then** `truncate_left` applies and the newest turns are kept.
3. **Given** `[MASK]` in the document or question text, **Then** it is replaced by a space (as Laya does); **Given** `[CLS]`, `[SEP]`, `[PAD]`, `[UNK]` in user text, **Then** they tokenize to the real special ids, matching `marker_like_strings`.
4. **Given** many options and a small `head_max_len`, **Then** the per-option cap applies and `options_stats` (`options`, `options_distinct`, `tokens_per_option`) equals the record; **Given** markers are lost to the cut, **Then** the builder returns an error naming the question and `head_max_len` (Laya's guard).

---

### User Story 3 - Collation (Priority: P2)

Items from one or several states are padded into a batch the way `collate_items` does: `input_ids` padded with the pad id, `attention_mask` 1 over real tokens, `marker_pos` and `marker_mask` padded to the widest item, `qtype` per row.

**Why this priority**: 006 consumes batches; padding behavior must be fixed before the forward pass relies on it.

**Independent Test**: Unit tests derived from `collate_items`, plus agreement with the `short`, `medium` and `padded` forward goldens: building their `input` block yields their unpadded `input_ids` and `marker_pos`, and collating with `pad_to` yields their padded `input_ids`, `attention_mask` and `marker_mask`.

**Acceptance Scenarios**:

1. **Given** items of different lengths and marker counts, **When** collated, **Then** shapes are `(n, max_len_in_batch)` and `(n, max_markers)` with Laya's padding values.
2. **Given** the `padded` forward golden input, **When** built and padded to 160, **Then** `input_ids`, `attention_mask`, `marker_pos`, `marker_mask`, `qtype` equal the golden tensors.
3. **Given** an empty item list, **Then** collation returns nothing (Laya returns `None`).

---

### User Story 4 - Report Accidental Behavior (Priority: P3)

Places where Laya's behavior looks accidental or underspecified are listed for the researcher, not fixed.

**Why this priority**: Parity is the goal; the observations feed later specs.

**Independent Test**: Review the PR description for the list.

**Acceptance Scenarios**:

1. **Given** the port is complete, **When** the PR is opened, **Then** it lists each suspected accident (starting from laya:005 `findings.md`: only `[MASK]` is sanitized, a `[PAD]` id in text is not masked, truncation is silent, `unicode` is mostly `[UNK]` on the fixture vocabulary) and confirms Raya still matches Laya.

### Edge Cases

- Empty state, empty instructions, empty option text (a label with no description renders as the label alone).
- A state or option whose token count is exactly the available room, or room is 0 (`truncate_left` with no room keeps none of the state).
- `max_len` smaller than the head; `head_max_len` too small for even the minimum per-option cap (markers lost, guard error).
- Options that collapse to identical token spans under the cap (`options_distinct < options`) are reported in the stats, not rejected.
- A `[MASK]` string inside a criterion value, a label or the instructions.
- Non-ASCII text in dict state or structured criteria stays unescaped in the serialized state.
- Tokenizer file missing, unreadable or lacking `[CLS]`/`[SEP]`/`[MASK]`/`[PAD]`: explicit error naming the file.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Port the layout `[CLS] <type> question: instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]`, including the 48-token option cap, the head budget and per-option cap, the head cut, the `max_len` cut and marker filtering, as in `build_sequence` (`laya/common.py`).
- **FR-002**: Match state serialization (string passthrough; dict/list as `json.dumps(ensure_ascii=False)` text), question normalization (`_to_internal`: list choice criteria to ordered labels, lowercase noul keys, non-string instructions to JSON text), option rendering (`render_options`, `render_criterion`, noul default labels and descriptions) and mask-token replacement in instructions, options and state, including left truncation for list states.
- **FR-003**: Produce padding and attention inputs as `collate_items` does (pad id from the tokenizer, `attention_mask`, `marker_pos`, `marker_mask`, `qtype`).
- **FR-004**: Tests compare ids, markers, lengths, stats and truncation flags against `tests/data/tokenizer/` exactly (all 11 records), and collation against the forward goldens; all offline.
- **FR-005**: MUST NOT "fix" Laya behavior. Accidental-looking behavior is reproduced and reported (User Story 4).
- **FR-006**: Support reusing one tokenized state across several questions, as `Agent._encode_state` does (state tokenized once, sliced per question).
- **FR-007**: Errors use the shared `raya_core::Error`: bad question definitions return `Error::Config` naming the question, unsupported label types `Error::Unimplemented`, tokenizer file problems an error naming the path, and lost option markers an error naming the question and `head_max_len`. Nothing is silently truncated or skipped beyond what Laya itself does.
- **FR-008**: No `unsafe` in `raya-tokenizer`; the crate stays `#![forbid(unsafe_code)]`. The builder never enables tokenizer-side truncation or padding.
- **FR-009**: Special-token ids are read from the tokenizer and checked against the golden `meta.json` in tests; nothing hard-codes ids in library code.

### Key Entities

- **Question (public form)**: `type` (`choice`/`score`/`noul`), `instructions`, optional `criteria` (dict, list or absent by type), optional `labels` (noul only).
- **Normalized question**: Laya's internal `t`, `ins`, `crit`, `labels`.
- **State**: string, JSON object or JSON array; its serialized text and token ids are computed once per request.
- **Item**: `ids`, `markers`, `qtype`, option stats, plus the derived truncation facts (`state_tokens_total/kept`, `state_truncated`, `sequence_clipped`, `head_length`) the goldens record.
- **Batch**: padded `input_ids`, `attention_mask`, `marker_pos`, `marker_mask`, `qtype`.

## Success Criteria *(mandatory)*

- **SC-001**: Every one of the 11 golden records matches exactly (ids, markers, lengths, stats, truncation flags).
- **SC-002**: Tests run offline, with no network and no submodule checkout (vendored fixtures only).
- **SC-003**: The `short`, `medium` and `padded` forward goldens' inputs rebuild to their recorded `input_ids`, `marker_pos`, `marker_mask`, `attention_mask` and `qtype`.
- **SC-004**: Vendored fixtures and `reference/` are byte-unchanged (`git diff --stat main -- tests/data reference` empty).
- **SC-005**: `cargo build`, `fmt --check`, `clippy -D warnings` and `cargo test --workspace` pass on the branch.

## Assumptions

- The vendored goldens record the laya-sparse commit they were generated from (`cefb300`); regenerate, never hand-edit.
- The `tokenizers` crate 0.22 encodes the fixture tokenizer like the Python `tokenizers` 0.23.2 that wrote the goldens; any divergence found by the goldens is reported as a finding, not worked around silently.
- fp32 and CPU only apply to the model; this feature is integer sequence handling with no numerics beyond JSON number text.
- Shared tokenization across threads is read-only; no global lock.
- Not in scope: the real (non-fixture) checkpoint tokenizer, BPE/other tokenizer types beyond what `tokenizer.json` describes, the forward pass, answer decoding, HTTP-facing validation messages.
