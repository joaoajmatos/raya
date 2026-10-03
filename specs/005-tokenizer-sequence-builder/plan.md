# Implementation Plan: Tokenizer and Sequence Builder

**Branch**: `005-sequence-builder` | **Date**: 2026-10-03 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/005-tokenizer-sequence-builder/spec.md`

## Summary

Port Laya's sequence-building path to `raya-tokenizer`: normalize a public question definition, render its options, tokenize state once, build `[CLS] head [SEP] ([MASK] option)* [SEP] state [SEP]` per question with Laya's budgets and cuts, enforce the lost-marker guard, and collate items into a padded batch. A small order-preserving JSON value type reproduces Python's `json.dumps(ensure_ascii=False)` text (state and structured criteria), because option order and state text both depend on dict insertion order. The `tokenizers` crate does all tokenization; the builder never enables its truncation or padding. Verification is exact comparison with the 11 vendored tokenizer records plus the `short`/`medium`/`padded` forward-golden inputs for collation. No model math, no validation beyond what rendering needs.

## Technical Context

**Language/Version**: Rust, toolchain pinned by `rust-toolchain.toml` (workspace `rust-version = 1.94`), edition 2021

**Primary Dependencies**: existing `tokenizers` 0.22 (`onig`), `raya-core` (`Error`), `serde` and `serde_json` (already workspace dependencies; `serde_json` is new for this crate). No new crate in `Cargo.lock`. Dev: `tempfile` only if a test needs it; `safetensors` (workspace) to read the int64 forward-golden tensors.

**Storage**: `tokenizer.json` read by path; tests read the vendored `tests/data/tokenizer/{tokenizer.json,meta.json,records/*.json}` and `tests/data/{short,medium,padded}/{meta.json,tensors.safetensors}`

**Testing**: `cargo test --workspace`; integration tests in `crates/raya-tokenizer/tests/`; offline

**Target Platform**: Linux x86_64, macOS arm64, Windows

**Project Type**: library crate (`raya-tokenizer`); no CLI change in this feature

**Performance Goals**: none claimed (constitution III)

**Constraints**: `#![forbid(unsafe_code)]`; no tokenizer-side truncation or padding; vendored fixtures and `reference/` never modified; golden JSON key order is sorted (see research R5)

**Scale/Scope**: fixture vocabulary 231; records up to 96 tokens; real Laya sequences up to a few thousand tokens, so state is tokenized once and sliced per question

## Constitution Check

| Principle | Status |
|---|---|
| I Laya is the specification | Pass. Port of `build_sequence`, `render_options`, `_to_internal`, `collate_items`; accidental behavior (only `[MASK]` sanitized, silent truncation, `[PAD]` ids unmasked) reproduced and reported, not fixed. |
| II Parity by goldens | Pass. All 11 records compared exactly, per-record failure messages; collation checked against forward-golden tensors. Gap: records are key-sorted, so dict order is verified through `state_serialized` only (R5, open question 1). |
| III Measure, do not assume | Pass. No performance claims. The assumption "tokenizers 0.22 equals Python tokenizers 0.23.2 on the fixture" is checked by the goldens, not asserted. |
| IV Raya-L, fp32 | Pass. Integer sequence handling only. |
| V Portable first | Pass. No SIMD, no `unsafe`. |
| VI Correctness before speed | Pass. Straight port, no optimization; the shared-state tokenization is Laya's own behavior. |
| VII Small surface, explicit errors | Pass. No new crate in the lock file; explicit errors for bad questions, unsupported label types, missing tokenizer or special tokens, lost markers. Validation limited to rendering needs. |

Re-check after design: unchanged.

## Project Structure

### Documentation (this feature)

```text
specs/005-tokenizer-sequence-builder/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/sequence-api.md
├── checklists/requirements.md
└── tasks.md           # /speckit-tasks
```

### Source Code

```text
crates/raya-tokenizer/
├── Cargo.toml                 # + serde, serde_json; dev: safetensors
├── src/
│   ├── lib.rs                 # re-exports (keeps `Tokenizer`)
│   ├── json.rs                # JsonValue (ordered), Python-compatible dumps
│   ├── question.rs            # QuestionDef, normalize (_to_internal), render_options
│   ├── builder.rs             # SequenceBuilder, build_sequence, encode_state, Item, Stats
│   └── collate.rs             # Batch, collate
└── tests/
    ├── common/mod.rs          # load records, meta, forward-golden inputs
    ├── json_text.rs           # dumps parity, float repr, escapes, key order
    ├── render_options.rs      # option text per type, labels, defaults
    ├── goldens.rs             # 11 records, exact
    ├── collate.rs             # collate_items semantics + short/medium/padded
    └── errors.rs              # bad questions, label types, missing tokenizer/specials, lost markers
```

**Structure Decision**: all logic in `raya-tokenizer`; `raya-core::Error` is reused without a new variant (`Error::Config` for bad questions and tokenizer problems, `Error::Unimplemented` for unsupported labels), so no public API change outside this crate.

## Complexity Tracking

| Item | Why | Simpler alternative rejected |
|---|---|---|
| Own ordered `JsonValue` instead of `serde_json::Value` | `serde_json::Value` sorts keys unless the `preserve_order` feature is on, which would add `indexmap` to the lock file and change map ordering for every workspace crate (feature unification). Laya's option order and state text depend on insertion order. | `preserve_order` feature (new transitive crates, workspace-wide effect) |
| Python float `repr` for dict-state numbers | `json.dumps` writes floats with `repr`; Rust `Display` differs from it outside roughly `1e-4..1e16`. About 30 lines using `{:e}` digits. | Reject floats (would break valid states) |
