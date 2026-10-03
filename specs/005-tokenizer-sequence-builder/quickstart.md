# Quickstart: validating the sequence builder

Offline; needs no submodule checkout (vendored fixtures only).

1. `cargo test -p raya-tokenizer --test goldens`: all 11 records of `tests/data/tokenizer/records/` (ids, markers, lengths, stats, truncation fields); failures name the record and question.
2. `cargo test -p raya-tokenizer --test collate`: `collate_items` semantics, and the `short`, `medium` and `padded` forward-golden inputs rebuilt from their `meta.json` `input` blocks, compared with `input_ids`, `marker_pos`, `marker_mask`, `attention_mask`, `qtype` in `tensors.safetensors`.
3. `cargo test -p raya-tokenizer --test json_text --test render_options --test errors`: Python `json.dumps` parity (escapes, floats, key order), option rendering per question type, explicit errors.
4. `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`; `git diff --stat main -- tests/data reference` must be empty.

Failure demo: change a character in a copied record's `state_serialized`; the golden test reports the record name and the first differing position.
