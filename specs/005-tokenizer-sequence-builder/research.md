# Research: Tokenizer and Sequence Builder

## R1. Tokenization through the `tokenizers` crate
- **Decision**: `Tokenizer::from_file`; `encode(text, false)` for every call; ids via `get_ids()`. Never call `with_truncation`/`with_padding`. The 48-token option cap is `ids.truncate(48)` after encoding.
- **Rationale**: Python's `truncation=True, max_length=48` keeps the first 48 tokens (right side), the same result as truncating the id list. Skipping tokenizer-side truncation also keeps `&Tokenizer` read-only, so Laya's `_TOKENIZE_LOCK` has no counterpart. The fixture `tokenizer.json` has `truncation: null, padding: null`.
- **Edge**: `[CLS]`, `[SEP]`, `[PAD]`, `[UNK]` typed in text match as added tokens in both implementations (golden `marker_like_strings`). The goldens are the test that crate 0.22.2 and Python 0.23.2 agree; a divergence is a finding, not worked around.

## R2. Special tokens
- **Decision**: read ids with `token_to_id` for `[CLS]`, `[SEP]`, `[MASK]`, `[PAD]` at load time; a missing token gives `Error::Config` naming the file and token. Token *strings* are constants (Laya takes them from `tok.mask_token`, which the fixture sets to `[MASK]`; `meta.json` `mask_token`); ids are never hard-coded (FR-009). Tests compare the ids with `meta.json` `special_tokens`.

## R3. Ordered JSON and Python `json.dumps`
- **Decision**: own `JsonValue` (Null, Bool, Int(i64), UInt(u64), Float(f64), Str, Array, Object as `Vec<(String, JsonValue)>`) with an order-keeping `Deserialize`, and `dumps()` writing Python's default separators `", "` and `": "`.
- **Escapes (ensure_ascii=False)**: `"` `\\` `\n` `\r` `\t` `\b` `\f` as short escapes, other code points below 0x20 as `\u00xx` (lowercase hex), everything else (including DEL and non-ASCII) literal.
- **Floats**: Python `repr`: shortest round-trip digits; scientific form `d.ddde±XX` (exponent at least two digits, e.g. `1e-07`, `1e+16`) when the decimal exponent is below -4 or at least 16, otherwise positional with a trailing `.0` for integral values. Implemented from Rust's `{:e}` digits. Integers: decimal. `NaN`/`Infinity` cannot occur in parsed JSON.
- **Duplicate object keys** in input: Python's `json.loads` keeps the last value at the first position; `JsonValue` does the same.
- **Alternatives**: `serde_json` `preserve_order` (adds `indexmap`, changes ordering workspace-wide through feature unification); passing pre-serialized text from the caller (pushes Laya-specific formatting onto every caller and leaves it untested).

## R4. Question normalization and option rendering
- `_to_internal`: choice with list criteria becomes an ordered label list with no descriptions; noul dict keys are lowercased; non-string `instructions` become `json.dumps(ensure_ascii=False)` text.
- `render_options` per type: choice `label` or `label: <desc>` (`str(label)`; description `None`/`""` means none; a non-string description goes through `render_criterion`: compact JSON with `", "`/`": "` separators); score `level i: <desc>`; noul `false_label: ...`, `true_label: ...` with default descriptions "no, the statement does not hold" / "yes, the statement holds" when absent or empty, labels default `false`/`true`, `.strip()`ped.
- **Label types**: string, integer and bool for list-form choice labels (`str()` gives `True`/`False`); floats and others give `Error::Unimplemented`. Dict-form labels are JSON keys, i.e. strings. List-form duplicates give `Error::Config` under Python equality for the supported types (equal strings; `true`/`1` and `false`/`0` and equal integers collide); floats are rejected earlier, so `1.0` never reaches the comparison.
- **Validation**: type, criteria shape, `labels` only on noul with the two distinct non-empty strings (spec Clarifications).

## R5. Verification gap: golden key order
- Records are written with `sort_keys=True`. Dict and list `state` text is recovered from `state_serialized`: the tests parse it into an ordered value and assert `parse(s).dumps() == s`, then build with that value. Criteria order is not recoverable, but the generator (`experiments/golden/tokenizer.py`) uses `opt0..opt11` (stored sorted as `opt0, opt1, opt10, ...`), and that record's options collapse to identical token spans, so order cannot change its ids; the other records are alphabetical already. Net: criteria order is covered by unit tests with hand-derived expectations, not by goldens.
- **Open**: ask laya-sparse for ordered pairs (spec open question 1).

## R6. Builder arithmetic (transcribed from `build_sequence`)
```text
head  = encode("{t} question: {ins with [MASK]->' '}")
opt_i = [MASK] + encode(" " + option_i with [MASK]->' ')[:48]
budget = head_max_len - sum(len(opt_i))
if budget < 16: per = max(4, (head_max_len - 16) // n_opts)   # floor division, n >= 1
                opt_i = opt_i[:per]; budget = head_max_len - sum(len(opt_i))
head = head[: max(8, budget)]
ids = [CLS] + head + [SEP]; markers = start index of each opt_i; ids += opt_i...; ids += [SEP]
room = max(0, max_len - len(ids) - 1)
state = last `room` ids if truncate_left else first `room`   # room 0 gives none
ids += state + [SEP]; ids = ids[:max_len]; markers = [m for m in markers if m < max_len]
```
- `option_order` is a permutation hook Laya exposes but `Agent._encode_state` never passes; not ported.
- Arithmetic is signed: `head_max_len - sum` and `(head_max_len - 16) // n` can be negative in Python, so Rust uses `isize` with floor division.
- Stats (`options`, `options_distinct` over the capped lists before assembly, `tokens_per_option`) and the derived fields (`head_length`, `state_budget = room`, `state_tokens_total/kept`, `state_truncated`, `sequence_clipped`) follow the golden generator (`experiments/golden/tokenizer.py`) and are checked on every record.

## R7. Guard and collation
- Guard (`Agent._encode_state`): `len(markers) != n_options` gives `Error::Config { field: "head_max_len", reason: "question 'q' options exceed head_max_len=N" }`. The raw builder returns the short marker list, as `build_sequence` does.
- `collate`: `n` = items, `L` = longest item, `K` = most markers (0 if none); `ids` padded with the pad id, `attention_mask` 1 over real tokens, `marker_pos` zero-padded, `marker_mask` true over real markers, `qtype` per row; no items gives `None`. `pad_to` (forward goldens) is a test-side extension of `L`, not part of `collate_items`, so `collate` takes an optional minimum length, documented as the one addition.
