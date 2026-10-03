# Data Model: Tokenizer and Sequence Builder

## JsonValue (`json.rs`)
`Null | Bool | Int(i64) | UInt(u64) | Float(f64) | Str(String) | Array(Vec) | Object(Vec<(String, JsonValue)>)`. `parse(&str)`, `Deserialize`, `dumps()` (Python `json.dumps(ensure_ascii=False)`, separators `", "` and `": "`; the criterion path uses the same text). Insertion order is kept; duplicate keys keep the last value at the first position.

## QuestionDef / Question (`question.rs`)
- `QuestionDef` (public form, `Deserialize`): `type: String`, `instructions: JsonValue`, `criteria: Option<JsonValue>`, `labels: Option<JsonValue>`.
- `Question` (normalized): `qtype: QType {Choice=0, Score=1, Noul=2}`, `ins: String`, `crit: Crit`, `labels: Option<(String, String)>`; `Crit` is `Labels(Vec<JsonValue>)` (list-form choice), `Described(Vec<(String, JsonValue)>)` (dict-form choice), `Levels(Vec<JsonValue>)` (score), `Noul { false_: Option<JsonValue>, true_: Option<JsonValue> }`.
- `Question::from_def(id, &QuestionDef) -> Result<Question>` (validation per the spec Clarifications) and `Question::options(&self) -> Result<Vec<String>>`.

## State (`builder.rs`)
`State::Text(String) | State::Json(JsonValue)`; `truncate_left()` is true only for `Json(Array)`; `serialized() -> String`.

## SequenceBuilder, Item, Stats, Trace (`builder.rs`)
- `SequenceBuilder { tok: Tokenizer, cls, sep, mask, pad: u32 }`; `from_file(path)`, `from_dir(dir)`.
- `Item { ids: Vec<u32>, markers: Vec<usize>, qtype: QType, stats: Stats, trace: Trace }`.
- `Stats { options, options_distinct, tokens_per_option: Option<usize> }`.
- `Trace` (derived, mirrors the golden fields): `head_length`, `state_budget`, `state_tokens_total`, `state_tokens_kept`, `state_truncated`, `sequence_clipped`.
- `Item::position_ids() -> Vec<u32>` = `0..len`.

## Batch (`collate.rs`)
`Batch { n, len, max_markers, input_ids: Vec<u32> (n*len), attention_mask: Vec<u8>, marker_pos: Vec<u32> (n*K), marker_mask: Vec<bool>, qtype: Vec<u8> }`, row-major. Built by `collate(groups: &[Vec<Item>], pad_id, min_len) -> Option<Batch>`.

## State transitions
None; every function is pure given the tokenizer.
