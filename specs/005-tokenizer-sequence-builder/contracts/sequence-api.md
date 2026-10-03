# Contract: `raya-tokenizer` public API

```text
SequenceBuilder::from_file(tokenizer_json: &Path) -> Result<SequenceBuilder>
SequenceBuilder::from_dir(dir: &Path) -> Result<SequenceBuilder>        // dir/tokenizer.json
SequenceBuilder::state_ids(&self, state: &State) -> Result<Vec<u32>>
SequenceBuilder::build_sequence(&self, state_ids: &[u32], truncate_left: bool, q: &Question,
                                max_len: usize, head_max_len: usize) -> Result<Item>
SequenceBuilder::encode_state(&self, state: &State, questions: &[(String, QuestionDef)],
                              max_len: usize, head_max_len: usize) -> Result<Vec<Item>>
collate(groups: &[Vec<Item>], pad_id: u32, min_len: usize) -> Option<Batch>
Question::from_def(id: &str, def: &QuestionDef) -> Result<Question>
Question::options(&self) -> Result<Vec<String>>
State, JsonValue, QuestionDef, Question, Item, Stats, Trace, Batch, QType
```

Behavior:
- `encode_state` mirrors `Agent._encode_state`: tokenize the serialized state once (`[MASK]` replaced by a space), `truncate_left` for array states, one `Item` per question in the given order, error if any item lost option markers.
- `build_sequence` mirrors `build_sequence(..., return_stats=True)` and does not run the guard.
- Errors: `Error::Config { field, reason }` (bad question: field `questions.<id>.<key>`; tokenizer file or special token problems name the path or token; lost markers: field `head_max_len`), `Error::Unimplemented` (list label that is not string, integer or bool). `tokenizers` returns opaque errors, so tokenizer read and encode failures are reported as `Config` naming the path; no new `Error` variant is added.
- Not ported: `option_order`, `target`/`label`/`meta` collation fields, Laya's full question validation.
- The `pub use tokenizers::Tokenizer` re-export stays.
