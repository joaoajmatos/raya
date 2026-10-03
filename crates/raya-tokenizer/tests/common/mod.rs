#![allow(dead_code)]

use std::path::{Path, PathBuf};

use raya_tokenizer::{JsonValue, QuestionDef, SequenceBuilder, State};

pub fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/data")
}

pub fn tokenizer_dir() -> PathBuf {
    data().join("tokenizer")
}

pub fn builder() -> SequenceBuilder {
    SequenceBuilder::from_file(&tokenizer_dir().join("tokenizer.json")).unwrap()
}

pub fn read_json(path: &Path) -> serde_json::Value {
    let text = std::fs::read_to_string(path).unwrap();
    serde_json::from_str(&text).unwrap()
}

pub fn meta() -> serde_json::Value {
    read_json(&tokenizer_dir().join("meta.json"))
}

pub fn record_names() -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(tokenizer_dir().join("records"))
        .unwrap()
        .map(|e| {
            let name = e.unwrap().file_name().into_string().unwrap();
            name.strip_suffix(".json").unwrap().to_owned()
        })
        .collect();
    names.sort();
    names
}

pub fn record(name: &str) -> serde_json::Value {
    read_json(&tokenizer_dir().join("records").join(format!("{name}.json")))
}

/// Records are written with sorted keys, so a dict or list state is rebuilt from
/// `state_serialized`, which keeps Laya's real key order.
pub fn state_of(rec: &serde_json::Value) -> State {
    if let serde_json::Value::String(s) = &rec["state"] {
        return State::Text(s.clone());
    }
    let text = rec["state_serialized"].as_str().unwrap();
    let value = JsonValue::parse(text).unwrap();
    assert_eq!(value.dumps(), text, "state text must round-trip");
    State::Json(value)
}

pub fn questions_of(questions: &serde_json::Value) -> Vec<(String, QuestionDef)> {
    questions
        .as_object()
        .unwrap()
        .iter()
        .map(|(id, def)| (id.clone(), serde_json::from_value(def.clone()).unwrap()))
        .collect()
}

pub fn def(json: &str) -> QuestionDef {
    serde_json::from_str(json).unwrap()
}
