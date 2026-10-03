mod common;

use common::def;
use raya_core::Error;
use raya_tokenizer::{SequenceBuilder, State};

const CHOICE: &str = r#"{"type":"choice","instructions":"which one","criteria":{"a":"first","b":"second","c":"third"}}"#;

fn config_field(e: Error) -> (String, String) {
    match e {
        Error::Config { field, reason } => (field, reason),
        other => panic!("expected Config, got {other}"),
    }
}

#[test]
fn missing_tokenizer_file_names_the_path() {
    let path = common::tokenizer_dir().join("nope.json");
    let (field, _) = config_field(SequenceBuilder::from_file(&path).err().unwrap());
    assert!(field.contains("nope.json"), "{field}");
}

#[test]
fn tokenizer_without_a_special_token_names_it() {
    let tmp = tempfile::tempdir().unwrap();
    let mut json = common::read_json(&common::tokenizer_dir().join("tokenizer.json"));
    json["model"]["vocab"]
        .as_object_mut()
        .unwrap()
        .remove("[MASK]");
    json["added_tokens"]
        .as_array_mut()
        .unwrap()
        .retain(|t| t["content"] != "[MASK]");
    let path = tmp.path().join("tokenizer.json");
    std::fs::write(&path, json.to_string()).unwrap();
    let (field, reason) = config_field(SequenceBuilder::from_dir(tmp.path()).err().unwrap());
    assert!(field.contains("tokenizer.json"), "{field}");
    assert!(reason.contains("[MASK]"), "{reason}");
}

#[test]
fn unknown_question_type_names_the_question() {
    let b = common::builder();
    let q = vec![("only".to_owned(), def(r#"{"type":"x","instructions":"i"}"#))];
    let (field, _) = config_field(
        b.encode_state(&State::Text("s".into()), &q, 96, 64)
            .unwrap_err(),
    );
    assert_eq!(field, "questions.only.type");
}

#[test]
fn lost_markers_name_question_and_limit() {
    let b = common::builder();
    let q = vec![("big".to_owned(), def(CHOICE))];
    // head_max_len 8 leaves no room for three options after the cuts and `max_len` 12 clips them.
    let (field, reason) = config_field(
        b.encode_state(&State::Text("some state".into()), &q, 12, 8)
            .unwrap_err(),
    );
    assert_eq!(field, "head_max_len");
    assert!(
        reason.contains("big") && reason.contains("head_max_len=8"),
        "{reason}"
    );
}

#[test]
fn build_sequence_returns_the_short_marker_list_without_the_guard() {
    let b = common::builder();
    let q = raya_tokenizer::Question::from_def("big", &def(CHOICE)).unwrap();
    let item = b.build_sequence(&[], false, &q, 12, 8).unwrap();
    assert!(item.markers.len() < 3);
    assert!(item.ids.len() <= 12);
}

#[test]
fn small_max_len_does_not_panic_and_never_exceeds_it() {
    let b = common::builder();
    let q = raya_tokenizer::Question::from_def("q", &def(CHOICE)).unwrap();
    let ids = b.state_ids(&State::Text("some state text".into())).unwrap();
    for max_len in 0..40 {
        for left in [false, true] {
            let item = b.build_sequence(&ids, left, &q, max_len, 64).unwrap();
            assert!(item.ids.len() <= max_len, "max_len {max_len}");
            assert!(item.markers.iter().all(|m| *m < max_len));
        }
    }
}

#[test]
fn no_room_keeps_none_of_the_state_even_when_truncating_left() {
    let b = common::builder();
    let q = raya_tokenizer::Question::from_def("q", &def(CHOICE)).unwrap();
    let ids = b
        .state_ids(&State::Text("alpha beta gamma".into()))
        .unwrap();
    let head_only = b.build_sequence(&[], true, &q, 200, 64).unwrap();
    // Exactly the head plus the closing [SEP]: room is 0.
    let max_len = head_only.ids.len();
    let item = b.build_sequence(&ids, true, &q, max_len, 64).unwrap();
    assert_eq!(item.trace.state_tokens_kept, 0);
    assert_eq!(item.ids, head_only.ids);
}

#[test]
fn empty_state_and_empty_instructions_build() {
    let b = common::builder();
    let q = vec![("q".to_owned(), def(r#"{"type":"noul","instructions":""}"#))];
    let items = b
        .encode_state(&State::Text(String::new()), &q, 96, 64)
        .unwrap();
    assert_eq!(items[0].stats.options, 2);
    assert_eq!(items[0].trace.state_tokens_total, 0);
}

#[test]
fn mask_token_in_user_text_is_replaced_by_a_space() {
    let b = common::builder();
    let q = vec![("q".to_owned(), def(CHOICE))];
    let with = b
        .encode_state(&State::Text("x [MASK] y".into()), &q, 96, 64)
        .unwrap();
    let without = b
        .encode_state(&State::Text("x   y".into()), &q, 96, 64)
        .unwrap();
    assert_eq!(with[0].ids, without[0].ids);
    let mask = b.mask_id();
    assert_eq!(with[0].ids.iter().filter(|i| **i == mask).count(), 3);
}
