mod common;

use common::def;
use raya_core::Error;
use raya_tokenizer::{QType, Question};

fn options(json: &str) -> Vec<String> {
    Question::from_def("q", &def(json)).unwrap().options()
}

fn err(json: &str) -> Error {
    Question::from_def("q", &def(json)).unwrap_err()
}

#[test]
fn choice_with_descriptions_keeps_object_order() {
    assert_eq!(
        options(
            r#"{"type":"choice","instructions":"i","criteria":{"stop":"halt","continue":"go","skip":null,"none":""}}"#
        ),
        ["stop: halt", "continue: go", "skip", "none"]
    );
}

#[test]
fn choice_list_labels_use_python_str() {
    assert_eq!(
        options(r#"{"type":"choice","instructions":"i","criteria":["a",7,true,false,-3]}"#),
        ["a", "7", "True", "False", "-3"]
    );
}

#[test]
fn structured_descriptions_are_compact_json() {
    assert_eq!(
        options(
            r#"{"type":"choice","instructions":"i","criteria":{"a":{"desc":"x","n":[1,2.5]},"b":0,"c":false}}"#
        ),
        [r#"a: {"desc": "x", "n": [1, 2.5]}"#, "b: 0", "c: false"]
    );
}

#[test]
fn score_levels_are_numbered() {
    assert_eq!(
        options(r#"{"type":"score","instructions":"i","criteria":["zero","low",{"k":1}]}"#),
        ["level 0: zero", "level 1: low", r#"level 2: {"k": 1}"#]
    );
}

#[test]
fn noul_defaults_labels_and_partial_criteria() {
    assert_eq!(
        options(r#"{"type":"noul","instructions":"i"}"#),
        [
            "false: no, the statement does not hold",
            "true: yes, the statement holds"
        ]
    );
    assert_eq!(
        options(
            r#"{"type":"noul","instructions":"i","criteria":{"True":"is late","false":""},"labels":{"false":" on time ","true":"late"}}"#
        ),
        ["on time: no, the statement does not hold", "late: is late"]
    );
}

#[test]
fn non_string_instructions_become_json_text() {
    let q = Question::from_def(
        "q",
        &def(r#"{"type":"noul","instructions":{"ünï":1,"b":[true]}}"#),
    )
    .unwrap();
    assert_eq!(q.ins, r#"{"ünï": 1, "b": [true]}"#);
    assert_eq!(q.qtype, QType::Noul);
    assert_eq!(q.qtype.id(), 2);
}

#[test]
fn bad_questions_are_config_errors_naming_the_question() {
    for json in [
        r#"{"type":"other","instructions":"i"}"#,
        r#"{"type":"choice","instructions":"i"}"#,
        r#"{"type":"choice","instructions":"i","criteria":[]}"#,
        r#"{"type":"choice","instructions":"i","criteria":{}}"#,
        r#"{"type":"choice","instructions":"i","criteria":["a","a"]}"#,
        r#"{"type":"choice","instructions":"i","criteria":[true,1]}"#,
        r#"{"type":"choice","instructions":"i","criteria":[0,false]}"#,
        r#"{"type":"choice","instructions":"i","criteria":[null]}"#,
        r#"{"type":"choice","instructions":"i","criteria":[["x"]]}"#,
        r#"{"type":"choice","instructions":"i","criteria":["a"],"labels":{"false":"a","true":"b"}}"#,
        r#"{"type":"choice","instructions":"i","criteria":["a"],"labels":null}"#,
        r#"{"type":"score","instructions":"i","criteria":["a"],"labels":null}"#,
        r#"{"type":"score","instructions":"i","criteria":{"a":"b"}}"#,
        r#"{"type":"score","instructions":"i","criteria":[]}"#,
        r#"{"type":"score","instructions":"i","criteria":["a",null]}"#,
        r#"{"type":"noul","instructions":"i","criteria":{"maybe":"x"}}"#,
        r#"{"type":"noul","instructions":"i","criteria":["x"]}"#,
        r#"{"type":"noul","instructions":"i","labels":{"false":"same","true":"same"}}"#,
        r#"{"type":"noul","instructions":"i","labels":{"false":" ","true":"t"}}"#,
        r#"{"type":"noul","instructions":"i","labels":{"false":"f"}}"#,
        r#"{"type":"noul","instructions":"i","labels":{"false":"f","true":"t","x":"y"}}"#,
    ] {
        match err(json) {
            Error::Config { field, .. } => {
                assert!(field.starts_with("questions.q."), "{json}: {field}");
            }
            other => panic!("{json}: {other}"),
        }
    }
}

#[test]
fn float_labels_are_unimplemented() {
    assert!(matches!(
        err(r#"{"type":"choice","instructions":"i","criteria":[1.5]}"#),
        Error::Unimplemented(_)
    ));
}

#[test]
fn malformed_question_json_is_a_config_error_naming_the_question() {
    for json in [
        r#"{"instructions":"i"}"#,
        r#"{"type":"choice"}"#,
        r#"{"type":7,"instructions":"i"}"#,
        "not json",
    ] {
        match raya_tokenizer::QuestionDef::from_json("q7", json) {
            Err(Error::Config { field, reason }) => {
                assert_eq!(field, "questions.q7", "{json}");
                assert!(!reason.is_empty());
            }
            other => panic!("{json}: {other:?}"),
        }
    }
    assert!(
        raya_tokenizer::QuestionDef::from_json("q7", r#"{"type":"noul","instructions":"i"}"#)
            .is_ok()
    );
}

#[test]
fn null_labels_on_noul_mean_default_labels() {
    assert_eq!(
        options(r#"{"type":"noul","instructions":"i","labels":null}"#),
        [
            "false: no, the statement does not hold",
            "true: yes, the statement holds"
        ]
    );
}
