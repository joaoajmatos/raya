//! `raya predict` against the golden probabilities (spec 006 US2, SC-002).
#![allow(
    clippy::float_cmp,
    clippy::excessive_precision,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use safetensors::SafeTensors;
use serde_json::Value;

const TOL: f32 = 1e-5;

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/data")
}

fn json(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn golden(case: &str, name: &str) -> Vec<f32> {
    let bytes = std::fs::read(data().join(case).join("tensors.safetensors")).unwrap();
    let t = SafeTensors::deserialize(&bytes).unwrap();
    t.tensor(name)
        .unwrap()
        .data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

struct Setup {
    dir: tempfile::TempDir,
    document: PathBuf,
    question: Value,
}

/// A checkpoint directory assembled from the vendored fixture for `case`, with the generator's
/// per-call `max_len` / `head_max_len` written into `rl_agent_config.json`.
fn setup(case: &str, with_tokenizer: bool) -> Setup {
    let meta = json(&data().join(case).join("meta.json"));
    let config_dir = data().join(case).join(meta["config_dir"].as_str().unwrap());
    let weights = data()
        .join(case)
        .join(meta["weights_file"].as_str().unwrap());
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join("encoder")).unwrap();
    std::fs::copy(
        config_dir.join("encoder/config.json"),
        dir.path().join("encoder/config.json"),
    )
    .unwrap();
    let mut agent = json(&config_dir.join("rl_agent_config.json"));
    agent["max_len"] = meta["input"]["max_len"].clone();
    agent["head_max_len"] = meta["input"]["head_max_len"].clone();
    std::fs::write(dir.path().join("rl_agent_config.json"), agent.to_string()).unwrap();
    std::fs::copy(weights, dir.path().join("model.safetensors")).unwrap();
    if with_tokenizer {
        std::fs::copy(
            data().join("tokenizer/tokenizer.json"),
            dir.path().join("tokenizer.json"),
        )
        .unwrap();
    }
    let document = dir.path().join("document.txt");
    std::fs::write(&document, meta["input"]["state"].as_str().unwrap()).unwrap();
    Setup {
        dir,
        document,
        question: meta["input"]["question"].clone(),
    }
}

fn predict(model: &Path, document: &Path, question: &str, extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_raya"))
        .arg("predict")
        .args(["--model".as_ref(), model.as_os_str()])
        .args(["--document".as_ref(), document.as_os_str()])
        .args(["--question", question])
        .args(extra)
        .output()
        .unwrap()
}

fn check_case(case: &str) {
    let s = setup(case, true);
    let out = predict(s.dir.path(), &s.document, &s.question.to_string(), &[]);
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: Value = serde_json::from_slice(&out.stdout).unwrap();

    let want = golden(case, "probabilities");
    let got: Vec<f32> = v["probabilities"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    assert_eq!(got.len(), want.len(), "{case}");
    for (g, w) in got.iter().zip(&want) {
        assert!((g - w).abs() <= TOL, "{case}: {g} vs golden {w}");
    }
    assert!((got.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    assert_eq!(v["options"].as_array().unwrap().len(), want.len());

    // act_probability = softmax(act_logits)[0]
    let act = golden(case, "act_logits");
    let max = act.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = act.iter().map(|a| (a - max).exp()).sum();
    let want_act = (act[0] - max).exp() / sum;
    let got_act = v["act_probability"].as_f64().unwrap() as f32;
    assert!(
        (got_act - want_act).abs() <= TOL,
        "{case}: {got_act} vs {want_act}"
    );
}

#[test]
fn short_choice_matches_golden() {
    check_case("short");
}

#[test]
fn medium_score_matches_golden() {
    check_case("medium");
}

#[test]
fn fastpath_off_h128_matches_golden() {
    check_case("fastpath_off");
}

#[test]
fn question_can_be_a_file_and_the_tokenizer_a_flag() {
    let s = setup("short", false);
    let qfile = s.dir.path().join("question.json");
    std::fs::write(&qfile, s.question.to_string()).unwrap();
    let tok = data().join("tokenizer/tokenizer.json");
    let out = predict(
        s.dir.path(),
        &s.document,
        qfile.to_str().unwrap(),
        &["--tokenizer", tok.to_str().unwrap()],
    );
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn fails_cleanly(out: &Output, needle: &str) {
    assert_eq!(out.status.code(), Some(1));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains(needle), "stderr: {err}");
    assert!(!err.contains("panicked"), "stderr: {err}");
    assert!(out.stdout.is_empty());
}

#[test]
fn errors_exit_1_with_a_message() {
    let s = setup("short", true);
    let q = s.question.to_string();
    let missing = s.dir.path().join("nope.txt");
    fails_cleanly(&predict(s.dir.path(), &missing, &q, &[]), "nope.txt");
    fails_cleanly(
        &predict(s.dir.path(), &s.document, "{not json", &[]),
        "question",
    );
    fails_cleanly(
        &predict(s.dir.path(), &s.document, "no-such-question-file.json", &[]),
        "--question",
    );
    let bad_type = r#"{"type":"rank","instructions":"x","criteria":["a","b"]}"#;
    fails_cleanly(&predict(s.dir.path(), &s.document, bad_type, &[]), "rank");

    let no_tok = setup("short", false);
    fails_cleanly(
        &predict(no_tok.dir.path(), &no_tok.document, &q, &[]),
        "tokenizer",
    );
    let nowhere = s.dir.path().join("elsewhere");
    fails_cleanly(
        &predict(&nowhere, &s.document, &q, &[]),
        "cannot load model",
    );
}

#[test]
fn missing_arguments_are_a_usage_error() {
    let out = Command::new(env!("CARGO_BIN_EXE_raya"))
        .arg("predict")
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("--model"));
}
