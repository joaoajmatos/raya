//! `raya predict` with temperatures other than the fixtures' 1.0, a `noul` question and the
//! inline-JSON rule. The golden probabilities of `predict.rs` cannot see a calibration bug because
//! every fixture temperature is 1.0; here the expectation is `softmax(golden logits / T)`.
#![allow(
    clippy::float_cmp,
    clippy::excessive_precision,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use safetensors::SafeTensors;
use serde_json::{json, Value};

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/data")
}

fn read_json(path: &Path) -> Value {
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

/// A checkpoint directory assembled from the vendored fixture for `case`, with `tweak` applied to
/// the agent config before it is written.
fn setup(case: &str, tweak: impl FnOnce(&mut Value)) -> Setup {
    let meta = read_json(&data().join(case).join("meta.json"));
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
    let mut agent = read_json(&config_dir.join("rl_agent_config.json"));
    agent["max_len"] = meta["input"]["max_len"].clone();
    agent["head_max_len"] = meta["input"]["head_max_len"].clone();
    tweak(&mut agent);
    std::fs::write(dir.path().join("rl_agent_config.json"), agent.to_string()).unwrap();
    std::fs::copy(weights, dir.path().join("model.safetensors")).unwrap();
    std::fs::copy(
        data().join("tokenizer/tokenizer.json"),
        dir.path().join("tokenizer.json"),
    )
    .unwrap();
    let document = dir.path().join("document.txt");
    std::fs::write(&document, meta["input"]["state"].as_str().unwrap()).unwrap();
    Setup {
        dir,
        document,
        question: meta["input"]["question"].clone(),
    }
}

fn predict(s: &Setup, question: &str) -> Output {
    Command::new(env!("CARGO_BIN_EXE_raya"))
        .arg("predict")
        .args(["--model".as_ref(), s.dir.path().as_os_str()])
        .args(["--document".as_ref(), s.document.as_os_str()])
        .args(["--question", question])
        .output()
        .unwrap()
}

fn stdout_json(out: &Output) -> Value {
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).unwrap()
}

fn probabilities(v: &Value) -> Vec<f32> {
    v["probabilities"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

fn softmax_scaled(logits: &[f32], t: f32) -> Vec<f32> {
    let z: Vec<f32> = logits.iter().map(|l| l / t).collect();
    let max = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = z.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = e.iter().sum();
    e.iter().map(|x| x / sum).collect()
}

fn run_case(case: &str, tweak: impl FnOnce(&mut Value)) -> Vec<f32> {
    let s = setup(case, tweak);
    probabilities(&stdout_json(&predict(&s, &s.question.to_string())))
}

fn at_temp(case: &str, t: f32) -> Vec<f32> {
    run_case(case, |a| a["temperature"] = json!([t, t, t]))
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

// The fixture logits are all about -0.0067 and within 1e-4 of each other, so a temperature moves
// a probability by only ~1e-5. The comparisons below therefore use a tolerance far below that
// (the golden tolerance of 1e-5 would hide a wrong temperature) and compare runs of the CLI with
// each other: the temperature-1 run gives the logits back as `ln p` up to a constant.

#[test]
fn a_non_unit_temperature_is_applied() {
    let unit = run_case("short", |_| {});
    let logits: Vec<f32> = unit.iter().map(|p| p.ln()).collect();
    for t in [0.5f32, 2.5, 5.0] {
        let got = at_temp("short", t);
        assert!(
            max_diff(&got, &softmax_scaled(&logits, t)) < 2e-7,
            "T = {t}: {got:?}"
        );
    }
    assert!(
        max_diff(&at_temp("short", 0.5), &unit) > 5e-6,
        "T has no effect"
    );
    assert!(
        max_diff(&at_temp("short", 5.0), &unit) > 5e-6,
        "T has no effect"
    );
    assert_eq!(max_diff(&at_temp("short", 1.0), &unit), 0.0);
}

#[test]
fn the_bucket_temperature_wins_and_is_clamped() {
    assert_eq!(golden("short", "logits").len(), 3); // choice:3-5
    let bucket = |set: f32| {
        run_case("short", |a| {
            a["temperature"] = json!([1.0, 1.0, 1.0]);
            a["temperature_by_options"] = json!({ "choice:3-5": set });
        })
    };
    assert_eq!(bucket(4.0), at_temp("short", 4.0));
    assert_eq!(bucket(0.01), at_temp("short", 0.5));
    assert_eq!(bucket(99.0), at_temp("short", 5.0));
    assert!(max_diff(&bucket(4.0), &at_temp("short", 1.0)) > 5e-6);
}

#[test]
fn the_type_temperature_is_clamped() {
    assert_eq!(at_temp("short", 0.0), at_temp("short", 0.5));
    assert_eq!(at_temp("short", 50.0), at_temp("short", 5.0));
}

#[test]
fn a_score_question_uses_the_score_temperature() {
    let mixed = run_case("medium", |a| a["temperature"] = json!([9.0, 2.0, 9.0]));
    assert_eq!(mixed, at_temp("medium", 2.0));
    assert!(max_diff(&mixed, &at_temp("medium", 9.0)) > 1e-6);
}

#[test]
fn a_noul_question_runs_end_to_end() {
    let s = setup("short", |_| {});
    let q = r#"{"type":"noul","instructions":"does the document contain a stop request"}"#;
    let v = stdout_json(&predict(&s, q));
    assert_eq!(v["type"], "noul");
    let p = probabilities(&v);
    assert!(!p.is_empty());
    assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    assert_eq!(v["options"].as_array().unwrap().len(), p.len());
    assert!((0.0..=1.0).contains(&v["act_probability"].as_f64().unwrap()));
}

#[test]
fn inline_json_may_have_leading_whitespace_and_an_array_is_not_inline() {
    let s = setup("short", |_| {});
    let spaced = format!("  \n\t{}", s.question);
    assert_eq!(probabilities(&stdout_json(&predict(&s, &spaced))).len(), 3);
    // not starting with `{`: treated as a path, so the failure names --question
    let out = predict(&s, "[1, 2]");
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("--question"));
    assert!(out.stdout.is_empty());
}
