use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/data")
}

fn raya(dir: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_raya"))
        .arg("info")
        .arg(dir)
        .output()
        .unwrap()
}

/// Configs come from `inventory.json` `meta`, which matches the h64 weights
/// (`fixture-tiny/` has a different `vocab_size`).
fn checkpoint(with_weights: bool) -> tempfile::TempDir {
    let tmp = tempfile::tempdir().unwrap();
    let inventory = std::fs::read_to_string(data().join("inventory.json")).unwrap();
    let meta = &serde_json::from_str::<serde_json::Value>(&inventory).unwrap()["meta"];
    std::fs::create_dir(tmp.path().join("encoder")).unwrap();
    std::fs::write(
        tmp.path().join("encoder/config.json"),
        meta["encoder_config"].to_string(),
    )
    .unwrap();
    std::fs::write(
        tmp.path().join("rl_agent_config.json"),
        meta["agent_config"].to_string(),
    )
    .unwrap();
    if with_weights {
        std::fs::copy(
            data().join("weights-h64.safetensors"),
            tmp.path().join("model.safetensors"),
        )
        .unwrap();
    }
    tmp
}

#[test]
fn info_prints_computed_numbers() {
    let tmp = checkpoint(true);
    let out = raya(tmp.path());
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8(out.stdout).unwrap();
    for line in [
        "hidden_size: 64",
        "encoder_layers: 3",
        "head_layers: 1",
        "tensors: 44",
        "parameters: 210822",
        &format!("bytes: {}", 210_822 * 4),
    ] {
        assert!(
            text.lines().any(|l| l == line),
            "missing `{line}` in:\n{text}"
        );
    }
}

#[test]
fn missing_weights_exit_1_with_path_on_stderr() {
    let tmp = checkpoint(false);
    let out = raya(tmp.path());
    assert_eq!(out.status.code(), Some(1));
    assert!(out.stdout.is_empty());
    assert!(String::from_utf8_lossy(&out.stderr).contains("model.safetensors"));
}

#[test]
fn missing_argument_is_a_usage_error() {
    let out = Command::new(env!("CARGO_BIN_EXE_raya"))
        .arg("info")
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
}
