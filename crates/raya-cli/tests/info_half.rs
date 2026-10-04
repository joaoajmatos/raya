use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/data")
}

/// A checkpoint dir whose weights are the h64 fixture narrowed to bf16 (high half of each f32),
/// generated in a temp dir; the vendored file is only read.
fn bf16_checkpoint() -> tempfile::TempDir {
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

    let bytes = std::fs::read(data().join("weights-h64.safetensors")).unwrap();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let narrowed: BTreeMap<String, (Vec<usize>, Vec<u8>)> = st
        .tensors()
        .into_iter()
        .map(|(name, view)| {
            let half: Vec<u8> = view
                .data()
                .chunks_exact(4)
                .flat_map(|c| [c[2], c[3]])
                .collect();
            (name, (view.shape().to_vec(), half))
        })
        .collect();
    let views: Vec<(String, TensorView<'_>)> = narrowed
        .iter()
        .map(|(name, (shape, data))| {
            (
                name.clone(),
                TensorView::new(Dtype::BF16, shape.clone(), data).unwrap(),
            )
        })
        .collect();
    safetensors::serialize_to_file(views, None, &tmp.path().join("model.safetensors")).unwrap();
    tmp
}

#[test]
fn info_reports_bf16_source_and_fp32_resident_size() {
    let tmp = bf16_checkpoint();
    let out = Command::new(env!("CARGO_BIN_EXE_raya"))
        .arg("info")
        .arg(tmp.path())
        .output()
        .unwrap();
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8(out.stdout).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert!(lines.contains(&"parameters: 210822"), "{text}");
    assert!(lines.contains(&"source_dtype: BF16"), "{text}");
    // `bytes` counts the fp32 data the store holds, not the half-size file.
    let want = format!("bytes: {}", 210_822 * 4);
    assert!(lines.contains(&want.as_str()), "{text}");
}
