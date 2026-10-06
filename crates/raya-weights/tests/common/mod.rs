#![allow(dead_code)]

pub mod half;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use raya_core::ModelConfig;
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};

pub fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/data")
}

pub fn h64_path() -> PathBuf {
    data().join("weights-h64.safetensors")
}

pub fn h128_path() -> PathBuf {
    data().join("weights-h128.safetensors")
}

/// Built from `inventory.json` `meta`, which records the config the h64 weights were saved
/// with (`fixture-h64/` now carries the same config).
pub fn h64_config() -> ModelConfig {
    let meta = &inventory()["meta"];
    ModelConfig::from_json(
        &meta["encoder_config"].to_string(),
        &meta["agent_config"].to_string(),
    )
    .unwrap()
}

pub fn h128_config() -> ModelConfig {
    let text = std::fs::read_to_string(data().join("fastpath_off/meta.json")).unwrap();
    let meta: serde_json::Value = serde_json::from_str(&text).unwrap();
    ModelConfig::from_json(
        &meta["encoder_config"].to_string(),
        &meta["agent_config"].to_string(),
    )
    .unwrap()
}

pub fn inventory() -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(data().join("inventory.json")).unwrap()).unwrap()
}

/// One tensor of a mutated checkpoint: dtype, shape and raw bytes.
pub type Raw = (Dtype, Vec<usize>, Vec<u8>);

pub fn read_all(path: &Path) -> BTreeMap<String, Raw> {
    let bytes = std::fs::read(path).unwrap();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    st.tensors()
        .into_iter()
        .map(|(n, v)| (n, (v.dtype(), v.shape().to_vec(), v.data().to_vec())))
        .collect()
}

pub fn write_all(path: &Path, tensors: &BTreeMap<String, Raw>) {
    let views: Vec<(String, TensorView<'_>)> = tensors
        .iter()
        .map(|(n, (dt, shape, data))| {
            (
                n.clone(),
                TensorView::new(*dt, shape.clone(), data).unwrap(),
            )
        })
        .collect();
    safetensors::serialize_to_file(views, None, path).unwrap();
}

/// Copies h64 into `dir`, applying `mutate`, and returns the new path.
pub fn mutated_h64(dir: &Path, mutate: impl FnOnce(&mut BTreeMap<String, Raw>)) -> PathBuf {
    let mut tensors = read_all(&h64_path());
    mutate(&mut tensors);
    let path = dir.join("model.safetensors");
    write_all(&path, &tensors);
    path
}
