use std::collections::{BTreeSet, HashMap};
use std::ops::Range;
use std::path::Path;

use raya_core::{Error, LoadReport, ModelConfig, Result, ShapeMismatch};
use safetensors::{Dtype, SafeTensors};

use crate::buffer::{read_file, AlignedBuf};
use crate::key::ParamKey;
use crate::mapping::expected;

/// A borrowed view of one parameter inside the store's buffer.
#[derive(Debug, Clone, Copy)]
pub struct Tensor<'a> {
    pub data: &'a [f32],
    pub shape: &'a [usize],
}

/// Every parameter of one checkpoint, validated against its config; immutable once built.
pub struct WeightStore {
    buf: AlignedBuf,
    index: HashMap<ParamKey, (Range<usize>, Vec<usize>)>,
}

impl WeightStore {
    pub fn get(&self, key: ParamKey) -> Option<Tensor<'_>> {
        let (range, shape) = self.index.get(&key)?;
        let data = bytemuck::try_cast_slice(&self.buf.as_bytes()[range.clone()]).ok()?;
        Some(Tensor { data, shape })
    }

    pub fn tensor_count(&self) -> usize {
        self.index.len()
    }

    pub fn parameter_count(&self) -> usize {
        self.index
            .values()
            .map(|(_, shape)| shape.iter().product::<usize>())
            .sum()
    }

    pub fn total_bytes(&self) -> usize {
        self.index.values().map(|(range, _)| range.len()).sum()
    }
}

/// Reads `path` and checks it against the parameter set `config` implies.
pub fn load(path: &Path, config: &ModelConfig) -> Result<WeightStore> {
    let entries = expected(config)?;
    let buf = read_file(path)?;
    let mut report = LoadReport {
        path: path.to_path_buf(),
        missing: vec![],
        unused: vec![],
        shape: vec![],
        detail: None,
    };
    let st = match SafeTensors::deserialize(buf.as_bytes()) {
        Ok(st) => st,
        Err(e) => {
            report.detail = Some(e.to_string());
            return Err(Error::Weights(Box::new(report)));
        }
    };

    let mut non_f32: Vec<String> = st
        .tensors()
        .into_iter()
        .filter(|(_, view)| view.dtype() != Dtype::F32)
        .map(|(name, view)| format!("{name} ({:?})", view.dtype()))
        .collect();
    if !non_f32.is_empty() {
        non_f32.sort();
        return Err(Error::Unimplemented(format!(
            "non-fp32 tensors in {}: {} (dtype conversion is not part of Raya-L)",
            path.display(),
            non_f32.join(", ")
        )));
    }

    let known: BTreeSet<&str> = entries.iter().map(|e| e.laya_name.as_str()).collect();
    report.unused = st
        .names()
        .into_iter()
        .filter(|name| !known.contains(*name))
        .map(str::to_owned)
        .collect();
    let base = buf.as_bytes().as_ptr() as usize;
    let mut index = HashMap::new();
    let mut misaligned = Vec::new();
    for entry in &entries {
        let Ok(view) = st.tensor(&entry.laya_name) else {
            report.missing.push(entry.laya_name.clone());
            continue;
        };
        if view.shape() != entry.shape.as_slice() || view.data().is_empty() {
            report.shape.push(ShapeMismatch {
                name: entry.laya_name.clone(),
                expected: entry.shape.clone(),
                found: view.shape().to_vec(),
            });
            continue;
        }
        if bytemuck::try_cast_slice::<u8, f32>(view.data()).is_err() {
            misaligned.push(entry.laya_name.clone());
            continue;
        }
        let start = view.data().as_ptr() as usize - base;
        index.insert(
            entry.key,
            (start..start + view.data().len(), entry.shape.clone()),
        );
    }
    if !(report.missing.is_empty() && report.unused.is_empty() && report.shape.is_empty()) {
        return Err(Error::Weights(Box::new(report)));
    }
    if !misaligned.is_empty() {
        report.detail = Some(format!(
            "tensors not 4-byte aligned in the file: {}",
            misaligned.join(", ")
        ));
        return Err(Error::Weights(Box::new(report)));
    }
    drop(st);
    Ok(WeightStore { buf, index })
}

/// Loads a Laya checkpoint directory: `model.safetensors`, `rl_agent_config.json`,
/// `encoder/config.json`.
pub fn load_dir(dir: &Path) -> Result<(ModelConfig, WeightStore)> {
    let config = ModelConfig::from_dir(dir)?;
    let store = load(&dir.join("model.safetensors"), &config)?;
    Ok((config, store))
}
