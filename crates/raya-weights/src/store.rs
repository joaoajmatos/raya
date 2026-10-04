use std::collections::{BTreeSet, HashMap};
use std::ops::Range;
use std::path::Path;

use raya_core::{Error, LoadReport, ModelConfig, Result, ShapeMismatch};
use safetensors::{Dtype, SafeTensors};

use crate::buffer::{read_file, AlignedBuf};
use crate::convert::{bf16_to_f32, f16_to_f32};
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
    source: Vec<&'static str>,
}

impl WeightStore {
    /// The dtypes the file stored its tensors in (sorted, e.g. `["BF16"]` or `["BF16", "F32"]`).
    /// The store itself always serves fp32; F16 and BF16 are widened exactly on load.
    pub fn source_dtypes(&self) -> &[&'static str] {
        &self.source
    }

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
///
/// An all-F32 file is served zero-copy from the single file buffer. F16 and BF16 tensors are
/// widened to fp32 into a second buffer, so while loading such a file the peak memory is the
/// file plus its fp32 copy (about 1.5x the final fp32 size); the file buffer is dropped after.
pub fn load(path: &Path, config: &ModelConfig) -> Result<WeightStore> {
    let entries = expected(config)?;
    let buf = read_file(path)?;
    let mut report = LoadReport {
        path: path.to_path_buf(),
        missing: vec![],
        unused: vec![],
        duplicate: vec![],
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

    let mut unsupported: Vec<String> = st
        .tensors()
        .into_iter()
        .filter(|(_, view)| !matches!(view.dtype(), Dtype::F32 | Dtype::F16 | Dtype::BF16))
        .map(|(name, view)| format!("{name} ({:?})", view.dtype()))
        .collect();
    if !unsupported.is_empty() {
        unsupported.sort();
        return Err(Error::Unimplemented(format!(
            "unsupported tensor dtypes in {}: {} (only F32, F16 and BF16 are loaded; \
             F16 and BF16 are widened to fp32)",
            path.display(),
            unsupported.join(", ")
        )));
    }

    report.duplicate = duplicate_keys(buf.as_bytes());
    let known: BTreeSet<&str> = entries.iter().map(|e| e.laya_name.as_str()).collect();
    report.unused = st
        .names()
        .into_iter()
        .filter(|name| !known.contains(*name))
        .map(str::to_owned)
        .collect();
    let base = buf.as_bytes().as_ptr() as usize;
    let mut slots = Vec::new();
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
        let start = view.data().as_ptr() as usize - base;
        slots.push(Slot {
            key: entry.key,
            name: entry.laya_name.clone(),
            shape: entry.shape.clone(),
            dtype: view.dtype(),
            range: start..start + view.data().len(),
        });
    }
    if !(report.missing.is_empty()
        && report.unused.is_empty()
        && report.duplicate.is_empty()
        && report.shape.is_empty())
    {
        return Err(Error::Weights(Box::new(report)));
    }
    drop(st);

    let mut source: Vec<&'static str> = slots.iter().map(|s| dtype_name(s.dtype)).collect();
    source.sort_unstable();
    source.dedup();
    let (buf, index) = if slots.iter().all(|s| s.dtype == Dtype::F32) {
        zero_copy(buf, &slots, report)?
    } else {
        widen(&buf, &slots)
    };
    Ok(WeightStore { buf, index, source })
}

/// One validated tensor: where its bytes sit in the file buffer and how to read them.
struct Slot {
    key: ParamKey,
    name: String,
    shape: Vec<usize>,
    dtype: Dtype,
    range: Range<usize>,
}

type Index = HashMap<ParamKey, (Range<usize>, Vec<usize>)>;

fn dtype_name(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::F32 => "F32",
        Dtype::F16 => "F16",
        Dtype::BF16 => "BF16",
        _ => unreachable!("unsupported dtypes are rejected before this point"),
    }
}

/// All-fp32 file: the store aliases the file buffer, no copy.
fn zero_copy(
    buf: AlignedBuf,
    slots: &[Slot],
    mut report: LoadReport,
) -> Result<(AlignedBuf, Index)> {
    let misaligned: Vec<&str> = slots
        .iter()
        .filter(|s| bytemuck::try_cast_slice::<u8, f32>(&buf.as_bytes()[s.range.clone()]).is_err())
        .map(|s| s.name.as_str())
        .collect();
    if !misaligned.is_empty() {
        report.detail = Some(format!(
            "tensors not 4-byte aligned in the file: {}",
            misaligned.join(", ")
        ));
        return Err(Error::Weights(Box::new(report)));
    }
    let index = slots
        .iter()
        .map(|s| (s.key, (s.range.clone(), s.shape.clone())))
        .collect();
    Ok((buf, index))
}

/// File with F16 or BF16 tensors: every tensor is copied (and widened where needed) into one
/// new fp32 buffer; the caller then drops the file buffer.
fn widen(file: &AlignedBuf, slots: &[Slot]) -> (AlignedBuf, Index) {
    let bytes = file.as_bytes();
    let total: usize = slots
        .iter()
        .map(|s| s.shape.iter().product::<usize>())
        .sum();
    let mut words: Vec<f32> = Vec::with_capacity(total);
    let mut index = HashMap::with_capacity(slots.len());
    for s in slots {
        let src = &bytes[s.range.clone()];
        let start = words.len() * 4;
        match s.dtype {
            Dtype::F32 => words.extend(
                src.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
            ),
            Dtype::F16 => words.extend(
                src.chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))),
            ),
            _ => words.extend(
                src.chunks_exact(2)
                    .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))),
            ),
        }
        // The index is byte-based like the zero-copy path; `start` is a multiple of 4 and `words`
        // is a `Vec<f32>`, so every range is 4-aligned by construction (`get` casts it back).
        index.insert(s.key, (start..words.len() * 4, s.shape.clone()));
    }
    (AlignedBuf::from_words(words), index)
}

/// Loads a Laya checkpoint directory: `model.safetensors`, `rl_agent_config.json`,
/// `encoder/config.json`.
pub fn load_dir(dir: &Path) -> Result<(ModelConfig, WeightStore)> {
    let config = ModelConfig::from_dir(dir)?;
    let store = load(&dir.join("model.safetensors"), &config)?;
    Ok((config, store))
}

/// Tensor names that occur more than once in the safetensors header (`serde_json` and the
/// `safetensors` crate would silently keep the last one). Empty if the header is unreadable;
/// `SafeTensors::deserialize` has already validated it by the time this runs.
fn duplicate_keys(bytes: &[u8]) -> Vec<String> {
    struct Keys(Vec<String>);
    impl<'de> serde::de::Visitor<'de> for Keys {
        type Value = Vec<String>;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("a JSON object")
        }
        fn visit_map<A: serde::de::MapAccess<'de>>(
            mut self,
            mut map: A,
        ) -> std::result::Result<Self::Value, A::Error> {
            while let Some((key, _)) = map.next_entry::<String, serde::de::IgnoredAny>()? {
                self.0.push(key);
            }
            Ok(self.0)
        }
    }
    let Some(len) = bytes
        .first_chunk::<8>()
        .and_then(|n| usize::try_from(u64::from_le_bytes(*n)).ok())
    else {
        return vec![];
    };
    let Some(header) = bytes.get(8..).and_then(|b| b.get(..len)) else {
        return vec![];
    };
    let mut de = serde_json::Deserializer::from_slice(header);
    let Ok(keys) = serde::Deserializer::deserialize_map(&mut de, Keys(vec![])) else {
        return vec![];
    };
    let mut seen = BTreeSet::new();
    let mut dup = BTreeSet::new();
    for key in keys {
        if key != "__metadata__" && !seen.insert(key.clone()) {
            dup.insert(key);
        }
    }
    dup.into_iter().collect()
}
