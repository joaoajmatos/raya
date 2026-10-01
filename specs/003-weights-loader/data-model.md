# Data Model: Weights Loader

## ParamKey (`raya-weights/src/key.rs`)
Enum identifying one parameter: `TokEmbeddings`, `EmbedNorm`, `AttnNorm(layer)`, `Wqkv(layer)`, `AttnWo(layer)`, `MlpNorm(layer)`, `MlpWi(layer)`, `MlpWo(layer)`, `FinalNorm`, `Head { layer, part }` with `part` in `{InProjW, InProjB, OutProjW, OutProjB, Lin1W, Lin1B, Lin2W, Lin2B, Norm1W, Norm1B, Norm2W, Norm2B}`, `TypeEmb`, `Scorer { idx, bias }`, `ActHead { idx, bias }`, `Temperature`. `Copy + Eq + Hash + Debug`.

## Entry and mapping table (`mapping.rs`)
`Entry { key: ParamKey, laya_name: String, shape: Vec<usize> }`; `fn expected(cfg: &ModelConfig) -> Result<Vec<Entry>>`. Invariants: Laya names unique; `AttnNorm(0)` absent; head entries only when `head_layers > 0`; shapes per research R4.

## WeightStore (`store.rs`)
- Owns the aligned buffer and an index `ParamKey -> (byte range, shape)`.
- `get(&self, key) -> Option<Tensor<'_>>` with `Tensor { data: &[f32], shape: &[usize] }`.
- `tensor_count()`, `parameter_count()` (sum of element counts), `total_bytes()` (sum of tensor data bytes, header excluded).
- Built only by `load(path, &ModelConfig)` and `load_dir(dir)`; a store is always complete and validated.

## LoadReport (`raya-core/src/error.rs`)
`path: PathBuf`, `missing: Vec<String>`, `unused: Vec<String>`, `shape: Vec<ShapeMismatch { name, expected, found }>`, plus an optional `detail: Option<String>` for header-parse failures. `Error::Weights(LoadReport)`.

## State
None. Load is a pure function of (file, config); the store is immutable.
