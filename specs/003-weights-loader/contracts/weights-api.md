# Contract: `raya-weights` public API

```text
pub fn load(path: &Path, config: &ModelConfig) -> Result<WeightStore>
pub fn load_dir(dir: &Path) -> Result<(ModelConfig, WeightStore)>
    // dir = model.safetensors + rl_agent_config.json + encoder/config.json

WeightStore::get(ParamKey) -> Option<Tensor<'_>>
WeightStore::{tensor_count, parameter_count, total_bytes}
Tensor { data: &[f32], shape: &[usize] }
mapping::expected(&ModelConfig) -> Result<Vec<Entry>>
```

Behavior:
- `config` goes through `ModelShape::from_config` and `ensure_supported` before the file is opened.
- Success implies: every expected key present exactly once, shapes equal, dtype F32, no unexpected tensor.
- Failure: `Error::Io { path }` (cannot read), `Error::Weights(report)` (missing, unused, shape, header parse), `Error::Unimplemented` (non-F32 dtype, bias-enabled config), `Error::Config` (config validation).
- Slices alias the store's single buffer; no per-tensor allocation.
- The placeholder `SafeTensors` re-export in `raya-weights/src/lib.rs` is removed.
