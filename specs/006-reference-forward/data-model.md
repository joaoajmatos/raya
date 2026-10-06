# Data Model: Reference Forward Pass

All tensors are row-major fp32 `Vec<f32>` with an explicit shape.

## ParamKey, HeadPart (moved to `raya-core::params`)

Unchanged enums from `raya-weights` (typed identity of one Laya parameter). Re-exported by `raya-weights`.

## ParamSource (trait, `raya-core`)

`fn param(&self, key: ParamKey) -> Option<(&[f32], &[usize])>`. Implemented by `WeightStore`. Returns data and shape; `Model::bind` checks every required key and shape and returns `Error::Config` or `Error::Weights`-style messages naming the key.

## Model and Network

- `Model` (existing shell): validated `ModelConfig` and `ModelShape`.
- `Model::bind(&self, params: &dyn ParamSource) -> Result<Network<'_>>`: resolves every weight slice once.
- `Network<'w>`: borrowed slices per layer (`EncoderLayer { attn_norm: Option<..>, wqkv, wo, mlp_norm, wi, mlp_wo }`, `HeadLayer { in_proj_w, in_proj_b, out_proj_w, out_proj_b, lin1_w, lin1_b, lin2_w, lin2_b, norm1_w, norm1_b, norm2_w, norm2_b }`, `type_emb`, `scorer`, `act_head`).

## ForwardInput

| Field | Type | Rule |
|---|---|---|
| `input_ids` | `&[u32]` | length `seq`; each id `< vocab_size` |
| `valid_len` | `usize` | `1..=seq`; positions `>= valid_len` are padding |
| `marker_pos` | `&[usize]` | `k` entries, each `< valid_len` where `marker_mask` is true |
| `marker_mask` | `&[bool]` | same length as `marker_pos` |
| `qtype` | `usize` | `0..3` |

Built from a tokenizer `Batch` row (`attention_mask` must be a prefix of ones, else `Error::Unimplemented`).

## ForwardOutput

`logits: Vec<f32>` (len `k`, invalid markers `-1e4`), `act_logits: Vec<f32>` (len `n_act`), `features: [f32; 4]`, `pooled: Vec<f32>`.

## Trace

Ordered list of `(name: String, shape: Vec<usize>, data: Vec<f32>)`. Names: `embedding_output`, `encoder_layer_NN`, `encoder_final_norm`, `type_emb_add`, `head_input`, `head_layer_NN`, `marker_states`, `logits`, `probabilities`, `act_features`, `pooled`, `act_logits`. `Trace::get(name)`.

## Calibrated

`Calibrated { probabilities: Vec<f32>, temperature: f32, act_probability: f32 }`.
`calibrate(logits: &[f32] (k valid), qtype, agent: &AgentConfig) -> Result<Vec<f32>>` implements research R7.

## Mask (extended in `raya-kernels`)

`Mask::{Block3, Window64, Full, Sliding(usize)}`. `Full`: all keys `< valid_len`. `Sliding(w)`: `|i - j| <= w`. `Tiled` supports only `Block3` and `Window64`.

## GemmBackend (extended in `raya-kernels`)

`matmul_nt(&self, out: &mut [f32], a: &[f32], b: &[f32], m, n, k)`: `out[i][j] = sum_t a[i][t] * b[j][t]`; `Reference` implements it with plain loops.
