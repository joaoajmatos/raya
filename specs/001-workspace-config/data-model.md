# Data Model: Rust Workspace, CI and Config Types

All types live in `raya-core`.

## EncoderConfig (from `encoder/config.json`)

| Field | Type | Rule |
|---|---|---|
| hidden_size | usize | > 0, divisible by num_attention_heads |
| intermediate_size | usize | > 0 |
| num_hidden_layers | usize | > 0 |
| num_attention_heads | usize | > 0 |
| global_attn_every_n_layers | usize | >= 1 |
| local_attention | usize | >= 1 (total window width; +/-local_attention/2 per side) |
| global_rope_theta, local_rope_theta | f64 | > 0 |
| norm_eps | f64 | > 0 |
| norm_bias, attention_bias, mlp_bias | bool | |
| hidden_activation | `Activation::Gelu` | other values: `Unimplemented` |
| max_position_embeddings, vocab_size | usize | > 0 |
| pad/cls/sep/bos/eos_token_id | Option<u32> | recorded only |

Method: `is_global_layer(i) -> bool` = `i % global_attn_every_n_layers == 0`; `global_layer_count()`.

## AgentConfig (from `rl_agent_config.json`)

| Field | Type | Rule |
|---|---|---|
| encoder | String | reference name, informational |
| head_layers | usize | default 2 |
| max_len | usize | default 512, > 0 |
| head_max_len | usize | default 192, > 0 |
| temperature | [f32; 3] | exactly 3 finite values |
| temperature_by_options | BTreeMap<String, f32> | default empty |
| act_costs | BTreeMap<String, f32> | default empty; `n_act = len + 1` |

## HeadConfig (derived)

`layers` (= head_layers), `heads` = max(1, hidden/64), `ffn_size` = 4 * hidden. Invariant: hidden % heads == 0.

## ModelConfig

`{ encoder: EncoderConfig, agent: AgentConfig }` plus `head() -> HeadConfig`.
Constructors: `ModelConfig::from_dir(path)`, `from_json(encoder_json, agent_json)`.

## Shape types (`shape.rs`)

```text
ModelShape { segments: Vec<Segment>, compression: Compression, head: HeadConfig }
Segment    { layers: usize, attention: AttentionKind }
AttentionKind = Full | BlockLocal{w} | Gas{s} | Hybrid | LayaInterleaved{global_every, window}
Compression   = None
```

`ModelShape::from_config(&ModelConfig)` yields one segment of `num_hidden_layers` with
`LayaInterleaved { global_every: global_attn_every_n_layers, window: local_attention }` and `Compression::None`.
`ModelShape::ensure_supported()` returns `Error::Unimplemented` for any segment that is not `LayaInterleaved`,
any segment count other than one, or any compression other than `None`.

## Error

`Error::Config { field: String, reason: String }`, `Error::Io { path, source }`, `Error::Json { path, source }`,
`Error::Unimplemented(String)`.

## Model shell

`Model { config: ModelConfig, shape: ModelShape }`; `Model::new(config)` validates the shape is supported. No
forward pass.
