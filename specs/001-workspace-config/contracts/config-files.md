# Contract: Checkpoint configuration files

Raya reads these two files from a Laya checkpoint directory. Unknown keys are ignored.

## `encoder/config.json` (Hugging Face `ModernBertConfig`)

Required keys: `hidden_size`, `intermediate_size`, `num_hidden_layers`, `num_attention_heads`,
`global_attn_every_n_layers`, `local_attention`, `max_position_embeddings`, `vocab_size`, `norm_eps`, `norm_bias`,
`attention_bias`, `mlp_bias`, `hidden_activation`.

RoPE thetas: `global_rope_theta` and `local_rope_theta`, or `rope_parameters` with
`full_attention.rope_theta` / `sliding_attention.rope_theta`, or flat `rope_parameters.rope_theta` applying to both.
At least one form must supply both values.

Optional: `pad_token_id`, `cls_token_id`, `sep_token_id`, `bos_token_id`, `eos_token_id`.

## `rl_agent_config.json`

All optional except `temperature` when present must be a list of three numbers:
`encoder`, `head_layers` (2), `max_len` (512), `head_max_len` (192), `temperature` ([1,1,1]),
`temperature_by_options` ({}), `act_costs` ({}).

## Errors

A missing required key, a wrong type, or a violated rule produces `Error::Config` whose `field` is the JSON key
(for example `hidden_size`) and whose `reason` states the problem. Unsupported values (for example
`hidden_activation: "silu"`) produce `Error::Unimplemented`.
