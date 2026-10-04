# Contract: `raya info <checkpoint-dir>`

Input: a directory with `model.safetensors`, `rl_agent_config.json`, `encoder/config.json`.

Stdout on success, `key: value` lines in this order:

```text
checkpoint: <dir>
hidden_size: <d>
encoder_layers: <L>
attention_heads: <n>
head_layers: <H>
tensors: <count>
parameters: <sum of tensor element counts>
bytes: <sum of resident fp32 tensor bytes>
source_dtype: <dtypes in the file, sorted, joined by +, e.g. F32 or BF16>
```

Exit 0. On any loader error the report goes to stderr and the exit code is 1. A missing argument is a clap usage error (exit 2). The numbers are computed from loaded tensors, never read from docs or the inventory.
