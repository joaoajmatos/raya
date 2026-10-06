# Contract: forward API and `raya predict`

## Library (`raya-core`)

```rust
pub fn Model::bind(&self, params: &dyn ParamSource) -> Result<Network<'_>>;

impl Network<'_> {
    pub fn forward(
        &self,
        input: &ForwardInput<'_>,
        kernel: &dyn AttentionKernel,
        trace: Option<&mut Trace>,
    ) -> Result<ForwardOutput>;
}

pub fn calibrate(logits: &[f32], qtype: usize, agent: &AgentConfig) -> Result<Vec<f32>>;
```

- Deterministic, single-threaded, no global state. `kernel` is the only attention entry point (US3).
- Errors: `Error::Unimplemented` for unsupported shapes/activations/layer types and non-prefix masks; `Error::Config`-style messages for ids out of vocabulary, `valid_len` out of range, marker positions out of range, `qtype > 2`, or missing/mis-shaped parameters (naming the key).
- A kernel failure is surfaced as an error carrying the kernel's message.

## Trace names

See data-model.md. Tracing does not change the numbers.

## CLI

```text
raya predict --model <dir> --document <file> --question <json-or-path> [--tokenizer <path>]
```

- `--model`: directory with `model.safetensors`, `rl_agent_config.json`, `encoder/config.json` and (unless `--tokenizer`) `tokenizer.json`.
- `--document`: UTF-8 text file; its content is the state string.
- `--question`: a Laya question definition (`{"type": "choice" | "score" | "noul", "instructions": ..., "criteria": ...}`), either the JSON text itself or a path to a file containing it.
- stdout, one JSON object: `{"type": ..., "labels": [...], "probabilities": [...], "act_probability": ...}` with unrounded f32 values; `labels` are the option labels in marker order.
- Exit codes: 0 success; 1 for any error (message on stderr, no panic).
- Uses the agent config's `max_len` and `head_max_len` and the `SequenceBuilder` from spec 005.
