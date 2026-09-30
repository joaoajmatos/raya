# Research: Rust Workspace, CI and Config Types

## R1. Where Laya's configuration lives

**Decision**: Read two files from a checkpoint directory: `encoder/config.json` and `rl_agent_config.json`.

**Rationale**: `build_model` (`laya/common.py`) builds the encoder with `AutoConfig.from_pretrained(encoder_dir)` and
takes `head_layers`, `act_costs` and the rest from `rl_agent_config.json`; `Agent` reads `max_len`, `head_max_len`,
`temperature`, `temperature_by_options` from the same agent file. The test fixture
(`tests/experiments/conftest.py`) writes exactly this layout. A real `ModernBertConfig` dump was generated with
transformers 4.57.1 and is checked in as the fixture.

**Alternatives considered**: a single merged Raya config file. Rejected: Raya must load Laya checkpoints unchanged
(feasibility section 3.3).

## R2. Field mapping

| Raya field | Source | Notes |
|---|---|---|
| hidden_size, intermediate_size, num_hidden_layers, num_attention_heads, max_position_embeddings, vocab_size | encoder | required |
| global_attn_every_n_layers, local_attention | encoder | required; layer `i` is global iff `i % n == 0` (fixture: n=2, layers 0 and 2 global) |
| global_rope_theta, local_rope_theta | encoder | flat keys (4.x) or `rope_parameters.{full_attention,sliding_attention}.rope_theta` (5.x), as `_apply_rope_config` does; a flat `rope_parameters.rope_theta` is the fallback |
| norm_eps, norm_bias, attention_bias, mlp_bias, hidden_activation | encoder | required; only `gelu` accepted, anything else is `Unimplemented` |
| pad/cls/sep/bos/eos token ids | encoder | optional; recorded, not interpreted |
| head_layers | agent | default 2 as in `build_model` |
| act_costs | agent | `n_act = len(act_costs) + 1`, as in `build_model` |
| max_len (512), head_max_len (192) | agent | defaults as in `Agent` |
| temperature (3 floats), temperature_by_options | agent | list of exactly 3 floats, as `Agent` enforces; clamping is forward-pass behavior (feature 006) |

**Marker tokens**: the `[MASK]` marker id is a tokenizer property (`tok.mask_token_id`), absent from the encoder
config. The config carries the special-token ids it has; the marker is resolved by feature 005.

## R3. Expressing Laya's interleaved attention as one segment

**Decision**: add `AttentionKind::LayaInterleaved { global_every, window }` and convert Laya's shape to one
`Segment { layers, attention: LayaInterleaved }`. `Full`, `BlockLocal`, `Gas` and `Hybrid` stay defined and return
`Error::Unimplemented` when a model is built from them.

**Rationale**: Laya's encoder mixes global and local layers, so no single one of the four listed kinds describes
it. Mapping local layers to `BlockLocal{w}` would be wrong: native local layers use an exact +/-window mask,
while block-local is a different, approximately equivalent kernel pattern that feature 007 verifies.

**Alternatives considered**: per-layer attention vectors in `Segment` (heavier, and the repetition hides the simple
rule); mapping to `Hybrid` (its meaning in the feasibility doc is token-level global tokens plus local, not layer
interleaving).

## R4. Decision-head sizing

**Decision**: `HeadConfig { layers: head_layers, heads: max(1, hidden/64), ffn: 4*hidden }`, validated so
`hidden % heads == 0`; encoder with `norm_first` semantics is forward-pass detail.

**Rationale**: `DecisionModel.__init__`: `nhead = max(1, d // 64)`, `TransformerEncoderLayer(d, nhead, 4*d, norm_first=True)`.

## R5. Validation policy

**Decision**: required fields must be present; no silent defaults for architecture fields. Agent-level fields with
defaults in Laya keep those defaults. Errors are `Error::Config { field, reason }` naming the JSON key.

**Rationale**: a wrong guess at an architecture field would produce wrong numbers far from its cause (constitution
VII). Consistency checks: positive sizes; `hidden % num_attention_heads == 0`; `global_attn_every_n_layers >= 1`;
`local_attention >= 1`; `temperature` has three finite values; `head_layers` may be 0 (Laya allows it).

## R6. Workspace lints and unsafe policy

**Decision**: `[workspace.lints]` with `clippy::pedantic` as warn (priority -1) and a short allow list
(`module_name_repetitions`, `missing_errors_doc`, `missing_panics_doc`, `must_use_candidate`); CI runs
`clippy --all-targets -D warnings`. `#![forbid(unsafe_code)]` sits in each crate root except `raya-kernels`; a
test reads each crate root and fails if one is missing, because workspace lints cannot express a per-crate
exception to `forbid`.

## R7. Dependency audit

**Decision**: `cargo audit` through the `rustsec/audit-check` action (spec allows cargo-deny or cargo-audit).

**Rationale**: no policy file to maintain; license policy is out of scope.

## R8. `tokenizers` features

**Decision**: `default-features = false, features = ["onig"]`.

**Rationale**: the default set pulls progress bars and other unused pieces; the regex backend is required by
ModernBERT-style tokenizers. To be revisited in feature 005 if the tokenizer needs more.

## R9. CLI stubs

**Decision**: `info`, `predict`, `bench` take no arguments yet; each prints `not implemented (feature NNN)` to
stderr and exits with status 2.

**Rationale**: later features own their argument shapes; declaring them now would duplicate and drift.
