# Research: Reference Forward Pass

Sources: pinned laya-sparse `357ad53` (`laya/common.py`, `laya/agent.py`, `experiments/golden/forward.py`) and the `transformers` 5.18 `modeling_modernbert.py` that produced the goldens (`meta.json` `versions`).

## R1. What the encoder is

Laya's encoder is Hugging Face `ModernBertModel`; its source is not in the submodule. The goldens fix the version (transformers 5.18.0). Behavior ported: embeddings = `tok_embeddings` then `LayerNorm(eps=norm_eps, bias=norm_bias)`; layer 0 has `attn_norm = Identity`; pre-norm residual attention and MLP; `final_norm`. The golden hooks record each layer's output and the final-norm output, and `encoder_output` equals `encoder_final_norm`.

## R2. RoPE

`inv_freq = 1 / theta^(arange(0, head_dim, 2) / head_dim)`, `freqs = pos * inv_freq`, `emb = cat(freqs, freqs)`, `q' = q cos + rotate_half(q) sin` with `rotate_half(x) = cat(-x2, x1)` over the two halves of `head_dim`. Theta per layer type: `full_attention` -> `global_rope_theta`, `sliding_attention` -> `local_rope_theta`. Positions are `0..seq` (the golden `position_ids`).

## R3. Local window

`sliding_window = local_attention / 2` and the mask is `|i - j| <= sliding_window` (`sliding_window_bidirectional_overlay`). Fixture: `local_attention = 16`, so 8 per side; the real checkpoint uses 128, so 64 per side (the 007 `WINDOW`). `layer_types` in the golden `meta.json` agree with `is_global_layer` (`layer % global_attn_every_n_layers == 0`).

## R4. Padding

Encoder: the HF attention mask removes padded keys; padded query rows still compute and produce non-zero values in Laya. Head: `src_key_padding_mask`. The 007 trait zeroes padded query rows and only supports a prefix of valid keys. Valid rows are identical either way because padded keys are masked everywhere, so goldens are compared on rows `0..valid_len`. The `padded` case is right-padded (`unpadded_length` 82 of 160). A non-prefix mask is rejected with `Error::Unimplemented`; this is a Raya limitation, not a Laya behavior.

## R5. Decision head layer

`nn.TransformerEncoderLayer(d, nhead = max(1, d / 64), 4d, dropout, batch_first=True, norm_first=True)` with default `activation=relu`, `layer_norm_eps=1e-5`, and `_DynamicMultiheadAttention` (packed `in_proj_weight`/`in_proj_bias`, `out_proj`, scale via `scaled_dot_product_attention`, i.e. `head_dim^-0.5`). The hidden-128 case uses 2 heads of 64, so `head_dim` is a multiple of 8 and the 007 shape check passes; hidden 64 uses 1 head of 64.

## R6. Scorer, features, act head

`scorer = LayerNorm(d) -> Linear(d, d) -> GELU() -> Linear(d, 1)`; `GELU()` is exact (erf). Logits for invalid markers are set to `-1e4`. `k = marker_mask.sum.clamp(min=2)`, `ent = -sum(p ln max(p, 1e-9)) / ln k`, features `[top1, top1 - top2, ent, k / 255]`. For one marker, softmax is 1.0 and `top2 = 0`. Note `ent` uses `k` clamped to at least 2 and `p` over all marker slots including masked ones (their probability is about `e^-1e4 = 0`). `act_head = Linear(d + 4, 256) -> GELU -> Linear(256, n_act)`. `pooled = h[:, 0]` is the CLS row after the head.

## R7. Calibration (outside `forward`)

`DecisionModel.forward` does not use `temperature`. `Agent._decode_answers`: `k = number of markers`, `t = temperature_by_options[temp_bucket(qtype, k)]` else `temperature[qtype]`, both stored after `clamp_temperature` (non-finite becomes 1.0, then clamp to [0.5, 5.0]); `z = logits[:k] / t`; `p = exp(z - max z) / sum`. `temp_bucket(qtype, k) = "<qtype name>:<size>"` with size `"2"` for `k <= 2`, `"3-5"`, `"6-10"`, `"11+"`. Per-language overrides (`lang_temperatures`) are not part of the checkpoint and out of scope. Observation to report, not fix: Laya clamps the shipped `choice:11+` temperature 0.1006 up to 0.5.

## R8. Exact GELU needs `erf`

Rust std has no `erf` and the spec limits dependencies, so `ops.rs` implements it (f64: Taylor series below |x| 2.5, continued fraction for `erfc` above, saturation at |x| 6). Checked against `erf(0.5) = 0.5204998778130465`, `erf(1) = 0.8427007929497149`, `erf(2) = 0.9953222650189527`, `erf(3) = 0.9999779095030014`.

## R9. Tolerances

Spec: 1e-4 absolute for embedding, encoder layers and head layers; 1e-5 for probabilities. `logits` and `act_logits` are pre-softmax values of order 1 and are held to 1e-4 with the other intermediates; `probabilities` and `act_features` to 1e-5. After the first run the measured maxima are recorded in `quickstart.md`; tightening happens only if the measured error is at least 10x below the stated bound, with that evidence cited. Torch's fp32 GEMM (MKL) and ours sum in different orders, so errors of a few 1e-6 are expected and are not a defect.

## R10. Tensor runtime (for `docs/tensor-runtime.md`)

Options: (a) `candle`, (b) `ndarray` plus a BLAS binding, (c) plain Rust loops. Decision for 006: (c). Reasons that can be stated without a benchmark: no new dependency (constitution VII), builds identically on Windows, macOS and Linux (BLAS bindings and `candle`'s optional MKL/Accelerate backends are the usual source of Windows and CI friction), and the attention kernel stays ours. Large-GEMM throughput of (a) and (b) is expected to exceed plain loops; that is a hypothesis, no run exists. The decision is revisited by feature 008 numbers on the target CPU, which are human-owned.

## R11. Dependency cycle

`raya-weights` depends on `raya-core`, so `raya-core` cannot use `WeightStore`. `ParamKey` and `HeadPart` move to `raya-core::params` (re-exported by `raya-weights`, same paths), and `raya-core` defines `ParamSource`, implemented by `WeightStore`.

## R12. Predict inputs

`raya predict --model <dir> --document <file> --question <json|path> [--tokenizer <path>]`. The sequence is built with `SequenceBuilder::build_sequence` using the agent config's `max_len` and `head_max_len`, matching `Agent._encode_state`; the state is the document text (a plain string). The tokenizer from 005 is used unchanged.
