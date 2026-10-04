# Research: Weights Loader

## R1. Memory-map or read once
- **Decision**: read once into a `Vec<f32>` of `ceil(len/4)` words, viewed as bytes with `bytemuck::cast_slice_mut` and filled by `File::read_exact`. Tensors are `bytemuck::try_cast_slice` views into the same buffer.
- **Rationale**: `Vec<f32>` guarantees 4-byte alignment; safetensors pads its header to 8 bytes so tensor data starts aligned; no `unsafe` (constitution, spec FR-008). One buffer: peak memory equals file size.
- **Alternatives**: `memmap2` (needs `unsafe`, deferred); `Vec<u8>` plus per-tensor copy (extra memory); `slice::from_raw_parts` (`unsafe`).
- **Edge**: a file length that is not a multiple of 4 is padded in the buffer and the view truncated to the real length; a tensor whose offset is not 4-aligned is an error naming the tensor (checked, not assumed).

## R2. Header parsing
- **Decision**: `safetensors::SafeTensors::deserialize` on the byte view for header validation, dtype, shape and offsets. No hand-rolled parser.
- **Rationale**: already a workspace dependency; it validates offsets against the buffer length. `__metadata__` is ignored; its errors are wrapped with the file path.

## R3. Endianness
- **Decision**: safetensors is little-endian; the crate has a `compile_error!` on big-endian targets instead of byte-swapping.
- **Rationale**: all supported targets are little-endian; swapping would be an unrequested code path.

## R4. Expected parameter set (from `inventory.json`, `laya/common.py`)
With `d = hidden_size`, `i = intermediate_size`, `V = vocab_size`, `L = num_hidden_layers`, `H = head_layers`, `n_act = AgentConfig::n_act()`:

| Laya name | Shape |
|---|---|
| `encoder.embeddings.tok_embeddings.weight` | [V, d] |
| `encoder.embeddings.norm.weight` | [d] |
| `encoder.layers.N.attn_norm.weight` (N >= 1 only) | [d] |
| `encoder.layers.N.attn.Wqkv.weight` | [3d, d] |
| `encoder.layers.N.attn.Wo.weight` | [d, d] |
| `encoder.layers.N.mlp_norm.weight` | [d] |
| `encoder.layers.N.mlp.Wi.weight` | [2i, d] |
| `encoder.layers.N.mlp.Wo.weight` | [d, i] |
| `encoder.final_norm.weight` | [d] |
| `head.layers.K.self_attn.in_proj_weight` / `in_proj_bias` | [3d, d] / [3d] |
| `head.layers.K.self_attn.out_proj.weight` / `.bias` | [d, d] / [d] |
| `head.layers.K.linear1.weight` / `.bias` | [4d, d] / [4d] |
| `head.layers.K.linear2.weight` / `.bias` | [d, 4d] / [d] |
| `head.layers.K.norm1.{weight,bias}`, `norm2.{weight,bias}` | [d] |
| `type_emb.weight` | [3, d] |
| `scorer.0.{weight,bias}` | [d] |
| `scorer.1.weight` / `.bias` | [d, d] / [d] |
| `scorer.3.weight` / `.bias` | [1, d] / [1] |
| `act_head.0.weight` / `.bias` | [256, d+4] / [256] |
| `act_head.2.weight` / `.bias` | [n_act, 256] / [n_act] |
| `temperature` | [3] |

Hypothesis check against the h64 inventory (d=64, i=128, V=231, L=3, H=1, n_act=2): 44 keys and 210822 parameters; h128 (`act_head.0` is [256,132]) confirms `d+4`. The table was cross-checked against the h64 inventory with a throwaway script (44 keys, every shape equal); the Rust coverage test must still verify it. Note `Wi` is [2i, d] = [256, 64] in the inventory and `Wo` is [d, i] = [64, 128].
- Configs with `attention_bias`, `mlp_bias` or `norm_bias` true would add bias keys absent from the inventory: return `Unimplemented`. Implementation reuses whatever `ModelShape::ensure_supported` already rejects; extend only if it does not cover them.
- `n_act` comes from `raya-core` as is; a disagreement with the file surfaces as a shape error naming `act_head.2.weight`.

## R5. Typed keys and the table
- **Decision**: `ParamKey` enum (kind plus layer index), `Entry { key, laya_name, shape }`, table built by a function of `&ModelConfig`. Lookups are by `ParamKey`; Laya strings appear only in `mapping.rs`.
- **Alternatives**: string-keyed `HashMap` (typos would fail late, in 006); proc-macro (over-engineered).

## R9. Widening half precision (amendment 2026-10-04)
- **Decision**: F16 and BF16 are widened by hand-written conversions in `raya-weights/src/convert.rs` (bf16: `bits << 16`; fp16: binary16 to binary32 with subnormal normalization via `ilog2`, NaN payload shifted by 13). No `half` dependency.
- **Verification**: exhaustive over all 65536 fp16 patterns against an f64 computation from the format definition (value, sign of zero, NaN payload), all bf16 patterns, special values; integration tests narrow the vendored h64 weights in a temp dir with independent round-to-nearest-even code and compare the loaded tensors bit for bit with the independently widened values.
- **Memory**: all-F32 files keep the zero-copy index into the file buffer. Otherwise every tensor is copied or widened into one new `Vec<f32>` (wrapped as the store buffer) and the file buffer is dropped, so the store keeps one buffer either way.
- **Alternatives**: `half` crate (new dependency for ~20 lines); converting lazily per access (every consumer would pay it and `Tensor` would stop being a plain `&[f32]`); per-tensor `Vec<f32>` (more allocations, no benefit).

## R6. Error shape
- **Decision**: one new `Error::Weights(LoadReport)` in `raya-core` with `path`, `missing`, `unused`, `shape` (name, expected, found), listing every offender, multi-line `Display`, sorted lists. Open/read failure reuses `Error::Io{path}`; header parse failure is reported through the same `Weights` variant with the safetensors message and the path. A non-F32 dtype returns `Error::Unimplemented` naming every offending tensor and dtype (spec US1 scenario 4); dtype checks run before the missing/unused/shape comparison.
- **Rationale**: 001 left the enum open for later features; one variant keeps downstream `match` sites small.

## R7. `raya info`
- Takes a checkpoint directory, uses `load_dir`, prints plain `key: value` lines (see contracts/info-cli.md). Failures go to stderr with exit 1. The temp checkpoint directory in the test is assembled by copying `fixture-tiny/` configs and `weights-h64.safetensors` as `model.safetensors`; no vendored file is touched.

## R8. Dev dependencies
- `tempfile` for temp dirs. Mutated safetensors files are produced with the `safetensors` crate's serializer (already a dependency).
