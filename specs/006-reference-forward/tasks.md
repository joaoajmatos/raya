# Tasks: Reference Forward Pass (Correctness Only)

**Input**: Design documents from `/specs/006-reference-forward/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/forward-api.md, quickstart.md

**Tests**: Required (constitution II; spec SC-001 to SC-003).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: can run in parallel (different files, no dependency)
- **[Story]**: US1 (layer-by-layer parity), US2 (`raya predict`), US3 (pluggable attention and runtime doc)
- Windows: `RUSTUP_TOOLCHAIN=1.94.1`, call `%USERPROFILE%\.cargo\bin\cargo.exe`, unset `CC`.

## Phase 1: Setup

- [X] T001 `crates/raya-core/Cargo.toml`: add `raya-kernels` (workspace path dependency); `crates/raya-cli/Cargo.toml`: add `raya-tokenizer` and `serde_json` as dependencies, `raya-kernels` as a dependency; no new crate may appear in `Cargo.lock` (check the diff).
- [X] T002 Move `ParamKey` and `HeadPart` from `crates/raya-weights/src/key.rs` into `crates/raya-core/src/params.rs` (same derives), re-export them from `raya-weights` (`pub use raya_core::{HeadPart, ParamKey}`) so every existing path keeps working; add the `ParamSource` trait (`fn param(&self, key: ParamKey) -> Option<(&[f32], &[usize])>`) and implement it for `WeightStore` in `crates/raya-weights/src/store.rs`. Existing weights tests must pass unchanged.

## Phase 2: Foundational (blocks all stories)

- [X] T003 `crates/raya-kernels/src/attention.rs`: add `Mask::Full` (all keys `< valid_len`) and `Mask::Sliding(usize)` (`|i - j| <= half_window`), extending `name`, `key_range` and `allows`; `crates/raya-kernels/src/tiled/mod.rs`: return `KernelError::Unsupported` for the two new variants; update the exhaustive match in `crates/raya-kernels/tests/common/mod.rs`. 007 tests must pass unchanged.
- [X] T004 [P] `crates/raya-kernels/tests/masks_dense.rs`: `Naive` with `Full` and `Sliding(8)` equals a straightforward dense masked softmax written in the test (including `valid_len < seq`, zero padded rows, window larger than seq), and `Tiled` returns `Unsupported` for both.
- [X] T005 `crates/raya-kernels/src/gemm.rs`: `GemmBackend::matmul_nt(&self, out: &mut [f32], a: &[f32], b: &[f32], m: usize, n: usize, k: usize)` computing `out[i][j] = sum_t a[i][t] * b[j][t]` and a `Reference` struct implementing it with plain loops (sequential f32 accumulation, no threads, no `unsafe`); re-export from `lib.rs`. Test in `crates/raya-kernels/tests/gemm_reference.rs` against hand-computed values.
- [X] T006 `crates/raya-core/src/ops.rs`: `erf` (f64: Taylor series for `|x| < 2.5`, continued-fraction `erfc` above, saturation at `|x| >= 6`), exact `gelu`, `relu`, `layer_norm(x, weight, bias: Option, eps, rows, d)` (biased variance, `(x - mean) / sqrt(var + eps)`), `softmax` (max-shifted, f32), `linear(out, x, w, bias, m, in, out_features, &dyn GemmBackend)`, `rope` (rotate-half, `inv_freq = theta^(-2i/head_dim)`). `crates/raya-core/tests/ops.rs`: `erf(0.5) = 0.5204998778130465`, `erf(1) = 0.8427007929497149`, `erf(2) = 0.9953222650189527`, `erf(3) = 0.9999779095030014` to 1e-12, oddness and saturation; GELU at 0, 1, -1 (`0.8413447460685429`, `-0.15865525393145707`); LayerNorm, softmax and RoPE (position 0 is identity) on hand-computed inputs.
- [X] T007 [P] `crates/raya-core/src/trace.rs`: `Trace` (ordered `(name, shape, data)`, `push`, `get`, `names`); `crates/raya-core/src/calibrate.rs`: `temp_bucket(qtype, k)`, `clamp_temperature` (non-finite becomes 1.0, then `[0.5, 5.0]`) and `calibrate(logits, qtype, agent)` per research R7 (bucket entry in `temperature_by_options`, else `temperature[qtype]`, clamped; `z = logits / t`; max-shifted softmax); invalid `qtype` or empty logits are errors. `crates/raya-core/tests/calibrate.rs`: bucket boundaries (k = 2, 3, 5, 6, 10, 11), clamp of 0.1006 to 0.5 and 9.0 to 5.0, NaN to 1.0, bucket override beats the qtype temperature, probabilities sum to 1.
- [X] T008 `crates/raya-core/src/model.rs`: `Model::bind(&self, &dyn ParamSource) -> Result<Network<'_>>` resolving every parameter of the config (embeddings, per-layer weights with layer 0 lacking `AttnNorm`, `Wqkv` `[3d, d]`, head layers via `HeadPart`, `TypeEmb`, `Scorer`, `ActHead`) with shape checks that name the missing or mis-shaped key; `ForwardInput`/`ForwardOutput` per data-model.md (`input_ids` ids `< vocab_size`, `valid_len` `1..=seq`, marker positions `< valid_len` where the mask is true, `qtype` `0..3`; violations are errors, a non-`LayaInterleaved` shape or non-prefix mask is `Error::Unimplemented`). Keep `Model::new` as is.

**Checkpoint**: ops, masks, GEMM, trace and calibration exist and are tested; no forward yet.

## Phase 3: User Story 1 - Layer-by-Layer Parity (Priority: P1) MVP

**Goal**: every golden tensor of the four cases matches Laya within tolerance; failures name case, tensor and layer index.

**Independent Test**: `cargo test -p raya-cli --test forward_golden -- --nocapture`.

- [X] T009 [US1] `crates/raya-core/src/encoder.rs`: embeddings (`tok_embeddings` row gather, LayerNorm with `norm_eps`, bias only if `norm_bias`), `encoder_layer(h, layer)` per plan decision 2 (layer 0 skips the attention norm; `Wqkv` viewed `[seq, 3, heads, head_dim]`; per-layer-type theta; mask `Full` for global layers and `Sliding(local_attention / 2)` otherwise; attention through `&dyn AttentionKernel` with `AttnParams { mask, block: BLOCK, valid_len, scale: head_dim^-0.5 }`; `Wo`; residual; MLP `Wi` split `input`/`gate`, `gelu(input) * gate`, `Wo`, residual), final norm. Heads need `head_dim % 8 == 0`; a kernel error becomes an error with the kernel's message.
- [X] T010 [US1] `crates/raya-core/src/head.rs`: head layer per plan decision 3 (norm1, packed `in_proj` split into q, k, v, `heads = max(1, d / 64)`, scale `head_dim^-0.5`, mask `Full` with `valid_len`, `out_proj`, residual; norm2, `linear1`, `relu`, `linear2`, residual; eps 1e-5), marker gather (`marker_pos` clamped at 0), scorer (`LayerNorm(d)` eps 1e-5, `Linear`, exact GELU, `Linear(d, 1)`), `-1e4` for invalid markers, softmax, features `[top1, top1 - top2, entropy / ln(max(k, 2)), k / 255]` with `entropy = -sum(p ln max(p, 1e-9))` over all marker slots and `k` the valid-marker count (a single slot gives top2 = 0), `pooled = h[0]`, `act_head` (`Linear(d + 4, 256)`, GELU, `Linear(256, n_act)`).
- [X] T011 [US1] `crates/raya-core/src/model.rs`: `Network::forward(&self, input, kernel, trace)` chaining embeddings, layers, final norm, `type_emb` addition, head, scoring; when `trace` is `Some`, push `embedding_output`, `encoder_layer_NN`, `encoder_final_norm`, `type_emb_add`, `head_input`, `head_layer_NN`, `marker_states`, `logits`, `probabilities`, `act_features`, `pooled`, `act_logits` with golden shapes (`[1, seq, d]`, `[1, 1, d]`, `[1, k]`, `[1, 4]`, ...). Export the new items from `crates/raya-core/src/lib.rs`.
- [X] T012 [P] [US1] `crates/raya-core/tests/forward_tiny.rs`: a tiny network built in the test (hidden 16, 2 heads of 8, 2 encoder layers (one global, one local), 1 head layer, a local `ParamSource` over a `HashMap`, weights from a fixed integer pattern) with structural checks (finite outputs, trace names and shapes, features follow Laya's formula, padding after `valid_len` does not change valid rows, tracing does not change numbers; numerical parity itself is T013's job); plus negative cases: id out of vocabulary, `valid_len` 0 or above `seq`, marker out of range, missing key names the key, non-prefix mask, `qtype` 3. Single-marker case: `features` is `[1.0, 1.0, 0.0, 2/255]` (`k` is clamped to at least 2, as in Laya) and `probabilities` is `[1.0]` (Laya's documented behavior, the golden fixtures have no one-option case).
- [X] T013 [US1] `crates/raya-cli/tests/common/mod.rs` (new) and `crates/raya-cli/tests/forward_golden.rs`: load `tests/data/<case>/tensors.safetensors` (`safetensors`, int64 and bool tensors included) and `meta.json` (`config_dir`, `weights_file`, `input`), build the model with `ModelConfig::from_dir(case dir / config_dir)` and `raya_weights::load`, run `Network::forward` with `Naive` and a `Trace`, compare each trace tensor on rows `0..valid_len` (`unpadded_length` from `meta.json`; non-sequence tensors in full): max absolute difference printed in a table; assert `<= 1e-4` for intermediates and `logits`, `act_logits`, `pooled`, `marker_states`, and `<= 1e-5` for `probabilities` and `act_features`; failure messages name case, tensor and, for `encoder_layer_NN` and `head_layer_NN`, the first divergent layer index (stop at the first). Cases: `short`, `medium`, `padded`, `fastpath_off` (hidden 128, 2 head heads). Also assert the padded case's valid rows equal the unpadded-length forward of the same ids (padding does not change valid-token outputs).
- [X] T014 [US1] Run the golden suite, copy the measured maxima into the table in `specs/006-reference-forward/quickstart.md`, and tighten a tolerance in T013 only if the measured error is at least 10x below the bound (cite the measurement in the test comment). Report any gap above tolerance honestly instead of loosening the test; investigate against Laya's source first.

**Checkpoint**: US1 complete; `cargo test --workspace` green.

## Phase 4: User Story 2 - Runnable Prediction (Priority: P2)

**Goal**: `raya predict` prints calibrated probabilities equal to Laya's.

**Independent Test**: `cargo test -p raya-cli --test predict`.

- [X] T015 [US2] `crates/raya-cli/src/predict.rs` and `crates/raya-cli/src/main.rs`: `Command::Predict { model, document, question, tokenizer }` per `contracts/forward-api.md`: `load_dir(model)`, `SequenceBuilder::from_file(tokenizer or model/tokenizer.json)`, parse the question (inline JSON if it starts with `{`, else read as a path) with `QuestionDef::from_json`, state = document text (`State` string form), `build_sequence` with the agent config's `max_len` and `head_max_len`, one forward (`Naive`), `calibrate`, print the JSON object. Every failure goes to stderr with exit code 1; no panics. Replace the "not implemented (feature 006)" stub.
- [X] T016 [US2] `crates/raya-cli/tests/predict.rs`: assemble a temporary model directory from the vendored fixture (`fixture-h64` configs, `weights-h64.safetensors` as `model.safetensors`, `tokenizer/tokenizer.json`), write the `short` case's question JSON (`meta.json` `input.question`) and state to files, run the binary (`env!("CARGO_BIN_EXE_raya")`), parse stdout and compare `probabilities` with the golden `probabilities` within 1e-5 (SC-002), also check `labels` order and `act_probability` equals `softmax(act_logits)[0]`; negative cases: missing document file, malformed question JSON, missing tokenizer, all exit 1 with a message and no panic.

**Checkpoint**: US1 and US2 independently green.

## Phase 5: User Story 3 - Pluggable Attention and Documented Runtime Choice (Priority: P3)

**Goal**: a second attention implementation runs unchanged; the runtime choice is documented.

**Independent Test**: `cargo test -p raya-core --test stub_kernel`; read `docs/tensor-runtime.md`.

- [X] T017 [P] [US3] `crates/raya-core/tests/stub_kernel.rs`: a counting wrapper around `Naive` implementing `AttentionKernel` is passed to `Network::forward` on the tiny network of T012 without any model change; assert identical outputs, that it was called once per encoder layer plus once per head layer with the expected `Mask` values (global layers `Full`, local layers `Sliding(local_attention / 2)`), and that a kernel returning `KernelError::Unsupported` surfaces as an error.
- [X] T018 [P] [US3] `docs/tensor-runtime.md`: compare (a) `candle`, (b) `ndarray` plus a BLAS binding, (c) plain Rust loops on Windows/macOS/Linux build simplicity, large-GEMM performance and ability to call our own attention kernel; recommend (c) for 006 per research R10; label every throughput statement a hypothesis (no run exists); state that the choice is revisited with feature 008 numbers on the target CPU and that precision and target CPU remain human decisions.

## Phase 6: Polish & Cross-Cutting

- [X] T019 [P] Update `README.md` / `docs` mentions and the 006 `spec.md` status to `Implemented` (with the PR), `specs/README.md` row if it tracks status, and make `raya --help` text match (`Predict` is no longer "feature 006"); confirm `raya predict` with missing arguments exits with clap's usage error.
- [X] T020 Constitution gates: `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`; confirm `git status` shows no change under `reference/` or `tests/data/`, `Cargo.lock` gains no crate, and `unsafe` appears nowhere outside `raya-kernels`.
- [X] T021 PR description: what is done and not done, the measured gaps table, divergences from the spec (none intended), Laya observations (research R4, R6, R7), the laya-sparse SHA used (`357ad53`), and that no performance or hardware claim is made.

## Dependencies

- T001, T002 first. T003 to T008 depend on them; T004 needs T003, T008 needs T002 and T007's trace type only for `forward` (T011).
- US1: T009, T010 need T003, T005, T006, T008; T011 needs T009, T010, T007; T012 needs T011; T013 needs T011; T014 needs T013.
- US2 needs US1 (T015 needs T011 and T007); T016 needs T015.
- US3: T017 needs T011/T012; T018 is independent.
- Polish after the stories.

## Parallel opportunities

T004 with T005 and T006; T007 with T006; T012 with T013; T017 with T018 and with US2.

## Implementation strategy

MVP is US1 (parity). Build bottom-up (ops, masks, GEMM, then encoder, head, forward), prove each layer against the goldens in order, then add the CLI and the doc.
