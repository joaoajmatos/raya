# Quickstart: validating the reference forward pass

Windows: use `RUSTUP_TOOLCHAIN=1.94.1`, call `%USERPROFILE%\.cargo\bin\cargo.exe`, and unset `CC` (see the role notes). Everything below is offline.

1. Unit and kernel tests: `cargo test -p raya-core -p raya-kernels`. Covers `erf`/GELU values, LayerNorm, softmax, RoPE, a tiny deterministic network (shape, trace, padding, error and single-marker checks), calibration buckets and clamping, the single-marker case, the stub attention kernel (US3), and the extended masks against the oracle.
2. Golden parity: `cargo test -p raya-cli --test forward_golden -- --nocapture`. For each of `short`, `medium`, `padded`, `fastpath_off` it loads the vendored weights with the config from the case's `config_dir`, runs the forward with tracing and prints a table of the maximum absolute difference per tensor (valid rows). It fails at the first tensor above tolerance, naming case, tensor and layer index. Measured gaps are in the table below.
3. End to end: `cargo test -p raya-cli --test predict` builds a temporary model directory from the vendored fixture (`fixture-h64` config, `weights-h64.safetensors` as `model.safetensors`, `tokenizer/tokenizer.json`), runs `raya predict` with the question and state of the `short`, `medium` and `fastpath_off` cases, and compares the printed probabilities with the golden within 1e-5.
4. Whole workspace: `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`.

## Tolerances

The spec starts at 1e-4 (layer outputs, logits) and 1e-5 (probabilities). The first run measured the gaps below (Windows `x86_64`, rustc 1.94.1, portable scalar loops, laya-sparse `357ad53` goldens), so `forward_golden.rs` asserts 1e-5 and 1e-6 (14x and 33x above the worst measurement).

| Case | Valid rows | Worst intermediate (tensor) | Max abs diff | Worst probability diff |
|---|---|---|---|---|
| `short` | 62 | `encoder_layer_00` .. `pooled` | 4.8e-7 | 0 |
| `medium` | 225 | `encoder_final_norm` | 7.2e-7 | 0 |
| `padded` | 82 of 160 | `encoder_layer_00` .. `pooled` | 4.8e-7 | 3.0e-8 |
| `fastpath_off` | 128 | `head_layer_00` | 7.2e-7 | 3.0e-8 |

Logits differ by at most 1.1e-8 and `act_logits` by 1.5e-8 in every case. These are single runs on one machine; nothing here is a performance statement.
