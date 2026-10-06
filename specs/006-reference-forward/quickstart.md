# Quickstart: validating the reference forward pass

Windows: use `RUSTUP_TOOLCHAIN=1.94.1`, call `%USERPROFILE%\.cargo\bin\cargo.exe`, and unset `CC` (see the role notes). Everything below is offline.

1. Unit and kernel tests: `cargo test -p raya-core -p raya-kernels`. Covers `erf`/GELU values, LayerNorm, softmax, RoPE, a hand-computed tiny network, calibration buckets and clamping, the single-marker case, the stub attention kernel (US3), and the extended masks against the oracle.
2. Golden parity: `cargo test -p raya-cli --test forward_golden -- --nocapture`. For each of `short`, `medium`, `padded`, `fastpath_off` it loads the vendored weights with the config from the case's `config_dir`, runs the forward with tracing and prints a table of the maximum absolute difference per tensor (valid rows). It fails at the first tensor above tolerance, naming case, tensor and layer index. Measured gaps are copied into the table below after the first green run.
3. End to end: `cargo test -p raya-cli --test predict` builds a temporary model directory from the vendored fixture (`fixture-h64` config, `weights-h64.safetensors` as `model.safetensors`, `tokenizer/tokenizer.json`), runs `raya predict` with the `short` case's question and state, and compares the printed probabilities with the golden within 1e-5.
4. Whole workspace: `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`.

## Tolerances

Layer outputs and logits 1e-4 absolute, probabilities 1e-5 (spec). Measured maxima: recorded below once measured.

| Case | Worst tensor | Max abs diff |
|---|---|---|
| (filled in by the first run) | | |
