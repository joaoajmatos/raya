# Quickstart: validating the weights loader

Offline; needs no submodule checkout (vendored fixtures only).

1. The tests assemble the h64 checkpoint dir in a temp directory: configs written from `tests/data/inventory.json` `meta` (not `fixture-tiny/`, whose `vocab_size` 100 does not match the weights) plus `weights-h64.safetensors` copied as `model.safetensors`.
2. `cargo test -p raya-weights`: mapping coverage (44 keys, none unmapped or double-mapped), h64 and h128 loads with bit equality, half-precision loads (`load_half`: bf16, fp16 and mixed copies of h64 generated in a temp dir, widened bit-exactly), negative cases (missing, extra, duplicate, wrong shape, unsupported dtype, bad path, `fixture-tiny` config rejected).
3. `cargo test -p raya-cli --test info`: `raya info` prints `parameters: 210822`.
4. Manual: `cargo run -p raya-cli -- info <dir>`; compare `parameters` with `n_parameters` in `tests/data/inventory.json`.
5. `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`; `git diff --stat -- tests/data reference` must be empty.

Failure demo: delete a key from a temp copy; the loader error lists it under `missing`.
