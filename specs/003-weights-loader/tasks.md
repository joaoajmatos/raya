# Tasks: Weights Loader

**Input**: Design documents from `/specs/003-weights-loader/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Required (constitution II and VII; spec SC-001 to SC-005).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: can run in parallel (different files, no dependency)
- **[Story]**: US1 (load exactly), US2 (fail loudly), US3 (`raya info`)

## Phase 1: Setup

- [X] T001 Add `bytemuck` to `[workspace.dependencies]` in `Cargo.toml` (latest version compatible with rustc 1.94) and `tempfile` as a workspace dev-dependency; in `crates/raya-weights/Cargo.toml` add `bytemuck`, `serde_json` (dev) and `tempfile` (dev); in `crates/raya-cli/Cargo.toml` add `raya-core`, `raya-weights` and `tempfile` (dev). No other dependency. Confirm `cargo audit` stays clean via the CI `audit` job
- [X] T002 [P] `crates/raya-weights/src/lib.rs`: replace the placeholder `SafeTensors` re-export with `mod key; mod mapping; mod buffer; mod store;`, public re-exports per `contracts/weights-api.md`, `#![forbid(unsafe_code)]` kept, and `#[cfg(target_endian = "big")] compile_error!("raya-weights assumes little-endian")` (research R3)

## Phase 2: Foundational (blocks all stories)

- [X] T003 `crates/raya-core/src/error.rs`: add `ShapeMismatch { name, expected: Vec<usize>, found: Vec<usize> }`, `LoadReport { path, missing, unused, shape, detail: Option<String> }` and `Error::Weights(LoadReport)` per `data-model.md`; `Display` prints each non-empty group on its own lines with sorted names; re-export from `lib.rs`. Unit test in the same file for the message format
- [X] T004 [P] `crates/raya-weights/src/key.rs`: `ParamKey` enum and `HeadPart` enum exactly as in `data-model.md` (`Copy + Eq + Hash + Debug`)
- [X] T005 `crates/raya-weights/src/mapping.rs`: `Entry { key, laya_name, shape }` and `expected(&ModelConfig) -> Result<Vec<Entry>>` implementing the table in `research.md` R4 (no `attn_norm` for layer 0; head entries only when `head_layers > 0`; `act_head.0` input `d + 4`; `Wi` is `[2*intermediate_size, d]`, `Wo` is `[d, intermediate_size]`; `n_act` from `AgentConfig::n_act()`). Calls `ModelShape::from_config` and `ensure_supported` first; any of `attention_bias`, `mlp_bias`, `norm_bias` true returns `Error::Unimplemented` unless `ensure_supported` already rejects it. Laya string names appear only in this file
- [X] T006 [P] `crates/raya-weights/src/buffer.rs`: `AlignedBuf` holding a `Vec<f32>` plus the real byte length; `read_file(path) -> Result<AlignedBuf>` using `File::read_exact` into `bytemuck::cast_slice_mut`; `as_bytes()` truncated to the real length; open/read errors become `Error::Io { path }`. No `unsafe`
- [X] T007 [P] Test helpers in `crates/raya-weights/tests/common/mod.rs`: build the h64 `ModelConfig` from `encoder_config`/`agent_config` in `tests/data/inventory.json` `meta` (at the time `fixture-tiny/` had `vocab_size` 100 against 231 in the weights; since the 357ad53 re-vendor `fixture-h64/` carries the matching config), the h128 config from `encoder_config`/`agent_config` in `tests/data/fastpath_off/meta.json` via `ModelConfig::from_json`, read `tests/data/inventory.json`, and write mutated copies of a fixture into a `tempfile` dir with the `safetensors` serializer (drop a key, add a key, change a shape, change a dtype). Never write under `tests/data/`

**Checkpoint**: error type, key, table and buffer exist; no story logic yet.

## Phase 3: User Story 1 - Load Laya Weights Exactly (Priority: P1) MVP

**Goal**: `load` returns a complete `WeightStore` for h64 and h128, bit-identical to the file, without per-tensor copies.

**Independent Test**: `cargo test -p raya-weights --test load_fixtures --test mapping_coverage`.

- [X] T008 [P] [US1] `crates/raya-weights/tests/mapping_coverage.rs`: for the h64 config, assert `expected()` has exactly 44 entries, Laya names unique, the set of names and every shape equal `tests/data/inventory.json` `keys`, no `AttnNorm(0)`, and that the sum of element counts is `n_parameters` (210822). Repeat for h128 against the keys and dtype of `weights-h128.safetensors`' header (SC-002)
- [X] T009 [P] [US1] `crates/raya-weights/tests/load_fixtures.rs`: load h64 and h128; assert `tensor_count() == 44`, every `ParamKey` from `expected()` resolves with the entry's shape, each slice is bit-identical (`f32::to_bits`) to the tensor read independently with `safetensors::SafeTensors`, `parameter_count()` equals the sum of numel, and slice pointers lie inside one buffer (no per-tensor copy). Test that the `temperature` and `type_emb` tensors load
- [X] T010 [US1] `crates/raya-weights/src/store.rs`: `WeightStore`, `Tensor<'_>`, `load(path, &ModelConfig)` and `load_dir(dir)` per `contracts/weights-api.md`: expected table first, `read_file`, `SafeTensors::deserialize` on `as_bytes()`, build the key index with `bytemuck::try_cast_slice::<u8, f32>` per tensor (misaligned offset is an error naming the tensor), `get`, `tensor_count`, `parameter_count`, `total_bytes`. `load_dir` reads `model.safetensors` and uses `ModelConfig::from_dir`
- [X] T011 [US1] Dtype handling in `store.rs`: any tensor whose dtype is not `F32` returns `Error::Unimplemented` naming every such tensor and dtype, before the missing/unused/shape comparison. Add the test in `load_negative.rs` (T013) for a bf16 copy of one tensor

**Checkpoint**: US1 independently green (MVP).

## Phase 4: User Story 2 - Fail Loudly on Mismatch (Priority: P1)

**Goal**: every mismatch is reported in one error that names the offenders.

**Independent Test**: `cargo test -p raya-weights --test load_negative`.

- [X] T012 [US2] `store.rs`: after dtype check, compare the file's tensor names with the table; collect missing, unused and shape mismatches (name, expected, found) and return one `Error::Weights(LoadReport)` if any; `__metadata__` ignored; header parse failure from `safetensors` becomes `Error::Weights` with `detail` and `path`; zero-length tensors are errors
- [X] T013 [US2] `crates/raya-weights/tests/load_negative.rs` using T007 helpers: missing key lists it under `missing`; extra key under `unused`; both at once lists both; wrong shape names the tensor with expected and found; bf16 tensor gives `Unimplemented` (T011); nonexistent path gives `Error::Io` with the path; truncated file and garbage header give `Error::Weights` with the path; a config that `mapping::expected` rejects (bias flags, unsupported shape; `ensure_supported` itself does not check biases) returns its error without the file being opened (point the path at a nonexistent file and expect the config error, not `Io`); a duplicate header key is reported under `duplicate`; the h128 config (`fixture-h128`) against `weights-h64` is a shape error on the embedding table ([231, 128] expected, [231, 64] found; originally the `fixture-tiny` config, vocab 100, against 231); the vendored fixtures are byte-unchanged after the tests (hash before and after)

**Checkpoint**: US1 and US2 green.

## Phase 5: User Story 3 - Inspect a Checkpoint (Priority: P2)

**Goal**: `raya info <dir>` prints computed numbers.

**Independent Test**: `cargo test -p raya-cli --test info`.

- [X] T014 [US3] `crates/raya-cli/src/main.rs`: change `Info` to `Info { dir: PathBuf }`; call `raya_weights::load_dir`, print the lines of `contracts/info-cli.md` in that order; on error print the error to stderr and return `ExitCode::from(1)`; `Predict` and `Bench` stay stubs. Keep formatting in a small `info.rs` function returning a `String` so it can be unit tested
- [X] T015 [US3] `crates/raya-cli/tests/info.rs`: assemble a checkpoint dir in a `tempfile` dir (write `encoder/config.json` and `rl_agent_config.json` from `tests/data/inventory.json` `meta`, copy `weights-h64.safetensors` as `model.safetensors`); run the built binary (`env!("CARGO_BIN_EXE_raya")`); assert exit 0 and `parameters: 210822`, `tensors: 44`, and `bytes` equal to `210822 * 4`; a dir missing `model.safetensors` exits 1 with the path on stderr (SC-003)

## Phase 6: Polish & Cross-Cutting

- [X] T016 Run `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`; confirm `git diff --stat main -- tests/data reference` is empty (SC-005); record the output in the PR description
- [X] T017 Update `specs/003-weights-loader/spec.md` Status to Implemented only after T016 passes; update `specs/README.md` row for 003 if it tracks status; mention in `README.md`/`AGENTS.md` only if the commands list changes
- [X] T018 Report to the orchestrator for the TODO note: mmap deferred (needs `unsafe` home), bf16/fp16 conversion left to the human, packing hook dropped

## Dependencies & Execution Order

- Phase 1 then Phase 2. T004, T006, T007 are parallel with each other; T005 needs T004; T003 is independent of T004 to T007.
- US1 (Phase 3) needs Phase 2. T008 and T009 need only T005/T007 and T010 respectively (write them first; they fail until T010).
- US2 (Phase 4) extends `store.rs` from US1 (T012 after T010/T011).
- US3 (Phase 5) needs US1 only (`load_dir`); it can run in parallel with US2 once T010 is merged, since it touches only `raya-cli`.
- Polish last.

## Parallel Example

After T003: T004, T006 and T007 in parallel; after T005 and T010: T012 (store.rs) and T014 (raya-cli) in parallel.

## Implementation Strategy

1. MVP = Phases 1 to 3 (load h64 and h128 exactly).
2. Add loud failure (US2) before any other crate depends on the store: feature 006 must never see a partially valid store.
3. `raya info` last; it is a thin consumer.

## Phase 7: Follow-up - widen half precision on load (amendment 2026-10-04)

- [X] T019 `crates/raya-weights/src/convert.rs`: `bf16_to_f32` and `f16_to_f32` (exact; subnormals, signed zero, infinities, NaN payloads) with in-file unit tests exhaustive over all 65536 fp16 patterns against an f64 computation from the format definition, all bf16 patterns, and special values
- [X] T020 `crates/raya-weights/src/store.rs` and `buffer.rs`: accept F32, F16 and BF16 (anything else returns `Error::Unimplemented` naming every tensor and dtype); all-F32 files keep the zero-copy path; files with any F16/BF16 tensor are widened into one new fp32 buffer and the file buffer is dropped; `WeightStore::source_dtypes()`; missing/unused/duplicate/shape reporting unchanged
- [X] T021 [P] `crates/raya-weights/tests/common/half.rs` and `tests/load_half.rs`: independent narrowing (round to nearest even) and reference widening; bf16, fp16 and mixed copies of h64 generated in a temp dir load bit-exactly to the widened originals (44 tensors, 210822 parameters, `total_bytes` = 4 x parameters); rounding-error bounds; unsupported dtypes (F64, I32) named; half-precision files still get the full mismatch report; all-F32 file reports `["F32"]`. Never write under `tests/data/`
- [X] T022 `crates/raya-weights/tests/load_negative.rs`: the former bf16 `Unimplemented` case now uses F64
- [X] T023 `crates/raya-cli/src/info.rs`, `tests/info.rs`, `tests/info_half.rs`: `source_dtype` line (last), `bytes` is the resident fp32 size; a bf16 checkpoint generated in a temp dir prints `source_dtype: BF16`
- [X] T024 Update spec, plan, research R9, contracts, quickstart for the amendment; Status stays Implemented
- [X] T025 Run `cargo build --workspace`, `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace` (VS env, `CC=cl`); `git diff --stat main -- tests/data reference` empty

## Notes

- Risk: the R4 shape table is verified by a throwaway script only; T008 is the real gate, and h128 is the only test of the `d + 4` rule at a second width.
- Risk: `ModelShape::ensure_supported` may not reject bias-enabled configs; T005 covers that case explicitly.
- Risk: `bytemuck` alignment assumption rests on safetensors' 8-byte header padding; T010 checks alignment per tensor instead of assuming it.
- Risk: adding a variant to `raya_core::Error` is a public-API change for the other crates; no `match` on `Error` exists outside `raya-core` today (verify at T003).
- Human-owned, not decided: target CPU and any weight packing. (bf16/fp16 widening on load was decided 2026-10-04, Phase 7.)
