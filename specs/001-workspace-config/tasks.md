# Tasks: Rust Workspace, CI and Config Types

**Input**: Design documents from `/specs/001-workspace-config/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Required (constitution, workflow: every change ships with tests).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: can run in parallel (different files, no dependency)
- **[Story]**: US1 (buildable, gated workspace), US2 (config parses into Raya types)

## Phase 1: Setup

- [X] T001 Create root `Cargo.toml` (workspace members `crates/*`, shared package metadata, `[workspace.lints]` per research R6, pinned dependency versions), `rust-toolchain.toml` (1.94.1, rustfmt, clippy), extend `.gitignore` for `Cargo.lock` policy (commit the lockfile: binary workspace)
- [X] T002 [P] Create crate skeletons `crates/raya-core`, `raya-kernels`, `raya-weights`, `raya-tokenizer`, `raya-cli` with their `Cargo.toml` (dependencies exactly per plan) and inheriting workspace lints

## Phase 2: Foundational

- [X] T003 `crates/raya-core/src/error.rs`: `Error` enum (`Config`, `Io`, `Json`, `Unimplemented`) and `Result` alias
- [X] T004 Generate fixture `tests/data/fixture-tiny/encoder/config.json` from a real `ModernBertConfig` and `tests/data/fixture-tiny/rl_agent_config.json` (done during planning; verify they are committed; later replaced by `fixture-h64/` and `fixture-h128/` vendored from laya-sparse 357ad53)

**Checkpoint**: `cargo build --workspace` succeeds with empty crates.

## Phase 3: User Story 1 - Buildable, Gated Workspace (P1)

**Goal**: clean build, enforced lint/safety rules, CI, stubbed CLI.

**Independent Test**: `cargo build/fmt/clippy/test` green; CI green on both platforms.

- [X] T005 [P] [US1] `crates/raya-kernels/src/{lib.rs,attention.rs,gemm.rs}`: empty modules and trait declarations (`AttentionKernel`, `GemmBackend`) with doc comments; no `forbid(unsafe_code)` here
- [X] T006 [P] [US1] `crates/raya-weights/src/lib.rs` and `crates/raya-tokenizer/src/lib.rs`: `forbid(unsafe_code)`, re-export of `safetensors` / `tokenizers` entry types so the dependencies are exercised
- [X] T007 [US1] `crates/raya-cli/src/main.rs`: clap subcommands `info`, `predict`, `bench` as stubs per `contracts/cli.md` (stderr message, exit 2)
- [X] T008 [P] [US1] `crates/raya-cli/tests/stubs.rs`: run the binary via `CARGO_BIN_EXE_raya`; assert stderr text, empty stdout, exit status 2 for each command, and `--help` exit 0
- [X] T009 [P] [US1] `crates/raya-core/tests/forbid_unsafe.rs`: read each crate's `src/lib.rs` or `main.rs`; assert `#![forbid(unsafe_code)]` everywhere except `raya-kernels`, and that `raya-kernels` does not claim it
- [X] T010 [US1] `.github/workflows/ci.yml`: jobs `fmt`, `clippy` (`-D warnings`), `test` (matrix ubuntu-latest, macos-14), `audit` (`rustsec/audit-check`); `Swatinem/rust-cache`; submodule checkout disabled (not needed to build)
- [X] T011 [US1] Rewrite `README.md`: what Raya is, constitution scope limits (Raya-L, fp32, inference only, no SIMD target chosen), build/test commands, submodule note

**Checkpoint**: US1 scenarios 1-4 verified locally (scenario 2 on the PR).

## Phase 4: User Story 2 - Laya Config Parses into Raya Types (P2)

**Goal**: typed, validated `ModelConfig` and `ModelShape` for Laya's shape.

**Independent Test**: `cargo test -p raya-core` against the fixture.

- [X] T012 [P] [US2] `crates/raya-core/src/config/encoder.rs`: `EncoderConfig` with serde raw struct, RoPE theta resolution (three forms), validation, `Activation`, `is_global_layer`
- [X] T013 [P] [US2] `crates/raya-core/src/config/agent.rs`: `AgentConfig` with Laya defaults and validation (`temperature` length 3)
- [X] T014 [US2] `crates/raya-core/src/config/mod.rs`: `ModelConfig`, `HeadConfig`, `from_dir`, `from_json`, head-size derivation and invariant
- [X] T015 [US2] `crates/raya-core/src/shape.rs` and `model.rs`: `Segment`, `AttentionKind` (incl. `LayaInterleaved`), `Compression`, `ModelShape::{from_config, ensure_supported}`, `Model::new`
- [X] T016 [US2] `crates/raya-core/tests/config.rs`: fixture parse asserting every field; one-segment/no-compression conversion; missing-field, wrong-type, inconsistent-field errors naming the key; `Unimplemented` for each other `AttentionKind`, for extra compression via constructed shapes, and for `hidden_activation: silu`; `rope_parameters` forms; defaults
- [X] T017 [US2] Wire `lib.rs` exports and doc comments (why, not what)

**Checkpoint**: all US2 scenarios covered by tests.

## Phase 5: Polish

- [X] T018 Run `cargo fmt`, `clippy -D warnings`, `cargo test --workspace`; run `cargo audit` locally if installable
- [X] T019 Check spec FR-007 dependency list against every `Cargo.toml`; check `quickstart.md` commands
- [X] T020 Verify by hand that adding `unsafe {}` to a forbidden crate fails the build, then revert

## Dependencies

- T001 before T002 before everything else; T003 before T012-T015.
- US1 tasks (T005-T011) and US2 tasks (T012-T017) are independent after Phase 2 and can proceed in parallel.
- T018-T020 last.

## Implementation Strategy

MVP is Phase 1-3 (workspace and gates). US2 adds config on top. Commit per phase.

## Implementation Notes

- `cargo audit` (local, advisory DB of 2026-09-30): no vulnerabilities; one allowed warning, `paste`
  (RUSTSEC-2024-0436, unmaintained), pulled in transitively by `tokenizers`.
- Added `crates/raya-core/src/config/fields.rs`: serde's type errors do not name the offending key, so keys
  are type-checked first to satisfy "the error names the field" (spec US2 scenario 2).
- CI could only be verified by running it on GitHub; see the PR.
