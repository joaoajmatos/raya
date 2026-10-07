# Quickstart: validate the workspace

Prerequisites: Rust 1.94.1 (installed automatically by `rust-toolchain.toml` through rustup).

```bash
cargo build --workspace
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Expected: all four succeed.

**Config parsing** (spec User Story 2): `cargo test -p raya-core` runs the fixture-config tests against
`tests/data/fixture-h64/` and `tests/data/fixture-h128/` (vendored from laya-sparse; originally a hand-made `fixture-tiny/`). The h64 fixture is a real `ModernBertConfig` (3 layers, hidden 64, global layers 0 and 2,
window 16) plus a Laya agent config.

**CLI stubs** (User Story 1, scenario 4):

```bash
cargo run -q -p raya-cli -- info; echo $?     # prints "not implemented (feature 003)" on stderr, exit 2
```

**Unsafe rule** (scenario 3): `cargo test -p raya-core --test forbid_unsafe`. To see the compiler enforce it, add
`unsafe {}` to any non-kernel crate and run `cargo build`: it fails.

**CI** (scenario 2): open a PR and confirm the `fmt`, `clippy`, `test (ubuntu-latest)`, `test (macos-14)` and
`audit` checks run.
