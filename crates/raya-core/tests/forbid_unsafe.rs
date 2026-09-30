//! Workspace lints cannot carve out a per-crate exception to `forbid`, so the rule
//! (constitution V: `unsafe` only in `raya-kernels`) is checked on the crate roots.
use std::fs;
use std::path::PathBuf;

const ATTRIBUTE: &str = "#![forbid(unsafe_code)]";

fn crate_root(name: &str) -> String {
    let crates = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let dir = crates.join(name).join("src");
    let file = if dir.join("lib.rs").exists() {
        "lib.rs"
    } else {
        "main.rs"
    };
    fs::read_to_string(dir.join(file)).unwrap()
}

#[test]
fn every_crate_but_kernels_forbids_unsafe() {
    for name in ["raya-core", "raya-weights", "raya-tokenizer", "raya-cli"] {
        assert!(
            crate_root(name).contains(ATTRIBUTE),
            "{name} must forbid unsafe code"
        );
    }
    assert!(
        !crate_root("raya-kernels").contains(ATTRIBUTE),
        "raya-kernels is the one crate allowed to use unsafe"
    );
}

#[test]
fn every_crate_in_the_workspace_is_covered() {
    let crates = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut found: Vec<String> = fs::read_dir(crates)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    found.sort();
    assert_eq!(
        found,
        [
            "raya-cli",
            "raya-core",
            "raya-kernels",
            "raya-tokenizer",
            "raya-weights"
        ]
    );
}
