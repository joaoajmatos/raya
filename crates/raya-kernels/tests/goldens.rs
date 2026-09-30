use std::path::PathBuf;

/// PyTorch goldens for the `block_local` pattern come from feature 002. Until they are
/// vendored, this test reports the skip and passes.
#[test]
fn block_local_goldens_if_present() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/attention-goldens");
    if !dir.exists() {
        eprintln!("goldens absent (feature 002 not merged): skipped");
        return;
    }
    panic!("attention goldens found at {}; wire them into this test (format defined by feature 002)", dir.display());
}
