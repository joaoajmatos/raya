use std::process::Command;

fn raya(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_raya"))
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn stubs_report_not_implemented_and_exit_2() {
    let out = raya(&["bench"]);
    assert_eq!(out.status.code(), Some(2));
    assert!(out.stdout.is_empty(), "bench wrote to stdout");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr).trim(),
        "not implemented (feature 008)"
    );
}

#[test]
fn help_succeeds() {
    for args in [
        &["--help"][..],
        &["info", "--help"],
        &["predict", "--help"],
        &["bench", "--help"],
    ] {
        assert!(raya(args).status.success(), "{args:?}");
    }
}
