use std::process::Command;

fn raya(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_raya"))
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn stubs_report_not_implemented_and_exit_2() {
    for (command, feature) in [("predict", "006"), ("bench", "008")] {
        let out = raya(&[command]);
        assert_eq!(out.status.code(), Some(2), "{command}");
        assert!(out.stdout.is_empty(), "{command} wrote to stdout");
        assert_eq!(
            String::from_utf8_lossy(&out.stderr).trim(),
            format!("not implemented (feature {feature})")
        );
    }
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
