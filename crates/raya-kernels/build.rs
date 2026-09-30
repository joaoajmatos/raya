use std::process::Command;

fn run(cmd: &str, args: &[&str]) -> String {
    Command::new(cmd)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map_or_else(|| "unknown".to_string(), |o| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

fn main() {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    println!("cargo:rustc-env=RAYA_RUSTC_VERSION={}", run(&rustc, &["-V"]));
    println!("cargo:rustc-env=RAYA_GIT_COMMIT={}", run("git", &["rev-parse", "HEAD"]));
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=build.rs");
}
