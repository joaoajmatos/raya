#![forbid(unsafe_code)]

use std::process::ExitCode;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "raya",
    version,
    about = "Rust CPU inference for Laya-style decision models"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Inspect a checkpoint (feature 003)
    Info,
    /// Run a decision (feature 006)
    Predict,
    /// Benchmark on this machine (feature 008)
    Bench,
}

fn main() -> ExitCode {
    let feature = match Cli::parse().command {
        Command::Info => "003",
        Command::Predict => "006",
        Command::Bench => "008",
    };
    eprintln!("not implemented (feature {feature})");
    ExitCode::from(2)
}
