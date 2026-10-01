#![forbid(unsafe_code)]

mod info;

use std::path::PathBuf;
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
    /// Inspect a checkpoint directory
    Info {
        /// Directory with `model.safetensors`, `rl_agent_config.json` and `encoder/config.json`
        dir: PathBuf,
    },
    /// Run a decision (feature 006)
    Predict,
    /// Benchmark on this machine (feature 008)
    Bench,
}

fn main() -> ExitCode {
    let feature = match Cli::parse().command {
        Command::Info { dir } => {
            return match raya_weights::load_dir(&dir) {
                Ok((config, store)) => {
                    print!("{}", info::render(&dir, &config, &store));
                    ExitCode::SUCCESS
                }
                Err(err) => {
                    eprintln!("{err}");
                    ExitCode::from(1)
                }
            };
        }
        Command::Predict => "006",
        Command::Bench => "008",
    };
    eprintln!("not implemented (feature {feature})");
    ExitCode::from(2)
}
