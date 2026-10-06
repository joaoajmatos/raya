#![forbid(unsafe_code)]

mod info;
mod predict;

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
    /// Run a decision: probabilities for one question about one document
    Predict {
        /// Directory with `model.safetensors`, `rl_agent_config.json`, `encoder/config.json`
        /// and `tokenizer.json`
        #[arg(long)]
        model: PathBuf,
        /// Text file whose content is the document (the state)
        #[arg(long)]
        document: PathBuf,
        /// Question definition as JSON (`{"type": "choice", "instructions": ..., "criteria": ...}`)
        /// or the path of a file holding it
        #[arg(long)]
        question: String,
        /// `tokenizer.json` to use instead of `<model>/tokenizer.json`
        #[arg(long)]
        tokenizer: Option<PathBuf>,
    },
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
        Command::Predict {
            model,
            document,
            question,
            tokenizer,
        } => {
            let args = predict::Args {
                model,
                document,
                question,
                tokenizer,
            };
            return match predict::run(&args) {
                Ok(json) => {
                    println!("{json}");
                    ExitCode::SUCCESS
                }
                Err(err) => {
                    eprintln!("error: {err:#}");
                    ExitCode::from(1)
                }
            };
        }
        Command::Bench => "008",
    };
    eprintln!("not implemented (feature {feature})");
    ExitCode::from(2)
}
