//! Configuration, errors and the model shell for Raya-L: Laya's shape, fp32, inference only.
#![forbid(unsafe_code)]

pub mod config;
pub mod error;
pub mod model;
pub mod shape;

pub use config::{Activation, AgentConfig, EncoderConfig, HeadConfig, ModelConfig};
pub use error::{Error, Result};
pub use model::Model;
pub use shape::{AttentionKind, Compression, ModelShape, Segment};
