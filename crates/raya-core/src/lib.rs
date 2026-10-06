//! Configuration, errors and the model shell for Raya-L: Laya's shape, fp32, inference only.
#![forbid(unsafe_code)]

pub mod calibrate;
pub mod config;
pub mod encoder;
pub mod error;
pub mod head;
pub mod model;
pub mod ops;
pub mod params;
pub mod shape;
pub mod trace;

pub use calibrate::calibrate;
pub use config::{Activation, AgentConfig, EncoderConfig, HeadConfig, ModelConfig};
pub use error::{Error, LoadReport, Result, ShapeMismatch};
pub use model::{ForwardInput, ForwardOutput, Model, Network};
pub use params::{HeadPart, ParamKey, ParamSource};
pub use shape::{AttentionKind, Compression, ModelShape, Segment};
pub use trace::Trace;
