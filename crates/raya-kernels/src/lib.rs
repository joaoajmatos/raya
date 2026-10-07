//! Attention and GEMM kernels. The only crate allowed to use `unsafe`.
#![allow(clippy::many_single_char_names, clippy::similar_names)]

pub mod attention;
pub mod gemm;
pub mod meta;
pub mod naive;
pub mod simd;
pub mod threads;
pub mod tiled;

pub use attention::{AttentionKernel, AttnParams, AttnShape, KernelError, Mask, BLOCK, WINDOW};
pub use gemm::{GemmBackend, Reference};
pub use naive::Naive;
pub use simd::Backend;
pub use tiled::{StageTimes, Tiled, TiledConfig};
