//! Loads a Laya `model.safetensors` into a validated, immutable `WeightStore` of fp32 tensors.
#![forbid(unsafe_code)]

#[cfg(target_endian = "big")]
compile_error!("raya-weights assumes little-endian");

mod buffer;
mod convert;
mod mapping;
mod store;

pub use mapping::{expected, Entry};
pub use raya_core::{HeadPart, ParamKey};
pub use store::{load, load_dir, Tensor, WeightStore};
