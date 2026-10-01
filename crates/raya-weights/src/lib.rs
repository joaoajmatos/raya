//! Loads a Laya `model.safetensors` into a validated, immutable `WeightStore` of fp32 tensors.
#![forbid(unsafe_code)]

#[cfg(target_endian = "big")]
compile_error!("raya-weights assumes little-endian");

mod buffer;
mod key;
mod mapping;
mod store;

pub use key::{HeadPart, ParamKey};
pub use mapping::{expected, Entry};
pub use store::{load, load_dir, Tensor, WeightStore};
