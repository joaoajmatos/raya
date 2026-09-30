use crate::config::ModelConfig;
use crate::error::Result;
use crate::shape::ModelShape;

/// Shell only: holds a validated configuration. The forward pass arrives with feature 006.
#[derive(Debug, Clone)]
pub struct Model {
    pub config: ModelConfig,
    pub shape: ModelShape,
}

impl Model {
    pub fn new(config: ModelConfig) -> Result<Self> {
        let shape = ModelShape::from_config(&config)?;
        shape.ensure_supported()?;
        Ok(Self { config, shape })
    }
}
