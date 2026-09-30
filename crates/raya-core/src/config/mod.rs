mod agent;
mod encoder;
mod fields;

use std::path::Path;

pub use agent::AgentConfig;
pub use encoder::{Activation, EncoderConfig};

use crate::error::{Error, Result};

/// Decision-head sizing, derived the way Laya's `DecisionModel.__init__` does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeadConfig {
    pub layers: usize,
    pub heads: usize,
    pub ffn_size: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    pub encoder: EncoderConfig,
    pub agent: AgentConfig,
}

impl ModelConfig {
    /// Reads `encoder/config.json` and `rl_agent_config.json` from a Laya checkpoint directory.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let read = |path: std::path::PathBuf| -> Result<serde_json::Value> {
            let text = std::fs::read_to_string(&path).map_err(|source| Error::Io {
                path: path.clone(),
                source,
            })?;
            serde_json::from_str(&text).map_err(|source| Error::Json { path, source })
        };
        let encoder = read(dir.join("encoder").join("config.json"))?;
        let agent = read(dir.join("rl_agent_config.json"))?;
        Self::from_values(encoder, agent)
    }

    pub fn from_json(encoder_json: &str, agent_json: &str) -> Result<Self> {
        let parse = |text: &str, name: &str| {
            serde_json::from_str(text).map_err(|source| Error::Json {
                path: name.into(),
                source,
            })
        };
        Self::from_values(
            parse(encoder_json, "encoder/config.json")?,
            parse(agent_json, "rl_agent_config.json")?,
        )
    }

    fn from_values(encoder: serde_json::Value, agent: serde_json::Value) -> Result<Self> {
        let cfg = Self {
            encoder: EncoderConfig::from_value(encoder)?,
            agent: AgentConfig::from_value(agent)?,
        };
        cfg.head()?;
        Ok(cfg)
    }

    /// `nhead = max(1, d / 64)`, feed-forward `4 * d`; the head needs `d % nhead == 0`.
    pub fn head(&self) -> Result<HeadConfig> {
        let d = self.encoder.hidden_size;
        let heads = (d / 64).max(1);
        if !d.is_multiple_of(heads) {
            return Err(Error::config(
                "hidden_size",
                format!("{d} is not divisible by the decision head's {heads} heads"),
            ));
        }
        Ok(HeadConfig {
            layers: self.agent.head_layers,
            heads,
            ffn_size: 4 * d,
        })
    }
}
