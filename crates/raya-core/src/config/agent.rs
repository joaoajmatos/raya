use std::collections::BTreeMap;

use serde::Deserialize;

use super::fields::{check_types, Kind};
use crate::error::{Error, Result};

/// Laya's `rl_agent_config.json`. Defaults are the ones `build_model` and `Agent` apply.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct AgentConfig {
    #[serde(default)]
    pub encoder: String,
    #[serde(default = "default_head_layers")]
    pub head_layers: usize,
    #[serde(default = "default_max_len")]
    pub max_len: usize,
    #[serde(default = "default_head_max_len")]
    pub head_max_len: usize,
    #[serde(default = "default_temperature")]
    pub temperature: Vec<f32>,
    #[serde(default)]
    pub temperature_by_options: BTreeMap<String, f32>,
    #[serde(default)]
    pub act_costs: BTreeMap<String, f32>,
}

fn default_head_layers() -> usize {
    2
}
fn default_max_len() -> usize {
    512
}
fn default_head_max_len() -> usize {
    192
}
fn default_temperature() -> Vec<f32> {
    vec![1.0; 3]
}

impl AgentConfig {
    /// Number of action logits: one per cost entry plus the default action, as in `build_model`.
    #[must_use]
    pub fn n_act(&self) -> usize {
        self.act_costs.len() + 1
    }

    pub(crate) fn from_value(value: serde_json::Value) -> Result<Self> {
        check_types(
            &value,
            &[
                ("encoder", Kind::Str),
                ("head_layers", Kind::Uint),
                ("max_len", Kind::Uint),
                ("head_max_len", Kind::Uint),
                ("temperature", Kind::FloatList),
                ("temperature_by_options", Kind::FloatMap),
                ("act_costs", Kind::FloatMap),
            ],
        )?;
        let cfg: Self = serde_json::from_value(value).map_err(|e| {
            let msg = e.to_string();
            let field = msg.split('`').nth(1).unwrap_or("rl_agent_config.json");
            Error::config(field, msg.clone())
        })?;
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        for (field, value) in [
            ("max_len", self.max_len),
            ("head_max_len", self.head_max_len),
        ] {
            if value == 0 {
                return Err(Error::config(field, "must be at least 1"));
            }
        }
        if self.temperature.len() != 3 || self.temperature.iter().any(|t| !t.is_finite()) {
            return Err(Error::config(
                "temperature",
                "must be a list of 3 finite numbers",
            ));
        }
        let finite = |m: &BTreeMap<String, f32>| m.values().all(|v| v.is_finite());
        if !finite(&self.temperature_by_options) {
            return Err(Error::config(
                "temperature_by_options",
                "values must be finite",
            ));
        }
        if !finite(&self.act_costs) {
            return Err(Error::config("act_costs", "values must be finite"));
        }
        Ok(())
    }
}
