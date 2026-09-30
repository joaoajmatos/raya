use serde::Deserialize;

use super::fields::{check_types, Kind};
use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Gelu,
}

/// The architecture half of a Laya checkpoint: a Hugging Face `ModernBertConfig`.
#[derive(Debug, Clone, PartialEq)]
pub struct EncoderConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub global_attn_every_n_layers: usize,
    /// Total window width of a local layer; each token sees `local_attention / 2` per side.
    pub local_attention: usize,
    pub global_rope_theta: f64,
    pub local_rope_theta: f64,
    pub norm_eps: f64,
    pub norm_bias: bool,
    pub attention_bias: bool,
    pub mlp_bias: bool,
    pub hidden_activation: Activation,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub pad_token_id: Option<u32>,
    pub cls_token_id: Option<u32>,
    pub sep_token_id: Option<u32>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
}

impl EncoderConfig {
    #[must_use]
    pub fn is_global_layer(&self, layer: usize) -> bool {
        layer.is_multiple_of(self.global_attn_every_n_layers)
    }

    #[must_use]
    pub fn global_layer_count(&self) -> usize {
        (0..self.num_hidden_layers)
            .filter(|&i| self.is_global_layer(i))
            .count()
    }

    pub(crate) fn from_value(value: serde_json::Value) -> Result<Self> {
        use Kind::{Bool, Float, Str, Uint};
        check_types(
            &value,
            &[
                ("hidden_size", Uint),
                ("intermediate_size", Uint),
                ("num_hidden_layers", Uint),
                ("num_attention_heads", Uint),
                ("global_attn_every_n_layers", Uint),
                ("local_attention", Uint),
                ("max_position_embeddings", Uint),
                ("vocab_size", Uint),
                ("norm_eps", Float),
                ("global_rope_theta", Float),
                ("local_rope_theta", Float),
                ("norm_bias", Bool),
                ("attention_bias", Bool),
                ("mlp_bias", Bool),
                ("hidden_activation", Str),
                ("pad_token_id", Uint),
                ("cls_token_id", Uint),
                ("sep_token_id", Uint),
                ("bos_token_id", Uint),
                ("eos_token_id", Uint),
            ],
        )?;
        let raw: Raw = serde_json::from_value(value).map_err(|e| {
            // serde names the field in its message, e.g. "missing field `hidden_size`".
            let msg = e.to_string();
            let field = msg.split('`').nth(1).unwrap_or("encoder/config.json");
            Error::config(field, msg.clone())
        })?;
        let (global_rope_theta, local_rope_theta) = raw.rope_thetas()?;
        let hidden_activation = match raw.hidden_activation.as_str() {
            "gelu" => Activation::Gelu,
            other => {
                return Err(Error::Unimplemented(format!(
                    "hidden_activation `{other}` (only `gelu` is supported)"
                )))
            }
        };
        let cfg = Self {
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            global_attn_every_n_layers: raw.global_attn_every_n_layers,
            local_attention: raw.local_attention,
            global_rope_theta,
            local_rope_theta,
            norm_eps: raw.norm_eps,
            norm_bias: raw.norm_bias,
            attention_bias: raw.attention_bias,
            mlp_bias: raw.mlp_bias,
            hidden_activation,
            max_position_embeddings: raw.max_position_embeddings,
            vocab_size: raw.vocab_size,
            pad_token_id: raw.pad_token_id,
            cls_token_id: raw.cls_token_id,
            sep_token_id: raw.sep_token_id,
            bos_token_id: raw.bos_token_id,
            eos_token_id: raw.eos_token_id,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        let positive = [
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            (
                "global_attn_every_n_layers",
                self.global_attn_every_n_layers,
            ),
            ("local_attention", self.local_attention),
            ("max_position_embeddings", self.max_position_embeddings),
            ("vocab_size", self.vocab_size),
        ];
        for (field, value) in positive {
            if value == 0 {
                return Err(Error::config(field, "must be at least 1"));
            }
        }
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(Error::config(
                "hidden_size",
                format!(
                    "{} is not divisible by num_attention_heads {}",
                    self.hidden_size, self.num_attention_heads
                ),
            ));
        }
        let positive_f = [
            ("global_rope_theta", self.global_rope_theta),
            ("local_rope_theta", self.local_rope_theta),
            ("norm_eps", self.norm_eps),
        ];
        for (field, value) in positive_f {
            if !(value.is_finite() && value > 0.0) {
                return Err(Error::config(field, "must be a positive finite number"));
            }
        }
        Ok(())
    }
}

#[derive(Deserialize)]
struct Raw {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    global_attn_every_n_layers: usize,
    local_attention: usize,
    norm_eps: f64,
    norm_bias: bool,
    attention_bias: bool,
    mlp_bias: bool,
    hidden_activation: String,
    max_position_embeddings: usize,
    vocab_size: usize,
    global_rope_theta: Option<f64>,
    local_rope_theta: Option<f64>,
    rope_parameters: Option<RopeParameters>,
    pad_token_id: Option<u32>,
    cls_token_id: Option<u32>,
    sep_token_id: Option<u32>,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
}

#[derive(Deserialize)]
struct RopeParameters {
    rope_theta: Option<f64>,
    full_attention: Option<RopeTheta>,
    sliding_attention: Option<RopeTheta>,
}

#[derive(Deserialize)]
struct RopeTheta {
    rope_theta: Option<f64>,
}

impl Raw {
    /// Same precedence as Laya's `_apply_rope_config`: per-layer-type values from
    /// `rope_parameters` (transformers 5), else the flat 4.x keys, else the flat
    /// `rope_parameters.rope_theta`.
    fn rope_thetas(&self) -> Result<(f64, f64)> {
        let rp = self.rope_parameters.as_ref();
        let flat = rp.and_then(|r| r.rope_theta);
        let pick = |typed: Option<&RopeTheta>, legacy: Option<f64>| {
            typed.and_then(|t| t.rope_theta).or(legacy).or(flat)
        };
        let global = pick(
            rp.and_then(|r| r.full_attention.as_ref()),
            self.global_rope_theta,
        );
        let local = pick(
            rp.and_then(|r| r.sliding_attention.as_ref()),
            self.local_rope_theta,
        );
        let global = global.ok_or_else(|| Error::config("global_rope_theta", "missing"))?;
        let local = local.ok_or_else(|| Error::config("local_rope_theta", "missing"))?;
        Ok((global, local))
    }
}
