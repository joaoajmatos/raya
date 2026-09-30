use crate::config::{HeadConfig, ModelConfig};
use crate::error::{Error, Result};

/// Attention pattern of a segment. Only `LayaInterleaved` is implemented; the rest
/// express the shapes in `docs/feasibility.md` section 4.2 and are rejected on use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionKind {
    Full,
    BlockLocal {
        w: usize,
    },
    Gas {
        s: usize,
    },
    Hybrid,
    /// Laya's native schedule: every `global_every`-th layer is global, the rest are
    /// local with a window of `window` tokens in total. Native local layers use an exact
    /// window mask, which is not the same pattern as `BlockLocal`.
    LayaInterleaved {
        global_every: usize,
        window: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segment {
    pub layers: usize,
    pub attention: AttentionKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compression {
    None,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelShape {
    pub segments: Vec<Segment>,
    pub compression: Compression,
    pub head: HeadConfig,
}

impl ModelShape {
    /// Laya's shape: one segment, no compression.
    pub fn from_config(cfg: &ModelConfig) -> Result<Self> {
        Ok(Self {
            segments: vec![Segment {
                layers: cfg.encoder.num_hidden_layers,
                attention: AttentionKind::LayaInterleaved {
                    global_every: cfg.encoder.global_attn_every_n_layers,
                    window: cfg.encoder.local_attention,
                },
            }],
            compression: Compression::None,
            head: cfg.head()?,
        })
    }

    pub fn ensure_supported(&self) -> Result<()> {
        if self.segments.len() != 1 {
            return Err(Error::Unimplemented(format!(
                "{} segments (only a single segment is supported)",
                self.segments.len()
            )));
        }
        match self.segments[0].attention {
            AttentionKind::LayaInterleaved { .. } => Ok(()),
            other => Err(Error::Unimplemented(format!("attention kind {other:?}"))),
        }
    }
}
