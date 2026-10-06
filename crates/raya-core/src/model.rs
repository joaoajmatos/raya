use raya_kernels::{AttentionKernel, GemmBackend, Reference};

use crate::config::ModelConfig;
use crate::error::{Error, Result};
use crate::ops::{gelu, layer_norm, linear, softmax_in_place};
use crate::params::{HeadPart, ParamKey, ParamSource};
use crate::shape::ModelShape;
use crate::trace::Trace;
use crate::{encoder, head};

/// Holds a validated configuration; `bind` resolves weights into a runnable `Network`.
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

    /// Resolves every parameter the configuration implies from `params`, checking shapes.
    pub fn bind<'w>(&'w self, params: &'w dyn ParamSource) -> Result<Network<'w>> {
        Network::bind(self, params)
    }
}

pub(crate) struct Norm<'w> {
    pub w: &'w [f32],
    pub b: Option<&'w [f32]>,
}

pub(crate) struct Lin<'w> {
    pub w: &'w [f32],
    pub b: Option<&'w [f32]>,
    pub in_f: usize,
    pub out_f: usize,
}

impl Lin<'_> {
    pub fn apply(&self, gemm: &dyn GemmBackend, x: &[f32], rows: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * self.out_f];
        linear(
            gemm, &mut out, x, self.w, self.b, rows, self.in_f, self.out_f,
        );
        out
    }
}

pub(crate) struct EncoderLayer<'w> {
    pub attn_norm: Option<Norm<'w>>,
    pub wqkv: Lin<'w>,
    pub wo: Lin<'w>,
    pub mlp_norm: Norm<'w>,
    pub wi: Lin<'w>,
    pub mlp_wo: Lin<'w>,
    pub global: bool,
}

pub(crate) struct HeadLayer<'w> {
    pub norm1: Norm<'w>,
    pub in_proj: Lin<'w>,
    pub out_proj: Lin<'w>,
    pub norm2: Norm<'w>,
    pub lin1: Lin<'w>,
    pub lin2: Lin<'w>,
}

/// A model with its weights bound: the reference forward pass of `DecisionModel`.
pub struct Network<'w> {
    pub(crate) config: &'w ModelConfig,
    pub(crate) head_heads: usize,
    pub(crate) tok_emb: &'w [f32],
    pub(crate) embed_norm: Norm<'w>,
    pub(crate) layers: Vec<EncoderLayer<'w>>,
    pub(crate) final_norm: Norm<'w>,
    pub(crate) head: Vec<HeadLayer<'w>>,
    pub(crate) type_emb: &'w [f32],
    pub(crate) scorer_norm: Norm<'w>,
    pub(crate) scorer1: Lin<'w>,
    pub(crate) scorer2: Lin<'w>,
    pub(crate) act1: Lin<'w>,
    pub(crate) act2: Lin<'w>,
    pub(crate) gemm: Reference,
}

/// One sequence, as Laya's `collate_items` produces it for a batch row.
#[derive(Debug, Clone, Copy)]
pub struct ForwardInput<'a> {
    pub input_ids: &'a [u32],
    /// Positions at or beyond this are padding (right padding only).
    pub valid_len: usize,
    pub marker_pos: &'a [usize],
    pub marker_mask: &'a [bool],
    /// Question type: 0 choice, 1 score, 2 noul.
    pub qtype: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ForwardOutput {
    /// Marker logits; invalid markers hold `-1e4`.
    pub logits: Vec<f32>,
    /// `softmax(logits)` without temperature.
    pub probabilities: Vec<f32>,
    pub act_logits: Vec<f32>,
    /// `[top1, top1 - top2, normalized entropy, k / 255]`.
    pub features: [f32; 4],
    pub pooled: Vec<f32>,
}

fn fetch<'w>(p: &'w dyn ParamSource, key: ParamKey, shape: &[usize]) -> Result<&'w [f32]> {
    let (data, found) = p
        .param(key)
        .ok_or_else(|| Error::Input(format!("missing parameter {key:?}")))?;
    if found != shape {
        return Err(Error::Input(format!(
            "parameter {key:?}: expected shape {shape:?}, found {found:?}"
        )));
    }
    Ok(data)
}

fn norm(p: &dyn ParamSource, w: ParamKey, b: Option<ParamKey>, d: usize) -> Result<Norm<'_>> {
    Ok(Norm {
        w: fetch(p, w, &[d])?,
        b: b.map(|k| fetch(p, k, &[d])).transpose()?,
    })
}

fn lin(
    p: &dyn ParamSource,
    w: ParamKey,
    b: Option<ParamKey>,
    in_f: usize,
    out_f: usize,
) -> Result<Lin<'_>> {
    Ok(Lin {
        w: fetch(p, w, &[out_f, in_f])?,
        b: b.map(|k| fetch(p, k, &[out_f])).transpose()?,
        in_f,
        out_f,
    })
}

impl<'w> Network<'w> {
    fn bind(model: &'w Model, p: &'w dyn ParamSource) -> Result<Self> {
        model.shape.ensure_supported()?;
        let cfg = &model.config;
        let e = &cfg.encoder;
        let (d, inter) = (e.hidden_size, e.intermediate_size);
        let head_cfg = model.shape.head;
        let mut layers = Vec::with_capacity(e.num_hidden_layers);
        for i in 0..e.num_hidden_layers {
            layers.push(EncoderLayer {
                attn_norm: if i == 0 {
                    None
                } else {
                    Some(norm(p, ParamKey::AttnNorm(i), None, d)?)
                },
                wqkv: lin(p, ParamKey::Wqkv(i), None, d, 3 * d)?,
                wo: lin(p, ParamKey::AttnWo(i), None, d, d)?,
                mlp_norm: norm(p, ParamKey::MlpNorm(i), None, d)?,
                wi: lin(p, ParamKey::MlpWi(i), None, d, 2 * inter)?,
                mlp_wo: lin(p, ParamKey::MlpWo(i), None, inter, d)?,
                global: e.is_global_layer(i),
            });
        }
        let mut head_layers = Vec::with_capacity(head_cfg.layers);
        for layer in 0..head_cfg.layers {
            let k = |part| ParamKey::Head { layer, part };
            head_layers.push(HeadLayer {
                norm1: norm(p, k(HeadPart::Norm1W), Some(k(HeadPart::Norm1B)), d)?,
                in_proj: lin(
                    p,
                    k(HeadPart::InProjW),
                    Some(k(HeadPart::InProjB)),
                    d,
                    3 * d,
                )?,
                out_proj: lin(p, k(HeadPart::OutProjW), Some(k(HeadPart::OutProjB)), d, d)?,
                norm2: norm(p, k(HeadPart::Norm2W), Some(k(HeadPart::Norm2B)), d)?,
                lin1: lin(
                    p,
                    k(HeadPart::Lin1W),
                    Some(k(HeadPart::Lin1B)),
                    d,
                    head_cfg.ffn_size,
                )?,
                lin2: lin(
                    p,
                    k(HeadPart::Lin2W),
                    Some(k(HeadPart::Lin2B)),
                    head_cfg.ffn_size,
                    d,
                )?,
            });
        }
        let sc = |idx, bias| ParamKey::Scorer { idx, bias };
        let act = |idx, bias| ParamKey::ActHead { idx, bias };
        Ok(Self {
            config: cfg,
            head_heads: head_cfg.heads,
            tok_emb: fetch(p, ParamKey::TokEmbeddings, &[e.vocab_size, d])?,
            embed_norm: norm(p, ParamKey::EmbedNorm, None, d)?,
            layers,
            final_norm: norm(p, ParamKey::FinalNorm, None, d)?,
            head: head_layers,
            type_emb: fetch(p, ParamKey::TypeEmb, &[3, d])?,
            scorer_norm: norm(p, sc(0, false), Some(sc(0, true)), d)?,
            scorer1: lin(p, sc(1, false), Some(sc(1, true)), d, d)?,
            scorer2: lin(p, sc(3, false), Some(sc(3, true)), d, 1)?,
            act1: lin(p, act(0, false), Some(act(0, true)), d + 4, 256)?,
            act2: lin(p, act(2, false), Some(act(2, true)), 256, cfg.agent.n_act())?,
            gemm: Reference,
        })
    }

    fn check(&self, input: &ForwardInput<'_>) -> Result<()> {
        let seq = input.input_ids.len();
        let vocab = self.config.encoder.vocab_size;
        if let Some(&id) = input.input_ids.iter().find(|&&id| id as usize >= vocab) {
            return Err(Error::Input(format!(
                "token id {id} is outside the vocabulary of {vocab}"
            )));
        }
        if input.valid_len == 0 || input.valid_len > seq {
            return Err(Error::Input(format!(
                "valid_len {} is not in 1..={seq}",
                input.valid_len
            )));
        }
        if input.marker_pos.len() != input.marker_mask.len() || input.marker_pos.is_empty() {
            return Err(Error::Input(
                "marker_pos and marker_mask must be non-empty and the same length".into(),
            ));
        }
        for (&pos, &valid) in input.marker_pos.iter().zip(input.marker_mask) {
            if pos >= seq || (valid && pos >= input.valid_len) {
                return Err(Error::Input(format!(
                    "marker position {pos} is outside the sequence (valid_len {}, length {seq})",
                    input.valid_len
                )));
            }
        }
        if !input.marker_mask.iter().any(|&m| m) {
            return Err(Error::Input("no valid marker".into()));
        }
        if input.qtype > 2 {
            return Err(Error::Input(format!(
                "qtype {} is not 0, 1 or 2",
                input.qtype
            )));
        }
        Ok(())
    }

    /// Runs `DecisionModel.forward` on one sequence, recording every stage in `trace`.
    pub fn forward(
        &self,
        input: &ForwardInput<'_>,
        kernel: &dyn AttentionKernel,
        mut trace: Option<&mut Trace>,
    ) -> Result<ForwardOutput> {
        self.check(input)?;
        let d = self.config.encoder.hidden_size;
        let seq = input.input_ids.len();
        let gemm: &dyn GemmBackend = &self.gemm;
        let mut record = |name: String, shape: &[usize], data: &[f32]| {
            if let Some(t) = trace.as_deref_mut() {
                t.push(name, shape, data);
            }
        };

        let mut h = encoder::embed(self, input.input_ids);
        record("embedding_output".into(), &[1, seq, d], &h);
        for (i, layer) in self.layers.iter().enumerate() {
            encoder::layer(self, layer, &mut h, seq, input.valid_len, kernel)?;
            record(format!("encoder_layer_{i:02}"), &[1, seq, d], &h);
        }
        let mut normed = vec![0.0f32; h.len()];
        let eps = encoder::eps(self);
        layer_norm(
            &mut normed,
            &h,
            self.final_norm.w,
            self.final_norm.b,
            eps,
            d,
        );
        h = normed;
        record("encoder_final_norm".into(), &[1, seq, d], &h);
        record("encoder_output".into(), &[1, seq, d], &h);

        let add = &self.type_emb[input.qtype * d..(input.qtype + 1) * d];
        record("type_emb_add".into(), &[1, 1, d], add);
        for row in h.chunks_mut(d) {
            for (x, a) in row.iter_mut().zip(add) {
                *x += a;
            }
        }
        record("head_input".into(), &[1, seq, d], &h);
        for (i, layer) in self.head.iter().enumerate() {
            head::layer(self, layer, &mut h, seq, input.valid_len, kernel)?;
            record(format!("head_layer_{i:02}"), &[1, seq, d], &h);
        }

        let k = input.marker_pos.len();
        let mut states = Vec::with_capacity(k * d);
        for &pos in input.marker_pos {
            states.extend_from_slice(&h[pos * d..(pos + 1) * d]);
        }
        record("marker_states".into(), &[1, k, d], &states);
        let mut logits = head::score(self, &states, k);
        for (l, &valid) in logits.iter_mut().zip(input.marker_mask) {
            if !valid {
                *l = -1e4;
            }
        }
        record("logits".into(), &[1, k], &logits);
        let mut probabilities = logits.clone();
        softmax_in_place(&mut probabilities);
        record("probabilities".into(), &[1, k], &probabilities);

        let valid_markers = input.marker_mask.iter().filter(|&&m| m).count();
        let features = head::features(&probabilities, valid_markers);
        record("act_features".into(), &[1, 4], &features);
        let pooled = h[..d].to_vec();
        record("pooled".into(), &[1, d], &pooled);
        let mut act_in = pooled.clone();
        act_in.extend_from_slice(&features);
        let mut hidden = self.act1.apply(gemm, &act_in, 1);
        for v in &mut hidden {
            *v = gelu(*v);
        }
        let act_logits = self.act2.apply(gemm, &hidden, 1);
        record("act_logits".into(), &[1, act_logits.len()], &act_logits);

        Ok(ForwardOutput {
            logits,
            probabilities,
            act_logits,
            features,
            pooled,
        })
    }
}
