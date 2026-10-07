//! Decision head: `nn.TransformerEncoderLayer(norm_first=True)` layers over the full sequence,
//! the marker scorer, the score features and the `act_head` input.
use raya_kernels::{AttentionKernel, Mask};

use crate::encoder::attend;
use crate::error::Result;
use crate::model::{HeadLayer, Network};
use crate::ops::{gelu, layer_norm, relu};

/// torch's default `layer_norm_eps`, used by the head layers and the scorer's `LayerNorm`.
const HEAD_EPS: f32 = 1e-5;

/// One head layer in place: `h += out_proj(attn(norm1(h)))`, then `h += lin2(relu(lin1(norm2(h))))`.
pub(crate) fn layer(
    net: &Network<'_>,
    layer: &HeadLayer<'_>,
    h: &mut [f32],
    seq: usize,
    valid_len: usize,
    kernel: &dyn AttentionKernel,
) -> Result<()> {
    let d = net.config.encoder.hidden_size;
    let mut normed = vec![0.0f32; h.len()];
    layer_norm(&mut normed, h, layer.norm1.w, layer.norm1.b, HEAD_EPS, d);
    let qkv = layer.in_proj.apply(&net.gemm, &normed, seq);
    let ctx = attend(
        kernel,
        &qkv,
        seq,
        net.head_heads,
        d / net.head_heads,
        None,
        Mask::Full,
        valid_len,
    )?;
    let attn_out = layer.out_proj.apply(&net.gemm, &ctx, seq);
    for (x, a) in h.iter_mut().zip(&attn_out) {
        *x += a;
    }

    layer_norm(&mut normed, h, layer.norm2.w, layer.norm2.b, HEAD_EPS, d);
    let mut hidden = layer.lin1.apply(&net.gemm, &normed, seq);
    for v in &mut hidden {
        *v = relu(*v);
    }
    let ff = layer.lin2.apply(&net.gemm, &hidden, seq);
    for (x, f) in h.iter_mut().zip(&ff) {
        *x += f;
    }
    Ok(())
}

/// `scorer`: `LayerNorm -> Linear -> GELU -> Linear(d, 1)` on the `k` gathered marker rows.
pub(crate) fn score(net: &Network<'_>, states: &[f32], k: usize) -> Vec<f32> {
    let d = net.config.encoder.hidden_size;
    let mut normed = vec![0.0f32; states.len()];
    layer_norm(
        &mut normed,
        states,
        net.scorer_norm.w,
        net.scorer_norm.b,
        HEAD_EPS,
        d,
    );
    let mut hidden = net.scorer1.apply(&net.gemm, &normed, k);
    for v in &mut hidden {
        *v = gelu(*v);
    }
    net.scorer2.apply(&net.gemm, &hidden, k)
}

/// `[top1, top1 - top2, entropy / ln(k), k / 255]` with `k = max(valid markers, 2)`; a single
/// marker has no second probability, so `top2` is 0 (Laya's `DecisionModel.forward`).
#[allow(clippy::cast_precision_loss)]
pub(crate) fn features(p: &[f32], valid_markers: usize) -> [f32; 4] {
    let k = valid_markers.max(2) as f32;
    let entropy = -p.iter().map(|&x| x * x.max(1e-9).ln()).sum::<f32>() / k.ln();
    let mut top1 = f32::NEG_INFINITY;
    let mut top2 = f32::NEG_INFINITY;
    for &x in p {
        if x > top1 {
            top2 = top1;
            top1 = x;
        } else if x > top2 {
            top2 = x;
        }
    }
    if p.len() < 2 {
        top2 = 0.0;
    }
    [top1, top1 - top2, entropy, k / 255.0]
}
