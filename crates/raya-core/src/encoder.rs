//! `ModernBERT` encoder (transformers 5.18 `ModernBertModel`): embeddings, pre-norm layers, final
//! norm. Eval mode, so dropout is the identity.
use raya_kernels::{AttentionKernel, AttnParams, AttnShape, Mask};

use crate::error::{Error, Result};
use crate::model::{EncoderLayer, Network};
use crate::ops::{gelu, layer_norm, rope_in_place};

#[allow(clippy::cast_possible_truncation)]
pub(crate) fn eps(net: &Network<'_>) -> f32 {
    net.config.encoder.norm_eps as f32
}

/// `tok_embeddings` lookup followed by the embedding `LayerNorm`. Ids are checked by the caller.
pub(crate) fn embed(net: &Network<'_>, ids: &[u32]) -> Vec<f32> {
    let d = net.config.encoder.hidden_size;
    let mut raw = Vec::with_capacity(ids.len() * d);
    for &id in ids {
        let row = id as usize * d;
        raw.extend_from_slice(&net.tok_emb[row..row + d]);
    }
    let mut out = vec![0.0f32; raw.len()];
    layer_norm(
        &mut out,
        &raw,
        net.embed_norm.w,
        net.embed_norm.b,
        eps(net),
        d,
    );
    out
}

/// Splits a packed `[seq, 3 * d]` projection (`[seq, 3, heads, head_dim]`) into head-major
/// `[heads][seq][head_dim]` q, k, v.
pub(crate) fn split_qkv(qkv: &[f32], seq: usize, heads: usize, head_dim: usize) -> [Vec<f32>; 3] {
    let d = heads * head_dim;
    let mut parts = [
        vec![0.0f32; seq * d],
        vec![0.0f32; seq * d],
        vec![0.0f32; seq * d],
    ];
    for t in 0..seq {
        for (which, part) in parts.iter_mut().enumerate() {
            for h in 0..heads {
                let src = t * 3 * d + which * d + h * head_dim;
                let dst = (h * seq + t) * head_dim;
                part[dst..dst + head_dim].copy_from_slice(&qkv[src..src + head_dim]);
            }
        }
    }
    parts
}

#[allow(clippy::too_many_arguments)]
/// Runs the kernel and returns the context as `[seq, heads * head_dim]`.
pub(crate) fn attend(
    kernel: &dyn AttentionKernel,
    qkv: &[f32],
    seq: usize,
    heads: usize,
    head_dim: usize,
    rope_theta: Option<f32>,
    mask: Mask,
    valid_len: usize,
) -> Result<Vec<f32>> {
    let [mut q, mut k, v] = split_qkv(qkv, seq, heads, head_dim);
    if let Some(theta) = rope_theta {
        rope_in_place(&mut q, heads, seq, head_dim, theta);
        rope_in_place(&mut k, heads, seq, head_dim, theta);
    }
    let shape = AttnShape {
        heads,
        seq,
        head_dim,
    };
    let params = AttnParams::new(mask, valid_len, head_dim);
    let mut out = vec![0.0f32; shape.numel()];
    kernel
        .attend(shape, params, &q, &k, &v, &mut out)
        .map_err(|e| Error::Kernel(e.to_string()))?;
    let d = heads * head_dim;
    let mut ctx = vec![0.0f32; seq * d];
    for h in 0..heads {
        for t in 0..seq {
            let src = (h * seq + t) * head_dim;
            ctx[t * d + h * head_dim..t * d + (h + 1) * head_dim]
                .copy_from_slice(&out[src..src + head_dim]);
        }
    }
    Ok(ctx)
}

/// One encoder layer in place: `h += attn(norm(h))`, then `h += mlp(norm(h))`.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn layer(
    net: &Network<'_>,
    layer: &EncoderLayer<'_>,
    h: &mut [f32],
    seq: usize,
    valid_len: usize,
    kernel: &dyn AttentionKernel,
) -> Result<()> {
    let e = &net.config.encoder;
    let d = e.hidden_size;
    let (heads, eps) = (e.num_attention_heads, eps(net));
    let normed = match &layer.attn_norm {
        Some(n) => {
            let mut out = vec![0.0f32; h.len()];
            layer_norm(&mut out, h, n.w, n.b, eps, d);
            out
        }
        None => h.to_vec(),
    };
    let qkv = layer.wqkv.apply(&net.gemm, &normed, seq);
    let (theta, mask) = if layer.global {
        (e.global_rope_theta, Mask::Full)
    } else {
        (e.local_rope_theta, Mask::Sliding(e.local_attention / 2))
    };
    let ctx = attend(
        kernel,
        &qkv,
        seq,
        heads,
        d / heads,
        Some(theta as f32),
        mask,
        valid_len,
    )?;
    let attn_out = layer.wo.apply(&net.gemm, &ctx, seq);
    for (x, a) in h.iter_mut().zip(&attn_out) {
        *x += a;
    }

    let mut normed = vec![0.0f32; h.len()];
    layer_norm(&mut normed, h, layer.mlp_norm.w, layer.mlp_norm.b, eps, d);
    let wi = layer.wi.apply(&net.gemm, &normed, seq);
    let inter = e.intermediate_size;
    let mut gated = vec![0.0f32; seq * inter];
    for (t, row) in wi.chunks(2 * inter).enumerate() {
        let (input, gate) = row.split_at(inter);
        for (j, g) in gated[t * inter..(t + 1) * inter].iter_mut().enumerate() {
            *g = gelu(input[j]) * gate[j];
        }
    }
    let mlp_out = layer.mlp_wo.apply(&net.gemm, &gated, seq);
    for (x, m) in h.iter_mut().zip(&mlp_out) {
        *x += m;
    }
    Ok(())
}
