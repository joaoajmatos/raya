use raya_core::{Error, HeadPart, ModelConfig, ModelShape, ParamKey, Result};

/// One expected parameter: typed key, Laya's name for it, and its shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub key: ParamKey,
    pub laya_name: String,
    pub shape: Vec<usize>,
}

fn entry(key: ParamKey, name: impl Into<String>, shape: &[usize]) -> Entry {
    Entry {
        key,
        laya_name: name.into(),
        shape: shape.to_vec(),
    }
}

/// The parameter set a Laya checkpoint of this config must contain.
pub fn expected(cfg: &ModelConfig) -> Result<Vec<Entry>> {
    ModelShape::from_config(cfg)?.ensure_supported()?;
    let e = &cfg.encoder;
    for (flag, name) in [
        (e.attention_bias, "attention_bias"),
        (e.mlp_bias, "mlp_bias"),
        (e.norm_bias, "norm_bias"),
    ] {
        if flag {
            return Err(Error::Unimplemented(format!(
                "{name}=true (bias tensors are not part of Raya-L)"
            )));
        }
    }
    let mut out = encoder_entries(cfg);
    for layer in 0..cfg.agent.head_layers {
        out.extend(head_layer_entries(e.hidden_size, layer));
    }
    out.extend(output_entries(cfg));
    Ok(out)
}

fn encoder_entries(cfg: &ModelConfig) -> Vec<Entry> {
    let e = &cfg.encoder;
    let (d, i) = (e.hidden_size, e.intermediate_size);
    let mut out = vec![
        entry(
            ParamKey::TokEmbeddings,
            "encoder.embeddings.tok_embeddings.weight",
            &[e.vocab_size, d],
        ),
        entry(ParamKey::EmbedNorm, "encoder.embeddings.norm.weight", &[d]),
    ];
    for n in 0..e.num_hidden_layers {
        let p = format!("encoder.layers.{n}");
        // Layer 0 has no attn_norm tensor in the checkpoint (see the inventory).
        if n > 0 {
            out.push(entry(
                ParamKey::AttnNorm(n),
                format!("{p}.attn_norm.weight"),
                &[d],
            ));
        }
        out.extend([
            entry(
                ParamKey::Wqkv(n),
                format!("{p}.attn.Wqkv.weight"),
                &[3 * d, d],
            ),
            entry(ParamKey::AttnWo(n), format!("{p}.attn.Wo.weight"), &[d, d]),
            entry(ParamKey::MlpNorm(n), format!("{p}.mlp_norm.weight"), &[d]),
            entry(
                ParamKey::MlpWi(n),
                format!("{p}.mlp.Wi.weight"),
                &[2 * i, d],
            ),
            entry(ParamKey::MlpWo(n), format!("{p}.mlp.Wo.weight"), &[d, i]),
        ]);
    }
    out.push(entry(
        ParamKey::FinalNorm,
        "encoder.final_norm.weight",
        &[d],
    ));
    out
}

fn head_layer_entries(d: usize, layer: usize) -> Vec<Entry> {
    let p = format!("head.layers.{layer}");
    let parts: [(HeadPart, &str, Vec<usize>); 12] = [
        (
            HeadPart::InProjW,
            "self_attn.in_proj_weight",
            vec![3 * d, d],
        ),
        (HeadPart::InProjB, "self_attn.in_proj_bias", vec![3 * d]),
        (HeadPart::OutProjW, "self_attn.out_proj.weight", vec![d, d]),
        (HeadPart::OutProjB, "self_attn.out_proj.bias", vec![d]),
        (HeadPart::Lin1W, "linear1.weight", vec![4 * d, d]),
        (HeadPart::Lin1B, "linear1.bias", vec![4 * d]),
        (HeadPart::Lin2W, "linear2.weight", vec![d, 4 * d]),
        (HeadPart::Lin2B, "linear2.bias", vec![d]),
        (HeadPart::Norm1W, "norm1.weight", vec![d]),
        (HeadPart::Norm1B, "norm1.bias", vec![d]),
        (HeadPart::Norm2W, "norm2.weight", vec![d]),
        (HeadPart::Norm2B, "norm2.bias", vec![d]),
    ];
    parts
        .into_iter()
        .map(|(part, suffix, shape)| {
            entry(
                ParamKey::Head { layer, part },
                format!("{p}.{suffix}"),
                &shape,
            )
        })
        .collect()
}

/// `type_emb`, the scorer, the action head and the temperature.
fn output_entries(cfg: &ModelConfig) -> Vec<Entry> {
    let d = cfg.encoder.hidden_size;
    let n_act = cfg.agent.n_act();
    let scorer =
        |idx, bias, name: &str, shape: &[usize]| entry(ParamKey::Scorer { idx, bias }, name, shape);
    let act = |idx, bias, name: &str, shape: &[usize]| {
        entry(ParamKey::ActHead { idx, bias }, name, shape)
    };
    vec![
        entry(ParamKey::TypeEmb, "type_emb.weight", &[3, d]),
        scorer(0, false, "scorer.0.weight", &[d]),
        scorer(0, true, "scorer.0.bias", &[d]),
        scorer(1, false, "scorer.1.weight", &[d, d]),
        scorer(1, true, "scorer.1.bias", &[d]),
        scorer(3, false, "scorer.3.weight", &[1, d]),
        scorer(3, true, "scorer.3.bias", &[1]),
        // Input width is d + 4 (research R4; h128 in mapping_coverage confirms it).
        act(0, false, "act_head.0.weight", &[256, d + 4]),
        act(0, true, "act_head.0.bias", &[256]),
        act(2, false, "act_head.2.weight", &[n_act, 256]),
        act(2, true, "act_head.2.bias", &[n_act]),
        entry(ParamKey::Temperature, "temperature", &[3]),
    ]
}
