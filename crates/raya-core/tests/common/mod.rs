#![allow(dead_code, clippy::cast_precision_loss)]
use std::collections::HashMap;

use raya_core::{HeadPart, ModelConfig, ParamKey, ParamSource};

pub const D: usize = 16;
pub const VOCAB: usize = 20;

pub fn config() -> ModelConfig {
    let enc = format!(
        r#"{{"hidden_size":{D},"intermediate_size":16,"num_hidden_layers":2,
        "num_attention_heads":2,"global_attn_every_n_layers":2,"local_attention":4,
        "norm_eps":1e-5,"norm_bias":false,"attention_bias":false,"mlp_bias":false,
        "hidden_activation":"gelu","max_position_embeddings":64,"vocab_size":{VOCAB},
        "global_rope_theta":160000.0,"local_rope_theta":10000.0}}"#
    );
    let agent = r#"{"head_layers":1,"max_len":64,"head_max_len":32,
        "temperature":[1.0,1.0,1.0],"act_costs":{"abstain":0.1}}"#;
    ModelConfig::from_json(&enc, agent).unwrap()
}

pub struct Params(pub HashMap<ParamKey, (Vec<f32>, Vec<usize>)>);

impl ParamSource for Params {
    fn param(&self, key: ParamKey) -> Option<(&[f32], &[usize])> {
        self.0.get(&key).map(|(d, s)| (d.as_slice(), s.as_slice()))
    }
}

fn fill(shape: &[usize], seed: usize, norm: bool) -> (Vec<f32>, Vec<usize>) {
    let n: usize = shape.iter().product();
    let data = (0..n)
        .map(|i| {
            let x = ((i * 7919 + seed * 104_729) % 1009) as f32 / 1009.0 - 0.5;
            if norm {
                1.0 + 0.1 * x
            } else {
                0.4 * x
            }
        })
        .collect();
    (data, shape.to_vec())
}

/// Deterministic weights for every parameter of `config()` (layer 0 has no attention norm).
pub fn params() -> Params {
    let cfg = config();
    let e = &cfg.encoder;
    let (d, inter) = (e.hidden_size, e.intermediate_size);
    let mut m = HashMap::new();
    let mut seed = 0;
    let mut put = |key: ParamKey, shape: &[usize], norm: bool| {
        seed += 1;
        m.insert(key, fill(shape, seed, norm));
    };
    put(ParamKey::TokEmbeddings, &[e.vocab_size, d], false);
    put(ParamKey::EmbedNorm, &[d], true);
    for i in 0..e.num_hidden_layers {
        if i > 0 {
            put(ParamKey::AttnNorm(i), &[d], true);
        }
        put(ParamKey::Wqkv(i), &[3 * d, d], false);
        put(ParamKey::AttnWo(i), &[d, d], false);
        put(ParamKey::MlpNorm(i), &[d], true);
        put(ParamKey::MlpWi(i), &[2 * inter, d], false);
        put(ParamKey::MlpWo(i), &[d, inter], false);
    }
    put(ParamKey::FinalNorm, &[d], true);
    let k = |part| ParamKey::Head { layer: 0, part };
    put(k(HeadPart::InProjW), &[3 * d, d], false);
    put(k(HeadPart::InProjB), &[3 * d], false);
    put(k(HeadPart::OutProjW), &[d, d], false);
    put(k(HeadPart::OutProjB), &[d], false);
    put(k(HeadPart::Lin1W), &[4 * d, d], false);
    put(k(HeadPart::Lin1B), &[4 * d], false);
    put(k(HeadPart::Lin2W), &[d, 4 * d], false);
    put(k(HeadPart::Lin2B), &[d], false);
    put(k(HeadPart::Norm1W), &[d], true);
    put(k(HeadPart::Norm1B), &[d], false);
    put(k(HeadPart::Norm2W), &[d], true);
    put(k(HeadPart::Norm2B), &[d], false);
    put(ParamKey::TypeEmb, &[3, d], false);
    let sc = |idx, bias| ParamKey::Scorer { idx, bias };
    put(sc(0, false), &[d], true);
    put(sc(0, true), &[d], false);
    put(sc(1, false), &[d, d], false);
    put(sc(1, true), &[d], false);
    put(sc(3, false), &[1, d], false);
    put(sc(3, true), &[1], false);
    let act = |idx, bias| ParamKey::ActHead { idx, bias };
    put(act(0, false), &[256, d + 4], false);
    put(act(0, true), &[256], false);
    put(act(2, false), &[2, 256], false);
    put(act(2, true), &[2], false);
    Params(m)
}

pub fn ids(n: usize) -> Vec<u32> {
    (0..n)
        .map(|i| u32::try_from((i * 3 + 1) % VOCAB).unwrap())
        .collect()
}
