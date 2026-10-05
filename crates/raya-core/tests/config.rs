use std::path::PathBuf;

use raya_core::{
    Activation, AttentionKind, Compression, Error, Model, ModelConfig, ModelShape, Segment,
};
use serde_json::{json, Value};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/fixture-h64")
}

fn fixture_json() -> (Value, Value) {
    let dir = fixture_dir();
    let read = |p: PathBuf| serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    (
        read(dir.join("encoder/config.json")),
        read(dir.join("rl_agent_config.json")),
    )
}

fn parse(encoder: &Value, agent: &Value) -> Result<ModelConfig, Error> {
    ModelConfig::from_json(&encoder.to_string(), &agent.to_string())
}

fn config_error_field(result: Result<ModelConfig, Error>) -> String {
    match result {
        Err(Error::Config { field, .. }) => field,
        other => panic!("expected a config error, got {other:?}"),
    }
}

#[test]
fn fixture_parses_with_every_field() {
    let cfg = ModelConfig::from_dir(&fixture_dir()).unwrap();
    let e = &cfg.encoder;
    assert_eq!(
        (
            e.num_hidden_layers,
            e.hidden_size,
            e.intermediate_size,
            e.num_attention_heads
        ),
        (3, 64, 128, 2)
    );
    assert_eq!((e.global_attn_every_n_layers, e.local_attention), (2, 16));
    assert_eq!(
        (e.global_rope_theta, e.local_rope_theta),
        (160_000.0, 10_000.0)
    );
    assert!((e.norm_eps - 1e-5).abs() < f64::EPSILON);
    assert!(!e.norm_bias && !e.attention_bias && !e.mlp_bias);
    assert_eq!(e.hidden_activation, Activation::Gelu);
    assert_eq!((e.max_position_embeddings, e.vocab_size), (2048, 231));
    assert_eq!(
        (
            e.pad_token_id,
            e.cls_token_id,
            e.sep_token_id,
            e.bos_token_id,
            e.eos_token_id
        ),
        (Some(0), Some(2), Some(3), Some(2), Some(3))
    );
    // Layers 0 and 2 are global, layer 1 local (matches the fixture's documented layout).
    assert_eq!(
        (0..3).map(|i| e.is_global_layer(i)).collect::<Vec<_>>(),
        [true, false, true]
    );
    assert_eq!(e.global_layer_count(), 2);

    let a = &cfg.agent;
    assert_eq!((a.head_layers, a.max_len, a.head_max_len), (1, 256, 96));
    assert_eq!(a.temperature, [1.0, 1.0, 1.0]);
    assert_eq!(a.n_act(), 2);

    let head = cfg.head().unwrap();
    assert_eq!((head.layers, head.heads, head.ffn_size), (1, 1, 256));
}

#[test]
fn laya_shape_is_one_segment_without_compression() {
    let cfg = ModelConfig::from_dir(&fixture_dir()).unwrap();
    let shape = ModelShape::from_config(&cfg).unwrap();
    assert_eq!(
        shape.segments,
        [Segment {
            layers: 3,
            attention: AttentionKind::LayaInterleaved {
                global_every: 2,
                window: 16
            },
        }]
    );
    assert_eq!(shape.compression, Compression::None);
    shape.ensure_supported().unwrap();
    Model::new(cfg).unwrap();
}

#[test]
fn head_sizing_follows_laya() {
    let (mut enc, agent) = fixture_json();
    enc["hidden_size"] = json!(1024);
    enc["num_attention_heads"] = json!(16);
    let head = parse(&enc, &agent).unwrap().head().unwrap();
    assert_eq!((head.heads, head.ffn_size), (16, 4096));
}

#[test]
fn head_needing_indivisible_heads_is_rejected() {
    let (mut enc, agent) = fixture_json();
    // 200 / 64 = 3 decision-head heads, and 200 % 3 != 0.
    enc["hidden_size"] = json!(200);
    enc["num_attention_heads"] = json!(8);
    assert_eq!(config_error_field(parse(&enc, &agent)), "hidden_size");
}

#[test]
fn missing_required_field_names_the_key() {
    for key in [
        "hidden_size",
        "num_hidden_layers",
        "global_attn_every_n_layers",
        "local_attention",
        "norm_eps",
        "vocab_size",
    ] {
        let (mut enc, agent) = fixture_json();
        enc.as_object_mut().unwrap().remove(key);
        assert_eq!(config_error_field(parse(&enc, &agent)), key);
    }
}

#[test]
fn missing_rope_theta_names_the_key() {
    let (mut enc, agent) = fixture_json();
    let obj = enc.as_object_mut().unwrap();
    obj.remove("local_rope_theta");
    obj.insert(
        "rope_parameters".into(),
        json!({"full_attention": {"rope_theta": 160_000.0}}),
    );
    assert_eq!(config_error_field(parse(&enc, &agent)), "local_rope_theta");
}

#[test]
fn wrong_type_names_the_key() {
    let (mut enc, agent) = fixture_json();
    enc["hidden_size"] = json!("sixty-four");
    assert_eq!(config_error_field(parse(&enc, &agent)), "hidden_size");

    let (enc, mut agent) = fixture_json();
    agent["temperature"] = json!("hot");
    assert_eq!(config_error_field(parse(&enc, &agent)), "temperature");
}

#[test]
fn inconsistent_fields_are_rejected() {
    let cases: [(&str, Value); 4] = [
        ("num_attention_heads", json!(3)),
        ("global_attn_every_n_layers", json!(0)),
        ("local_attention", json!(0)),
        ("norm_eps", json!(0.0)),
    ];
    for (key, value) in cases {
        let (mut enc, agent) = fixture_json();
        enc[key] = value;
        let field = config_error_field(parse(&enc, &agent));
        // A heads mismatch is reported against hidden_size, the field that cannot be divided.
        let expected = if key == "num_attention_heads" {
            "hidden_size"
        } else {
            key
        };
        assert_eq!(field, expected);
    }
}

#[test]
fn bad_temperature_is_rejected() {
    let (enc, mut agent) = fixture_json();
    agent["temperature"] = json!([1.0, 1.0]);
    assert_eq!(config_error_field(parse(&enc, &agent)), "temperature");
}

#[test]
fn agent_defaults_match_laya() {
    let (enc, _) = fixture_json();
    let cfg = parse(&enc, &json!({})).unwrap();
    let a = cfg.agent;
    assert_eq!((a.head_layers, a.max_len, a.head_max_len), (2, 512, 192));
    assert_eq!(a.temperature, [1.0, 1.0, 1.0]);
    assert_eq!(a.n_act(), 1);
}

#[test]
fn unsupported_activation_is_unimplemented() {
    let (mut enc, agent) = fixture_json();
    enc["hidden_activation"] = json!("silu");
    assert!(matches!(parse(&enc, &agent), Err(Error::Unimplemented(_))));
}

#[test]
fn rope_parameters_forms() {
    let (mut enc, agent) = fixture_json();
    let obj = enc.as_object_mut().unwrap();
    obj.remove("global_rope_theta");
    obj.remove("local_rope_theta");

    // transformers 5: per-layer-type values.
    obj.insert(
        "rope_parameters".into(),
        json!({"full_attention": {"rope_theta": 123.0}, "sliding_attention": {"rope_theta": 45.0}}),
    );
    let e = parse(&enc, &agent).unwrap().encoder;
    assert_eq!((e.global_rope_theta, e.local_rope_theta), (123.0, 45.0));

    // Flat theta applies to both, as for mmBERT.
    enc.as_object_mut()
        .unwrap()
        .insert("rope_parameters".into(), json!({"rope_theta": 160_000.0}));
    let e = parse(&enc, &agent).unwrap().encoder;
    assert_eq!(
        (e.global_rope_theta, e.local_rope_theta),
        (160_000.0, 160_000.0)
    );
}

#[test]
fn other_attention_kinds_and_shapes_are_unimplemented() {
    let cfg = ModelConfig::from_dir(&fixture_dir()).unwrap();
    let base = ModelShape::from_config(&cfg).unwrap();
    for attention in [
        AttentionKind::Full,
        AttentionKind::BlockLocal { w: 128 },
        AttentionKind::Gas { s: 4 },
        AttentionKind::Hybrid,
    ] {
        let mut shape = base.clone();
        shape.segments[0].attention = attention;
        assert!(
            matches!(shape.ensure_supported(), Err(Error::Unimplemented(_))),
            "{attention:?}"
        );
    }
    let mut two = base.clone();
    two.segments.push(two.segments[0]);
    assert!(matches!(
        two.ensure_supported(),
        Err(Error::Unimplemented(_))
    ));
}

#[test]
fn missing_directory_is_an_io_error() {
    let err = ModelConfig::from_dir(&fixture_dir().join("nope")).unwrap_err();
    assert!(matches!(err, Error::Io { .. }));
}
