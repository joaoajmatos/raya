#![allow(
    clippy::float_cmp,
    clippy::excessive_precision,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
use std::collections::BTreeMap;

use raya_core::calibrate::{clamp_temperature, temp_bucket};
use raya_core::{calibrate, AgentConfig, Error, ModelConfig};

fn agent(temperature: &str, by_options: &str) -> AgentConfig {
    let enc = r#"{"hidden_size":64,"intermediate_size":128,"num_hidden_layers":3,
        "num_attention_heads":2,"global_attn_every_n_layers":2,"local_attention":16,
        "norm_eps":1e-5,"norm_bias":false,"attention_bias":false,"mlp_bias":false,
        "hidden_activation":"gelu","max_position_embeddings":2048,"vocab_size":231,
        "global_rope_theta":160000.0,"local_rope_theta":10000.0}"#;
    let ag = format!(
        r#"{{"head_layers":1,"max_len":256,"head_max_len":96,"temperature":{temperature},
        "temperature_by_options":{by_options},"act_costs":{{"abstain":0.1}}}}"#
    );
    ModelConfig::from_json(enc, &ag).unwrap().agent
}

#[test]
fn bucket_boundaries() {
    for (k, want) in [
        (1, "choice:2"),
        (2, "choice:2"),
        (3, "choice:3-5"),
        (5, "choice:3-5"),
        (6, "choice:6-10"),
        (10, "choice:6-10"),
        (11, "choice:11+"),
        (200, "choice:11+"),
    ] {
        assert_eq!(temp_bucket(0, k).unwrap(), want);
    }
    assert_eq!(temp_bucket(1, 4).unwrap(), "score:3-5");
    assert_eq!(temp_bucket(2, 2).unwrap(), "noul:2");
    assert!(matches!(temp_bucket(3, 2), Err(Error::Input(_))));
}

#[test]
fn temperatures_are_clamped_like_laya() {
    assert_eq!(clamp_temperature(0.1006), 0.5);
    assert_eq!(clamp_temperature(9.0), 5.0);
    assert_eq!(clamp_temperature(2.0), 2.0);
    assert_eq!(clamp_temperature(f32::NAN), 1.0);
    assert_eq!(clamp_temperature(f32::INFINITY), 1.0);
}

#[test]
fn calibration_scales_logits_and_sums_to_one() {
    let a = agent("[2.0, 1.0, 1.0]", "{}");
    let p = calibrate(&[2.0, 0.0, -2.0], &[true; 3], 0, &a).unwrap();
    assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    // T = 2 halves the logits: softmax([1, 0, -1])
    let e = [1.0f32.exp(), 1.0, (-1.0f32).exp()];
    let s: f32 = e.iter().sum();
    for (g, w) in p.iter().zip(e) {
        assert!((g - w / s).abs() < 1e-6);
    }
}

#[test]
fn bucket_entry_overrides_the_type_temperature() {
    let a = agent("[1.0, 1.0, 1.0]", r#"{"choice:3-5": 4.0}"#);
    let flat = calibrate(&[2.0, 0.0, -2.0], &[true; 3], 0, &a).unwrap(); // k = 3: bucket entry, T = 4
    let plain = calibrate(&[2.0, 0.0, -2.0], &[true; 3], 1, &a).unwrap(); // score: temperature[1] = 1
    assert!(flat[0] < plain[0]);
    let _: BTreeMap<String, f32> = a.temperature_by_options.clone();
}

#[test]
fn degenerate_inputs_are_errors() {
    let a = agent("[1.0, 1.0, 1.0]", "{}");
    assert!(matches!(calibrate(&[], &[], 0, &a), Err(Error::Input(_))));
    assert!(matches!(
        calibrate(&[1.0], &[true], 3, &a),
        Err(Error::Input(_))
    ));
    assert!(matches!(
        calibrate(&[1.0], &[false], 0, &a),
        Err(Error::Input(_))
    ));
    assert!(matches!(
        calibrate(&[1.0, 2.0], &[true], 0, &a),
        Err(Error::Input(_))
    ));
    // a single option stays at probability 1
    assert_eq!(calibrate(&[0.3], &[true], 2, &a).unwrap(), [1.0]);
}

/// `k` is the number of valid markers, as in `Agent._decode_answers`: a forward output with
/// invalid (`-1e4`) slots must use the bucket of the valid count, not of the slot count.
#[test]
fn invalid_markers_are_dropped_and_not_counted_in_k() {
    // choice:3-5 (k = 3) runs hot at 4.0, choice:6-10 (k = 6 slots) would be 1.0
    let a = agent("[1.0, 1.0, 1.0]", r#"{"choice:3-5": 4.0}"#);
    let logits = [2.0, 0.0, -2.0, -1e4, -1e4, -1e4];
    let mask = [true, true, true, false, false, false];
    let p = calibrate(&logits, &mask, 0, &a).unwrap();
    assert_eq!(p.len(), 3);
    assert_eq!(p, calibrate(&logits[..3], &[true; 3], 0, &a).unwrap());
}
