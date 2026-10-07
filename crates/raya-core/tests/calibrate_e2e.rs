//! `calibrate` end to end: clamping, fallback, missing temperatures and non-prefix masks, which
//! `calibrate.rs` only checks through `clamp_temperature` in isolation.
#![allow(
    clippy::float_cmp,
    clippy::excessive_precision,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
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

const LOGITS: [f32; 3] = [2.0, 0.0, -2.0];

fn cal(a: &AgentConfig, qtype: usize) -> Vec<f32> {
    calibrate(&LOGITS, &[true; 3], qtype, a).unwrap()
}

fn with_t(t: f32) -> Vec<f32> {
    let a = agent(&format!("[{t:?}, {t:?}, {t:?}]"), "{}");
    cal(&a, 0)
}

#[test]
fn out_of_range_temperatures_are_clamped_end_to_end() {
    // below the floor behaves exactly like 0.5, above the ceiling exactly like 5.0
    assert_eq!(with_t(0.1), with_t(0.5));
    assert_eq!(with_t(0.0), with_t(0.5));
    assert_eq!(with_t(-3.0), with_t(0.5));
    assert_eq!(with_t(9.0), with_t(5.0));
    assert_eq!(with_t(1e30), with_t(5.0));
    // the boundaries themselves are kept, and in between is not clamped
    assert_ne!(with_t(0.5), with_t(0.51));
    assert_ne!(with_t(5.0), with_t(4.99));
    // the floor sharpens, the ceiling flattens
    assert!(with_t(0.5)[0] > with_t(1.0)[0]);
    assert!(with_t(5.0)[0] < with_t(1.0)[0]);
}

#[test]
fn clamping_also_applies_to_bucket_entries() {
    let a = agent("[1.0, 1.0, 1.0]", r#"{"choice:3-5": 0.1}"#);
    assert_eq!(cal(&a, 0), with_t(0.5));
    let a = agent("[1.0, 1.0, 1.0]", r#"{"choice:3-5": 50.0}"#);
    assert_eq!(cal(&a, 0), with_t(5.0));
}

#[test]
fn non_finite_temperatures_fall_back_to_one_end_to_end() {
    let want = with_t(1.0);
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut a = agent("[1.0, 1.0, 1.0]", "{}");
        a.temperature[0] = bad;
        assert_eq!(cal(&a, 0), want, "type temperature {bad}");

        let mut a = agent("[3.0, 3.0, 3.0]", "{}");
        a.temperature_by_options.insert("choice:3-5".into(), bad);
        assert_eq!(cal(&a, 0), want, "bucket temperature {bad}");
    }
    assert_eq!(clamp_temperature(f32::NEG_INFINITY), 1.0);
    assert_eq!(clamp_temperature(0.5), 0.5);
    assert_eq!(clamp_temperature(5.0), 5.0);
}

#[test]
fn a_bucket_for_another_qtype_or_size_is_not_used() {
    let a = agent(
        "[1.0, 2.0, 1.0]",
        r#"{"choice:6-10": 4.0, "score:3-5": 4.0}"#,
    );
    assert_eq!(cal(&a, 0), with_t(1.0)); // choice, k = 3: neither entry matches
    let plain = agent("[1.0, 4.0, 1.0]", "{}");
    assert_eq!(cal(&a, 1), cal(&plain, 1)); // score:3-5 matches: T = 4, not temperature[1] = 2
}

#[test]
fn a_missing_type_temperature_is_an_explicit_error() {
    let mut a = agent("[1.0, 1.0, 1.0]", "{}");
    a.temperature = vec![1.0];
    assert!(calibrate(&LOGITS, &[true; 3], 0, &a).is_ok());
    let err = calibrate(&LOGITS, &[true; 3], 2, &a).unwrap_err();
    assert!(
        matches!(&err, Error::Input(m) if m.contains("temperature has 1 entries")),
        "{err}"
    );
    // a bucket entry makes the type temperature unnecessary
    a.temperature_by_options.insert("noul:3-5".into(), 2.0);
    assert!(calibrate(&LOGITS, &[true; 3], 2, &a).is_ok());
    a.temperature.clear();
    assert!(calibrate(&LOGITS, &[true; 3], 0, &a).is_err());
}

#[test]
fn zero_options_land_in_the_smallest_bucket() {
    assert_eq!(temp_bucket(0, 0).unwrap(), "choice:2");
    assert_eq!(temp_bucket(1, 0).unwrap(), "score:2");
}

#[test]
fn non_prefix_masks_pick_the_valid_logits_in_order() {
    let a = agent("[1.0, 1.0, 1.0]", r#"{"choice:3-5": 4.0}"#);
    let want = calibrate(&[2.0, 0.0], &[true; 2], 0, &a).unwrap(); // k = 2: T = 1
    for (logits, mask) in [
        ([-1e4, 2.0, 0.0], [false, true, true]),
        ([2.0, -1e4, 0.0], [true, false, true]),
        ([2.0, 0.0, -1e4], [true, true, false]),
    ] {
        assert_eq!(calibrate(&logits, &mask, 0, &a).unwrap(), want, "{mask:?}");
    }
    // a lone valid marker in the last slot is a single option
    assert_eq!(
        calibrate(&[-1e4, -1e4, 0.7], &[false, false, true], 0, &a).unwrap(),
        [1.0]
    );
    // order is kept: the larger logit stays first
    let p = calibrate(&[-1e4, 0.0, 2.0], &[false, true, true], 0, &a).unwrap();
    assert!(p[0] < p[1]);
}
