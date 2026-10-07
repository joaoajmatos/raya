//! Forward-pass edge cases (spec 006 hardening): non-prefix marker masks, the shortest sequence,
//! `qtype` 2, and the score features for k >= 3. Tests only; they pin current behavior.
#![allow(
    clippy::float_cmp,
    clippy::excessive_precision,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
mod common;

use raya_core::{ForwardInput, ForwardOutput, ParamKey, Trace};
use raya_kernels::Naive;

fn run(ids: &[u32], valid: usize, pos: &[usize], mask: &[bool], qtype: usize) -> ForwardOutput {
    run_traced(ids, valid, pos, mask, qtype, None)
}

fn run_traced(
    ids: &[u32],
    valid: usize,
    pos: &[usize],
    mask: &[bool],
    qtype: usize,
    trace: Option<&mut Trace>,
) -> ForwardOutput {
    let model = raya_core::Model::new(common::config()).unwrap();
    let params = common::params();
    let net = model.bind(&params).unwrap();
    net.forward(
        &ForwardInput {
            input_ids: ids,
            valid_len: valid,
            marker_pos: pos,
            marker_mask: mask,
            qtype,
        },
        &Naive,
        trace,
    )
    .unwrap()
}

fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-6, "{x} vs {y}");
    }
}

#[test]
fn a_leading_invalid_marker_leaves_the_valid_ones_unchanged() {
    let ids = common::ids(12);
    let full = run(&ids, 12, &[5, 8], &[true, true], 0);
    let out = run(&ids, 12, &[2, 5, 8], &[false, true, true], 0);
    assert_eq!(out.logits[0], -1e4);
    assert_eq!(out.probabilities[0], 0.0);
    // the head layers do not look at which markers are valid: same rows, same logits
    close(&out.logits[1..], &full.logits);
    close(&out.probabilities[1..], &full.probabilities);
    close(&out.features, &full.features);
    assert!((out.probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-6);
}

#[test]
fn an_invalid_marker_in_the_middle_keeps_slots_aligned() {
    let ids = common::ids(12);
    let packed = run(&ids, 12, &[2, 8], &[true, true], 0);
    let out = run(&ids, 12, &[2, 5, 8], &[true, false, true], 0);
    assert_eq!(out.logits[1], -1e4);
    assert_eq!(out.probabilities[1], 0.0);
    close(&[out.logits[0], out.logits[2]], &packed.logits);
    close(
        &[out.probabilities[0], out.probabilities[2]],
        &packed.probabilities,
    );
    close(&out.features, &packed.features);
}

#[test]
fn the_shortest_sequence_runs() {
    let ids = common::ids(1);
    let out = run(&ids, 1, &[0], &[true], 0);
    assert!(out.logits.iter().all(|v| v.is_finite()));
    assert!(out.pooled.iter().all(|v| v.is_finite()));
    assert!(out.act_logits.iter().all(|v| v.is_finite()));
    assert_eq!(out.probabilities, [1.0]);
}

#[test]
fn valid_len_one_inside_a_longer_padded_sequence_matches_seq_one() {
    let ids = common::ids(6);
    let short = run(&ids[..1], 1, &[0], &[true], 0);
    let padded = run(&ids, 1, &[0], &[true], 0);
    close(&padded.logits, &short.logits);
    close(&padded.pooled, &short.pooled);
    close(&padded.act_logits, &short.act_logits);
}

#[test]
fn qtype_two_selects_the_third_type_embedding_row() {
    let ids = common::ids(9);
    let (pos, mask) = ([1usize, 4, 6], [true; 3]);
    let mut t2 = Trace::new();
    let out2 = run_traced(&ids, 9, &pos, &mask, 2, Some(&mut t2));
    assert!(out2.logits.iter().all(|v| v.is_finite()));
    assert!(out2.act_logits.iter().all(|v| v.is_finite()));

    let (te, shape) = common::params().0.remove(&ParamKey::TypeEmb).unwrap();
    let d = common::D;
    assert_eq!(shape, [3, d]);
    let (_, added) = t2.get("type_emb_add").unwrap();
    close(added, &te[2 * d..3 * d]);

    // and the three types give three different results
    let out0 = run(&ids, 9, &pos, &mask, 0);
    let out1 = run(&ids, 9, &pos, &mask, 1);
    assert_ne!(out2.logits, out0.logits);
    assert_ne!(out2.logits, out1.logits);
    assert_ne!(out0.logits, out1.logits);
}

#[test]
fn features_for_four_options_follow_laya_formula() {
    let ids = common::ids(12);
    let out = run(&ids, 12, &[1, 3, 5, 7], &[true; 4], 0);
    let p = &out.probabilities;
    assert_eq!(p.len(), 4);
    assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    let mut sorted = p.clone();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let entropy = -p.iter().map(|&x| x * x.max(1e-9).ln()).sum::<f32>() / 4.0f32.ln();
    assert!((out.features[0] - sorted[0]).abs() < 1e-7);
    assert!((out.features[1] - (sorted[0] - sorted[1])).abs() < 1e-7);
    assert!((out.features[2] - entropy).abs() < 1e-6);
    assert!(out.features[2] > 0.0 && out.features[2] <= 1.0 + 1e-6);
    assert!((out.features[3] - 4.0 / 255.0).abs() < 1e-9);
}

#[test]
fn features_count_valid_markers_only_for_k() {
    let ids = common::ids(12);
    let out = run(&ids, 12, &[1, 3, 5, 7], &[true, false, true, true], 0);
    assert_eq!(out.probabilities[1], 0.0);
    // k = 3 valid markers, not 4 slots
    assert!((out.features[3] - 3.0 / 255.0).abs() < 1e-9);
    let p: Vec<f32> = [0, 2, 3].iter().map(|&i| out.probabilities[i]).collect();
    let entropy = -p.iter().map(|&x| x * x.max(1e-9).ln()).sum::<f32>() / 3.0f32.ln();
    assert!((out.features[2] - entropy).abs() < 1e-6);
}
