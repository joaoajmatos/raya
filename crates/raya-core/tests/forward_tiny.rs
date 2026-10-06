#![allow(
    clippy::float_cmp,
    clippy::excessive_precision,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
mod common;

use raya_core::{Error, ForwardInput, Model, ParamKey, Trace};
use raya_kernels::Naive;

fn input<'a>(ids: &'a [u32], valid: usize, pos: &'a [usize], mask: &'a [bool]) -> ForwardInput<'a> {
    ForwardInput {
        input_ids: ids,
        valid_len: valid,
        marker_pos: pos,
        marker_mask: mask,
        qtype: 0,
    }
}

#[test]
fn forward_produces_finite_outputs_and_the_documented_trace() {
    let model = Model::new(common::config()).unwrap();
    let params = common::params();
    let net = model.bind(&params).unwrap();
    let ids = common::ids(12);
    let (pos, mask) = ([2usize, 5, 8], [true, true, false]);
    let mut trace = Trace::new();
    let out = net
        .forward(&input(&ids, 10, &pos, &mask), &Naive, Some(&mut trace))
        .unwrap();

    assert!(out.logits.iter().all(|v| v.is_finite()));
    assert_eq!(out.logits[2], -1e4, "invalid marker");
    assert_eq!(out.probabilities[2], 0.0);
    assert!((out.probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    assert_eq!(out.act_logits.len(), 2);
    assert_eq!(out.pooled.len(), common::D);
    // two valid markers: k = 2 and the features follow Laya's formula
    let (p0, p1) = (out.probabilities[0], out.probabilities[1]);
    assert!((out.features[0] - p0.max(p1)).abs() < 1e-7);
    assert!((out.features[1] - (p0.max(p1) - p0.min(p1))).abs() < 1e-7);
    assert!((out.features[3] - 2.0 / 255.0).abs() < 1e-9);

    let names: Vec<&str> = trace.names().collect();
    assert_eq!(
        names,
        [
            "embedding_output",
            "encoder_layer_00",
            "encoder_layer_01",
            "encoder_final_norm",
            "encoder_output",
            "type_emb_add",
            "head_input",
            "head_layer_00",
            "marker_states",
            "logits",
            "probabilities",
            "act_features",
            "pooled",
            "act_logits",
        ]
    );
    assert_eq!(trace.get("encoder_layer_01").unwrap().0, [1, 12, common::D]);
    assert_eq!(trace.get("marker_states").unwrap().0, [1, 3, common::D]);
    assert_eq!(trace.get("type_emb_add").unwrap().0, [1, 1, common::D]);
    assert_eq!(trace.get("logits").unwrap().1, out.logits.as_slice());
}

#[test]
fn tracing_does_not_change_the_numbers() {
    let model = Model::new(common::config()).unwrap();
    let params = common::params();
    let net = model.bind(&params).unwrap();
    let ids = common::ids(9);
    let (pos, mask) = ([1usize, 4], [true, true]);
    let a = net
        .forward(&input(&ids, 9, &pos, &mask), &Naive, None)
        .unwrap();
    let b = net
        .forward(
            &input(&ids, 9, &pos, &mask),
            &Naive,
            Some(&mut Trace::new()),
        )
        .unwrap();
    assert_eq!(a, b);
}

#[test]
fn a_single_marker_follows_laya_single_option_rule() {
    let model = Model::new(common::config()).unwrap();
    let params = common::params();
    let net = model.bind(&params).unwrap();
    let ids = common::ids(8);
    let out = net
        .forward(&input(&ids, 8, &[3], &[true]), &Naive, None)
        .unwrap();
    assert_eq!(out.probabilities, [1.0]);
    // top1 = 1, top1 - top2 = 1 (top2 padded with 0), entropy 0, k clamped to 2
    assert_eq!(out.features[0], 1.0);
    assert_eq!(out.features[1], 1.0);
    assert!(out.features[2].abs() < 1e-7);
    assert!((out.features[3] - 2.0 / 255.0).abs() < 1e-9);
}

#[test]
fn padding_after_valid_len_does_not_change_valid_rows() {
    let model = Model::new(common::config()).unwrap();
    let params = common::params();
    let net = model.bind(&params).unwrap();
    let mut ids = common::ids(14);
    let (pos, mask) = ([1usize, 6], [true, true]);
    let (mut full, mut cut) = (Trace::new(), Trace::new());
    let a = net
        .forward(&input(&ids, 9, &pos, &mask), &Naive, Some(&mut full))
        .unwrap();
    ids.truncate(9);
    let b = net
        .forward(&input(&ids, 9, &pos, &mask), &Naive, Some(&mut cut))
        .unwrap();
    for (x, y) in a.logits.iter().zip(&b.logits) {
        assert!((x - y).abs() < 1e-6);
    }
    let d = common::D;
    let (_, rows_full) = full.get("head_layer_00").unwrap();
    let (_, rows_cut) = cut.get("head_layer_00").unwrap();
    for (x, y) in rows_full[..9 * d].iter().zip(&rows_cut[..9 * d]) {
        assert!((x - y).abs() < 1e-6);
    }
}

#[test]
fn bad_inputs_are_explicit_errors() {
    let model = Model::new(common::config()).unwrap();
    let params = common::params();
    let net = model.bind(&params).unwrap();
    let ids = common::ids(8);
    let ok_pos = [1usize, 3];
    let ok_mask = [true, true];
    let run = |i: ForwardInput<'_>| net.forward(&i, &Naive, None).unwrap_err();

    let mut bad_ids = ids.clone();
    bad_ids[2] = 20;
    assert!(
        matches!(run(input(&bad_ids, 8, &ok_pos, &ok_mask)), Error::Input(m) if m.contains("vocabulary"))
    );
    assert!(
        matches!(run(input(&ids, 0, &ok_pos, &ok_mask)), Error::Input(m) if m.contains("valid_len"))
    );
    assert!(
        matches!(run(input(&ids, 9, &ok_pos, &ok_mask)), Error::Input(m) if m.contains("valid_len"))
    );
    assert!(
        matches!(run(input(&ids, 8, &[1, 8], &ok_mask)), Error::Input(m) if m.contains("marker position"))
    );
    assert!(
        matches!(run(input(&ids, 5, &[1, 6], &ok_mask)), Error::Input(m) if m.contains("marker position"))
    );
    assert!(
        matches!(run(input(&ids, 8, &[1, 2], &[false, false])), Error::Input(m) if m.contains("no valid marker"))
    );
    assert!(matches!(
        run(input(&ids, 8, &[1, 2], &[true])),
        Error::Input(_)
    ));
    let mut q3 = input(&ids, 8, &ok_pos, &ok_mask);
    q3.qtype = 3;
    assert!(matches!(run(q3), Error::Input(m) if m.contains("qtype")));
}

#[test]
fn missing_or_misshaped_parameters_name_the_key() {
    let model = Model::new(common::config()).unwrap();
    let mut params = common::params();
    let removed = params.0.remove(&ParamKey::AttnNorm(1)).unwrap();
    let err = model.bind(&params).err().unwrap();
    assert!(
        matches!(&err, Error::Input(m) if m.contains("AttnNorm(1)")),
        "{err}"
    );

    params
        .0
        .insert(ParamKey::AttnNorm(1), (vec![1.0; 3], vec![3]));
    let err = model.bind(&params).err().unwrap();
    assert!(
        matches!(&err, Error::Input(m) if m.contains("AttnNorm(1)") && m.contains("[16]")),
        "{err}"
    );

    params.0.insert(ParamKey::AttnNorm(1), removed);
    assert!(model.bind(&params).is_ok());
}
