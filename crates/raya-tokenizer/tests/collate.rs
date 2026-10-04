mod common;

use common::def;
use raya_tokenizer::{collate, Item, State};
use safetensors::SafeTensors;

fn items(texts: &[&str], q: &str) -> Vec<Item> {
    let b = common::builder();
    texts
        .iter()
        .flat_map(|t| {
            b.encode_state(
                &State::Text((*t).into()),
                &[("q".to_owned(), def(q))],
                96,
                64,
            )
            .unwrap()
        })
        .collect()
}

#[test]
fn pads_to_the_longest_item_and_widest_marker_list() {
    let choice = r#"{"type":"choice","instructions":"i","criteria":["a","b","c"]}"#;
    let noul = r#"{"type":"noul","instructions":"i"}"#;
    let mut a = items(&["short"], choice);
    let b = items(&["a much longer document than the other one"], noul);
    let (la, lb) = (a[0].ids.len(), b[0].ids.len());
    a.extend(b.clone());
    let pad = common::builder().pad_id();
    let batch = collate(&[a.clone()], pad, 0).unwrap();
    let len = la.max(lb);
    assert_eq!((batch.n, batch.len, batch.max_markers), (2, len, 3));
    assert_eq!(&batch.input_ids[..la], a[0].ids.as_slice());
    assert!(batch.input_ids[la..len].iter().all(|i| *i == pad));
    assert_eq!(
        batch.attention_mask[..len]
            .iter()
            .map(|m| usize::from(*m))
            .sum::<usize>(),
        la
    );
    assert_eq!(batch.marker_mask, [true, true, true, true, true, false]);
    assert_eq!(
        &batch.marker_pos[3..],
        [
            u32::try_from(b[0].markers[0]).unwrap(),
            u32::try_from(b[0].markers[1]).unwrap(),
            0
        ]
    );
    assert_eq!(batch.qtype, [0, 2]);
}

#[test]
fn groups_are_flattened_in_order() {
    let q = r#"{"type":"noul","instructions":"i"}"#;
    let g1 = items(&["one"], q);
    let g2 = items(&["two two"], q);
    let batch = collate(&[g1.clone(), g2.clone()], 0, 0).unwrap();
    assert_eq!(batch.n, 2);
    assert_eq!(&batch.input_ids[..g1[0].ids.len()], g1[0].ids.as_slice());
}

#[test]
fn no_items_gives_none() {
    assert!(collate(&[], 0, 0).is_none());
    assert!(collate(&[Vec::new()], 0, 0).is_none());
}

fn i64_tensor(st: &SafeTensors, name: &str) -> Vec<i64> {
    st.tensor(name)
        .unwrap()
        .data()
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn forward_case(case: &str) {
    let dir = common::data().join(case);
    let meta = common::read_json(&dir.join("meta.json"));
    let input = &meta["input"];
    let max_len = usize::try_from(input["max_len"].as_u64().unwrap()).unwrap();
    let head_max_len = usize::try_from(input["head_max_len"].as_u64().unwrap()).unwrap();
    let pad_to = input["pad_to"]
        .as_u64()
        .map_or(0, |n| usize::try_from(n).unwrap());
    let question = serde_json::from_value(input["question"].clone()).unwrap();
    let state = State::Text(input["state"].as_str().unwrap().to_owned());

    let b = common::builder();
    let built = b
        .encode_state(&state, &[("q".to_owned(), question)], max_len, head_max_len)
        .unwrap();
    let batch = collate(std::slice::from_ref(&built), b.pad_id(), pad_to).unwrap();

    let bytes = std::fs::read(dir.join("tensors.safetensors")).unwrap();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let ids: Vec<i64> = batch.input_ids.iter().map(|i| i64::from(*i)).collect();
    assert_eq!(ids, i64_tensor(&st, "input_ids"), "{case}: input_ids");
    let att: Vec<i64> = batch.attention_mask.iter().map(|i| i64::from(*i)).collect();
    assert_eq!(
        att,
        i64_tensor(&st, "attention_mask"),
        "{case}: attention_mask"
    );
    let pos: Vec<i64> = batch.marker_pos.iter().map(|i| i64::from(*i)).collect();
    assert_eq!(pos, i64_tensor(&st, "marker_pos"), "{case}: marker_pos");
    let qtype: Vec<i64> = batch.qtype.iter().map(|i| i64::from(*i)).collect();
    assert_eq!(qtype, i64_tensor(&st, "qtype"), "{case}: qtype");
    let mask: Vec<bool> = st
        .tensor("marker_mask")
        .unwrap()
        .data()
        .iter()
        .map(|b| *b != 0)
        .collect();
    assert_eq!(batch.marker_mask, mask, "{case}: marker_mask");
    let unpadded = usize::try_from(input["unpadded_length"].as_u64().unwrap()).unwrap();
    assert_eq!(built[0].ids.len(), unpadded, "{case}: unpadded length");
    let positions: Vec<i64> = built[0]
        .position_ids()
        .iter()
        .map(|p| i64::from(*p))
        .collect();
    assert_eq!(
        positions,
        i64_tensor(&st, "position_ids")[..unpadded],
        "{case}: position_ids"
    );
}

#[test]
fn short_forward_golden_input_rebuilds() {
    forward_case("short");
}

#[test]
fn medium_forward_golden_input_rebuilds() {
    forward_case("medium");
}

#[test]
fn padded_forward_golden_input_rebuilds() {
    forward_case("padded");
}
