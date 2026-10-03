mod common;

use raya_tokenizer::QType;
use serde_json::Value;

fn usize_vec(v: &Value) -> Vec<usize> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| usize::try_from(x.as_u64().unwrap()).unwrap())
        .collect()
}

fn u32_vec(v: &Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| u32::try_from(x.as_u64().unwrap()).unwrap())
        .collect()
}

fn first_diff(a: &[u32], b: &[u32]) -> Option<usize> {
    (0..a.len().max(b.len())).find(|&i| a.get(i) != b.get(i))
}

#[test]
fn special_token_ids_match_meta() {
    let b = common::builder();
    let special = &common::meta()["special_tokens"];
    assert_eq!(u64::from(b.cls_id()), special["cls"].as_u64().unwrap());
    assert_eq!(u64::from(b.sep_id()), special["sep"].as_u64().unwrap());
    assert_eq!(u64::from(b.mask_id()), special["mask"].as_u64().unwrap());
    assert_eq!(u64::from(b.pad_id()), special["pad"].as_u64().unwrap());
}

#[test]
fn all_eleven_records_are_present() {
    let names = common::record_names();
    let meta = common::meta();
    let listed: Vec<&str> = meta["records"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(names, listed);
    assert_eq!(names.len(), 11);
}

#[test]
fn every_record_matches_exactly() {
    let b = common::builder();
    for name in common::record_names() {
        let rec = common::record(&name);
        let max_len = usize::try_from(rec["max_len"].as_u64().unwrap()).unwrap();
        let head_max_len = usize::try_from(rec["head_max_len"].as_u64().unwrap()).unwrap();
        let state = common::state_of(&rec);
        assert_eq!(
            state.truncate_left(),
            rec["truncate_left"].as_bool().unwrap(),
            "{name}: truncate_left"
        );
        let questions = common::questions_of(&rec["questions"]);
        let items = b
            .encode_state(&state, &questions, max_len, head_max_len)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let golden = rec["items"].as_array().unwrap();
        assert_eq!(items.len(), golden.len(), "{name}: item count");
        // The record stores `questions` key-sorted but `items` in the generator's question
        // order, so items are matched by question id.
        for (item, (qid, _)) in items.iter().zip(&questions) {
            let g = golden
                .iter()
                .find(|g| g["qid"] == qid.as_str())
                .unwrap_or_else(|| panic!("{name}: no golden item for {qid}"));
            let ctx = format!("{name}/{qid}");
            let want_ids = u32_vec(&g["input_ids"]);
            assert_eq!(
                item.ids,
                want_ids,
                "{ctx}: input_ids differ at {:?}",
                first_diff(&item.ids, &want_ids)
            );
            assert_eq!(
                item.position_ids(),
                u32_vec(&g["position_ids"]),
                "{ctx}: position_ids"
            );
            assert_eq!(item.markers, usize_vec(&g["markers"]), "{ctx}: markers");
            assert_eq!(
                item.ids.len() as u64,
                g["length"].as_u64().unwrap(),
                "{ctx}"
            );
            assert_eq!(
                u64::from(item.qtype.id()),
                g["qtype"].as_u64().unwrap(),
                "{ctx}"
            );
            assert_eq!(
                item.qtype.name(),
                g["qtype_name"].as_str().unwrap(),
                "{ctx}: qtype_name"
            );
            let opts = &g["options_stats"];
            assert_eq!(
                item.stats.options as u64,
                opts["options"].as_u64().unwrap(),
                "{ctx}"
            );
            assert_eq!(
                item.stats.options_distinct as u64,
                opts["options_distinct"].as_u64().unwrap(),
                "{ctx}"
            );
            assert_eq!(
                item.stats.tokens_per_option.map(|n| n as u64),
                opts["tokens_per_option"].as_u64(),
                "{ctx}: tokens_per_option"
            );
            let t = &item.trace;
            for (field, got) in [
                ("head_length", t.head_length as u64),
                ("state_budget", t.state_budget as u64),
                ("state_tokens_total", t.state_tokens_total as u64),
                ("state_tokens_kept", t.state_tokens_kept as u64),
            ] {
                assert_eq!(got, g[field].as_u64().unwrap(), "{ctx}: {field}");
            }
            assert_eq!(
                t.state_truncated,
                g["state_truncated"].as_bool().unwrap(),
                "{ctx}"
            );
            assert_eq!(
                t.sequence_clipped,
                g["sequence_clipped"].as_bool().unwrap(),
                "{ctx}"
            );
        }
    }
}

#[test]
fn single_question_build_sequence_matches_encode_state() {
    let b = common::builder();
    let rec = common::record("primary_one_question");
    let state = common::state_of(&rec);
    let questions = common::questions_of(&rec["questions"]);
    let (id, def) = &questions[0];
    let q = raya_tokenizer::Question::from_def(id, def).unwrap();
    assert_eq!(q.qtype, QType::Choice);
    let state_ids = b.state_ids(&state).unwrap();
    let direct = b.build_sequence(&state_ids, false, &q, 96, 64).unwrap();
    let want = u32_vec(&rec["items"][0]["input_ids"]);
    assert_eq!(direct.ids, want);
}

#[test]
fn fixed_text_encodes_to_golden_ids() {
    // `empty_document` has no state, so its ids before the state are tokenizer output for the
    // question head and options alone; the head ids start after [CLS].
    let b = common::builder();
    let rec = common::record("primary_one_question");
    let ids = u32_vec(&rec["items"][0]["input_ids"]);
    let encoded = b
        .tokenizer()
        .encode(
            "choice question: which option best matches the document",
            false,
        )
        .unwrap();
    assert_eq!(encoded.get_ids(), &ids[1..=encoded.get_ids().len()]);
}
