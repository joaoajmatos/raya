mod common;

use std::collections::HashSet;

use raya_weights::expected;

#[test]
fn h64_table_matches_inventory() {
    let entries = expected(&common::h64_config()).unwrap();
    assert_eq!(entries.len(), 44);
    let names: HashSet<_> = entries.iter().map(|e| e.laya_name.as_str()).collect();
    assert_eq!(names.len(), 44, "Laya names must be unique");
    let keys: HashSet<_> = entries.iter().map(|e| e.key).collect();
    assert_eq!(keys.len(), 44, "typed keys must be unique");
    assert!(!names.contains("encoder.layers.0.attn_norm.weight"));

    let inv = common::inventory();
    let inv_keys = inv["keys"].as_object().unwrap();
    assert_eq!(inv_keys.len(), 44);
    for e in &entries {
        let item = inv_keys
            .get(&e.laya_name)
            .unwrap_or_else(|| panic!("{} not in inventory", e.laya_name));
        let shape: Vec<usize> = item["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| usize::try_from(v.as_u64().unwrap()).unwrap())
            .collect();
        assert_eq!(e.shape, shape, "{}", e.laya_name);
    }
    let total: usize = entries
        .iter()
        .map(|e| e.shape.iter().product::<usize>())
        .sum();
    assert_eq!(total as u64, inv["n_parameters"].as_u64().unwrap());
    assert_eq!(total, 210_822);
}

#[test]
fn h128_table_matches_file_header() {
    let entries = expected(&common::h128_config()).unwrap();
    let tensors = common::read_all(&common::h128_path());
    assert_eq!(entries.len(), tensors.len());
    for e in &entries {
        let (dtype, shape, _) = tensors
            .get(&e.laya_name)
            .unwrap_or_else(|| panic!("{} not in h128 file", e.laya_name));
        assert_eq!(*dtype, safetensors::Dtype::F32);
        assert_eq!(&e.shape, shape, "{}", e.laya_name);
    }
}

#[test]
fn bias_configs_are_unimplemented() {
    for flag in ["attention_bias", "mlp_bias", "norm_bias"] {
        let mut cfg = common::h64_config();
        match flag {
            "attention_bias" => cfg.encoder.attention_bias = true,
            "mlp_bias" => cfg.encoder.mlp_bias = true,
            _ => cfg.encoder.norm_bias = true,
        }
        assert!(
            matches!(expected(&cfg), Err(raya_core::Error::Unimplemented(m)) if m.contains(flag)),
            "{flag}"
        );
    }
}
