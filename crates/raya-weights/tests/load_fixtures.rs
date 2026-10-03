mod common;

use raya_weights::{expected, load, ParamKey};
use safetensors::SafeTensors;

fn check(path: &std::path::Path, cfg: &raya_core::ModelConfig) {
    let store = load(path, cfg).unwrap();
    let entries = expected(cfg).unwrap();
    assert_eq!(store.tensor_count(), entries.len());

    let bytes = std::fs::read(path).unwrap();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let mut numel = 0;
    let mut ranges = Vec::new();
    for e in &entries {
        let t = store
            .get(e.key)
            .unwrap_or_else(|| panic!("{}", e.laya_name));
        assert_eq!(t.shape, e.shape.as_slice(), "{}", e.laya_name);
        let raw = st.tensor(&e.laya_name).unwrap();
        let want: Vec<u32> = raw
            .data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()).to_bits())
            .collect();
        let got: Vec<u32> = t.data.iter().map(|x| x.to_bits()).collect();
        assert_eq!(got, want, "{}", e.laya_name);
        numel += t.data.len();
        ranges.push((t.data.as_ptr() as usize, t.data.len() * 4));
    }
    assert_eq!(store.parameter_count(), numel);
    assert_eq!(store.total_bytes(), numel * 4);
    // No per-tensor copy: all slices fall inside one file-sized span.
    let lo = ranges.iter().map(|r| r.0).min().unwrap();
    let hi = ranges.iter().map(|r| r.0 + r.1).max().unwrap();
    assert!(hi - lo <= bytes.len(), "slices span more than one buffer");
}

#[test]
fn h64_loads_bit_exact() {
    check(&common::h64_path(), &common::h64_config());
}

#[test]
fn h128_loads_bit_exact() {
    check(&common::h128_path(), &common::h128_config());
}

#[test]
fn temperature_and_type_emb_load() {
    let store = load(&common::h64_path(), &common::h64_config()).unwrap();
    assert_eq!(store.get(ParamKey::Temperature).unwrap().shape, &[3]);
    assert_eq!(store.get(ParamKey::TypeEmb).unwrap().shape, &[3, 64]);
    assert!(store.get(ParamKey::AttnNorm(0)).is_none());
}
