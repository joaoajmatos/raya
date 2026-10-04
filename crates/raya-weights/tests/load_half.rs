mod common;

use common::half;
use raya_core::Error;
use raya_weights::{expected, load};
use safetensors::Dtype;

fn f32_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Loads a narrowed copy and checks every tensor equals the independently widened value of the
/// independently narrowed original, bit for bit.
fn check(dtype: Dtype, pick: impl Fn(&str) -> bool + Copy, expect_source: &[&str]) {
    let tmp = tempfile::tempdir().unwrap();
    let path = half::narrowed_h64(tmp.path(), dtype, pick);
    let cfg = common::h64_config();
    let store = load(&path, &cfg).unwrap();
    assert_eq!(store.source_dtypes(), expect_source);
    assert_eq!(store.tensor_count(), 44);
    assert_eq!(store.parameter_count(), 210_822);
    assert_eq!(store.total_bytes(), 210_822 * 4, "the store holds fp32");

    let original = common::read_all(&common::h64_path());
    for e in expected(&cfg).unwrap() {
        let t = store
            .get(e.key)
            .unwrap_or_else(|| panic!("{}", e.laya_name));
        assert_eq!(t.shape, e.shape.as_slice(), "{}", e.laya_name);
        let orig = f32_values(&original[&e.laya_name].2);
        assert_eq!(t.data.len(), orig.len(), "{}", e.laya_name);
        for (i, (got, x)) in t.data.iter().zip(&orig).enumerate() {
            let want = if pick(&e.laya_name) {
                match dtype {
                    Dtype::BF16 => half::widen_bf16(half::narrow_bf16(*x)),
                    _ => half::widen_f16(half::narrow_f16(*x)),
                }
            } else {
                *x
            };
            assert_eq!(got.to_bits(), want.to_bits(), "{}[{i}]", e.laya_name);
        }
    }
}

#[test]
fn bf16_checkpoint_widens_exactly() {
    check(Dtype::BF16, |_| true, &["BF16"]);
}

#[test]
fn f16_checkpoint_widens_exactly() {
    check(Dtype::F16, |_| true, &["F16"]);
}

#[test]
fn mixed_dtype_checkpoint_keeps_fp32_tensors_untouched() {
    let keep = |n: &str| n != "temperature" && n != "type_emb.weight";
    check(Dtype::F16, keep, &["F16", "F32"]);
    check(Dtype::BF16, keep, &["BF16", "F32"]);
}

#[test]
fn widened_values_are_within_rounding_of_the_originals() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = common::h64_config();
    let original = common::read_all(&common::h64_path());
    for (dtype, rel) in [(Dtype::BF16, 2f32.powi(-8)), (Dtype::F16, 2f32.powi(-11))] {
        let path = half::narrowed_h64(tmp.path(), dtype, |_| true);
        let store = load(&path, &cfg).unwrap();
        for e in expected(&cfg).unwrap() {
            let t = store.get(e.key).unwrap();
            for (got, x) in t.data.iter().zip(f32_values(&original[&e.laya_name].2)) {
                // fp16 subnormals have an absolute step of 2^-24.
                let tol = (x.abs() * rel).max(2f32.powi(-24));
                assert!(
                    (got - x).abs() <= tol,
                    "{dtype:?} {}: {got} vs {x}",
                    e.laya_name
                );
            }
        }
    }
}

#[test]
fn unsupported_dtypes_still_fail_naming_every_offender() {
    let tmp = tempfile::tempdir().unwrap();
    let path = common::mutated_h64(tmp.path(), |t| {
        *t.get_mut("temperature").unwrap() = (Dtype::F64, vec![3], vec![0; 24]);
        *t.get_mut("type_emb.weight").unwrap() = (Dtype::I32, vec![3, 64], vec![0; 3 * 64 * 4]);
    });
    match load(&path, &common::h64_config()).err().unwrap() {
        Error::Unimplemented(m) => {
            assert!(m.contains("temperature (F64)"), "{m}");
            assert!(m.contains("type_emb.weight (I32)"), "{m}");
        }
        other => panic!("{other}"),
    }
}

#[test]
fn half_precision_files_still_get_the_full_mismatch_report() {
    let tmp = tempfile::tempdir().unwrap();
    let narrowed = half::narrowed_h64(tmp.path(), Dtype::BF16, |_| true);
    let mut tensors = common::read_all(&narrowed);
    tensors.remove("temperature");
    tensors.get_mut("type_emb.weight").unwrap().1 = vec![64, 3];
    tensors.insert("bogus.weight".into(), (Dtype::BF16, vec![2], vec![0; 4]));
    let broken = tmp.path().join("broken.safetensors");
    common::write_all(&broken, &tensors);
    match load(&broken, &common::h64_config()).err().unwrap() {
        Error::Weights(r) => {
            assert_eq!(r.missing, ["temperature"]);
            assert_eq!(r.unused, ["bogus.weight"]);
            assert_eq!(r.shape.len(), 1);
            assert_eq!(r.shape[0].name, "type_emb.weight");
        }
        other => panic!("{other}"),
    }
}

#[test]
fn all_fp32_file_reports_f32_only() {
    let store = load(&common::h64_path(), &common::h64_config()).unwrap();
    assert_eq!(store.source_dtypes(), ["F32"]);
}
