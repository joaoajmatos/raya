mod common;

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use raya_core::{Error, LoadReport, ModelConfig};
use raya_weights::load;
use safetensors::Dtype;

fn hash_file(p: &Path) -> u64 {
    let mut h = DefaultHasher::new();
    std::fs::read(p).unwrap().hash(&mut h);
    h.finish()
}

fn report(err: Error) -> LoadReport {
    match err {
        Error::Weights(r) => *r,
        other => panic!("expected Weights, got {other}"),
    }
}

fn cfg() -> ModelConfig {
    common::h64_config()
}

fn bogus(t: &mut std::collections::BTreeMap<String, common::Raw>) {
    t.insert("bogus.weight".into(), (Dtype::F32, vec![1], vec![0; 4]));
}

#[test]
fn negative_cases() {
    let fixtures: [PathBuf; 2] = [common::h64_path(), common::h128_path()];
    let before: Vec<_> = fixtures.iter().map(|p| hash_file(p)).collect();
    let tmp = tempfile::tempdir().unwrap();

    // missing
    let p = common::mutated_h64(tmp.path(), |t| {
        t.remove("temperature");
    });
    let r = report(load(&p, &cfg()).err().unwrap());
    assert_eq!(r.missing, ["temperature"]);
    assert!(r.unused.is_empty() && r.shape.is_empty());

    // extra
    let p = common::mutated_h64(tmp.path(), bogus);
    let r = report(load(&p, &cfg()).err().unwrap());
    assert_eq!(r.unused, ["bogus.weight"]);

    // both at once
    let p = common::mutated_h64(tmp.path(), |t| {
        t.remove("type_emb.weight");
        bogus(t);
    });
    let r = report(load(&p, &cfg()).err().unwrap());
    assert_eq!(r.missing, ["type_emb.weight"]);
    assert_eq!(r.unused, ["bogus.weight"]);
    let text = Error::Weights(Box::new(r)).to_string();
    assert!(text.contains("type_emb.weight") && text.contains("bogus.weight"));

    // wrong shape
    let p = common::mutated_h64(tmp.path(), |t| {
        t.get_mut("type_emb.weight").unwrap().1 = vec![64, 3];
    });
    let r = report(load(&p, &cfg()).err().unwrap());
    assert_eq!(r.shape.len(), 1);
    assert_eq!(r.shape[0].name, "type_emb.weight");
    assert_eq!(r.shape[0].expected, [3, 64]);
    assert_eq!(r.shape[0].found, [64, 3]);

    // bf16 tensor
    let p = common::mutated_h64(tmp.path(), |t| {
        *t.get_mut("temperature").unwrap() = (Dtype::BF16, vec![3], vec![0; 6]);
    });
    match load(&p, &cfg()).err().unwrap() {
        Error::Unimplemented(m) => {
            assert!(m.contains("temperature") && m.contains("BF16"), "{m}");
        }
        other => panic!("{other}"),
    }

    // missing path
    let nope = tmp.path().join("nope.safetensors");
    match load(&nope, &cfg()).err().unwrap() {
        Error::Io { path, .. } => assert_eq!(path, nope),
        other => panic!("{other}"),
    }

    // truncated and garbage
    let bytes = std::fs::read(common::h64_path()).unwrap();
    let trunc = tmp.path().join("trunc.safetensors");
    std::fs::write(&trunc, &bytes[..bytes.len() / 2]).unwrap();
    let r = report(load(&trunc, &cfg()).err().unwrap());
    assert_eq!(r.path, trunc);
    let junk = tmp.path().join("junk.safetensors");
    std::fs::write(&junk, b"not a safetensors file at all").unwrap();
    let r = report(load(&junk, &cfg()).err().unwrap());
    assert_eq!(r.path, junk);
    assert!(r.detail.is_some());

    // config rejected before the file is opened
    let mut bad = cfg();
    bad.encoder.mlp_bias = true;
    assert!(matches!(
        load(&nope, &bad).err().unwrap(),
        Error::Unimplemented(_)
    ));

    let after: Vec<_> = fixtures.iter().map(|p| hash_file(p)).collect();
    assert_eq!(before, after, "vendored fixtures changed");
}
