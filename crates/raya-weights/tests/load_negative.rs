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

    // unsupported dtype (bf16 and fp16 are widened instead; see load_half.rs)
    let p = common::mutated_h64(tmp.path(), |t| {
        *t.get_mut("temperature").unwrap() = (Dtype::F64, vec![3], vec![0; 24]);
    });
    match load(&p, &cfg()).err().unwrap() {
        Error::Unimplemented(m) => {
            assert!(m.contains("temperature") && m.contains("F64"), "{m}");
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

/// Appends a second `temperature` entry to the header of h64, keeping every offset valid.
fn with_duplicate_temperature(dir: &Path) -> PathBuf {
    let bytes = std::fs::read(common::h64_path()).unwrap();
    let n = usize::try_from(u64::from_le_bytes(bytes[..8].try_into().unwrap())).unwrap();
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let entry = header["temperature"].to_string();
    let text = std::str::from_utf8(&bytes[8..8 + n]).unwrap().trim_end();
    let mut new = format!("{},\"temperature\":{entry}}}", &text[..text.len() - 1]);
    while new.len() % 8 != 0 {
        new.push(' ');
    }
    let mut out = (new.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(new.as_bytes());
    out.extend_from_slice(&bytes[8 + n..]);
    let path = dir.join("dup.safetensors");
    std::fs::write(&path, out).unwrap();
    path
}

#[test]
fn duplicate_key_is_reported() {
    let tmp = tempfile::tempdir().unwrap();
    let r = report(
        load(&with_duplicate_temperature(tmp.path()), &cfg())
            .err()
            .expect("duplicate key must be an error"),
    );
    assert_eq!(r.duplicate, ["temperature"]);
    assert!(r.missing.is_empty() && r.unused.is_empty() && r.shape.is_empty());
    assert!(Error::Weights(Box::new(r)).to_string().contains(
        "duplicate tensors:
    temperature"
    ));
}

/// `fixture-tiny/` has `vocab_size` 100; `weights-h64` was saved with 231. The loader must
/// reject the pair, naming the embedding table, rather than load a mismatched model.
#[test]
fn fixture_tiny_config_does_not_match_h64_weights() {
    let dir = common::data().join("fixture-tiny");
    let config = ModelConfig::from_dir(&dir).unwrap();
    assert_eq!(config.encoder.vocab_size, 100);
    let r = report(load(&common::h64_path(), &config).err().unwrap());
    assert_eq!(r.shape.len(), 1, "{r}");
    assert_eq!(r.shape[0].name, "encoder.embeddings.tok_embeddings.weight");
    assert_eq!(r.shape[0].expected, [100, 64]);
    assert_eq!(r.shape[0].found, [231, 64]);
}
