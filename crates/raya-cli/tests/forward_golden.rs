//! Layer-by-layer parity of the reference forward pass with Laya's golden fixtures
//! (`tests/data/<case>/`, spec 006 US1): every golden tensor, on the valid rows.
#![allow(
    clippy::float_cmp,
    clippy::excessive_precision,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
use std::path::{Path, PathBuf};

use raya_core::{ForwardInput, Model, ModelConfig, Trace};
use raya_kernels::Naive;
use safetensors::{Dtype, SafeTensors};
use serde_json::Value;

/// The spec starts at 1e-4 (layer outputs) and 1e-5 (probabilities) and allows tightening to the
/// measured error. First run on all four cases (Windows `x86_64`, rustc 1.94.1, portable scalar
/// loops): worst intermediate 7.2e-7, worst probability 3.0e-8, so the bounds are 14x and 33x
/// above the measurements (headroom for libm differences on macOS arm64 and Linux).
/// `logits`, `act_logits` and the other intermediates share the layer bound.
const LAYER_TOL: f32 = 1e-5;
const PROB_TOL: f32 = 1e-6;

fn data() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/data")
}

struct Golden {
    bytes: Vec<u8>,
}

impl Golden {
    fn open(case: &str) -> Self {
        let path = data().join(case).join("tensors.safetensors");
        Self {
            bytes: std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
        }
    }

    fn view(&self) -> SafeTensors<'_> {
        SafeTensors::deserialize(&self.bytes).unwrap()
    }

    fn f32s(&self, name: &str) -> (Vec<usize>, Vec<f32>) {
        let t = self.view();
        let v = t
            .tensor(name)
            .unwrap_or_else(|_| panic!("no tensor {name}"));
        assert_eq!(v.dtype(), Dtype::F32, "{name}");
        let data = v
            .data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        (v.shape().to_vec(), data)
    }

    fn ints(&self, name: &str) -> Vec<usize> {
        let t = self.view();
        let v = t.tensor(name).unwrap();
        assert_eq!(v.dtype(), Dtype::I64, "{name}");
        v.data()
            .chunks_exact(8)
            .map(|c| usize::try_from(i64::from_le_bytes(c.try_into().unwrap())).unwrap())
            .collect()
    }

    fn bools(&self, name: &str) -> Vec<bool> {
        let t = self.view();
        let v = t.tensor(name).unwrap();
        assert_eq!(v.dtype(), Dtype::BOOL, "{name}");
        v.data().iter().map(|&b| b != 0).collect()
    }
}

struct Case {
    name: &'static str,
    config: ModelConfig,
    weights: PathBuf,
    valid_len: usize,
    golden: Golden,
}

fn load_case(name: &'static str) -> Case {
    let dir = data().join(name);
    let meta: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
    let config_dir = dir.join(meta["config_dir"].as_str().unwrap());
    let weights = dir.join(meta["weights_file"].as_str().unwrap());
    Case {
        name,
        config: ModelConfig::from_dir(&config_dir).unwrap(),
        weights,
        valid_len: usize::try_from(meta["input"]["unpadded_length"].as_u64().unwrap()).unwrap(),
        golden: Golden::open(name),
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// `(tensor name, tolerance)` in execution order for a model with `layers` encoder layers and
/// `head_layers` head layers.
fn stages(layers: usize, head_layers: usize) -> Vec<(String, f32)> {
    let mut v = vec![("embedding_output".to_string(), LAYER_TOL)];
    v.extend((0..layers).map(|i| (format!("encoder_layer_{i:02}"), LAYER_TOL)));
    for n in [
        "encoder_final_norm",
        "encoder_output",
        "type_emb_add",
        "head_input",
    ] {
        v.push((n.to_string(), LAYER_TOL));
    }
    v.extend((0..head_layers).map(|i| (format!("head_layer_{i:02}"), LAYER_TOL)));
    for (n, tol) in [
        ("marker_states", LAYER_TOL),
        ("logits", LAYER_TOL),
        ("probabilities", PROB_TOL),
        ("act_features", PROB_TOL),
        ("pooled", LAYER_TOL),
        ("act_logits", LAYER_TOL),
    ] {
        v.push((n.to_string(), tol));
    }
    v
}

fn run(case: &Case, ids: &[u32], valid_len: usize, trace: &mut Trace) {
    let model = Model::new(case.config.clone()).unwrap();
    let store = raya_weights::load(&case.weights, &case.config).unwrap();
    let net = model.bind(&store).unwrap();
    let g = &case.golden;
    let (mask, pos, qtype) = (
        g.bools("marker_mask"),
        g.ints("marker_pos"),
        g.ints("qtype")[0],
    );
    let input = ForwardInput {
        input_ids: ids,
        valid_len,
        marker_pos: &pos,
        marker_mask: &mask,
        qtype,
    };
    net.forward(&input, &Naive, Some(trace)).unwrap();
}

fn golden_ids(case: &Case) -> Vec<u32> {
    case.golden
        .ints("input_ids")
        .into_iter()
        .map(|i| u32::try_from(i).unwrap())
        .collect()
}

/// Max abs difference per stage over the valid rows; prints a table.
fn compare(case: &Case, trace: &Trace) -> Vec<(String, f32, f32)> {
    let g = &case.golden;
    let head_layers = case.config.agent.head_layers;
    let mut rows = Vec::new();
    for (name, tol) in stages(case.config.encoder.num_hidden_layers, head_layers) {
        let (shape, want) = g.f32s(&name);
        let (got_shape, got) = trace
            .get(&name)
            .unwrap_or_else(|| panic!("{} no trace {name}", case.name));
        assert_eq!(got_shape, shape.as_slice(), "{} {name} shape", case.name);
        // sequence-shaped tensors: only rows below valid_len are defined
        let n = if shape.len() == 3 && shape[1] == g.ints("input_ids").len() {
            case.valid_len * shape[2]
        } else {
            want.len()
        };
        rows.push((name, max_abs_diff(&got[..n], &want[..n]), tol));
    }
    println!("case {}: valid_len {}", case.name, case.valid_len);
    for (name, diff, tol) in &rows {
        println!("  {name:<20} max|diff| {diff:.3e}  (tol {tol:.0e})");
    }
    rows
}

fn assert_parity(case_name: &str, rows: &[(String, f32, f32)]) {
    if let Some((name, diff, tol)) = rows.iter().find(|(_, d, t)| d > t) {
        let layer = name
            .rsplit_once('_')
            .filter(|_| name.contains("_layer_"))
            .map(|(_, i)| {
                format!(
                    " (first divergent layer index {})",
                    i.trim_start_matches('0').parse::<usize>().unwrap_or(0)
                )
            })
            .unwrap_or_default();
        panic!(
            "case {case_name}: tensor `{name}`{layer} differs from the golden by {diff:.3e}, tolerance {tol:.0e}"
        );
    }
}

fn check_case(name: &'static str) {
    assert!(Path::new(&data().join(name)).exists());
    let case = load_case(name);
    let ids = golden_ids(&case);
    let mut trace = Trace::new();
    run(&case, &ids, case.valid_len, &mut trace);
    let rows = compare(&case, &trace);
    assert_parity(case.name, &rows);
}

#[test]
fn short_matches_golden() {
    check_case("short");
}

#[test]
fn medium_matches_golden() {
    check_case("medium");
}

#[test]
fn padded_matches_golden() {
    check_case("padded");
}

#[test]
fn fastpath_off_matches_golden() {
    check_case("fastpath_off");
}

/// Padding must not change valid-token outputs: the padded sequence and the same ids cut to
/// their valid length give the same valid rows and outputs.
#[test]
fn padding_does_not_change_valid_rows() {
    let case = load_case("padded");
    let ids = golden_ids(&case);
    assert!(ids.len() > case.valid_len, "fixture must be padded");
    let (mut padded, mut cut) = (Trace::new(), Trace::new());
    run(&case, &ids, case.valid_len, &mut padded);
    run(&case, &ids[..case.valid_len], case.valid_len, &mut cut);
    let d = case.config.encoder.hidden_size;
    for name in [
        "embedding_output",
        "encoder_layer_00",
        "encoder_layer_02",
        "head_layer_00",
    ] {
        let (_, a) = padded.get(name).unwrap();
        let (_, b) = cut.get(name).unwrap();
        let diff = max_abs_diff(&a[..case.valid_len * d], &b[..case.valid_len * d]);
        assert!(diff <= 1e-5, "{name}: padded vs cut differ by {diff:.3e}");
    }
    for name in ["logits", "probabilities", "act_logits", "pooled"] {
        let diff = max_abs_diff(padded.get(name).unwrap().1, cut.get(name).unwrap().1);
        assert!(diff <= 1e-5, "{name}: padded vs cut differ by {diff:.3e}");
    }
}

#[test]
#[should_panic(expected = "case demo: tensor `encoder_layer_01` (first divergent layer index 1)")]
fn failure_names_case_tensor_and_layer() {
    let rows = vec![
        ("embedding_output".to_string(), 1e-7, LAYER_TOL),
        ("encoder_layer_00".to_string(), 1e-7, LAYER_TOL),
        ("encoder_layer_01".to_string(), 3e-3, LAYER_TOL),
        ("encoder_layer_02".to_string(), 4e-3, LAYER_TOL),
    ];
    assert_parity("demo", &rows);
}
