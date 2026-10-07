//! `raya predict`: tokenize, run the reference forward pass, print calibrated probabilities.
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use raya_core::ops::softmax_in_place;
use raya_core::{calibrate, ForwardInput, Model};
use raya_kernels::Naive;
use raya_tokenizer::{Question, QuestionDef, SequenceBuilder, State};
use serde_json::{json, Value};

pub struct Args {
    pub model: PathBuf,
    pub document: PathBuf,
    /// A question definition as JSON text, or the path of a file holding it.
    pub question: String,
    pub tokenizer: Option<PathBuf>,
}

/// A float as the shortest decimal that round-trips the f32, so nothing is rounded to 4 places
/// (Laya's `predict_batch` does) and no f32-to-f64 noise digits appear.
fn number(x: f32) -> Value {
    format!("{x}")
        .parse::<f64>()
        .ok()
        .and_then(serde_json::Number::from_f64)
        .map_or(Value::Null, Value::Number)
}

fn question_text(arg: &str) -> Result<String> {
    if arg.trim_start().starts_with('{') {
        return Ok(arg.to_owned());
    }
    std::fs::read_to_string(Path::new(arg))
        .with_context(|| format!("--question is neither JSON nor a readable file: {arg}"))
}

pub fn run(args: &Args) -> Result<String> {
    let (config, store) = raya_weights::load_dir(&args.model)
        .with_context(|| format!("cannot load model {}", args.model.display()))?;
    let builder = match &args.tokenizer {
        Some(path) => SequenceBuilder::from_file(path),
        None => SequenceBuilder::from_dir(&args.model),
    }
    .context(
        "cannot load the tokenizer (put tokenizer.json in the model directory or pass --tokenizer)",
    )?;

    let document = std::fs::read_to_string(&args.document)
        .with_context(|| format!("cannot read {}", args.document.display()))?;
    let def = QuestionDef::from_json("question", &question_text(&args.question)?)?;
    let question = Question::from_def("question", &def)?;
    let items = builder.encode_state(
        &State::Text(document),
        &[("question".to_owned(), def)],
        config.agent.max_len,
        config.agent.head_max_len,
    )?;
    let Some(item) = items.first() else {
        bail!("the sequence builder returned no item");
    };

    let model = Model::new(config.clone())?;
    let net = model.bind(&store)?;
    let qtype = usize::from(item.qtype.id());
    let marker_mask = vec![true; item.markers.len()];
    let out = net.forward(
        &ForwardInput {
            input_ids: &item.ids,
            valid_len: item.ids.len(),
            marker_pos: &item.markers,
            marker_mask: &marker_mask,
            qtype,
        },
        &Naive,
        None,
    )?;
    let probabilities = calibrate(&out.logits, &marker_mask, qtype, &config.agent)?;
    let mut act = out.act_logits.clone();
    softmax_in_place(&mut act);

    Ok(json!({
        "type": item.qtype.name(),
        "options": question.options(),
        "probabilities": probabilities.iter().copied().map(number).collect::<Vec<_>>(),
        "act_probability": number(act[0]),
    })
    .to_string())
}
