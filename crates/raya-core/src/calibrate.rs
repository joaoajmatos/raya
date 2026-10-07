//! Temperature calibration, ported from `Agent._decode_answers` (the model's forward does not
//! use the temperature).
use crate::config::AgentConfig;
use crate::error::{Error, Result};
use crate::ops::softmax_in_place;

const TEMP_MIN: f32 = 0.5;
const TEMP_MAX: f32 = 5.0;
const QTYPE_NAMES: [&str; 3] = ["choice", "score", "noul"];

/// A usable temperature: confined to `[0.5, 5.0]`, 1.0 if not finite (`clamp_temperature`).
#[must_use]
pub fn clamp_temperature(t: f32) -> f32 {
    if t.is_finite() {
        t.clamp(TEMP_MIN, TEMP_MAX)
    } else {
        1.0
    }
}

/// Key into `temperature_by_options` (`temp_bucket`): `"<qtype>:<2|3-5|6-10|11+>"`.
pub fn temp_bucket(qtype: usize, k: usize) -> Result<String> {
    let name = QTYPE_NAMES
        .get(qtype)
        .ok_or_else(|| Error::Input(format!("qtype {qtype} is not 0, 1 or 2")))?;
    let size = match k {
        0..=2 => "2",
        3..=5 => "3-5",
        6..=10 => "6-10",
        _ => "11+",
    };
    Ok(format!("{name}:{size}"))
}

/// Calibrated option probabilities for one question: `softmax(logits[valid] / T)` over the `k`
/// markers `marker_mask` marks valid (Laya's `logits[:k]`, `k` = number of valid markers), with
/// `T = temperature_by_options[bucket(qtype, k)]`, else `temperature[qtype]`. Pass
/// `ForwardOutput::logits` and the mask the forward was given; invalid markers (`-1e4`) are
/// dropped and never counted in `k`.
pub fn calibrate(
    logits: &[f32],
    marker_mask: &[bool],
    qtype: usize,
    agent: &AgentConfig,
) -> Result<Vec<f32>> {
    if logits.len() != marker_mask.len() {
        return Err(Error::Input(format!(
            "{} logits but {} mask entries",
            logits.len(),
            marker_mask.len()
        )));
    }
    let logits: Vec<f32> = logits
        .iter()
        .zip(marker_mask)
        .filter_map(|(&l, &valid)| valid.then_some(l))
        .collect();
    if logits.is_empty() {
        return Err(Error::Input("no valid option logits to calibrate".into()));
    }
    let bucket = temp_bucket(qtype, logits.len())?;
    let t = match agent.temperature_by_options.get(&bucket) {
        Some(&t) => t,
        None => *agent.temperature.get(qtype).ok_or_else(|| {
            Error::Input(format!(
                "temperature has {} entries, qtype {qtype} needs one",
                agent.temperature.len()
            ))
        })?,
    };
    let t = clamp_temperature(t);
    let mut z: Vec<f32> = logits.iter().map(|l| l / t).collect();
    softmax_in_place(&mut z);
    Ok(z)
}
