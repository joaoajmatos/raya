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

/// Calibrated option probabilities for the `k = logits.len()` valid markers of one question:
/// `softmax(logits / T)` with `T = temperature_by_options[bucket]`, else `temperature[qtype]`.
pub fn calibrate(logits: &[f32], qtype: usize, agent: &AgentConfig) -> Result<Vec<f32>> {
    if logits.is_empty() {
        return Err(Error::Input("no option logits to calibrate".into()));
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
