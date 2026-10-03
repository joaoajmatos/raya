use raya_core::{Error, Result};
use serde::Deserialize;

use crate::json::JsonValue;

/// Question kinds, with Laya's `QTYPES` ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QType {
    Choice = 0,
    Score = 1,
    Noul = 2,
}

impl QType {
    #[must_use]
    pub fn id(self) -> u8 {
        self as u8
    }

    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Choice => "choice",
            Self::Score => "score",
            Self::Noul => "noul",
        }
    }
}

/// A question as a caller writes it (Laya's public `type`/`instructions`/`criteria`/`labels`).
#[derive(Debug, Clone, Deserialize)]
pub struct QuestionDef {
    #[serde(rename = "type")]
    pub kind: String,
    pub instructions: JsonValue,
    #[serde(default)]
    pub criteria: Option<JsonValue>,
    #[serde(default)]
    pub labels: Option<JsonValue>,
}

impl QuestionDef {
    /// Parses one question from JSON text; a missing or ill-typed field is an `Error::Config`
    /// naming the question instead of a bare serde error at the caller.
    pub fn from_json(id: &str, text: &str) -> Result<Self> {
        serde_json::from_str(text).map_err(|e| bad(id, "", e.to_string()))
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Crit {
    /// Choice given as a list of labels: no descriptions.
    Labels(Vec<JsonValue>),
    /// Choice given as `{label: description}`.
    Described(Vec<(String, JsonValue)>),
    Levels(Vec<JsonValue>),
    Noul {
        false_desc: Option<JsonValue>,
        true_desc: Option<JsonValue>,
    },
}

/// Laya's normalized question (`Agent._to_internal`).
#[derive(Debug, Clone, PartialEq)]
pub struct Question {
    pub qtype: QType,
    pub ins: String,
    crit: Crit,
    noul_labels: (String, String),
}

fn bad(id: &str, key: &str, reason: impl Into<String>) -> Error {
    Error::Config {
        field: if key.is_empty() {
            format!("questions.{id}")
        } else {
            format!("questions.{id}.{key}")
        },
        reason: reason.into(),
    }
}

fn is_blank(v: &JsonValue) -> bool {
    matches!(v, JsonValue::Null) || matches!(v, JsonValue::Str(s) if s.is_empty())
}

/// `render_criterion`: strings pass through, anything else becomes compact JSON text.
fn render_criterion(v: &JsonValue) -> String {
    match v {
        JsonValue::Str(s) => s.clone(),
        other => other.dumps(),
    }
}

/// Python `str(label)` for the label types supported here.
fn label_text(id: &str, i: usize, v: &JsonValue) -> Result<String> {
    match v {
        JsonValue::Str(s) => Ok(s.clone()),
        JsonValue::Int(n) => Ok(n.to_string()),
        JsonValue::UInt(n) => Ok(n.to_string()),
        JsonValue::Bool(true) => Ok("True".to_owned()),
        JsonValue::Bool(false) => Ok("False".to_owned()),
        JsonValue::Float(_) => Err(Error::Unimplemented(format!(
            "question `{id}`: choice label {i} is a float (Python `str(float)` parity is not ported)"
        ))),
        other => Err(bad(
            id,
            "criteria",
            format!(
                "choice label {i} must be a string, number or bool, got {}",
                other.dumps()
            ),
        )),
    }
}

/// Python dict-key equality for the supported label types: `True == 1` and `False == 0`.
fn same_label(a: &JsonValue, b: &JsonValue) -> bool {
    let number = |v: &JsonValue| match v {
        JsonValue::Bool(x) => Some(i128::from(*x)),
        JsonValue::Int(x) => Some(i128::from(*x)),
        JsonValue::UInt(x) => Some(i128::from(*x)),
        _ => None,
    };
    match (a, b) {
        (JsonValue::Str(x), JsonValue::Str(y)) => x == y,
        _ => number(a).is_some() && number(a) == number(b),
    }
}

fn noul_labels(id: &str, labels: Option<&JsonValue>) -> Result<(String, String)> {
    let err = || {
        bad(
            id,
            "labels",
            "noul labels must map exactly 'false' and 'true' to distinct non-empty strings",
        )
    };
    let Some(labels) = labels else {
        return Ok(("false".to_owned(), "true".to_owned()));
    };
    let JsonValue::Object(pairs) = labels else {
        return Err(err());
    };
    let get = |key: &str| pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v);
    let (Some(JsonValue::Str(f)), Some(JsonValue::Str(t))) = (get("false"), get("true")) else {
        return Err(err());
    };
    if pairs.len() != 2 {
        return Err(err());
    }
    let (f, t) = (f.trim(), t.trim());
    if f.is_empty() || t.is_empty() || f == t {
        return Err(err());
    }
    Ok((f.to_owned(), t.to_owned()))
}

impl Question {
    /// Validates only what rendering needs, then normalizes like `Agent._to_internal`.
    pub fn from_def(id: &str, def: &QuestionDef) -> Result<Self> {
        let qtype = match def.kind.as_str() {
            "choice" => QType::Choice,
            "score" => QType::Score,
            "noul" => QType::Noul,
            other => {
                return Err(bad(
                    id,
                    "type",
                    format!("`{other}` is not one of choice, score, noul"),
                ))
            }
        };
        if def.labels.is_some() && qtype != QType::Noul {
            return Err(bad(id, "labels", "only supported for noul questions"));
        }
        let ins = match &def.instructions {
            JsonValue::Str(s) => s.clone(),
            other => other.dumps(),
        };
        let crit = match (qtype, def.criteria.as_ref()) {
            (QType::Choice, Some(JsonValue::Array(items))) if !items.is_empty() => {
                for (i, item) in items.iter().enumerate() {
                    label_text(id, i, item)?;
                    if let Some(first) = items[..i].iter().position(|p| same_label(p, item)) {
                        return Err(bad(
                            id,
                            "criteria",
                            format!("choice label {i} repeats label {first}"),
                        ));
                    }
                }
                Crit::Labels(items.clone())
            }
            (QType::Choice, Some(JsonValue::Object(pairs))) if !pairs.is_empty() => {
                Crit::Described(pairs.clone())
            }
            (QType::Choice, _) => {
                return Err(bad(
                    id,
                    "criteria",
                    "a choice question needs a non-empty list or object of criteria",
                ))
            }
            (QType::Score, Some(JsonValue::Array(items))) if !items.is_empty() => {
                if let Some(i) = items.iter().position(|v| matches!(v, JsonValue::Null)) {
                    return Err(bad(id, "criteria", format!("score level {i} is null")));
                }
                Crit::Levels(items.clone())
            }
            (QType::Score, _) => {
                return Err(bad(
                    id,
                    "criteria",
                    "a score question takes a non-empty list of level descriptions",
                ))
            }
            (QType::Noul, None | Some(JsonValue::Null)) => Crit::Noul {
                false_desc: None,
                true_desc: None,
            },
            (QType::Noul, Some(JsonValue::Object(pairs))) => {
                let mut false_desc = None;
                let mut true_desc = None;
                for (k, v) in pairs {
                    match k.to_lowercase().as_str() {
                        "false" => false_desc = Some(v.clone()),
                        "true" => true_desc = Some(v.clone()),
                        _ => {
                            return Err(bad(
                                id,
                                "criteria",
                                format!("noul criteria are keyed only 'true'/'false', got `{k}`"),
                            ))
                        }
                    }
                }
                Crit::Noul {
                    false_desc,
                    true_desc,
                }
            }
            (QType::Noul, Some(_)) => {
                return Err(bad(
                    id,
                    "criteria",
                    "a noul question takes criteria as an object keyed 'true'/'false', or none",
                ))
            }
        };
        Ok(Self {
            qtype,
            ins,
            crit,
            noul_labels: noul_labels(id, def.labels.as_ref())?,
        })
    }

    /// Option texts in label-index order (`render_options`).
    #[must_use]
    pub fn options(&self) -> Vec<String> {
        match &self.crit {
            Crit::Labels(items) => items
                .iter()
                .enumerate()
                .map(|(i, v)| label_text("", i, v).expect("labels checked in from_def"))
                .collect(),
            Crit::Described(pairs) => pairs
                .iter()
                .map(|(k, v)| {
                    if is_blank(v) {
                        k.clone()
                    } else {
                        format!("{k}: {}", render_criterion(v))
                    }
                })
                .collect(),
            Crit::Levels(items) => items
                .iter()
                .enumerate()
                .map(|(i, c)| format!("level {i}: {}", render_criterion(c)))
                .collect(),
            Crit::Noul {
                false_desc,
                true_desc,
            } => {
                let text = |desc: &Option<JsonValue>, default: &str| match desc {
                    Some(d) if !is_blank(d) => render_criterion(d),
                    _ => default.to_owned(),
                };
                vec![
                    format!(
                        "{}: {}",
                        self.noul_labels.0,
                        text(false_desc, "no, the statement does not hold")
                    ),
                    format!(
                        "{}: {}",
                        self.noul_labels.1,
                        text(true_desc, "yes, the statement holds")
                    ),
                ]
            }
        }
    }
}
