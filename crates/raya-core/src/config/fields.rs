use serde_json::Value;

use crate::error::{Error, Result};

#[derive(Clone, Copy)]
pub(super) enum Kind {
    Uint,
    Float,
    Bool,
    Str,
    FloatList,
    FloatMap,
}

impl Kind {
    fn accepts(self, v: &Value) -> bool {
        match self {
            Kind::Uint => v.is_u64(),
            Kind::Float => v.is_number(),
            Kind::Bool => v.is_boolean(),
            Kind::Str => v.is_string(),
            Kind::FloatList => v.as_array().is_some_and(|a| a.iter().all(Value::is_number)),
            Kind::FloatMap => v
                .as_object()
                .is_some_and(|m| m.values().all(Value::is_number)),
        }
    }

    fn expected(self) -> &'static str {
        match self {
            Kind::Uint => "a non-negative integer",
            Kind::Float => "a number",
            Kind::Bool => "a boolean",
            Kind::Str => "a string",
            Kind::FloatList => "a list of numbers",
            Kind::FloatMap => "an object with numeric values",
        }
    }
}

/// serde's own type errors do not say which key was wrong, so check the keys we read first.
/// Absent keys are left to the required/default rules; `null` counts as absent, as in Python.
pub(super) fn check_types(value: &Value, fields: &[(&str, Kind)]) -> Result<()> {
    let Some(map) = value.as_object() else {
        return Err(Error::config("<root>", "expected a JSON object"));
    };
    for (key, kind) in fields {
        if let Some(v) = map.get(*key).filter(|v| !v.is_null()) {
            if !kind.accepts(v) {
                return Err(Error::config(
                    key,
                    format!("expected {}, found {v}", kind.expected()),
                ));
            }
        }
    }
    Ok(())
}
