use std::fmt::{self, Write};

use serde::de::{Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};

/// A JSON value that keeps object key order, so it can be written back the way Python's
/// `json.dumps` would write the same dict.
#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    Str(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

impl JsonValue {
    pub fn parse(text: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(text)
    }

    /// `json.dumps(value, ensure_ascii=False)` with Python's default separators.
    #[must_use]
    pub fn dumps(&self) -> String {
        let mut out = String::new();
        self.write(&mut out);
        out
    }

    fn write(&self, out: &mut String) {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Self::Int(i) => {
                let _ = write!(out, "{i}");
            }
            Self::UInt(u) => {
                let _ = write!(out, "{u}");
            }
            Self::Float(f) => out.push_str(&float_repr(*f)),
            Self::Str(s) => write_str(s, out),
            Self::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    item.write(out);
                }
                out.push(']');
            }
            Self::Object(pairs) => {
                out.push('{');
                for (i, (key, value)) in pairs.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    write_str(key, out);
                    out.push_str(": ");
                    value.write(out);
                }
                out.push('}');
            }
        }
    }
}

fn write_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python's `repr(float)`: shortest round-trip digits, scientific form when the decimal
/// exponent is below -4 or at least 16.
fn float_repr(x: f64) -> String {
    let sign = if x.is_sign_negative() { "-" } else { "" };
    if x == 0.0 {
        return format!("{sign}0.0");
    }
    let sci = format!("{:e}", x.abs());
    let (mantissa, exp) = sci.split_once('e').expect("`{:e}` always has an exponent");
    let exp: i32 = exp.parse().expect("`{:e}` exponent is an integer");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    if !(-4..16).contains(&exp) {
        let frac = &digits[1..];
        let dot = if frac.is_empty() { "" } else { "." };
        let esign = if exp < 0 { '-' } else { '+' };
        return format!("{sign}{}{dot}{frac}e{esign}{:02}", &digits[..1], exp.abs());
    }
    if exp >= 0 {
        let int_len = usize::try_from(exp).expect("non-negative") + 1;
        let (int_part, frac) = if digits.len() > int_len {
            (digits[..int_len].to_owned(), digits[int_len..].to_owned())
        } else {
            (format!("{digits:0<int_len$}"), "0".to_owned())
        };
        format!("{sign}{int_part}.{frac}")
    } else {
        let zeros = "0".repeat(usize::try_from(-exp - 1).expect("non-negative"));
        format!("{sign}0.{zeros}{digits}")
    }
}

impl<'de> Deserialize<'de> for JsonValue {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = JsonValue;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("any JSON value")
            }

            fn visit_unit<E>(self) -> Result<JsonValue, E> {
                Ok(JsonValue::Null)
            }

            fn visit_bool<E>(self, v: bool) -> Result<JsonValue, E> {
                Ok(JsonValue::Bool(v))
            }

            fn visit_i64<E>(self, v: i64) -> Result<JsonValue, E> {
                Ok(JsonValue::Int(v))
            }

            fn visit_u64<E>(self, v: u64) -> Result<JsonValue, E> {
                Ok(JsonValue::UInt(v))
            }

            fn visit_f64<E>(self, v: f64) -> Result<JsonValue, E> {
                Ok(JsonValue::Float(v))
            }

            fn visit_str<E>(self, v: &str) -> Result<JsonValue, E> {
                Ok(JsonValue::Str(v.to_owned()))
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<JsonValue, A::Error> {
                let mut items = Vec::new();
                while let Some(item) = seq.next_element()? {
                    items.push(item);
                }
                Ok(JsonValue::Array(items))
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<JsonValue, A::Error> {
                let mut pairs: Vec<(String, JsonValue)> = Vec::new();
                while let Some((key, value)) = map.next_entry::<String, JsonValue>()? {
                    // Python's dict keeps the first position and the last value.
                    match pairs.iter_mut().find(|(k, _)| *k == key) {
                        Some(slot) => slot.1 = value,
                        None => pairs.push((key, value)),
                    }
                }
                Ok(JsonValue::Object(pairs))
            }
        }
        d.deserialize_any(V)
    }
}
