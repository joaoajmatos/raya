use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// `field` is the JSON key, so a bad config points at the line to fix.
    #[error("invalid config: `{field}`: {reason}")]
    Config { field: String, reason: String },
    #[error("cannot read {}: {source}", path.display())]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("cannot parse {}: {source}", path.display())]
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("not implemented: {0}")]
    Unimplemented(String),
    /// A forward-pass input outside what the model accepts.
    #[error("invalid input: {0}")]
    Input(String),
    /// The attention kernel rejected or failed a call.
    #[error("attention kernel: {0}")]
    Kernel(String),
    #[error("{0}")]
    Weights(Box<LoadReport>),
}

/// A tensor whose shape in the file differs from the one the config implies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeMismatch {
    pub name: String,
    pub expected: Vec<usize>,
    pub found: Vec<usize>,
}

/// Every way a weights file disagreed with the model config, reported at once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadReport {
    pub path: PathBuf,
    pub missing: Vec<String>,
    pub unused: Vec<String>,
    /// Names that appear more than once in the file header.
    pub duplicate: Vec<String>,
    pub shape: Vec<ShapeMismatch>,
    /// Set when the file could not be parsed at all.
    pub detail: Option<String>,
}

impl std::fmt::Display for LoadReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid weights file {}", self.path.display())?;
        if let Some(detail) = &self.detail {
            write!(f, ": {detail}")?;
        }
        for (title, names) in [
            ("missing tensors", &self.missing),
            ("unexpected tensors", &self.unused),
            ("duplicate tensors", &self.duplicate),
        ] {
            if names.is_empty() {
                continue;
            }
            write!(f, "\n  {title}:")?;
            let mut sorted: Vec<_> = names.iter().collect();
            sorted.sort();
            for name in sorted {
                write!(f, "\n    {name}")?;
            }
        }
        if !self.shape.is_empty() {
            write!(f, "\n  shape mismatches:")?;
            let mut shapes: Vec<_> = self.shape.iter().collect();
            shapes.sort_by(|a, b| a.name.cmp(&b.name));
            for m in shapes {
                write!(
                    f,
                    "\n    {}: expected {:?}, found {:?}",
                    m.name, m.expected, m.found
                )?;
            }
        }
        Ok(())
    }
}

impl Error {
    pub(crate) fn config(field: &str, reason: impl Into<String>) -> Self {
        Self::Config {
            field: field.to_owned(),
            reason: reason.into(),
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_lists_each_group_sorted() {
        let report = LoadReport {
            path: "m.safetensors".into(),
            missing: vec!["b".into(), "a".into()],
            unused: vec!["x".into()],
            duplicate: vec!["d".into()],
            shape: vec![ShapeMismatch {
                name: "w".into(),
                expected: vec![2, 3],
                found: vec![3, 2],
            }],
            detail: None,
        };
        assert_eq!(
            Error::Weights(Box::new(report)).to_string(),
            "invalid weights file m.safetensors\n  missing tensors:\n    a\n    b\n  \
             unexpected tensors:\n    x\n  duplicate tensors:\n    d\n  shape mismatches:\n    w: expected [2, 3], found [3, 2]"
        );
    }

    #[test]
    fn empty_groups_are_omitted() {
        let report = LoadReport {
            path: "m".into(),
            missing: vec![],
            unused: vec![],
            duplicate: vec![],
            shape: vec![],
            detail: Some("bad header".into()),
        };
        assert_eq!(report.to_string(), "invalid weights file m: bad header");
    }
}
