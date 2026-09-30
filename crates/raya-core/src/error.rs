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
