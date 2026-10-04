use std::cmp::max;
use std::collections::HashSet;
use std::path::Path;

use raya_core::{Error, Result};
use tokenizers::Tokenizer;

use crate::json::JsonValue;
use crate::question::{QType, Question, QuestionDef};

const MASK_TOKEN: &str = "[MASK]";
/// Laya cuts each option's tokens to this many before the head budget is applied.
const OPTION_TOKEN_CAP: usize = 48;

/// The document or conversation a request is about.
#[derive(Debug, Clone, PartialEq)]
pub enum State {
    Text(String),
    Json(JsonValue),
}

impl State {
    /// Laya serializes strings as they are and anything else as JSON text.
    #[must_use]
    pub fn serialized(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Json(v) => v.dumps(),
        }
    }

    /// Conversation lists keep their newest end when cut (`truncate_left`).
    #[must_use]
    pub fn truncate_left(&self) -> bool {
        matches!(self, Self::Json(JsonValue::Array(_)))
    }
}

/// What the head budget did to the options (`build_sequence(return_stats=True)`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stats {
    pub options: usize,
    pub options_distinct: usize,
    pub tokens_per_option: Option<usize>,
}

/// Facts about truncation, derived the way the laya:005 golden generator derives them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Trace {
    pub head_length: usize,
    pub state_budget: usize,
    pub state_tokens_total: usize,
    pub state_tokens_kept: usize,
    pub state_truncated: bool,
    pub sequence_clipped: bool,
}

/// One built sequence for one question.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Item {
    pub ids: Vec<u32>,
    pub markers: Vec<usize>,
    pub qtype: QType,
    pub stats: Stats,
    pub trace: Trace,
}

impl Item {
    /// Laya returns no position ids; the encoder derives `arange(len)`.
    #[must_use]
    pub fn position_ids(&self) -> Vec<u32> {
        (0..u32::try_from(self.ids.len()).expect("sequence length fits u32")).collect()
    }
}

pub struct SequenceBuilder {
    tok: Tokenizer,
    cls: u32,
    sep: u32,
    mask: u32,
    pad: u32,
}

impl SequenceBuilder {
    pub fn from_file(path: &Path) -> Result<Self> {
        let file_err = |reason: String| Error::Config {
            field: path.display().to_string(),
            reason,
        };
        let tok = Tokenizer::from_file(path).map_err(|e| file_err(e.to_string()))?;
        let id = |token: &str| {
            tok.token_to_id(token)
                .ok_or_else(|| file_err(format!("tokenizer has no `{token}` token")))
        };
        Ok(Self {
            cls: id("[CLS]")?,
            sep: id("[SEP]")?,
            mask: id(MASK_TOKEN)?,
            pad: id("[PAD]")?,
            tok,
        })
    }

    /// Reads `tokenizer.json` from a checkpoint directory.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        Self::from_file(&dir.join("tokenizer.json"))
    }

    #[must_use]
    pub fn pad_id(&self) -> u32 {
        self.pad
    }

    #[must_use]
    pub fn cls_id(&self) -> u32 {
        self.cls
    }

    #[must_use]
    pub fn sep_id(&self) -> u32 {
        self.sep
    }

    #[must_use]
    pub fn mask_id(&self) -> u32 {
        self.mask
    }

    #[must_use]
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }

    /// Never enables tokenizer-side truncation or padding, so `&self` stays read-only.
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.tok
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| Error::Config {
                field: "tokenizer".to_owned(),
                reason: e.to_string(),
            })
    }

    /// The state's token ids, computed once and reused for every question.
    pub fn state_ids(&self, state: &State) -> Result<Vec<u32>> {
        self.encode(&state.serialized().replace(MASK_TOKEN, " "))
    }

    /// `build_sequence(..., return_stats=True)`; does not run the lost-marker guard.
    pub fn build_sequence(
        &self,
        state_ids: &[u32],
        truncate_left: bool,
        q: &Question,
        max_len: usize,
        head_max_len: usize,
    ) -> Result<Item> {
        let ins = q.ins.replace(MASK_TOKEN, " ");
        let head = self.encode(&format!("{} question: {ins}", q.qtype.name()))?;
        let mut opt_ids: Vec<Vec<u32>> = Vec::new();
        for option in q.options() {
            let mut tokens = self.encode(&format!(" {}", option.replace(MASK_TOKEN, " ")))?;
            tokens.truncate(OPTION_TOKEN_CAP);
            let mut ids = vec![self.mask];
            ids.extend(tokens);
            opt_ids.push(ids);
        }
        // Signed on purpose: Python's budget and floor division go negative.
        let head_max = to_isize(head_max_len);
        let total = |opts: &[Vec<u32>]| to_isize(opts.iter().map(Vec::len).sum());
        let mut budget = head_max - total(&opt_ids);
        let mut tokens_per_option = None;
        if budget < 16 {
            let count = to_isize(max(1, opt_ids.len()));
            let per = usize::try_from(max(4, (head_max - 16).div_euclid(count)))
                .expect("per-option cap is at least 4");
            tokens_per_option = Some(per);
            for o in &mut opt_ids {
                o.truncate(per);
            }
            budget = head_max - total(&opt_ids);
        }
        let keep = usize::try_from(max(8, budget)).expect("head budget is at least 8");
        let mut ids = vec![self.cls];
        ids.extend(head.iter().take(keep));
        ids.push(self.sep);
        let mut markers = Vec::new();
        for o in &opt_ids {
            markers.push(ids.len());
            ids.extend(o);
        }
        ids.push(self.sep);

        let pre_state = ids.len();
        let room = max_len.saturating_sub(pre_state + 1);
        let state_part = if truncate_left {
            &state_ids[state_ids.len().saturating_sub(room)..]
        } else {
            &state_ids[..room.min(state_ids.len())]
        };
        ids.extend(state_part);
        ids.push(self.sep);
        ids.truncate(max_len);
        markers.retain(|m| *m < max_len);

        // Derived like the golden generator: the head is what `build_sequence` yields for an
        // empty state, minus its closing [SEP].
        let head_length = (pre_state + 1).min(max_len).saturating_sub(1);
        let state_budget = max_len.saturating_sub(head_length + 1);
        let kept = state_ids.len().min(state_budget);
        let distinct: HashSet<&Vec<u32>> = opt_ids.iter().collect();
        Ok(Item {
            ids,
            markers,
            qtype: q.qtype,
            stats: Stats {
                options: opt_ids.len(),
                options_distinct: distinct.len(),
                tokens_per_option,
            },
            trace: Trace {
                head_length,
                state_budget,
                state_tokens_total: state_ids.len(),
                state_tokens_kept: kept,
                state_truncated: kept < state_ids.len(),
                sequence_clipped: head_length + kept + 1 > max_len,
            },
        })
    }

    /// `Agent._encode_state`: tokenize the state once, build one item per question, and fail
    /// when the cuts lost option markers.
    pub fn encode_state(
        &self,
        state: &State,
        questions: &[(String, QuestionDef)],
        max_len: usize,
        head_max_len: usize,
    ) -> Result<Vec<Item>> {
        let state_ids = self.state_ids(state)?;
        let mut items = Vec::with_capacity(questions.len());
        for (id, def) in questions {
            let q = Question::from_def(id, def)?;
            let item =
                self.build_sequence(&state_ids, state.truncate_left(), &q, max_len, head_max_len)?;
            if item.markers.len() != q.options().len() {
                return Err(Error::Config {
                    field: "head_max_len".to_owned(),
                    reason: format!("question `{id}` options exceed head_max_len={head_max_len}"),
                });
            }
            items.push(item);
        }
        Ok(items)
    }
}

fn to_isize(n: usize) -> isize {
    isize::try_from(n).expect("length fits isize")
}
