use std::ops::Range;

/// Query/key block size of Laya's `block_local` pattern.
pub const BLOCK: usize = 128;
/// Half-width of the exact local window (`|i - j| <= WINDOW`).
pub const WINDOW: usize = 64;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum KernelError {
    #[error("invalid shape: {0}")]
    Shape(String),
    #[error("buffer `{name}` has length {got}, expected {expected}")]
    Length { name: &'static str, got: usize, expected: usize },
    #[error("unsupported: {0}")]
    Unsupported(String),
}

/// Which keys a query may attend (besides the `j < valid_len` rule).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mask {
    /// Laya's `block_local`: query block `i` sees key blocks `i-1`, `i`, `i+1`.
    Block3,
    /// Exact sliding window `|i - j| <= WINDOW`.
    Window64,
}

impl Mask {
    pub fn name(self) -> &'static str {
        match self {
            Mask::Block3 => "block3",
            Mask::Window64 => "window64",
        }
    }

    /// Keys the query at row `i` may attend, clipped to `valid_len`.
    pub fn key_range(self, i: usize, valid_len: usize) -> Range<usize> {
        let (lo, hi) = match self {
            Mask::Block3 => {
                let b = i / BLOCK;
                (b.saturating_sub(1) * BLOCK, (b + 2) * BLOCK)
            }
            Mask::Window64 => (i.saturating_sub(WINDOW), i + WINDOW + 1),
        };
        lo.min(valid_len)..hi.min(valid_len)
    }

    /// Whether query `i` may attend key `j`, ignoring `valid_len`.
    pub fn allows(self, i: usize, j: usize) -> bool {
        match self {
            Mask::Block3 => (i / BLOCK).abs_diff(j / BLOCK) <= 1,
            Mask::Window64 => i.abs_diff(j) <= WINDOW,
        }
    }
}

/// Dimensions of one attention call. Buffers are `[heads][seq][head_dim]`, row-major.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttnShape {
    pub heads: usize,
    pub seq: usize,
    pub head_dim: usize,
}

impl AttnShape {
    pub fn numel(self) -> usize {
        self.heads * self.seq * self.head_dim
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttnParams {
    pub mask: Mask,
    pub block: usize,
    /// Keys at or beyond this position are padding; query rows there output zeros.
    pub valid_len: usize,
    pub scale: f32,
}

impl AttnParams {
    /// Default parameters: block 128 and scale `1/sqrt(head_dim)`.
    #[allow(clippy::cast_precision_loss)]
    pub fn new(mask: Mask, valid_len: usize, head_dim: usize) -> Self {
        Self { mask, block: BLOCK, valid_len, scale: 1.0 / (head_dim as f32).sqrt() }
    }
}

/// Single entry point for attention so kernels can be swapped without touching model code.
///
/// `q`, `k`, `v` and `out` are `[heads][seq][head_dim]` row-major f32. For `i < valid_len`,
/// `out[h][i]` is the softmax-weighted sum of `v[h][j]` over `j < valid_len` allowed by
/// `params.mask`; other rows are zero. `out` is fully overwritten.
pub trait AttentionKernel: Send + Sync {
    fn name(&self) -> &'static str;

    fn attend(
        &self,
        shape: AttnShape,
        params: AttnParams,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &mut [f32],
    ) -> Result<(), KernelError>;
}

/// Shared argument checks; kernels return errors instead of panicking on bad input.
pub fn validate(
    shape: AttnShape,
    params: &AttnParams,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &[f32],
) -> Result<(), KernelError> {
    if shape.heads == 0 || shape.head_dim == 0 {
        return Err(KernelError::Shape("heads and head_dim must be at least 1".into()));
    }
    if shape.head_dim % 8 != 0 {
        return Err(KernelError::Shape(format!("head_dim {} is not a multiple of 8", shape.head_dim)));
    }
    if params.valid_len > shape.seq {
        return Err(KernelError::Shape(format!(
            "valid_len {} exceeds seq {}",
            params.valid_len, shape.seq
        )));
    }
    if params.block != BLOCK {
        return Err(KernelError::Unsupported(format!("block {} (only {BLOCK} is supported)", params.block)));
    }
    let expected = shape.numel();
    for (name, buf) in [("q", q), ("k", k), ("v", v), ("out", out)] {
        if buf.len() != expected {
            return Err(KernelError::Length { name, got: buf.len(), expected });
        }
    }
    Ok(())
}
