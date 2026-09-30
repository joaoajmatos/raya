# Contract: `AttentionKernel`

```rust
pub trait AttentionKernel: Send + Sync {
    fn name(&self) -> &'static str;
    fn attend(&self, shape: AttnShape, params: AttnParams,
              q: &[f32], k: &[f32], v: &[f32], out: &mut [f32]) -> Result<(), KernelError>;
}
```

- Layout: `[heads][seq][head_dim]`, row-major, contiguous f32, identical for `q`, `k`, `v`, `out`.
- Semantics: for `i < valid_len`, `out[h][i] = sum_j softmax(scale * q_i . k_j) v_j` over `j` with `|i-j| <= window` and `j < valid_len`; `out[h][i] = 0` for `i >= valid_len`.
- Errors: wrong buffer length, `head_dim % 8 != 0`, `valid_len > seq`, unsupported `window`/`block` return `KernelError`; never panic on bad shapes.
- Numerics: stable softmax (subtract running max); result never NaN for finite inputs, including logits up to +/-1e4; agreement with `Naive` within 1e-5 absolute.
- Feature 006 adopts this signature unchanged; it adapts activation layout on its side.
- Note: this is the only change to the trait introduced by 007; `name()` is unchanged from the current stub.

Construction: `Naive`, `Tiled::new(TiledConfig { threads: Option<usize>, backend: Backend })`. `threads: None` reads `RAYA_THREADS`, then `available_parallelism`.
