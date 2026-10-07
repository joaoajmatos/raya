# Data Model: Fused Block-Local Attention Kernel

No persisted entities except bench output. Types live in `raya-kernels`.

| Type | Fields | Rules |
|---|---|---|
| `AttnShape` | `heads`, `seq`, `head_dim` | `seq` in `0..=8192` allowed by the spike; `head_dim` multiple of 8 (validated), `heads >= 1` |
| `Mask` | `Block3`, `Window64` | `Block3`: `|i/128 - j/128| <= 1`; `Window64`: `|i-j| <= 64` |
| `AttnParams` | `mask`, `block` (128), `valid_len`, `scale` | `valid_len <= seq`; `block = 128` is the only value tested; other values return `KernelError::Unsupported` |
| `AttentionKernel` (trait) | `name()`, `attend(&self, shape, params, q, k, v, out)` | `q`, `k`, `v`, `out` are `[heads][seq][head_dim]` row-major f32, each of length `heads*seq*head_dim`; `out` fully overwritten |
| `KernelError` | `Shape`, `Length`, `Unsupported` | returned instead of panicking (constitution VII) |
| `Naive` | none | single-threaded scalar; oracle and speed baseline |
| `Tiled` | `threads`, `backend`, `mr`, `nc` | `backend` is `Portable`, `Avx2` or `Neon`; unavailable backend falls back to `Portable` at construction, never at run time |
| `StageTimes` (feature `stage-timing`) | `qk_ns`, `softmax_ns`, `pv_ns`, `tiles` | summed over threads |
| `BenchMetadata` | `cpu_model`, `logical_cpus`, `physical_cpus`, `threads`, `features`, `rustflags`, `rustc`, `git_commit`, `timestamp`, `label` | `label` is "target machine" when `RAYA_TARGET_MACHINE=1`, "cloud-agent hardware, indicative only" when `RAYA_TARGET_MACHINE=cloud-agent`, otherwise "unlabeled machine, indicative only" (constitution 1.2.0; originally the cloud label was the default, which made a hardware claim the code cannot know) |

## State and invariants

- Rows `i >= valid_len` of `out` are zero.
- Every row `i < valid_len` sums its softmax weights over at most 129 keys (`Window64`) or 384 keys (`Block3`), intersected with `j < valid_len`.
- No buffer larger than one micro-tile of scores is allocated per worker.
