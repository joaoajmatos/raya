# Contract: GEMM shapes of the Laya-shape model

`matmul_nt(out[m,n], a[m,k], b[n,k])` with `b` a Laya weight `[out, in]`; `m` is the number of rows (tokens). Model: hidden `d` = 1024, 16 heads of 64, gated MLP intermediate 2624 (`Wi` is `[2 * 2624, d]`), 28 encoder layers, 2 head layers over the full sequence.

Source of the numbers: `docs/feasibility.md` (section 1: "1024->5248, 2624->1024", 28 layers, hidden 1024, 16 heads), the vendored inventory `tests/data/inventory.json` for the structure (`Wqkv [3d, d]`, `Wo [d, d]`, `mlp.Wi [2i, d]`, `mlp.Wo [d, i]`, head `in_proj [3d, d]`, `out_proj [d, d]`, `linear1 [4d, d]` at hidden 64, `linear2 [d, 4d]`, `scorer.1 [d, d]`, `scorer.3 [1, d]`, `act_head.0 [256, d+4]`, `act_head.2 [2, 256]`). The head feed-forward width 4d at hidden 1024 and `act_head` input width are extrapolated from the fixture and confirmed by task T001 against the real checkpoint before any run.

| Name | Used by | n (out) | k (in) | count per forward | m |
|---|---|---|---|---|---|
| `enc_wqkv` | encoder attention input | 3072 | 1024 | 28 | tokens |
| `enc_wo` | encoder attention output | 1024 | 1024 | 28 | tokens |
| `enc_mlp_wi` | encoder gated MLP in | 5248 | 1024 | 28 | tokens |
| `enc_mlp_wo` | encoder gated MLP out | 1024 | 2624 | 28 | tokens |
| `head_in_proj` | head attention input | 3072 | 1024 | 2 | tokens |
| `head_out_proj` | head attention output | 1024 | 1024 | 2 | tokens |
| `head_linear1` | head feed-forward in | 4096 (to confirm) | 1024 | 2 | tokens |
| `head_linear2` | head feed-forward out | 1024 | 4096 (to confirm) | 2 | tokens |
| `scorer_1` | marker scorer | 1024 | 1024 | 1 | k_markers (2 to 6) |
| `act_0` | act head | 256 | 1028 (to confirm) | 1 | 1 |

`m` values for the token GEMMs: 512, 1024, 2048, 4096, 8192. Small-`m` GEMMs (`scorer_1`, `act_0`) are measured at their own `m` (1, 2, 3, 6) and are weighted by count in the budget but are expected to be negligible.

Tiny and odd shapes for the correctness gate only (not timed): `[m, n, k]` in `{(1,1,1), (3,5,7), (17,9,33), (64,64,64), (130,31,257)}` to cover tails of any blocking.

GEMM budget per length L and candidate c: `sum over rows of (count x p50(c, shape, L))`, in seconds. It is a sum of kernel times, not a forward time (spec FR-010).

FLOP count per token-row for the token GEMMs (`2 * n * k` summed, times count): encoder `28 * 2 * (3072*1024 + 1024*1024 + 5248*1024 + 1024*2624)` = about 0.686 GFLOP per token; head `2 * 2 * (3072*1024 + 1024*1024 + 4096*1024 + 1024*4096)` = about 0.050 GFLOP per token (head width to confirm). At 512 tokens that is about 0.38 TFLOP per forward and at 8192 about 6.0 TFLOP, arithmetic only; no timing is implied.
