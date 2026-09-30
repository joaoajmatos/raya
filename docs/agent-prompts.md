# Raya scaffolding: prompts for a cloud agent

Two repos are involved:

- **raya** (https://github.com/joaoajmatos/raya.git): all Rust code, the crates at the repo root. Prompts
  P1, P3-P7 run here.
- **laya-sparse** (this repo): Python-side work only. P2 (golden fixtures) and the compare command of P7
  run here. Run these with an agent on laya-sparse, then vendor the output into raya (see "Bridging").

Scope: build only what does not depend on Phase 2 results. Raya-L means "Laya's exact shape, inference
only, loads Laya's safetensors". Nothing here implements compression or a new architecture.

Run order (P = can run in parallel):

```
P0 preamble (prepend to every prompt)
P1 workspace + CI        ─┐
P2 golden fixtures (py)  ─┼─> P3 weights loader ─┐
P6 block-local kernel    ─┘   P4 tokenizer/seq  ─┼─> P5 reference forward ─> P7 bench harness
```

P1, P2 and P6 are independent. P3 and P4 need P1 (and P2's fixtures for their tests). P5 needs P2, P3, P4.
P7 needs P5 and P6. One branch and one PR per prompt; merge before starting its dependents.

## Bridging the two repos

laya-sparse is a git submodule of raya at `reference/laya-sparse/`, pinned to a commit on `phase2-tooling`
(public repo). Agents must run `git submodule update --init` first; if `reference/laya-sparse/` is empty,
stop and say so. To move to newer Laya: in laya-sparse push the change, then in raya
`git submodule update --remote reference/laya-sparse` and commit the new pointer. Never edit files under
`reference/`.

`docs/feasibility.md` lives in raya (source of truth for scope). Fixtures from P2 are generated in
laya-sparse and copied into raya `tests/data/` (with the laya-sparse SHA in each meta.json); re-run P2 and
re-copy when Laya behaviour changes. Never edit copied fixtures by hand.

---

## P0. Preamble (prepend to every prompt)

```
You are working in the `raya` repo: a Rust CPU inference runtime for Laya-style encoder-only decision
models (Laya lives in a separate research repo, joaoajmatos/laya-sparse, checked out as the read-only git
submodule reference/laya-sparse/). First run `git submodule update --init`. Before changing anything, read
README.md and docs/feasibility.md.
docs/feasibility.md is the source of truth for scope. If a file the task names is missing, stop and say so.

Rules:
- Laya's behaviour (reference/laya-sparse/laya/) is the spec: match it, do not improve it. Never edit reference/ or vendored fixtures.
- Claims need measurements. Any number you produce on your own machine is NOT a measurement of the
  target CPU: label it "cloud-agent hardware, indicative only" and never compare it to numbers in
  the feasibility doc or Laya's result directories. Mark unmeasured statements as hypotheses.
- Do not implement compression, architecture variants, or int8/fp16/bf16 paths. Raya-L = Laya's shape,
  fp32, inference only.
- Do not choose a SIMD target for the project; keep a portable path and make others optional.
- Add tests with every change. Offline tests only (no network, no real checkpoint) unless marked slow.
- Keep comments sparse and explain why, not what. Match surrounding style.
- Work on a branch named as given in the task, open one PR, and summarize what is done, what is not,
  and any divergence from these instructions. If something is ambiguous and blocks you, stop and
  write the question in the PR description instead of guessing.
```

---

## P1. Rust workspace, CI, config types

Repo: raya. Branch: `p1-workspace`

```
Create a Cargo workspace at the repo root. Crates:
- raya-core     config types, errors, the model struct shell (no forward logic yet)
- raya-kernels  attention and GEMM kernels (empty modules + traits for now)
- raya-weights  safetensors loading (empty for now)
- raya-tokenizer wraps the Hugging Face `tokenizers` crate (empty for now)
- raya-cli      binary `raya` with subcommands stubbed: `info`, `predict`, `bench`

Requirements:
1. Rust stable, edition 2021, `rust-toolchain.toml` pinned, workspace-level lints (clippy pedantic
   allowed where noisy, `#![forbid(unsafe_code)]` everywhere except raya-kernels).
2. raya-core config: parse Laya's model config (look at reference/laya-sparse/laya/ and, once P2 fixtures exist,
   tests/data/ for the fields: layers, hidden size, heads, intermediate size,
   global-attention layer pattern, local window, RoPE thetas, norm eps, head layers, marker tokens).
   Also define the future-facing shape from docs/feasibility.md section 4.2 as types:
   `Segment { layers, attention: AttentionKind }`, `AttentionKind::{Full, BlockLocal{w}, Gas{s}, Hybrid}`,
   and `Compression` as an enum with a single variant `None`. Anything else returns
   `Error::Unimplemented`. Laya's shape must round-trip into a single segment with no compression.
3. GitHub Actions: fmt, clippy -D warnings, test on ubuntu-latest (x86_64) and macos-14 (arm64),
   cargo-deny or cargo-audit. Cache dependencies.
4. `README.md`: what this is, the scope limits from the preamble, how to build and test.
5. Unit tests for config parsing using a config JSON checked in under tests/data/.

Do not add model math. Do not add dependencies beyond: serde, serde_json, thiserror, safetensors,
tokenizers, clap, criterion (dev), anyhow (cli only).
```

---

## P2. Golden fixtures from Laya (Python)

Repo: **laya-sparse** (not raya). Branch: `raya/p2-golden-fixtures`. Output goes to `artifacts/raya-golden/` and is
copied into raya `tests/data/` by hand.

```
Add `experiments/golden/` (Python, runs with `python -m experiments golden ...`) that exports the
contract Raya's Rust tests will verify against. Use the existing tiny fixture model from
tests/experiments/ so everything runs offline and deterministically (fixed seeds, fp32, eval mode,
single thread for determinism).

Deliverables:
1. `golden weights-inventory`: dump every state_dict key with shape and dtype to
   artifacts/raya-golden/inventory.json, for the fixture model AND (if a real checkpoint is present locally,
   behind a flag and not in offline tests) the real one. This is the source for the Rust key mapping.
2. `golden export --out artifacts/raya-golden/golden/<case>/`: for each case save, as safetensors files:
   input_ids, attention inputs, embedding output, the output of EVERY encoder layer, final encoder
   output, decision-head per-layer outputs, final logits/probabilities, and the raw weights of the
   fixture model. Also a meta.json with the config, seed, torch version, and the git commit.
   Cases: short (fits one local window), medium (several blocks, crosses global/local layer
   boundaries), and a padded batch-of-one case. Also one case where the decision head's PyTorch fast
   path is forced off, and assert outputs equal within 1e-6 (this is known behaviour, see
   docs/raya-feasibility.md 1.5).
3. `golden kernels`: for the block-local (block 128, +/-64 window) and gather-attend-scatter kernels in
   experiments/kernels/, export random q/k/v inputs, the native-layer reference output, and the
   kernel output, for seq lengths 256, 1000 (non-multiple of block), 2048, with 16 heads, head dim 64.
   Record the max abs diff between native and kernel and fail if it exceeds 1e-5.
6. Offline pytest in tests/experiments/ that runs export into a tmp dir and validates file presence,
   shapes, and that re-running gives byte-identical output.
7. Keep fixtures small: commit only the tiny-model cases (< 5 MB total); everything else git-ignored.
```

---

## P3. Weights loader

Repo: raya. Branch: `p3-weights` (needs P1, P2 merged)

```
Implement raya-weights:
1. Memory-mapped safetensors loading (single file and sharded index), fp32 only; bf16/fp16 tensors are
   converted to fp32 on load, never kept.
2. Key mapping from Laya's state_dict names to Raya's internal names, driven by
   tests/data/inventory.json. Mapping lives in one table in code, with a test that every
   inventory key is mapped exactly once and every Raya parameter is covered (no silent missing or
   unused tensors: loading must fail loudly with a report of both lists).
3. Shape validation against raya-core config; clear errors naming the tensor.
4. A `WeightStore` API exposing tensors by internal name as `&[f32]` + shape (no copies for fp32),
   plus a helper that transposes or repacks GEMM weights once at load time (make the packing a
   pluggable function; default is row-major no-op).
5. `raya info <checkpoint-dir>`: prints config, parameter count computed from the loaded tensors
   (not quoted from docs), and total bytes.
6. Tests: load the fixture-model weights from tests/data/golden/*/ and compare a few tensors
   bit-for-bit against the file; negative tests (missing key, wrong shape, extra key).

No model math here.
```

---

## P4. Tokenizer and sequence builder

Repo: raya. Branch: `p4-tokenizer` (needs P4b output vendored) (needs P1, P2 merged)

```
Implement raya-tokenizer: a Rust port of the sequence builder in Laya (read reference/laya-sparse/laya/agent.py and
reference/laya-sparse/laya/common.py for the exact marker-token layout, state handling, question types choice/score/noul,
truncation and max_len behaviour, padding).

1. Load tokenizer.json via the `tokenizers` crate.
2. Build input_ids, position ids and the attention layout for: one document + one question (the
   primary workload), one document + several questions, and state. Match Laya exactly, including what
   happens when the document exceeds max_len.
3. Goldens come from P4b (laya-sparse); vendor them into tests/data/tokenizer/. They contain tokenizer outputs (ids, marker positions, truncation flags) for a fixed list of texts, including
   empty document, unicode, very long text, and text containing marker-like strings.
4. Rust tests compare ids and marker positions against those golden files exactly.
5. Report in the PR any place where Laya's behaviour looks accidental or underspecified. Do not
   "fix" it in Rust; parity with Laya is the goal.
```

---

## P4b. Tokenizer goldens (run in laya-sparse, before P4's tests)

Repo: **laya-sparse**. Branch: `raya/p4b-tokenizer-goldens`. Needs P2's `experiments/golden/` module.

```
Extend `experiments/golden/` with `python -m experiments golden tokenizer --out artifacts/raya-golden/tokenizer/`.
Use the tokenizer of the tiny fixture model if it has one; otherwise the real tokenizer.json only when a
local checkpoint is given by flag (never in offline tests). Read laya/agent.py and laya/common.py for the
real sequence builder and call it, do not reimplement it.

For each fixed input save a JSON record: document text, questions (with types choice/score/noul), state,
max_len, resulting input_ids, position ids, marker positions, truncation flags. Inputs: empty document;
one document + one question (primary workload); one document + several questions; with state; unicode;
text containing marker-like strings; a document far longer than max_len; a document exactly max_len.
Also save the tokenizer.json used and a meta.json with the laya-sparse commit.
Offline pytest: re-running gives byte-identical output. Do not change any existing Laya behaviour; if
behaviour looks accidental, list it in the PR description.
```

---

## P5. Reference forward pass (correctness only)

Repo: raya. Branch: `p5-forward` (needs P2, P3, P4 merged)

```
Implement a correct, unoptimized Raya-L forward pass in raya-core, verified layer by layer against the
P2 golden fixtures. Correctness is the only goal; speed is P6/P7.

1. First, write `docs/tensor-runtime.md` comparing (a) building on `candle`, (b) ndarray plus a
   BLAS binding (system OpenBLAS / Accelerate), (c) plain Rust loops for the reference path. Recommend
   one with reasons (build simplicity on Windows, macOS and Linux, GEMM performance on large shapes,
   ability to call our own attention kernel). Implement with the recommendation; keep tensor use
   behind a small internal trait so it can be swapped.
2. Implement: embeddings, RoPE (separate theta for global and local layers as in Laya), RMS/LayerNorm
   as in Laya, gated MLP, dense attention with both the global pattern and the local +/-window mask
   (mask applied to dense scores is fine here), the layer pattern from config, the 2-layer decision
   head over the full sequence, marker-token scoring, and temperature-calibrated probabilities.
3. Add a forward hook/trace facility so tests can capture every intermediate activation.
4. Tests: for each P2 golden case, compare embedding output, every encoder layer output, head layer
   outputs, and final probabilities against the goldens. Tolerance 1e-4 absolute for fp32 layer
   outputs (justify and tighten if achievable); probabilities 1e-5. A failing layer must be reported
   by index so divergences are localizable.
5. `raya predict --model <dir> --document <file> --question <text>` wired to tokenizer + forward.
6. Attention is invoked through a trait `AttentionKernel` so P6 kernels plug in without touching the
   model code; the dense reference implements it.

Do not optimize. Do not add threading beyond what the chosen GEMM library does by default.
```

---

## P6. Fused block-local attention kernel (independent spike)

Repo: raya. Branch: `p6-block-local-kernel` (needs P1; uses P2's `golden kernels` output for tests if merged,
otherwise generates its own reference with a naive Rust implementation)

```
Implement the block-local attention kernel in raya-kernels and benchmark it. Read
reference/laya-sparse/laya/experiments/kernels/local.py and docs/feasibility.md sections 2.2 and 2.3 (Phase 1 kernel
numbers are quoted there).

Spec: seq up to 8192, 16 heads, head dim 64, block size 128, each query block attends to its own block
plus one neighbour on each side (equivalent to Laya's +/-64 local window: verify the equivalence
semantically and write down exactly how edge blocks and padding are handled).

1. `AttentionKernel` trait (shared with P5; if P5 is not merged, define it in raya-kernels and P5 will
   adopt it). Implementations:
   a. `naive`: per-score dot products, obviously correct, used as the test oracle.
   b. `tiled`: FlashAttention-style. Per (head, query block): load the Q tile, iterate K/V tiles,
      compute the score tile with a small register-blocked micro-GEMM, online softmax, accumulate into
      the output tile. Never materialize a full score matrix. Multithreaded across (head, block) with
      rayon or std threads; thread count configurable.
   c. Portable SIMD via `std::simd` or the `wide` crate; optional `avx2` and `neon` micro-kernels
      behind cargo features with runtime dispatch and a portable fallback. Do not make either the
      default; the target CPU is an open decision.
2. Correctness tests: tiled vs naive on random inputs, lengths 1, 127, 128, 129, 1000, 2048, padded
   tail, all-masked-row safety, extreme values (large logits) for softmax stability. Tolerance 1e-5.
   If P2 goldens exist, also compare against the PyTorch kernel output.
3. Criterion benchmarks for one layer at 2048, 4096, 8192 tokens, plus a per-stage breakdown (QK tile,
   softmax, PV tile). Output a JSON file with cpu model, core count, feature flags, rustc version,
   git commit.
4. A short `docs/kernel-notes.md`: design, what was tried, what was slower than expected. Report
   numbers as "cloud-agent hardware, indicative only". The reference to beat is 85 ms per layer at 8K
   on the project's i7-8700 (PyTorch block-local); state plainly that this comparison is not valid
   until the bench is run on that machine via P7.

Negative results are welcome: if the tiled kernel does not beat the naive one or cannot be made
competitive, say so and why.
```

---

## P7. Benchmark harness that the researcher runs locally

Repo: raya for steps 1, 4, 5, 6; laya-sparse for steps 2-3 (separate PR). Branch: `p7-bench-harness` (needs P5, P6 merged)

```
Add `raya bench` and a Python-side comparison so the researcher can produce a valid measurement on the
project's target machine.

1. `raya bench --model <dir> --tokens 512,2048,4096,8192 --repeats N --threads T --warmup W`:
   synthetic but fixed token input of the requested length plus one question, model loaded once
   (weights resident), measure end-to-end predict latency and per-component time (embedding, each
   encoder layer, head) with a low-overhead timer. Report p50/p90, peak RSS, and whether a time cap was
   hit, mirroring the conventions of the Laya sweep (read reference/laya-sparse/laya/experiments/cli.py).
2. Write results to `results/<run-id>/raya-bench.json` (git-ignored) using the same manifest fields as
   Laya runs (cpu, cores, torch-free environment info, git commit, checkpoint revision, flags),
   plus the AttentionKernel implementation used.
3. Add `python -m experiments compare-raya --run-id <id>` that reads the Laya run and the Raya run for
   the same machine and emits a table (latency, peak memory, max abs output difference on identical
   inputs). Refuse to compare runs whose manifest CPU fields differ.
4. Output-parity gate: `raya bench` first runs one predict and compares probabilities with a Laya
   output saved from the same input; abort the benchmark if they differ beyond tolerance, so nobody
   benchmarks a wrong model.
5. PowerShell script `scripts/build_windows.ps1` that builds release with `target-cpu=native`
   off by default (and an explicit flag to turn it on), matching the Windows conventions in
   reference/laya-sparse/laya/experiments/setup_windows.ps1.
6. Document in README.md the exact steps for the researcher to run on the i7-8700 machine.

Do not include any benchmark numbers from the cloud in docs/ as comparisons. Step 3's `compare-raya`
command is implemented in laya-sparse (it reads both result directories); in raya, only emit the JSON.
```

---

## After the scaffold (human tasks, not for the agent)

- Fix the target CPU and precision before any kernel tuning beyond P6.
- Run P7 on the i7-8700, then decide whether the Rust kernel beats the 85 ms/layer reference.
- Revisit compression, model sizing and the training pipeline only when Phase 2 results exist.
