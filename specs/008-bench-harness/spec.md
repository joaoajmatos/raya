# Feature Specification: Benchmark Harness for the Target Machine

**Feature Branch**: `p7-bench-harness` (raya) and a separate PR in laya-sparse for the compare command (specification directory only; no speckit branch created)

**Created**: 2026-09-30

**Status**: Draft

**Origin**: Prompt P7 of the former `docs/agent-prompts.md`.

**Depends on**: 006, 007. **Blocks**: the human decision on whether Rust beats PyTorch.

**Input**: User description: "`raya bench` and a Python-side comparison so the researcher can produce a valid measurement on the i7-8700 and compare Raya to Laya on identical inputs, with an output-parity gate."

## User Scenarios & Testing *(mandatory)*

The user is the researcher on the target machine, who needs a trustworthy Raya-versus-Laya comparison.

### User Story 1 - Benchmark Raya (Priority: P1)

`raya bench --model <dir> --tokens 512,2048,4096,8192 --repeats N --threads T --warmup W` measures end-to-end predict latency for a fixed synthetic token input plus one question, with weights resident, and per-component times (embedding, each encoder layer, head).

**Why this priority**: It is the measurement the whole project is waiting for.

**Independent Test**: Run on the fixture model; check JSON fields and component sums.

**Acceptance Scenarios**:

1. **Given** the options above, **When** run, **Then** it reports p50 and p90, peak RSS and whether a time cap was hit, following the conventions of `experiments/cli.py` in laya-sparse.
2. **Given** a run, **When** it finishes, **Then** `results/<run-id>/raya-bench.json` (git-ignored) records the manifest fields used by Laya runs (CPU, cores, environment info, git commit, checkpoint revision, flags) plus the `AttentionKernel` implementation used.
3. **Given** the benchmark starts, **When** a first predict runs on an input with a saved Laya output, **Then** a probability difference above tolerance aborts the benchmark.

---

### User Story 2 - Compare with Laya (Priority: P2)

`python -m experiments compare-raya --run-id <id>` (in laya-sparse) reads the Laya run and the Raya run and emits a table of latency, peak memory and max abs output difference on identical inputs.

**Why this priority**: Turns two result directories into a decision table.

**Independent Test**: Run on paired sample results; run on mismatched manifests.

**Acceptance Scenarios**:

1. **Given** matching runs, **When** compared, **Then** a table shows latency, peak memory and max abs output difference.
2. **Given** runs whose manifest CPU fields differ, **When** compared, **Then** the command refuses.

---

### User Story 3 - Windows Build and Researcher Steps (Priority: P3)

`scripts/build_windows.ps1` builds release without `target-cpu=native` by default and with it behind an explicit flag; `README.md` documents exact steps for the i7-8700 machine.

**Why this priority**: The target machine runs Windows 11.

**Independent Test**: Follow the README on the target machine.

**Acceptance Scenarios**:

1. **Given** the script, **When** run with no flags, **Then** `target-cpu=native` is off.
2. **Given** the README, **When** followed, **Then** the researcher produces a valid `raya-bench.json` and compare output.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Benchmark lengths, repeats, threads and warmup are configurable; model load is excluded from timing and reported separately.
- **FR-002**: Low-overhead per-component timer.
- **FR-003**: Output-parity gate before any timing.
- **FR-004**: Manifest parity with Laya runs so `compare-raya` can validate machines.
- **FR-005**: `compare-raya` lives in laya-sparse (separate PR); raya only emits JSON.
- **FR-006**: No cloud benchmark numbers in `docs/` as comparisons (constitution III).
- **FR-007**: Windows script matches the conventions of `experiments/setup_windows.ps1`.

## Success Criteria *(mandatory)*

- **SC-001**: A researcher can go from clone to a compared result on the target machine using only the README.
- **SC-002**: A wrong model cannot be benchmarked (gate aborts) and mismatched machines cannot be compared.

## Assumptions

- The Laya sweep's CLI (`experiments/cli.py`) is the reference for run-id, result layout and time-cap conventions.
