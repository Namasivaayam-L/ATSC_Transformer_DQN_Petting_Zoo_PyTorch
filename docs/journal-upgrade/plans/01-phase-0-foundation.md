# Phase 0 — Foundation & Rigor Scaffolding (Weeks 0–1)

## Context recap (why this phase exists)
Journal acceptance is won or lost on rigor, not novelty. Before touching the model we build the
reproducibility, tracking, evaluation, and metrics infrastructure. The current repo has none of this:
configs are `.ini` files with hardcoded absolute paths (e.g. a `/home/namachu/...` sys.path append in
`dqn/main.py`), results are loose CSVs, and "evaluation" reuses the training loop.

## Prerequisites
- Repo cloned at `../ATSC_Transformer_DQN_Petting_Zoo_PyTorch`.
- SUMO installed and `SUMO_HOME` set; `libsumo` importable.
- `uv` available for Python execution.

## Tasks

### 0.1 — Strip the repo to a clean core
DELETE (move to a `legacy/` folder rather than hard-delete, to preserve history):
- `experiments/trf_multi_agent/sac/` (dead TensorFlow SAC; mixes TF with the PyTorch codebase).
- `experiments/trf_multi_agent/dqn/models.py` (entirely commented-out model graveyard).
- `experiments/trf_dqn/` (single-agent leftover; superseded by the multi-agent track).
Remove the hardcoded absolute `sys.path.append('/home/namachu/...')` in `dqn/main.py`; replace with a
proper package install (`uv pip install -e .`) or a relative path resolved at runtime.

### 0.2 — Adopt Hydra for configuration
- `uv add hydra-core omegaconf`.
- Create `conf/` with a structured config: `conf/config.yaml` (defaults) + groups
  `conf/env/{grid4x4,cologne3}.yaml`, `conf/agent/{idqn,trf_coord,trf_temporal}.yaml`,
  `conf/reward/{dwt,pressure,queue}.yaml`.
- Every former `.ini` key becomes a typed config field. No constant may be hardcoded in `.py`.
- Each run writes its resolved config into its output dir (Hydra does this automatically).

### 0.3 — Wire experiment tracking
- `uv add wandb` (or use TensorBoard if offline). `wandb login` with the academic account.
- Log: all five metrics per episode, epsilon, loss, seed, full config, and (later) attention maps.
- Group runs by `scenario/method/reward` and tag with the seed.

### 0.4 — Build the metrics module
Create `eval/metrics.py` that parses SUMO's `tripinfo.xml` / `summary.xml` outputs and computes:
- **average travel time** (primary), average waiting time, mean queue length, throughput
  (vehicles that completed their trip), total time loss.
- Provide both per-episode and aggregate-over-seeds (mean ± 95% bootstrap CI) functions.
- Unit-test the parser on a tiny recorded run.

### 0.5 — Separate, frozen evaluation loop
Create `eval/evaluate.py`:
- Loads a checkpoint, sets `epsilon = 0`, runs N evaluation seeds, NEVER calls `learn()` or updates weights.
- Writes per-seed metrics to disk and logs aggregates to W&B.
- This permanently fixes the contaminated-evaluation bug (#5).

### 0.6 — Aggregate-statistics tooling
- `uv add rliable`.
- Create `eval/aggregate.py` that consumes per-seed metric arrays and produces IQM, performance
  profiles, and stratified-bootstrap 95% CIs (the NeurIPS-standard results methodology).

### 0.7 — Determinism harness
Create `utils/seeding.py` setting seeds for `random`, `numpy`, `torch` (+ `torch.use_deterministic_algorithms`
where feasible) and passing `--seed` to SUMO. Call it at the start of every entry point.

## Files created/edited in this phase
- New: `conf/**`, `eval/metrics.py`, `eval/evaluate.py`, `eval/aggregate.py`, `utils/seeding.py`.
- Edited: `experiments/trf_multi_agent/dqn/main.py` (remove hardcoded path; read Hydra config).
- Moved: dead code → `legacy/`.

## Commands
```bash
uv add hydra-core omegaconf wandb rliable
uv pip install -e .
# smoke test the fixed-time baseline end-to-end (see Phase 1 for the agent; here just prove the pipeline):
uv run python -m eval.evaluate agent=fixed_time env=grid4x4 seeds=5
```

## ACCEPTANCE GATE (Phase 0)
A do-nothing **fixed-time** controller runs end-to-end on `grid4x4`, and the pipeline logs all five
metrics to W&B across 5 seeds with 95% CI bands, fully driven by a Hydra config with zero hardcoded
paths. If any metric cannot be computed or any path is hardcoded, this gate FAILS — fix before Phase 1.

## Dependencies
None (this is the first phase). Everything downstream depends on this.
