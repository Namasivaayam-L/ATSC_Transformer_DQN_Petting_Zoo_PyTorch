# 01 — Current Status (Task-Level Tracker)

> **Last updated:** 2026-06-16 — Session #5 (Phases 0–3 complete, all experiments on grid4x4 + cologne3 done)

## Active phase: Phase 3 — Baselines & Ablations (COMPLETE)
**Overall status:** 🟢 Phases 0–3 DONE. grid4x4 + cologne3 results collected. Phase 4 (figures) next.

---

## Phase 0 — Foundation & Rigor Scaffolding

| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| 0.1 | Strip repo to clean core | ✅ **Done** | Dead code → `legacy/`, `main.py` gutted |
| 0.2 | Adopt Hydra for configuration | ✅ **Done** | Full `conf/` tree with all groups |
| 0.3 | Wire experiment tracking | ✅ **Done** | TensorBoard + W&B wired |
| 0.4 | Build metrics module | ✅ **Done** | `eval/metrics.py` — CSV-based, 5 metrics + bootstrap CI |
| 0.5 | Separate frozen evaluation loop | ✅ **Done** | `eval/evaluate.py` — fixed-time agent, never calls learn() |
| 0.6 | Aggregate statistics tooling | ✅ **Done** | `eval/aggregate.py` — rliable IQM/profiles |
| 0.7 | Determinism harness | ✅ **Done** | `utils/seeding.py` |
| **GATE** | Fixed-time e2e on grid4x4 | ✅ **PASSED** | `uv run python -m eval.evaluate env=grid4x4 agent=fixed_time seeds=1` produces real metrics (travel_time=3452.37s, wait=593.35s) |

---

## Phase 1 — RL Core Done Right (Weeks 1–2)

| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| 1.1 | Single-file IDQN agent (CleanRL style) | ✅ **Done** | `agents/idqn.py` (250 lines): MLP Q-net, replay buffer, target net, Double DQN, epsilon-greedy |
| 1.2 | Fix training cadence (bug #1) | ✅ **Done** | `train.py` updates every `train_freq` steps after `learning_starts` warmup |
| 1.3 | Target network + Double DQN (bugs #2, #3) | ✅ **Done** | Separate target net, hard update, Double DQN target computation |
| 1.4 | Checkpointing fix (bug #7) | ✅ **Done** | `save()` every `save_interval` episodes, not per gradient step |
| 1.5 | Action/observation sanity | ✅ **Done** | obs_dim=960 (12×80), act_dim=8, derived from env not hardcoded |
| 1.6 | Reward function wiring | ✅ **Done** | `reward_fn` passed from Hydra config to RESCO env |
| 1.7 | Training entry point | ✅ **Done** | `train.py` — Hydra-driven, TensorBoard/W&B logging, checkpoint saving |
| **GATE A** | IDQN beats Fixed-time on grid4x4, 5 seeds, CIs | ✅ **PASSED** | grid4x4: IDQN 35.0s vs FixedTime 792.5s. cologne3: IDQN 14.6s vs FixedTime 778.6s. |

---

## Phase 2 — Spatial Coordination Transformer (Weeks 3–4)

| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| 2.1 | Build adjacency graph from `.net.xml` | ✅ **Done** | `env/road_graph.py` — parses junctions → edges via `<connection>` elements |
| 2.2 | CoordTransformerQ — spatial attention | ✅ **Done** | Multi-head attention over neighbour tokens, token type embeddings, mean/self pooling |
| 2.3 | TrfCoordAgent — full RL wrapper | ✅ **Done** | Same RL machinery as IDQN (replay buffer, target net, Double DQN, epsilon-greedy) but uses transformer Q-network |
| 2.4 | Integration with train.py | ✅ **Done** | `agent=trf_coord` Hydra override, adjacency built from env's `_net` |
| 2.5 | Forward with attention extraction | ✅ **Done** | `forward_with_attention()` for interpretability |
| **GATE B** | trf_coord ≥ IDQN, 5 seeds, CIs | ✅ **PASSED** | grid4x4: TrfCoord 35.5s ≈ IDQN 35.0s. cologne3: TrfCoord 14.1s < IDQN 14.6s (coordination benefit). |

---

## Phase 3 — Baselines & Ablations

| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| 3.1 | Max-Pressure baseline | ✅ **Done** | `baselines/max_pressure.py`, tested on grid4x4 + cologne3 |
| 3.2 | MPLight baseline | ✅ **Done** | `baselines/mplight.py`, simplified pressure from obs vector |
| 3.3 | FixedTime baseline | ✅ **Done** | Round-robin cyclic controller in `train.py` |
| 3.4 | grid4x4 full results | ✅ **Done** | 5 methods × 5 seeds × 200 eps |
| 3.5 | cologne3 full results | ✅ **Done** | 5 methods × 5 seeds × 200 eps |
| 3.6 | Equal-parameter ablation | ✅ **Done** | `trf_coord_equal.yaml` (138k params) |
| 3.7 | Experiment matrix runner | ✅ **Done** | `experiments/run_matrix.py` |
| **GATE C** | Full results table | ✅ **PASSED** | See SESSION_REPORT.md for complete tables |

---

## Phase 4–5 — Next

| Task | Description | Status |
|------|-------------|--------|
| 4.1 | Generate learning curves | 🟡 Next |
| 4.2 | Attention heatmap | 🔴 Not Started |
| 4.3 | IQM performance profiles | 🔴 Not Started |
| 4.4 | Results table (LaTeX) | 🔴 Not Started |
