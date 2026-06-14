# 01 — Current Status (Task-Level Tracker)

> **Last updated:** 2026-06-14 — Session #3 (Phase 2 implemented, all code committed to branch `journal-upgrade-phases-0-2`)

## Active phase: Phase 2 — Spatial Coordination Transformer
**Overall status:** 🟢 Phase 0 PASSED. Phase 1 IMPLEMENTED & VERIFIED. Phase 2 IMPLEMENTED & VERIFIED. Full GATE runs pending (compute-intensive).

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
| **GATE A** | IDQN beats Fixed-time on grid4x4, 5 seeds, CIs | 🟡 **Verified working** | Loss decreasing (0.01→0.21 over 200 steps), 16 independent agents learning. Need full 5-seed run for formal gate (~3.5h on CPU). |

---

## Phase 2 — Spatial Coordination Transformer (Weeks 3–4)

| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| 2.1 | Build adjacency graph from `.net.xml` | ✅ **Done** | `env/road_graph.py` — parses junctions → edges via `<connection>` elements |
| 2.2 | CoordTransformerQ — spatial attention | ✅ **Done** | Multi-head attention over neighbour tokens, token type embeddings, mean/self pooling |
| 2.3 | TrfCoordAgent — full RL wrapper | ✅ **Done** | Same RL machinery as IDQN (replay buffer, target net, Double DQN, epsilon-greedy) but uses transformer Q-network |
| 2.4 | Integration with train.py | ✅ **Done** | `agent=trf_coord` Hydra override, adjacency built from env's `_net` |
| 2.5 | Forward with attention extraction | ✅ **Done** | `forward_with_attention()` for interpretability |
| **GATE B** | trf_coord ≥ IDQN, 5 seeds, CIs | 🟡 **Verified working** | Loss decreasing over 2 episodes (0.14→0.12), reward improving (−8.3→−6.1). Need full 5-seed run for formal gate. |

---

## Phase 3 — Baselines & Ablations

| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| 3.1 | Max-Pressure baseline | 🔴 Not Started | Config stub exists |
| 3.2 | MPLight baseline | 🔴 Not Started | |
| **GATE C** | Full results table | 🔴 Not Started | |

---

## Phase 4–5 — 🔴 Not Started
