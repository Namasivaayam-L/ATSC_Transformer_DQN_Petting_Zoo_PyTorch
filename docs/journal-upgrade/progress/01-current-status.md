# 01 — Current Status (Task-Level Tracker)

> **Last updated:** 2026-06-29 — Session #7 (Ep count study verified, 35/50 complete)

## Active phase: Phase 3.8 — Episode Count Convergence Study (PARTIAL)
**Overall status:** 🟢 Phases 0–3 DONE. Ep count study 35/50 combos verified clean. 15 remaining (9 in-progress, 6 missing). Phase 4 (figures) next after completion.

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

## Phase 3.8 — Episode Count Convergence Study

| Task | Description | Status |
|------|-------------|--------|
| 3.8.0 | Create plan & scripts (`run_ep_count.sh`, `eval_ep_count.py`) | ✅ **Done** |
| 3.8.1 | Run trf_coord × trf_coord_equal × 5 nets × [50,100,200,400,500] eps | 🟡 **35/50 combos verified** |
| 3.8.2 | Verify aggregate integrity | ✅ **Done** — all 35 pass (no NaN, no zero travel, variance OK) |
| 3.8.3 | Run `eval_ep_count.py` analysis | 🔴 Pending (after all runs complete) |

### Completeness Matrix (verified 2026-06-29)

```
                    eps50    eps100   eps200   eps400   eps500
─────────────────────────────────────────────────────────────
trf_coord
  grid4x4          [5/5]    [5/5]    [5/5]    [3/5]    [0/5]
  cologne3         [5/5]    [5/5]    [5/5]    [5/5]    [1/5]
  cologne8         [5/5]    [5/5]    [5/5]    [5/5]    ─
  ingolstadt7      [5/5]    [5/5]    [5/5]    [4/5]    ─
  ingolstadt21     [5/5]    [5/5]    [5/5]    [1/5]    ─

trf_coord_equal
  grid4x4          [5/5]    [5/5]    [5/5]    [2/5]    [0/5]
  cologne3         [5/5]    [5/5]    [5/5]    [5/5]    [0/5]
  cologne8         [5/5]    [5/5]    [5/5]    [5/5]    ─
  ingolstadt7      [5/5]    [5/5]    [5/5]    [5/5]    ─
  ingolstadt21     [5/5]    [5/5]    [5/5]    [0/5]    ─
```

### Verified Results — ATT (s) mean [lo, hi]

**grid4x4:**
| Ep | trf_coord | trf_coord_equal |
|----|-----------|-----------------|
| 50 | 39.3 [31.5, 45.3] | 40.8 [33.3, 48.1] |
| 100 | 41.0 [30.5, 50.4] | 41.1 [34.0, 48.2] |
| 200 | 36.4 [27.7, 45.7] | **26.7 [20.3, 32.5]** |

**cologne3:**
| Ep | trf_coord | trf_coord_equal |
|----|-----------|-----------------|
| 50 | 19.0 [11.6, 26.7] | 38.0 [15.6, 73.6] |
| 100 | 14.7 [9.6, 21.8] | 59.7 [11.4, 148.0] |
| 200 | 12.7 [6.2, 20.9] | 35.1 [6.5, 80.1] |
| 400 | 17.6 [10.2, 24.6] | **11.9 [3.8, 20.8]** |

**cologne8:**
| Ep | trf_coord | trf_coord_equal |
|----|-----------|-----------------|
| 50 | 39.0 [32.7, 46.5] | 42.0 [35.5, 50.1] |
| 100 | 41.3 [30.4, 52.0] | 46.1 [28.5, 66.2] |
| 200 | 28.1 [19.5, 35.8] | 29.4 [23.1, 35.2] |
| 400 | 24.0 [14.7, 34.8] | **16.9 [10.9, 24.4]** |

**ingolstadt7:**
| Ep | trf_coord | trf_coord_equal |
|----|-----------|-----------------|
| 50 | 16.9 [13.6, 20.4] | 18.0 [14.8, 21.2] |
| 100 | 14.8 [11.1, 19.2] | 14.9 [12.5, 17.2] |
| 200 | 19.8 [12.0, 27.9] | 16.3 [10.1, 23.6] |
| 400 | ─ | **10.8 [6.8, 18.1]** |

**ingolstadt21:**
| Ep | trf_coord | trf_coord_equal |
|----|-----------|-----------------|
| 50 | 38.9 [23.3, 66.3] | 29.7 [25.0, 34.7] |
| 100 | 30.8 [20.7, 45.9] | 25.8 [22.7, 29.3] |
| 200 | **21.9 [15.1, 29.2]** | 23.1 [16.9, 33.1] |

### Key Findings
1. **cologne3 trf_coord_equal unstable** — very wide CIs (59.7 [11.4, 148.0] at 100 eps)
2. **Convergence trend**: cologne8, ingolstadt7 show clear ATT improvement 50→400 eps
3. **grid4x4 flat** — already good at 50 eps, marginal gain to 200
4. **Resume**: `bash run_ep_count_parallel.sh` (auto-skips completed combos)

---

## Phase 4–5 — Next

| Task | Description | Status |
|------|-------------|--------|
| 4.1 | Generate learning curves | 🟡 Pending (after ep count study) |
| 4.2 | Attention heatmap | 🔴 Not Started |
| 4.3 | IQM performance profiles | 🔴 Not Started |
| 4.4 | Results table (LaTeX) | 🔴 Not Started |
