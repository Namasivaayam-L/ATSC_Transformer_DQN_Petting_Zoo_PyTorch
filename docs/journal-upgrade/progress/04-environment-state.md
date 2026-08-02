# 04 — Environment State (Setup & Configuration)

Tracks what's installed, configured, and any environment-specific notes. Update this whenever
you install packages, change configs, or discover environment quirks.

> **Last updated:** 2026-06-26 12:00 +05:30 — Session #6

---

## System
- **OS:** Linux
- **Python manager:** `uv` (per plan — bare `python` may resolve to a shim)
- **GPU:** NVIDIA GTX 1650 (4GB VRAM), CUDA 12.13 detected, PyTorch 2.12.0+cu132

## Repo location
- **Path:** `/home/namachu/Documents/personal/ATSC_Transformer_DQN_Petting_Zoo_PyTorch`
- **Journal-upgrade dir:** `./journal-upgrade/`
- **Current branch:** `journal-upgrade-phases-0-2` (plus ep count study files untracked)

## SUMO
- **Status:** ✅ Verified working
- **`SUMO_HOME`:** `/usr/share/sumo`
- **`libsumo`:** Importable via `sumo_rl` (vendored fork)
- **Check command:** `uv run python -c "import sumo_rl; print('OK')"` or `python train.py ...`

## Python dependencies
| Package | Version | Notes |
|---------|---------|-------|
| hydra-core | 1.3.3 | Configuration management |
| omegaconf | 2.3.1 | YAML config parsing |
| torch | 2.12.0+cu132 | GPU-enabled (CUDA 13.2) |
| sumo-rl | (vendored in `./sumo_rl/`) | RESCO envs + PettingZoo wrapper |
| rliable | installed | Statistical aggregation (IQM, CIs) |
| pandas | 3.0.3 | CSV/metrics processing |
| numpy | latest | Array operations |
| tensorboard | available | Experiment tracking (default) |
| wandb | available | Optional experiment tracking |

## Directories created
| Directory | Status | Description |
|-----------|--------|-------------|
| `conf/` | ✅ Committed | Full Hydra config tree (config.yaml + 9 group configs) |
| `agents/` | ✅ Committed | `idqn.py` + `trf_coord.py` + `__init__.py` |
| `env/` | ✅ Committed | `road_graph.py` — adjacency graph from SUMO net |
| `eval/` | ✅ Committed | Full evaluation pipeline (metrics, evaluate, aggregate) |
| `baselines/` | ✅ Committed | `max_pressure.py`, `mplight.py` + `__init__.py` |
| `utils/` | ✅ Committed | `seeding.py` added |
| `legacy/` | ✅ Committed | Dead code preserved for reference (sac/, trf_dqn/, models.py) |
| `journal-upgrade/` | ✅ Committed | Complete planning + progress tracking |
| `results/` | ⚡ Untracked | Centralized output: `results/{scenario}/{agent}/{reward}/` with `seed_*/`, `aggregate.json`, `tb/` per experiment |
| `nets/RESCO/` | ✅ External | Downloaded from GitHub, NOT tracked in git |
| `figures/` | ✅ Committed | `make_figures.py` — regenerates all figures/tables |
| `paper/` | 🔴 Not created | Phase 5 |
| `run_ep_count.sh` | ⚡ Untracked | Sequential episode count study runner |
| `run_ep_count_parallel.sh` | ⚡ Untracked | Parallel episode count study runner (MAX_PARALLEL=8) |
| `eval_ep_count.py` | ⚡ Untracked | Episode count convergence analysis + plots |
| `results_ep_count/` | ⚡ Untracked | Study output (large, ~500MB+), DO NOT commit to git |
| `run_ep_count_nohup.log` | ⚡ Untracked | Sequential run log |
| `run_ep_count_parallel_nohup.log` | ⚡ Untracked | Parallel run log |

## Known gotchas
1. ✅ **Observation shape**: Space says `(5,)` but actual obs is `(12, 80)` = 960 dims. Use `env.reset()` output, not `observation_spaces`.
2. ✅ **CSV saving**: sumo_rl only saves on `reset()` not `close()`. Code explicitly calls `save_csv` after each episode.
3. ✅ **Reward function names**: Must match what's registered in `traffic_signal.py` (line 321-326): `dwt`, `average-speed`, `queue`, `pressure`. Use `dwt` not `diff-waiting-time`.
4. ✅ **`save_csv` attribute**: sumo_rl env doesn't always expose `save_csv`; use `env.unwrapped.env.save_csv` or set via `out_csv_name` at construction.
5. ⚠️ **Simulation speed**: ~60s per 3600-step episode on CPU. GPU (GTX 1650) available but not yet utilized for simulation step (SUMO runs on CPU). Only neural network forward/backward benefits from GPU.
6. ✅ **Agent count**: grid4x4 = 16 agents (A0-D3), cologne3 = 3 agents. Each agent in grid4x4 has 2–4 neighbours.
7. ⚠️ **RESCO nets NOT tracked in git**: Download from `Pi-Star-Lab/RESCO` GitHub if doing a fresh clone.
8. ✅ **`train_freq` fix**: Gradient updates every `train_freq` env steps (not per episode), after `learning_starts` warmup.
9. ✅ **Target network**: Separate target net with hard update every `target_update_interval` steps. Optional Polyak averaging via `tau` config.
10. ✅ **Buffer size check**: Both agents check `max(learning_starts, batch_size)` before sampling, preventing batch-size-greater-than-buffer crashes.
11. ✅ **`setsid` for background training**: opencode bash tool sends SIGTERM on timeout. Use `setsid` not `nohup` for fully detached background processes.
12. ✅ **Max 8 concurrent SUMO instances (REVISED)**: Original assumption said max 4, but testing shows SUMO port conflicts don't exist — each instance gets its own random TCP port. 8 jobs run fine on GTX 1650 / 22GB RAM at ~1200 MiB GPU.
13. ✅ **Heterogeneous obs padding**: cologne3 has agents with 5/6/8 lanes. Pad all obs to max (640-dim) with zeros.
14. ✅ **MPLight simplified**: Compute pressure from obs vector only, no SUMO API calls (avoids TraCI errors with parallel envs).
15. ✅ **`find_csv` returns last episode**: First episode CSV has all zeros (simulation startup). Always use last for metrics.
16. ✅ **Ep count study timing**: grid4x4 eps=50 ~82 min, cologne3 eps=50 ~27 min, cologne3/8 eps=200 ~50 min, ingolstadt21 eps=50 ~111 min, cologne8 eps=100 ~14.4h. Larger × more eps scales ~linearly.
17. ✅ **Resume robust**: `train.py` saves `latest.pt` per seed and `seeds_done.json`. Re-running automatically skips completed seeds. Safe to kill at any time.
18. ✅ **`run_ep_count_parallel.sh` uses bash job queue**: Launches up to MAX_PARALLEL (default 8) combos in background, tracks PIDs, waits for completion. Handles resume automatically, skips combos with 5-seed `aggregate.json`.
