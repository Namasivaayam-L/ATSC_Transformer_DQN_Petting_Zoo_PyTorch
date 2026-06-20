# Session #2 — "happy-eagle" (Jun 14–Jun 20, 2026)

**Conversation ID:** `ses_13a94cc0cffeqR9fygwnY0sUWn`
**Duration:** ~7 days (multi-resume, long-running experiments)
**Total messages:** 868 | **Parts (tool calls + text):** 4,320
**Tokens:** 2,580,594 input / 203,953 output | **Cost:** $0.00

---

## Executive Summary

This session performed the bulk of the journal-upgrade execution across Phases 0–3. It began by auditing existing (uncommitted) work [#msg 6], then systematically completed Phase 0 verification [#msg 91], implemented Phases 1 & 2 [#msg 94][#msg 144], ran all GATE experiments on both grid4x4 and cologne3 [#msg 594][#msg 720], performed the Phase 3 reward sweep and equal-param ablation [#msg 344][#msg 852], and committed/pushed everything to a dedicated branch [#msg 213]. GPU acceleration was enabled mid-session, drastically speeding up later runs [#msg 811][#msg 840]. Results are organized in per-experiment directories with CSVs, TensorBoard logs, and model checkpoints.

---

## Timeline of Work

### Phase 0 — Foundation Verification (Jun 14)

- **Progress audit:** Discovered Phase 0 code was already written (staged + untracked) but progress files said "Not Started" — updated all 4 progress files [#msg 6][#msg 14]
- **Gate verification:** Ran fixed-time end-to-end on grid4x4 → `travel_time=3452s` with all 5 metrics working [#msg 91]
- **Fixes during gate:**
  - Downloaded missing RESCO nets from GitHub to `nets/RESCO/` [#msg 24]
  - Fixed reward config names (`"diff-waiting-time"` → `"dwt"`) [#msg 41]
  - Rewrote `eval/metrics.py` to parse CSV output (sumo_rl doesn't emit tripinfo.xml) [#msg 62]
  - Fixed `evaluate.py` to pass `reward_fn` and save CSV after episodes [#msg 43]
- **BLOCKER-001 resolved:** Conference paper was published (user confirmed), journal submission framed as substantial extension [#msg 16]

### Phase 1 — RL Core: IDQN Agent (Jun 14)

- Implemented `agents/idqn.py` — CleanRL-style single-file IDQN with target network, Double DQN, proper training cadence [#msg 94]
- Implemented `train.py` — training entry point with env creation, replay buffer, logging, checkpointing [#msg 96]
- Tested: loss decreasing (0.14 → 0.12) over 2 episodes, reward improving (-8.3 → -6.1) [#msg 110]

### Phase 2 — Spatial Coordination Transformer (Jun 14)

- Implemented `env/road_graph.py` — lane-level adjacency from SUMO `.net.xml` (grid4x4: 16 lanes → 16 tokens) [#msg 144]
- Implemented `agents/trf_coord.py` — transformer Q-network with learned positional embeddings, 2-layer encoder, 2 attention heads [#msg 144]
- Fixed `norm_first` iteration bug in transformer forward pass (rewrote to use standard `self.transformer()` + separate `forward_with_attention()`) [#msg 154][#msg 186]
- Fixed buffer size check: both agents now use `max(learning_starts, batch_size)` [#msg 169]
- Tested: TrfCoord working, loss decreasing on short runs [#msg 173]

### GPU Enablement (Jun 14)

- Machine: GTX 1650 with 4GB VRAM (CUDA 13.2) [#msg 120]
- User installed CUDA packages; session resumed with GPU detected [#msg 153]
- Added `device: auto` to config with CUDA auto-detection [#msg 812]
- GPU verified: `Using device: cuda` confirmed [#msg 811]

### Committing & Branching (Jun 14)

Created branch `journal-upgrade-phases-0-2` with 3 commits [#msg 191][#msg 213]:

| Commit | Message | Files |
|--------|---------|-------|
| `b8c03c2` | Phase 0: Foundation infrastructure | 59 files, 3,117 insertions |
| `0ff4b91` | Phase 1: RL Core — IDQN agent | 2 files, 532 lines |
| `d6cdcc2` | Phase 2: Spatial Coordination Transformer | 2 files, 421 lines |

Branch pushed to origin [#msg 213].

### GATE A & GATE B Experiments — CPU (Jun 15–18)

Full 5-seed, 1000-episode runs on grid4x4 + cologne3 across all agents. Experiments were launched in background and ran for multiple days [#msg 217][#msg 387]:

**grid4x4 reward sweep results (CPU)** [#msg 594]:

| Agent | DWT | Pressure | Queue |
|-------|-----|----------|-------|
| Fixed-time | 30.56s | — | — |
| IDQN | 34.98s | 38.87s | 33.76s |
| TrfCoord | 35.47s | 40.06s | 34.04s |
| MaxPressure | ~27s | ~27s | ~27s |
| MPLight | ~30s | ~30s | ~30s |

**cologne3 sweep** also completed with 5-seed runs for all agents [#msg 720]:

| Method | DWT Travel Time | 95% CI |
|--------|----------------|--------|
| IDQN | 14.65s | [7.60, 22.33] |
| TrfCoord | ~18s | — |
| MaxPressure | ~480s | deterministic |
| MPLight | ~480s | deterministic |
| Fixed-Time | ~480s | deterministic |

### Data Saving & Model Checkpoints (Jun 19)

- User requested models/results be saved to disk [#msg 813]
- Fixed output directory structure: `results/logs/grid4x4/` and `results/logs/cologne3/` subfolders [#msg 812]
- Identified missing `trf_coord_grid4x4_dwt` run (ran before output dir fix) — re-ran on GPU [#msg 823]
- Fixed `.gitignore` to track CSVs, aggregate.json, checkpoints (.pt), TensorBoard logs (except Hydra outputs and RESCO nets) [#msg 860][#msg 862]
- Actual commit message: `feat: equal-param ablation complete + save all results and checkpoints` [#msg 812]

### Parallel GPU Runs & Equal-Param Ablation (Jun 19–20)

- GPU underutilized: only 180MB / 4GB used, 16 CPU cores available [#msg 830][#msg 831]
- Launched 4 parallel runs on GPU (8× faster than CPU, ~420 eps/hr each) [#msg 833][#msg 838][#msg 840]
- **trf_coord_equal ablation:** equal-parameter transformer (~140K params vs 138K for IDQN MLP) across 3 rewards [#msg 344][#msg 355]
- Results after all 3 trf_coord_equal runs completed [#msg 851][#msg 852]:

| Agent | DWT | Pressure | Queue |
|-------|-----|----------|-------|
| trf_coord_equal | 32.97s | 37.98s | 37.09s |
| IDQN (reference) | 34.98s | 38.87s | 33.76s |

- Small IDQN GPU run contamination (overwrote `seed_0/` of idqn_grid4x4_dwt) — aggregate regenerated from intact CSVs [#msg 857]

### Output Directory Reorganization

- Consolidated all 20 experiment output dirs from root into `results/{scenario}/{agent}/{reward}/`
- `logs/`, `tb/`, `outputs/` also moved inside `results/` for single top-level results root
- `.gitignore` updated: `outputs` → `results/outputs`, `tb/events.*` → `results/tb/events.*`

---

## Key Files Created/Modified

| File | Purpose |
|------|---------|
| `agents/idqn.py` | CleanRL-style IDQN agent (532 lines) |
| `agents/trf_coord.py` | Spatial coordination transformer agent (421 lines) |
| `env/road_graph.py` | Lane-level adjacency graph from SUMO net |
| `train.py` | Unified training entry point |
| `eval/evaluate.py` | Frozen evaluation loop (277 lines) |
| `eval/metrics.py` | CSV-based metrics (173 lines) |
| `eval/aggregate.py` | rliable wrappers with IQM + CIs (66 lines) |
| `conf/` | Full Hydra config tree (10 files) |
| `baselines/max_pressure.py` | Max-pressure baseline agent |
| `baselines/mplight.py` | MPLight baseline agent |
| `figures/make_figures.py` | Learning curve figure generation |
| `.gitignore` | Updated to track results + checkpoints |

---

## Current Status

| Phase | Status | Notes |
|-------|--------|-------|
| **0** Foundation | ✅ Complete | Gate verified, committed |
| **1** IDQN Agent | ✅ Complete | GATE A results available |
| **2** Transformer | ✅ Complete | GATE B results available |
| **3** Baselines | 🟡 In Progress | Reward sweep + equal-param done; formal comparison pending |
| **4** Figures | 🔴 Not Started | `make_figures.py` exists but not run |
| **5** Write-up | 🔴 Not Started | |

## Next Steps

1. Run Phase 3 formal comparisons with statistical testing
2. Generate Phase 4 figures (learning curves, ablation tables, attention interpretability)
3. Begin Phase 5 manuscript integration
4. Consider re-running cologne3 on GPU for consistent checkpointing
