# Episode Count Study — Session Handoff

## What Was the Goal

Determine the optimal training episode count for `trf_coord` and `trf_coord_equal`
agents across 5 network topologies by sweeping episode counts [50, 100, 200, 400, 500].
Identify the "knee" where additional episodes yield diminishing returns.

## Files Created

| File | Purpose |
|------|---------|
| `run_ep_count.sh` | Original sequential script (10 combos × 5 ep_counts × 5 seeds = 250 runs) |
| `run_ep_count_parallel.sh` | **Primary script** — runs up to 8 combos in parallel (no SUMO port conflicts). Handles resume automatically. |
| `eval_ep_count.py` | Post-study analysis — generates convergence plots and optimal ep count recommendations |
| `conf/env/ingolstadt7.yaml` | Env config for ingolstadt7 (7 agents, 3600s episodes) |
| `conf/env/ingolstadt21.yaml` | Env config for ingolstadt21 (24 agents, 3600s episodes) |

## Current Progress: 34/50 Combos Complete

| Ep Count | grid4x4 | cologne3 | cologne8 | ingolstadt7 | ingolstadt21 |
|----------|---------|----------|----------|-------------|--------------|
| **50** | ✅✅ | ✅✅ | ✅✅ | ✅✅ | ✅✅ |
| **100** | ✅✅ | ✅✅ | ✅✅ | ✅✅ | ✅✅ |
| **200** | ✅✅ | ✅✅ | ✅✅ | ✅✅ | ✅✅ |
| **400** | 🔄3 🔄2 | ✅✅ | ✅✅ | 🔄3 🔄4 | 🔄0 🔄0 |
| **500** | 🔄0 🔄0 | 🔄0 ❌ | ❌ ❌ | ❌ ❌ | ❌ ❌ |

✅ = done (5 seeds), 🔄 = partial (seeds_done shown), ❌ = not started

### Incomplete Combos (9)

| Combo | Seeds Done |
|-------|-----------|
| trf_coord_grid4x4_dwt_eps400 | [0,1,2] |
| trf_coord_equal_grid4x4_dwt_eps400 | [0,1] |
| trf_coord_ingolstadt7_dwt_eps400 | [0,1,2] |
| trf_coord_equal_ingolstadt7_dwt_eps400 | [0,1,2,3] |
| trf_coord_ingolstadt21_dwt_eps400 | [0] |
| trf_coord_equal_ingolstadt21_dwt_eps400 | [0] |
| trf_coord_cologne3_dwt_eps500 | [] |
| trf_coord_grid4x4_dwt_eps500 | [] |
| trf_coord_equal_grid4x4_dwt_eps500 | [] |

### Not Started (7)

```
trf_coord_cologne8_dwt_eps500
trf_coord_equal_cologne8_dwt_eps500
trf_coord_ingolstadt7_dwt_eps500
trf_coord_equal_ingolstadt7_dwt_eps500
trf_coord_ingolstadt21_dwt_eps500
trf_coord_equal_ingolstadt21_dwt_eps500
trf_coord_equal_cologne3_dwt_eps500
```

## How to Resume

```bash
bash run_ep_count_parallel.sh
```

- Uses `resume=true` — picks up from `latest.pt` checkpoints
- Skips combos with `aggregate.json` containing 5 seeds
- Runs up to 8 combos concurrently (safe on GTX 1650 with 4GB VRAM)
- Logs go to `results_ep_count/_logs/`

## After All Runs Complete

```bash
.venv/bin/python eval_ep_count.py results_ep_count/ results_ep_count/eval/
```

Outputs: `convergence_analysis.json`, convergence plots per network, `optimal_ep_count.txt`

## Key Notes

- **GPU**: GTX 1650 (4GB VRAM). Each job uses ~100-150 MiB. 8 parallel jobs = ~1200 MiB.
- **CPU**: 16 cores. Each job needs ~2 cores (SUMO sim + PyTorch training). 8 parallel = 16 cores = tight but works.
- **No SUMO port conflicts** — each instance uses its own TraCI TCP connection. Safe to parallelize.
- **Network timing**: grid4x4 eps=50 takes ~80 min (5 seeds). ingolstadt21 eps=500 takes ~14+ hours per combo. The larger combos (ingolstadt21 × eps=500) are the slowest.
- **Resume is robust**: process interrupted at any point, re-running the script picks up cleanly.
- The sequential script `run_ep_count.sh` works too but is much slower.
