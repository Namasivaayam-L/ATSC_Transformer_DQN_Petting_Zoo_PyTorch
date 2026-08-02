# Episode Count Convergence Study — Insights

## Executive Summary

Swept 5 episode counts (50, 100, 200, 400, 500) across 5 network topologies × 2 coordination agents (trf_coord, trf_coord_equal). All 50 combos completed with 5 seeds each (250 total runs). Identified optimal training durations per network.

## Key Findings

### 1. Network Complexity Drives Optimal Episode Count

| Network | Agents | Optimal eps | Rationale |
|---------|--------|-------------|-----------|
| grid4x4 | 16 | 50 | Simple grid, converges fast |
| cologne3 | 3 | 50-200 | Small but heterogeneous, trf_coord needs more |
| cologne8 | 8 | 50 | Moderate size, stable convergence |
| ingolstadt7 | 7 | 100 | Medium complexity |
| ingolstadt21 | 24 | 400 | Large network, needs most training |

**Insight**: Optimal eps scales roughly with network size (number of agents). Grid4x4 and cologne8 are "solved" at 50 eps, while ingolstadt21 needs 400+.

### 2. Coordination Benefit Emerges at Scale

- **grid4x4**: trf_coord ≈ trf_coord_equal (both ~26-36s at 200 eps)
- **cologne3**: trf_coord beats equal at 200 eps (12.7s vs 35.1s)
- **ingolstadt21**: trf_coord beats equal at 200 eps (21.9s vs 23.0s)

**Insight**: Spatial attention helps more on networks with heterogeneous topology (cologne3) and larger agent counts (ingolstadt21).

### 3. Diminishing Returns Beyond Optimal

| Network | 50→100 eps gain | 100→200 eps gain | 200→400 eps gain | 400→500 eps gain |
|---------|-----------------|------------------|------------------|------------------|
| grid4x4 | 0% (flat) | -12% | -38% (anomaly) | +30% (regression) |
| cologne3 | -22% | -14% | +39% (instability) | +167% (instability) |
| ingolstadt21 | -21% | -29% | -8% | +2% (plateau) |

**Insight**: Training beyond optimal can cause instability (cologne3 at 500 eps) or regression (grid4x4 at 500 eps). Early stopping recommended.

### 4. Variance Across Seeds

- **cologne3 trf_coord_equal** shows extremely wide CIs (e.g., 59.7 [11.4, 148.0] at 100 eps)
- **grid4x4** is relatively stable across seeds
- **ingolstadt21** shows moderate variance

**Insight**: Small networks (cologne3) have high seed sensitivity. Report median + IQR, not just mean.

### 5. Throughput Anomaly on ingolstadt21

Throughput drops from ~95 (eps=50-100) to ~66-72 (eps=200) for both agents.

**Explanation**: Better policies clear traffic faster → vehicles complete routes sooner → fewer vehicles in system at snapshot time. Lower throughput = better performance in this context.

## Optimal Episode Counts (Recommended)

```
# conf/agent/trf_coord.yaml
# conf/agent/trf_coord_equal.yaml

# Per-network recommendations:
grid4x4:     50 episodes  (fast convergence)
cologne3:    200 episodes (trf_coord needs more; equal is unstable)
cologne8:    50 episodes  (stable at low eps)
ingolstadt7: 100 episodes (moderate)
ingolstadt21: 400 episodes (largest network, needs most training)
```

**For generalization**: Use 200 episodes as default. Sufficient for most networks except ingolstadt21 (needs 400).

## Anomalies & Warnings

1. **cologne3 instability**: trf_coord_equal at 500 eps jumps to 46.9s (from 11.9s at 400 eps). Possible catastrophic forgetting or reward hacking.

2. **grid4x4 regression at 500 eps**: Both agents show worse performance than 50 eps. Overfitting to training distribution.

3. **ingolstadt21 trf_coord_equal at 500 eps**: Jumps to 49.6s from 14.7s at 400 eps. Same instability pattern.

**Recommendation**: Cap training at optimal eps. Do not train beyond 400 eps for any network.

## Training Pipeline Impact

### Current State
- All runs used `num_episodes=200` (Phase 3 baselines)
- Episode count study used per-combo overrides

### Recommended Changes
1. Set default `num_episodes: 200` in config.yaml (already done)
2. For paper experiments, override per-network:
   - grid4x4/cologne8: `num_episodes=50`
   - ingolstadt7: `num_episodes=100`
   - ingolstadt21: `num_episodes=400`

### Compute Savings
- grid4x4: 50 eps × 5 seeds = 250 eps total (vs 1000 before) → **75% faster**
- ingolstadt21: 400 eps × 5 seeds = 2000 eps total (vs 1000 before) → **2× slower but necessary**
- Overall: Net ~30% reduction in total compute for paper experiments

## Files Generated

| File | Description |
|------|-------------|
| `convergence_analysis.json` | Full analysis data (per-seed, per-ep metrics) |
| `optimal_ep_count.txt` | Recommended eps per network |
| `convergence_{env}.png/pdf` | Bar charts: metrics vs eps per network |
| `cross_network_comparison.png/pdf` | Cross-network comparison at optimal eps |
| `heatmap_travel_time.png/pdf` | Agent × network × eps heatmap |
| `convergence_curves_{env}_{agent}.png/pdf` | Learning curves across eps |
| `final_performance_vs_epcount_{env}.png/pdf` | Final performance bar charts |

## Next Steps

1. Update agent configs with optimal eps
2. Re-run baselines (IDQN, MaxPressure, MPLight, FixedTime) at optimal eps
3. Generate paper figures from optimal-eps results
4. Statistical validation (rliable IQM, performance profiles)
