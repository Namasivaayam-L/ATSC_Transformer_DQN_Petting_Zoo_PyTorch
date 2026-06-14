# Phase 4 — Statistical Validation & Figures (Week 7)

## Context recap
Replace the original paper's qualitative "blue line below red line" with rigorous, regenerable
statistics and figures. This is what separates a journal paper from a project report.

## Prerequisites
- Phase 3 complete: per-seed metric arrays for all methods/scenarios/rewards logged.

## Tasks

### 4.1 — Aggregate performance (rliable)
- IQM (interquartile mean) + performance profiles across seeds for the primary metric.
- Stratified bootstrap 95% CIs. Save vector figures.

### 4.2 — Significance / probability of improvement
- Probability-of-improvement of the proposed method over MPLight and over IDQN (rliable).
- Report effect sizes, not just p-values.

### 4.3 — Learning curves with CI bands
- Mean ± 95% CI training curves (return and average travel time) per method, per scenario.

### 4.4 — Attention interpretability figure (the money shot)
- Heatmap of which neighbours each agent attends to on `grid4x4` (and a real-topology example on
  `cologne3`). Tie specific attention patterns to traffic situations (e.g. attending upstream during a
  platoon arrival). This is the qualitative evidence that coordination is being learned.

### 4.5 — Ablation table rendering
- Render the equal-parameter ablation and any secondary ablations as publication tables.

## Files created/edited
- New: `figures/make_figures.py` (regenerates EVERY figure/table from logged data — reproducibility),
  `figures/` output dir.

## Commands
```bash
uv run python figures/make_figures.py --all
```

## ACCEPTANCE GATE (Phase 4)
Every figure and table in the paper is produced by `figures/make_figures.py` from logged data with no
manual editing, includes CI bands / significance, and the attention figure clearly visualises learned
coordination. Anything hand-tweaked in an image editor FAILS this gate.

## Dependencies
Requires Phase 3 results.
