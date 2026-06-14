# 00 — Overview & Execution Order

## Purpose
Turn the existing ATSC repo into a journal-grade, simulation-only contribution. The central
scientific problem with the current work is NOT the idea — it is that the experiment never ran
correctly. This plan first makes the science valid, then makes the architecture novel.

## The diagnosis that motivates this plan (grounded in the actual repo)
The current `experiments/trf_multi_agent/dqn/` agent has bugs that invalidate all prior results:

| # | Bug | Location | Effect |
|---|-----|----------|--------|
| 1 | Only ONE gradient update per episode (learn() only at terminal step, then break) | `dqn/main.py` (training while-loop, terminal branch) | ~1 update/episode → agent barely learns |
| 2 | `self.eval()` called inside `forward()` | `dqn/dqn.py` (DeepQNetwork.forward) | BatchNorm/Dropout never in train mode |
| 3 | No target network; bootstraps off the same net | `dqn/dqn.py` (DQN.learn) | unstable/divergent Q-values |
| 4 | `self.num_states` hardcoded to 80, contradicts config `num_states=8` | `dqn/dqn.py` (DeepQNetwork.__init__) vs `dqn/config.ini` | wrong tensor shapes |
| 5 | Evaluation reuses `train()` → keeps learning + exploring during "test" | `dqn/main.py` (second train() call block) | contaminated results |
| 6 | `num_heads = 1` | `dqn/config.ini` | "multi-head" attention is single-head |
| 7 | Model saved to disk every gradient step | `dqn/dqn.py` (end of DQN.learn) | cripples throughput |
| 8 | Dead TensorFlow SAC + commented-out model graveyard | `sac/sac.py`, `dqn/models.py` | reproducibility hazard; remove |

These are GOOD news: once fixed, there is real upside because the idea was never properly tested.

## What is good and must be kept
- SUMO-RL + PettingZoo + Gymnasium stack (the credible standard for traffic-signal RL).
- `sumo_rl/environment/resco_envs.py` — RESCO benchmark scenarios are already vendored. This is the
  single biggest asset for journal acceptance and is currently underused.
- The three reward functions already wired: `dwt` (difference in waiting time), `pressure`, `queue`.
- `libsumo` fast backend.

## Execution order (8-week sprint)
- **Phase 0 — Foundation & rigor scaffolding** (`01-phase-0-foundation.md`) — Weeks 0–1
- **Phase 1 — RL core done right (CleanRL IDQN)** (`02-phase-1-rl-core.md`) — Weeks 1–2  → **GATE A**
- **Phase 2 — Spatial-coordination transformer** (`03-phase-2-transformer.md`) — Weeks 3–4 → **GATE B**
- **Phase 3 — Baselines & ablations** (`04-phase-3-baselines-ablations.md`) — Weeks 5–6
- **Phase 4 — Statistical validation & figures** (`05-phase-4-stats-figures.md`) — Week 7
- **Phase 5 — Write-up integration** (`06-phase-5-writeup-integration.md`) — Week 8
- Decision gates summarized in `99-decision-gates.md`.

## Global conventions (apply to every phase)
- **Reproducibility first.** Every run is fully specified by a Hydra config; no hardcoded paths or constants.
- **Seeds:** minimum 5 seeds per configuration. Report mean ± 95% CI (bootstrap).
- **Tracking:** log every metric/seed/run to Weights & Biases (academic free tier) or TensorBoard.
- **Never** evaluate with a loop that calls `learn()`. Evaluation = frozen weights, epsilon=0.
- **Primary metric:** average travel time. Secondary: avg waiting time, queue length, throughput, time loss.
- **Determinism:** set seeds for python/numpy/torch and SUMO (`--seed`), and log them.
- **Environment manager:** use `uv` for Python on this machine (`uv run`, `uv pip`, `uv add`). Bare `python` resolves to the Windows Store shim and fails.

## Definition of done for the technical track
A results table with absolute numbers ± 95% CI for every method × scenario × metric, plus an
ablation isolating the coordination transformer against an equal-parameter MLP, plus an attention
interpretability figure — all regenerable from logged data.
