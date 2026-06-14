# Phase 3 — Baselines & Ablations (Weeks 5–6)

## Context recap
A journal reviewer will reject "beats vanilla DQN." We must beat methods the field respects and prove
the transformer's coordination mechanism — not mere capacity — is what wins. This phase produces the
headline results table and the one ablation that is the paper's spine.

## Prerequisites
- GATE B resolved (either `trf_coord` passed, or `trf_temporal` fallback active).

## Tasks

### 3.1 — Baselines (all on identical scenarios / seeds / metrics)
- **Fixed-time** (cyclic, fixed phases) — already used as the Phase-0 sanity controller.
- **Max-Pressure** (classic non-RL benchmark; implement from the standard max-pressure rule).
- **IDQN** (the Phase-1 rebuilt DQN; this is the "no transformer" control).
- **MPLight** (the standard pressure-based MARL/attention baseline; THE one the novelty must beat).
  Use a faithful re-implementation or an established reference; document the source and settings.

### 3.2 — The decisive ablation (paper's spine)
- Coordination transformer **vs.** IDQN-MLP at **equal parameter count**, identical everything else
  (scenario, seeds, reward, training budget). If the transformer wins, the coordination MECHANISM is the
  cause, not extra parameters. Produce this as a standalone table.

### 3.3 — Scenario sweep
- Run all methods on **both** `grid4x4` and `cologne3`, **5 seeds each**.
- Run across all three reward functions (`dwt`, `pressure`, `queue`) — your existing strength, now
  measured properly.

### 3.4 — Secondary ablations (only if time remains)
- Number of attention heads (2 vs 4 vs 8); neighbour radius (1-hop vs 2-hop); coordination variant (a vs b).
- These are nice-to-have; the equal-parameter ablation (3.2) is mandatory.

## Files created/edited
- New: `baselines/fixed_time.py`, `baselines/max_pressure.py`, `baselines/mplight.py`,
  `experiments/run_matrix.py` (orchestrates method × scenario × reward × seed).
- New: `conf/agent/{fixed_time,max_pressure,mplight}.yaml`.

## Commands
```bash
# full matrix (method x scenario x reward x 5 seeds) — schedule overnight on the single GPU:
uv run python experiments/run_matrix.py
# equal-parameter ablation:
uv run python experiments/run_matrix.py +ablation=equal_params
```

## ACCEPTANCE GATE (Phase 3)
A results table with absolute numbers ± 95% CI for every method × scenario × metric exists, and the
equal-parameter ablation table is complete. The proposed method should show a statistically clear
improvement on average travel time over Fixed-time, Max-Pressure, and IDQN; the comparison vs MPLight
is the headline claim (win, or a clearly-argued competitive trade-off). If the method does not beat
MPLight, document honestly and lean on the interpretability + coordination-ablation contribution.

## Dependencies
Requires Phase 2 (the proposed model) and Phase 1 (IDQN baseline + machinery).
