 # MASTER BUILD SPEC — Traffic Sense Journal Upgrade

> **What this file is.** This is a self-contained build specification. Paste it into a coding agent (Claude Code, Cursor, etc.) and it will create a `journal-upgrade/` directory tree containing a complete, phase-wise execution plan plus a non-technical backlog. The plans are detailed enough to begin executing immediately.
>
> **How to use it (instructions to the coding agent — follow exactly):**
>
> 1. You are operating inside the research folder: `C:\Users\NLakshmanaSwamy\OneDrive - Knex\Documents\IAB\POC\research`. The repo to be upgraded is the sibling folder `ATSC_Transformer_DQN_Petting_Zoo_PyTorch` (clone it from `https://github.com/Namasivaayam-L/ATSC_Transformer_DQN_Petting_Zoo_PyTorch` if not present).
> 2. Run the **single bash script** in the "BUILD SCRIPT" section below, verbatim, from the research folder root. It is written for **Git Bash on Windows** and uses quoted heredocs (`<<'EOF'`) so nothing inside is shell-expanded. It is idempotent (`mkdir -p`, overwrites files).
> 3. After the script completes, it prints the created tree. Confirm it matches the "EXPECTED TREE" section.
> 4. Then open `journal-upgrade/plans/00-overview.md` and begin Phase 0. **Do not ask clarifying questions** — every decision is already made and recorded in the plan files. Execute phases in order, respecting each phase's acceptance gate before proceeding.
>
> **Authoritative decisions already made (do not revisit):**
> - Target: SCI/Scopus-indexed journal. RL core rebuilt on **CleanRL** single-file style (PyTorch). Novelty = **spatial neighbor-coordination transformer**, with **temporal history-attention as the Week-4 fallback**. Scenarios = **RESCO `grid4x4` + `cologne3`**. Primary metric = **average travel time**. Scope = **simulation only (CV/DETR/dlib cut)**. Timeline = **<2 months, single GPU**. Baselines = **Fixed-time, Max-Pressure, IDQN, MPLight**.

---

## EXPECTED TREE

```
journal-upgrade/
├── README.md
├── plans/
│   ├── 00-overview.md
│   ├── 01-phase-0-foundation.md
│   ├── 02-phase-1-rl-core.md
│   ├── 03-phase-2-transformer.md
│   ├── 04-phase-3-baselines-ablations.md
│   ├── 05-phase-4-stats-figures.md
│   ├── 06-phase-5-writeup-integration.md
│   └── 99-decision-gates.md
├── backlog/
│   ├── 00-backlog-overview.md
│   ├── 01-publication-status-check.md
│   ├── 02-target-journal-shortlist.md
│   ├── 03-authorship-advisor.md
│   ├── 04-reproducibility-package.md
│   ├── 05-compute-budget.md
│   ├── 06-licensing-ethics.md
│   └── 07-submission-calendar.md
└── memory/
    ├── 00-session-reproduction-guide.md
    ├── 01-session-context.md
    ├── 02-decisions-log.md
    ├── 03-options-presented.md
    ├── 04-diagnosis-and-suggestions.md
    └── 05-open-tensions-and-risks.md
```

> **memory/** captures everything needed to reproduce this planning session in a fresh chat: the full
> repo + paper context, every decision and the exact value chosen, every option that was offered (incl.
> the roads not taken), the technical diagnosis/suggestions, and the unresolved tensions. Start a new
> session by feeding the agent `memory/00-session-reproduction-guide.md`.

---

## BUILD SCRIPT

Run this whole block from the research folder root (`.../IAB/POC/research`).

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="journal-upgrade"
mkdir -p "$ROOT/plans" "$ROOT/backlog" "$ROOT/memory"
cd "$ROOT"

# ============================================================
# README.md
# ============================================================
cat > README.md <<'EOF'
# Journal Upgrade — Traffic Sense (Simulation Track)

This directory holds the full execution plan and non-technical backlog for upgrading
the "Traffic Sense: Optimizing City Traffic with Transformer-infused DRL and Computer Vision"
project from a Final-Year-Project paper into a journal-grade submission.

## Locked decisions
- **Target:** SCI/Scopus-indexed journal
- **RL core:** CleanRL single-file (PyTorch), rebuilt from scratch
- **Novelty:** Spatial neighbor-coordination transformer (temporal history-attention = fallback)
- **Scenarios:** RESCO `grid4x4` (16 agents) + `cologne3` (real topology)
- **Primary metric:** average travel time (secondary: waiting time, queue, throughput, time loss)
- **Scope:** simulation only — computer-vision (DETR/dlib/video) is CUT
- **Timeline:** <2 months, single GPU — ruthless prioritization
- **Baselines:** Fixed-time, Max-Pressure, IDQN, MPLight

## How to execute
1. Read `plans/00-overview.md`.
2. Execute phases in order: `01` → `06`. Each phase has an acceptance gate that MUST pass before the next.
3. `plans/99-decision-gates.md` holds the two hard go/no-go checkpoints (Week-2 and Week-4).
4. Work the `backlog/` items in parallel — `backlog/01-publication-status-check.md` is the highest-priority non-technical risk and must be resolved in Week 0.

## Resuming in a fresh session
`memory/` reconstructs this entire planning session: start by reading `memory/00-session-reproduction-guide.md`,
which points to the context, decisions, the options that were offered, the diagnosis, and the open tensions.

## Repo under upgrade
`../ATSC_Transformer_DQN_Petting_Zoo_PyTorch`
(GitHub: https://github.com/Namasivaayam-L/ATSC_Transformer_DQN_Petting_Zoo_PyTorch)

## The thesis the paper will defend
> Equipping each intersection's DRL agent with a self-attention module over its neighbouring
> intersections' states enables learned traffic coordination that outperforms independent agents
> and pressure-based methods on average travel time, with attention weights providing interpretable
> evidence of which neighbours drive each signal's decisions.
EOF

# ============================================================
# plans/00-overview.md
# ============================================================
cat > plans/00-overview.md <<'EOF'
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
EOF

# ============================================================
# plans/01-phase-0-foundation.md
# ============================================================
cat > plans/01-phase-0-foundation.md <<'EOF'
# Phase 0 — Foundation & Rigor Scaffolding (Weeks 0–1)

## Context recap (why this phase exists)
Journal acceptance is won or lost on rigor, not novelty. Before touching the model we build the
reproducibility, tracking, evaluation, and metrics infrastructure. The current repo has none of this:
configs are `.ini` files with hardcoded absolute paths (e.g. a `/home/namachu/...` sys.path append in
`dqn/main.py`), results are loose CSVs, and "evaluation" reuses the training loop.

## Prerequisites
- Repo cloned at `../ATSC_Transformer_DQN_Petting_Zoo_PyTorch`.
- SUMO installed and `SUMO_HOME` set; `libsumo` importable.
- `uv` available for Python execution.

## Tasks

### 0.1 — Strip the repo to a clean core
DELETE (move to a `legacy/` folder rather than hard-delete, to preserve history):
- `experiments/trf_multi_agent/sac/` (dead TensorFlow SAC; mixes TF with the PyTorch codebase).
- `experiments/trf_multi_agent/dqn/models.py` (entirely commented-out model graveyard).
- `experiments/trf_dqn/` (single-agent leftover; superseded by the multi-agent track).
Remove the hardcoded absolute `sys.path.append('/home/namachu/...')` in `dqn/main.py`; replace with a
proper package install (`uv pip install -e .`) or a relative path resolved at runtime.

### 0.2 — Adopt Hydra for configuration
- `uv add hydra-core omegaconf`.
- Create `conf/` with a structured config: `conf/config.yaml` (defaults) + groups
  `conf/env/{grid4x4,cologne3}.yaml`, `conf/agent/{idqn,trf_coord,trf_temporal}.yaml`,
  `conf/reward/{dwt,pressure,queue}.yaml`.
- Every former `.ini` key becomes a typed config field. No constant may be hardcoded in `.py`.
- Each run writes its resolved config into its output dir (Hydra does this automatically).

### 0.3 — Wire experiment tracking
- `uv add wandb` (or use TensorBoard if offline). `wandb login` with the academic account.
- Log: all five metrics per episode, epsilon, loss, seed, full config, and (later) attention maps.
- Group runs by `scenario/method/reward` and tag with the seed.

### 0.4 — Build the metrics module
Create `eval/metrics.py` that parses SUMO's `tripinfo.xml` / `summary.xml` outputs and computes:
- **average travel time** (primary), average waiting time, mean queue length, throughput
  (vehicles that completed their trip), total time loss.
- Provide both per-episode and aggregate-over-seeds (mean ± 95% bootstrap CI) functions.
- Unit-test the parser on a tiny recorded run.

### 0.5 — Separate, frozen evaluation loop
Create `eval/evaluate.py`:
- Loads a checkpoint, sets `epsilon = 0`, runs N evaluation seeds, NEVER calls `learn()` or updates weights.
- Writes per-seed metrics to disk and logs aggregates to W&B.
- This permanently fixes the contaminated-evaluation bug (#5).

### 0.6 — Aggregate-statistics tooling
- `uv add rliable`.
- Create `eval/aggregate.py` that consumes per-seed metric arrays and produces IQM, performance
  profiles, and stratified-bootstrap 95% CIs (the NeurIPS-standard results methodology).

### 0.7 — Determinism harness
Create `utils/seeding.py` setting seeds for `random`, `numpy`, `torch` (+ `torch.use_deterministic_algorithms`
where feasible) and passing `--seed` to SUMO. Call it at the start of every entry point.

## Files created/edited in this phase
- New: `conf/**`, `eval/metrics.py`, `eval/evaluate.py`, `eval/aggregate.py`, `utils/seeding.py`.
- Edited: `experiments/trf_multi_agent/dqn/main.py` (remove hardcoded path; read Hydra config).
- Moved: dead code → `legacy/`.

## Commands
```bash
uv add hydra-core omegaconf wandb rliable
uv pip install -e .
# smoke test the fixed-time baseline end-to-end (see Phase 1 for the agent; here just prove the pipeline):
uv run python -m eval.evaluate agent=fixed_time env=grid4x4 seeds=5
```

## ACCEPTANCE GATE (Phase 0)
A do-nothing **fixed-time** controller runs end-to-end on `grid4x4`, and the pipeline logs all five
metrics to W&B across 5 seeds with 95% CI bands, fully driven by a Hydra config with zero hardcoded
paths. If any metric cannot be computed or any path is hardcoded, this gate FAILS — fix before Phase 1.

## Dependencies
None (this is the first phase). Everything downstream depends on this.
EOF

# ============================================================
# plans/02-phase-1-rl-core.md
# ============================================================
cat > plans/02-phase-1-rl-core.md <<'EOF'
# Phase 1 — RL Core Done Right (CleanRL IDQN) (Weeks 1–2)

## Context recap
Rebuild the agent so results are trustworthy, using CleanRL single-file style for transparency and
citability. NO transformer yet — this phase produces the credible MLP baseline (IDQN) and proves the
rebuilt core actually learns. This directly fixes bugs #1, #2, #3, #4, #7.

## Prerequisites
- Phase 0 complete and its gate passed (Hydra, W&B, metrics, frozen eval, seeding all working).

## Tasks

### 1.1 — Single-file IDQN agent (CleanRL style)
Create `agents/idqn.py` as a self-contained Independent-DQN over the PettingZoo parallel env:
- One Q-network per agent (shared architecture, independent weights), MLP only.
- Input dim derived from the env observation space (NOT hardcoded). Delete any `num_states=80` constant.
- Replay buffer per agent (or a keyed shared buffer); store (s, a, r, s', done).

### 1.2 — Fix the training cadence (bug #1)
- Gradient update every `train_freq` environment steps (e.g. every 1–4 steps) AFTER a warmup of
  `learning_starts` steps, sampling a minibatch each time. Do NOT gate learning on episode termination.
- Track updates-per-episode in W&B to confirm the fix (should be hundreds–thousands, not 1).

### 1.3 — Target network + Double DQN (bugs #2, #3)
- Add a separate target network; sync via hard update every `target_update_interval` steps OR Polyak (tau).
- Implement **Double DQN** target: action selected by online net, evaluated by target net.
- REMOVE the `self.eval()` call from inside `forward()`. Manage train/eval mode explicitly:
  `model.train()` during the update, `model.eval()` only inside the frozen evaluation loop.
- Optionally add a **Dueling** head (value/advantage decomposition) behind a config flag.

### 1.4 — Checkpointing (bug #7)
- Save checkpoints every `save_interval` updates (or best-eval), NOT every gradient step.

### 1.5 — Action / observation sanity
- Confirm the action space matches the env (per-agent discrete phase selection). Document the
  state→action mapping in a docstring (the original paper conflated an 8-dim state with a length-4 action;
  make the real shapes explicit and correct here).

### 1.6 — Reward functions
- Wire `dwt`, `pressure`, `queue` from the env via Hydra `reward` group. Verify each returns sane scales.

## Files created/edited
- New: `agents/idqn.py`, `agents/replay_buffer.py`, `train.py` (Hydra entry point), `conf/agent/idqn.yaml`.
- Edited/retired: the old `dqn/dqn.py`, `dqn/main.py`, `dqn/memory.py` logic is superseded; keep under
  `legacy/` for reference, do not import from them.

## Commands
```bash
uv run python train.py agent=idqn env=grid4x4 reward=dwt seed=0
# sweep seeds:
for s in 0 1 2 3 4; do uv run python train.py agent=idqn env=grid4x4 reward=dwt seed=$s; done
uv run python -m eval.evaluate agent=idqn env=grid4x4 reward=dwt seeds=5
```

## ACCEPTANCE GATE A (hard go/no-go — see 99-decision-gates.md)
On `grid4x4`, the rebuilt **IDQN beats Fixed-time and approaches Max-Pressure** on average travel time,
across 5 seeds with non-overlapping/clearly-better CIs vs Fixed-time. Updates-per-episode is in the
hundreds+. If IDQN does not beat Fixed-time, STOP and debug here — nothing downstream can succeed until
the core learns. This gate protects the entire timeline.

## Dependencies
Requires Phase 0 (config, metrics, frozen eval, seeding).
EOF

# ============================================================
# plans/03-phase-2-transformer.md
# ============================================================
cat > plans/03-phase-2-transformer.md <<'EOF'
# Phase 2 — Spatial Neighbour-Coordination Transformer (Weeks 3–4)

## Context recap
This is the novelty and the fix for "I used the transformer for the sake of using it." A self-attention
module over an 8-scalar vector has no meaningful long-range dependency to capture. We make the state
structured so attention is justified: each agent attends over the states of its NEIGHBOURING
intersections, learning coordination. This makes "long-range dependency" literally true (spatial,
across the road graph) and yields an interpretability story (attention over neighbours).

## Prerequisites
- GATE A passed (IDQN beats Fixed-time). Do NOT start the transformer on an unproven core.

## Tasks

### 2.1 — Build the road-graph neighbourhood
Create `env/road_graph.py`:
- Parse the scenario `.net.xml` to build the intersection adjacency graph (which traffic lights are
  connected by a road segment).
- For each agent, produce an ordered neighbour list (self + adjacent intersections). Cache it.

### 2.2 — Tokenised observation
- Wrap the env (or post-process observations) so each agent's input is a set of tokens:
  `[self_obs, neighbour_1_obs, ..., neighbour_k_obs]`, each token = that intersection's per-lane vector.
- Add an identity/positional encoding per token (self vs neighbour-i). Pad/mask variable neighbour counts.

### 2.3 — Coordination transformer Q-network
Create `agents/trf_coord.py`:
- `nn.TransformerEncoder` over the neighbour tokens with **multi-head** attention (num_heads > 1; the old
  `num_heads=1` is explicitly rejected). `norm_first=True`, GELU, dropout.
- Pool the self-token (or attention-pooled summary) → Q-head over the agent's discrete actions.
- **Match parameter count to the Phase-1 MLP** as closely as practical, so the later ablation is fair.
- Reuse the exact same Double-DQN + target-network + train-cadence machinery from Phase 1 (only the
  network body changes).

### 2.4 — Two coordination variants to A/B
- (a) attend over neighbours' **current** observations.
- (b) additionally include neighbours' **last action / current phase** as part of each neighbour token
  (richer coordination signal). Config flag selects the variant.

### 2.5 — Attention logging for interpretability
- Expose per-head attention weights; log periodic snapshots to W&B for the heatmap figure in Phase 4.

## Files created/edited
- New: `env/road_graph.py`, `agents/trf_coord.py`, `conf/agent/trf_coord.yaml`,
  `conf/agent/trf_temporal.yaml` (fallback, see below).
- Edited: observation wrapper in `env/` to emit tokenised neighbour observations.

## Commands
```bash
uv run python train.py agent=trf_coord env=grid4x4 reward=dwt seed=0 agent.num_heads=4
for s in 0 1 2 3 4; do uv run python train.py agent=trf_coord env=grid4x4 reward=dwt seed=$s; done
uv run python -m eval.evaluate agent=trf_coord env=grid4x4 reward=dwt seeds=5
```

## DECISION GATE B (Week 4 — hard checkpoint; see 99-decision-gates.md)
- **PASS:** the spatial transformer trains cleanly and **>= matches IDQN** within ~2 weeks of tuning →
  spatial coordination is the paper's contribution. Proceed to Phase 3 with `trf_coord`.
- **FALL BACK:** if it is unstable / not converging / consuming all compute → switch to the TEMPORAL
  fallback below. You lose ~3 days, not the deadline.

### Temporal history-attention fallback (`agents/trf_temporal.py`)
- State per agent = sequence of the last K timesteps of that intersection's own observation.
- Transformer attends over time; justifies "long-range temporal dependency" and addresses partial
  observability (ties to the DTQN reference in the original paper).
- Lower engineering risk, still a real, publishable contribution. Use the SAME Phase-1 RL machinery.

## Dependencies
Requires Phase 1 (the RL machinery and the IDQN baseline it must match/beat).
EOF

# ============================================================
# plans/04-phase-3-baselines-ablations.md
# ============================================================
cat > plans/04-phase-3-baselines-ablations.md <<'EOF'
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
EOF

# ============================================================
# plans/05-phase-4-stats-figures.md
# ============================================================
cat > plans/05-phase-4-stats-figures.md <<'EOF'
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
EOF

# ============================================================
# plans/06-phase-5-writeup-integration.md
# ============================================================
cat > plans/06-phase-5-writeup-integration.md <<'EOF'
# Phase 5 — Write-up Integration (Week 8 + buffer)

## Context recap
Fold the real results into the manuscript and fix the correctness/clarity defects flagged in the
original paper. This phase also absorbs schedule overruns.

## Prerequisites
- Phase 4 figures/tables generated.

## Tasks

### 5.1 — Rewrite Methods
- Describe the spatial-coordination transformer precisely: tokenisation, neighbour graph, multi-head
  attention, Q-head, and the Double-DQN/target-network training (which the original CLAIMED via the
  theta-prime in its loss equation but did not implement).
- State the true observation and action shapes (the original conflated an 8-dim state with a length-4
  action and misused "max-pooling" for "argmax"). Correct all of this.

### 5.2 — Fix the equations
- Bellman optimality (original Eq. 1) is missing the summation over (s', r): restore the expectation/sum.
- DQN loss (original Eq. 3) is missing the expectation/mean over the minibatch: write it correctly with
  the target-network parameters.

### 5.3 — Rewrite Results around absolute numbers
- Lead every table with average travel time ± 95% CI. Report secondary metrics. Replace qualitative
  plot descriptions with the rliable figures and significance statements.

### 5.4 — Fix figure hygiene
- The original reused the IDENTICAL caption ("architecture of our Transformer-based DQN model…") for
  Figs. 3, 4, 7, 8, 9. Give every figure a unique, accurate caption. Remove placeholder figures.

### 5.5 — Reconcile narrative with scope
- Remove or reframe the computer-vision (DETR/dlib) sections — CV is cut. Move any CV mention to a brief
  future-work/deployment paragraph only.
- Fix the OpenCV-vs-dlib tooling contradiction by simply not making real-time CV claims.

### 5.6 — Limitations & ethics
- Add sim-to-real gap, no real-world deployment claims, fairness across directions, and the scope of the
  scenarios (grid4x4 + cologne3).

### 5.7 — Reproducibility statement
- Point to the cleaned repo, configs, seeds, and the one-command figure regeneration.

## Files created/edited
- The manuscript (LaTeX). New: `paper/` if the manuscript is brought into this tree, else edit in place.

## ACCEPTANCE GATE (Phase 5)
The manuscript's every numeric claim is backed by a logged result; equations are correct; every figure
has a unique accurate caption; CV claims are removed/reframed; a reproducibility statement is present.
Run a similarity check (see backlog) before declaring done.

## Dependencies
Requires Phase 4. Also depends on backlog item 01 (publication-status) being resolved so the framing
(new submission vs. substantial extension) is correct.
EOF

# ============================================================
# plans/99-decision-gates.md
# ============================================================
cat > plans/99-decision-gates.md <<'EOF'
# 99 — Decision Gates (Hard Checkpoints)

Two gates protect the <2-month timeline. Treat them as go/no-go; do not proceed past a failed gate by
"hoping it works out later."

## GATE A — end of Phase 1 (≈ Week 2): Does the rebuilt core learn?
- **Criterion:** On `grid4x4`, rebuilt IDQN beats Fixed-time on average travel time across 5 seeds with
  clearly-better CIs, and updates-per-episode are in the hundreds+.
- **PASS →** proceed to Phase 2 (transformer).
- **FAIL →** STOP. Debug the RL core (most likely culprits: training cadence, target network, eval-mode
  leakage, observation shape). Nothing downstream can succeed until this passes. Do not start the
  transformer.

## GATE B — Week 4: Does the spatial transformer train?
- **Criterion:** `trf_coord` trains stably and at least matches IDQN within ~2 weeks of tuning.
- **PASS →** spatial coordination is the contribution; proceed to Phase 3 with `trf_coord`.
- **FALL BACK →** switch to `trf_temporal` (history-attention). Lower risk, still publishable. You lose
  ~3 days, not the deadline. Proceed to Phase 3 with `trf_temporal`.
- **Rule:** make this call ON Week 4. Do not let transformer tuning consume Weeks 5–6 (reserved for
  baselines/ablations).

## Standing rule
If at any point the full method×scenario×reward×seed matrix will not fit the compute budget before the
deadline, cut SECONDARY ablations first, then drop `cologne3` to a single reward, before ever cutting
seeds below 5 or dropping a mandatory baseline (Fixed-time, Max-Pressure, IDQN, MPLight). Log every cut
explicitly in the paper's experimental-setup section — never silently truncate.
EOF

# ============================================================
# backlog/00-backlog-overview.md
# ============================================================
cat > backlog/00-backlog-overview.md <<'EOF'
# Backlog — Non-Technical Work (the stuff that sinks papers even when the tech is solid)

These run in PARALLEL with the technical phases. Priority order:

1. `01-publication-status-check.md` — **HIGHEST RISK. Resolve in Week 0.**
2. `02-target-journal-shortlist.md` — Week 0–1.
3. `03-authorship-advisor.md` — Week 0.
4. `04-reproducibility-package.md` — ongoing, finalize by Phase 5.
5. `05-compute-budget.md` — Week 0 (gates the whole experiment matrix).
6. `06-licensing-ethics.md` — by Phase 5.
7. `07-submission-calendar.md` — Week 0, then maintained.
EOF

# ============================================================
# backlog/01-publication-status-check.md
# ============================================================
cat > backlog/01-publication-status-check.md <<'EOF'
# Backlog 01 — Prior-Publication / Self-Plagiarism Check  [HIGHEST PRIORITY]

## Why this is the biggest non-technical risk
The repo's `images/` folder contains an email titled "[ACCEPTED] Notification from ADICS - 2024".
The original paper was apparently ACCEPTED somewhere. If it was actually PUBLISHED, submitting a close
variant to a journal risks a dual-submission / self-plagiarism rejection (and can damage reputation).

## Actions
- Determine definitively: was the original paper **published** (conference proceedings / indexed), or
  merely **accepted and then withdrawn / not published**? Get documentary proof.
- If PUBLISHED:
  - This journal submission MUST be a clearly-extended work. Most journals require a substantial
    increment (commonly ~30%+ new content). The new RL core, the spatial-coordination novelty, RESCO
    benchmarks, proper baselines, and statistical validation comfortably provide this — but the
    extension must be explicitly stated and the prior paper cited.
  - Check the target journal's specific policy on extended conference papers.
- If NOT published:
  - Freer to submit as new work, but still run a similarity check (backlog 04 / Phase 5).

## Definition of done
A written, evidence-backed determination of the original paper's status, and a one-line framing decision:
"new submission" vs. "substantial extension of [citation]". Phase 5 write-up depends on this.
EOF

# ============================================================
# backlog/02-target-journal-shortlist.md
# ============================================================
cat > backlog/02-target-journal-shortlist.md <<'EOF'
# Backlog 02 — Target Journal Shortlist & Scope-Fit

## Actions
- Shortlist 2–3 candidate venues, e.g.: IEEE Transactions on ITS (T-ITS), IEEE Open Journal of ITS,
  Transportation Research Part C, Applied Intelligence. (Confirm current scope/indexing yourself.)
- For each: read the scope statement + one recent ATSC/RL paper to calibrate the expected bar
  (baselines, scenarios, statistical rigor).
- Record: page/length limits, open-access fees, code/data availability requirements, review timeline.
- Pick a primary + a backup before writing (Phase 5 formatting depends on the chosen template).

## Definition of done
A short table of 2–3 venues with scope-fit notes, fees, length limits, and a chosen primary target.
EOF

# ============================================================
# backlog/03-authorship-advisor.md
# ============================================================
cat > backlog/03-authorship-advisor.md <<'EOF'
# Backlog 03 — Authorship & Advisor Sign-off

## Actions
- Confirm co-authorship and author order (the original lists Dr. J. Angela Jennifa Sujana and
  L. Namasivaayam). Agree contribution statements.
- Get explicit advisor buy-in on the NEW direction (simulation-only, dropped CV, spatial-coordination
  novelty, RESCO benchmarks) BEFORE major work — this is a material change from the original paper.
- Agree on the target venue (links to backlog 02).

## Definition of done
Written agreement on authors, order, and the new scope/direction.
EOF

# ============================================================
# backlog/04-reproducibility-package.md
# ============================================================
cat > backlog/04-reproducibility-package.md <<'EOF'
# Backlog 04 — Reproducibility Package

## Actions
- Clean `README` with exact setup (uv, SUMO_HOME, libsumo), and a one-command path to reproduce a
  headline result and to regenerate all figures (`figures/make_figures.py --all`).
- Pin the environment (`pyproject.toml` / lockfile). Record SUMO version.
- Commit all Hydra configs and the seed list used for the paper.
- Mint a code DOI (e.g. Zenodo GitHub release) for the submission.
- Run a similarity / plagiarism scan (Turnitin / iThenticate) on the manuscript before submission —
  especially important given reused prose from the FYP.

## Definition of done
A reviewer can clone, install, and reproduce a headline number and all figures from the committed
configs + seeds; a DOI exists; similarity scan passed.
EOF

# ============================================================
# backlog/05-compute-budget.md
# ============================================================
cat > backlog/05-compute-budget.md <<'EOF'
# Backlog 05 — Compute Budget Reality-Check

## Why (Week 0)
The plan targets <2 months on a single GPU. The experiment matrix (methods × scenarios × rewards × 5
seeds × episodes) must be estimated up front or the timeline silently breaks.

## Actions
- Estimate GPU-hours: count configurations × episodes × wall-clock per episode (measure one short run
  with `libsumo`). Multiply out.
- Confirm it fits the available GPU before the deadline. If not, secure backup: lab cluster, Colab Pro,
  or Kaggle.
- Use `libsumo` (not TraCI) and run seeds in parallel where the GPU allows. Cache the road graph.
- Define the cut-order (from `99-decision-gates.md` standing rule) if compute runs short.

## Definition of done
A GPU-hour estimate, a confirmed fit (or a secured backup), and a documented cut-order.
EOF

# ============================================================
# backlog/06-licensing-ethics.md
# ============================================================
cat > backlog/06-licensing-ethics.md <<'EOF'
# Backlog 06 — Dataset/Asset Licensing & Ethics

## Actions
- Confirm RESCO scenario licenses and that the `.net.xml` / route files can be redistributed in the repo.
- Cite SUMO and RESCO correctly.
- Prepare the limitations/ethics content for the paper: sim-to-real gap, no real-world deployment claims
  (CV is cut), fairness across traffic directions, and the limited scenario scope.

## Definition of done
License compliance confirmed for all redistributed assets; ethics/limitations content drafted.
EOF

# ============================================================
# backlog/07-submission-calendar.md
# ============================================================
cat > backlog/07-submission-calendar.md <<'EOF'
# Backlog 07 — Submission Calendar

## Actions
- Build a realistic calendar beyond the 8-week build: internal review → advisor review → revision →
  submission. Add 2–3 weeks buffer.
- Map the two decision gates (Week 2, Week 4) onto the calendar as explicit milestones.
- Track journal review timelines (from backlog 02) to set expectations for decisions/revisions.

## Definition of done
A dated calendar from today through submission, including the decision gates and review/revision buffer.
EOF

# ============================================================
# memory/00-session-reproduction-guide.md
# ============================================================
cat > memory/00-session-reproduction-guide.md <<'EOF'
# 00 — Session Reproduction Guide

## What this folder is
A complete record of the planning session that produced this `journal-upgrade/` tree. Its purpose is to
let a NEW chat session (or a future you) reconstruct exactly where we landed and WHY, without re-deriving
anything. If you are an agent resuming this work, read these files in order before acting.

## Read order
1. `01-session-context.md` — the paper, the repo, the tech stack, the goal. The "where we started".
2. `02-decisions-log.md` — every decision made and the exact value chosen. The "what we settled on".
3. `03-options-presented.md` — every option that was offered for each decision, including the ones NOT
   chosen and the trade-off previews. The "what else was on the table".
4. `04-diagnosis-and-suggestions.md` — the technical diagnosis (repo bugs, paper flaws) and the package/
   method recommendations. The "why the plan looks the way it does".
5. `05-open-tensions-and-risks.md` — unresolved tensions and the single biggest risk. The "watch out for".

## How to reproduce the session in a fresh chat
Paste this prompt to a new agent:

> Read `journal-upgrade/memory/` files 00→05 in order, then `journal-upgrade/plans/00-overview.md`. The
> repo under upgrade is `../ATSC_Transformer_DQN_Petting_Zoo_PyTorch`. All planning decisions are already
> made and recorded — do not re-ask them. Confirm you understand the locked decisions and the two
> decision gates, then continue from wherever the plans/ phases were last left off.

## One-line summary of the whole session
Diagnosed that the original FYP paper's transformer-DQN was never properly trained (a cluster of RL
bugs), then planned a simulation-only, journal-grade rebuild on CleanRL with a spatial neighbour-
coordination transformer as the real novelty, benchmarked on RESCO grid4x4 + cologne3, primary metric
average travel time, under a <2-month single-GPU budget with a Week-4 fallback to a temporal transformer.

## Provenance
- Original paper: `fyp_journal_ieee.pdf` (11 pages, IEEE format, dated 2024-12-16). Title: "Traffic Sense:
  Optimizing City Traffic with Transformer-infused DRL and Computer Vision."
- Repo: https://github.com/Namasivaayam-L/ATSC_Transformer_DQN_Petting_Zoo_PyTorch
- Planning session date: 2026-06-13.
EOF

# ============================================================
# memory/01-session-context.md
# ============================================================
cat > memory/01-session-context.md <<'EOF'
# 01 — Session Context (Where We Started)

## The original paper (consumed in full, all 11 pages)
- Title: "Traffic Sense: Optimizing City Traffic with Transformer-infused DRL and Computer Vision."
- Authors: Dr. J. Angela Jennifa Sujana (HOD) and L. Namasivaayam, Mepco Schlenk Engineering College.
- IEEE-format LaTeX, created 2024-12-16. Reads as a Final-Year-Project write-up.
- Core claim: a DQN with a transformer layer ("TRF-DQN") beats vanilla DQN at minimizing vehicle waiting
  time at signalized intersections. State extracted from SUMO during training and from real traffic
  camera video via DETR + dlib tracking at inference. Evaluated in SUMO; "implemented" on real city
  camera footage with a homegrown multi-video simulator.
- Reward functions: difference-in-waiting-time, pressure-based, queue-based.
- The author's own framing in this session: "I just used the transformer layer for the sake of using it.
  I had no idea what I was doing back then."

## The author's goal for this session
Make the work PUBLISHABLE in an indexed journal. Specifically:
- Make it technically strong; get the metrics base and achieved (ABSOLUTE quantification).
- Author was confused about which packages/methods to use → wanted sophisticated, defensible recommendations.
- FOCUS ON SIMULATION ONLY (the computer-vision half is set aside).
- Wanted a detailed phase-wise plan + a non-technical backlog, delivered as a build spec.

## The repo and its ACTUAL tech stack (read directly, not assumed)
Repo: ATSC_Transformer_DQN_Petting_Zoo_PyTorch.
- **Good foundation (keep):** SUMO-RL (vendored under `sumo_rl/`) + PettingZoo 1.24.3 + Gymnasium 0.29.1,
  PyTorch 2.3.0, `libsumo` 1.20.0 fast backend. `sumo_rl/environment/resco_envs.py` is present — RESCO is
  THE standard traffic-signal RL benchmark and was barely used.
- **Active model:** `experiments/trf_multi_agent/dqn/dqn.py` — a TransformerEncoder (embedding → encoder →
  5 FC layers → Q-values), RMSprop, HuberLoss. Config: 2x2 grid, num_states=8, num_heads=1, num_enc_layers=2,
  embedding_dim=32, width=40, batch_size=300, 2000 episodes.
- **Dead code (cut):** `experiments/trf_multi_agent/sac/sac.py` (TensorFlow SAC, continuous actions forced
  onto a discrete problem; TF deps not even in requirements). `experiments/trf_multi_agent/dqn/models.py`
  (entirely commented-out graveyard of MLP/LSTM/CNN/Transformer experiments). `experiments/trf_dqn/`
  (single-agent leftover).
- **Missing:** no stable-baselines3, no RLlib, no Tianshou, no wandb/tensorboard, no Hydra, no Double/Dueling
  DQN, no target network, no statistical tooling. Config via `.ini` files + a hardcoded `/home/namachu/...`
  absolute path in `dqn/main.py`.

## What was NOT in scope
- The computer-vision pipeline (DETR/dlib/video) — explicitly cut for this paper.
- Syntax/prose issues in the paper — explicitly set aside by the author ("we can set aside the syntax issues").
EOF

# ============================================================
# memory/02-decisions-log.md
# ============================================================
cat > memory/02-decisions-log.md <<'EOF'
# 02 — Decisions Log (What We Settled On)

Every decision the author made in this session, with the exact value chosen. These are LOCKED; do not
re-litigate. (The full menus, including unchosen options, are in `03-options-presented.md`.)

| # | Decision | CHOSEN VALUE | Notes |
|---|----------|--------------|-------|
| 1 | Architectural scope | **"Recommend what fits"** | Author deferred to assessment rather than pre-committing to keep-DQN vs upgrade-MARL. |
| 2 | Publication bar | **Indexed journal (SCI/Scopus)** | The most demanding tier: strong baselines, stats, ablations, multiple scenarios. |
| 3 | Simulation setup | **"Whatever's most rigorous"** | Author wanted the most defensible/reproducible sim+benchmark stack even if it changes the repo. |
| 4 | RL library philosophy | **CleanRL** (single-file, transparent, citable) | Chosen over Tianshou and RLlib. Multi-agent is DIY but full control of the agent. |
| 5 | Transformer novelty angle | **Spatial: neighbour coordination** | Chosen over temporal-only and spatio-temporal. The "most engineering effort" option. |
| 6 | Timeline / compute | **Tight (<2 months), limited compute (single GPU)** | Forces ruthless scoping. |
| 7 | Risk gate handling | **Build spatial, temporal as fallback** | Explicit Week-4 decision gate; drop to temporal history-attention if spatial isn't training cleanly. |
| 8 | Computer-vision scope | **Simulation only — cut CV entirely** | Drop DETR/dlib/video from this paper. |
| 9 | Plan deliverable format | **"Just show me here first"** (then → build-spec .md) | Author first wanted it in chat; then asked for a single master build-spec .md that recreates the folder tree. |
| 10 | Coordination scale | **Small grid + 1 real-ish net** | RESCO grid4x4 (16 agents, controlled) + cologne3 (3 agents, real topology). Chosen over single-corridor and larger-network. |
| 11 | Primary evaluation metric | **Average travel time** | Chosen over waiting time and throughput. Secondary: waiting time, queue, throughput, time loss. |
| 12 | Root directory name | **`journal-upgrade/`** | Chosen over `traffic-sense-journal` and `atsc-paper-v2`. |
| 13 | Build mechanism | **Shell-script style** (mkdir + quoted heredocs) | Chosen over embedded-contents style. Hardened for Git Bash on Windows + OneDrive path with spaces. |
| 14 | Phase self-containment | **Fully standalone per phase** | Each phase repeats context, exact paths, commands, acceptance gates. |
| 15 | Code grounding | **Fully grounded in the real repo** | Reference real paths/line-level bugs/configs/vendored RESCO. Assumes the repo is cloned. |
| 16 | Memory folder (this request) | **Add `memory/` with full session record** | Capture decisions, options, suggestions; enable exact session reproduction. |

## Derived/recommended values that became part of the plan
- Baselines (mandatory): Fixed-time, Max-Pressure, IDQN, MPLight.
- Seeds: minimum 5 per configuration; report mean ± 95% bootstrap CI.
- Mandatory ablation: coordination transformer vs IDQN-MLP at EQUAL parameter count.
- Reward functions carried forward: dwt, pressure, queue.
- Decision gates: GATE A (end Phase 1 — IDQN must beat Fixed-time) and GATE B (Week 4 — spatial trains or fall back to temporal).
EOF

# ============================================================
# memory/03-options-presented.md
# ============================================================
cat > memory/03-options-presented.md <<'EOF'
# 03 — Options Presented (What Else Was On The Table)

Full menus offered during the session. ✅ = chosen. This preserves the roads not taken and the trade-offs,
so a future session understands WHY the chosen path was chosen and what the alternatives cost.

## Q1 — Architectural scope
- Keep DQN, do it right (Double/Dueling/Rainbow; lower risk, modest novelty).
- Upgrade to modern MARL (IPPO/QMIX/justified transformer; higher effort, stronger novelty).
- ✅ Recommend what fits (decide after repo assessment).

## Q2 — Publication bar
- Solid conference / workshop (clean quant + a few baselines + seeds; weeks).
- ✅ Indexed journal (SCI/Scopus; strong baselines, stats, ablations, multiple scenarios; months).
- Not sure yet.

## Q3 — Simulation setup
- SUMO + a clean RL wrapper (SUMO-RL / PettingZoo / Gymnasium).
- ✅ Whatever's most rigorous (most defensible/reproducible stack, even if it changes the repo).
- Match the existing repo.

## Q4 — RL library philosophy
- ✅ CleanRL — single-file, transparent, citable; easiest to defend/modify; multi-agent is DIY.
- Tianshou — batteries-included PyTorch RL (DQN/Double/Dueling/Rainbow/PPO/SAC); more abstraction.
- RLlib (Ray) — industrial, native multi-agent, distributed; steep learning curve, opaque.

## Q5 — Transformer novelty angle
- ✅ Spatial: neighbour coordination — attend over neighbouring intersections; strongest journal angle;
  makes "long-range dependency" literally true; MOST engineering effort.
- Temporal: history attention — attend over a time-window per intersection; justifies temporal long-range;
  ties to DTQN; medium effort. (This is the chosen FALLBACK.)
- Both spatial + temporal — strongest novelty, highest risk/scope-creep.
- Let me recommend per timeline.

## Q6 — Timeline / compute
- 3–4 months, single GPU.
- 6+ months, decent compute.
- ✅ Tight (<2 months), limited compute.

## Q7 — Risk gate handling
- ✅ Build spatial, temporal as fallback (explicit Week-4 gate; safest path to finishing).
- Spatial or bust (no fallback; higher ceiling, real deadline risk).
- Start temporal, upgrade if time (safest for deadline, slightly lower ceiling).

## Q8 — Computer-vision scope
- ✅ Simulation only — cut CV entirely.
- Sim now, CV as 'future work'.
- Keep CV as a real component (much larger scope; not recommended under <2mo).

## Q9 — Deliverable format
- Markdown plan file in the repo.
- Plan + ADO backlog items (via ado-ops skill).
- ✅ Just show me here first. (Later refined by the author into: one master build-spec .md.)

## Q10 — Coordination scale
- ✅ Small grid + 1 real-ish net — RESCO grid4x4 (16 agents) + cologne3 (3 agents, real topology).
- Single corridor / arterial — cheapest compute, weaker 'network' novelty.
- Larger network (ingolstadt/grid) — strongest scalability claim, heavy compute, risky under <2mo.

## Q11 — Primary metric
- ✅ Average travel time — RESCO/literature standard; hardest to game; best comparability.
- Average waiting time — continuity with original paper; easier to game.
- Throughput — network efficiency framing; sensitive to demand setup.

## Q12 — Root directory name
- traffic-sense-journal.
- ✅ journal-upgrade.
- atsc-paper-v2.

## Q13 — Build mechanism
- Embedded contents + instructions (agent-agnostic, readable).
- ✅ Shell-script style (mkdir + heredoc cat; hardened for Windows Git Bash + spaces-in-path).

## Q14 — Phase self-containment
- ✅ Fully standalone per phase.
- Lean phases + shared context file.

## Q15 — Code grounding
- ✅ Fully grounded in the real repo (paths/line-level bugs/configs).
- Mostly grounded, some generic.
- Higher-level / portable.

## Q16 — CV scope re-confirmation (asked alongside risk gate)
- ✅ Simulation only — cut CV entirely (re-affirmed).
EOF

# ============================================================
# memory/04-diagnosis-and-suggestions.md
# ============================================================
cat > memory/04-diagnosis-and-suggestions.md <<'EOF'
# 04 — Diagnosis & Suggestions (Why The Plan Looks Like This)

## A. Repo bugs that invalidated the original results (grounded, code-level)
| # | Bug | Location | Effect |
|---|-----|----------|--------|
| 1 | Only ONE gradient update per episode — learn() called only at the terminal step, then break | `experiments/trf_multi_agent/dqn/main.py` (training while-loop terminal branch) | ~1 update/episode → agent barely learns. Likely explains everything. |
| 2 | `self.eval()` called INSIDE `forward()` | `dqn/dqn.py` (DeepQNetwork.forward) | BatchNorm/Dropout never in train mode; running stats corrupt. |
| 3 | No target network; bootstraps off the same net | `dqn/dqn.py` (DQN.learn, next_q from self.model) | unstable/divergent Q-values. Paper's loss eq CLAIMED theta-prime but code lacked it. |
| 4 | `self.num_states` hardcoded to 80, contradicts config `num_states=8` | `dqn/dqn.py` (DeepQNetwork.__init__) vs `dqn/config.ini` | wrong tensor shapes. |
| 5 | Evaluation reuses `train()` → keeps learning + exploring during "test" | `dqn/main.py` (second train() block) | contaminated/invalid results. |
| 6 | `num_heads = 1` | `dqn/config.ini` | "multi-head" attention is single-head. |
| 7 | Model saved to disk every gradient step | `dqn/dqn.py` (end of DQN.learn) | cripples throughput (explains "12 hours"). |
| 8 | Dead TF SAC + commented model graveyard | `sac/sac.py`, `dqn/models.py` | reproducibility hazard; remove. |

## B. Paper-level flaws (from reading the PDF)
- No quantification anywhere — entire results section is "blue line below red line"; zero numbers, no
  variance, no significance, no seeds.
- Core novelty thin: self-attention over an 8-element non-sequential vector has no real long-range
  dependency to capture; the phrase is repeated ~6x but never demonstrated.
- Duplicate/placeholder figure captions: Figs 3,4,7,8,9 share the identical caption.
- Two incorrect equations: Bellman optimality (Eq.1) missing the summation over (s',r); DQN loss (Eq.3)
  missing the expectation/mean.
- Action/state mismatch: state vector length 8 (sub-lanes) vs action array length 4, never reconciled;
  "max-pooling" misused for "argmax".
- Tooling contradiction: abstract says OpenCV for tracking, body says dlib.
- Tiny network (width 4 in the paper / 40 in the repo config), big claims; no baselines beyond vanilla DQN.

## C. The novelty fix (the heart of the plan)
The transformer must EARN its place by making the state structured. Chosen: SPATIAL neighbour coordination
— each agent attends over neighbouring intersections' states. This makes "long-range dependency" literally
true (spatial, across the road graph), yields an interpretability story (attention heatmaps over neighbours),
and connects to current MARL+transformer literature. Fallback: TEMPORAL history-attention (attend over the
last K timesteps per intersection), lower risk, still publishable, ties to DTQN.

## D. Package/method recommendations (the "what should I use" answer)
| Concern | Was | Recommended → |
|---------|-----|---------------|
| RL algorithms | hand-rolled buggy DQN + dead TF-SAC | CleanRL single-file (chosen); Tianshou/RLlib were alternatives |
| Multi-agent | independent hand-rolled agents | keep PettingZoo; IDQN now, IPPO/QMIX optional |
| Benchmark scenarios | 2x2 grid only | RESCO suite (already vendored): grid4x4 + cologne3 |
| Config | .ini + hardcoded paths | Hydra + structured configs |
| Tracking | loose CSVs | Weights & Biases (or TensorBoard) |
| Sim speed | libsumo (good) | keep libsumo; parallelize seeds |
| Stats/plots | matplotlib one-offs | rliable (IQM, performance profiles, bootstrap CIs) |
| Baselines | vanilla DQN only | Fixed-time, Max-Pressure, IDQN, MPLight |

## E. Why these specific choices under <2mo
CleanRL avoids fighting a framework on limited compute. grid4x4 isolates the coordination effect cleanly;
cologne3 adds real-topology credibility without a heavy network. Average travel time = direct comparability
with prior work. The single equal-parameter ablation is the cheapest experiment that actually proves the thesis.
EOF

# ============================================================
# memory/05-open-tensions-and-risks.md
# ============================================================
cat > memory/05-open-tensions-and-risks.md <<'EOF'
# 05 — Open Tensions & Risks (Watch Out For)

## TENSION 1 — Ambitious novelty vs. tight constraint (flagged honestly in-session)
The author chose the MOST ambitious novelty (spatial neighbour-coordination — the "most engineering effort"
option) together with the TIGHTEST constraint (<2 months, single GPU). These pull in opposite directions.
Resolution baked into the plan:
- One scenario family only (grid4x4 + cologne3), not the full RESCO sweep.
- Minimum viable but non-negotiable baselines (Fixed-time, Max-Pressure, IDQN, MPLight).
- A single decisive ablation (equal-parameter transformer vs MLP), skipping head-count sweeps unless time remains.
- A HARD Week-4 fallback to the temporal transformer if spatial isn't training cleanly (lose ~3 days, not the deadline).

## RISK 1 (HIGHEST, non-technical) — Prior-publication status
The repo's `images/` folder contains an email "[ACCEPTED] Notification from ADICS - 2024". The original
paper may already be PUBLISHED. If so, a close variant risks dual-submission/self-plagiarism rejection.
MUST be resolved in Week 0 (see `backlog/01-publication-status-check.md`). If published, this becomes a
"substantial extension" submission (cite the prior paper; the new RL core + spatial novelty + RESCO +
baselines + stats easily provide the required increment). This changes the paper's framing, so it gates
the Phase-5 write-up.

## RISK 2 (technical) — The RL core might still underperform after the rebuild
Mitigated by GATE A: if rebuilt IDQN doesn't beat Fixed-time on grid4x4, STOP and debug before building the
transformer. Nothing downstream can succeed on an unproven core.

## RISK 3 (technical) — Spatial transformer instability
Multi-agent + attention + Double-DQN can be finicky on limited compute. Mitigated by GATE B and the temporal
fallback. Standing rule: do NOT let transformer tuning bleed into Weeks 5–6 (reserved for baselines/ablations).

## RISK 4 (scope) — Compute budget overrun
The method×scenario×reward×seed matrix may not fit a single GPU in time. Standing cut-order (never silently
truncate; log every cut in the paper): cut SECONDARY ablations first → drop cologne3 to a single reward →
only then touch the (mandatory) 5 seeds or the (mandatory) baseline set.

## RISK 5 (reproducibility/integrity) — Reused FYP prose
Run a similarity scan (Turnitin/iThenticate) before submission; rewrite results around real numbers; fix the
two incorrect equations and the duplicate figure captions; remove/reframe all CV claims.

## Things deliberately deferred (not forgotten)
- ADO work items: offered, not chosen yet. Author may later want the backlog mirrored as ADO epics/features/
  stories via the ado-ops skill.
- Computer-vision pipeline: cut for this paper; candidate for a future deployment-focused paper.
- Secondary ablations (head count, neighbour radius, coordination variant a vs b): nice-to-have if time remains.
EOF

# ============================================================
# Verification
# ============================================================
echo ""
echo "=== Created tree under $(pwd) ==="
find . -type f | sort
echo ""
echo "=== File count (expected 23: 1 README + 8 plans + 8 backlog + 6 memory) ==="
find . -type f | wc -l
echo ""
echo "Done. New session? Read memory/00-session-reproduction-guide.md."
echo "Executing? Open plans/00-overview.md and begin Phase 0."
```

---

## POST-BUILD CHECKLIST (for the agent)

After running the script:

1. Verify the tree matches **EXPECTED TREE** (23 files: 1 README + 8 plans + 8 backlog + 6 memory).
2. **If resuming/reproducing a session:** read `memory/00-session-reproduction-guide.md` first (it routes you through context → decisions → options → diagnosis → tensions). **If executing fresh:** skip to step 3.
3. Open `plans/00-overview.md`, then `plans/01-phase-0-foundation.md`, and **begin executing Phase 0**.
4. Honor **GATE A** (end of Phase 1) and **GATE B** (Week 4) in `plans/99-decision-gates.md` as hard go/no-go checkpoints.
5. Resolve `backlog/01-publication-status-check.md` in Week 0 — it can change the entire framing.
6. Do not ask clarifying questions; all decisions are recorded in `memory/02-decisions-log.md`. If a genuine blocker arises, note it against the relevant phase/backlog file and continue with the lowest-risk path described in that file.

---
