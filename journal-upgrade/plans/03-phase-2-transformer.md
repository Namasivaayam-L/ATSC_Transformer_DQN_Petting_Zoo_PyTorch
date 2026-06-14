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
