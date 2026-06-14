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
