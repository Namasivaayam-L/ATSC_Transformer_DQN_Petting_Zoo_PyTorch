# 03 — Blockers & Decisions (Open Items)

Items here need USER input or are runtime decisions that deviate from the original plan.
Planning-phase decisions are in `../memory/02-decisions-log.md` — do NOT duplicate them here.

---

## Open Blockers

### (none currently open)

---

## Resolved Blockers

### BLOCKER-001 — Prior Publication Status (CRITICAL)
- **Raised:** Session #1 (2026-06-13)
- **Source:** `../backlog/01-publication-status-check.md`
- **Question:** Was the original paper ("Traffic Sense") actually PUBLISHED in ADICS-2024 proceedings,
  or was it only accepted/presented but not formally published in indexed proceedings?
- **Status:** ✅ RESOLVED (Session #2, 2026-06-14)
- **Resolution:** **PUBLISHED.** The ADICS-2024 conference paper is a DQN with various reward functions (no transformer layer). This journal submission **must be framed as a substantial extension** and cite the prior paper. Key framing:
  - The conference paper established the baseline: independent DQN agents with DWT/pressure/queue rewards.
  - This work extends it with a spatial coordination transformer and rigorous experimental methodology (proper target network, frozen evaluation, multi-seed statistical validation).
  - Must run similarity check (backlog #4) and cite the prior work explicitly in the introduction.

---

## Runtime Decisions (execution-time deviations from plan)

### Decision #1 — Simulation episode length for training vs evaluation (2026-06-14)
- **Context:** Full 3600-step episodes take ~60s wall time each on CPU. Running GATE A (200 eps × 5 seeds) would take ~16.7 hours total.
- **Decision:** Use `num_seconds=800` for training (faster iteration, ~3.5h total), then re-evaluate trained policies with `num_seconds=3600` for paper-quality metrics.
- **Impact:** Training is faster at the cost of not seeing full traffic patterns. Evaluation at 3600 ensures comparability with prior work.

### Decision #2 — One independent IDQN agent per traffic signal (2026-06-14)
- **Context:** Original plan suggested parameter sharing. Multi-agent PettingZoo env with 16 agents.
- **Decision:** Each intersection gets its own independent IDQN agent (separate weights, separate replay buffer). This is the standard IDQN formulation and mirrors the original ADICS-2024 approach.
- **Impact:** 16× more parameters, 16× more gradient computations per step, but simpler and more comparable to baselines.

### Decision #3 — Adjacency from connection elements not road topology (2026-06-14)
- **Context:** RESCO grid4x4 net has well-defined `<connection>` elements between junctions.
- **Decision:** Parse `<connection via="...">` to build edge-to-junction mapping, then connect agents whose shared edge has via connections. Verified: grid4x4 corners=2 neighbours, edges=3, centres=4.
- **Impact:** More accurate than road-distance-based adjacency. Works across any SUMO net.

### Decision #4 — Manual transformer forward pass simplified (2026-06-14)
- **Context:** Original implementation manually iterated encoder layers to extract attention weights, but `norm_first=True` sub-layer order was wrong, causing shape mismatch.
- **Decision:** Use standard `nn.TransformerEncoder.forward()` in the main forward path. Add separate `forward_with_attention()` for interpretability that correctly handles `norm_first` sub-layer order.
- **Impact:** Cleaner, correct forward pass. Attention weights still available for analysis.
