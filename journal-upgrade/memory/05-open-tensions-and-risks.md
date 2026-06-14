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
