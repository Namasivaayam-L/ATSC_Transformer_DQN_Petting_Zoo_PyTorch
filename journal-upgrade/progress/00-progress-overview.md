# 00 — Progress Overview (Start Here)

## Purpose
This folder tracks execution progress across agent sessions. **New agents: read this file FIRST,**
then check `01-current-status.md` for the exact state of work, and `02-session-log.md` for
the chronological history of what was done and by whom (which session).

## How to use this folder

### If you are a NEW agent session:
1. Read `00-progress-overview.md` (this file) — understand the tracking system.
2. Read `01-current-status.md` — know the CURRENT state of each phase/task.
3. Read `02-session-log.md` — see what was done in prior sessions and any blockers.
4. Read `03-blockers-and-decisions.md` — check for open blockers needing user input.
5. Read `04-environment-state.md` — know what's installed, what's running, paths, etc.
6. Consult `../memory/` for the full planning context if needed.
7. Consult `../plans/` for the detailed phase specs.
8. **Update this folder** before ending your session.

### Before ending YOUR session:
1. Update `01-current-status.md` with what you completed and what's next.
2. Append to `02-session-log.md` with a dated entry of your work.
3. Update `03-blockers-and-decisions.md` if any new blockers or decisions arose.
4. Update `04-environment-state.md` if you installed/changed anything.

## Quick reference — The plan at a glance
| Phase | Name | Weeks | Gate | Status |
|-------|------|-------|------|--------|
| 0 | Foundation & Rigor Scaffolding | 0–1 | Fixed-time e2e on grid4x4, 5 seeds, W&B | 🟢 **PASSED** — fixed-time runs end-to-end, all metrics logged |
| 1 | RL Core (CleanRL IDQN) | 1–2 | **GATE A**: IDQN beats Fixed-time | 🟢 **IMPLEMENTED & VERIFIED** — `agents/idqn.py` + `train.py` working, loss decreasing. Full 5-seed run needed for formal GATE A. |
| 2 | Spatial Coordination Transformer | 3–4 | **GATE B**: trf_coord ≥ IDQN | 🟢 **IMPLEMENTED & VERIFIED** — `agents/trf_coord.py` + `env/road_graph.py` working, loss decreasing over 2 episodes. Full 5-seed run needed for formal GATE B. |
| 3 | Baselines & Ablations | 5–6 | Full results table + equal-param ablation | 🔴 Not Started |
| 4 | Statistical Validation & Figures | 7 | All figures from `make_figures.py` | 🔴 Not Started |
| 5 | Write-up Integration | 8 | Manuscript with real numbers | 🔴 Not Started |

## Key backlog items (parallel track)
| # | Item | Priority | Status |
|---|------|----------|--------|
| 1 | Publication status check (ADICS-2024) | 🔴 CRITICAL | ✅ **RESOLVED** — published as conference paper, frame as substantial extension |
| 2 | Target journal shortlist | High | 🔴 Not Started |
| 3 | Authorship / advisor | Medium | 🔴 Not Started |
| 5 | Compute budget estimation | High | 🔴 Not Started |

## Project start date
- Planning session: 2026-06-13
- Execution start: 2026-06-13 (initial code work done same day)
- Phase 0 + 1 + 2 coded: 2026-06-14
