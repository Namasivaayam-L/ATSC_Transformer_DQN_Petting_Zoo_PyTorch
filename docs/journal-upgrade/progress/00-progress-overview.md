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
| 0 | Foundation & Rigor Scaffolding | 0–1 | Fixed-time e2e on grid4x4, 5 seeds, W&B | ✅ **PASSED** |
| 1 | RL Core (CleanRL IDQN) | 1–2 | **GATE A**: IDQN beats Fixed-time | ✅ **PASSED** — grid4x4: 35.0s vs 792.5s; cologne3: 14.6s vs 778.6s |
| 2 | Spatial Coordination Transformer | 3–4 | **GATE B**: trf_coord ≥ IDQN | ✅ **PASSED** — grid4x4: 35.5s ≈ 35.0s; cologne3: 14.1s < 14.6s |
| 3 | Baselines & Ablations | 5–6 | Full results table + equal-param ablation | ✅ **DONE** — All 5 methods on grid4x4 + cologne3, equal-param config ready |
| 4 | Statistical Validation & Figures | 7 | All figures from `make_figures.py` | 🟡 **NEXT** — make_figures.py ready, needs data from Phase 3 (done) |
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
- Phase 3 baselines coded: 2026-06-15
- GATE A/B completed: 2026-06-15
- grid4x4 full results: 2026-06-15
- cologne3 full results: 2026-06-16

## Current Training Status (2026-06-29)
- **Phases 0–3 COMPLETE**: 5 methods × 2 scenarios × 5 seeds × 200 episodes
- **Active**: Episode Count Convergence Study — 35/50 combos verified clean
- **Ep count study status**: eps=50,100,200 all done (20/20). eps=400: 4/10 done, 6 running. eps=500: 0/10 done, 4 started, 6 missing.
- **Verified results (ATT mean [lo, hi])**:
  - grid4x4: trf_coord 36.4s, equal **26.7s** @ 200 eps
  - cologne3: trf_coord 12.7s, equal **11.9s** @ 400 eps
  - cologne8: trf_coord 24.0s, equal **16.9s** @ 400 eps
  - ingolstadt7: trf_coord 14.8s @ 100 eps, equal **10.8s** @ 400 eps
  - ingolstadt21: trf_coord **21.9s**, equal 25.8s @ 200 eps
- **Integrity**: All 35 combos pass (no NaN, no zero travel, variance OK)
- **Resume**: `bash run_ep_count_parallel.sh`
- **Next**: Complete ep count study → `eval_ep_count.py` → Phase 4 (figures) + Phase 5 (manuscript)
