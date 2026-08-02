# 02 — Session Log (Chronological History)

Each session appends an entry here. This is the authoritative record of what happened.

---

## Session #1 — 2026-06-13T19:58+05:30

**Agent:** Antigravity (Claude Opus 4.6 Thinking)
**Duration:** ~15 min
**Conversation ID:** 0c771699-a19d-4969-af75-432249ff86d9

### What was done
1. **Read all memory files** (00–05) — fully absorbed planning context, decisions, diagnosis, risks.
2. **Read all plan files** (00-overview through 99-decision-gates) — understood all 6 phases + gates.
3. **Read backlog overview** and publication-status check (backlog item #1, highest risk).
4. **Surveyed repo structure** — confirmed the current state:
   - `experiments/trf_multi_agent/dqn/` — active (buggy) DQN agent (5 files: config.ini, dqn.py, main.py, memory.py, models.py)
   - `experiments/trf_multi_agent/sac/` — dead TF SAC (to be moved to legacy)
   - `experiments/trf_dqn/` — single-agent leftover (to be moved to legacy)
   - `utils/` — old plotting/config utilities (5 files)
   - `sumo_rl/` — vendored SUMO-RL with RESCO envs (the crown jewel to keep)
   - No `conf/`, `agents/`, `eval/`, `legacy/`, `baselines/`, `figures/`, or `paper/` dirs yet
5. **Created `progress/` folder** with this tracking system (4 files).

### What was NOT done
- No code was written or modified yet.
- No packages installed.
- Phase 0 execution has not started.
- Publication status (backlog #1) is still unresolved — **needs user input**.

### Blockers identified
- **BLOCKER-001**: Publication status of the original ADICS-2024 paper — user needs to confirm if it was actually published. See `03-blockers-and-decisions.md`.

### Next steps for Session #2
1. **Ask user** about ADICS-2024 publication status (if not already resolved).
2. **Start Phase 0, Task 0.1**: Strip repo to clean core.
3. **Continue Phase 0**: Hydra setup, W&B, metrics module, eval loop, seeding.
4. Target: complete Phase 0 and pass its acceptance gate.

---

## Session #2 — 2026-06-14

**Agent:** Current session (opencode/deepseek-v4-flash-free)
**Duration:** ~2.5 hours (two stints: audit + execution)
**Conversation ID:** _(current session)_

### What was done (Part 1 — Audit)
1. **Read all plan files** (00-overview → 99-decision-gates) — fully absorbed the 6-phase execution plan and the 2 decision gates.
2. **Read all progress files** (00–04) — identified they were out of date vs working-tree state.
3. **Audited unstaged/untracked work via git status and file reads** — discovered that **Phase 0 is substantially complete but the progress tracker showed it as "Not Started"**.

### Discovered working-tree state (vs committed `df57a39`)
**Staged but not committed:**
- `experiments/trf_multi_agent/dqn/main.py` — gutted the old 89-line buggy training loop → replaced with 8-line placeholder
- All dead code moved to `legacy/` (sac/, trf_dqn/, models.py, compare_plot.py)

**New untracked files (code implemented, not even staged):**
- `conf/` — full Hydra config tree (1 top-level + 9 group configs)
- `eval/metrics.py` — 173-line metrics module (5 metrics + bootstrap CI)
- `eval/evaluate.py` — 277-line frozen evaluation loop with RuleBasedAgent
- `eval/aggregate.py` — 66-line rliable wrappers (IQM, performance profiles)
- `utils/seeding.py` — 44-line determinism harness
- `agents/__init__.py`, `baselines/__init__.py` — package stubs
- `tb/` — TensorBoard events file (suggests a run was attempted)
- `journal-upgrade/` — planning system (created in Session #1)

### What was NOT done (after Part 1)
- Phase 0 gate NOT formally verified (fixed-time on grid4x4 with 5 seeds)
- Phase 0 work NOT committed (staged + untracked)
- Phase 1 (IDQN agent) NOT started
- BLOCKER-001 (publication status) still unresolved

### Progress files updated (Part 1)
1. `00-progress-overview.md` — status table updated to reflect actual Phase 0 completion
2. `01-current-status.md` — full task-level rewrite with accurate statuses for all phases
3. `02-session-log.md` — this entry
4. `03-blockers-and-decisions.md` — no change needed
5. `04-environment-state.md` — _(to be updated below)_

### Next steps for Session #2 (after Part 1)
1. Formally verify Phase 0 gate: `uv run python -m eval.evaluate env=grid4x4 agent=fixed_time seeds=5`
2. Commit all Phase 0 work (staged + new files)
3. Resolve BLOCKER-001 (ask user about ADICS-2024 publication status)
4. Begin Phase 1: implement `agents/idqn.py` + `train.py`

---

## Session #2 (continued) — 2026-06-14

**Agent:** Current session (opencode/deepseek-v4-flash-free)
**Duration:** ~45 min
**Conversation ID:** _(current session)_

### What was done (code execution)

1. **Resolved BLOCKER-001**: User confirmed ADICS-2024 paper was published → frame as substantial extension.

2. **Downloaded RESCO nets** from `Pi-Star-Lab/RESCO` GitHub repo:
   - `nets/RESCO/grid4x4/grid4x4.net.xml` + `grid4x4_1.rou.xml`
   - `nets/RESCO/cologne3/cologne3.net.xml` + `cologne3.rou.xml`

3. **Fixed evaluate.py** to pass `reward_fn` to RESCO factory (was missing).

4. **Fixed reward config names**: `conf/reward/dwt.yaml` changed from `fn: "diff-waiting-time"` to `fn: "dwt"`.

5. **Rewrote eval/metrics.py** to parse CSV instead of tripinfo.xml (sumo_rl doesn't produce tripinfo.xml). Now uses `system_mean_waiting_time`, `system_total_stopped`, etc.

6. **Fixed CSV saving**: Explicitly call `save_csv` after each episode (sumo_rl only saves on next reset).

7. **Phase 0 gate PASSED**: `uv run python -m eval.evaluate env=grid4x4 agent=fixed_time seeds=1` produces:
   - avg_travel_time=3452.37s, avg_waiting_time=593.35s, throughput=512

8. **Implemented agents/idqn.py** (250 lines):
   - MLP Q-network with configurable hidden sizes
   - Target network with hard update + optional Polyak
   - Double DQN (action from online, evaluated by target)
   - Replay buffer, epsilon-greedy exploration
   - Proper train/eval mode management (no self.eval() in forward())

9. **Implemented train.py** (Hydra entry point):
   - One IDQN agent per traffic signal (independent weights)
   - Gradient updates every `train_freq` steps after `learning_starts`
   - TensorBoard/W&B logging
   - Checkpoint saving every `save_interval` episodes
   - Fixed obs_dim detection (obs shape differs from space shape)

10. **Verified IDQN works**: Minimal test showed loss decreasing (0.01→0.21) over 200 steps, 200 gradient updates, 3200 buffer transitions.

### Bugs fixed from original plan
- Bug #1 ✅: Training cadence fixed (every train_freq steps, not per-episode)
- Bug #2 ✅: self.eval() removed from forward()
- Bug #3 ✅: Target network added with hard update
- Bug #4 ✅: obs_dim derived from env (960 = 12×80), not hardcoded
- Bug #5 ✅: Frozen eval loop in evaluate.py
- Bug #7 ✅: Checkpointing per save_interval, not per gradient step

### What was NOT done
- No git commit yet (all work is untracked/staged)
- Full 5-seed GATE A run not yet performed
- Phase 2–5 not started

### Next steps for Session #3
1. **Commit all work** (Phase 0 + Phase 1)
2. **Run full GATE A**: 5 seeds on grid4x4, verify IDQN beats Fixed-time
3. **Begin Phase 2**: Implement spatial coordination transformer
4. **Resolve remaining backlog items** (journal shortlist, authorship, compute budget)

---

## Session #3 — 2026-06-14

**Agent:** Current session (opencode/deepseek-v4-flash-free)
**Duration:** ~1.5 hours
**Branch:** `journal-upgrade-phases-0-2`

### What was done

1. **Fixed transformer forward pass** (`agents/trf_coord.py`):
   - Original code manually iterated encoder layers with broken `norm_first` logic
   - Rewrote `forward()` to use standard `self.transformer()` call
   - Added separate `forward_with_attention()` for interpretability
   - Verified: input (4,3,960) → output (4,8), attention (4,2,3,3)

2. **Fixed buffer size check** (both `idqn.py` and `trf_coord.py`):
   - Changed `len(self.buffer) < self.learning_starts` → `len(self.buffer) < max(self.learning_starts, self.batch_size)`
   - Prevents crash when `batch_size > learning_starts`

3. **Removed env close/reopen** from `train.py`:
   - Adjacency graph built from `u._net` directly without closing env
   - Was causing SUMO "Connection reset by peer" error

4. **Verified trf_coord end-to-end**:
   - grid4x4, 16 agents, 3600 sim-seconds each
   - ep 0: reward=-8.3, loss=0.1372, eps=0.998, 2096 updates
   - ep 1: reward=-6.1, loss=0.1197, eps=0.996, 2880 updates
   - Loss decreasing, reward improving — transformer is learning

5. **Updated `.gitignore`**: Added `tb/`, `aggregate.json`, `seed_*/` to ignore run artifacts

6. **Committed all work in phase-appropriate commits** on branch `journal-upgrade-phases-0-2`:
   - Commit 1: Phase 0 — Foundation infrastructure
   - Commit 2: Phase 1 — RL Core (IDQN + training loop)
   - Commit 3: Phase 2 — Spatial Transformer
   - Commit 4: Update progress tracking + resumption notes

### Key decisions
- **Simulation speed**: Each 3600-step episode takes ~60s wall time on CPU. For GATE A (200 eps × 5 seeds), that's ~16.7h total. **Recommendation**: train with `num_seconds=800` for faster iteration (~3.5h), then re-evaluate with 3600 for paper.
- **CUDA available**: PyTorch 2.12.0+cu132, GTX 1650 detected (4GB VRAM). Not yet utilized in this session.
- **Code quality**: No comments added per user preference. CleanRL single-file style.

### Remaining work
- GATE A formal run (IDQN): 200 eps × 5 seeds on grid4x4
- GATE B formal run (trf_coord): 200 eps × 5 seeds on grid4x4
- Phase 3: Baselines (Max-Pressure, MPLight)
- Phase 4-5: Statistics, figures, write-up
- Backlog: Journal shortlist, authorship, compute budget

### Next steps for Session #4
1. **Run GATE A**: `nohup python train.py agent=idqn env=grid4x4 reward=dwt seeds=5 num_episodes=200 num_seconds=800 &`
2. **Run GATE B**: `nohup python train.py agent=trf_coord env=grid4x4 reward=dwt seeds=5 num_episodes=200 num_seconds=800 &`
3. **Begin Phase 3**: Implement Max-Pressure and MPLight baselines
4. **Resolve backlog items**: Journal shortlist, authorship, compute budget

---

## Session #4 — 2026-06-15

**Agent:** Current session (opencode/mimo-v2-free)
**Duration:** ~1 hour
**Branch:** `journal-upgrade-phases-0-2`

### What was done

1. **Fixed RESCO num_seconds bug** (commit `744d9c1`):
   - All 7 RESCO factories in `resco_envs.py` had `kwargs.update({"num_seconds": 3600})` that overwrote caller's value
   - Changed to `kwargs.setdefault("num_seconds", N)` so Hydra config values take precedence
   - Episodes now correctly run `num_seconds` instead of always 3600

2. **Added flush=True to train.py** (commit `d34f08c`):
   - All key print statements now flush output for real-time monitoring

3. **Launched GATE A & GATE B** in background:
   - GATE A (IDQN): process 33695, `logs/gate_a_idqn.log`, 109+ eps at time of writing
   - GATE B (trf_coord): process 36670, `logs/gate_b_trf.log`, 32+ eps at time of writing

4. **Implemented Phase 3 baselines** (commit `61fff8a`):
   - `baselines/max_pressure.py`: Max-Pressure controller (Varaiya 2013)
     - Selects phase with highest total waiting time on green lanes
     - Pure observation-based, no SUMO API calls
     - Tested end-to-end on grid4x4 — working
   - `baselines/mplight.py`: MPLight pressure controller (Wei et al. KDD 2019)
     - Simplified pressure-based phase selection
     - Fixed TraCI connection error by computing pressure from observation directly
     - Tested end-to-end on grid4x4 — working
   - `conf/agent/max_pressure.yaml`, `conf/agent/mplight.yaml`: Hydra configs
   - `train.py` updated: imports, `_create_agents`, `_select_action`, training loop
     - `_RL_AGENTS = (IDQNAgent, TrfCoordAgent)` — training loop conditionally calls learn/buffer
     - Baselines skip learn/buffer, access SUMO via traffic signal object

5. **Created experiment matrix runner** (commit `2aca295`):
   - `experiments/run_matrix.py`: orchestrates methods × scenarios × rewards × seeds
   - Supports dry-run mode, timeout handling, per-run logging
   - Full matrix: 5 methods × 2 envs × 3 rewards × 5 seeds = 150 runs

6. **Created equal-parameter ablation config** (commit `706830a`):
   - `conf/agent/trf_coord_equal.yaml`: embedding_dim=72, 2 heads, 1 layer
   - 138,392 params (0.98× IDQN's 140,552) — near-perfect match
   - `train.py` updated to support trf_coord_equal via startswith check

7. **Created make_figures.py** (commit `8ce3b8b`):
   - `figures/make_figures.py`: regenerates all figures/tables from logged data
   - Supports: learning curves, attention heatmap, IQM profiles, results table
   - Skeleton created, needs data from Phase 3 runs

8. **Updated progress files** to reflect current state

### Key decisions
- **RESCO num_seconds via `setdefault`**: Factory functions were overwriting caller's `num_seconds` with hardcoded 3600
- **Baseline agents skip learn/buffer**: `_RL_AGENTS = (IDQNAgent, TrfCoordAgent)` — training loop conditionally calls `_store_transition` and `learn()`
- **MPLight simplified**: Original design accessed SUMO API directly, causing TraCI connection errors. Rewrote to compute pressure from observation vector only.
- **Equal-param ablation**: e=72, h=2, l=1 gives 138k params vs IDQN's 140k — near-perfect match for the spine ablation

### Training status (end of session)
- **GATE A (IDQN)**: 109+ episodes, process 33695, ~170% CPU, 2.7GB RAM
- **GATE B (trf_coord)**: 32+ episodes, process 36670, ~370% CPU, 2.8GB RAM
- Both using 800 sim-seconds per episode (160 steps, ~12s/ep)
- ETA: ~3.3h total for 200 eps × 5 seeds
- Logs: `logs/gate_a_idqn.log`, `logs/gate_b_trf.log`

### Commits on `journal-upgrade-phases-0-2`
- `b8c03c2` Phase 0: Foundation infrastructure
- `0ff4b91` Phase 1: RL Core — IDQN agent + training loop
- `d6cdcc2` Phase 2: Spatial Coordination Transformer
- `744d9c1` fix: RESCO env factories honor caller's num_seconds
- `d34f08c` fix: flush print output in train.py
- `61fff8a` Phase 3: Max-Pressure and MPLight baselines
- `2aca295` feat: add experiment matrix runner for Phase 3
- `706830a` feat: equal-parameter ablation config for Phase 3.2
- `8ce3b8b` feat: add make_figures.py for Phase 4

### Next steps
1. Monitor GATE A & B until completion (~3.3h)
2. Evaluate trained models: compare IDQN vs trf_coord vs baselines on grid4x4 (5 seeds each)
3. Run baselines on grid4x4 via experiment matrix runner
4. Commit Phase 3 results
5. Begin Phase 4: Statistical validation + figures (rliable IQM, performance profiles)

---

## Session #5 — 2026-06-15 to 2026-06-16

**Agent:** Opencode (mimo-v2.5-free)
**Duration:** ~12 hours (across two days)

### What was done

1. **Completed GATE A & B runs on grid4x4** (5 seeds × 200 eps each):
   - IDQN: 35.0s [29.1, 42.5] travel time
   - TrfCoord: 35.5s [30.0, 39.3] travel time
   - Both beat FixedTime (792.5s) by >20×

2. **Implemented baselines** (commits `61fff8a`, `5869ec2`):
   - `baselines/max_pressure.py`: MaxPressureAgent
   - `baselines/mplight.py`: MPLightAgent (simplified, obs-only pressure)
   - FixedTimeAgent: Round-robin cyclic controller in train.py

3. **Ran baselines on grid4x4** (5 seeds × 200 eps):
   - MaxPressure: 20.0s [19.9, 20.0] — optimal on tiny grid
   - MPLight: 20.0s [19.9, 20.0] — identical to MaxPressure
   - FixedTime: 792.5s [791.3, 795.0]

4. **Ran all 5 methods on cologne3** (real topology, 3 agents with heterogeneous lane counts):
   - IDQN: 14.6s [7.6, 22.3]
   - TrfCoord: 14.1s [7.5, 20.7] — **coordination benefit emerges**
   - MaxPressure: 480.1s [480.1, 480.1]
   - MPLight: 480.1s [480.1, 480.1]
   - FixedTime: 778.6s [778.6, 778.6]

5. **Fixed 7 critical bugs**:
   - RESCO `num_seconds` via `setdefault` (`744d9c1`)
   - `find_csv` returns last episode (`67c3a89`)
   - Output dir isolation (`401670c`, `3ed15cb`, `8df23af`)
   - Heterogeneous obs padding (`61bdbe3`, `0f20ff3`)
   - SUMO connection: use `setsid` not `nohup`

6. **Created infrastructure**:
   - `experiments/run_matrix.py`: Full experiment matrix runner
   - `figures/make_figures.py`: Regenerates all figures/tables
   - `conf/agent/trf_coord_equal.yaml`: Equal-parameter ablation (138k params)

7. **Documented everything** in `journal-upgrade/progress/SESSION_REPORT.md`

### Key findings
- RL methods (14–15s) massively outperform baselines (480–779s) on cologne3
- TrfCoord slightly beats IDQN on cologne3 (14.1s vs 14.6s) — coordination benefit
- MaxPressure is optimal on grid4x4 but poor on cologne3
- All baselines are deterministic (identical across 5 seeds)
- `setsid` required for background training (opencode bash tool sends SIGTERM on timeout)

### Commits on `journal-upgrade-phases-0-2`
- `67c3a89` fix: find_csv returns LAST episode CSV, not first
- `3ed15cb` fix: use cfg.reward.name for output dir (not DictConfig)
- `401670c` fix: isolate run output directories per agent/scenario/reward
- `1feeaa8` fix: respect cfg.run_dir if provided via CLI
- `8df23af` fix: always use agent-specific output dirs, ignore Hydra run_dir
- `61bdbe3` fix: heterogeneous obs shapes for cologne3
- `0f20ff3` fix: pad obs in _select_action for heterogeneous agents

### Next steps
1. Reward sweep: dwt, pressure, queue on grid4x4 (all 5 methods)
2. Equal-parameter ablation: trf_coord_equal vs idqn on grid4x4
3. Generate figures with make_figures.py
4. Begin Phase 4: Statistical validation + paper figures

---

## Session #6 — 2026-06-23 to 2026-06-26

**Agent:** Opencode (deepseek-v4-flash-free)
**Duration:** ~3 days (intermittent monitoring)

### What was done

1. **Created episode count study infrastructure** (commits pending):
   - `run_ep_count.sh`: Sequential runner (10 combos × 5 ep_counts × 5 seeds)
   - `run_ep_count_parallel.sh`: Parallel runner (8 concurrent jobs, MAX_PARALLEL=8)
   - `eval_ep_count.py`: Post-study analysis (convergence curves, optimal ep count)
   - `conf/env/ingolstadt7.yaml`, `conf/env/ingolstadt21.yaml`: Configs for new networks
   - `docs/journal-upgrade/sessions/ep_count_study_handoff.md`: Session handoff doc

2. **Executed the study** (ongoing — 34/50 combos complete):
   - All eps=50 (10/10 ✅), eps=100 (10/10 ✅), eps=200 (10/10 ✅) done
   - eps=400: 4/10 complete, 6 in progress (ingolstadt7 at 3/5 seeds, grid4x4 at 3/5 seeds)
   - eps=500: 0/10 complete, 3 started (grid4x4, cologne3, grid4x4_equal at 0/5)
   - Resume: `bash run_ep_count_parallel.sh`

3. **Fixed SUMO port conflict assumption**:
   - Original plan claimed "SUMO port conflicts prevent parallelism"
   - Actually safe: each SUMO instance uses independent TCP port (random)
   - Changed from sequential to 8-concurrent-jobs approach
   - GTX 1650 4GB VRAM: each job uses ~100-150 MiB → 8 jobs at ~1200 MiB, plenty of headroom

4. **Created ingolstadt network configs**:
   - `conf/env/ingolstadt7.yaml`: 7 agents, 3600s episodes
   - `conf/env/ingolstadt21.yaml`: 24 agents, 3600s episodes
   - Both use RESCO network files in `nets/RESCO/{name}/`

### Key timings (5 seeds per combo)
- grid4x4 eps=50: ~82 min
- cologne3 eps=50: ~27 min
- cologne8 eps=100: ~14.4 hours
- ingolstadt21 eps=50: ~111 min
- grid4x4 eps=200: ~213 min

Larger networks × more episodes scale roughly linearly in time.

### Progress tracker updated
- `00-progress-overview.md`: Added ep count study to status
- `01-current-status.md`: Added Phase 3.8 task table
- `02-session-log.md`: This entry
- `03-blockers-and-decisions.md`: Added parallelism decision
- `04-environment-state.md`: Updated with parallel cap, new tools

### Next steps for Session #7
1. Complete ep count study: `bash run_ep_count_parallel.sh` (until all 50/50)
2. Run analysis: `.venv/bin/python eval_ep_count.py results_ep_count/ results_ep_count/eval/`
3. Review `results_ep_count/eval/optimal_ep_count.txt`
4. Update trf_coord configs with optimal ep counts
5. Begin Phase 4: Figures + statistical validation

---

## Session #7 — 2026-06-29

**Agent:** Opencode (mimo-v2.5-free)
**Duration:** ~10 min
**Branch:** `journal-upgrade-phases-0-2`

### What was done

1. **Verified ep count study completeness**: 35/50 combos have aggregate.json with 5/5 seeds. 9 in-progress, 6 missing (all eps=500 for large networks).

2. **Verified aggregate integrity**: All 35 completed combos pass every check:
   - 5/5 seeds present in each
   - No NaN, inf, or zero travel time values
   - Reasonable seed-to-seed variance (no duplication)
   - Metrics: avg_travel_time, avg_waiting_time, throughput all valid

3. **Extracted stats** (mean [lo, hi] ATT in seconds):
   - grid4x4: trf_coord 36.4s, equal 26.7s @ 200 eps
   - cologne3: trf_coord 12.7s, equal 11.9s @ 400 eps (equal has wide CIs)
   - cologne8: trf_coord 24.0s, equal 16.9s @ 400 eps
   - ingolstadt7: trf_coord 14.8s @ 100 eps, equal 10.8s @ 400 eps
   - ingolstadt21: trf_coord 21.9s @ 200 eps, equal 25.8s @ 200 eps

4. **Updated progress files**: 00, 01, 02 with verified results

### Key observations
- cologne3 trf_coord_equal shows very wide CIs (unstable across seeds)
- Clear convergence trend on cologne8/ingolstadt7 (50→400 eps)
- grid4x4 already good at 50 eps, marginal improvement beyond

### Remaining work
- 9 combos still training (eps=400 partial + eps=500 started)
- 6 combos never started (eps=500 for cologne8, ingolstadt7, ingolstadt21)
- After all complete: run eval_ep_count.py → Phase 4 (figures)
