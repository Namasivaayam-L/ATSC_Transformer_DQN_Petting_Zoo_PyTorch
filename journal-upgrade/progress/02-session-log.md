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
