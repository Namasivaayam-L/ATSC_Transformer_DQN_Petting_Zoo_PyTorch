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
