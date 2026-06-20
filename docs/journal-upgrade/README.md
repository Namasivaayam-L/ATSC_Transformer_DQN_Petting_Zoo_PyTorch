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
