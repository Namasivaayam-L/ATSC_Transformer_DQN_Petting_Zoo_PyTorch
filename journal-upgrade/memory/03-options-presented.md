# 03 — Options Presented (What Else Was On The Table)

Full menus offered during the session. ✅ = chosen. This preserves the roads not taken and the trade-offs,
so a future session understands WHY the chosen path was chosen and what the alternatives cost.

## Q1 — Architectural scope
- Keep DQN, do it right (Double/Dueling/Rainbow; lower risk, modest novelty).
- Upgrade to modern MARL (IPPO/QMIX/justified transformer; higher effort, stronger novelty).
- ✅ Recommend what fits (decide after repo assessment).

## Q2 — Publication bar
- Solid conference / workshop (clean quant + a few baselines + seeds; weeks).
- ✅ Indexed journal (SCI/Scopus; strong baselines, stats, ablations, multiple scenarios; months).
- Not sure yet.

## Q3 — Simulation setup
- SUMO + a clean RL wrapper (SUMO-RL / PettingZoo / Gymnasium).
- ✅ Whatever's most rigorous (most defensible/reproducible stack, even if it changes the repo).
- Match the existing repo.

## Q4 — RL library philosophy
- ✅ CleanRL — single-file, transparent, citable; easiest to defend/modify; multi-agent is DIY.
- Tianshou — batteries-included PyTorch RL (DQN/Double/Dueling/Rainbow/PPO/SAC); more abstraction.
- RLlib (Ray) — industrial, native multi-agent, distributed; steep learning curve, opaque.

## Q5 — Transformer novelty angle
- ✅ Spatial: neighbour coordination — attend over neighbouring intersections; strongest journal angle;
  makes "long-range dependency" literally true; MOST engineering effort.
- Temporal: history attention — attend over a time-window per intersection; justifies temporal long-range;
  ties to DTQN; medium effort. (This is the chosen FALLBACK.)
- Both spatial + temporal — strongest novelty, highest risk/scope-creep.
- Let me recommend per timeline.

## Q6 — Timeline / compute
- 3–4 months, single GPU.
- 6+ months, decent compute.
- ✅ Tight (<2 months), limited compute.

## Q7 — Risk gate handling
- ✅ Build spatial, temporal as fallback (explicit Week-4 gate; safest path to finishing).
- Spatial or bust (no fallback; higher ceiling, real deadline risk).
- Start temporal, upgrade if time (safest for deadline, slightly lower ceiling).

## Q8 — Computer-vision scope
- ✅ Simulation only — cut CV entirely.
- Sim now, CV as 'future work'.
- Keep CV as a real component (much larger scope; not recommended under <2mo).

## Q9 — Deliverable format
- Markdown plan file in the repo.
- Plan + ADO backlog items (via ado-ops skill).
- ✅ Just show me here first. (Later refined by the author into: one master build-spec .md.)

## Q10 — Coordination scale
- ✅ Small grid + 1 real-ish net — RESCO grid4x4 (16 agents) + cologne3 (3 agents, real topology).
- Single corridor / arterial — cheapest compute, weaker 'network' novelty.
- Larger network (ingolstadt/grid) — strongest scalability claim, heavy compute, risky under <2mo.

## Q11 — Primary metric
- ✅ Average travel time — RESCO/literature standard; hardest to game; best comparability.
- Average waiting time — continuity with original paper; easier to game.
- Throughput — network efficiency framing; sensitive to demand setup.

## Q12 — Root directory name
- traffic-sense-journal.
- ✅ journal-upgrade.
- atsc-paper-v2.

## Q13 — Build mechanism
- Embedded contents + instructions (agent-agnostic, readable).
- ✅ Shell-script style (mkdir + heredoc cat; hardened for Windows Git Bash + spaces-in-path).

## Q14 — Phase self-containment
- ✅ Fully standalone per phase.
- Lean phases + shared context file.

## Q15 — Code grounding
- ✅ Fully grounded in the real repo (paths/line-level bugs/configs).
- Mostly grounded, some generic.
- Higher-level / portable.

## Q16 — CV scope re-confirmation (asked alongside risk gate)
- ✅ Simulation only — cut CV entirely (re-affirmed).
