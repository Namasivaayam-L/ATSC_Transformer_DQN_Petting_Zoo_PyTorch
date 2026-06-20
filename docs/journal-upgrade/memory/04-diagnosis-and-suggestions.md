# 04 — Diagnosis & Suggestions (Why The Plan Looks Like This)

## A. Repo bugs that invalidated the original results (grounded, code-level)
| # | Bug | Location | Effect |
|---|-----|----------|--------|
| 1 | Only ONE gradient update per episode — learn() called only at the terminal step, then break | `experiments/trf_multi_agent/dqn/main.py` (training while-loop terminal branch) | ~1 update/episode → agent barely learns. Likely explains everything. |
| 2 | `self.eval()` called INSIDE `forward()` | `dqn/dqn.py` (DeepQNetwork.forward) | BatchNorm/Dropout never in train mode; running stats corrupt. |
| 3 | No target network; bootstraps off the same net | `dqn/dqn.py` (DQN.learn, next_q from self.model) | unstable/divergent Q-values. Paper's loss eq CLAIMED theta-prime but code lacked it. |
| 4 | `self.num_states` hardcoded to 80, contradicts config `num_states=8` | `dqn/dqn.py` (DeepQNetwork.__init__) vs `dqn/config.ini` | wrong tensor shapes. |
| 5 | Evaluation reuses `train()` → keeps learning + exploring during "test" | `dqn/main.py` (second train() block) | contaminated/invalid results. |
| 6 | `num_heads = 1` | `dqn/config.ini` | "multi-head" attention is single-head. |
| 7 | Model saved to disk every gradient step | `dqn/dqn.py` (end of DQN.learn) | cripples throughput (explains "12 hours"). |
| 8 | Dead TF SAC + commented model graveyard | `sac/sac.py`, `dqn/models.py` | reproducibility hazard; remove. |

## B. Paper-level flaws (from reading the PDF)
- No quantification anywhere — entire results section is "blue line below red line"; zero numbers, no
  variance, no significance, no seeds.
- Core novelty thin: self-attention over an 8-element non-sequential vector has no real long-range
  dependency to capture; the phrase is repeated ~6x but never demonstrated.
- Duplicate/placeholder figure captions: Figs 3,4,7,8,9 share the identical caption.
- Two incorrect equations: Bellman optimality (Eq.1) missing the summation over (s',r); DQN loss (Eq.3)
  missing the expectation/mean.
- Action/state mismatch: state vector length 8 (sub-lanes) vs action array length 4, never reconciled;
  "max-pooling" misused for "argmax".
- Tooling contradiction: abstract says OpenCV for tracking, body says dlib.
- Tiny network (width 4 in the paper / 40 in the repo config), big claims; no baselines beyond vanilla DQN.

## C. The novelty fix (the heart of the plan)
The transformer must EARN its place by making the state structured. Chosen: SPATIAL neighbour coordination
— each agent attends over neighbouring intersections' states. This makes "long-range dependency" literally
true (spatial, across the road graph), yields an interpretability story (attention heatmaps over neighbours),
and connects to current MARL+transformer literature. Fallback: TEMPORAL history-attention (attend over the
last K timesteps per intersection), lower risk, still publishable, ties to DTQN.

## D. Package/method recommendations (the "what should I use" answer)
| Concern | Was | Recommended → |
|---------|-----|---------------|
| RL algorithms | hand-rolled buggy DQN + dead TF-SAC | CleanRL single-file (chosen); Tianshou/RLlib were alternatives |
| Multi-agent | independent hand-rolled agents | keep PettingZoo; IDQN now, IPPO/QMIX optional |
| Benchmark scenarios | 2x2 grid only | RESCO suite (already vendored): grid4x4 + cologne3 |
| Config | .ini + hardcoded paths | Hydra + structured configs |
| Tracking | loose CSVs | Weights & Biases (or TensorBoard) |
| Sim speed | libsumo (good) | keep libsumo; parallelize seeds |
| Stats/plots | matplotlib one-offs | rliable (IQM, performance profiles, bootstrap CIs) |
| Baselines | vanilla DQN only | Fixed-time, Max-Pressure, IDQN, MPLight |

## E. Why these specific choices under <2mo
CleanRL avoids fighting a framework on limited compute. grid4x4 isolates the coordination effect cleanly;
cologne3 adds real-topology credibility without a heavy network. Average travel time = direct comparability
with prior work. The single equal-parameter ablation is the cheapest experiment that actually proves the thesis.
