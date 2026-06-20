# 01 — Session Context (Where We Started)

## The original paper (consumed in full, all 11 pages)
- Title: "Traffic Sense: Optimizing City Traffic with Transformer-infused DRL and Computer Vision."
- Authors: Dr. J. Angela Jennifa Sujana (HOD) and L. Namasivaayam, Mepco Schlenk Engineering College.
- IEEE-format LaTeX, created 2024-12-16. Reads as a Final-Year-Project write-up.
- Core claim: a DQN with a transformer layer ("TRF-DQN") beats vanilla DQN at minimizing vehicle waiting
  time at signalized intersections. State extracted from SUMO during training and from real traffic
  camera video via DETR + dlib tracking at inference. Evaluated in SUMO; "implemented" on real city
  camera footage with a homegrown multi-video simulator.
- Reward functions: difference-in-waiting-time, pressure-based, queue-based.
- The author's own framing in this session: "I just used the transformer layer for the sake of using it.
  I had no idea what I was doing back then."

## The author's goal for this session
Make the work PUBLISHABLE in an indexed journal. Specifically:
- Make it technically strong; get the metrics base and achieved (ABSOLUTE quantification).
- Author was confused about which packages/methods to use → wanted sophisticated, defensible recommendations.
- FOCUS ON SIMULATION ONLY (the computer-vision half is set aside).
- Wanted a detailed phase-wise plan + a non-technical backlog, delivered as a build spec.

## The repo and its ACTUAL tech stack (read directly, not assumed)
Repo: ATSC_Transformer_DQN_Petting_Zoo_PyTorch.
- **Good foundation (keep):** SUMO-RL (vendored under `sumo_rl/`) + PettingZoo 1.24.3 + Gymnasium 0.29.1,
  PyTorch 2.3.0, `libsumo` 1.20.0 fast backend. `sumo_rl/environment/resco_envs.py` is present — RESCO is
  THE standard traffic-signal RL benchmark and was barely used.
- **Active model:** `experiments/trf_multi_agent/dqn/dqn.py` — a TransformerEncoder (embedding → encoder →
  5 FC layers → Q-values), RMSprop, HuberLoss. Config: 2x2 grid, num_states=8, num_heads=1, num_enc_layers=2,
  embedding_dim=32, width=40, batch_size=300, 2000 episodes.
- **Dead code (cut):** `experiments/trf_multi_agent/sac/sac.py` (TensorFlow SAC, continuous actions forced
  onto a discrete problem; TF deps not even in requirements). `experiments/trf_multi_agent/dqn/models.py`
  (entirely commented-out graveyard of MLP/LSTM/CNN/Transformer experiments). `experiments/trf_dqn/`
  (single-agent leftover).
- **Missing:** no stable-baselines3, no RLlib, no Tianshou, no wandb/tensorboard, no Hydra, no Double/Dueling
  DQN, no target network, no statistical tooling. Config via `.ini` files + a hardcoded `/home/namachu/...`
  absolute path in `dqn/main.py`.

## What was NOT in scope
- The computer-vision pipeline (DETR/dlib/video) — explicitly cut for this paper.
- Syntax/prose issues in the paper — explicitly set aside by the author ("we can set aside the syntax issues").
