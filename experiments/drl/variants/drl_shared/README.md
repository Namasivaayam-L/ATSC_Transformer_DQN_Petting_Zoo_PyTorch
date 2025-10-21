# Parameter-Sharing DQN (experiments/drl_shared)

This folder implements a shared-parameter DQN variant for multi-agent traffic signal control:

- **Shared Network**: one `DeepQNetwork` instance for all agents; actions are chosen per-agent but gradients update the same weights.
- **NoisyNet Exploration**: learnable noise in linear layers replacing ε-greedy.
- **LayerNorm**: stabilizes training with small batches.
- **Prioritized Experience Replay**: proportional priorities + importance-sampling weights.
- **n-Step Returns**: aggregates multi-step rewards.
- **AdamW + StepLR**: optimizer with weight decay and LR scheduling.

---

## File Structure

- `dqn_shared.py`  
  Defines `NoisyLinear`, `DeepQNetwork`, and `SharedDQN` class.

- `memory.py`  
  Reusable prioritized & n-step replay buffer.

- `main.py`  
  Training script using `sumo_rl.parallel_env`, reads `config.ini`, logs to console, saves models and CSVs.

- `config.ini`  
  Shared with `experiments/drl`; defines SUMO and learning hyperparameters.

---

## Usage

```bash
cd experiments/drl_shared
python main.py
```

Folder uses same `config.ini` in `experiments/drl`; customize hyperparameters there.

Logs print per-episode info; models saved to `out_dir/models/`, metrics to `out_dir/csv/`.

---

## Dependencies

- Python 3.8+  
- PyTorch 2.x  
- `sumo_rl`, `numpy`, `tqdm`

Ensure `SUMO_HOME` is set for SUMO tools.
