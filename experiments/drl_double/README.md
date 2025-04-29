# Double DQN Variant (experiments/drl_double)

This folder runs the Double DQN algorithm to mitigate overestimation bias in multi-agent traffic signal control:

- **Double Q-Learning**: separate online and target networks; target network updates at fixed intervals.
- **NoisyNet Exploration**: Gaussian noise layers replace ε-greedy scheduling.
- **LayerNorm** on hidden layers.
- **Prioritized Experience Replay**: proportional priorities + importance sampling.
- **n-Step Returns**: multi-step reward aggregation.

---

## File Structure

- `main.py`  
  Entry-point: reads `config.ini`, sets up `sumo_rl.parallel_env`, initializes `DoubleDQN`, runs episodes, logs metrics, saves CSVs & model snapshots.

- **Shared Modules (from `experiments/drl`)**
  - `dqn_double.py`: defines `DoubleDQN` class and `DeepQNetwork` architecture.
  - `memory.py`: prioritized & n-step replay buffer.
  - `config.ini`: shared configuration file for SUMO and model parameters.

---

## Usage

```bash
cd experiments/drl_double
python main.py
```

By default, reads `experiments/drl/config.ini`. Adjust hyperparameters there.

Outputs are saved under `out_dir` as defined in `config.ini`:
- `csv/`: per-episode metrics
- `models/`: saved `.pth` files
- `logs/`: training logs (INFO level)

---

## Dependencies

- Python 3.8+  
- PyTorch 2.x  
- `sumo_rl`, `numpy`, `tqdm`

Ensure `SUMO_HOME` is set for SUMO tools.

---

## Configuration Highlights

See `experiments/drl/config.ini` for details:
- **[Memory]**: `buffer_size`, `n_step`
- **[Model]**: `learning_rate`, `gamma`, `batch_size`, `target_update_freq` (optional)
- **[Sumo]**: network, routes, signal timings, simulation length

---

## Other Variants

- `experiments/drl_shared/`: shared-parameter DQN
- `experiments/drl_functorch/`: functorch-vectorized DQN
