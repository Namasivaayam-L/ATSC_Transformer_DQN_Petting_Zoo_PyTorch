# DQN Experiments (experiments/drl)

This folder contains the original Deep Q-Network (DQN) implementation for multi-agent traffic signal control, with enhancements:

- **NoisyNet exploration** instead of ε-greedy
- **LayerNorm** layers for stability with small batches
- **Prioritized experience replay** (proportional)
- **n-step returns** for faster reward propagation

---

## File Structure

- **dqn.py**  
  - `NoisyLinear`: learnable Gaussian noise layer  
  - `DeepQNetwork`: transformer-based Q-network with noisy linear heads  
  - `DQN` agent: handles action selection, learning, model saving

- **memory.py**  
  - `Memory`: deque-based n-step buffer + proportional priorities + importance sampling weights

- **main.py**  
  Orchestrates training loop using `sumo_rl.parallel_env`, logs metrics, updates CSV, saves models.

- **config.ini**  
  INI file defining SUMO settings, model hyperparameters, and memory parameters.

---

## Dependencies

- Python 3.8+  
- PyTorch 2.x  
- `sumo_rl` (SUMO 1.22 bindings)  
- `numpy`, `tqdm`  
- SUMO and `SUMO_HOME` environment variable

Install core libs via:
```bash
pip install torch torchvision torchaudio
pip install numpy tqdm sumo_rl
```

---

## Configuration (`config.ini`)

### [Sumo]
- `net_file`: path to SUMO network XML
- `route_file`: path to route definition XML
- `out_dir`: base output directory
- `single_agent`: True/False
- `use_gui`: True/False
- `num_seconds`: simulation length (s)
- `yellow_time`, `min_green`, `max_green`: signal timings
- `reward_fn`: reward function key
- `num_states`: observation vector size per agent
- `num_bins`: discretization bins (if applicable)

### [Model]
- `model_name`: identifier ("dqn")
- `num_episodes`: training episodes
- `batch_size`: replay batch size
- `learning_rate`: optimizer LR
- `epsilon`, `decay`, `min_epsilon`: ε schedule (legacy; replaced by NoisyNet)
- `gamma`: discount factor
- `num_heads`, `num_enc_layers`: transformer encoder config
- `embedding_dim`, `width`: network dimensions
- `fine_tune`: True/False
- `test_num_episodes`: evaluation episodes
- `fine_tune_model_path`: path for loading pretrained model

### [Memory]
- `buffer_size`: max replay buffer length
- `n_step`: number of steps for multi-step returns

---

## Usage

1. Ensure `SUMO_HOME` is set and SUMO tools are in `$PATH`.
2. Edit `config.ini` for your scenario.
3. From project root, run:
   ```bash
   python experiments/drl/main.py
   ```
4. Check outputs in `out_dir`:
   - `csv/`: per-episode metrics
   - `models/`: saved `.pth` snapshots
   - `logs/`: debug logs if enabled

---

## Logging & Outputs

- Uses Python `logging` (INFO level by default).
- Progress printed via `tqdm`.
- CSV metrics updated each step via `utils.update_csv`.

---

## Extending

Variants live in sibling folders:
- `experiments/drl_shared/` (parameter-sharing DQN)
- `experiments/drl_double/` (Double-DQN)
- `experiments/drl_functorch/` (functorch vmap)

Please refer to their respective `README.md` for details.
