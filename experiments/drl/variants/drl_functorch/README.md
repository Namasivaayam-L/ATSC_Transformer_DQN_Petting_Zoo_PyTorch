# Functorch DQN Variant (experiments/drl_functorch)

This folder implements a functional DQN using PyTorch’s Functorch API for vectorized multi-agent updates in traffic signal control:

- **Functional Networks**: Converts modules to functional form (`make_functional_with_buffers`, `functional_call`).
- **vmap Updates**: Applies `vmap` to perform simultaneous gradient updates across agents.
- **NoisyNet Exploration**: Learnable noise layers for exploration.
- **LayerNorm**: Normalization for stable training.
- **Prioritized Experience Replay** & **n-Step Returns**: Shared from core `memory.py`.

---

## File Structure

- `dqn_functorch.py`  
  Defines `DeepQNetwork`, `NoisyLinear`, functional wrappers (`initialize_agents`, `loss_fn`, `update_fn`, `multiagent_update`).

- `main.py`  
  Entry-point: reads `config.ini` via `utils.setup_env`, initializes functional agents, runs a stub multi-agent update, logs results.

- `config.ini`  
  Shared file in `experiments/drl`; defines SUMO settings, model and memory hyperparameters.

- `memory.py`  
  Uses same prioritized & n-step buffer from core folder.

---

## Usage

```bash
cd experiments/drl_functorch
python main.py
```

Reads `experiments/drl/config.ini`. Adjust parameters there.

---

## Dependencies

- Python 3.8+  
- PyTorch 2.x (with Functorch integrated)  
- `sumo_rl`, `numpy`, `tqdm`

Ensure `SUMO_HOME` is set.

---

## Configuration Highlights

See `experiments/drl/config.ini`:
- **[Sumo]**: network & route, signal timings, simulation length
- **[Model]**: `batch_size`, `learning_rate`, `gamma`, transformer dims
- **[Memory]**: `buffer_size`, `n_step`

---

## Related Variants

- `experiments/drl/`: original DQN  
- `experiments/drl_shared/`: parameter-sharing DQN  
- `experiments/drl_double/`: Double DQN
