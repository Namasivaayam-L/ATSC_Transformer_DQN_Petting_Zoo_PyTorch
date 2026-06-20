# 🚦 ATSC Transformer DQN with PettingZoo

A research framework for **Adaptive Traffic Signal Control (ATSC)** using **Transformer-based Deep Q-Networks (DQN)** within **PettingZoo** multi-agent environments. Integrates with **SUMO** for realistic traffic simulation.

---

## 🚀 Features

- ✅ Multi-agent Reinforcement Learning for ATSC
- ✅ Transformer-based DQN for optimal signal control
- ✅ Seamless integration with SUMO traffic simulator
- ✅ PettingZoo compatibility for scalable multi-agent environments
- ✅ Focus on traffic optimization and flow management

---

## 🧰 Tech Stack

| Component            | Tech                                   |
|----------------------|----------------------------------------|
| RL Framework         | PettingZoo                             |
| Deep Learning        | PyTorch                                |
| Simulation           | SUMO                                   |
| Algorithms           | DQN, Transformer                       |
| Language             | Python                                 |
| Data Handling        | NumPy, Pandas                          |
| Environment Inter.   | Gymnasium                              |

---

## 🛠️ Setup Instructions

To get started, clone the repository and install the dependencies:

```bash
git clone https://github.com/Namasivaayam-L/ATSC_Transformer_DQN_Petting_Zoo_PyTorch.git
cd ATSC_Transformer_DQN_Petting_Zoo_PyTorch
pip install -r requirements.txt
```

### 🔧 SUMO

Ensure SUMO is properly installed and configured on your system for traffic simulations.

---

## 🧑‍💻 Run Experiments

The `experiments/` directory contains various implementations:

-   **Transformer-based DQN**: `experiments/trf_dqn/`
-   **Multi-agent DQN**: `experiments/trf_multi_agent/`
-   **SAC Implementations**: Explore `experiments/` for Soft Actor-Critic examples.

---

## 📂 Project Structure

```
.
├── experiments/                  # Contains various experiment implementations (trf_dqn, trf_multi_agent, SAC)
├── sumo_rl/                      # Core PettingZoo environment wrappers for SUMO
│   ├── agents/                   # Agent implementations
│   ├── environment/              # SUMO-PettingZoo environment setup
│   └── util/                     # Utility functions for RL environments
├── utils/                        # General utilities (plotting, config reading)
├── README.md                     # Project overview and documentation
├── pyproject.toml                # Project metadata and dependencies
└── requirements.txt              # Python dependencies
```

---

Built for research in **AI/ML**, **Reinforcement Learning**, **Traffic Control**, and **Multi-Agent Systems**.
