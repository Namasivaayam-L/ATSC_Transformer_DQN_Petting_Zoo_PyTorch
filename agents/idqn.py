"""Independent DQN (IDQN) — CleanRL-style single-file implementation.

One Q-network per agent with shared architecture, independent weights.
Fixes bugs #1 (training cadence), #2 (eval mode), #3 (target network),
#4 (hardcoded shapes), #7 (checkpointing).

Usage:
    from agents.idqn import IDQNAgent, ReplayBuffer
    agent = IDQNAgent(obs_dim=960, act_dim=8, cfg=cfg)
"""
from __future__ import annotations

import copy
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-size FIFO replay buffer."""

    def __init__(self, capacity: int):
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool) -> None:
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self.buffer, batch_size)
        states = torch.as_tensor(np.stack([t.state for t in batch]), dtype=torch.float32)
        actions = torch.as_tensor([t.action for t in batch], dtype=torch.long)
        rewards = torch.as_tensor([t.reward for t in batch], dtype=torch.float32)
        next_states = torch.as_tensor(np.stack([t.next_state for t in batch]), dtype=torch.float32)
        dones = torch.as_tensor([t.done for t in batch], dtype=torch.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Q-network (MLP)
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """MLP Q-network: obs_dim -> hidden -> hidden -> act_dim."""

    def __init__(self, obs_dim: int, act_dim: int,
                 hidden_sizes: Tuple[int, ...] = (128, 128),
                 dueling: bool = False):
        super().__init__()
        self.dueling = dueling
        layers: List[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h

        if dueling:
            # Value stream
            self.value_stream = nn.Sequential(*layers, nn.Linear(in_dim, 1))
            # Advantage stream
            self.advantage_stream = nn.Sequential(*layers, nn.Linear(in_dim, act_dim))
        else:
            self.feature = nn.Sequential(*layers)
            self.q_head = nn.Linear(in_dim, act_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dueling:
            val = self.value_stream(x)
            adv = self.advantage_stream(x)
            return val + adv - adv.mean(dim=-1, keepdim=True)
        return self.q_head(self.feature(x))


# ---------------------------------------------------------------------------
# IDQN Agent
# ---------------------------------------------------------------------------

class IDQNAgent:
    """Independent DQN agent for a single intersection.

    One agent instance per traffic signal. Each has its own Q-network,
    target network, and replay buffer.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        *,
        hidden_sizes: Tuple[int, ...] = (128, 128),
        lr: float = 1e-4,
        gamma: float = 0.99,
        tau: Optional[float] = None,
        target_update_interval: int = 1000,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        learning_starts: int = 1000,
        train_freq: int = 4,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_episodes: int = 500,
        dueling: bool = False,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.target_update_interval = target_update_interval
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.device = torch.device(device)

        self.online_net = QNetwork(obs_dim, act_dim, hidden_sizes, dueling).to(self.device)
        self.target_net = copy.deepcopy(self.online_net)
        self.target_net.eval()
        for p in self.target_net.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

        self._update_count = 0
        self._episode_count = 0

    # -- epsilon schedule ---------------------------------------------------

    @property
    def epsilon(self) -> float:
        frac = min(self._episode_count / max(self.epsilon_decay_episodes, 1), 1.0)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    # -- action selection (exploitation + exploration) ----------------------

    def act(self, obs: np.ndarray, epsilon: Optional[float] = None,
            valid_actions: Optional[int] = None) -> int:
        """Epsilon-greedy action selection."""
        if epsilon is None:
            epsilon = self.epsilon
        if random.random() < epsilon:
            n = valid_actions if valid_actions is not None else self.act_dim
            return random.randrange(n)
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.online_net(obs_t)
            if valid_actions is not None and valid_actions < self.act_dim:
                mask = torch.full_like(q, float("-inf"))
                mask[:, :valid_actions] = 0
                q = q + mask
            return int(q.argmax(dim=-1).item())

    # -- learning -----------------------------------------------------------

    def learn(self) -> Optional[float]:
        """Sample a minibatch and perform one gradient step.

        Returns the loss value, or None if not enough data / not time to train.
        """
        if len(self.buffer) < max(self.learning_starts, self.batch_size):
            return None
        if self._update_count % self.train_freq != 0:
            self._update_count += 1
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q values
        q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: action selected by online net, evaluated by target net
        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(dim=-1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q_values * (1.0 - dones)

        loss = F.mse_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        self._update_count += 1

        # Hard update of target network
        if self._update_count % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        # Soft update (Polyak)
        if self.tau is not None:
            for p, tp in zip(self.online_net.parameters(), self.target_net.parameters()):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

        return loss.item()

    def on_episode_end(self) -> None:
        """Called at the end of each episode."""
        self._episode_count += 1

    # -- save / load --------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save({
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self._update_count,
            "episode_count": self._episode_count,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._update_count = ckpt.get("update_count", 0)
        self._episode_count = ckpt.get("episode_count", 0)
