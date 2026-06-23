"""Spatial Neighbour-Coordination Transformer Q-network.

Each agent attends over the states of its **neighbouring** intersections
(self + adjacent TLS nodes) via multi-head self-attention.  This makes
"long-range spatial dependency" literally true and yields an
interpretability story (attention over neighbours).

Usage:
    from agents.trf_coord import TrfCoordAgent
    agent = TrfCoordAgent(obs_dim=960, act_dim=8, adj={"A0": ["A1", "B0"]}, ...)
"""
from __future__ import annotations

import copy
import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Replay buffer (shared with IDQN — duplicated here for single-file clarity)
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    state: np.ndarray      # (1+K, obs_dim) token matrix
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
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
# Coordination Transformer Q-network
# ---------------------------------------------------------------------------

class CoordTransformerQ(nn.Module):
    """Transformer encoder over neighbour tokens → Q-values.

    Architecture:
        1. Linearly project each obs_dim token to ``embedding_dim``.
        2. Add a learnable ``token_type`` embedding (self vs neighbour-i).
        3. Stack of ``TransformerEncoderLayer`` with multi-head attention.
        4. Pool: take the [CLS]-like self-token (index 0) or mean-pool.
        5. MLP head → Q-values.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        max_tokens: int = 9,          # 1 self + up to 8 neighbours
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_enc_layers: int = 2,
        dropout: float = 0.1,
        pool: str = "self",           # "self" | "mean"
    ):
        super().__init__()
        self.pool = pool

        # Token projection
        self.token_proj = nn.Linear(obs_dim, embedding_dim)

        # Learnable positional / type embedding: token_type[i] for position i
        self.token_type = nn.Parameter(torch.randn(1, max_tokens, embedding_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_enc_layers)
        self.norm = nn.LayerNorm(embedding_dim)

        # Q-head
        self.q_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, act_dim),
        )

        # Store attention weights for interpretability
        self._attn_weights: Optional[torch.Tensor] = None

    @property
    def attn_weights(self) -> Optional[torch.Tensor]:
        """Per-head attention weights from the last forward pass (for logging)."""
        return self._attn_weights

    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            tokens: (batch, n_tokens, obs_dim) — stacked self + neighbour obs.
            mask:   (batch, n_tokens) bool True = masked (padding).

        Returns:
            Q-values: (batch, act_dim)
        """
        B, N, _ = tokens.shape

        # Project to embedding space
        h = self.token_proj(tokens) + self.token_type[:, :N, :]

        # Transformer encoder (standard forward)
        h = self.transformer(h, src_key_padding_mask=mask)
        self._attn_weights = None  # populated separately for interpretability

        h = self.norm(h)

        # Pool
        if self.pool == "self":
            pooled = h[:, 0, :]            # (B, emb)
        else:
            if mask is not None:
                h = h.masked_fill(mask.unsqueeze(-1), 0.0)
                pooled = h.sum(dim=1) / (~mask).float().sum(dim=1, keepdim=True).clamp(min=1)
            else:
                pooled = h.mean(dim=1)

        return self.q_head(pooled)

    def forward_with_attention(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Forward pass that also returns attention weights for interpretability."""
        B, N, _ = tokens.shape
        h = self.token_proj(tokens) + self.token_type[:, :N, :]

        attn_maps = []
        for layer in self.transformer.layers:
            # norm_first: norm -> attn -> residual -> norm -> ffn -> residual
            h1 = layer.norm1(h)
            h_attn, attn_w = layer.self_attn(h1, h1, h1, need_weights=True, average_attn_weights=False)
            h = h + h_attn
            h2 = layer.norm2(h)
            h = h + layer.linear2(layer.activation(layer.linear1(h2)))
            attn_maps.append(attn_w)

        self._attn_weights = torch.stack(attn_maps, dim=1)
        h = self.norm(h)

        if self.pool == "self":
            pooled = h[:, 0, :]
        else:
            pooled = h.mean(dim=1)

        return self.q_head(pooled), self._attn_weights


# ---------------------------------------------------------------------------
# Spatial Coordination Agent
# ---------------------------------------------------------------------------

class TrfCoordAgent:
    """Spatial coordination transformer agent — one per traffic signal.

    Same RL machinery as IDQN (target net, Double DQN, replay buffer)
    but the Q-network body is the CoordTransformerQ instead of an MLP.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        *,
        adj: Dict[str, List[str]],
        agent_id: str,
        max_neighbours: int = 8,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_enc_layers: int = 2,
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
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.adj = adj
        self.agent_id = agent_id
        self.max_neighbours = max_neighbours
        self.n_tokens = 1 + min(len(adj.get(agent_id, [])), max_neighbours)
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

        self.online_net = CoordTransformerQ(
            obs_dim, act_dim,
            max_tokens=self.n_tokens,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_enc_layers=num_enc_layers,
        ).to(self.device)
        self.target_net = copy.deepcopy(self.online_net)
        self.target_net.eval()
        for p in self.target_net.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

        self._update_count = 0
        self._episode_count = 0

    @property
    def epsilon(self) -> float:
        frac = min(self._episode_count / max(self.epsilon_decay_episodes, 1), 1.0)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def _build_tokens(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Build token matrix (n_tokens, obs_dim) from raw obs dict.

        Pads shorter obs to self.obs_dim (max across all agents).
        """
        import numpy as np
        self_obs = obs_dict[self.agent_id].flatten().astype(np.float32)
        neighbours = self.adj.get(self.agent_id, [])
        k = min(len(neighbours), self.max_neighbours)
        tokens = np.zeros((1 + k, self.obs_dim), dtype=np.float32)
        tokens[0, :len(self_obs)] = self_obs[:self.obs_dim]
        for i, n_id in enumerate(neighbours[:k]):
            n_obs = obs_dict[n_id].flatten().astype(np.float32)
            tokens[1 + i, :len(n_obs)] = n_obs[:self.obs_dim]
        return tokens

    def act(self, obs_dict: Dict[str, np.ndarray], epsilon: Optional[float] = None,
            valid_actions: Optional[int] = None) -> int:
        """Epsilon-greedy action selection from tokenised observations.

        Args:
            valid_actions: if set, mask Q-values beyond this count (for
            heterogeneous action spaces like cologne8).
        """
        if epsilon is None:
            epsilon = self.epsilon
        if random.random() < epsilon:
            n = valid_actions if valid_actions is not None else self.act_dim
            return random.randrange(n)
        tokens = self._build_tokens(obs_dict)
        with torch.no_grad():
            tokens_t = torch.as_tensor(tokens, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.online_net(tokens_t)
            if valid_actions is not None and valid_actions < self.act_dim:
                mask = torch.full_like(q, float("-inf"))
                mask[:, :valid_actions] = 0
                q = q + mask
            return int(q.argmax(dim=-1).item())

    def learn(self) -> Optional[float]:
        if len(self.buffer) < max(self.learning_starts, self.batch_size):
            return None
        if self._update_count % self.train_freq != 0:
            self._update_count += 1
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = states.to(self.device)           # (B, n_tokens, obs_dim)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(dim=-1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        loss = F.mse_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        self._update_count += 1

        if self._update_count % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        if self.tau is not None:
            for p, tp in zip(self.online_net.parameters(), self.target_net.parameters()):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

        return loss.item()

    def on_episode_end(self) -> None:
        self._episode_count += 1

    def save(self, path: str) -> None:
        torch.save({
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self._update_count,
            "episode_count": self._episode_count,
            "adj": self.adj,
            "agent_id": self.agent_id,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "n_tokens": self.n_tokens,
            "config": {
                "embedding_dim": self.token_proj.out_features if hasattr(self, 'token_proj') else self.online_net.token_proj.out_features,
                "num_heads": self.online_net.transformer.layers[0].self_attn.num_heads,
                "num_enc_layers": len(self.online_net.transformer.layers),
            },
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._update_count = ckpt.get("update_count", 0)
        self._episode_count = ckpt.get("episode_count", 0)
