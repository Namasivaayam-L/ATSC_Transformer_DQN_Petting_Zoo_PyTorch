"""
Double-DQN variant with NoisyNet exploration, prioritized replay, and n-step returns.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import logging
from pathlib import Path

from memory import Memory

# NoisyNet linear layer
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters(sigma_init)
        self.reset_noise()

    def reset_parameters(self, sigma_init):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(sigma_init)
        self.bias_sigma.data.fill_(sigma_init)

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            w, b = self.weight_mu, self.bias_mu
        return F.linear(x, w, b)

# Core Q-network
class DeepQNetwork(nn.Module):
    def __init__(self, num_states, num_bins, num_actions,
                 embedding_dim, num_heads, num_enc_layers, width):
        super().__init__()
        self.embedding = nn.Linear(num_states, embedding_dim)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads,
            dim_feedforward=width, dropout=0.1,
            activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_enc_layers)
        self.fc1 = NoisyLinear(num_states * embedding_dim, width)
        self.norm1 = nn.LayerNorm(width)
        self.fc2 = NoisyLinear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.fc3 = NoisyLinear(width, width // 2)
        self.norm3 = nn.LayerNorm(width // 2)
        self.fc4 = NoisyLinear(width // 2, width // 4)
        self.norm4 = nn.LayerNorm(width // 4)
        self.fc5 = NoisyLinear(width // 4, num_actions)

    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        x = self.embedding(state)
        x = self.transformer(x)
        x = x.flatten(start_dim=1)
        x = torch.relu(self.norm1(self.fc1(x)))
        x = torch.relu(self.norm2(self.fc2(x)))
        x = torch.relu(self.norm3(self.fc3(x)))
        x = torch.relu(self.norm4(self.fc4(x)))
        return self.fc5(x)

# Double DQN agent
class DoubleDQN:
    def __init__(self, num_actions, num_states, width,
                 num_heads, num_enc_layers, embedding_dim, num_bins,
                 batch_size, gamma, learning_rate,
                 buffer_size, n_step, target_update_freq=1000,
                 alpha=0.6, beta=0.4, beta_increment_per_sampling=1e-3):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DeepQNetwork(num_states, num_bins, num_actions,
                                  embedding_dim, num_heads,
                                  num_enc_layers, width).to(self.device)
        self.target_model = DeepQNetwork(num_states, num_bins, num_actions,
                                         embedding_dim, num_heads,
                                         num_enc_layers, width).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.memory = Memory(buffer_size, alpha, beta,
                             beta_increment_per_sampling,
                             n_step, gamma)
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_step = n_step
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        self.loss_fn = nn.HuberLoss()
        self.learn_step_counter = 0
        self.target_update_freq = target_update_freq

    def act(self, state):
        state = torch.tensor(state, device=self.device).unsqueeze(0)
        self.model.train(); self.model.reset_noise()
        with torch.no_grad():
            q_vals = self.model(state)
        return int(q_vals.argmax().item())

    def learn(self):
        # sample a batch
        states, actions, rewards, next_states, indices, weights = \
            self.memory.sample(self.batch_size)
        curr = torch.tensor(np.array(states), device=self.device)
        nxt = torch.tensor(np.array(next_states), device=self.device)
        actions = torch.tensor(actions, device=self.device).long()
        rewards = torch.tensor(rewards, device=self.device).float()
        weights = torch.tensor(weights, device=self.device).float()
        curr = curr.view(self.batch_size, self.n_step, -1)
        nxt = nxt.view(self.batch_size, self.n_step, -1)

        # compute Q and targets
        self.model.train(); self.optimizer.zero_grad()
        q_eval = self.model(curr).gather(1, actions.view(-1,1)).squeeze()
        # Double-DQN: select action via online model, evaluate via target
        next_actions = self.model(nxt).argmax(dim=1, keepdim=True)
        next_q_target = self.target_model(nxt).gather(1, next_actions).squeeze().detach()
        target = rewards + (self.gamma ** self.n_step) * next_q_target

        loss_per = F.smooth_l1_loss(q_eval, target, reduction='none')
        loss = (loss_per * weights).mean()
        loss.backward(); self.optimizer.step(); self.scheduler.step()

        # update priorities and target network
        new_prios = loss_per.abs().cpu().numpy()
        self.memory.update_priorities(indices, new_prios)
    def save(self, path):
        """Save model state to file."""
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), str(Path(path) / f"{self.ts}.pth"))
        self.logger.info(f"Model saved to {path}/{self.ts}.pth")

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
