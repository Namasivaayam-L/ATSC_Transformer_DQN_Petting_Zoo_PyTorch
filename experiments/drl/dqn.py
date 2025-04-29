import os
import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import torch.nn.functional as F
import logging
from pathlib import Path

# module logger
logger = logging.getLogger(__name__)

# NoisyNet linear layer for learned exploration
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

    def forward(self, input):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            w, b = self.weight_mu, self.bias_mu
        return F.linear(input, w, b)

class DeepQNetwork(nn.Module):
    def __init__(self, num_states, num_bins, num_actions, embedding_dim, num_heads, num_enc_layers, width, learning_rate):
        super(DeepQNetwork, self).__init__()

        self.num_states = num_states
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim

        # Embedding layer for continuous input
        self.embedding = nn.Linear(self.num_states, self.embedding_dim)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=width,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
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

        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.loss = nn.HuberLoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        # learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        logger.info(f"DeepQNetwork initialized | num_states={self.num_states} | num_bins={self.num_bins} | num_actions={self.num_actions} | embedding_dim={self.embedding_dim} | num_heads={self.num_heads} | enc_layers={self.num_enc_layers} | width={self.width} | learning_rate={learning_rate} | device={self.device}")

    def forward(self, state):
        logger.debug(f"DeepQNetwork forward | input shape={state.shape}")
        if state.dim() == 1:
            state = state.unsqueeze(0)
        # Apply embedding layer
        x = self.embedding(state)

        x = self.transformer(x)
        x = x.flatten(start_dim=1)
        x = torch.relu(self.norm1(self.fc1(x)))
        x = torch.relu(self.norm2(self.fc2(x)))
        x = torch.relu(self.norm3(self.fc3(x)))
        x = torch.relu(self.norm4(self.fc4(x)))
        q_values = self.fc5(x)
        logger.debug(f"DeepQNetwork forward | output shape={q_values.shape}")
        return q_values


class DQN:
    def __init__(self, ts, num_actions, num_states, width, num_heads, num_enc_layers, embedding_dim, num_bins, batch_size, gamma, learning_rate, model_path, fine_tune_model_path=None, logger=None, n_step=1, ):
        self.ts = ts
        self.width = width
        self.num_actions = num_actions
        self.num_heads = num_heads
        self.num_enc_layers = num_enc_layers
        self.num_states = num_states
        self.embedding_dim = embedding_dim
        self.num_bins = num_bins
        self.batch_size = batch_size
        self.model_path = model_path
        self.gamma = gamma 
        self.learning_rate = learning_rate
        self.n_step = n_step
        self.model = DeepQNetwork(self.num_states, self.num_bins, self.num_actions, self.embedding_dim, self.num_heads, self.num_enc_layers, self.width, self.learning_rate)
        # use main's logger or fallback
        self.logger = logger or logging.getLogger(__name__)
        if fine_tune_model_path:
            self.model.load_state_dict(torch.load(fine_tune_model_path+ts+'.pth', map_location=self.model.device))
            self.model.eval()
            self.logger.info(f"Loaded model for TS {ts} from {fine_tune_model_path}{ts}.pth")
            print(f"Loaded {ts} Model Successfully...!")
        self.logger.info(f"DQN initialized | TS={self.ts} | actions={self.num_actions} | states={self.num_states} | bins={self.num_bins} | batch_size={self.batch_size} | n_step={self.n_step}")

    def act(self, state, epsilon=None):
        # Noisy-Net exploration: reset noise and act greedily
        state = torch.tensor(state).unsqueeze(0).to(self.model.device)
        self.model.train()
        self.model.reset_noise()
        self.logger.debug("NoisyNet noise reset for action selection")
        with torch.no_grad():
            q_vals = self.model(state)
        action = q_vals.argmax().item()
        self.logger.info(f"DQN.act | selected action={action}")
        return action

    def learn(self, key, ep, experience):
        self.logger.info(f"DQN.learn | key={key} | ep={ep} | batch_size={len(experience[0])} | n_step={self.n_step}")
        states, actions, rewards, next_states, indices, weights = experience
        # to tensors
        curr_state = torch.tensor(np.array(states), device=self.model.device)
        actions = torch.tensor(actions, device=self.model.device).long()
        rewards = torch.tensor(rewards, device=self.model.device).float()
        next_state = torch.tensor(np.array(next_states), device=self.model.device)
        weights = torch.tensor(weights, device=self.model.device).float()
        # legacy reshape
        curr_state, next_state = tuple(map(lambda x: torch.reshape(x, [self.batch_size, self.num_states, 80]), (curr_state, next_state)))
        self.model.train()
        self.model.optimizer.zero_grad()
        # forward
        q_values = self.model(curr_state)
        next_q_values = self.model(next_state)
        # select Q for taken actions
        q_value_actions = q_values.gather(1, actions.view(-1,1)).squeeze()
        # compute n-step target
        next_q_max = next_q_values.max(dim=1)[0].detach()
        target_q = rewards + (self.gamma ** self.n_step) * next_q_max
        # compute per-sample Huber loss and weight
        import torch.nn.functional as F
        loss_per_sample = F.smooth_l1_loss(q_value_actions, target_q, reduction='none')
        loss = (loss_per_sample * weights).mean()
        self.logger.debug(f"Loss computed: {loss.item()}")
        loss.backward()
        self.model.optimizer.step()
        # step LR scheduler
        self.model.scheduler.step()
        # save model
        path = Path(self.model_path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), str(path / f"{self.ts}.pth"))
        self.logger.info(f"DQN.learn | model saved to {path / f"{self.ts}.pth"}")
        # return indices and new priorities (abs TD-errors)
        new_prios = loss_per_sample.abs().cpu().detach().numpy()
        return indices, new_prios