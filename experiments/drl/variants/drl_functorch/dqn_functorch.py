"""
Functional version of DQN using functorch for vectorized multi-agent updates.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from functorch import make_functional_with_buffers, vmap, grad
from torch.func import functional_call
import math

# NoisyNet linear layer as before
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

# Original DeepQNetwork architecture
class DeepQNetwork(nn.Module):
    def __init__(self, num_states, num_bins, num_actions, embedding_dim, num_heads, num_enc_layers, width):
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

# Functionalize a batch of agents
def initialize_agents(num_agents, device, **net_args):
    """
    Returns:
      func: functional model callable
      params: list of param-pytrees stacked over agent-dim
      buffers: list of buffer-pytrees stacked over agent-dim
      lr: learning rate
    """
    nets = [DeepQNetwork(**net_args).to(device) for _ in range(num_agents)]
    func, params_list, buffers_list = zip(*[
        make_functional_with_buffers(net) for net in nets
    ])
    # assume same func/buffers structure for all agents
    func = func[0]
    # stack each param/buffer tensor along new agent-dim
    stacked_params = [torch.stack([p[i] for p in params_list], dim=0)
                      for i in range(len(params_list[0]))]
    stacked_buffers = [torch.stack([b[i] for b in buffers_list], dim=0)
                       for i in range(len(buffers_list[0]))]
    return func, stacked_params, stacked_buffers

# loss for one agent
def loss_fn(params, buffers, batch, func, gamma):
    states, actions, rewards, next_states = batch
    q = functional_call(func, (params, buffers), states)
    q_a = q.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        q_next = functional_call(func, (params, buffers), next_states)
        target = rewards + gamma * q_next.max(dim=-1)[0]
    return F.smooth_l1_loss(q_a, target)

# update function per agent
def update_fn(params, buffers, batch, func, lr, gamma):
    grads = grad(lambda p: loss_fn(p, buffers, batch, func, gamma))(params)
    # simple gradient descent update
    new_params = [p - lr * g for p, g in zip(params, grads)]
    return new_params, buffers

# vectorized multi-agent update
def multiagent_update(stacked_params, stacked_buffers, batch, func, lr, gamma):
    # vmap over agent-dim (dim=0)
    v_update = vmap(lambda p, b: update_fn(p, b, batch, func, lr, gamma), in_dims=(0,0))
    new_params, new_buffers = v_update(stacked_params, stacked_buffers)
    return new_params, new_buffers

# Example stub
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_agents = 4
    func, params, buffers = initialize_agents(
        num_agents, device,
        num_states=10, num_bins=5, num_actions=2,
        embedding_dim=32, num_heads=1, num_enc_layers=2, width=40
    )
    # dummy batch
    batch = (
        torch.randn(8, 10, device=device),            # states
        torch.randint(0, 2, (8,), device=device),      # actions
        torch.randn(8, device=device),                 # rewards
        torch.randn(8, 10, device=device),             # next_states
    )
    new_params, new_buffers = multiagent_update(
        params, buffers, batch, func, lr=1e-3, gamma=0.99
    )
    print("Updated parameters for all agents.")
