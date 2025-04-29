import os
from venv import logger
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F


class ActorCriticNetwork(nn.Module):
    def __init__(self, num_states, num_bins, num_actions, embedding_dim, num_heads,
                 num_enc_layers, width, learning_rate):
        super(ActorCriticNetwork, self).__init__()

        self.num_states = num_states
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.num_states,
            nhead=num_heads,
            dim_feedforward=width,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            transformer_layer, num_layers=num_enc_layers)

        # self.embedding = nn.Linear(self.num_states, self.num_states * self.embedding_dim)

        self.fc1 = nn.Linear(num_states , width)
        self.norm1 = nn.BatchNorm1d(width)

        self.fc2 = nn.Linear(width, width)
        self.norm2 = nn.BatchNorm1d(width)

        self.fc3 = nn.Linear(width, width // 2)
        self.norm3 = nn.BatchNorm1d(width // 2)

        self.fc4 = nn.Linear(width // 2, width // 4)
        self.norm4 = nn.BatchNorm1d(width // 4)

        self.fc_actor = nn.Linear(width // 4, num_actions)
        self.fc_critic = nn.Linear(width // 4, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)  # Combining optimizers
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        print(f"Shape at input layer: {state.shape}")
        if state.dim() == 1:
            state = state.unsqueeze(0)
        # Apply embedding layer
        # x = self.embedding(state)
        print(f"Shape before embedding layer: {state.shape}")
        x = self.embedding(state)
        print(f"Shape after embedding layer: {x.shape}")
        x = self.transformer(x)
        print(f"Shape after transformer layer: {x.shape}")
        x = x.flatten(start_dim=1)
        print(f"Shape after flattening: {x.shape}")
        x = torch.relu(self.norm1(self.fc1(x)))
        print(f"Shape after layer 1: {x.shape}")
        x = torch.relu(self.norm2(self.fc2(x)))
        print(f"Shape after layer 2: {x.shape}")
        x = torch.relu(self.norm3(self.fc3(x)))
        print(f"Shape after layer 3: {x.shape}")
        x = torch.relu(self.norm4(self.fc4(x)))
        print(f"Shape after layer 4: {x.shape}")

        logits = self.fc_actor(x)
        dist = torch.distributions.Categorical(logits=logits)

        value = self.fc_critic(x)

        return dist, value




class A2C:
    def __init__(self, ts, num_actions, num_states, width, num_heads, num_enc_layers, embedding_dim, num_bins, batch_size, gamma, learning_rate, model_path, fine_tune_model_path=None):
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
        self.model = ActorCriticNetwork(self.num_states, self.num_bins, self.num_actions, self.embedding_dim, self.num_heads, self.num_enc_layers, self.width, self.learning_rate)
        if fine_tune_model_path:
            self.model.load_state_dict(torch.load(fine_tune_model_path+ts+'.pth',map_location = self.model.device))
            self.model.eval()
            print(f"Loaded {ts} Model Successfully...!")

    def act(self, state):
        state = torch.tensor(state).unsqueeze(0).to(self.model.device)
        dist, _ = self.model(state)
        action = dist.sample()
        return action.item()


    def learn(self, key, ep, experience):
        self.model.train()
        curr_state, actions, rewards, next_state = tuple(map(lambda x: torch.tensor(np.array(x)).to(self.model.device), experience))
        curr_state, next_state = tuple(map(lambda x: torch.reshape(x,[self.batch_size,self.num_states, 80]), (curr_state, next_state)))
        
        dist, value = self.model(curr_state)
        _, next_value = self.model(next_state)

        advantage = rewards + self.gamma * next_value.squeeze(-1) - value.squeeze(-1)
        
        actor_loss = -(dist.log_prob(actions) * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss

        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()

        with open(os.path.join(self.model_path, 'actor_loss.csv'), 'a+') as f:
            f.write(f'{key},{ep},{str(actor_loss.item())}\n')
        with open(os.path.join(self.model_path, 'critic_loss.csv'), 'a+') as f:
            f.write(f'{key},{ep},{str(critic_loss.item())}\n')

        torch.save(self.model.state_dict(), self.model_path+f"{self.ts}.pth")
