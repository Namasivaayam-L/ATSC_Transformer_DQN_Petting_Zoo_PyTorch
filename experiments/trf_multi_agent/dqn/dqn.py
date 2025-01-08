import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, num_states, num_bins, num_actions, embedding_dim, num_heads, num_enc_layers, width, learning_rate):
        super(DeepQNetwork, self).__init__()

        self.num_states = num_states
        self.num_bins = num_bins
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=num_bins,
            nhead=num_heads,
            dim_feedforward=width,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_enc_layers)

        self.fc1 = nn.Linear(num_states * num_bins, width)
        self.norm1 = nn.BatchNorm1d(width)

        self.fc2 = nn.Linear(width, width)
        self.norm2 = nn.BatchNorm1d(width)

        self.fc3 = nn.Linear(width, width // 2)
        self.norm3 = nn.BatchNorm1d(width // 2)

        self.fc4 = nn.Linear(width // 2, width // 4)
        self.norm4 = nn.BatchNorm1d(width // 4)

        self.fc5 = nn.Linear(width // 4, num_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=learning_rate)
        self.loss = nn.HuberLoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        self.eval()
        if state.dim() == 1:
            state = state.unsqueeze(0)

        x = self.transformer(state.float())
        x = x.flatten(start_dim=1)

        x = torch.relu(self.norm1(self.fc1(x)))
        x = torch.relu(self.norm2(self.fc2(x)))
        x = torch.relu(self.norm3(self.fc3(x)))
        x = torch.relu(self.norm4(self.fc4(x)))
        q_values = self.fc5(x)

        return q_values


class DQN:
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
        self.model = DeepQNetwork(self.num_states, self.num_bins, self.num_actions, self.embedding_dim, self.num_heads, self.num_enc_layers, self.width, self.learning_rate)
        if fine_tune_model_path:
            self.model.load_state_dict(torch.load(fine_tune_model_path+ts+'.pth',map_location = self.model.device))
            self.model.eval()
            print(f"Loaded {ts} Model Successfully...!")
    def act(self, state, epsilon):
        if np.random.rand() <= epsilon:
            actions = np.random.randint(self.num_actions)
        else:
            # print(type(state[0]),type(state))
            state = torch.tensor(state).unsqueeze(0).to(self.model.device)
            preds = self.model.forward(state)
            # print(preds.shape)
            actions = torch.argmax(preds).item()
            # print(actions)
        return actions

    def learn(self, experience):
        self.model.train()
        self.model.optimizer.zero_grad()
        curr_state, actions, rewards, next_state = tuple(map(lambda x: torch.tensor(np.array(x)).to(self.model.device), experience))
        # print(torch.tensor(curr_state).shape)
        curr_state, next_state = tuple(map(lambda x: torch.reshape(x,[self.batch_size,self.num_states, self.num_bins]), (curr_state, next_state)))
        q_values = self.model.forward(curr_state)
        next_q_values = self.model.forward(next_state)
        q_target = rewards + self.gamma * torch.max(next_q_values, dim=1)[0].detach()
        q_target_vec = q_values.clone()
        actions = actions.long()
        q_target_vec[np.arange(self.batch_size), actions] = q_target.float()
        loss = self.model.loss(q_target_vec, q_values).to(self.model.device)
        loss.backward() 
        self.model.optimizer.step()
        torch.save(self.model.state_dict(), self.model_path+f"{self.ts}.pth")