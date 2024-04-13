import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe = T.zeros(max_len, d_model)
        position = T.arange(0, max_len, dtype=T.float).unsqueeze(1)
        div_term = T.exp(T.arange(0, d_model, 2) * (-T.log(T.tensor(10000.0)) / d_model))
        print(position.shape, div_term.shape)
        self.pe[:, 0::2] = T.sin(position * div_term)
        self.pe[:, 1::2] = T.cos(position * div_term)
        self.register_buffer('pe', self.pe)

    def forward(self, x):
        # Ensure that positional encoding matches the batch size of the input tensor
        pe = self.pe[:x.size(0), :self.d_model]  # Adjust dimensions based on input size
        return x + pe.unsqueeze(1)  # Unsqueeze to match batch size dimension

class DeepQNetwork(nn.Module):
    def __init__(self, num_states, num_actions, num_layers, learning_rate, width, num_heads, dim_feedforward, dropout):
        super(DeepQNetwork, self).__init__()
        self.num_states = num_states
        self.width = width
        self.num_actions = num_actions
        self.d_model = num_states
        self.positional_encoding = PositionalEncoding(self.d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.d_model, num_heads, dim_feedforward, self.dropout),
            num_layers=num_layers
        )
        self.fc1 = nn.Linear(self.d_model, self.d_model)
        self.fc2 = nn.Linear(self.d_model, 4)  # Output dimensionality is 2 for action probabilities
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.layer_norm1 = nn.LayerNorm(self.d_model)
        self.layer_norm2 = nn.LayerNorm(self.d_model)
        self.layer_norm3 = nn.LayerNorm(4)  # Output dimensionality is 2 for action probabilities
        self.dropout = nn.Dropout(dropout)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, x):
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = self.layer_norm1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.layer_norm2(x)
        x = self.fc2(x)
        x = self.layer_norm3(x)
        x = self.sigmoid(x)
        return x

class DQN:
    def __init__(self, ts, num_actions, num_states, width, num_heads, num_layers, dim_feedforward, dropout, batch_size, gamma, learning_rate, model_path, fine_tune_model_path=None):
        self.ts = ts
        self.width = width
        self.num_actions = num_actions
        self.num_heads = num_heads
        self.num_states = num_states
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.model_path = model_path
        self.gamma = gamma 
        self.learning_rate = learning_rate
        self.model = DeepQNetwork(self.num_states, self.num_actions, self.num_layers, self.learning_rate, self.width, self.num_heads, dim_feedforward, dropout)   
        if fine_tune_model_path:
            self.model.load_state_dict(T.load(fine_tune_model_path+ts+'.pth'))
            self.model.eval()
            print(f"Loaded {ts} Model Successfully...!")
    def act(self, state, epsilon):
        if np.random.rand() <= epsilon:
            actions = np.random.randint(self.num_actions)
        else:
            # print(type(state[0]),type(state))
            state = T.tensor(state).to(self.model.device)
            preds = self.model.forward(state)
            # print(preds.shape)
            actions = T.argmax(preds).item()
            # print(actions)
        return actions

    def learn(self, experience):
        self.model.train()
        self.model.optimizer.zero_grad()
        curr_state, actions, rewards, next_state = tuple(map(lambda x: T.tensor(np.array(x)).to(self.model.device), experience))
        # print(T.tensor(curr_state).shape)
        curr_state, next_state = tuple(map(lambda x: T.reshape(x,[self.batch_size,self.num_states]), (curr_state, next_state)))
        q_values = self.model.forward(curr_state)
        next_q_values = self.model.forward(next_state)
        q_target = rewards + self.gamma * T.max(next_q_values, dim=1)[0]
        q_target_vec = q_values.clone().detach().to(T.float32)  # Convert to float
        actions = actions.long()
        q_target_vec[np.arange(self.batch_size), actions] = q_target.detach().to(T.float32)
        loss = self.model.loss(q_target_vec, q_values).to(self.model.device)
        loss.backward()
        self.model.optimizer.step()
        T.save(self.model.state_dict(), self.model_path+f"{self.ts}.pth")