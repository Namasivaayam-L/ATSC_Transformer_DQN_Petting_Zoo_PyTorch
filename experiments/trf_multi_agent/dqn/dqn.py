import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np
        
class DeepQNetwork(nn.Module):
    def __init__(self, num_states, num_actions, num_layers, learning_rate, width):
        super(DeepQNetwork, self).__init__()
        self.num_states = num_states
        self.width = width
        self.num_actions = num_actions

        self.transformer = nn.Transformer(
            d_model=self.num_states,
            nhead=4,
            num_encoder_layers=3,
            dropout=0.1,
            batch_first=True
        )

        self.layers = nn.ModuleList([
            nn.Linear(self.num_states, self.width),
            nn.LayerNorm(self.width),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.width, self.width),
            nn.LayerNorm(self.width),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.width, self.num_actions)
        ])
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        self.eval()
        if state.dim() == 1:
            state = state.unsqueeze(0)
        preds = self.transformer(state, state)
        for layer in self.layers:    
            preds = layer(preds)   
        return preds
        
class DQN:
    def __init__(self, ts, num_actions, num_states, width, num_layers,batch_size, gamma, learning_rate, model_path, fine_tune_model_path=None):
        self.ts = ts
        self.width = width
        self.num_actions = num_actions
        self.num_states = num_states
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.model_path = model_path
        self.gamma = gamma 
        self.learning_rate = learning_rate
        self.model = DeepQNetwork(self.num_states, self.num_actions, self.num_layers, self.learning_rate, self.width)   
        if fine_tune_model_path:
            self.model.load_state_dict(T.load(fine_tune_model_path+ts+'.pth'))
            self.model.eval()
            print(f"Loaded {ts} Model Successfully...!")
    def act(self, state, epsilon):
        if np.random.rand() <= epsilon:
            actions = np.random.randint(self.num_actions)
        else:
            print(type(state[0]),type(state))
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