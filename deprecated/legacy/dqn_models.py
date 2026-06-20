# import stat
# import torch as T
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import numpy as np
# from torchinfo import summary 
    # self.layers = nn.ModuleList([
    #         nn.Linear(self.num_states, self.width),
    #         nn.BatchNorm1d(self.width),
    #         nn.ReLU(),
    #         nn.Dropout(p=0.5),  # Adjust the dropout rate as needed
    #         *[nn.Linear(self.width, self.width) for _ in range(num_layers)],
    #         nn.BatchNorm1d(self.width),
    #         nn.ReLU(),
    #         nn.Dropout(p=0.5),
    #         nn.Linear(self.width, self.num_actions)
    #     ])
        
    # def forward(self, state):
    #     self.eval()
    #     if state.dim() == 1:
    #         # If input is 1D, add a batch dimension
    #         state = state.unsqueeze(0)
        
    #     preds = state  # Initialize with input state
    #     for layer in self.layers:
    #         preds = layer(preds)
    #         if isinstance(layer, nn.Linear):
    #             preds = F.relu(preds)  # Apply ReLU after Linear layers
    #     return preds
        # self.layers = nn.ModuleList([
        #     nn.Linear(self.num_states, self.width),  # Input layer with adjusted width
        #     *[nn.Linear(self.width, self.width) for _ in range(num_layers)],  # Hidden layers with new width
        #     nn.Linear(self.width, self.num_actions)  # Output layer remains unchanged
        # ])
    # def forward(self, state):
    #     lstm_output, _ = self.lstm(state)
    #     lstm_output = lstm_output[:, -1, :]  # Use the output from the last time step
    #     for layer in self.layers:
    #         lstm_output = layer(lstm_output)
    #     return lstm_output
        # print(self.num_states, num_actions)
# self.lstm_hidden_size = self.num_states * 2
# # LSTM layer
# self.lstm = nn.LSTM(input_size= 1, hidden_size=self.lstm_hidden_size, batch_first=True)
# # Feedforward layers
# self.layers = nn.ModuleList([
#     nn.Linear(self.lstm_hidden_size, self.width),
#     nn.BatchNorm1d(self.width),
#     nn.ReLU(),
#     nn.Dropout(p=0.5),
#     *[nn.Linear(self.width, self.width) for _ in range(num_layers)],
#     nn.BatchNorm1d(self.width),
#     nn.ReLU(),
#     nn.Dropout(p=0.5),
#     nn.Linear(self.width, self.num_actions)
# ])

# self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
# self.loss = nn.MSELoss()
# self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
# self.to(self.device)

# def forward(self, state):
# # Assuming `state` is a tensor of shape (batch_size, num_states)
# # Add a sequence dimension
# state = state.view(1, 21, 21)  # Assuming state has shape (1, 21)
# print('State shape: ', state.shape)
# # LSTM layer
# lstm_output, _ = self.lstm(state)
# # Extract the output from the last time step
# lstm_output = lstm_output[:, -1, :]
# # Feedforward layers
# for layer in self.layers:
#     lstm_output = layer(lstm_output)
# return lstm_output



        # Transformer layer
    #     self.transformer = nn.Transformer(
    #         d_model=num_states,
    #         nhead= 3,
    #         num_encoder_layers=num_layers,
    #         dropout= 0.2,
    #         batch_first=True
    #     )

    #     # Feedforward layers
    #     self.layers = nn.ModuleList([
    #         nn.Linear(num_states, self.width),  # Adjusted input size
    #         nn.BatchNorm1d(self.width),
    #         nn.ReLU(),
    #         nn.Dropout(p=0.5),
    #         *[nn.Linear(self.width, self.width) for _ in range(num_layers)],
    #         nn.BatchNorm1d(self.width),
    #         nn.ReLU(),
    #         nn.Dropout(p=0.5),
    #         nn.Linear(self.width, num_actions)
    #     ])

    #     self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    #     self.loss = nn.HuberLoss()
    #     self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    #     self.to(self.device)

    # def forward(self, state):
    #     self.eval()
    #     if state.dim() == 1:
    #         # If input is 1D, add a batch dimension
    #         state = state.unsqueeze(0)
    #     # print('State Shape: ', state.shape)
    #     preds = self.transformer(state, state)
    #     # print('Transformer Output Shape',preds.shape)
    #     preds = T.mean(preds, dim=0)
    #     preds = preds.unsqueeze(0)
    #     # print('Transformer Output Shape',preds.shape)

    #     for layer in self.layers:
    #         preds = layer(preds)
    #         # if isinstance(layer, nn.Linear):
    #         #     preds = F.relu(preds)  # Apply ReLU after Linear layers
    #     return preds



        # CNN layers
    #     self.cnn_layers = nn.Sequential(
    #         nn.Conv1d( 1, 64, kernel_size=3, stride=1, padding=1),
    #         nn.ReLU(),
    #         nn.MaxPool1d(kernel_size=2, stride=2),
    #         nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
    #         nn.ReLU(),
    #         nn.MaxPool1d(kernel_size=2, stride=2),
    #     )

    #     # Feedforward layers
    #     self.linear_layers = nn.Sequential(
    #         nn.Linear(128, self.width),  # Adjusted output size based on CNN output
    #         nn.BatchNorm1d(self.width),
    #         nn.ReLU(),
    #         nn.Dropout(p=0.5),
    #         *[nn.Linear(self.width, self.width) for _ in range(num_layers)],
    #         nn.BatchNorm1d(self.width),
    #         nn.ReLU(),
    #         nn.Dropout(p=0.5),
    #         nn.Linear(self.width, self.num_actions)
    #     )

    #     self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    #     self.loss = nn.HuberLoss()
    #     self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    #     self.to(self.device)

    # def forward(self, state):
    #     self.eval()
    #     if state.dim() == 1:
    #         # If input is 1D, add a batch dimension
    #         state = state.unsqueeze(0)
    #     # Apply CNN layers
    #     preds = self.cnn_layers(state)
    #     print('Cnn op shape: ', preds.shape)
    #     # Flatten the CNN output
    #     preds = preds.view(preds.size(1), -1)
    #     preds = F.relu(self.linear_layers(preds))
    #     print('Output shape',preds.shape)
    #     print(T.argmax(preds))
    #     return preds


# class DeepQNetwork(nn.Module):
#     def __init__(self, num_states, num_actions, num_layers, learning_rate, width):
#         super(DeepQNetwork, self).__init__()
#         self.num_states = num_states
#         self.width = width
#         self.num_actions = num_actions
#         # Transformer layer
#         self.transformer = nn.Transformer(
#             d_model=num_states,
#             nhead= 3,
#             num_encoder_layers= 3,
#             dropout= 0.2,
#             batch_first=True
#         )

#         # Feedforward layers
#         self.layers = nn.ModuleList([
#             nn.Linear(num_states, self.width),  # Adjusted input size
#             nn.BatchNorm1d(self.width),
#             nn.ReLU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(self.width, self.width),
#             nn.BatchNorm1d(self.width),
#             nn.ReLU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(self.width, num_actions)
#         ])

#         self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
#         self.loss = nn.HuberLoss()
#         self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
#         self.to(self.device)

#     def forward(self, state):
#         self.eval()
#         if state.dim() == 1:
#             # If input is 1D, add a batch dimension
#             state = state.unsqueeze(0)
#         # print('State Shape: ', state.shape)
#         preds = self.transformer(state, state)
#         # print('Transformer Output Shape',preds.shape)
#         # preds = T.mean(preds)
#         # preds = preds.unsqueeze(0)
#         # print('Transformer Output Shape',preds.shape)

#         for layer in self.layers:
#             preds = layer(preds)
#             # if isinstance(layer, nn.Linear):
#             #     preds = F.relu(preds)  # Apply ReLU after Linear layers
#         return preds

#PE EMBedding
#          self.nhead = 3
#         self.dropout = .5
#         # Embedding layer
#         self.embedding = nn.Embedding(num_states, self.width)
        
#         self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.width, nhead=self.nhead, dim_feedforward=width, dropout=self.dropout)
#         self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)

#         self.output_layer = nn.Linear(num_states, num_actions)

#         self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
#         self.loss = nn.CrossEntropyLoss()  # Assuming classification task
#         self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
#         self.to(self.device)

#     def forward(self, state):
#         self.eval()
#         # Embedding layer
#         state = state.to(T.long)
#         embedded_state = self.embedding(state)
#         trf_preds = self.transformer_encoder(embedded_state)
#         # print("trf preds", trf_preds.shape, len(trf_preds.shape))
#         # Max pooling over the sequence dimension
#         if len(trf_preds.shape)==3:  #Batch training
#             trf_preds,_ = T.max(trf_preds, dim=2)
#         else:  #Predictions
#             trf_preds,_ = T.max(trf_preds, dim=1)
#         # print("trf preds", trf_preds.shape)
#         # Apply softmax to the output logits
#         action_logits = self.output_layer(trf_preds)
#         # print('preds :', action_logits.shape)
#         return action_logits



        # self.final_layer = nn.Linear(self.width, self.num_actions)

        # self.layers = nn.ModuleList([
        #         nn.Linear(self.num_states, self.width),
        #         nn.BatchNorm1d(self.width),
        #         nn.ReLU(),
        #         nn.Dropout(p=0.5),  # Adjust the dropout rate as needed
        #         *[nn.Linear(self.width, self.width) for _ in range(num_layers)],
        #         nn.BatchNorm1d(self.width),
        #         nn.ReLU(),
        #         nn.Dropout(p=0.5),
        #         nn.Linear(self.width, self.num_actions)
        #     ])
        # self.layers = nn.ModuleList([
        #     nn.Linear(self.num_states, self.width),
        #     nn.LayerNorm(self.width),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.1),
        #     nn.Linear(self.width, self.width),
        #     nn.LayerNorm(self.width),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.1),
        # ])
        #   def forward(self, state):
        #         self.eval()
        #         if state.dim() == 1:
        #             # If input is 1D, add a batch dimension
        #             state = state.unsqueeze(0)
                
        #         preds = state  # Initialize with input state
        #         for layer in self.layers:
        #             preds = layer(preds)
        #             if isinstance(layer, nn.Linear):
        #                 preds = nn.functional.relu(preds)  # Apply ReLU after Linear layers
        #         return preds




















