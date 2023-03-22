import torch
import torch.nn as nn
import torch.nn.functional as F




class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.max_action = args.high_action
        self.lstm = nn.LSTM(input_size=sum(args.obs_shape) + sum(args.action_shape), hidden_size=args.rnn_hidden_dim, num_layers=1,
                            batch_first=True)
        self.fc1 = nn.Linear(args.rnn_hidden_dim, 64)
        self.fc2 = nn.Linear(64, 64)

        self.q_out = nn.Linear(64, 1)#output q score of certain action under certain observation

    def forward(self, state, action):
        state = torch.cat(state, dim=-1)
        for i in range(len(action)):
            action[i] /= self.max_action
        action = torch.cat(action, dim=-1)
        x = torch.cat([state, action], dim=-1)
        x,_=self.lstm(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        q_value = self.q_out(x)[:,-1,:]
        return q_value
