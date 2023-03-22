import torch
import torch.nn as nn
import torch.nn.functional as F


class MADDPGCritic(nn.Module):
    def __init__(self, scheme, args):
        super(MADDPGCritic, self).__init__()
        self.args = args
        self.max_action = args.high_action


        # Set up network layers
        self.fc1 = nn.Linear(sum(args.obs_shape) + sum(args.action_shape), args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.q_out = nn.Linear(args.rnn_hidden_dim, 1)



    def forward(self, state, action, hidden_state=None):
        state = torch.cat(state, dim=1)
        for i in range(len(action)):
            action[i] /= self.max_action
        action = torch.cat(action, dim=1)
        x = torch.cat([state, action], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q = self.q_out(x)

        return q, hidden_state
