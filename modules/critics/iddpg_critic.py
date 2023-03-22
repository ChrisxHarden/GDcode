import torch
import torch.nn as nn
import torch.nn.functional as F


class IDDPGCritic(nn.Module):
    def __init__(self,  args):
        super(IDDPGCritic, self).__init__()
        self.args = args
        self.max_action=self.args.high_action

        self.init_hidden()

        # Set up network layers
        self.fc1 = nn.Linear(args.obs_shape[0] + args.action_shape[0], args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.q_out = nn.Linear(args.rnn_hidden_dim, 1)

    def init_hidden(self):
        # make hidden states on same device as model
        self.hidden_states = None

    def forward(self, obs, action, hidden_state=None):

        action /= self.max_action
        x = torch.cat([obs, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q = self.q_out(x)
        return q
