import torch
import torch.nn as nn
import torch.nn.functional as F


class FACMACCritic(nn.Module):
    def __init__(self, agent_id, args):
        super(FACMACCritic, self).__init__()
        self.args = args
        self.max_action=self.args.high_action

        self.init_hidden()

        # Set up network layers
        self.fc1 = nn.Linear(sum(args.obs_shape) + args.action_shape[agent_id], args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.q_out = nn.Linear(args.rnn_hidden_dim, 1)

    def init_hidden(self):
        # make hidden states on same device as model
        self.hidden_states = None

    def forward(self, obs, action, hidden_state=None):
        state=torch.cat(obs,dim=1)
        action /= self.max_action
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q = self.q_out(x)
        return q
