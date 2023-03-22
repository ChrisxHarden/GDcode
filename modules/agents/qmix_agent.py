import torch.nn as nn
import torch.nn.functional as F


class QMIXRNNAgent(nn.Module):
    def __init__(self, args, agent_id):
        super(QMIXRNNAgent, self).__init__()
        self.args = args
        input_shape=args.obs_shape[agent_id]

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.action_shape[agent_id])

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


class FFAgent(nn.Module):
    def __init__(self, args, agent_id):
        super(FFAgent, self).__init__()
        self.args = args
        input_shape=args.obs_shape[agent_id]

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.action_shape[agent_id])

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))

        h = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q, h