import torch.nn as nn
import torch
import torch.nn.functional as F


class RNNAgent(nn.Module):
    def __init__(self, args, agent_id):
        super(RNNAgent, self).__init__()

        self.args = args
        self.max_action = args.high_action
        input_shape = args.obs_shape[agent_id]

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.action_out=nn.Linear(args.rnn_hidden_dim, args.action_shape[agent_id])
        self.hidden_state=self.init_hidden()

    def init_hidden(self,evaluate=True):
        # make hidden states on same device as model
        if evaluate:
            return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
        else:
            return self.fc1.weight.new(self.args.batch_size, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state=None, actions=None):
        x = F.relu(self.fc1(inputs))
        hidden_state=self.hidden_state
        h_in = torch.tensor(hidden_state.reshape(-1, self.args.rnn_hidden_dim),device=torch.device('cuda'))


        h = self.rnn(x, h_in)
        x = F.relu(self.fc2(h))
        actions=F.tanh(self.action_out(x))
        self.hidden_state=h
        return {"actions": actions, "hidden_state": h}