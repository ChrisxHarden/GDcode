import torch.nn as nn
import torch.nn.functional as F


class MLPAgent(nn.Module):
    def __init__(self, args,agent_id):
        super(MLPAgent, self).__init__()
        self.args = args
        self.max_action = args.high_action
        input_shape=args.obs_shape[agent_id]

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.action_out=nn.Linear(args.rnn_hidden_dim,args.action_shape[agent_id])

        self.agent_return_logits = getattr(self.args, "agent_return_logits", False)
        self.hidden_state=self.init_hidden()

    def init_hidden(self,eval=None):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()#将hidden清零

    def forward(self, inputs, hidden_state=None, actions=None):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        if self.agent_return_logits:
            actions = self.action_out(x)
        else:
            actions = self.max_action * F.tanh(self.action_out(x))
        #self.hidden_state=hidden_state
        return {"actions": actions, "hidden_state": hidden_state}