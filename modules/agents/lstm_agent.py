import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor, self).__init__()
        self.max_action = args.high_action
        self.lstm=nn.LSTM(input_size=args.obs_shape[agent_id],hidden_size=args.rnn_hidden_dim,num_layers=1,batch_first=True)
        self.fc1 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)#obs_shape是什么shape?
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 64)
        #self.fc3 = nn.Linear(64, 64)
        self.action_out = nn.Linear(64, args.action_shape[agent_id])
        # self.h=None
        # self.c=None

    def forward(self, x):#input shape[batch_size,seq_length,obs_shape]
        # if type(self.h)==None :
        #     seq,a=self.lstm(x,[self.h,self.c])#[batch_size,seq_length,rnn_hidden_dim]
        #     self.h=a[0]
        #     self.c=a[1]
        # else:
        #     seq,a=self.lstm(x)
        #     self.h=a[0]
        #     self.c=a[1]
        seq,a=self.lstm(x)





        x = F.relu(self.fc1(seq))
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))
        if len(actions.shape)==3:
            actions=actions[:,-1,:]

        return actions
