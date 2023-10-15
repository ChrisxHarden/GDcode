import torch
import torch.nn as nn


import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor, self).__init__()
        self.max_action = args.high_action
        
        
        self.embedding=nn.Linear(args.obs_shape[agent_id],args.obs_shape[agent_id])
        self.transformer = nn.Transformer(
            d_model=args.obs_shape[agent_id],
            nhead=8,
            num_encoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        
    
        self.action_out = nn.Linear(args.obs_shape[agent_id], args.action_shape[agent_id])


    def forward(self, x):#input shape[batch_size,seq_length,obs_shape]

        x=self.embedding(x)
        x=self.transformer(x,x)
        actions = self.max_action * torch.tanh(self.action_out(x))
        if len(actions.shape)==3:
            actions=actions[:,-1,:]

        return actions
