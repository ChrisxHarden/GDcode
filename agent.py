import numpy as np
import torch
import os
from algorithms.facmac import FACMAClearner as FACMAC
from algorithms.maddpg import MADDPG
from algorithms.iddpg import IDDPG
from algorithms.MADDPG_lstm_agent import MADDPG_lstm_actor
from algorithms.FACMAC_lstm_agent import FACMAC_lstm_actor
from algorithms.test_fac import FAC
from algorithms.FACMAC_SCH import FACMACSCH
from algorithms.FACLSTM import FACLSTM
from algorithms.MADDPGLSTM import MADDPG_lstm as MADDPGLSTM

class Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        policy_name=getattr(self.args,"algorithms","MADDPG")
        self.policy_class={
            "FACMAC":{"policy":FACMAC(args,agent_id),"sep_action":True},
            "MADDPG":{"policy":MADDPG(args,agent_id),"sep_action":False},
            "IDDPG":{"policy":IDDPG(args,agent_id),"sep_action":True},
            "MADDPGLSTMactor":{"policy":MADDPG_lstm_actor(args,agent_id),"sep_action":False},
            "FAC":{"policy":FAC(args,agent_id),"sep_action":True},
            "FACMAC_SCH":{"policy":FACMACSCH(args,agent_id),"sep_action":True},
            "MADDPGLSTM":{"policy":MADDPGLSTM(args,agent_id), "sep_action":False},           
        }
        
        self.policy=self.policy_class[policy_name]["policy"]
        self.sep_action=self.policy_class[policy_name]["sep_action"]
        

    def select_action(self, o, noise_rate, epsilon):
        device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        if np.random.uniform() < epsilon:
            # epsilon-greedy solution, my personal thought is that's not enough.
            u=torch.rand(self.args.action_shape[self.agent_id],device=device)
            u=2*self.args.high_action*u-self.args.high_action

        else:
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0).to(device)
            
            
            if self.sep_action:
                u = self.policy.actor_network(inputs)["actions"].squeeze(0)

            else:
                u = self.policy.actor_network(inputs).squeeze(0)

            noise = noise_rate*self.args.high_action * torch.randn(*u.shape,device=device)
            u += noise

        return u

    def learn(self, transitions, other_agents):
        self.policy.train(transitions, other_agents)

