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
            "MADDPGLSTMactor":{"policy":MADDPG_lstm_actor(agent_id),"sep_action":False},
            "FAC":{"policy":FAC(args,agent_id),"sep_action":True},
            "FACMAC_SCH":{"policy":FACMACSCH(args,agent_id),"sep_action":True},
            "MADDPGLSTM":{"policy":MADDPGLSTM(args,agent_id), "sep_action":False},           
        }
        
        self.policy=self.policy_class[policy_name]["policy"]
        self.sep_action=self.policy_class[policy_name]["sep_action"]
        
        
        # if getattr(self.args,"algorithm","FACMAC") == "FACMAC":
        #     self.policy=FACMAC(args,agent_id)
        # elif getattr(self.args,"algorithm","FACMAC") == "MADDPG":
        #     self.policy = MADDPG(args, agent_id)
        # elif getattr(self.args,"algorithm","FACMAC") == "IDDPG":
        #     self.policy = IDDPG(args, agent_id)
        # elif getattr(self.args,"algorithm","FACMAC") == "MADDPGLSTMactor":
        #     self.policy = MADDPG_lstm_actor(args, agent_id)
        # elif getattr(self.args,"algorithm","FACMAC") == "FACMACLSTMactor":
        #     self.policy = FACMAC_lstm_actor(args, agent_id)
        # elif getattr(self.args,"algorithm","FACMAC") == "FAC":
        #     self.policy = FAC(args, agent_id)
        # elif getattr(self.args,"algorithm","FACMAC") == "FACMAC_SCH":
        #     self.policy = FACMACSCH(args, agent_id)
        # elif getattr(self.args,"algorithm","FACMAC") == "FACLSTM":
        #     self.policy = FACLSTM(args, agent_id)

        # elif getattr(self.args,"algorithm","FACMAC") == "MADDPGLSTM":
        #     self.policy = MADDPGLSTM(args, agent_id)

    def select_action(self, o, noise_rate, epsilon):
        device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        if np.random.uniform() < epsilon:
            # epsilon-greedy solution, my personal thought is that's not enough.
            u = np.random.uniform(-self.args.high_action, self.args.high_action, self.args.action_shape[self.agent_id])

        else:
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0).to(device)
            
            
            # if getattr(self.args, "algorithm", "FACMAC") == "FACMAC" or getattr(self.args, "algorithm", "FACMAC") == "IDDPG" or getattr(self.args, "algorithm", "FACMAC") == "FAC" or getattr(self.args,"algorithm","FACMAC") == "FACMAC_SCH":
            if self.sep_action:
                pi = self.policy.actor_network(inputs)["actions"].squeeze(0)

            # elif getattr(self.args, "algorithm", "FACMAC") == "MADDPG"or getattr(self.args, "algorithm", "FACMAC") == "MADDPGLSTMactor" or getattr(self.args, "algorithm", "FACMAC") == "FACMACLSTMactor" or getattr(self.args,"algorithm","FACMAC") == "FACLSTM" or getattr(self.args,"algorithm","FACMAC") == "MADDPGLSTM":
            else:
                pi = self.policy.actor_network(inputs).squeeze(0)

            
            u = pi.cpu().numpy()

            noise = noise_rate * self.args.high_action * np.random.randn(*u.shape)  # gaussian noise
            u += noise
            u = np.clip(u, -self.args.high_action, self.args.high_action)


        u_cal=torch.tensor(u.copy(),device=device)

        return u_cal

    def learn(self, transitions, other_agents):
        self.policy.train(transitions, other_agents)

