import torch
import os
from modules.agents.lstm_agent import Actor as LSTMActor
from modules.critics.facmac import FACMACCritic as Critic
from modules.mixers.qmix import QMixer
from modules.mixers.qmix_ablations import QMixerNonmonotonic
from modules.mixers.vdn import VDNMixer


class FACMAC_lstm_actor:
    def __init__(self, args, agent_id):  # 因为不同的agent的obs、act维度可能不一样，所以神经网络不同,需要agent_id来区分
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0

        # create the network
        self.actor_network = LSTMActor(args, agent_id)
        self.critic_network = Critic(agent_id,args)

        # build up the target network
        self.actor_target_network = LSTMActor(args, agent_id)
        self.critic_target_network = Critic(agent_id,args)

        self.mixer_network = None
        self.mixer_target_network = None

        # create the mixer network
        if getattr(self.args, "mixer", "qmix") == "qmix":
            self.mixer_network = QMixer(args)
            self.mixer_target_network = QMixer(args)
        elif getattr(self.args, "mixer", "qmix") == "qmix_non":
            self.mixer_network = QMixerNonmonotonic(args)
            self.mixer_target_network = QMixerNonmonotonic(args)
        elif getattr(self.args, "mixer", "qmix") == "vdn":
            self.mixer_network = VDNMixer()
            self.mixer_target_network = VDNMixer()
        else:
            raise Exception("unknown mixer {}".format(getattr(self.args, "mixer", "qmix")))

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        self.mixer_target_network.load_state_dict(self.mixer_network.state_dict())

        # create the optimizer

        self.critic_param = list(self.critic_network.parameters()) + list(self.mixer_network.parameters())

        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_param, lr=self.args.lr_critic)

        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir + '/' + self.args.scenario_name+'/share_param='+str(self.args.share_param)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + self.args.algorithm
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + 'agent_%d' % agent_id
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        # 加载模型
        if os.path.exists(self.model_path + '/actor_params.pkl'):
            self.actor_network.load_state_dict(torch.load(self.model_path + '/actor_params.pkl'))
            self.critic_network.load_state_dict(torch.load(self.model_path + '/critic_params.pkl'))
            self.mixer_network.load_state_dict(torch.load(self.model_path + '/mixer_params.pkl'))
            print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
                                                                          self.model_path + '/actor_params.pkl'))
            print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,
                                                                           self.model_path + '/critic_params.pkl'))


        if torch.cuda.is_available():
            self.cuda()

    # soft update
    def _soft_update_target_network(self):#对target_network的更新
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)


        for target_param, param in zip(self.mixer_target_network.parameters(), self.mixer_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    # update the network
    def train(self, transitions, other_agents):
        device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32,device=device)
        r = transitions['r_%d' % self.agent_id]  # 训练时只需要自己的reward
        r_s=r[:,-1].squeeze()
        o, u, o_next = [], [], []  # 用来装每个agent经验中的各项 seq_version
        o_s,u_s,o_next_s=[],[],[]
        for agent_id in range(self.args.n_agents):
            o.append(transitions['o_%d' % agent_id])
            o_s.append(transitions['o_%d' % agent_id][:,-1,:].squeeze())
            u.append(transitions['u_%d' % agent_id])
            u_s.append(transitions['u_%d' % agent_id][:, -1, :].squeeze())
            o_next.append(transitions['o_next_%d' % agent_id])
            o_next_s.append(transitions['o_next_%d' % agent_id][:, -1, :].squeeze())

        state= torch.stack(o_s, dim=-1)
        state_next = torch.stack(o_next_s, dim=-1)

        # calculate the target Q value function
        u_next = []
        q_next = []
        with torch.no_grad():
            # 得到下一个状态对应的动作
            if not self.args.share_param:
                index = 0
                for agent_id in range(self.args.n_agents):
                    if agent_id == self.agent_id:
                        u_next.append(self.actor_target_network.forward(o_next[agent_id]))
                        q_next_agent = self.critic_target_network.forward(o_next_s, u_next[agent_id]).detach()
                        q_next.append(q_next_agent)
                    else:
                        #
                        u_next.append(other_agents[index].policy.actor_target_network(o_next[agent_id]))
                        q_next_agent = other_agents[index].policy.critic_target_network.forward(o_next_s,
                                                                                                u_next[agent_id]).detach()
                        q_next.append(q_next_agent)
                        index += 1
            else:

                for agent_id in range(self.args.n_agents):

                    u_next.append(self.actor_target_network.forward(o_next[agent_id]))
                    q_next_agent = self.critic_target_network.forward(o_next_s, u_next[agent_id]).detach()
                    q_next.append(q_next_agent)



            q_next = torch.stack(q_next, dim=-1)
            if self.mixer_network is not None and getattr(self.args, "mixer", "qmix") == "vdn":
                q_next = self.mixer_target_network.forward(q_next)
            elif self.mixer_network is None:
                pass
            else:
                q_next = self.mixer_target_network.forward(q_next, state_next)

            target_q = (r_s.unsqueeze(1) + self.args.gamma * q_next).detach()

        # the q loss

        index = 0
        q = []
        if not self.args.share_param:
            for agent_id in range(self.args.n_agents):
                if agent_id == self.agent_id:
                    # u.append(self.actor_network.forward(o[agent_id], self.actor_network.hidden_state)["actions"])
                    q_agent = self.critic_network.forward(o_s, u_s[agent_id])
                    q.append(q_agent)
                else:

                    # u.append(other_agents[index].policy.actor_network.forward(o[agent_id], other_agents[index].policy.actor_network.hidden_state)["actions"])
                    q_agent = other_agents[index].policy.critic_network.forward(o_s, u_s[agent_id])
                    q.append(q_agent)

                    index += 1
        else:
            for agent_id in range(self.args.n_agents):
                q_agent = self.critic_network.forward(o_s, u_s[agent_id])
                q.append(q_agent)


        q = torch.stack(q, dim=-1)

        # q.squeeze()
        if self.mixer_network is not None and getattr(self.args, "mixer", "qmix") == "vdn":
            q = self.mixer_network.forward(q)
        elif self.mixer_network is None:
            pass
        else:
            q = self.mixer_network.forward(q, state)

        critic_loss = (target_q - q).pow(2).mean()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # the actor loss
        index_update = 0
        actions_taken = []
        q_val_of_actions_taken = []
        if not self.args.share_param:
            for agent_id in range(self.args.n_agents):
                if agent_id == self.agent_id:
                    actions_taken.append(
                        self.actor_network.forward(o[agent_id]))
                    q_agent = self.critic_network.forward(o_s, actions_taken[agent_id])
                    q_val_of_actions_taken.append(q_agent)
                else:

                    actions_taken.append(other_agents[index_update].policy.actor_network.forward(o[agent_id]))
                    q_agent = other_agents[index_update].policy.critic_network.forward(o_s, actions_taken[agent_id])
                    q_val_of_actions_taken.append(q_agent)

                    index_update += 1
        else:
            for agent_id in range(self.args.n_agents):

                actions_taken.append(
                    self.actor_network.forward(o[agent_id]))
                q_agent = self.critic_network.forward(o_s, actions_taken[agent_id])
                q_val_of_actions_taken.append(q_agent)


        q_val_of_actions_taken = torch.stack(q_val_of_actions_taken, dim=-1)
        q_val_of_actions_taken.squeeze()
        if self.mixer_network is not None and getattr(self.args, "mixer", "qmix") == "vdn":
            q_val_of_actions_taken = self.mixer_target_network.forward(q_val_of_actions_taken)
        elif self.mixer_network is None:
            pass
        else:
            q_val_of_actions_taken = self.mixer_network.forward(q_val_of_actions_taken, state)

        pi = torch.stack(actions_taken, dim=-1)
        actor_loss = -q_val_of_actions_taken.mean() + (pi ** 2).mean() * 1e-3

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()


        self._soft_update_target_network()
        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)
        self.train_step += 1

    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = model_path +'/share_param='+str(self.args.share_param)+ '/' + self.args.algorithm
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.actor_network.state_dict(), model_path + '/' + num + '_actor_params.pkl')
        torch.save(self.critic_network.state_dict(),  model_path + '/' + num + '_critic_params.pkl')

    def cuda(self,device="cuda:0"):
        self.actor_network.cuda(device=device)
        self.actor_target_network.cuda(device=device)
        self.critic_network.cuda(device=device)
        self.critic_target_network.cuda(device=device)
        if self.mixer_network is not None:
            self.mixer_network.cuda(device=device)
            self.mixer_target_network.cuda(device=device)


