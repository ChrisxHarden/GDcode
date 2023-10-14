from tqdm import tqdm
import time
from agent import Agent
from runner import Runner
from common.replay_buffer import Buffer
import torch
import os
import numpy as np
import matplotlib.pyplot as plt


class Lstm_Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.agents = self._init_agents()
        self.buffer = Buffer(args)
        self.temp_buffer = dict()
        self.eval_temp_buffer=dict()
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name+'/share_param='+str(self.args.share_param)
        self.result_path=self.save_path + '/'+self.args.algorithm+'/'+self.args.run_id

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents

    def run(self):
        returns = []
        buffer_count=0

        temp_buffer_limit=self.args.seq_length-1
        device = self.device


        for time_step in tqdm(range(self.args.time_steps+self.args.exploration_steps)):
            # reset the environment
            if time_step % self.episode_limit == 0:
                s = self.env.reset()
                s_init=[]
                for obs in s:
                    obs_tensor=torch.tensor(obs,dtype=torch.float32,device=self.device)
                    s_init.append(obs_tensor.unsqueeze(0))



                buffer_count=0
                for i in range(self.args.n_agents):
                    self.temp_buffer['o_%d' % i] = torch.zeros([self.args.seq_length, self.args.obs_shape[i]],dtype=torch.float32,device=device)
                    self.temp_buffer['u_%d' % i] = torch.zeros([self.args.seq_length, self.args.action_shape[i]],dtype=torch.float32,device=device)
                    self.temp_buffer['r_%d' % i] = torch.zeros([self.args.seq_length], dtype=torch.float32, device=device)
                    self.temp_buffer['o_next_%d' % i] = torch.zeros([self.args.seq_length, self.args.obs_shape[i]],dtype=torch.float32,device=device)
            u = []
            actions = []
            with torch.no_grad():
                if not self.args.share_agent:
                    for agent_id, agent in enumerate(self.agents):
                        if time_step % self.episode_limit == 0:
                            action = agent.select_action(s_init[agent_id], self.noise, self.epsilon)

                        else:

                            action = agent.select_action(self.temp_buffer['o_%d' % agent_id][:buffer_count+1], self.noise, self.epsilon)
                        u.append(action)
                        actions.append(action)
                else:
                    for agent_id, agent in enumerate(self.agents):
                        if time_step % self.episode_limit == 0:
                            action = self.agents[0].select_action(s_init[agent_id], self.noise, self.epsilon)
                        else:

                            action = self.agents[0].select_action(self.temp_buffer['o_%d' % agent_id][:buffer_count + 1],self.noise, self.epsilon)
                        u.append(action)
                        actions.append(action)
            for i in range(self.args.n_agents, self.args.n_players):
                actions.append(torch.tensor([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0],device=self.device))
                
            s_next, r, done, info = self.env.step(actions)
            

            self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents], s_next[:self.args.n_agents])
            self.temp_buffer_store(self.temp_buffer,s[:self.args.n_agents], u, r[:self.args.n_agents], s_next[:self.args.n_agents],idx=buffer_count,limit=temp_buffer_limit)
            s = s_next
            buffer_count = min(buffer_count + 1, temp_buffer_limit)



            if self.buffer.current_size >= self.args.exploration_steps:
                transitions = self.buffer.seq_sample(self.args.batch_size,self.args.seq_length)
                if not self.args.share_agent:
                    for agent in self.agents:
                        other_agents = self.agents.copy()
                        other_agents.remove(agent)
                        agent.learn(transitions, other_agents)
                else:
                    other_agents = self.agents.copy()
                    other_agents.remove(self.agents[0])
                    self.agents[0].learn(transitions, other_agents)

            if time_step > self.args.exploration_steps and time_step % self.args.evaluate_rate == 0:
                return_for_this_round=self.evaluate()
                print("Avg_return for this round is",return_for_this_round)
                returns.append(return_for_this_round)
                
                plt.figure()
                plt.plot(range(len(returns)), returns)
                plt.xlabel('episode * ' + str(self.args.evaluate_rate / self.episode_limit))
                plt.ylabel('average returns')
                plt.savefig(self.result_path+'/plt.png', format='png')
                self.noise = max(0.05, self.noise - 0.0000005)
                self.epsilon = max(0.05, self.epsilon - 0.0000005)
                np.save(self.result_path+'/returns.pkl', returns)

    def evaluate(self):
        returns = []
        buffer_count = 0
        temp_buffer_limit = self.args.seq_length - 1
        device = self.device
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s = self.env.reset()
            rewards = 0
            s_init = []
            for obs in s:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                s_init.append(obs_tensor.unsqueeze(0))
            for i in range(self.args.n_agents):
                self.eval_temp_buffer['o_%d' % i] = torch.zeros([self.args.seq_length, self.args.obs_shape[i]],
                                                           dtype=torch.float32, device=device)
                self.eval_temp_buffer['u_%d' % i] = torch.zeros([self.args.seq_length, self.args.action_shape[i]],
                                                           dtype=torch.float32, device=device)
                self.eval_temp_buffer['r_%d' % i] = torch.zeros([self.args.seq_length], dtype=torch.float32, device=device)
                self.eval_temp_buffer['o_next_%d' % i] = torch.zeros([self.args.seq_length, self.args.obs_shape[i]],
                                                                dtype=torch.float32, device=device)

            for time_step in range(self.args.evaluate_episode_len):
                self.env.render()
                actions = []
                u=[]
                with torch.no_grad():
                    if not self.args.share_agent:
                        for agent_id, agent in enumerate(self.agents):
                            if time_step==0:
                                action=agent.select_action(s_init[agent_id],0,0)
                            else:
                                action = agent.select_action(self.eval_temp_buffer['o_%d' % agent_id][:buffer_count+1], 0, 0)
                            actions.append(action)
                            u.append(action)
                    else:
                        for agent_id, agent in enumerate(self.agents):
                            if time_step==0:
                                action=self.agents[0].select_action(s_init[agent_id],0,0)
                            else:
                                action = self.agents[0].select_action(self.eval_temp_buffer['o_%d' % agent_id][:buffer_count+1], 0, 0)
                            actions.append(action)
                            u.append(action)


                for i in range(self.args.n_agents, self.args.n_players):
                    actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                s_next, r, done, info = self.env.step(actions)
                rewards += r[0]
                self.temp_buffer_store(self.eval_temp_buffer, s[:self.args.n_agents], u, r[:self.args.n_agents],
                                       s_next[:self.args.n_agents], idx=buffer_count, limit=temp_buffer_limit)

                s = s_next
                buffer_count = min(buffer_count + 1, temp_buffer_limit)
            returns.append(rewards)
        return sum(returns) / self.args.evaluate_episodes

    def temp_buffer_store(self, temp_buffer,o, u, r, o_next,idx,limit):


        for i in range(self.args.n_agents):
            if idx < limit:
                temp_buffer['o_%d' % i][idx] = torch.tensor(o[i], device=self.device)
                temp_buffer['u_%d' % i][idx] = torch.tensor(u[i], device=self.device)
                temp_buffer['r_%d' % i][idx] = torch.tensor(r[i], device=self.device)
                temp_buffer['o_next_%d' % i][idx] = torch.tensor(o_next[i], device=self.device)
            else:
                temp_buffer['o_%d' % i][:idx].data=self.temp_buffer['o_%d' % i][1:].data
                temp_buffer['o_%d' % i][idx] = torch.tensor(o[i], device=self.device)
                temp_buffer['u_%d' % i][:idx].data = self.temp_buffer['u_%d' % i][1:].data
                temp_buffer['u_%d' % i][idx] = torch.tensor(u[i], device=self.device)
                temp_buffer['r_%d' % i][:idx].data= self.temp_buffer['r_%d' % i][1:].data
                temp_buffer['r_%d' % i][idx] = torch.tensor(r[i], device=self.device)
                temp_buffer['o_next_%d' % i][:idx].data= self.temp_buffer['o_next_%d' % i][1:].data
                temp_buffer['o_next_%d' % i][idx] = torch.tensor(o_next[i], device=self.device)




