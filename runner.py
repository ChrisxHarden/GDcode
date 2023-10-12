from tqdm import tqdm
import time
from agent import Agent
from common.replay_buffer import Buffer
import torch
import os
import numpy as np
import matplotlib.pyplot as plt


class Runner:
    # Runner is used to execute the training process for no-temporal message fusion algorithms.
    # Potentially, we can merge Runner with the LSTM Runner
    
    
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.agents = self._init_agents()
        self.buffer = Buffer(args)
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
        for time_step in tqdm(range(self.args.time_steps+self.args.exploration_steps)):
            # reset the environment
            if time_step % self.episode_limit == 0:
                s = self.env.reset()


            u = []
            actions = []
            with torch.no_grad():
                if not self.args.share_agent:
                    #This is to let the agents' actor network not sharing the parameters, can be used for ablation study
                    
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], self.noise, self.epsilon)
                        u.append(action)
                        actions.append(action)
                
                
                else:
                    for agent_id in range(self.args.n_agents):
                        action=self.agents[0].select_action(s[agent_id],self.noise,self.epsilon)
                        u.append(action)
                        actions.append(action)
            
            for i in range(self.args.n_agents, self.args.n_players):
                actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])# for the adversial 
            
            
            s_next, r, done, info = self.env.step(actions)
            
            self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents], s_next[:self.args.n_agents])
            s = s_next
            
            if self.buffer.current_size >= self.args.exploration_steps:
                # start_time=time.time()
                transitions = self.buffer.sample(self.args.batch_size)
                # end_time=time.time()
                # buffer_sample_time=end_time-start_time
                
                # with open("time_cosumption.txt","+a") as file:
                #     file.write("buffer sample time per step\n")
                #     file.write(str(buffer_sample_time))
                #     file.write("\n")

                
                start_time=time.time()
                if not self.args.share_agent:
                    
                    for agent in self.agents:
                        other_agents = self.agents.copy()
                        other_agents.remove(agent)
                        
                        agent.learn(transitions, other_agents)
                else:
                    other_agents = self.agents.copy()
                    other_agents.remove(self.agents[0])
                    self.agents[0].learn(transitions, other_agents)
                # end_time=time.time()
                
                # execution_time=end_time-start_time
                # with open("time_cosumption.txt","+a") as file:
                #     file.write("training time per step\n")
                #     file.write(str(execution_time))
                #     file.write("\n")

            if time_step > 0 and time_step % self.args.evaluate_rate == 0:
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
                np.save(self.result_path+ '/returns.pkl', returns)

    def evaluate(self):
        returns = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s = self.env.reset()
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                self.env.render()
                actions = []
                with torch.no_grad():
                    if not self.args.share_agent:
                        for agent_id, agent in enumerate(self.agents):
                            action = agent.select_action(s[agent_id], 0, 0)
                            actions.append(action)
                    else:
                        for agent_id, agent in enumerate(self.agents):
                            action = self.agents[0].select_action(s[agent_id], 0, 0)
                            actions.append(action)
                for i in range(self.args.n_agents, self.args.n_players):
                    actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                s_next, r, done, info = self.env.step(actions)
                rewards += r[0]
                s = s_next
            returns.append(rewards)
            # print('Returns is', rewards)
        return sum(returns) / self.args.evaluate_episodes
