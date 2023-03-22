import threading
import torch
import numpy as np


class Buffer:
    def __init__(self, args):
        self.size = args.buffer_size
        self.args = args
        # memory management
        self.current_size = 0
        # create the buffer to store info
        self.buffer = dict()
        # for i in range(self.args.n_agents):
        #     self.buffer['o_%d' % i] = np.empty([self.size, self.args.obs_shape[i]])
        #     self.buffer['u_%d' % i] = np.empty([self.size, self.args.action_shape[i]])
        #     self.buffer['r_%d' % i] = np.empty([self.size])
        #     self.buffer['o_next_%d' % i] = np.empty([self.size, self.args.obs_shape[i]])
        self.device=torch.device('cuda' if torch.cuda.is_available else 'cpu')
        print(self.device)
        for i in range(self.args.n_agents):
            self.buffer['o_%d' % i] = torch.zeros([self.size, self.args.obs_shape[i]],dtype=torch.float32,device=self.device)
            self.buffer['u_%d' % i] = torch.zeros([self.size, self.args.action_shape[i]],dtype=torch.float32,device=self.device)
            self.buffer['r_%d' % i] = torch.zeros([self.size],dtype=torch.float32,device=self.device)
            self.buffer['o_next_%d' % i] = torch.zeros([self.size, self.args.obs_shape[i]],dtype=torch.float32,device=self.device)

        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, o, u, r, o_next):
        idxs = self._get_storage_idx(inc=1)  # 以transition的形式存，每次只存一条经验
        # for i in range(self.args.n_agents):
        #     with self.lock:
        #         self.buffer['o_%d' % i][idxs] = o[i].cpu() if type(o[i])==torch.Tensor else o[i]
        #         self.buffer['u_%d' % i][idxs] = u[i].cpu() if type(u[i])==torch.Tensor else u[i]
        #         self.buffer['r_%d' % i][idxs] = r[i].cpu() if type(r[i])==torch.Tensor else r[i]
        #         self.buffer['o_next_%d' % i][idxs] = o_next[i].cpu() if type(o_next[i])==torch.Tensor else o_next[i]



        for i in range(self.args.n_agents):
            with self.lock:
                self.buffer['o_%d' % i][idxs] = torch.tensor(o[i],device=self.device)
                self.buffer['u_%d' % i][idxs] = torch.tensor(u[i],device=self.device)
                self.buffer['r_%d' % i][idxs] = torch.tensor(r[i],device=self.device)
                self.buffer['o_next_%d' % i][idxs] = torch.tensor(o_next[i],device=self.device)



    
    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffer.keys():
            temp_buffer[key] = self.buffer[key][idx]
        return temp_buffer

    def seq_sample(self, batch_size,seq_length):
        temp_buffer = {}
        idxs = np.random.randint(seq_length, self.current_size, batch_size)
        for key in self.buffer.keys():
            temp_buffer[key]=[]
            for idx in idxs:
                temp_buffer[key].append (self.buffer[key][idx-seq_length:idx])
            temp_buffer[key]=torch.stack(temp_buffer[key])


        return temp_buffer



    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx
