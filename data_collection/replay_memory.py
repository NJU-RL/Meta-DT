import random
import numpy as np
from collections import OrderedDict
import pickle 

class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        np.random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done, mask):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done, mask)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, mask = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done, mask

    def __len__(self):
        return len(self.buffer)
    
    def save_buffer(self, save_path):
        states, actions, rewards, next_states, terminals, masks = map(np.stack, zip(*self.buffer))
        data = OrderedDict()
        data['observations'] = states
        data['actions'] = actions 
        data['rewards'] = rewards
        data['next_observations'] = next_states
        data['terminals'] = terminals
        data['masks'] = masks 

        print(f'Buffer length {len(self)}, save buffer to {save_path}')
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        f.close()
    









