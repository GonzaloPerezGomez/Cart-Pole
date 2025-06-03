from collections import deque
import random
import numpy as np

class ReplayBuffer:
    
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)
        self.maxlen = self.buffer.maxlen

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)