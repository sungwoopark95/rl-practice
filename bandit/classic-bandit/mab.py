import numpy as np
import random
from cfg import get_cfg


cfg = get_cfg()
      

class eGreedyMAB:
    def __init__(self, n_arms, eps_fixed):
        self.n_arms = n_arms
        self.eps_fixed = eps_fixed
        
    def initialize(self, epsilon):
        self.counts = np.zeros(self.n_arms)
        self.returns = np.zeros(self.n_arms)
        self.epsilon = epsilon
        self.rounds = 0
    
    def choose(self):
        if random.random() > self.epsilon:
            max_return = np.max(self.returns)
            argmaxes = np.where(self.returns == max_return)[0]
            idx = np.random.choice(argmaxes)
        else:
            idx = np.random.randint(self.n_arms)
        return idx
    
    def update(self, action, reward):
        # action: index of the chosen arm
        # reward: reward of the chosen arm
        self.counts[action] += 1
        value = self.returns[action]
        value = (value + reward) / self.counts[action]
        self.returns[action] = value
        
        self.rounds += 1
        if not self.eps_fixed:
            self.epsilon /= 0.8