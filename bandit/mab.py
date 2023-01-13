import numpy as np
from cfg import get_cfg

cfg = get_cfg()     

class eGreedyMAB:
    def __init__(self, n_arms, alpha=cfg.alpha, initial=cfg.initial):
        self.n_arms = n_arms
        self.alpha = alpha
        self.initial = initial          # set to None by default
        
    def initialize(self, epsilon):
        self.counts = np.zeros(self.n_arms)
        self.returns = np.zeros(self.n_arms) + self.initial
        if self.initial > 0:
            print(f"Optimistic bandit")
        else:
            print(f"Naive bandit")
        self.epsilon = epsilon
    
    def choose(self):
        if np.random.random() > self.epsilon:
            argmaxes = np.where(self.returns == np.max(self.returns))[0]
            idx = np.random.choice(argmaxes)
        else:
            idx = np.random.choice(self.n_arms)
        return idx
    
    def update(self, action, reward):
        # action: index of the chosen arm
        # reward: reward of the chosen arm
        self.counts[action] += 1
        
        value = self.returns[action]
        new_value = (value + reward) / self.counts[action]
        self.returns[action] = new_value
        
        self.epsilon *= self.alpha