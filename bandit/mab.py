import numpy as np
from abc import ABC, abstractmethod
from cfg import get_cfg

cfg = get_cfg()    


class Bandit(ABC):
    @abstractmethod
    def initialize(self): pass
    
    @abstractmethod
    def choose(self): pass
    
    @abstractmethod
    def update(self, action, reward): pass


class eGreedyMAB(Bandit):
    def __init__(self, n_arms, epsilon, alpha=cfg.alpha, initial=cfg.initial):
        self.n_arms = n_arms
        self.alpha = alpha
        self.initial = initial      # set to 0 by default
        self.epsilon = epsilon
        
    def initialize(self):
        self.counts = np.zeros(self.n_arms)
        self.returns = np.zeros(self.n_arms) + self.initial
        self.epsilon_ = self.epsilon
    
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
        

class UCB(Bandit):
    def __init__(self, n_arms, conf=cfg.conf):
        self.n_arms = n_arms
        self.conf = conf
    
    def initialize(self):
        self.counts = np.zeros(self.n_arms)
        self.returns = np.array([np.inf for _ in range(self.n_arms)], dtype='float')
        self.rounds = 0
    
    def choose(self):
        argmaxes = np.where(self.returns == np.max(self.returns))[0]
        idx = np.random.choice(argmaxes)
        return idx
    
    def update(self, action, reward):
        self.counts[action] += 1
        
        if self.rounds == 0:
            value = 0.
        else:
            value = self.returns[action]
        
        step = self.counts.sum()
        ucb = self.conf * np.sqrt(np.log(step)/self.counts[action])
        new_value = ((value + reward) / self.counts[action]) + ucb
        self.returns[action] = new_value
        
        self.rounds += 1