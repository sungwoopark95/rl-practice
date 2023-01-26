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
        if np.random.random() > self.epsilon_:
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
        n = self.counts[action]
        new_value = (((n-1)/n)*value) + ((1/n)*reward)
        self.returns[action] = new_value
        
        self.epsilon_ *= self.alpha
        

class UCB(Bandit):
    def __init__(self, n_arms, conf=cfg.conf):
        self.n_arms = n_arms
        self.conf = conf
    
    def initialize(self):
        self.counts = np.zeros(self.n_arms)
        self.qs = np.zeros(self.n_arms)
        self.ucbs = np.zeros(self.n_arms)
        self.step = 0
    
    def choose(self):
        self.step += 1
        for i, cnt in enumerate(self.counts):
            if cnt == 0:
                self.ucbs[i] = np.iinfo(np.int32).max
            else:
                self.ucbs[i] = np.sqrt(np.log(self.step) / cnt)
        returns = self.qs + (self.conf*self.ucbs)
        argmaxes = np.where(returns == np.max(returns))[0]
        return np.random.choice(argmaxes)
    
    def update(self, action, reward):
        value = self.qs[action]
        self.counts[action] += 1
        n = self.counts[action]
        new_value = (((n-1)/n)*value) + ((1/n)*reward)
        self.qs[action] = new_value
