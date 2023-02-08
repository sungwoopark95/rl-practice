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
        self.qs = np.zeros(self.n_arms) + self.initial
        self.epsilon_ = self.epsilon
    
    def choose(self):
        if np.random.random() > self.epsilon_:
            argmaxes = np.where(self.qs == np.max(self.qs))[0]
            idx = np.random.choice(argmaxes)
        else:
            idx = np.random.choice(self.n_arms)
        return idx
    
    def update(self, action, reward):
        # action: index of the chosen arm
        # reward: reward of the chosen arm
        self.counts[action] += 1
        
        value = self.qs[action]
        n = self.counts[action]
        new_value = (((n-1)/n)*value) + ((1/n)*reward)
        self.qs[action] = new_value
        
        self.epsilon_ *= self.alpha
        

class ETC(Bandit):
    ## Explore-then-commit Bandit
    def __init__(self, n_arms, explore, horizon=cfg.nsteps, initial=cfg.initial):
        assert explore * n_arms < horizon
        self.n_arms = n_arms
        self.initial = initial                     # set to 0 by default
        self.explore = explore
        
    def initialize(self):
        self.counts = np.zeros(self.n_arms)
        self.qs = np.zeros(self.n_arms) + self.initial
        self.step = 0
    
    def choose(self):
        ## explore step
        if self.step < self.explore * self.n_arms:
            idx = self.step % self.n_arms
        ## exploitation step
        else:
            argmaxes = np.where(self.qs == np.max(self.qs))[0]
            idx = np.random.choice(argmaxes)
        self.step += 1
        return idx
    
    def update(self, action, reward):
        # action: index of the chosen arm
        # reward: reward of the chosen arm
        self.counts[action] += 1
        value = self.qs[action]
        n = self.counts[action]
        new_value = (((n-1)/n)*value) + ((1/n)*reward)
        self.qs[action] = new_value


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
        log_step = np.log(self.step)
        for i, cnt in enumerate(self.counts):
            if cnt == 0:
                self.ucbs[i] = np.iinfo(np.int32).max
            else:
                self.ucbs[i] = self.conf * np.sqrt(log_step / cnt)
        returns = self.qs + self.ucbs
        argmaxes = np.where(returns == np.max(returns))[0]
        return np.random.choice(argmaxes)
    
    def update(self, action, reward):
        value = self.qs[action]
        self.counts[action] += 1
        n = self.counts[action]
        new_value = (((n-1)/n)*value) + ((1/n)*reward)
        self.qs[action] = new_value


class ThompsonSampling(Bandit):
    def __init__(self, n_arms, bernoulli=cfg.bernoulli):
        self.n_arms = n_arms
        if bernoulli:
            self.bernoulli = True
        else:
            self.bernoulli = False
    
    def initialize(self):
        self.counts = np.zeros(shape=self.n_arms)
        self.qs = np.zeros(shape=self.n_arms)
        if self.bernoulli:
            self.alphas = np.ones(shape=self.n_arms)
            self.betas = np.ones(shape=self.n_arms)
        else:
            self.mus = np.zeros(shape=self.n_arms)
            self.vars = np.ones(shape=self.n_arms)
    
    def choose(self):
        if self.bernoulli:
            thetas = np.array([np.random.beta(a=alpha, b=beta) for (alpha, beta) in zip(self.alphas, self.betas)])
        else:
            thetas = np.array([np.random.normal(loc=mu, scale=var) for (mu, var) in zip(self.mus, self.vars)])
        argmaxes = np.where(thetas == np.max(thetas))[0]
        return np.random.choice(argmaxes)
    
    def update(self, action, reward):
        ## reward update
        value = self.qs[action]
        self.counts[action] += 1
        n = self.counts[action]
        new_value = (((n-1)/n)*value) + ((1/n)*reward)
        self.qs[action] = new_value
        
        ## parameter update
        if self.bernoulli:
            self.alphas[action] += reward
            self.betas[action] += (1-reward)
        else:
            self.mus[action] = new_value
            self.vars[action] = (1/n)
