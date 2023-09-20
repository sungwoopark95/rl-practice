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
        self.initial = initial  # set to 0 by default
        self.epsilon = epsilon
        
    def initialize(self):
        self.counts = np.zeros(self.n_arms)
        self.qs = np.zeros(self.n_arms) + self.initial
        self.epsilon_ = self.epsilon
    
    def choose(self):
        p = np.random.uniform(low=0., high=1.)
        if p > self.epsilon_:
            argmaxes = np.where(self.qs == np.max(self.qs))[0]
            idx = np.random.choice(argmaxes)
        else:
            idx = np.random.choice(self.n_arms)
        return idx
    
    def update(self, action, reward):
        """
        action: index of the chosen arm
        reward: reward of the chosen arm
        """
        ## count update
        self.counts[action] += 1
        
        ## q update
        value = self.qs[action]
        n = self.counts[action]
        new_value = (((n-1)/n)*value) + ((1/n)*reward)
        self.qs[action] = new_value
        
        ## epsilon update
        self.epsilon_ *= self.alpha
        

class ETC(Bandit):
    ## Explore-then-commit Bandit
    def __init__(self, n_arms, explore, horizon=cfg.nsteps, initial=cfg.initial):
        assert explore * n_arms <= horizon
        self.n_arms = n_arms
        self.initial = initial  # set to 0 by default
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
        """
        action: index of the chosen arm
        reward: reward of the chosen arm
        """
        ## count update
        self.counts[action] += 1
        
        ## q update
        value = self.qs[action]
        n = self.counts[action]
        new_value = (((n-1)/n)*value) + ((1/n)*reward)
        self.qs[action] = new_value


class UCBNaive(Bandit):
    def __init__(self, n_arms:int, sigma:float, alpha:float, delta:float=0.1):
        self.n_arms = n_arms
        self.alpha = alpha
        self.delta = delta
        self.sigma = sigma
    
    def initialize(self):
        self.counts = np.zeros(self.n_arms)
        self.qs = np.zeros(self.n_arms)
        self.ucbs = np.array([np.iinfo(np.int32).max for _ in range(self.n_arms)])
        self.step = 0
    
    def choose(self):
        self.step += 1
        returns = self.qs + self.ucbs
        argmaxes = np.where(returns == np.max(returns))[0]
        return np.random.choice(argmaxes)
    
    def update(self, action, reward):
        """
        action: index of the chosen arm
        reward: reward of the chosen arm
        """
        ## count update
        self.counts[action] += 1
        
        ## q update
        value = self.qs[action]
        n = self.counts[action]
        new_value = (((n-1)/n)*value) + ((1/n)*reward)
        self.qs[action] = new_value
        
        ## ucb update
        inside = 2 * (self.sigma ** 2) * np.log(self.step/self.delta)
        self.ucbs[action] = self.alpha * np.sqrt(inside)


class UCBDelta(UCBNaive):
    def __init__(self, n_arms, delta):
        self.n_arms = n_arms
        self.delta = delta
    
    def update(self, action, reward):
        """
        action: index of the chosen arm
        reward: reward of the chosen arm
        """
        ## count update
        self.counts[action] += 1
        
        ## q update
        value = self.qs[action]
        n = self.counts[action]
        new_value = (((n-1)/n)*value) + ((1/n)*reward)
        self.qs[action] = new_value
        
        ## ucb update
        numerator = 2 * np.log(1/self.delta)
        self.ucbs[action] = np.sqrt(numerator / self.counts[action])
        
        
class UCBAsymptotic(UCBNaive):
    def __init__(self, n_arms):
        self.n_arms = n_arms
    
    def update(self, action, reward):
        """
        action: index of the chosen arm
        reward: reward of the chosen arm
        """
        ## count update
        self.counts[action] += 1
        
        ## q update
        value = self.qs[action]
        n = self.counts[action]
        new_value = (((n-1)/n)*value) + ((1/n)*reward)
        self.qs[action] = new_value
        
        ## ucb update
        ft = 1 + (self.step * (np.log(self.step)**2))
        numerator = 2 * np.log(ft)
        self.ucbs[action] = np.sqrt(numerator / self.counts[action])
        

class UCBMOSS(UCBNaive):
    def __init__(self, n_arms, nsim=cfg.nsim):
        self.n_arms = n_arms
        self.nsim = nsim
        
    def update(self, action, reward):
        """
        action: index of the chosen arm
        reward: reward of the chosen arm
        """
        ## count update
        self.counts[action] += 1
        
        ## q update
        value = self.qs[action]
        n = self.counts[action]
        new_value = (((n-1)/n)*value) + ((1/n)*reward)
        self.qs[action] = new_value
        
        ## ucb update
        left = 4 / n
        right = np.log(np.maximum(1, (self.nsim / (self.n_arms*n))))
        self.ucbs[action] = np.sqrt(left * right)


class ThompsonSampling(Bandit):
    def __init__(self, n_arms, bernoulli=cfg.bernoulli):
        self.n_arms = n_arms
        self.bernoulli = bernoulli
    
    def initialize(self):
        self.counts = np.zeros(shape=self.n_arms)
        self.qs = np.zeros(shape=self.n_arms)
        if self.bernoulli:
            self.alphas = np.ones(shape=self.n_arms)
            self.betas = np.ones(shape=self.n_arms)
        else:
            self.mus = np.zeros(shape=self.n_arms)
            self.devs = np.ones(shape=self.n_arms)
    
    def choose(self):
        if self.bernoulli:
            thetas = np.array([np.random.beta(a=alpha, b=beta) for (alpha, beta) in zip(self.alphas, self.betas)])
        else:
            thetas = np.array([np.random.normal(loc=mu, scale=var) for (mu, var) in zip(self.mus, self.devs)])
        argmaxes = np.where(thetas == np.max(thetas))[0]
        return np.random.choice(argmaxes)
    
    def update(self, action, reward):
        """
        action: index of the chosen arm
        reward: reward of the chosen arm
        """
        ## count update
        self.counts[action] += 1
        
        ## q update
        value = self.qs[action]
        n = self.counts[action]
        new_value = (((n-1)/n)*value) + ((1/n)*reward)
        self.qs[action] = new_value
        
        ## parameter update
        if self.bernoulli:
            self.alphas[action] += reward
            self.betas[action] += (1-reward)
        else:
            self.mus[action] = new_value
            self.devs[action] = np.sqrt(1/n)
