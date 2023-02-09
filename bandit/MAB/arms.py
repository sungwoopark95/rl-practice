import numpy as np
from abc import ABC, abstractmethod


class Arm(ABC):
    @abstractmethod
    def draw(self): pass


class BernoulliArm(Arm):
    def __init__(self, p):
        self.mu = p
    
    def draw(self):
        if np.random.random() > self.mu:
            return 0.0
        else:
            return 1.0
        

class GaussianArm(Arm):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        
    def draw(self):
        return np.random.normal(loc=self.mu, scale=self.sigma)
