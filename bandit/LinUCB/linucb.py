import numpy as np
from abc import ABC, abstractmethod
from cfg import get_cfg

cfg = get_cfg()    


class Bandit(ABC):
    @abstractmethod
    def initialize(self): pass
    
    @abstractmethod
    def choose(self, x): pass
    
    @abstractmethod
    def update(self, x, action, reward): pass


## Disjoint LinUCB
class LinUCB(Bandit):
    def __init__(self, arms, d, alpha):
        ## arms: K x l matrix, K - the number of actions, l - features of each action
        ## d: number of features - l (arm feature) + (d-l) (user feature)
        ## alpha: hyper-parameter that determines degree of exploration
        self.arms = arms
        self.n_arms = arms.shape[0]
        self.d = d  
        self.alpha = alpha
    
    def initialize(self):
        self.As = [np.identity(self.d) for _ in range(self.n_arms)]         # A = d x d
        self.bs = [np.zeros(shape=(self.d, 1)) for _ in range(self.n_arms)] # B = (d, 1) array
        self.ps = np.zeros(shape=self.n_arms)
        
    def choose(self, x):
        ## x: (d-l) shaped 1d-array - user feature
        ## return: index of arm which yields the highest payoff
        for i in range(self.n_arms):
            arm_feat = self.arms[i]
            A_a, b_a = self.As[i], self.bs[i]
            x_ta = np.append(arm_feat, x).reshape((-1, 1))
            theta_a = np.linalg.inv(A_a) @ b_a
            p = (theta_a.T @ x_ta) + (self.alpha * np.sqrt(x_ta.T@np.linalg.inv(A_a)@x_ta))
            self.ps[i] = p
        return np.argmax(self.ps)

    def update(self, x, a, r):
        ## x: (d-l) shaped 1d-array - user feature
        ## a: index of the chosen arm - return value of choose function
        ## r: reward yielded from the chosen arm
        chosen_arm_feat = self.arms[a]
        x_chosen = np.append(chosen_arm_feat, x).reshape((-1, 1))   # (d, 1) shaped array
        
        oldA, oldb = self.As[a], self.bs[a]
        newA = oldA + np.outer(x_chosen, x_chosen)
        newb = oldb + (r * x_chosen)
        
        self.As[a], self.bs[a] = newA, newb


# ## Hybrid LinUCB
# class HybridLinUCB:
#     def __init__(self, arms, user_features, alpha=1.):
    
    
#     def initialize(self):
        
        
#     def choose(self, x):
            
    
#     def update(self, x, a, r):
        