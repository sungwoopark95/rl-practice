import numpy as np
from abc import ABC, abstractmethod

class ContextualBandit(ABC):  
    @abstractmethod
    def choose(self, x): pass
    
    @abstractmethod
    def update(self, x, a, r): pass


## Disjoint LinUCB
class LinUCB(ContextualBandit):
    def __init__(self, arms, d, alpha):
        """
        arms: K x l matrix, K - the number of actions, l - features of each action
        d: number of features - l (arm feature) + (d-l) (user feature)
        alpha: hyper-parameter that determines degree of exploration
        """
        self.arms = arms
        self.n_arms = arms.shape[0]
        self.d = d  
        self.alpha = alpha
        self.As = [np.identity(d) for _ in range(self.n_arms)]         # A = d x d
        self.bs = [np.zeros(shape=(d, 1)) for _ in range(self.n_arms)] # B = (d, 1) array
        self.ps = np.zeros(shape=self.n_arms)
           
    def choose(self, x):
        """
        x: (d-l) shaped 1d-array - user feature
        return: index of arm which yields the highest payoff
        """
        for i in range(self.n_arms):
            arm_feat = self.arms[i]
            A_a, b_a = self.As[i], self.bs[i]
            A_a_inv = np.linalg.inv(A_a)
            x_ta = np.append(arm_feat, x).reshape((-1, 1))
            theta_a = A_a_inv @ b_a
            p = (theta_a.T @ x_ta) + (self.alpha * np.sqrt(x_ta.T@A_a_inv@x_ta))
            self.ps[i] = p.item()

        max_p = np.max(self.ps)
        tie = np.where(self.ps == max_p)[0]
        return np.random.choice(tie)

    def update(self, x, a, r):
        """
        x: (d-l) shaped 1d-array - user feature
        a: index of the chosen arm - return value of choose function
        r: reward yielded from the chosen arm
        """
        chosen_arm_feat = self.arms[a]
        x_chosen = np.append(chosen_arm_feat, x).reshape((-1, 1))   # (d, 1) shaped array
        oldA, oldb = self.As[a], self.bs[a]
        newA = oldA + np.outer(x_chosen, x_chosen)
        newb = oldb + (r * x_chosen)
        self.As[a], self.bs[a] = newA, newb


## eGreedy LinUCB
class eLinUCB(LinUCB):
    def __init__(self, arms, d, alpha, epsilon):
        super().__init__(arms, d, alpha)
        self.epsilon = epsilon
           
    def choose(self, x):
        p = np.random.uniform(low=0., high=1.)
        if p < self.epsilon:
            return np.random.choice(np.arange(self.n_arms))
        else:
            return super().choose(x)

    def update(self, x, a, r):
        super().update(x, a, r)


## Hybrid LinUCB
class HybridLinUCB(ContextualBandit):
    def __init__(self, arms, d, k, alpha=1.):
        """
        arms: K x l matrix, K - the number of actions, l - features of each action
        d: number of features - l (arm feature) + (d-l) (user feature)
        alpha: hyper-parameter that determines degree of exploration
        k: dimension of the shared parameter -> l (arm feature) * (d-l) (user feature)
        """
        self.arms = arms
        self.n_arms = arms.shape[0]
        self.d, self.k = d, k
        self.alpha = alpha
        self.A_node = np.identity(k)                                   # A_node = k x k
        self.b_node = np.zeros(shape=(k, 1))                           # B_node = k x 1
        self.As = [np.identity(d) for _ in range(self.n_arms)]         # A = d x d
        self.Bs = [np.zeros(shape=(d, k)) for _ in range(self.n_arms)] # B = d x k
        self.bs = [np.zeros(shape=(d, 1)) for _ in range(self.n_arms)] # b = d x 1 array
        self.ps = np.zeros(shape=self.n_arms)
        
    def choose(self, x):
        """
        x: (d-l) shaped 1d-array - user feature
        return: index of arm which yields the highest payoff
        """
        A_node_inv = np.linalg.inv(self.A_node)
        beta_hat = A_node_inv @ self.b_node                     # beta_hat = k x 1 array
        for i in range(self.n_arms):
            arm_feat = self.arms[i]
            x_ta = np.append(arm_feat, x).reshape((-1, 1))      # x_ta = d x 1
            z_ta = np.outer(arm_feat, x).reshape((-1, 1))       # z_ta = k x 1
            
            A_a, B_a, b_a = self.As[i], self.Bs[i], self.bs[i]
            A_a_inv = np.linalg.inv(A_a)
            B_a_T, z_ta_T, x_ta_T = B_a.T, z_ta.T, x_ta.T
            theta_a = A_a_inv @ (b_a-(B_a@beta_hat))            # theta_a = d x 1
            
            s = (                                               # s = scalar
                (z_ta_T @ A_node_inv @ z_ta)
                - (2*(z_ta_T @ A_node_inv @ B_a_T @ A_a_inv @ x_ta))
                + (x_ta_T @ A_a_inv @ x_ta)
                + (x_ta_T @ A_a_inv @ B_a @ A_node_inv @ B_a_T @ A_a_inv @ x_ta)
            ).item()
            
            p = (
                (z_ta_T @ beta_hat).item() 
                + (x_ta_T @ theta_a).item()
                + (self.alpha * np.sqrt(s))
            )
            self.ps[i] = p
            
        max_p = np.max(self.ps)
        tie = np.where(self.ps == max_p)[0]
        return np.random.choice(tie)
    
    def update(self, x, a, r):
        ## update shared parameter - phase 1
        chosenA, chosenB, chosenb = self.As[a], self.Bs[a], self.bs[a]
        chosenA_inv = np.linalg.inv(chosenA)
        chosenB_T = chosenB.T
        self.A_node += (chosenB_T @ chosenA_inv @ chosenB)
        self.b_node += (chosenB_T @ chosenA_inv @ chosenb)
        
        ## reward update
        chosen_arm_feat = self.arms[a]
        x_chosen = np.append(chosen_arm_feat, x).reshape((-1, 1))   # (d, 1) shaped array
        z_chosen = np.outer(chosen_arm_feat, x).reshape((-1, 1))    # (k, 1) shaped array
        chosenA += np.outer(x_chosen, x_chosen)
        chosenB += (x_chosen @ z_chosen.T)
        chosenb += (r * x_chosen)
        
        ## update shared parameter - phase 2
        chosenA_inv = np.linalg.inv(chosenA)    # update inverse
        chosenB_T = chosenB.T                   # update transpose
        self.A_node += np.outer(z_chosen, z_chosen) - (chosenB_T @ chosenA_inv @ chosenB)
        self.b_node += (r * z_chosen) - (chosenB_T @ chosenA_inv @ chosenb)

        ## assign updated params
        self.As[a] = chosenA
        self.Bs[a] = chosenB
        self.bs[a] = chosenb
