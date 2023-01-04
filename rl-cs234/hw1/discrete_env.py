import numpy as np

from gym import Env, spaces
from gym.utils import seeding

def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n) # convert a given object into a numpy array
    csprob_n = np.cumsum(prob_n) 
    # for ith element, get \sum_{j=1}^{i} {x_j}
    # equivalent to for item in x[:i]: sum += item
    # so the final element of csprob_n is always equal to 1
    
    # return index of the smallest value greater than a random number
    return (csprob_n > np_random.rand()).argmax() 


class DiscreteEnv(Env):
    """
    Has the following members
    - nS: number of states -> int
    - nA: number of actions -> int
    - P: transitions (*)
    - isd: initial state distribution (**)

    (*) dictionary dict of dicts of lists, where
      P[s][a] == [(probability, nextstate, reward, done), ...]
    (**) list or array of length nS
    """
    
    def __init__(self, nS, nA, P, isd):
        self.P = P
        self.isd = isd
        self.lastaction = None # for rendering
        self.nS = nS
        self.nA = nA

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.s = categorical_sample(self.isd, self.np_random) # sample an initial state
        self.lastaction = None
        return self.s

    def _step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        # transition: [(probability, nextstate, reward, done), ...]
        # t: (probability, nextstate, reward, done)
        # t[0]: probability
        # [t[0] for t in transitions]: probability distribution of each next state, given a (self.s, a) pair
        p, s, r, d = transitions[i] # s: sampled next state, r: sampled reward
        self.s = s # update the current state with the next state
        self.lastaction = a # update the current action
        return (s, r, d, {"prob" : p})
