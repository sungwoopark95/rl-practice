import numpy as np
import csv
import os


## Disjoint LinUCB
class LinUCB:
    def __init__(self, n_arms, features, alpha=1.):
        self.alpha = alpha
        self.n_arms = n_arms
        self.features = features
        self.A = [np.identity(len(features)) for _ in range(self.n_arms)]
        self.b = [np.zeros(shape=(len(features), 1)) for _ in range(self.n_arms)]
        
    def choose(self, x, arms):
        ps = np.zeros(shape=self.n_arms)
        feat_vec = np.array([x[feat] for feat in self.features]).reshape(-1, 1)
        for i in range(self.n_arms):
            theta_hat = np.linalg.inv(self.A[i]) @ self.b[i]
            p = (theta_hat.T@feat_vec) + (self.alpha*np.sqrt(feat_vec.T@np.linalg.inv(self.A[i])@feat_vec))
            ps[i] = p
        return arms[np.argmax(ps)]

    def update(self, x, arms, a, r):
        to_update = arms.index(a)
        feat_vec = np.array([x[feat] for feat in self.features]).reshape(-1, 1)
        self.A[to_update] += feat_vec @ feat_vec.T
        self.b[to_update] += r * feat_vec


## Hybrid LinUCB
