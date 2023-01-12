import random
import numpy as np
import pandas as pd
from cfg import get_cfg
from mab import eGreedyMAB
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle


class BernoulliArm:
    def __init__(self, p):
        self.p = p
    
    def draw(self):
        if np.random.random() > self.p:
            return 0.0
        else:
            return 1.0
        

class GaussianArm:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        
    def draw(self):
        return np.random.normal(loc=self.mu, scale=self.sigma)


def run(nsim, nsteps, learner, epsilon, arms, optimal_arm):
    epsilons = []
    epochs = []
    runs = []
    chosen_arms = []
    optimal_arms = []
    rewards = []
    
    if cfg.tqdm:
        bar = tqdm(range(nsim))
    else:
        bar = range(nsim)
    
    for sim in bar:
        learner.initialize(epsilon)
        for step in range(nsteps):
            chosen_arm = learner.choose()
            reward = arms[chosen_arm].draw()
            
            epsilons.append(epsilon)
            epochs.append(sim)
            runs.append(step)
            chosen_arms.append(chosen_arm)
            optimal_arms.append(optimal_arm)
            rewards.append(reward)
            
            learner.update(chosen_arm, reward)
    
    result = pd.DataFrame({
        'epsilon': epsilons,
        'sim': epochs,
        'step': runs,
        'chosen_arm': chosen_arms,
        'optimal_arm': optimal_arms,
        'reward': rewards
    })
    return result.reset_index(drop=True)


if __name__ == "__main__":
    cfg = get_cfg()
       
    if cfg.bernoulli:
        ps = np.random.uniform(low=0.0, high=1.0, size=cfg.n_arms)
        arms = [BernoulliArm(p) for p in ps]
        optimal_arm = np.argmax(ps)
        print(f"Action profile: {[arm.p for arm in arms]}")
        print(f"Optimal arm: {optimal_arm}")
    else:
        mus = np.around(np.linspace(start=1, stop=5, num=cfg.n_arms), decimals=3)
        mus = np.random.choice(mus, size=mus.shape[0], replace=False)
        arms = [GaussianArm(mu=mu, sigma=1) for mu in mus]
        true_vals = np.zeros(cfg.n_arms)
        for i, arm in enumerate(arms):
            true_mean = np.mean([arm.draw() for _ in range(1000)])
            true_vals[i] = true_mean
        optimal_arm = np.argmax(true_vals)
        print(f"Action profile: {true_vals}")
        print(f"Optimal arm: {optimal_arm}")
    
    epsilons = np.around(np.linspace(start=0.0, stop=0.5, num=8), decimals=3)
    results = []
    for eps in epsilons:
        learner = eGreedyMAB(n_arms=cfg.n_arms, alpha=cfg.alpha)
        result = run(nsim=cfg.nsim, nsteps=cfg.nsteps, learner=learner, epsilon=eps, arms=arms, optimal_arm=optimal_arm)
        results.append(result)

    ## save point
    with open(f"./{learner.__class__.__name__}_{cfg.nsim}_{cfg.nsteps}_{arms[0].__class__.__name__}_{cfg.alpha}_results.pkl", "wb") as f:
        pickle.dump(results, file=f, protocol=pickle.HIGHEST_PROTOCOL)
    