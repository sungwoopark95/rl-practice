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
        if random.random() > self.p:
            return 0
        else:
            return 1
        

class GaussianArm:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        
    def draw(self):
        return np.random.normal(loc=self.mu, scale=self.sigma)


def run(nsim, nsteps, learner, arms, epsilon, optimal_arm):
    result = pd.DataFrame(columns=['epsilon', 'epoch', 'run', 'chosen_arm', 'optimal_arm', 'reward'])
    
    if cfg.tqdm:
        bar = tqdm(range(nsim))
    else:
        bar = range(nsim)
    
    for sim in bar:
        learner.initialize(epsilon)
        for i, step in enumerate(range(nsteps)):
            chosen_arm = learner.choose()
            reward = arms[chosen_arm].draw()
            
            data = pd.DataFrame(
                {"epsilon": epsilon,
                "epoch": sim,
                "run": step,
                "chosen_arm": chosen_arm,
                "optimal_arm": optimal_arm,
                "reward": reward}, 
                index=[i]
            )
            result = pd.concat([result, data], axis=0)
            
            learner.update(chosen_arm, reward)
    
    return result.reset_index(drop=True)


if __name__ == "__main__":
    cfg = get_cfg()
       
    if cfg.bernoulli:
        ps = np.random.uniform(low=0, high=1, size=cfg.n_arms)
        arms = [BernoulliArm(p) for p in ps]
        optimal_arm = np.argmax(ps)
        print(f"Action profile: {[arm.p for arm in arms]}")
        print(f"Optimal arm: {optimal_arm}")
    else:
        vars = np.random.randint(low=1, high=10, size=cfg.n_arms)
        arms = [GaussianArm(mu=1, sigma=var) for var in vars]
        true_vals = np.zeros(cfg.n_arms)
        for i, arm in enumerate(arms):
            true_mean = np.mean([arm.draw() for _ in range(1000)])
            true_vals[i] = true_mean
        optimal_arm = np.argmax(true_vals)
        print(f"Action profile: {true_vals}")
        print(f"Optimal arm: {optimal_arm}")
    
    learner = eGreedyMAB(n_arms=cfg.n_arms, eps_fixed=cfg.eps_fixed)
    epsilons = np.linspace(start=0, stop=1, num=11)
    results = []
    for eps in epsilons:
        result = run(nsim=cfg.nsim, nsteps=cfg.nsteps, learner=learner, arms=arms, epsilon=eps, optimal_arm=optimal_arm)
        results.append(result)

    ## save point
    with open(f"/home/sungwoopark/rl-practice/bandit/classic-bandit/{learner.__class__.__name__}_{cfg.nsim}_{cfg.nsteps}_{arms[0].__class__.__name__}_results.pkl", "wb") as f:
        pickle.dump(results, file=f, protocol=pickle.HIGHEST_PROTOCOL)
    