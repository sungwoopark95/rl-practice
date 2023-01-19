import numpy as np
import pandas as pd
from cfg import get_cfg
from mab import eGreedyMAB, UCB
from linucb import LinUCB, HybridLinUCB
from arms import BernoulliArm, GaussianArm
from tqdm.auto import tqdm
import pickle


def run(nsim, nsteps, learner, arms, optimal_arm, is_ucb=True):   
    epochs = []
    runs = []
    chosen_arms = []
    optimal_arms = []
    rewards = []
    if is_ucb:
        confs = []
    else:
        epsilons = []
    
    if cfg.tqdm:
        bar = tqdm(range(nsim))
    else:
        bar = range(nsim)
    
    for sim in bar:
        learner.initialize()
        for step in range(nsteps):
            chosen_arm = learner.choose()
            reward = arms[chosen_arm].draw()
                
            if is_ucb:
                confs.append(learner.conf)
            else:
                epsilons.append(learner.epsilon)
            epochs.append(sim)
            runs.append(step)
            chosen_arms.append(chosen_arm)
            optimal_arms.append(optimal_arm)
            rewards.append(reward)
            
            learner.update(chosen_arm, reward)
    
    if is_ucb:
        result = pd.DataFrame({
            'conf': confs,
            'sim': epochs,
            'step': runs,
            'chosen_arm': chosen_arms,
            'optimal_arm': optimal_arms,
            'reward': rewards
        })
    else:
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
    
    if cfg.seed is not None:
        np.random.seed(cfg.seed)
    
    if cfg.initial > 0:
        mode = "Optimistic"
    else:
        mode = "Naive"
    print(f"{mode} {cfg.model} bandit")

    if cfg.bernoulli:
        mus = np.linspace(start=1, stop=10, num=20)
        mus = np.random.choice(mus, size=cfg.n_arms, replace=False)
        mus = np.around(mus / 10., decimals=2)
        arms = [BernoulliArm(p) for p in mus]
        optimal_arm = np.argmax(mus)
        print(f"Action profile: {[arm.p for arm in arms]}")
        print(f"Optimal arm: {optimal_arm}")
    
    else:
        mus = np.arange(15)
        mus = np.random.choice(mus, size=cfg.n_arms, replace=False)
        arms = [GaussianArm(mu=mu, sigma=1) for mu in mus]
        optimal_arm = np.argmax(mus)
        print(f"Action profile: {mus}")
        print(f"Optimal arm: {optimal_arm}")


    if cfg.model == 'mab':
        epsilons = [0., 0.01, 0.1, 0.3, 0.5, 1.0]
        
        results = []
        for eps in epsilons:
            learner = eGreedyMAB(n_arms=cfg.n_arms, epsilon=eps, alpha=cfg.alpha, initial=cfg.initial)
            result = run(nsim=cfg.nsim, nsteps=cfg.nsteps, learner=learner, arms=arms, optimal_arm=optimal_arm, is_ucb=False)
            results.append(result)
            
    elif cfg.model == 'ucb':
        confs = [0.5, 1.0, 2.0, 3.0, 4.0]
        
        results = []
        for conf in confs:
            learner = UCB(n_arms=cfg.n_arms, conf=conf)
            result = run(nsim=cfg.nsim, nsteps=cfg.nsteps, learner=learner, arms=arms, optimal_arm=optimal_arm)
            results.append(result)


    ## save point
    with open(f"./{learner.__class__.__name__}_{arms[0].__class__.__name__}_{cfg.alpha}_{mode}_results.pkl", "wb") as f:
        pickle.dump(results, file=f, protocol=pickle.HIGHEST_PROTOCOL)
    