import numpy as np
import pandas as pd
from cfg import get_cfg
from mab import *
from arms import BernoulliArm, GaussianArm
from tqdm.auto import tqdm
import pickle

cfg = get_cfg()
model_repr = {
    "mab": "K-armed Bandit", 
    "ucbnaive": "Naive UCB Approach", 
    "ucbdelta": "UCB-delta Algorithm", 
    "ucbasym": "Asymptotically Optimal UCB", 
    "ucbmoss": "UCB MOSS Algorithm", 
    "thompson": "Thompson Sampling",
    "etc": "Explore-then-commit"
}

def run(nsim, nsteps, learner, arms, optimal_arm, model=cfg.model.lower()):
    epochs = []
    runs = []
    chosen_arms = []
    optimal_arms = []
    rewards = []
    optimal_rewards = []
    container = []
    
    if cfg.tqdm:
        bar = tqdm(range(nsim))
    else:
        bar = range(nsim)
    
    for sim in bar:
        learner.initialize()
        for step in range(nsteps):
            chosen_arm = learner.choose()
            reward = arms[chosen_arm].draw()
            
            if model == "thompson":
                pass
            else:
                if model == "ucbnaive":
                    container.append(learner.c)
                elif model == "etc":
                    container.append(learner.explore)
                elif model == "mab":
                    container.append(learner.epsilon)
                elif model == "ucbdelta":
                    container.append(learner.delta)
                    
            epochs.append(sim)
            runs.append(step)
            chosen_arms.append(chosen_arm)
            optimal_arms.append(optimal_arm)
            optimal_rewards.append(arms[optimal_arm].mu)
            rewards.append(reward)
            
            learner.update(chosen_arm, reward)
    
    if model not in ["ucbnaive", 'ucbdelta', 'etc', 'mab']:
        result = pd.DataFrame({
            'sim': epochs,
            'step': runs,
            'chosen_arm': chosen_arms,
            'optimal_arm': optimal_arms,
            'optimal_reward': optimal_rewards,
            'reward': rewards
        })
    else:
        result = pd.DataFrame({
            'param': container,
            'sim': epochs,
            'step': runs,
            'chosen_arm': chosen_arms,
            'optimal_arm': optimal_arms,
            'optimal_reward': optimal_rewards,
            'reward': rewards
        })

    return result.reset_index(drop=True)


if __name__ == "__main__":
    if cfg.initial > 0:
        mode = "Optimistic"
    else:
        mode = "Naive"
    print(f"{mode} {model_repr[cfg.model.lower()]} with", end=' ')

    ## Arm Generation
    if cfg.is_definite:
        print("definite", end=' ')
        mus = np.append((np.linspace(start=1., stop=30., num=cfg.n_arms-1)) / 100., 0.9)
    else:
        print("ambiguous", end=' ')
        mus = (np.linspace(start=7., stop=9., num=cfg.n_arms)) / 10.
    mus = np.around(mus, decimals=2)
    np.random.shuffle(mus)
    if cfg.bernoulli:
        print(f"Bernoulli arms")
        arms = [BernoulliArm(p) for p in mus]
        optimal_arm = np.argmax(mus)
        print(f"Action profile: {mus}")
        print(f"Optimal arm: {optimal_arm}")
    else:
        print(f"Gaussian arms")
        arms = [GaussianArm(mu=mu, sigma=1) for mu in mus]
        optimal_arm = np.argmax(mus)
        print(f"Action profile: {mus}")
        print(f"Optimal arm: {optimal_arm}")

    ## Model
    if cfg.model.lower() == 'mab':
        epsilons = [0., 0.01, 0.1, 0.5, 0.9, 1.0]
        results = []
        for eps in epsilons:
            learner = eGreedyMAB(n_arms=cfg.n_arms, epsilon=eps, alpha=cfg.alpha, initial=cfg.initial)
            result = run(nsim=cfg.nsim, nsteps=cfg.nsteps, learner=learner, arms=arms, optimal_arm=optimal_arm)
            results.append(result)
            
    elif cfg.model.lower() == 'ucbnaive':
        confs = [0.01, 0.1, 0.5, 1., 3.]
        results = []
        for conf in confs:
            learner = UCBNaive(n_arms=cfg.n_arms, c=conf)
            result = run(nsim=cfg.nsim, nsteps=cfg.nsteps, learner=learner, arms=arms, optimal_arm=optimal_arm)
            results.append(result)

    elif cfg.model.lower() == 'ucbdelta':
        deltas = [0.01, 0.1, 0.5, 0.9, 0.99]
        results = []
        for d in deltas:
            learner = UCBDelta(n_arms=cfg.n_arms, delta=d)
            result = run(nsim=cfg.nsim, nsteps=cfg.nsteps, learner=learner, arms=arms, optimal_arm=optimal_arm)
            results.append(result)
 
    elif cfg.model.lower() == 'ucbasym':
        learner = UCBAsymptotic(n_arms=cfg.n_arms)
        results = [run(nsim=cfg.nsim, nsteps=cfg.nsteps, learner=learner, arms=arms, optimal_arm=optimal_arm)]
    
    elif cfg.model.lower() == 'ucbmoss':
        learner = UCBMOSS(n_arms=cfg.n_arms, nsim=cfg.nsim)
        results = [run(nsim=cfg.nsim, nsteps=cfg.nsteps, learner=learner, arms=arms, optimal_arm=optimal_arm)]
    
    elif cfg.model.lower() == 'etc':
        explores = [0, 10, 25, 50]
        results = []
        for exp in explores:
            learner = ETC(n_arms=cfg.n_arms, explore=exp, horizon=cfg.nsteps)
            result = run(nsim=cfg.nsim, nsteps=cfg.nsteps, learner=learner, arms=arms, optimal_arm=optimal_arm)
            results.append(result)
    
    elif cfg.model.lower() == 'thompson':
        learner = ThompsonSampling(n_arms=cfg.n_arms, bernoulli=cfg.bernoulli)
        results = [run(nsim=cfg.nsim, nsteps=cfg.nsteps, learner=learner, arms=arms, optimal_arm=optimal_arm)]

    ## save point
    if cfg.is_definite:
        fname = f"/home/sungwoopark/rl-practice/bandit/MAB/definite/{learner.__class__.__name__}_{arms[0].__class__.__name__}_{cfg.alpha}_{mode}_results.pkl"
    else:
        fname = f"/home/sungwoopark/rl-practice/bandit/MAB/indefinite/{learner.__class__.__name__}_{arms[0].__class__.__name__}_{cfg.alpha}_{mode}_results.pkl"
    
    with open(fname, "wb") as f:
        pickle.dump(results, file=f, protocol=pickle.HIGHEST_PROTOCOL)
    