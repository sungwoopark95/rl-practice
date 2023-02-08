import numpy as np
import pandas as pd
from cfg import get_cfg
from mab import eGreedyMAB, UCB, ThompsonSampling, ETC
from arms import BernoulliArm, GaussianArm
from tqdm.auto import tqdm
import pickle

cfg = get_cfg()
model_repr = {
    "mab": "K-armed Bandit", 
    "ucb": "UCB Approach", 
    "thompson": "Thompson Sampling",
    "etc": "Explore-then-commit"
}

def run(nsim, nsteps, learner, arms, optimal_arm, model=cfg.model.lower()):
    epochs = []
    runs = []
    chosen_arms = []
    optimal_arms = []
    rewards = []
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
                if model == "ucb":
                    container.append(learner.conf)
                elif model == "etc":
                    container.append(learner.explore)
                else:
                    container.append(learner.epsilon)
                    
            epochs.append(sim)
            runs.append(step)
            chosen_arms.append(chosen_arm)
            optimal_arms.append(optimal_arm)
            rewards.append(reward)
            
            learner.update(chosen_arm, reward)
    
    if model == "thompson":
        result = pd.DataFrame({
            'sim': epochs,
            'step': runs,
            'chosen_arm': chosen_arms,
            'optimal_arm': optimal_arms,
            'reward': rewards
        })
    else:
        result = pd.DataFrame({
            'param': container,
            'sim': epochs,
            'step': runs,
            'chosen_arm': chosen_arms,
            'optimal_arm': optimal_arms,
            'reward': rewards
        })

    return result.reset_index(drop=True)


if __name__ == "__main__":    
    if cfg.seed is not None:
        np.random.seed(cfg.seed)
    
    if cfg.initial > 0:
        mode = "Optimistic"
    else:
        mode = "Naive"
    print(f"{mode} {model_repr[cfg.model.lower()]} with", end=' ')

    ## Arm Generation
    if cfg.bernoulli:
        print(f"Bernoulli Arm")
        mus = np.linspace(start=0, stop=10, num=(cfg.n_arms*4))
        mus = np.random.choice(mus, size=cfg.n_arms, replace=False)
        mus = np.around(mus / 10., decimals=2)
        arms = [BernoulliArm(p) for p in mus]
        optimal_arm = np.argmax(mus)
        print(f"Action profile: {mus}")
        print(f"Optimal arm: {optimal_arm}")
    
    else:
        print(f"Gaussian Arm")
        mus = np.linspace(start=-5, stop=5, num=(cfg.n_arms*4))
        mus = np.around(np.random.choice(mus, size=cfg.n_arms, replace=False), decimals=2)
        mus = np.around(mus / 5., decimals=2)
        arms = [GaussianArm(mu=mu, sigma=1) for mu in mus]
        optimal_arm = np.argmax(mus)
        print(f"Action profile: {mus}")
        print(f"Optimal arm: {optimal_arm}")

    ## Model
    if cfg.model.lower() == 'mab':
        epsilons = [0., 0.01, 0.2, 0.5, 1.0]
        results = []
        for eps in epsilons:
            learner = eGreedyMAB(n_arms=cfg.n_arms, epsilon=eps, alpha=cfg.alpha, initial=cfg.initial)
            result = run(nsim=cfg.nsim, nsteps=cfg.nsteps, learner=learner, arms=arms, optimal_arm=optimal_arm)
            results.append(result)
            
    elif cfg.model.lower() == 'ucb':
        confs = [0., 0.5, 1.0, 2.0, 3.0]
        results = []
        for conf in confs:
            learner = UCB(n_arms=cfg.n_arms, conf=conf)
            result = run(nsim=cfg.nsim, nsteps=cfg.nsteps, learner=learner, arms=arms, optimal_arm=optimal_arm)
            results.append(result)
    
    elif cfg.model.lower() == 'etc':
        explores = [0, 10, 50, 100]
        results = []
        for exp in explores:
            learner = ETC(n_arms=cfg.n_arms, explore=exp, horizon=cfg.nsteps)
            result = run(nsim=cfg.nsim, nsteps=cfg.nsteps, learner=learner, arms=arms, optimal_arm=optimal_arm)
            results.append(result)
    
    elif cfg.model.lower() == 'thompson':
        learner = ThompsonSampling(n_arms=cfg.n_arms, bernoulli=cfg.bernoulli)
        results = run(nsim=cfg.nsim, nsteps=cfg.nsteps, learner=learner, arms=arms, optimal_arm=optimal_arm)

    ## save point
    with open(f"./{learner.__class__.__name__}_{arms[0].__class__.__name__}_{cfg.alpha}_{mode}_results.pkl", "wb") as f:
        pickle.dump(results, file=f, protocol=pickle.HIGHEST_PROTOCOL)
    