import numpy as np
import pandas as pd
from cfg import get_cfg
from mab import eGreedyMAB, UCB, ThompsonSampling
from arms import BernoulliArm, GaussianArm
from tqdm.auto import tqdm
import pickle


def run(nsim, nsteps, learner, arms, optimal_arm, is_ucb, is_thompson):
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
            
            if is_ucb:
                container.append(learner.conf)
            elif is_thompson:
                pass
            else:
                container.append(learner.epsilon)
            
            epochs.append(sim)
            runs.append(step)
            chosen_arms.append(chosen_arm)
            optimal_arms.append(optimal_arm)
            rewards.append(reward)
            
            learner.update(chosen_arm, reward)
    
    if is_thompson:
        result = pd.DataFrame({
            'sim': epochs,
            'step': runs,
            'chosen_arm': chosen_arms,
            'optimal_arm': optimal_arms,
            'reward': rewards
        })
    else:
        result = pd.DataFrame({
            'epsilon/conf': container,
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
        mus = np.random.randint(low=0, high=100, size=50)
        mus = np.random.choice(mus, size=cfg.n_arms, replace=False)
        mus = np.around(mus / 100., decimals=2)
        arms = [BernoulliArm(p) for p in mus]
        optimal_arm = np.argmax(mus)
        print(f"Action profile: {mus}")
        print(f"Optimal arm: {optimal_arm}")
    
    else:
        mus = np.linspace(start=-5, stop=10, num=100)
        mus = np.around(np.random.choice(mus, size=cfg.n_arms, replace=False), decimals=2)
        arms = [GaussianArm(mu=mu, sigma=1.5) for mu in mus]
        optimal_arm = np.argmax(mus)
        print(f"Action profile: {mus}")
        print(f"Optimal arm: {optimal_arm}")

    if cfg.model == 'mab':
        epsilons = [0., 0.01, 0.1, 0.3, 0.5, 1.0]
        
        results = []
        for eps in epsilons:
            learner = eGreedyMAB(n_arms=cfg.n_arms, epsilon=eps, alpha=cfg.alpha, initial=cfg.initial)
            result = run(nsim=cfg.nsim, nsteps=cfg.nsteps, learner=learner, arms=arms, optimal_arm=optimal_arm, is_ucb=False, is_thompson=False)
            results.append(result)
            
    elif cfg.model == 'ucb':
        confs = [0., 0.5, 1.0, 2.0, 3.0, 4.0]
        
        results = []
        for conf in confs:
            learner = UCB(n_arms=cfg.n_arms, conf=conf)
            result = run(nsim=cfg.nsim, nsteps=cfg.nsteps, learner=learner, arms=arms, optimal_arm=optimal_arm, is_ucb=True, is_thompson=False)
            results.append(result)
    
    elif cfg.model == 'thompson':
        learner = ThompsonSampling(n_arms=cfg.n_arms, bernoulli=cfg.bernoulli)
        results = run(nsim=cfg.nsim, nsteps=cfg.nsteps, learner=learner, arms=arms, optimal_arm=optimal_arm, is_ucb=False, is_thompson=True)

    ## save point
    with open(f"./{learner.__class__.__name__}_{arms[0].__class__.__name__}_{cfg.alpha}_{mode}_results.pkl", "wb") as f:
        pickle.dump(results, file=f, protocol=pickle.HIGHEST_PROTOCOL)
    