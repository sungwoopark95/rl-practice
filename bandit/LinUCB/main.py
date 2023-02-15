import matplotlib.pyplot as plt
from linucb import LinUCB, eLinUCB, HybridLinUCB, LinTS
from tqdm.auto import tqdm
from data import get_data
from cfg import get_cfg
from datetime import datetime

def run(learner, data, arms, users, nsim):
    aligned_ctr = []
    aligned_timestep = 0
    cum_reward = 0
    
    arm_list = arms['movieid']
    for sim in range(nsim):
        if sim == 0:
            data = data.copy()
            unused = []
        else:
            data = data.iloc[unused].copy()
            unused = []
        cnt = data.shape[0]
        if cfg.tqdm:
            bar = tqdm(range(data.shape[0]))
        else:
            bar = range(data.shape[0])
        for i in bar:
            user_id = data['userid'].iloc[i]
            movie_id = data['movieid'].iloc[i]
            user_feature = users[users['userid'] == user_id].iloc[:, 1:].to_numpy()
            chosen_arm = learner.choose(user_feature)
            if arm_list[chosen_arm] == movie_id:
                reward = data['reward'].iloc[i]
                learner.update(user_feature, chosen_arm, reward)

                aligned_timestep += 1
                cum_reward += reward
                aligned_ctr.append((cum_reward / aligned_timestep))
            else:
                unused.append(i)
        print(f"Hit count: {cnt - len(unused)}")
    
    return {
        "aligned_ctr": aligned_ctr,
        "aligned_timestamp": aligned_timestep,
        "cum_reward": cum_reward,
    }

def run_to_plot(model_name, data, arms, users, nsim, alphas):
    plt.figure(figsize=(6, 5))
    for alpha in alphas:
        print(f"alpha={alpha}")
        learner = bandit_init[model_name]
        learner.alpha = alpha
        result = run(learner, data, arms, users, nsim)
        plt.plot(result['aligned_ctr'], label=f"alpha={alpha}")
    
    plt.axhline(y=reward_mean, color="red")
    plt.ylim([reward_mean-0.3, 1.05])
    plt.title(f"{learner.__class__.__name__} top{cfg.topN} NSim={cfg.nsim}")
    plt.grid(True)
    plt.legend()
    plt.xlabel('Step')
    plt.ylabel('CTR')
    now = datetime.now()
    plt.savefig(f"/home/sungwoopark/rl-practice/bandit/LinUCB/plots/{learner.__class__.__name__}_top{cfg.topN}_{now}.png")
    print(f"Saved plot successfully at {now}!")


if __name__ == "__main__":
    ## prepare data and preprocess
    cfg = get_cfg()

    users, movies, ratings = get_data("users"), get_data("movies"), get_data("ratings")

    topN = cfg.topN
    top_movies = ratings[["movieid", "userid"]].groupby(by="movieid").count().sort_values(by=["userid"], ascending=False)
    top_movies.reset_index(drop=False, inplace=True)

    top_n_movies = movies[movies['movieid'].isin(top_movies.head(topN)['movieid'])]
    top_n_movies.sort_values(by='movieid', inplace=True)
    top_n_movies.reset_index(drop=True, inplace=True)

    top_n_ratings = ratings[ratings['movieid'].isin(top_n_movies['movieid'])]
    top_n_ratings.reset_index(drop=True, inplace=True)    

    ## prepare for training
    arm_to_use = top_n_movies.iloc[:, 1:].to_numpy()
    arm_features = movies.shape[1] - 1
    user_features = users.shape[1] - 1
    d = arm_features + user_features
    k = arm_features * user_features
    reward_mean = top_n_ratings["reward"].mean()
    alphas = [0., 0.5, 1.0, 2.0]
    print(f"Mean reward: {reward_mean}")
    
    ## training
    bandit_init = {
        'linucb': LinUCB(arms=arm_to_use, d=d),
        'elinucb': eLinUCB(arms=arm_to_use, d=d, epsilon=cfg.epsilon),
        'hybrid': HybridLinUCB(arms=arm_to_use, d=d, k=k),
        'lints': LinTS(arms=arm_to_use, d=d),
    }
    
    if cfg.model is None:
        print(f"Simulation on all algorithms")
        for model in bandit_init.keys():
            print(f"{model}")
            run_to_plot(
                model_name=model,
                data=top_n_ratings, 
                arms=top_n_movies,
                users=users,
                nsim=cfg.nsim,
                alphas=alphas
            )
    else:
        print(f"Simulation on {cfg.model.lower()}")
        run_to_plot(
            model_name=cfg.model.lower(),
            data=top_n_ratings, 
            arms=top_n_movies,
            users=users,
            nsim=cfg.nsim,
            alphas=alphas
        )
