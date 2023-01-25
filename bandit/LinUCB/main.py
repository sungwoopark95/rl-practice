import matplotlib.pyplot as plt
from linucb import LinUCB, eLinUCB, HybridLinUCB
from tqdm.auto import tqdm
from data import get_data
from cfg import get_cfg

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
            bar =tqdm(range(data.shape[0]))
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
    reward_mean = top_n_ratings["reward"].mean()
    print(f"Mean reward: {reward_mean}")
    
    ## training
    if cfg.model.lower() == "linucb":
        learner = LinUCB(arms=arm_to_use, d=d, alpha=cfg.alpha)
        print(learner.__class__.__name__)
        result = run(
            learner=learner, 
            data=top_n_ratings, 
            arms=top_n_movies,
            users=users,
            nsim=cfg.nsim
        )
        
    elif cfg.model.lower() == "elinucb":
        learner = eLinUCB(arms=arm_to_use, d=d, alpha=cfg.alpha, epsilon=cfg.epsilon)
        print(learner.__class__.__name__)
        result = run(
            learner=learner, 
            data=top_n_ratings, 
            arms=top_n_movies,
            users=users,
            nsim=cfg.nsim
        )
        
    elif cfg.model.lower() == "hybrid":
        k = arm_features * user_features
        learner = HybridLinUCB(arms=arm_to_use, d=d, k=k, alpha=cfg.alpha)
        print(learner.__class__.__name__)
        result = run(
            learner=learner, 
            data=top_n_ratings, 
            arms=top_n_movies,
            users=users,
            nsim=cfg.nsim
        )

    if cfg.save_plot:
        plt.figure(figsize=(7, 5))
        plt.plot(result['aligned_ctr'])
        plt.axhline(y=reward_mean, color="red")
        plt.ylim([reward_mean-0.3, 1.05])
        plt.title(f"{learner.__class__.__name__}_top{cfg.topN}_alpha={cfg.alpha}")
        plt.grid(True)
        plt.savefig(f"./{cfg.model}_top{cfg.topN}_alpha_{cfg.alpha}_nsim_{cfg.nsim}.png")
        print("Saved plot successfully!")
