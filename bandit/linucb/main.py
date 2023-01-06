from data import get_data
from linucb import LinUCB, HybridLinUCB
from cfg import get_cfg
import pandas as pd
import numpy as np
from tqdm import tqdm

## prepare dataset
users = get_data("users")
movies = get_data("movies")
ratings = get_data("ratings", drop_time=True)

## define running function
def run(learner, data):
    T = len(users)
    correct = np.zeros(shape=T)
    movie_ids = np.asarray(movies['movieid'])
    
    if cfg.tqdm:
        bar = tqdm(range(T))
    else:
        bar = range(T)
    
    for t in bar:
        user = dict(users.iloc[t])
        user_id = user['userid']
        del user['userid']
        user_preference = list(data[(data['userid'] == user_id) & (data['reward'] == 1)]['movieid'])
        
        x = dict(user)
        idx = learner.choose(x)
        movie_chosen = movie_ids[idx]
        correct[t] = (movie_chosen in user_preference)
        # print(f"{user_id}\tidx: {idx}, movie chosen: {movie_chosen}\tis in?: {correct[t]}")
        
        learner.update(x, idx, correct[t])
        
    return np.mean(correct)
        
if __name__ == "__main__":
    cfg = get_cfg()
    print(f"Start running...")
    
    n = 30
    top_movies_index = ratings.groupby("movieid").count().sort_values("userid", ascending = False).head(n).reset_index()["movieid"]
    top_movies_features = movies[movies["movieid"].isin(top_movies_index)]
    filtered_data_original = ratings[ratings["movieid"].isin(top_movies_index)]
    filtered_data_original["reward"] = np.where(filtered_data_original["ratings"] <5,0,1)
    filtered_data_original.reset_index(drop=True, inplace=True)
        
    for epoch in range(cfg.epoch):
        print(f"Epoch {epoch+1} running...")
        user_features = list(users.columns.values)[1:]
        arms = movies.drop("movieid", axis=1)
        linucb = LinUCB(arms=arms, user_features=user_features, alpha=0.5)
        hybrid_linucb = HybridLinUCB(arms=arms, user_features=user_features, alpha=0.5)
        
        print(f"\tLinUCB running...")
        linucb_result = run(linucb, data=filtered_data_original)
        print(f"\t[Epoch {epoch+1}] {linucb.__class__.__name__} result: {linucb_result}")
        
        # print(f"\tHybrid LinUCB running...")
        # hybrid_result = run(hybrid_linucb, data=filtered_data_original)
        # print(f"[Epoch {epoch+1}] {hybrid_linucb.__class__.__name__} result: {hybrid_result}")