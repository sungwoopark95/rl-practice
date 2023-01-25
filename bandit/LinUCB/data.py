import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
# from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

## process movielens-1m data
DATA = ["movies", "ratings", "users"]

def get_data(name:str, test_ratio=.2):
    assert name in DATA
    if name == "movies":
        movies = pd.read_csv("/home/sungwoopark/rl-practice/bandit/LinUCB/datasets/movies.csv")
        movies.drop('title', axis=1, inplace=True)
        
        genres = []
        for i in range(movies.shape[0]):
            genre_string = movies['genre'].iloc[i]
            genre_split = genre_string.split(', ')
            for g in genre_split:
                if g not in genres:
                    genres.append(g)
        genres = sorted(genres)
        
        ## genre one-hot
        genre_onehot = np.zeros(shape=(movies.shape[0], len(genres)), dtype=np.uint8)
        for i in range(movies.shape[0]):
            g_split = movies['genre'].iloc[i].split(', ')
            for g in g_split:
                idx = genres.index(g)
                genre_onehot[i, idx] = 1

        for j in range(len(genres)):
            genre_name = genres[j]
            movies[f"is_{genre_name}"] = genre_onehot[:, j]

        movies.drop('genre', axis=1, inplace=True)
        
        df = movies
    
    elif name == "users":
        users = pd.read_csv("/home/sungwoopark/rl-practice/bandit/LinUCB/datasets/users.csv")
        users.drop("zipcode", axis=1, inplace=True)
        
        ## preprocessing
        # gender
        users['gender'] = users['gender'].map(lambda x: 1 if x == "F" else 0)
        
        # age
        bins = [0, 20, 30, 40, 50, 60, np.inf]
        names = ['<20', '20-29', '30-39','40-49', '51-60', '60+']
        labels = [i for i in range(len(names))]
        name_label = dict()
        for name, label in zip(names, labels):
            name_label[name] = label
        users['agegroup'] = pd.cut(users['age'], bins=bins, labels=names)
        users['agegroup'] = users['agegroup'].map(name_label)
        users.drop('age', axis=1, inplace=True)
        
        age_onehot = np.zeros(shape=(users.shape[0], users['agegroup'].nunique()), dtype=np.uint8)
        for i in range(users.shape[0]):
            group_idx = users['agegroup'].iloc[i]
            age_onehot[i][group_idx] = 1

        for j in range(users['agegroup'].nunique()):
            users[f"agegroup_{j}"] = age_onehot[:, j]

        users.drop('agegroup', axis=1, inplace=True)
        
        # occupation
        occupation_onehot = np.zeros(shape=(users.shape[0], users['occupation'].nunique()), dtype=np.uint8)
        for i in range(users.shape[0]):
            group_idx = users['occupation'].iloc[i]
            occupation_onehot[i][group_idx] = 1

        for j in range(users['occupation'].nunique()):
            users[f"occupation_{j}"] = occupation_onehot[:, j]

        users.drop('occupation', axis=1, inplace=True)
        
        df = users
    
    else:
        ratings = pd.read_csv("/home/sungwoopark/rl-practice/bandit/LinUCB/datasets/ratings.csv")
        ratings.drop('timestamp', axis=1, inplace=True)
        
        user_mean = ratings[['userid', 'ratings']].groupby(by='userid').mean()
        user_mean.reset_index(drop=False, inplace=True)
        ratings = pd.merge(left=ratings, right=user_mean, on='userid', how='left')
        ratings['reward'] = (ratings['ratings_x'] > ratings['ratings_y']).astype(np.uint8)
        ratings.drop(['ratings_x', 'ratings_y'], axis=1, inplace=True)
        
        df = ratings
        
    return df
