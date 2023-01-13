import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
# from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

## process movielens-1m data
DATA = ["movies", "ratings", "users"]

def get_data(name:str, test_ratio=.2, drop_time=False):
    assert name in DATA
    if name == "movies":
        movies = pd.read_csv("/home/sungwoopark/rl-practice/bandit/linucb/datasets/movies.csv")
        movies.drop('title', axis=1, inplace=True)
        
        ## preprocessing to get only genres
        unique_genres = list(movies['genre'].unique())
        genres = []
        for item in unique_genres:
            split = item.split(', ')
            for jtem in split:
                if jtem not in genres:
                    genres.append(jtem)
                    
        df = pd.DataFrame(columns=['movieid']+genres)
        for i in range(movies.shape[0]):
            data = dict()
            row = movies.iloc[i]
            data['movieid'] = row['movieid']
            row_genre = row['genre'].split(', ')
            for g in genres:
                if g in row_genre:
                    data[g] = 1
                else:
                    data[g] = 0
            df = df.append(data, ignore_index=True)
    
    elif name == "users":
        user = pd.read_csv("/home/sungwoopark/rl-practice/bandit/linucb/datasets/users.csv")
        user.drop("zipcode", axis=1, inplace=True)
        
        ## preprocessing
        bins = [0, 20, 30, 40, 50, 60, np.inf]
        names = ['<20', '20-29', '30-39','40-49', '51-60', '60+']
        user['agegroup'] = pd.cut(user['age'], bins=bins, labels=names)
        user.drop("age", axis=1, inplace=True)
        
        columnsToEncode = ['agegroup', 'gender', 'occupation']
        myEncoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        myEncoder.fit(user[columnsToEncode])
        
        df = pd.concat([user.drop(columnsToEncode, 1), 
                        pd.DataFrame(myEncoder.transform(user[columnsToEncode]), 
                                     columns = myEncoder.get_feature_names_out(columnsToEncode))], 
                       axis=1).reindex()
    
    else:
        df = pd.read_csv("/home/sungwoopark/rl-practice/bandit/linucb/datasets/ratings.csv")
        
        if drop_time:
            df.drop("timestamp", axis=1, inplace=True)
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['year'] = df['timestamp'].dt.year
            df['month'] = df['timestamp'].dt.month
            df['day'] = df['timestamp'].dt.day
            df['hour'] = df['timestamp'].dt.hour
            df['minute'] = df['timestamp'].dt.minute
            df['second'] = df['timestamp'].dt.second
        
        # mean = df['ratings'].mean()
        # df['reward'] = df['ratings'].map(lambda x: 1 if x >= mean else 0)
        # df.drop('ratings', axis=1, inplace=True)
        
    return df