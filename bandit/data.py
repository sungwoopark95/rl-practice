import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

## process movielens-1m data
DATA = ["movies", "ratings", "users"]

def get_data(name:str, test_ratio=.2):
    assert name in DATA
    if name == "movies":
        df = pd.read_csv("./datasets/movies.csv")
        
        le_title = LabelEncoder()
        df['title'] = le_title.fit_transform(df['title'])
        
        le_genre = LabelEncoder()
        df['genre'] = le_genre.fit_transform(df['genre'])
        
        train, test = train_test_split(df, test_size=test_ratio)
    elif name == "users":
        df = pd.read_csv("./datasets/users.csv")
        
        le_gender = LabelEncoder()
        df['gender'] = le_gender.fit_transform(df['gender'])
        
        train, test = train_test_split(df, test_size=test_ratio)
    else:
        df = pd.read_csv("./datasets/ratings.csv")
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['second'] = df['timestamp'].dt.second
        
        train, test = train_test_split(df, test_size=test_ratio)
        
    return train, test