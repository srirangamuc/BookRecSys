import pandas as pd
import numpy as np
import re
import json

from sklearn.model_selection import train_test_split
from pipelines.constants import DATASET_PATH,MIN_NUM_RATINGS

def load_data():
    df = pd.read_csv(DATASET_PATH)\
        .drop(['web-scraper-order', 'web-scraper-start-url', 'genre', 'genre-href', 'book', 'book-href'], axis=1)
    df = df.dropna(how="any")
    return df

def clean_ratings(df):
    df['Rating_cleaned'] = df['total_rating'].apply(lambda x: re.sub(',(?!\s+\d$)','',x[:-8])).astype(np.int64)
    df = df[df['Rating_cleaned'] > MIN_NUM_RATINGS]
    return df

def clean_genre(df):
    df['genre_list_cleaned'] = df['list_genre'].apply(lambda x:[dict_genre['list_genre'] for dict_genre in json.loads(x)])
    df['genre_list_cleaned'] = df['genre_list_cleaned'].apply(lambda x:['-'.join(genre.split()) for genre in x[:-1]])
    return df