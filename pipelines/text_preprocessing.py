import re
import numpy as np
import pandas as pd
from pipelines.constants import NLP, STOPWORDS, PUNCTUATION
# from ingestion_and_cleaning import load_data,clean_ratings,clean_genre


def weighted_rating(R, v, C, m):    
    return (R * v) + (C * m) / (v + m)

class TextPreprocessor:
    def __init__(self, text):
        self.text = text

    def strip_html(self):
        clean = re.compile('<.*?>')
        self.text = re.sub(clean, '', self.text)

    def remove_stopwords(self):
        words = self.text.split()
        self.text = ' '.join([word for word in words if word not in STOPWORDS])

    def remove_digits(self):    
        self.text = re.sub(r'[0-9]', '', self.text)

    def remove_punctuation(self):
        self.text = ''.join([char for char in self.text if char not in PUNCTUATION])

    def get_keywords(self):
        doc = NLP(self.text)
        self.text = ' '.join([item.text.strip() for item in doc.ents])

    def parse_text(self):
        self.text = self.text.lower()
        self.strip_html()
        self.remove_stopwords()
        self.remove_digits()
        self.remove_punctuation()
        self.get_keywords()
        return self.text

def build_final_dataset(df):
    features = ['title', 'author', 'genre_list_cleaned','keywords']
    final_df = df.loc[:, features]
    final_df['genre_list_cleaned'] = final_df['genre_list_cleaned'].apply(lambda x: ' '.join(x))
    final_df['keywords'] = final_df['keywords'].apply(lambda x: ' '.join(list(set(x.split()))))
    final_df['corpus'] = final_df[['genre_list_cleaned','keywords']].agg(' '.join,axis=1).str.lower()
    return final_df

# ## Testing
# df = load_data()

# print('Cleaning Dataset')
# print('Cleaning Ratings')
# df = clean_ratings(df)

# print('Cleaning Genre')
# df = clean_genre(df)

# print("Starting Text Preprocessing.....")
# df['keywords'] = df['description'].apply(lambda x: TextPreprocessor(x).parse_text())

# print('Making Weighted Ratings')
# rating_mean = np.mean(df['Rating_cleaned'])
# min_rating = np.min(df['Rating_cleaned'])
# df['weighted_rating'] = weighted_rating(df['rating'], df['Rating_cleaned'],rating_mean,min_rating) / 1e6
# print(df[['title','weighted_rating']].head())

# print('Building Final Dataset')
# final_df = build_final_dataset(df)
# print(final_df.head())