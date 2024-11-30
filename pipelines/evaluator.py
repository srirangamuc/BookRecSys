import pandas as pd
import numpy as np

def build_map(df):
    return pd.Series(df.index, index=df['title'])

def get_recommendations(df, title, cosine_sim, map, top_n=10):
    if title not in map:
        raise ValueError(f"Title '{title}' not found in the dataset.")
        
    book_id = map[title]
    sim_scores = enumerate(cosine_sim[book_id])
    sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    
    book_indices = [idx for idx, _ in sorted_scores]
    scores = [score for _, score in sorted_scores]
    
    recommendations = df.iloc[book_indices][['title', 'author', 'genre_list_cleaned', 'cluster']]
    recommendations['score'] = scores
    recommendations['genre_list_cleaned'] = recommendations['genre_list_cleaned'].apply(lambda x: x.split())
    
    return recommendations

def get_relevant_items(df, title, cosine_sim, map, cluster):
    recommendations = get_recommendations(df, title, cosine_sim, map)
    relevant_items = recommendations.loc[
        (recommendations['cluster'] == cluster)
    ]
    return relevant_items

def relevant_and_recommended(df, title, cosine_sim, map, top_n=10):
    # Get the cluster of the input book
    book_index = map[title]
    cluster = df.loc[book_index, 'cluster']
    
    recommendations = get_recommendations(df, title, cosine_sim, map, top_n=top_n)
    relevant_items = get_relevant_items(df, title, cosine_sim, map, cluster)
    
    relevant_titles = set(relevant_items['title'])
    recommended_titles = set(recommendations['title'])
    
    recommended_and_relevant_titles = relevant_titles.intersection(recommended_titles)
    
    total_relevant_items = len(relevant_titles)
    total_recommended_items = len(recommended_titles)
    total_recommended_and_relevant_items = len(recommended_and_relevant_titles)
    
    return {
        'total_relevant_items': total_relevant_items,
        'total_recommended_items': total_recommended_items,
        'total_recommended_and_relevant_items': total_recommended_and_relevant_items,
        'precision': precision_k(total_recommended_and_relevant_items, total_recommended_items),
        'recall': recall_k(total_recommended_and_relevant_items, total_relevant_items),
        'recommended_relevant_titles': recommended_and_relevant_titles
    }

def precision_k(total_recommended_and_relevant_items, total_recommended_items):
    if total_recommended_items == 0:
        return 0.0
    return total_recommended_and_relevant_items / total_recommended_items

def recall_k(total_recommended_and_relevant_items, total_relevant_items):
    if total_relevant_items == 0:
        return 0.0
    return total_recommended_and_relevant_items / total_relevant_items