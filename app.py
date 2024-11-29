import streamlit as st
import numpy as np
from pipelines.text_preprocessing import *
from pipelines.ingestion_and_cleaning import *
from pipelines.model_builder import *
from pipelines.evaluator import *

# Load Data
df = load_data()

# Data Preprocessing
df = clean_ratings(df)
df = clean_genre(df)
df['keywords'] = df['description'].apply(lambda x: TextPreprocessor(x).parse_text())
rating_mean = df['Rating_cleaned'].mean()
min_rating = df['Rating_cleaned'].min()
df['weighted_rating'] = weighted_rating(df['rating'], df['Rating_cleaned'], rating_mean, min_rating)
final_df = build_final_dataset(df)

# Create Cosine Similarity Matrix
tfidf_matrix = vectorize_data(final_df)
cosine_sim_df = build_cosine_model(tfidf_matrix)

# Clustering
kmeans = KMeans(n_clusters=8, **CLUSTER_PARAMS)
kmeans.fit(np.array(cosine_sim_df))
final_df['cluster'] = kmeans.labels

def recommend_books(book_title):
    book_map = build_map(final_df)
    try:
        high_score = get_recommendations(final_df, book_title, cosine_sim_df, book_map)
        metrics = relevant_and_recommended(final_df, book_title, cosine_sim_df, book_map)
        return {
            'recommendations': high_score[['title', 'score']],
            'metrics': metrics
        }
    except ValueError as e:
        st.error(str(e))
        return None

# Streamlit App Interface
st.title("Book Recommendation System")

# Text Input for Book Search
book_name = st.text_input("Enter the book name:", "")

if book_name:
    st.subheader(f"Recommended Books for '{book_name}':")
    
    # Get book recommendations based on the input
    result = recommend_books(book_name)
    
    if result:
        # Display Recommendations
        st.write("Recommended Books:")
        st.dataframe(result['recommendations'])
        
        # Display Evaluation Metrics
        st.subheader("Recommendation Metrics:")
        metrics = result['metrics']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Precision", f"{metrics['precision']:.2f}")
        
        with col2:
            st.metric("Recall", f"{metrics['recall']:.2f}")
        
        with col3:
            st.metric("Relevant Items", metrics['total_relevant_items'])
        
        # Display Relevant Titles
        st.subheader("Relevant Titles:")
        st.write(list(metrics['recommended_relevant_titles']))
    else:
        st.write("No recommendations found or book not in the dataset.")