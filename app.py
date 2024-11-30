import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pipelines.text_preprocessing import *
from pipelines.ingestion_and_cleaning import *
from pipelines.model_builder import *
from pipelines.evaluator import *

# Set page configuration
st.set_page_config(page_title="Book Recommender", layout="wide")

# Load Data
@st.cache_data
def load_and_preprocess_data():
    print("Data Loading...")
    df = load_data()

    print("Data Preprocessing...")
    df = clean_ratings(df)
    df = clean_genre(df)
    df['keywords'] = df['description'].apply(lambda x: TextPreprocessor(x).parse_text())
    rating_mean = df['Rating_cleaned'].mean()
    min_rating = df['Rating_cleaned'].min()
    df['weighted_rating'] = weighted_rating(df['rating'], df['Rating_cleaned'], rating_mean, min_rating)
    final_df = build_final_dataset(df)

    print("Building Cosine Similarity Matrix...")
    tfidf_matrix = vectorize_data(final_df)
    cosine_sim_df = build_cosine_model(tfidf_matrix)

    print("Clustering...")
    kmeans = KMeans(n_clusters=8, **CLUSTER_PARAMS)
    kmeans.fit(np.array(cosine_sim_df))
    final_df['cluster'] = kmeans.labels

    print("Clustering Done!")
    return final_df, cosine_sim_df

final_df, cosine_sim_df = load_and_preprocess_data()

def recommend_books(book_title):
    book_map = build_map(final_df)
    try:
        high_score = get_recommendations(final_df, book_title, cosine_sim_df, book_map)
        metrics = relevant_and_recommended(final_df, book_title, cosine_sim_df, book_map)
        return {
            'recommendations': high_score[['title', 'score', 'cluster']],
            'metrics': metrics
        }
    except ValueError as e:
        st.error(str(e))
        return None

def plot_recommendation_metrics(metrics):
    precision = metrics['precision']
    recall = metrics['recall']
    relevant_items = metrics['total_relevant_items']
    recommended_items = metrics['total_recommended_items']

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    labels = ['Precision', 'Recall']
    values = [precision, recall]
    
    axs[0].bar(labels, values, color=['blue', 'green'])
    axs[0].set_title('Precision and Recall')
    axs[0].set_ylim(0, 1)
    
    for i, v in enumerate(values):
        axs[0].text(i, v, f'{v:.2f}', ha='center', va='bottom')
    
    items_data = [relevant_items, recommended_items]
    items_labels = ['Relevant Items', 'Total Recommended']
    
    axs[1].bar(items_labels, items_data, color=['red', 'orange'])
    axs[1].set_title('Items Comparison')
    
    for i, v in enumerate(items_data):
        axs[1].text(i, v, str(v), ha='center', va='bottom')

    plt.tight_layout()
    return fig

# Streamlit App Interface
def main():
    st.title("🚀 Intelligent Book Recommendation System")
    book_name = st.text_input("Enter the book name:", "")
    if book_name:
        st.subheader(f"Recommended Books for '{book_name}':")
        result = recommend_books(book_name)
        if result:
            # Create two columns for recommendations and metrics
            col1, col2 = st.columns([2, 1])
            with col1:
                # Display Recommendations
                st.write("Recommended Books:")
                st.dataframe(result['recommendations'])
            with col2:
                # Display Evaluation Metrics
                st.write("Recommendation Metrics:")
                metrics = result['metrics']
                
                st.metric("Precision", f"{metrics['precision']:.2f}")
                st.metric("Recall", f"{metrics['recall']:.2f}")
                st.metric("Relevant Items", metrics['total_relevant_items'])
            
            # Visualization of Metrics
            st.subheader("Metrics Visualization")
            fig = plot_recommendation_metrics(metrics)
            st.pyplot(fig)
            
            # Relevant Titles
            with st.expander("Relevant Titles"):
                st.write(list(metrics['recommended_relevant_titles']))
        
        else:
            st.write("No recommendations found or book not in the dataset.")

# Cluster Distribution Visualization
def plot_cluster_distribution():
    st.subheader("Cluster Distribution")
    cluster_counts = final_df['cluster'].value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    cluster_counts.plot(kind='bar', ax=ax)
    ax.set_title('Distribution of Books Across Clusters')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Books')
    
    st.pyplot(fig)

# Additional Analysis Page
def additional_analysis():
    st.header("System Analysis")
    
    # Cluster Distribution
    plot_cluster_distribution()
    
    # Optional: Similarity Score Distribution
    st.subheader("Similarity Score Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    similarity_scores = cosine_sim_df.flatten()
    sns.histplot(similarity_scores, kde=True, ax=ax)
    ax.set_title('Distribution of Similarity Scores')
    ax.set_xlabel('Similarity Score')
    st.pyplot(fig)

# Main App Navigation
def app():
    main()

# Run the app
if __name__ == "__main__":
    app()