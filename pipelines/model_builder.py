from pipelines.constants import VECTORIZER,CLUSTER_PARAMS,NUM_CLUSTERS,SSE
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from kneed import KneeLocator
import json

class KMeans:
    def __init__(self, n_clusters=3, max_iter=100, n_init=10, random_state=None, init='random'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.init = init
        self.centroids = None
        self.labels = None
        self.inertia_ = None
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
    def _initialize_centroids(self, X):
        X = np.array(X)
        
        if self.init == 'random':
            random_idx = np.random.choice(len(X), self.n_clusters, replace=False)
            self.centroids = X[random_idx]
        elif self.init == 'k-means++':
            self.centroids = self._initialize_centroids_kmeans_plus(X)
        else:
            raise ValueError("Unknown initialization method")

    def _initialize_centroids_kmeans_plus(self, X):
        X = np.array(X)
        centroids = []
        centroids.append(X[np.random.randint(0, len(X))])
        
        for _ in range(1, self.n_clusters):
            distances = np.min([np.linalg.norm(X - c, axis=1) for c in centroids], axis=0)
            prob_dist = distances / np.sum(distances)
            new_centroid = X[np.random.choice(len(X), p=prob_dist)]
            centroids.append(new_centroid)
            
        return np.array(centroids)

    def _assign_labels(self, X):
        X = np.array(X)
        distances = np.array([np.linalg.norm(X - centroid, axis=1) for centroid in self.centroids])
        return np.argmin(distances, axis=0)

    def _compute_inertia(self, X):
        X = np.array(X)
        total_distance = 0
        for i in range(self.n_clusters):
            mask = self.labels == i
            if np.any(mask):  
                cluster_points = X[mask]
                centroid = self.centroids[i]
                distances = np.sum((cluster_points - centroid) ** 2, axis=1)
                total_distance += np.sum(distances)
        return total_distance

    def _update_centroids(self, X):
        """Update centroids based on current label assignments."""
        X = np.array(X)
        new_centroids = np.array([X[self.labels == i].mean(axis=0) 
                                 for i in range(self.n_clusters)])
        return new_centroids

    def fit(self, X):
        X = np.array(X)  
        best_inertia = float('inf')
        best_centroids = None
        best_labels = None

        for i in range(self.n_init):
            print(f"Running initialization {i+1}/{self.n_init}...")
            self._initialize_centroids(X)

            for _ in range(self.max_iter):
                old_centroids = self.centroids.copy()
                self.labels = self._assign_labels(X)
                self.centroids = self._update_centroids(X)
                self.inertia_ = self._compute_inertia(X)
                if np.allclose(old_centroids, self.centroids):
                    break

            if self.inertia_ < best_inertia:
                best_inertia = self.inertia_
                best_centroids = self.centroids.copy()
                best_labels = self.labels.copy()

        self.centroids = best_centroids
        self.labels = best_labels
        self.inertia_ = best_inertia
        return self

    def predict(self, X):
        X = np.array(X)  
        return self._assign_labels(X)
    
    def save(self, filename):
        model_data = {
            "n_clusters": self.n_clusters,
            "max_iter": self.max_iter,
            "n_init": self.n_init,
            "random_state": self.random_state,
            "init": self.init,
            "centroids": self.centroids.tolist(),
            "labels": self.labels.tolist(),
            "inertia": self.inertia_
        }
        
        with open(filename, 'w') as f:
            json.dump(model_data, f)

        print(f"Model saved to {filename}")

    @classmethod
    def load(cls, filename):
        with open(filename, 'r') as f:
            model_data = json.load(f)

        model = cls(
            n_clusters=model_data['n_clusters'],
            max_iter=model_data['max_iter'],
            n_init=model_data['n_init'],
            random_state=model_data['random_state'],
            init=model_data['init']
        )
        
        model.centroids = np.array(model_data['centroids'])
        model.labels = np.array(model_data['labels'])
        model.inertia_ = model_data['inertia']
        
        print(f"Model loaded from {filename}")
        return model

def vectorize_data(df):
    tfidf_matrix = VECTORIZER.fit_transform(df['corpus'])
    return tfidf_matrix

def build_cosine_model(tfidf_matrix):
    cosine_sim_matrix = cosine_similarity(tfidf_matrix,tfidf_matrix)
    cosine_sim_matrix_df = pd.DataFrame(cosine_sim_matrix)
    return cosine_sim_matrix_df

def find_optimal_num_of_clusters(cosine_sim_df):
    X = np.array(cosine_sim_df)
    for k in NUM_CLUSTERS:
        print(f'Cluster {k}/40')
        kmeans = KMeans(n_clusters=k, **CLUSTER_PARAMS)
        kmeans.fit(X)
        SSE.append(kmeans.inertia_)
    locator = KneeLocator(NUM_CLUSTERS, SSE, curve='convex', direction='decreasing')
    print('Best Cluster for KMeans: ', locator.elbow)