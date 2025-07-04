"""
author:Bruce Zhao
date: 2025/7/4 20:35
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from config import Config

class TSCFExtractor:
    def __init__(self, window_size=Config.SLIDING_WINDOW_SIZE, n_clusters=Config.TSCF_CLUSTERS):
        self.window_size = window_size
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        self.is_fitted = False

    def extract_features(self, episode_data):
        """Extract features from action sequences in an episode."""
        actions = np.array([step[1] for step in episode_data])
        sequences = []
        for i in range(len(actions) - self.window_size + 1):
            window = actions[i:i + self.window_size].flatten()
            sequences.append(window)
        return np.array(sequences) if sequences else np.empty((0, actions.shape[1] * self.window_size))

    def fit(self, all_sequences):
        """Train TSCF model."""
        if len(all_sequences) == 0:
            return
        self.scaler.fit(all_sequences)
        scaled_data = self.scaler.transform(all_sequences)
        self.kmeans.fit(scaled_data)
        self.is_fitted = True

    def transform(self, episode_sequences):
        """Convert sequences to TSCF representation."""
        if not self.is_fitted or len(episode_sequences) == 0:
            return np.zeros(self.n_clusters)
        scaled_data = self.scaler.transform(episode_sequences)
        cluster_labels = self.kmeans.predict(scaled_data)
        tscf_vector = np.zeros(self.n_clusters)
        unique_clusters = np.unique(cluster_labels)
        for cluster in unique_clusters:
            tscf_vector[cluster] = 1
        return tscf_vector