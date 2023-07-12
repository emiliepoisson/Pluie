# Cluster Feature-Based Incremental Clustering Approach (CFICA) For Numerical Data
# Author: Boubacar Sow
# Date: July 10, 2023
# Copyright (c) 2023 Boubacar Sow

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
import math

class CFICA:
    def __init__(self, data, first_batch_size, batch_size, distance_threshold, merging_threshold):

        """
        Initializes the CFICA class.
        
        :param data: The data to be clustered.
        :param first_batch_size: The size of the first batch of data to be used for initialization.
        :param batch_size: The size of each subsequent batch of data to be used for clustering.
        :param distance_threshold: The threshold (to add to the maximum distance of the cluster) for adding a point to a cluster.
        :param merging_threshold: The threshold for merging two clusters.
        """
        self.data = data
        self.first_batch_size = first_batch_size
        self.batch_size = batch_size
        self.clusters = []
        self.cluster_features = []
        self.k = 0
        self.distance_threshold = distance_threshold  
        self.merging_threshold = merging_threshold
        self.p = 30
        self.centroids = []
        self.labels = []


    def find_optimal_k(self):
        
        """
        Finds the optimal number of clusters based on the highest silhouette score.
        """    
        X_batch = self.data[:self.first_batch_size]
        silhouette_scores = []
        k_range = range(2, 10)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X_batch)
            score = silhouette_score(X_batch, labels)
            silhouette_scores.append(score)
        
        # Find the optimal number of clusters based on the highest silhouette score
        self.k = k_range[np.argmax(silhouette_scores)]

    def initialize_clustering(self):

        """
        Initializes the clustering process by running KMeans on the first batch of data.
        """
        # Run KMeans on the first batch of data
        kmeans = KMeans(n_clusters=self.k, n_init=10).fit(self.data[:self.first_batch_size])
        
        # Create clusters based on KMeans labels
        self.clusters = [self.data[:self.first_batch_size][kmeans.labels_ == i] for i in range(self.k)]

        # Create labels based on KMeans labels
        self.labels = kmeans.labels_.tolist()
        
        # Compute the mean for each cluster
        means = [np.mean(cluster, axis=0) for cluster in self.clusters]
        
        # Compute p_farthest points for each cluster
        self.cluster_features = [(mean, cluster[np.argsort(np.linalg.norm(cluster - mean, axis=1))[-1:-self.p:-1]]) for mean, cluster in zip(means, self.clusters)]

    def fit(self):

        """
        Fits the CFICA model to the data.
        """
        self.find_optimal_k()
        self.initialize_clustering()
        num_batches = math.ceil((len(self.data) - self.first_batch_size) / self.batch_size)
        for i in range(num_batches):
            batch_start = self.first_batch_size + i * self.batch_size
            batch_end = min(self.first_batch_size + (i + 1) * self.batch_size, len(self.data))
            batch = self.data[batch_start:batch_end]

            for point in batch:
                min_distance = float('inf')
                min_cluster_index = None
                for i, (mean, p_farthest_points) in enumerate(self.cluster_features):
                    if len(p_farthest_points) > 0:
                        q = p_farthest_points[np.argmin(np.linalg.norm(p_farthest_points - point, axis=1))]
                        distance = np.linalg.norm(mean - point) + np.linalg.norm(q - point) * np.linalg.norm(mean - q)
                    else:
                        distance = np.linalg.norm(mean - point)

                    if distance < min_distance:
                        min_distance = distance
                        min_cluster_index = i


                if min_distance < self.distance_threshold:
                    # Add point to the closest cluster
                    self.clusters[min_cluster_index] = np.vstack([self.clusters[min_cluster_index], point])

                    # Update cluster features
                    mean = np.mean(self.clusters[min_cluster_index], axis=0)
                    distances = np.linalg.norm(self.clusters[min_cluster_index] - mean, axis=1)
                    third_quartile_plus_sigma = np.percentile(distances, 75) + np.std(distances)
                    non_outlier_points = self.clusters[min_cluster_index][distances < third_quartile_plus_sigma]
                    cluster_size = len(non_outlier_points)
                    num_farthest_points = min(cluster_size, self.p)  # Adjust the number of farthest points based on the cluster size
                    p_farthest_points = non_outlier_points[np.argsort(np.linalg.norm(non_outlier_points - mean, axis=1))[-1:-num_farthest_points-1:-1]]
                    self.cluster_features[min_cluster_index] = (mean, p_farthest_points)
                    self.labels.append(min_cluster_index)
                else:
                    # Form a new cluster
                    self.clusters.append(np.array([point]))
                    self.cluster_features.append((point, np.array([point])))
                    self.labels.append(self.k)
                    self.k += 1


            # Compute the mean for each cluster
            self.centroids = [np.mean(cluster, axis=0) for cluster in self.clusters]

            # Merge closest clusters after processing a batch
            while True:
                means = [cf[0] for cf in self.cluster_features]
                min_distance = float('inf')
                min_cluster_pair = None
                for i in range(len(self.clusters)):
                    for j in range(i+1, len(self.clusters)):
                        distance = np.linalg.norm(means[i] - means[j])
                        if distance < min_distance:
                            min_distance = distance
                            min_cluster_pair = (i, j)

                if min_distance < self.merging_threshold:
                    # Merge the closest cluster pair
                    i, j = min_cluster_pair
                    self.clusters[i] = np.vstack([self.clusters[i], self.clusters[j]])
                    del self.clusters[j]

                    # Update centroid and p_farthest points for the merged cluster
                    mean = np.mean(self.clusters[i], axis=0)
                    distances = np.linalg.norm(mean - self.clusters[i], axis=1)
                    third_quartile_plus_sigma = np.percentile(distances, 100)
                    non_outlier_points = self.clusters[i][distances <= third_quartile_plus_sigma]
                    cluster_size = len(non_outlier_points)
                    num_farthest_points = min(cluster_size, self.p)  # Adjust the number of farthest points based on the cluster size
                    p_farthest_points = non_outlier_points[np.argsort(np.linalg.norm(non_outlier_points - mean, axis=1))[-1:-num_farthest_points-1:-1]]
                    self.cluster_features[i] = (mean, p_farthest_points)
                    del means[j]
                    del self.cluster_features[j]
                    self.k -= 1
                    self.labels = [i if label == j else label for label in self.labels]
                    self.labels = [label - 1 if label > j else label for label in self.labels]

                else:
                    break

            print("After merging: ", self.k)
            # Compute the mean for each cluster
            self.centroids = [np.mean(cluster, axis=0) for cluster in self.clusters]
