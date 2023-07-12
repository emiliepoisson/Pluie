# This algorithm is very similar to the CFICA algorithm. The key difference is the introduction 
# of the concept of outlier.
# Author: Boubacar Sow
# Date: July 10, 2023
# Copyright (c) 2023 Boubacar Sow

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import math

class ProposedApproach:
    def __init__(self, data, first_batch_size, batch_size, distance_threshold=2, merging_threshold=1):

        """
        Initializes the ProposedApproach class.
        
        :param data: The data to be clustered.
        :param first_batch_size: The size of the first batch of data to be used for initialization.
        :param batch_size: The size of each subsequent batch of data to be used for clustering.
        :param distance_threshold: The threshold (to add to add to the maximum distance) for adding a point to a cluster.
        :param merging_threshold: The threshold for merging two clusters.
        """
        self.data = data
        self.first_batch_size = first_batch_size
        self.batch_size = batch_size
        self.clusters = []
        self.k = 0
        self.distance_threshold = distance_threshold# You need to set this value based on your specific needs
        self.merging_threshold = merging_threshold# You need to set this value based on your specific needs
        self.p = 30
        self.max_distances = []
        self.cluster_features = []
        self.centroids = []
        self.labels = []
        self.outliers = []
        self.clusters = []


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


    def cluster_outliers(self):

        """
        Clusters the outliers using KMeans.
        """
        data_to_cluster = np.concatenate(self.clusters)
        data_to_cluster = np.vstack([data_to_cluster, self.outliers])
        kmeans = KMeans(n_clusters=self.k, n_init=10, random_state=42, init=self.centroids).fit(data_to_cluster)
        self.labels = np.array(self.labels)
        self.labels[self.labels == -1] = kmeans.labels_[-len(self.outliers):]
        self.labels = self.labels.tolist()
        self.clusters = [[] for _ in range(self.k)]
        for i, label in enumerate(self.labels):
            self.clusters[label].append(data_to_cluster[i])

        self.centroids = [np.mean(cluster, axis=0) for cluster in self.clusters]
        for i in range(self.k):
            if len(self.clusters[i]) >= 30:
                distances = np.linalg.norm(self.clusters[i] - self.centroids[i], axis=1)
                self.max_distances[i] = np.percentile(distances, 95)
        
        # Compute p_farthest points for each cluster
        self.cluster_features = [(mean, np.array(cluster)[np.argsort(np.linalg.norm(cluster - mean, axis=1))[-1:-self.p:-1]]) for mean, cluster in zip(self.centroids, self.clusters)]


    def initialize_clustering(self):

        """
        Initializes the clustering process by running KMeans on the first batch of data.
        """
        kmeans = KMeans(n_clusters=self.k, n_init=10).fit(self.data[:self.first_batch_size])
        # Create clusters based on KMeans labels
        self.clusters = [self.data[:self.first_batch_size][kmeans.labels_ == i] for i in range(self.k)]
        
        # Compute the centroids of each cluster
        self.centroids = kmeans.cluster_centers_
        
        # Compute the labels
        self.labels = kmeans.labels_.tolist()

        # Compute p_farthest points for each cluster
        self.cluster_features = [(mean, cluster[np.argsort(np.linalg.norm(cluster - mean, axis=1))[-1:-self.p:-1]]) for mean, cluster in zip(self.centroids, self.clusters)]

        # Compute max_distances for each cluster
        for i in range(self.k):
            distances = np.linalg.norm(self.clusters[i] - self.centroids[i], axis=1)
            self.max_distances.append(np.percentile(distances, 95))


    def fit(self):

        """
        Fits the ProposedApproach model to the data.
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

                if np.linalg.norm(point - self.centroids[min_cluster_index]) < self.max_distances[min_cluster_index]:
                    # Add point to the closest cluster
                    self.clusters[min_cluster_index] = np.vstack([self.clusters[min_cluster_index], point])
                    self.labels.append(min_cluster_index)

                    # Update centroid and p_farthest points for the closest cluster
                    self.centroids[min_cluster_index] = np.mean(self.clusters[min_cluster_index], axis=0)

                    if len(self.clusters[min_cluster_index]) >= 30:
                        self.max_distances[min_cluster_index] = np.max(np.linalg.norm(self.clusters[min_cluster_index] - self.centroids[min_cluster_index], axis=1))

                    # Update cluster features
                    mean = np.mean(self.clusters[min_cluster_index], axis=0)
                    points = self.clusters[min_cluster_index]
                    p_farthest_points = points[np.argsort(np.linalg.norm(points - mean, axis=1))[-1:-len(points)-1:-1]]
                    self.cluster_features[min_cluster_index] = (mean, p_farthest_points)
            
                elif np.linalg.norm(point - self.centroids[min_cluster_index]) < self.max_distances[min_cluster_index] + self.distance_threshold:
                    # Add point to the outliers
                    self.outliers.append(point)
                    self.labels.append(-1)
                    if len(self.outliers) > 200:
                        self.cluster_outliers()
                        self.outliers = []
                else:
                    # Form a new cluster
                    print("new cluster")
                    self.clusters.append(np.array([point]))
                    self.labels.append(self.k)
                    self.centroids = np.vstack([self.centroids, point])
                    self.max_distances.append(np.mean(self.max_distances))
                    self.cluster_features.append((point, np.array([point])))
                    self.k += 1


            # Compute the mean for each cluster
            self.centroids = [np.mean(cluster, axis=0) for cluster in self.clusters]

            # Merge closest clusters after processing a batch
            while True:
                min_distance = float('inf')
                min_cluster_pair = None
                for i in range(len(self.clusters)):
                    for j in range(i+1, len(self.clusters)):
                        distance = np.linalg.norm(self.centroids[i] - self.centroids[j])
                        if distance < min_distance:
                            min_distance = distance
                            min_cluster_pair = (i, j)

                if min_distance < self.merging_threshold:
                    # Merge the closest cluster pair
                    i, j = min_cluster_pair
                    self.clusters[i] = np.vstack([self.clusters[i], self.clusters[j]])
                    del self.clusters[j]

                    self.centroids[i] = np.mean(self.clusters[i], axis=0)
                    del self.centroids[j]
                    self.k -= 1
                    self.labels = [i if label == j else label for label in self.labels]
                    self.labels = [label - 1 if label > j else label for label in self.labels]
                    del self.max_distances[j]
                    self.max_distances[i] = np.max(np.linalg.norm(self.clusters[i] - self.centroids[i], axis=1))
                    # Update cluster features
                    del self.cluster_features[j]
                    print("After merging: ", self.k)
                else:
                    break

            
            # Compute the mean for each cluster
            self.centroids = [np.mean(cluster, axis=0) for cluster in self.clusters]
