import math
from collections import Counter
from typing import Literal
import numpy as np

class Kmeans:
    def __init__(self, n_clusters: int, max_iterations: int = 1000, n_runs: int = 51, centroid_init: Literal['existing_points', 'random_points', 'precomputed'] = 'existing_points', empty_cluster: Literal['use_previous, add_random_point', 'remove_cluster'] = 'use_previous', centroids: np.array = None):
        self.X = None
        self.n_points = None
        self.n_features = None
        self.labels = None
        self.centroids = centroids

        self.k = n_clusters
        self.centroid_init = centroid_init
        self.empty_cluster = empty_cluster

        self.iterations = 0
        self.max_iterations = max_iterations
        self.n_runs = n_runs
        self.results = None
        self.final_labels = None

    def init_centroids(self):
        centroids = None
        if self.centroid_init == 'existing_points':
            c = np.random.choice(self.n_points, self.k)
            centroids = self.X[c, :]
        elif self.centroid_init == 'random_points':
            centroids = np.random.uniform(low=np.min(self.X), high=np.max(self.X), size=(self.k, self.n_features))
        elif self.centroid_init == 'precomputed':
            centroids = self.centroids
        centroids = centroids.astype('float64')
        return centroids

    def compute_distances(self, point):
        distances = np.zeros(self.k)
        for cent in range(len(self.centroids)):
            dist = np.linalg.norm(self.X[point] - self.centroids[cent])
            distances[cent] = dist
        return distances

    def assign_labels(self, distances):
        return np.argmin(distances)

    def deal_with_empty_cluster(self, cluster_label):
        cent = None
        if self.empty_cluster == 'use_previous':
            cent = self.centroids[cluster_label]
        elif self.empty_cluster == 'add_random_point':
            c = np.random.choice(self.n_points)
            self.labels[c] = cluster_label
            cent = self.X[c, :]
        elif self.empty_cluster == 'remove_cluster':
            cent = None

        return cent

    def update_centroid(self, cluster):
        pts = np.where(self.labels == cluster)
        pts = pts[0]
        pts = self.X[pts, :]

        if len(pts) > 0:
            l = len(pts)
            cent = np.sum(pts, axis=0)
            cent = cent / l
        else:
            cent = self.deal_with_empty_cluster(cluster)

        return cent

    def kmeans(self):
        self.iterations = 0
        self.labels = None

        self.centroids = self.init_centroids()
        initial_centroids = self.centroids

        self.labels = np.full(self.n_points, fill_value=-1)
        old_labels = np.full(self.n_points, fill_value=-1)

        for i in range(self.max_iterations):

            for point in range(self.n_points):
                distances = self.compute_distances(point)
                self.labels[point] = self.assign_labels(distances)

            clusters_to_remove = []
            for c in range(self.k):
                cent = self.update_centroid(c)
                if cent is None:
                    clusters_to_remove.append(c)
                else:
                    self.centroids[c] = cent

            clusters_to_remove.sort(reverse=True)
            for c in clusters_to_remove:
                self.centroids = np.delete(self.centroids, c, 0)
                self.k -= 1

            if np.array_equal(self.labels, old_labels):
                break
            else:
                old_labels = np.copy(self.labels)

            self.iterations += 1

        return self.iterations, initial_centroids, self.centroids, self.labels

    def fit(self, X: np.array):
        self.X = X
        self.n_points = X.shape[0]
        self.n_features = X.shape[1]

        self.results = np.zeros((self.n_points, self.n_runs), dtype=int)
        self.final_labels = np.full(self.n_points, fill_value=-1)

        threshold = math.floor(self.n_runs / self.k)

        for r in range(self.n_runs):
            it, initial_centroids, centroids, labels = self.kmeans()
            for l in range(len(labels)):
                self.results[l][r] = labels[l]

        for i in range(len(self.labels)):
            runs_i = Counter(self.results[i, :])
            for k, v in runs_i.items():
                if v > threshold:
                    self.final_labels[i] = k

        return self.final_labels, initial_centroids