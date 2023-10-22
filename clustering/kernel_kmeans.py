import copy
import math
from collections import Counter
from itertools import combinations, permutations
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np


class KernelKMeans:
    def __init__(self, n_clusters: int, max_iterations: int = 1000, n_runs:int = 51, kernel: Literal['linear', 'poly', 'rbf'] = 'linear',  gamma: float = None, coef0: float = 1.0, degree: int = 3, centroids: np.array = None):

        self.k = n_clusters
        self.iterations = 0
        self.max_iterations = max_iterations
        self.n_runs = n_runs
        self.centroids = centroids
        self.results = None
        self.final_labels = None

        if kernel == 'linear':
            self.kernel = self.linear_kernel
        elif kernel == 'poly':
            self.kernel = self.poly
        elif kernel == 'rbf':
            self.kernel = self.rbf

        self.X = None
        self.n_points = None
        self.n_features = None
        self.clusters = None

        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree

    def linear_kernel(self, x1, x2):
        k = x1.dot(x2.T)
        return k

    def poly(self, x1, x2):

        # poly = (gamma · <x,y> + coef0)^degree

        if self.gamma is None:
            self.gamma = 1 / self.X.shape[1]

        product = x1.dot(x2.T)

        k = ((self.gamma * product) + self.coef0) ** self.degree

        return k

    def rbf(self, x1, x2):

        if self.gamma is None:
            self.gamma = 1 / self.X.shape[1]

        dist = np.linalg.norm(x1 - x2)
        k = np.exp(-self.gamma * (dist ** 2))

        return k

    def get_random_cluster_assignments(self):
        clusters = {c: [] for c in range(self.k)}
        valid = False

        while not valid:
            for point in range(self.n_points):
                c = np.random.choice(self.k)
                clusters[c].append(point)

            assignments = list(clusters.values())
            valid = True
            for cl in assignments:
                if len(cl) == 0:
                    valid = False

        return clusters

    def compare_clusters(self, cluster_1, cluster_2):
        cl_1 = dict(zip(list(range(self.n_points)), [-1] * self.n_points))
        cl_2 = dict(zip(list(range(self.n_points)), [-1] * self.n_points))
        for cluster, points in cluster_1.items():
            for point in points:
                cl_1[point] = cluster
        for cluster, points in cluster_2.items():
            for point in points:
                cl_2[point] = cluster

        return np.array_equal(list(cl_1.values()), list(cl_2.values()))

    def get_labels(self):
        labels = [-1]*self.n_points
        for cluster, points in self.clusters.items():
            for point in points:
                labels[point] = cluster

        return labels

    def kernel_kmeans(self):
        # first: assign random cluster to every point
        self.clusters = self.get_random_cluster_assignments()
        initial_labels = self.get_labels()
        last_clusters = copy.deepcopy(self.clusters)

        for i in range(self.max_iterations):
            self.iterations += 1

            distance_array = np.zeros((self.n_points, self.k))

            for point in range(self.n_points):
                xi = self.X[point]
                old_c = -1
                for c, points in self.clusters.items():
                    if point in points:
                        old_c = c

                distances = self.k * [-1]
                for c in range(self.k):
                    cs = self.clusters[c]
                    sum_1 = 0
                    sum_2 = 0

                    for j in cs:
                        xj = self.X[j]
                        sum_1 += self.kernel(xi, xj)

                    point_pairs = []
                    point_pairs.extend(list(permutations(cs, 2)))
                    for p in cs:
                        point_pairs.append((p, p))
                    for j, l in point_pairs:
                        xj = self.X[j]
                        xl = self.X[l]
                        sum_2 += self.kernel(xj, xl)
                    dist = self.kernel(xi, xi) - (2 / len(cs)) * sum_1 + (1 / (len(cs) ** 2)) * sum_2

                    distance_array[point][c] = dist
                    distances[c] = dist

                new_c = np.argmin(distances)
                self.clusters[old_c].remove(point)
                self.clusters[new_c].append(point)

            if self.compare_clusters(self.clusters, last_clusters):
                break
            else:
                last_clusters = copy.deepcopy(self.clusters)

        return self.iterations, self.get_labels(), initial_labels

    def fit(self, X: np.array):
        self.X = X
        self.n_points = X.shape[0]
        self.n_features = X.shape[1]

        self.results = np.zeros((self.n_points, self.n_runs), dtype=int)
        self.final_labels = np.full(self.n_points, fill_value=-1)

        threshold = math.floor(self.n_runs / self.k)

        for r in range(self.n_runs):
            it, labels, initial_labels = self.kernel_kmeans()
            for l in range(len(labels)):
                self.results[l][r] = labels[l]

        for i in range(len(self.get_labels())):
            runs_i = Counter(self.results[i, :])
            for k, v in runs_i.items():
                if v > threshold:
                    self.final_labels[i] = k

        return self.final_labels, initial_labels