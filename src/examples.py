import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import homogeneity_score, silhouette_score
from sklearn.utils import check_random_state

from clustering.kernel_kmeans import KernelKMeans
from clustering.kmeans import Kmeans
from transformations.kernel_pca import KernelPCA
from transformations.pca import PCA


def PCA_comparison():

    n_components = 2
    svd_solver = 'full'
    eig_solver = 'arpack'
    kernel = 'rbf'
    random_state = check_random_state(seed=5)
    n_samples = 1000

    # moons, circles, blobs
    fig = plt.figure(figsize=(12, 12))

    # moons
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.set_title('Original Moons Data')
    X, y = make_moons(n_samples=n_samples, shuffle=True, noise=0.05, random_state=random_state)
    reg = LogisticRegression().fit(X, y)
    score = reg.score(X, y)
    print(f'Original Moons Data: {score}')
    ax1.scatter(X[:, 0], X[:, 1], marker="o", c=y, s=25, edgecolor="k")

    ax2 = fig.add_subplot(3, 3, 2)
    ax2.set_title('Moons Data after classic PCA')
    pca = PCA(n_components=n_components, svd_solver=svd_solver)
    new_x = pca.apply_transformation(X)
    reg = LogisticRegression().fit(new_x, y)
    score = reg.score(new_x, y)
    print(f'Moons Data after classic PCA: {score}')
    ax2.scatter(new_x[:, 0], new_x[:, 1], marker="o", c=y, s=25, edgecolor="k")

    ax3 = fig.add_subplot(3, 3, 3)
    ax3.set_title('Moons Data after Kernel-PCA')
    gamma = 17.5
    pca = KernelPCA(n_components=n_components, eig_solver=eig_solver, kernel=kernel, gamma=gamma)
    new_x = pca.apply_transformation(X)
    reg = LogisticRegression().fit(new_x, y)
    score = reg.score(new_x, y)
    print(f'Moons Data after Kernel-PCA with gamma={gamma}: {score}')
    ax3.scatter(new_x[:, 0], new_x[:, 1], marker="o", c=y, s=25, edgecolor="k")

    print()

    # circles
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.set_title('Original Circles Data')
    X, y = make_circles(n_samples=n_samples, factor=0.2, noise=0.05, shuffle=True, random_state=random_state)
    reg = LogisticRegression().fit(X, y)
    score = reg.score(X, y)
    print(f'Original Circles Data: {score}')
    ax4.scatter(X[:, 0], X[:, 1], marker="o", c=y, s=25, edgecolor="k")

    ax5 = fig.add_subplot(3, 3, 5)
    ax5.set_title('Circles Data after classic PCA')
    pca = PCA(n_components=n_components, svd_solver=svd_solver)
    new_x = pca.apply_transformation(X)
    reg = LogisticRegression().fit(new_x, y)
    score = reg.score(new_x, y)
    print(f'Circles Data after classic PCA: {score}')
    ax5.scatter(new_x[:, 0], new_x[:, 1], marker="o", c=y, s=25, edgecolor="k")

    ax6 = fig.add_subplot(3, 3, 6)
    ax6.set_title('Circles Data after Kernel-PCA')
    gamma = 1.0
    pca = KernelPCA(n_components=n_components, eig_solver=eig_solver, kernel=kernel, gamma=gamma)
    new_x = pca.apply_transformation(X)
    reg = LogisticRegression().fit(new_x, y)
    score = reg.score(new_x, y)
    print(f'Circles Data after Kernel-PCA with gamma={gamma}: {score}')
    ax6.scatter(new_x[:, 0], new_x[:, 1], marker="o", c=y, s=25, edgecolor="k")

    print()

    # blobs
    ax7 = fig.add_subplot(3, 3, 7)
    ax7.set_title('Original Blobs Data')
    X, y = make_blobs(n_samples=n_samples, cluster_std=[5.0, 5.0, 5.0], random_state=random_state)
    reg = LogisticRegression().fit(X, y)
    score = reg.score(X, y)
    print(f'Original Blobs Data: {score}')
    ax7.scatter(X[:, 0], X[:, 1], marker="o", c=y, s=25, edgecolor="k")

    ax8 = fig.add_subplot(3, 3, 8)
    ax8.set_title('Blobs Data after classic PCA')
    pca = PCA(n_components=n_components, svd_solver=svd_solver)
    new_x = pca.apply_transformation(X)
    reg = LogisticRegression().fit(new_x, y)
    score = reg.score(new_x, y)
    print(f'Blobs Data after classic PCA: {score}')
    ax8.scatter(new_x[:, 0], new_x[:, 1], marker="o", c=y, s=25, edgecolor="k")

    ax9 = fig.add_subplot(3, 3, 9)
    ax9.set_title('Blobs Data after Kernel-PCA')
    gamma = 0.007
    pca = KernelPCA(n_components=n_components, eig_solver=eig_solver, kernel=kernel, gamma=gamma)
    new_x = pca.apply_transformation(X)
    reg = LogisticRegression().fit(new_x, y)
    score = reg.score(new_x, y)
    print(f'Blobs Data after Kernel-PCA with gamma={gamma}: {score}')
    ax9.scatter(new_x[:, 0], new_x[:, 1], marker="o", c=y, s=25, edgecolor="k")

    print()

    plt.tight_layout()
    plt.show()

def KMeans_comparison():

    random_state = check_random_state(seed=5)

    n_samples = 1000
    n_runs = 101

    k = 2

    # moons, circles, blobs
    fig = plt.figure(figsize=(12, 12))

    # moons
    print('Moons Dataset')
    
    X, y = make_moons(n_samples=n_samples, shuffle=True, noise=0.05, random_state=random_state)
    kmeans = Kmeans(2, n_runs=n_runs, empty_cluster='add_random_point')
    labels, _ = kmeans.fit(X)
    
    print(f'Silhouette score: {silhouette_score(X, labels, random_state=random_state)}')
    print(f'Homogeneity score: {homogeneity_score(y, labels)}')

    ax1 = fig.add_subplot(3, 2, 1)
    ax1.set_title('Original Moons Data')
    ax1.scatter(X[:, 0], X[:, 1], marker="o", c=y, s=25, edgecolor="k")

    ax2 = fig.add_subplot(3, 2, 2)
    ax2.set_title('K-Means Moons Data clusters')
    ax2.scatter(X[:, 0], X[:, 1], marker="o", c=labels, s=25, edgecolor="k")

    print()

    # circles
    print('Circles Dataset')
    
    X, y = make_circles(n_samples=n_samples, factor=0.2, noise=0.05, shuffle=True, random_state=random_state)
    kmeans = Kmeans(2, n_runs=n_runs, empty_cluster='add_random_point')
    labels, _ = kmeans.fit(X)

    print(f'Silhouette score: {silhouette_score(X, labels, random_state=random_state)}')
    print(f'Homogeneity score: {homogeneity_score(y, labels)}')

    # circles
    ax4 = fig.add_subplot(3, 2, 3)
    ax4.set_title('Original Circles Data')
    ax4.scatter(X[:, 0], X[:, 1], marker="o", c=y, s=25, edgecolor="k")

    ax5 = fig.add_subplot(3, 2, 4)
    ax5.set_title('K-Means Circles Data clusters')
    ax5.scatter(X[:, 0], X[:, 1], marker="o", c=labels, s=25, edgecolor="k")

    print()

    # blobs
    print('Blobs dataset')

    X, y = make_blobs(n_samples=n_samples, cluster_std=[5.0, 5.0, 5.0], random_state=random_state)
    kmeans = Kmeans(3, n_runs=n_runs, empty_cluster='add_random_point')
    labels, initial_centroids = kmeans.fit(X)
    
    print(f'Silhouette score: {silhouette_score(X, labels, random_state=random_state)}')
    print(f'Homogeneity score: {homogeneity_score(y, labels)}')

    # blobs
    ax7 = fig.add_subplot(3, 2, 5)
    ax7.set_title('Original Blobs Data')
    ax7.scatter(X[:, 0], X[:, 1], marker="o", c=y, s=25, edgecolor="k")

    ax8 = fig.add_subplot(3, 2, 6)
    ax8.set_title('K-Means Blobs Data clusters')
    ax8.scatter(X[:, 0], X[:, 1], marker="o", c=labels, s=25, edgecolor="k")

    plt.show()

def Kernel_KMeans_comparison(dataset='Moons'):

    random_state = check_random_state(seed=5)

    n_clusters = 2
    kernel = 'rbf'

    n_points = 100
    n_runs = 101
    max_iter = 1000

    if dataset == 'Moons':
        X, y = make_moons(n_samples=n_points, shuffle=True, noise=0.05, random_state=random_state)
    elif dataset == 'Circles':
        X, y = make_circles(n_samples=n_points, factor=0.2, noise=0.05, shuffle=True, random_state=random_state)

    kernelKMeans = KernelKMeans(n_clusters=n_clusters, kernel=kernel, n_runs=n_runs, max_iterations=max_iter, gamma=5)
    labels, _ = kernelKMeans.fit(X)
    print(f'{dataset} Dataset')
    print(f'Silhouette score: {silhouette_score(X, labels, random_state=random_state)}')
    print(f'Homogeneity score: {homogeneity_score(y, labels)}')

    fig = plt.figure(figsize=(15, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title('Original ' + str(dataset) + ' Data')
    ax1.scatter(X[:, 0], X[:, 1], marker="o", c=y, s=25, edgecolor="k")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title('Kernel K-Means ' + str(dataset) + ' Data clusters')
    ax2.scatter(X[:, 0], X[:, 1], marker="o", c=labels, s=25, edgecolor="k")

    plt.show()


if __name__ == "__main__":
    # PCA_comparison()
    # KMeans_comparison()
    Kernel_KMeans_comparison()