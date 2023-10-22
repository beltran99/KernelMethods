import numpy as np

from scipy.linalg import svd
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import randomized_svd

from typing import Literal


class PCA:
    def __init__(self, n_components: int, normalize: bool = True, svd_solver: Literal['full', 'arpack', 'randomized'] = 'full', svd_flip: bool = True):
        self.n_components = n_components
        self.normalize = normalize
        self.svd_solver = svd_solver
        self.svd_flip = svd_flip

    def apply_transformation(self, X: np.array):

        if self.svd_solver == 'arpack' and not (0 < self.n_components < min(X.shape)):
            raise ValueError("Passed array X is not of the right shape") 
            return

        if self.normalize:
            mean = np.mean(X, axis=0)
            std = np.mean(X, axis=0)
            X = X - mean
            X = X / std

        u = s = vh = []

        if self.svd_solver == 'full':
            u, s, vh = svd(X)

        if self.svd_solver == 'arpack':
            u, s, vh = svds(X, self.n_components)
            order = np.argsort(-s)
            vh = vh[order]

        if self.svd_solver == 'randomized':
            u, s, vh = randomized_svd(X, self.n_components)

        if self.svd_flip:
            max_abs_columns = np.argmax(np.abs(u), axis=0)
            max_values = []
            col = 0
            for row in max_abs_columns:
                max_values.append(u[row, col])
                col += 1
            signs = np.sign(np.array(max_values))
            signs = np.reshape(signs, (-1, 1))
            u = np.multiply(u, signs)
            vh = np.multiply(vh[:self.n_components, :], signs[:self.n_components, :])

        pca_components = vh[:self.n_components, :]
        self.pca_components = pca_components

        new_X = np.matmul(X, np.transpose(pca_components))

        return new_X