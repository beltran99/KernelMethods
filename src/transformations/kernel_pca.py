import numpy as np

from typing import Literal

from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from sklearn.utils.extmath import randomized_svd


class KernelPCA:
    def __init__(self, n_components: int, eig_solver: Literal['dense', 'arpack'] = 'dense', svd_flip: bool = True, kernel: Literal['linear', 'poly', 'rbf'] = 'linear', gamma: float = None, coef0: float = 1.0, degree: int = 3):
        self.n_components = n_components
        self.eig_solver = eig_solver
        self.svd_flip = svd_flip
        self.kernel = kernel

        self.X = None
        self.n_points = None
        self.n_features = None

        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree

    def poly(self):
        # poly = (gamma · <x,y> + coef0)^degree

        if self.gamma is None:
            self.gamma = 1 / self.X.shape[1]

        product = self.X.dot(self.X.T)
        k = ((self.gamma * product) + self.coef0) ** self.degree

        return k

    def rbf(self):
        # rbf = exp(-gamma * d(xi,xj)**2)

        if self.gamma is None:
            self.gamma = 1 / self.X.shape[1]

        k = np.zeros((self.X.shape[0], self.X.shape[0]))
        for i in range(k.shape[0]):
            for j in range(k.shape[1]):
                dist = np.linalg.norm(self.X[i, :] - self.X[j, :])
                k[i][j] = np.exp(-self.gamma * (dist ** 2))

        return k

    def apply_transformation(self, x: np.array):

        self.X = x
        self.n_points = x.shape[0]
        self.n_features = x.shape[1]

        gram = None

        # mean = np.mean(x, axis=0)
        # x = x - mean

        self.X = np.array(self.X, dtype=np.float64)

        if self.kernel == 'linear':
            gram = self.X.dot(self.X.T)
        elif self.kernel == 'poly':
            gram = self.poly()
        elif self.kernel == 'rbf':
            gram = self.rbf()

        gram = np.array(gram, dtype=np.float64)
        one_n = np.full(shape=(self.n_points, self.n_points), fill_value=(1. / float(self.n_points)), dtype=np.float64)

        gram_centered = gram - np.dot(one_n, gram) - np.dot(gram, one_n) + np.dot(np.dot(one_n, gram), one_n)

        if self.eig_solver == 'dense':
            w, v = eigh(gram_centered, eigvals=(gram_centered.shape[0] - self.n_components, gram_centered.shape[0] - 1))
            order = w.argsort()[::-1]
            w = w[order]
            v = v[:, order]

        if self.eig_solver == 'arpack':
            w, v = eigsh(gram_centered, self.n_components)
            order = w.argsort()[::-1]
            w = w[order]
            v = v[:, order]

        if self.svd_flip:
            max_abs_columns = np.argmax(np.abs(v), axis=0)
            # signs = np.sign(u[max_abs_columns])
            max_values = []
            col = 0
            for row in max_abs_columns:
                max_values.append(v[row, col])
                col += 1
            signs = np.sign(np.array(max_values))
            signs = np.reshape(signs, (1, -1))
            v = np.multiply(v, signs)
            # w = np.multiply(w, signs[:, :self.n_components])

        return np.sqrt(w) * v
