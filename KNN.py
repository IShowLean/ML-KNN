import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
import numpy as np

class CustomKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5, metric='euclidean', kernel='uniform',
                 window_type='fixed', window_size=1.0, p=2, a=2, b=1):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.kernel = kernel
        self.window_type = window_type
        self.window_size = window_size
        self.p = p
        self.a = a
        self.b = b
        self.model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)

    def fit(self, X, y, sample_weight=None):
        self.X_train = X
        self.y_train = y
        self.sample_weight = sample_weight if sample_weight is not None else np.ones(len(y))
        self.classes_ = np.unique(y)
        self.model.fit(X)
        return self

    def predict(self, X):
        neighbors_distances, neighbors_indices = self.model.kneighbors(X)
        predictions = []

        for i in range(X.shape[0]):
            indices = neighbors_indices[i]
            distances = neighbors_distances[i]
            classes = self.y_train[indices]
            weights = self.sample_weight[indices]

            if self.window_type == 'variable':
                window = distances[-1] if distances[-1] != 0 else 1e-5
            else:
                window = self.window_size

            u = distances / window
            kernel_weights = self._apply_kernel(u)
            total_weights = kernel_weights * weights

            class_votes = {}
            for cls, w in zip(classes, total_weights):
                class_votes[cls] = class_votes.get(cls, 0) + w
            predicted_class = max(class_votes, key=class_votes.get)
            predictions.append(predicted_class)

        return np.array(predictions)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def get_params(self, deep=True):
        return {
            'n_neighbors': self.n_neighbors,
            'metric': self.metric,
            'kernel': self.kernel,
            'window_type': self.window_type,
            'window_size': self.window_size,
            'p': self.p,
            'a': self.a,
            'b': self.b
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        self.model = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric)
        return self

    def _apply_kernel(self, u):
        if self.kernel == 'uniform':
            return uniform_kernel(u)
        elif self.kernel == 'triangular':
            return triangular_kernel(u)
        elif self.kernel == 'gaussian':
            return gaussian_kernel(u)
        elif self.kernel == 'general':
            return general_kernel(u, a=2, b=1)
        else:
            return np.ones_like(u)

def euclidean_distance(X1, X2):
    return np.linalg.norm(X1[:, np.newaxis] - X2, axis=2)

def minkowski_distance(X1, X2, p=2):
    return np.sum(np.abs(X1[:, np.newaxis] - X2) ** p, axis=2) ** (1 / p)

def cosine_distance(X1, X2):
    X1_norm = np.linalg.norm(X1, axis=1)
    X2_norm = np.linalg.norm(X2, axis=1)
    dot_product = np.dot(X1, X2.T)
    return 1 - (dot_product / (X1_norm[:, np.newaxis] * X2_norm + 1e-10))

def manhattan_distance(X1, X2):
    return np.sum(np.abs(X1[:, np.newaxis] - X2), axis=2)

def uniform_kernel(u):
    return np.where(np.abs(u) < 1, 0.5, 0)

def gaussian_kernel(u):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u ** 2)

def general_kernel(u, a=2, b=1):
    return np.where(np.abs(u) < 1, (1 - np.abs(u) ** a) ** b, 0)

def triangular_kernel(u):
    return np.where(np.abs(u) < 1, 1 - np.abs(u), 0)