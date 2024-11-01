from sklearn.base import BaseEstimator, ClassifierMixin
from KNN import CustomKNN
import numpy as np

class CustomKNNWrapper(BaseEstimator, ClassifierMixin):
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

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        
        self.model = CustomKNN(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            kernel=self.kernel,
            window_type=self.window_type,
            window_size=self.window_size,
            p=self.p,
            a=self.a,
            b=self.b
        )
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)