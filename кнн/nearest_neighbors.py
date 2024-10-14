from distances import euclidean_distance, cosine_distance
import numpy as np
from sklearn.neighbors import NearestNeighbors


class KNNClassifier:
    def __init__(self, k=1, strategy='my_own', metric='euclidean',
                 weights=False, test_block_size=3):
        self.k = k
        self.strategy = strategy
        self.metric = metric
        self.weights = weights
        self.test_block_size = test_block_size
        if strategy == 'kd_tree' or strategy == 'ball_tree':
            if metric != 'euclidean':
                raise TypeError

    def fit(self, X, y):
        if self.strategy == 'my_own':
            self.X = X
        else:
            self.model = NearestNeighbors(n_neighbors=self.k, algorithm=self.strategy, metric=self.metric)
            self.model.fit(X, y)
        self.y = y

    def find_kneighbors(self, X, return_distance=True):
        KNN_distance, KNN_index = 0, 0
        if self.strategy == 'my_own':
            KNN_distance = np.zeros((X.shape[0], self.k))
            KNN_index = np.zeros((X.shape[0], self.k), dtype=int)
            for i in range(0, X.shape[0], self.test_block_size):
                if self.metric == 'euclidean':
                    all_dist = euclidean_distance(
                        X[i:i+self.test_block_size], self.X)
                else:
                    all_dist = cosine_distance(
                        X[i:i + self.test_block_size], self.X)
                all_dist_sort_index = np.argsort(
                    all_dist, axis=1)
                KNN_index[i:i+self.test_block_size, 0:self.X.shape[0]] = all_dist_sort_index[:, :self.k]
                KNN_distance[i:i + self.test_block_size, 0:self.X.shape[0]] = np.take_along_axis(
                    all_dist, all_dist_sort_index, axis=1
                )[:, :self.k]
        elif self.strategy == 'brute':
            KNN_distance = np.zeros((X.shape[0], self.k))
            KNN_index = np.zeros((X.shape[0], self.k), dtype=int)
            for i in range(0, X.shape[0], self.test_block_size):
                dist, index = self.model.kneighbors(
                    X[i:i+self.test_block_size], n_neighbors=self.k)
                KNN_index[i:i + self.test_block_size] = index[:, :self.k]
                KNN_distance[i:i + self.test_block_size] = dist[:, :self.k]
        else:
            KNN_distance, KNN_index = self.model.kneighbors(
                        X, n_neighbors=self.k)
        if return_distance:
            return KNN_distance, KNN_index
        else:
            return KNN_index

    def predict(self, X):

        def max_count_val(row):
            vals, counts = np.unique_counts(row)
            return vals[np.argmax(counts)]

        if not self.weights:
            KNN_index = self.find_kneighbors(X, False)
            if X.ndim == 2 and KNN_index.ndim == 2:
                KNN_classes = np.take_along_axis(
                    np.tile(self.y, X.shape[0]).reshape(X.shape[0], - 1), KNN_index, axis=1)
            else:
                KNN_classes = np.take(
                    self.y, KNN_index)
            prediction = np.apply_along_axis(max_count_val, axis=1, arr=KNN_classes)
            return prediction
        else:
            eps = 10 ** -5
            KNN_distance, KNN_index = self.find_kneighbors(X)
            KNN_classes = np.take_along_axis(
                np.tile(self.y, X.shape[0]).reshape(X.shape[0], - 1), KNN_index, axis=1)
            KNN_weights = np.ones_like(KNN_distance) / (KNN_distance + eps)
            vals = np.unique(KNN_classes, return_counts=False)
            class_weight_sums = np.zeros((X.shape[0], len(vals)))
            for i, cls in enumerate(vals):
                mask = (KNN_classes == cls)
                class_weight_sums[:, i] = np.sum(KNN_weights * mask, axis=1)
            max_class_indices = np.argmax(class_weight_sums, axis=1)
            prediction = vals[max_class_indices]
            return prediction
