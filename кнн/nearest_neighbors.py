import distances as ds
import numpy as np

class KNNClassifier:
    def __init__(self, k = 1, strategy = 'my_own', metric = 'euclidean', weights = False, test_block_size = 10):
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
            self.y = y

    def find_kneighbors(self, X, return_distance = True):
        if self.strategy == 'my_own':
            KNN_distance = np.zeros((X.shape[0], self.k))
            KNN_index = np.zeros((X.shape[0], self.k))
            now_size = 0
            while now_size < self.X.shape[0]:
                e_distance = ds.euclidean_distance(X, self.X[now_size : now_size+self.test_block_size])
                e_index = np.argsort(e_distance, axis=1) #по возрастанию
                now_size += self.test_block_size



    def predict(self, X):
        pass

M = KNNClassifier()

np.random.seed(45)
train_data = np.random.randint(0, 10, size=(10, 4))
train_target = np.random.randint(1, 3, size=(1, 10))
print("Тренировочная выборка:")
print(train_data)
print(train_target)


test_data = np.random.randint(0, 10, size=(3, 4))
print("Тестовая выборка:")
print(test_data)
M.fit(train_data, train_target)
M.find_kneighbors(test_data)