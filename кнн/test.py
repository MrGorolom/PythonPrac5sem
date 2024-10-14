from cross_validation import knn_cross_val_score
import numpy as np
np.random.seed(54)
train_data = np.random.randint(0, 30, size=(100, 4))
train_target = np.random.randint(1, 4, size=(100,))
print(train_data)
print(train_target)
KNN_index = np.array([[1, 9, 3],
                      [3,4,6],
                      [6, 2,1],
                      [5,3,1]])
KNN_index2 = np.array([[1, 1, 3],
                      [3,4,6],
                      [6, 2,1],
                      [5,3,1]])
print(knn_cross_val_score(train_data, train_target, [1, 3, 5, 7], strategy='kd_tree'))