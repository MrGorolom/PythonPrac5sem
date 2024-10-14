import numpy as np
np.random.seed(54)
train_data = np.random.randint(0, 30, size=(10, 4))
train_target = np.random.randint(1, 4, size=(10,))
k = 3
print(train_data)
print(train_target)
KNN_index = np.array([[1, 9, 3],
                      [3,4,6],
                      [6, 2,1],
                      [5,3,1]])
print( np.tile(train_target, 4).reshape(4, -1))
KNN_classes = np.take_along_axis(
                    np.tile(train_target, 4).reshape(4, -1), KNN_index, axis=1)
print(KNN_classes)