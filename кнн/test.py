import numpy as np
np.random.seed(54)
train_data = np.random.randint(0, 30, size=(10, 4))
train_target = np.random.randint(1, 4, size=(1, 10))

print(np.sum(train_data, axis=0))