import numpy as np
from nearest_neighbors import KNNClassifier


def kfold(n, n_folds, random_state=0):
    np.int = int
    curr_train = list()
    curr_valid = list()
    result = list()
    if random_state == 0:
        random_index = np.array(range(0, n))
    else:
        np.random.seed(random_state)
        random_index = np.random.randint(low=0, high=n, size=(n,))
    for i in range(0, n - n % n_folds, n // n_folds):
        if n - i < n//n_folds * 2:
            curr_valid = random_index[i:]
            curr_train = random_index[:i]
        else:
            curr_valid = random_index[i:i+n//n_folds]
            curr_train = np.concatenate((random_index[:i], random_index[i+n//n_folds:]))
        result.append((curr_train, curr_valid))
    return result


def knn_cross_val_score(X, y, k_list, score='accuracy', cv=None, **kwargs):
    result = dict()
    if cv is None:
        cv = kfold(len(y), 10)
    for k in k_list:
        accuracy = list()
        for train_index, val_index in cv:
            Model = KNNClassifier(k, **kwargs)
            Model.fit(X[np.ix_(train_index)], y[np.ix_(train_index)])
            model_y = Model.predict(X[np.ix_(val_index)])
            right_y = y[np.ix_(val_index)]
            accuracy_for_one = right_y == model_y
            accuracy.append(float(np.sum(accuracy_for_one) /
                                  accuracy_for_one.size))
        result[k] = accuracy
    return result
