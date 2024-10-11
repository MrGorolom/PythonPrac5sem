import numpy as np


def euclidean_distance(X, Y):
    X_sq = np.sum(X * X, axis=1).reshape(-1, 1)
    Y_sq = np.sum(Y * Y, axis=1).reshape(1, -1)
    XY_scal = X @ Y.T
    return np.sqrt(X_sq + Y_sq - 2 * XY_scal)


def cosine_distance(X, Y):
    X_sq = np.sum(X * X, axis=1).reshape(-1, 1)
    Y_sq = np.sum(Y * Y, axis=1).reshape(1, -1)
    XY_norm = np.sqrt(X_sq * Y_sq)
    XY_scal = X @ Y.T
    E = np.ones_like(XY_scal)
    return E - XY_scal/XY_norm
