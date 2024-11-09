import numpy as np


def grad_finite_diff(function, w, eps=1e-8):
    f = np.full_like(w, function(w), dtype = float)
    eps_matrix_e = np.eye(w.shape[0]) * eps
    W_matrix_merg = np.tile(w, w.shape[0]).reshape(w.shape[0], -1) + np.eye(w.shape[0]) * eps
    f_merg = np.apply_along_axis(lambda w: function(w), axis=1, arr=W_matrix_merg)
    return (f_merg - f) / eps


