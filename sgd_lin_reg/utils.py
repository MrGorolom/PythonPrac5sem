import numpy as np


def gradient(f, x, eps):
    def partial_derivative(xi, i):
        nonlocal x
        nonlocal f
        nonlocal eps
        grad_i = -f(x)
        x[i] += eps
        grad_i += f(x)
        x[i] -= eps
        return grad_i / eps

    deriv = np.vectorize(partial_derivative)
    return deriv(x, np.array(range(x.shape[0])))

def f(x):
    return x@x.T

x = np.array([1, 1, 1])
print(gradient(f, x, 0.0001))