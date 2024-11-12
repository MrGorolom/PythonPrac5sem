import numpy as np
from scipy.special import expit
from sgd_lin_reg.utils import grad_finite_diff


class BaseSmoothOracle:
    """
    Базовый класс для реализации оракулов.
    """
    def func(self, w):
        """
        Вычислить значение функции в точке w.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, w):
        """
        Вычислить значение градиента функции в точке w.
        """
        raise NotImplementedError('Grad oracle is not implemented.')


class BinaryLogistic(BaseSmoothOracle):
    """
    Оракул для задачи двухклассовой логистической регрессии.

    Оракул должен поддерживать l2 регуляризацию.
    """

    def __init__(self, l2_coef):
        """
        Задание параметров оракула.

        l2_coef - коэффициент l2 регуляризации
        """
        self.l2_coef = l2_coef

    def func(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w - одномерный numpy array
        """
        return (np.mean(np.logaddexp(0, -y * (X @ w)))
                + (self.l2_coef / 2) * (w @ w.T))

    def grad(self, X, y, w):
        """
        Вычислить градиент функционала в точке w на выборке X с ответами y.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w - одномерный numpy array
        """
        return (X.T*y * (expit(y * (X @ w)) - 1)).mean(axis=1) + self.l2_coef * w


'''X = np.array([[1, 2, 3], [1, 1, 1], [2, 3, 4]])
y = np.array([1, 1, 2])
w = np.array([1, 0, 1])
Log = BinaryLogistic(0.1)
print(Log.grad(X, y, w))
def part(w, x=X, y=y):
    return Log.func(x, y, w)
print(grad_finite_diff(part, w))'''