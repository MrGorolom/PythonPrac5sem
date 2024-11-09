from time import struct_time

import oracles
import time
import numpy as np
from scipy.special import expit


class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(
        self, loss_function, step_alpha=1, step_beta=0,
        tolerance=1e-5, max_iter=1000, **kwargs
    ):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия

        step_alpha - float, параметр выбора шага из текста задания

        step_beta- float, параметр выбора шага из текста задания

        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию.
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход

        max_iter - максимальное число итераций

        **kwargs - аргументы, необходимые для инициализации
        """
        if loss_function == 'binary_logistic':
            self.loss_function = oracles.BinaryLogistic(**kwargs)
        else:
            pass
        self.alpha = step_alpha
        self.beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.w = None

    def fit(self, X, y, w_0=None, trace=False):
        """
        Обучение метода по выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w_0 - начальное приближение в методе

        trace - переменная типа bool

        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)

        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
        """
        if w_0 is None:
            w_0 == np.zeros_like(y)
        if trace:
            history = dict()
            history['time'] = list()
            history['func'] = list()
            history['time'].append(0)
            prev_Q = self.loss_function.func(X, y, w_0)
            history['func'].append(prev_Q)
            prev_Q = 0
            for k in range(self.max_iter):
                start_time = time.time()
                dQ = self.loss_function.grad(X, y, w_0)
                eta = self.alpha / k ** self.beta
                w_0 = w_0 - eta * dQ
                Q = self.loss_function.func(X, y, w_0)
                end_time = time.time()
                history['time'].append(end_time - start_time)
                history['func'].append(float(Q))
                if abs(prev_Q - Q) < self.tolerance:
                    break
                else:
                    prev_Q = Q
            return history
        else:
            prev_Q = self.loss_function.func(X, y, w_0)
            prev_Q = 0
            for k in range(self.max_iter):
                dQ = self.loss_function.grad(X, y, w_0)
                eta = self.alpha / k ** self.beta
                w_0 = w_0 - eta * dQ
                Q = self.loss_function.func(X, y, w_0)
                if abs(prev_Q - Q) < self.tolerance:
                    break
                else:
                    prev_Q = Q
        self.w = w_0
        return

    def predict(self, X):
        """
        Получение меток ответов на выборке X

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        return: одномерный numpy array с предсказаниями
        """
        return np.sign(X @ self.w)

    def predict_proba(self, X):
        """
        Получение вероятностей принадлежности X к классу k

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        return: двумерной numpy array, [i, k] значение соответветствует вероятности
        принадлежности i-го объекта к классу k
        """
        return expit(X @ self.w)

    def get_objective(self, X, y):
        """
        Получение значения целевой функции на выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array

        return: float
        """
        return self.loss_function.func(X, y, self.w)

    def get_gradient(self, X, y):
        """
        Получение значения градиента функции на выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array

        return: numpy array, размерность зависит от задачи
        """
        return self.loss_function.grad(X, y, self.w)

    def get_weights(self):
        """
        Получение значения весов функционала
        """
        return self.w


class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(
        self, loss_function, batch_size, step_alpha=1, step_beta=0,
        tolerance=1e-5, max_iter=1000, random_seed=153, **kwargs
    ):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия

        batch_size - размер подвыборки, по которой считается градиент

        step_alpha - float, параметр выбора шага из текста задания

        step_beta- float, параметр выбора шага из текста задания

        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход

        max_iter - максимальное число итераций (эпох)

        random_seed - в начале метода fit необходимо вызвать np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости результатов на разных машинах.

        **kwargs - аргументы, необходимые для инициализации
        """
        if loss_function == 'binary_logistic':
            self.loss_function = oracles.BinaryLogistic(**kwargs)
        else:
            pass
        self.alpha = step_alpha
        self.beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.w = None
        self.batch_size = batch_size
        np.random.seed(random_seed)

    def fit(self, X, y, w_0=None, trace=False, log_freq=1):
        """
        Обучение метода по выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w_0 - начальное приближение в методе

        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Если обновлять history после каждой итерации, метод перестанет
        превосходить в скорости метод GD. Поэтому, необходимо обновлять историю метода лишь
        после некоторого числа обработанных объектов в зависимости от приближённого номера эпохи.
        Приближённый номер эпохи:
            {количество объектов, обработанных методом SGD} / {количество объектов в выборке}

        log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления.
        Обновление должно проиходить каждый раз, когда разница между двумя значениями приближённого номера эпохи
        будет превосходить log_freq.

        history['epoch_num']: list of floats, в каждом элементе списка будет записан приближённый номер эпохи:
        history['time']: list of floats, содержит интервалы времени между двумя соседними замерами
        history['func']: list of floats, содержит значения функции после текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат нормы разности векторов весов с соседних замеров
        (0 для самой первой точки)
        """
        if w_0 is None:
            w_0 == np.zeros_like(y)
        if trace:
            history = dict()
            history['time'] = list()
            history['func'] = list()
            history['time'].append(0)
            prev_Q = self.loss_function.func(X, y, w_0)
            history['func'].append(prev_Q)
            prev_Q = 0

            start_time = time.time()
            curr_count_x = 0
            for k in range(self.max_iter):

                np.random.shuffle(X)
                for i in range(0, X.shape[0], self.batch_size):
                    X_batch = X[i : i+self.batch_size]
                    y_batch = y[i : i+self.batch_size]
                    dQ = self.loss_function.grad(X_batch, y_batch, w_0)
                    eta = self.alpha / k ** self.beta
                    w_0 = w_0 - eta * dQ
                    Q = self.loss_function.func(X_batch, y_batch, w_0)
                    curr_count_x += y_batch.shape[0]
                    if curr_count_x / X.shape[0] > log_freq:
                        end_time = time.time()
                        history['time'].append(end_time - start_time)
                        history['func'].append(float(Q))
                        struct_time = time.time()
                    if abs(prev_Q - Q) < self.tolerance:
                        break
                    else:
                        prev_Q = Q
            return history
        else:
            prev_Q = 0
            for k in range(self.max_iter):
                np.random.shuffle(X)
                for i in range(0, X.shape[0], self.batch_size):
                    dQ = self.loss_function.grad(X, y, w_0)
                    eta = self.alpha / k ** self.beta
                    w_0 = w_0 - eta * dQ
                    Q = self.loss_function.func(X, y, w_0)
                    if abs(prev_Q - Q) < self.tolerance:
                        break
                    else:
                        prev_Q = Q
                if abs(prev_Q - Q) < self.tolerance:
                        break
        self.w = w_0
        return



model = GDClassifier(loss_function='binary_logistic', step_alpha=1,
    step_beta=0, tolerance=1e-4, max_iter=5, l2_coef=0.1)
l, d = 1000, 10
X = np.random.random((l, d))
y = np.random.randint(0, 2, l) * 2 - 1
w = np.random.random(d)
history = model.fit(X, y, w_0=np.zeros(d), trace=True)
print(' '.join([str(x) for x in history['func']]))