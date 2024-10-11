non_numpy_time = list()
semi_numpy_time = list()
full_numpy_time = list()



def timed_array(method, array):
    import time
    def __timed(*args, **kw):
        nonlo
        time_start = time.time()
        result = method(*args, **kw)
        time_end = time.time()
        array.append((time_end - time_start) * 1000)
        return result
    return __timed

for i in range(3):
    timed_array(calc_expectations_non_numpy(size_Oblast[i][0], size_Oblast[i][1], X_array[i], Q_array[i]), non_numpy_time)
print(non_numpy_time)