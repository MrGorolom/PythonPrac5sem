import numpy as np

# Исходный массив
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Новый столбец
new_column = np.array([10, 11])

# Позиция для вставки (например, после второго столбца)
position = 2

# Вставка столбца
result = np.insert(arr, position, values=new_column, axis=1)

print(result)