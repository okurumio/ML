import numpy as np


arr = np.arange(10)
arr[arr % 2 == 1] = 0
arr.reshape(2, -1)
# print(arr)

arr1 = np.arange(9).reshape(3, 3)

# print(arr1[:, [1, 0, 2]])  # 交换数组的列
# print(arr1[[1, 0, 2], :])  # 交换数组的行

# a = np.arange(1,8)
# print("a is:", a)
# print("a.prod is:", a.prod())
# print("a.cumprod is:", a.cumprod())
# print(np.ceil(a))