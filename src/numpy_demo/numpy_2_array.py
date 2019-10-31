import numpy as np

a = np.array([2, 23, 4])
print(a)
print(a[0])

#  指定数据类型
a = np.array([2, 23, 4], dtype=np.int)
print(a.dtype)  # int32

a = np.array([2, 23, 4], dtype=np.int64)
print(a.dtype)  # int64

a = np.array([2, 23, 4], dtype=np.int32)
print(a.dtype)  # int32

a = np.array([2, 23, 4], dtype=np.float)
print(a.dtype)  # float62

a = np.array([2, 23, 4], dtype=np.float32)
print(a.dtype)  # float32


