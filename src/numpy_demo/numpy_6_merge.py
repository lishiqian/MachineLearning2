import numpy as np

A = np.array([1, 1, 1])
B = np.array([2, 3, 4])

# 上下合并
C = np.vstack((A, B))
print("C:", C)

# 左右合并
D = np.hstack((A, B))
print('D:', D)

# A 序列转置 A[np.newaxis, :] 行上加一个维度
print(A.shape)
print(A[np.newaxis, :].shape)
# A[np.newaxis, :] 列上加一个维度
print(A[:, np.newaxis].shape)

A = np.array([1, 1, 1])[:, np.newaxis]
B = np.array([2, 3, 4])[:, np.newaxis]
# 多个array合并
C = np.concatenate((A, B, B, A))
print("np.concatenate(A, B, B, A)", C)

C = np.concatenate((A, B, B, A), axis=0)
print("np.concatenate((A, B, B, A), axis=0)", C)

D = np.concatenate((A, B, B, A), axis=1)
print("np.concatenate((A, B, B, A), axis=1)", D)
