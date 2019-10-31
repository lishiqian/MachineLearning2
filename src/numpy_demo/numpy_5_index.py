import numpy as np

A = np.arange(3, 15).reshape(3, 4)
'''
A: [[ 3  4  5  6]
 [ 7  8  9 10]
 [11 12 13 14]]
'''
print("A:", A)

# 一维索引
print("A[1]:", A[1])  # [ 7  8  9 10]

# 二维索引
print("A[1][1]:", A[1][1])  # 8
print("A[1, 1]:", A[1, 1])  # 8

# 范围切片
print('A[1, 1:3]:', A[1, 1:3])

# 逐行打印
for row in A:
    print(row)

# 逐列打印
for column in A.T:
    print(column)

# 将矩阵展开
print(A.flatten())

for item in A.flat:
    print(item)
