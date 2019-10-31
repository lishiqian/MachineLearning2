import numpy as np


'''
numpy基础运算
'''
A = np.arange(2, 14).reshape(3, 4)
print(A)
# array[[ 2  3  4  5]
#       [ 6  7  8  9]
#       [10 11 12 13]]

# 求最大值的索引
print("np.argmax(A):", np.argmax(A))
# 求最小值的索引
print("np.argmin(A):", np.argmin(A))

# 求平均值
print("np.mean(A):", np.mean(A))
print("np.average(A):", np.average(A))
# 对每一列求平均值
print("np.mean(A,axis=0):", np.mean(A, axis=0))
# 对每一行求平均值
print("np.mean(A,axis=1):", np.mean(A, axis=1))

# 求中位数
print("np.median(A):", np.median(A))

# 求当前项和下一项累加
print("np.cumsum(A):", np.cumsum(A))

# 求当前项和下一项累差
print("np.diff(A):", np.diff(A))

# 排序
B = np.arange(14, 2, -1).reshape((3, 4))
print(B)
print("np.sort(B):", np.sort(B))
# 装置
print("B.T:", B.T)
# 将全体值变为5到9之间
print(np.clip(A, 5, 9))
