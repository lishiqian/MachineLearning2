import numpy as np

# numpy 基本逻辑运算
a = np.array([10, 20, 30, 40])  # [10 20 30 40]
b = np.arange(4)  # [0 1 2 3]
print("a = ", a, ",b = ", b)

c = a - b
print("a - b = ", c)

c = a + b
print("a + b = ", c)

c = a * b
print("a * b = ", c)

c = a ** 2
print("a^2 = ", c)

c = np.sin(a)
print("sin(a) = ", c)

c = b < 3
print("b < 3", c)

t1 = np.array([[1, 1], [0, 3]])
t2 = np.arange(4).reshape([2, 2])
print("t1", t1)
print("t2", t2)
print("逐个相乘 t1 * t2：", t1 * t2)
print("矩阵相乘 t1.dot(t2)：", t1.dot(t2))
print("矩阵相乘 np.dot(t1, t2)：", np.dot(t1, t2))

t3 = np.array([1, 2, 3, 4, 5, 6]).reshape([2, 3])
print("t3:", t3)
# 全部值相加的值
print("np.sum(t3):", np.sum(t3))
# 全部数据的最大值
print("np.max(t3):", np.max(t3))
# 全部值得平均值
print("np.average(t3):", np.average(t3))
# 每一列相加
print("np.sum(t3, axis=0):", np.sum(t3, axis=0))
# 每一行相加
print("np.sum(t3, axis=1):", np.sum(t3, axis=1))
