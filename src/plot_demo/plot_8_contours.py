import numpy as np
import matplotlib.pyplot as plt


# 根据想x，y计算高度
def f(x, y):
    return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)


n = 256
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)
X, Y = np.meshgrid(x, y)

# use plt.contourf to filling contours 高度绑定颜色
plt.contourf(X, Y, f(X, Y), 10, alpha=0.75, cmap=plt.cm.hot)

# 画等高线
C = plt.contour(X, Y, f(X, Y), 10, color='black', linewidth=0.5)

# 添加高度数字
plt.clabel(C, inline=True, fontsize=10)

plt.show()
