import numpy as np
import matplotlib.pyplot as plt

n = 1024
# 随机生成中位数位0，方差为1的1024个数据
X = np.random.normal(0, 1, 1024)
Y = np.random.normal(0, 1, 1024)
T = np.arctan2(Y, X)

plt.scatter(X, Y, c=T, alpha=0.7)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)

# 隐藏x,y轴的数值标志
plt.xticks(())
plt.yticks(())

plt.show()
