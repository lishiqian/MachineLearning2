import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 100)
y1 = x * 2 + 1
y2 = x ** 2

plt.figure(num=3)

# x,y轴取值范围
plt.xlim(-3, 3)
plt.ylim(-3, 5)
# x,y轴标签
plt.xlabel("I am x")
plt.ylabel("I am y")
# 替换显示信息
plt.yticks([-2, -1, 0, 2, 3],
           [r'$really\ bad$', r'$bad\ \alpha$', 'normal', 'good', 'really good'])

l1, = plt.plot(x, y2, label='y2 = x ** 2')
l2, = plt.plot(x, y1, color='red', linewidth=2.0, linestyle='--', label='y1 = x * 2 + 1')
# plt.legend()
plt.legend(handles=[l1, l2], labels=['aaa', 'bbb'], loc='best')

plt.show()
