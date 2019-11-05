import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 100)
y1 = x * 2 + 1
y2 = x ** 2

plt.figure(num=3)
plt.plot(x, y2)
plt.plot(x, y1, color='red', linewidth=2.0, linestyle='--')
# x,y轴取值范围
plt.xlim(-1, 3)
plt.ylim(-3, 3)
# x,y轴标签
plt.xlabel("I am x")
plt.ylabel("I am y")
# 替换显示信息
plt.yticks([-2, -1, 0, 2, 3],
           [r'$really\ bad$', r'$bad\ \alpha$', 'normal', 'good', 'really good'])

# 'gca' = 'get current axis'
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
# 将X轴放在Y轴的1位置
ax.spines['bottom'].set_position(('data', 1))
# 将Y轴放在X轴的0位置
ax.spines['left'].set_position(('data', 0))

plt.show()
