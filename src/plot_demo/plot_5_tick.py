import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 100)
y = x * 0.1

plt.figure()
# 'gca' = 'get current axis'
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
# 将X轴放在Y轴的1位置
ax.spines['bottom'].set_position(('data', 0))
# 将Y轴放在X轴的0位置
ax.spines['left'].set_position(('data', 0))
plt.xlim(-3, 3)
plt.ylim(-3, 3)

# zorder 设置plot在Z轴的坐标 zorder越大，图像显示在越上面
plt.plot(x, y, lw=10.0, zorder=1)

# zorder 设置plot在Z轴的坐标 zorder越大，图像显示在越上面
# alpha 透明度
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(12)
    label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.5, zorder=2))

plt.show()
