import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 100)
y = x * 2 + 1

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

plt.plot(x, y)

x0 = 1
y0 = 2 * x0 + 1

plt.scatter(x0, y0, s=50)
plt.plot([x0, x0], [y0, 0], 'k--')
plt.plot([x0, 0], [y0, y0], 'k--')

# method 1
plt.annotate(r'$2x+1=%s$' % y0, xy=(x0, y0), xycoords='data',
             xytext=(+30, -30), textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
# method 2
plt.text(-3.7, 3, r"$This\ is\ a\ text!\ \alpha_a\ \mu_t$", fontdict={'size': 16, 'color': 'r'})

plt.show()
