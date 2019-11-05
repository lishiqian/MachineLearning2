import matplotlib.pyplot as plt
'''

plt.figure()
# 表示将图片分成两行两列的第一个图片
plt.subplot(2, 2, 1)
plt.plot([0, 1], [0, 1])
# 表示将图片分成两行两列的第一个图片
plt.subplot(2, 2, 2)
plt.plot([0, 1], [0, 2])
# 表示将图片分成两行两列的第一个图片
plt.subplot(2, 2, 3)
plt.plot([0, 1], [0, 3])
# 表示将图片分成两行两列的第一个图片
plt.subplot(2, 2, 4)
plt.plot([0, 1], [0, 4])

plt.show()
'''

plt.figure()
# 横跨三列
plt.subplot(2, 1, 1)
plt.plot([0, 1], [0, 1])

plt.subplot(2, 3, 4)
plt.plot([0, 1], [0, 2])

plt.subplot(2, 3, 5)
plt.plot([0, 1], [0, 3])

plt.subplot(2, 3, 6)
plt.plot([0, 1], [0, 4])

plt.show()
