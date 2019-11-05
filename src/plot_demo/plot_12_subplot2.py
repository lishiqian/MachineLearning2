import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 方法1  plt.subplot2grid()
plt.figure()
ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=1)
ax1.plot([0, 1], [0, 1])
ax1.set_title('ax1_title')

ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)

ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)

ax4 = plt.subplot2grid((3, 3), (2, 0))
ax4.scatter([0, 1], [0, 1])

ax5 = plt.subplot2grid((3, 3), (2, 1))

# 方法2  GridSpec
plt.figure()
gs = gridspec.GridSpec(3, 3)
ax6 = plt.subplot(gs[0, :])
ax7 = plt.subplot(gs[1, :2])
ax8 = plt.subplot(gs[1:3, 2])
ax9 = plt.subplot(gs[2, 0])
ax10 = plt.subplot(gs[2, 1])

# 方法3 plt.subplots
f, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2, sharex=True, sharey=True)
ax11.plot([0, 1], [0, 1])

plt.show()
