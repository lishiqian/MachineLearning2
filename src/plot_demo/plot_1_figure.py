import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 100)
y1 = x * 2 + 1
y2 = x ** 2

plt.figure()
plt.plot(x, y1)

plt.figure(num=3)
plt.plot(x, y2)
plt.plot(x, y1, color='red', linewidth=2.0, linestyle='--')

plt.show()
