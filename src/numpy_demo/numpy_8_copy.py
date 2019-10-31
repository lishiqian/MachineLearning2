import numpy as np

a = np.arange(4)
print(a)
b = a
c = a
d = b
d[0] = 11
print(a)

print(b is a)
print(c is a)
print(d is a)

d[1:3] = [22, 33]
print(a)
print(b)
print(c)

e = a.copy()
e[3] = 44
print("e:", e)
print(a)
print(b)
print(c)
