import numpy as np

# pop1 = np.zeros((1, 10))
# pop2 = np.ones((1, 10))
# print(pop1)
# print(pop2)
#
# index = np.random.randint(0, 2, size=10).astype(np.bool)
# print(index)
#
# pop1[0, index] = pop2[0, index]
# print(pop1)

data = np.random.randint(0,101,size=10)
print(data)

index = np.random.choice(np.arange(10),size=10, p = data/np.sum(data))
print(index)