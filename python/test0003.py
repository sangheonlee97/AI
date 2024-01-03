import numpy as np

a = np.array(range(10))

print(a)

b = a[:-3]

print(b)

b[:3] = 3

print(b)

print(a)