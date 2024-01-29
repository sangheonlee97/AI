import numpy as np

a = np.array(range(1,21))
size = 5

def split_X(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_X(a, size)
print(bbb)
print(bbb.shape)

X = bbb[:,:-1]
y = bbb[:, -1]
print(X, y)
print(X.shape, y.shape)