import numpy as np
import random
x = np.array([0, 1, 1, 2, 2, 2, 3, 4, 4, 4, 4, 4])
print("std: ",np.std(x))
print("mean: ",np.mean(x))
from sklearn.preprocessing import StandardScaler
x = x.reshape(-1, 1)
sc = StandardScaler()
x = sc.fit_transform(x)

print(np.unique(x, return_counts=True))