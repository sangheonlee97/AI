import pandas as pd
import numpy as np
n = []
for i in range(10000):
    if i % 2 == 1:
        n.append(i)

# n.to_csv("C:\\Study\\_data\\dacon\\ddarung\\test.csv", index=False)
np.savetxt('test.csv', n, fmt='%d', delimiter=', ')