import numpy as np
import pandas as pd
m = []
for i in range(100):
    if i%2 == 0:
        m.append({i, i + 2})
    np.savetxt('test_kaggle_bike.csv', m, fmt='%s', delimiter=', ')