import numpy as np
import random
x = np.array([0, 1, 2, 3, 4, 0,0,0,0,0,0,0,2, 3, 4, 4, 4, 4, 4])
from sklearn.preprocessing import StandardScaler
import pandas as pd
x = pd.get_dummies(x)
print(x)

# print(np.unique(x, return_counts=True))