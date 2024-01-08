import pandas as pd
import numpy as np

a = pd.DataFrame(range(10))
print(a.shape)
b = a.iloc[:3]
print("dsfsadfaf")
print(b)
b[:] = 1
print(a)
idx = 0
for i in a:
    idx = idx+1
print(a)