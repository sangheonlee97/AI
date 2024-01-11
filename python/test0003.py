import pandas as pd
import numpy as np

a = pd.DataFrame(range(5))
print(a)
a = a.T
print(a)
a[4] = 5
a = a.T
print(a)