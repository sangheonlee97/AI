import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer

data = pd.DataFrame([
    [2, np.nan, 6, 8, 10],
    [2, 4, np.nan, 8, 10],
    # [2, 4, 6, np.nan, 10],
    [2, 4, 6, 8, 10],
    [np.nan, 4, np.nan, 8, np.nan]
    ])

data = data.transpose()
data.columns = ['X1', 'X2', 'X3', 'X4']
print(data)

imputer = SimpleImputer(strategy='most_frequent')

data2 = imputer.fit_transform(data)
print(data2)

imputer2 = KNNImputer()
data3 = imputer2.fit_transform(data)
print(data3)

imputer3 = IterativeImputer()
data4 = imputer3.fit_transform(data)
print(data4.round(4))

