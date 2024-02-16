import numpy as np
import pandas as pd

data = pd.DataFrame([
    [2, np.nan, 6, 8, 10],
    [2, 4, np.nan, 8, 10],
    # [2, 4, 6, np.nan, 10],
    [2, 4, 6, 8, 10],
    [np.nan, 4, np.nan, 8, np.nan]
    ])

data = data.transpose()
data.columns = ['X1', 'X2', 'X3', 'X4']
print(data.shape)

# 결측치 확인
print(data.isnull().sum().sum())
print(data.info())

# 결측치 삭제
print(data.dropna())
print(data.dropna(axis=0))
print(data.dropna(axis=1))

# 결측치 임의 값으로 처리
means = data.mean()
print(means)
data2 = data.fillna(means)
print(data2)

med = data.median()
print(med)
data3 = data.fillna(med)
print(data3)

data4 = data.fillna(0)
print(data4)

data5 = data.ffill()
print(data5)

data6 = data.bfill()
print(data6)