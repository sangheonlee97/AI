import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
path = "..//_data//kaggle//jena//jena_climate_2009_2016.csv"

dataset = pd.read_csv(path, index_col=0)

######################################  이상치 처리
# 이상치 처리 풍속 : 343580 ~ 343598 = 3.47 
#       최대 풍속 : 343580 ~ 343598 = 4.98
print(np.min(dataset['wv (m/s)']))
print(np.min(dataset['max. wv (m/s)']))
dataset.iloc[343577:343597, -3] = 3.47
dataset.iloc[343577:343597, -2] = 4.98
print(np.min(dataset['wv (m/s)']))
print(np.min(dataset['max. wv (m/s)']))
######################################

######################################  정규분포 모양으로 전처리
# sqrt_values = dataset['VPmax (mbar)']
# # sqrt_values = np.sqrt(dataset['VPmax (mbar)'])

# plt.hist(sqrt_values, bins=20, edgecolor='black')
# plt.title('Histogram of sqrt(VPdef (mbar))')
# plt.xlabel('sqrt(VPdef (mbar))')
# plt.ylabel('Frequency')
# plt.show()
### VPdef (mbar), 'VPmax (mbar)' 컬럼은 log를 씌우는 편이 정규분포에 가까워진다
print("@@@@@@@@@@@@@@@@@@@@@@@")
print(np.max(dataset['VPmax (mbar)']))
print(np.max(dataset['VPdef (mbar)']))
dataset.iloc[:,5] = np.sqrt(dataset.iloc[:,5])
dataset.iloc[:,7] = np.sqrt(dataset.iloc[:,7])
print(np.max(dataset['VPmax (mbar)']))
print(np.max(dataset['VPdef (mbar)']))
######################################


# scaling
sc = MinMaxScaler()
dataset = sc.fit_transform(dataset)
dataset = pd.DataFrame(dataset)


def split_Xy(data, size, col):
    X, y = [], []
    for i in range(len(data) -  size - 144):
        X_s = data.iloc[i : i + size : 6, : ]
        y_s = data.iloc[i + size + 144, col]
        X.append(X_s)
        y.append(y_s)
    return np.array(X), np.array(y)
timesteps = 6 * 24 * 5
target_col = 1
X, y = split_Xy(dataset, timesteps, target_col)
path2 = "..//_data//kaggle//jena//"

np.save(path2 + "jena_X.npy", X)
np.save(path2 + "jena_y.npy", y)
print(y)