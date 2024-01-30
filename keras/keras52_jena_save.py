import numpy as np
import pandas as pd

path = "..//_data//kaggle//jena//jena_climate_2009_2016.csv"

dataset = pd.read_csv(path, index_col=0)

def split_Xy(data, size, col):
    X, y = [], []
    for i in range(len(data) -  size):
        X_s = data.iloc[i : i + size, : ]
        y_s = data.iloc[i + size, col]
        X.append(X_s)
        y.append(y_s)
    return np.array(X), np.array(y)
timesteps = 4
target_col = 1
X, y = split_Xy(dataset, timesteps, target_col)
path2 = "..//_data//kaggle//jena//"

np.save(path2 + "jena_X.npy", X)
np.save(path2 + "jena_y.npy", y)
print(y)