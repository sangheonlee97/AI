from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from keras.models import Sequential, save_model
from keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
import numpy as np
import pandas as pd
import time


datasets = fetch_california_housing()

# print(datasets.feature_names)       # ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']


X = datasets.data
y = datasets.target



# [실습]
# R2 0.55 ~ 0.6

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1228)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



# model = Sequential()
# model.add(Dense(10, input_dim=8))
# model.add(Dense(30))
# model.add(Dense(15))
# model.add(Dense(1))
from sklearn.utils import all_estimators
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score
kfold = KFold(5, shuffle=True, random_state=42)
allAlgorithms = all_estimators(type_filter='regressor')
best = [0, 'no', 999]
model = allAlgorithms[14][1]()
score = cross_val_score(model, X, y, cv=kfold)

print("r2 : ", score)

# best model :  HistGradientBoostingRegressor , idx :  14
# best r2 :  0.8429643520886949

# r2 :  [0.83377291 0.84235027 0.82615755 0.84944614 0.83214748]