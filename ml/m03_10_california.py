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
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
models = []
models.append(LinearSVR())
# models.append(Perceptron())
models.append(LinearRegression())
models.append(KNeighborsRegressor())
models.append(DecisionTreeRegressor())
models.append(RandomForestRegressor())

for model in models:
        
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    loss = model.score(X_test, y_test)

    r2 = r2_score(y_test, y_pred)

    print("loss : ", loss)
    print("r2 : ", r2)

    # cpu : 3.09
    # gpu : 5.46