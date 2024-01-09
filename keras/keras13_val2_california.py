from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from keras.models import Sequential
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

model = Sequential()
model.add(Dense(10, input_dim=8))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')



start_time = time.time()

model.fit(X_train, y_train, epochs=500, batch_size=142, validation_split=0.2)

end_time = time.time()

y_pred = model.predict(X_test)

loss = model.evaluate(X_test, y_test)

r2 = r2_score(y_test, y_pred)

print("loss : ", loss)
print("r2 : ", r2)
print("걸린 시간 : ", round(end_time - start_time,2), "초")

# loss :  0.5542855858802795
# r2 :  0.5767131444516536
# random_state = 1228
# epochs = 1000
# batch_size = 142