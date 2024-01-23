from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from keras.models import Sequential, save_model
from keras.layers import Dense, Conv2D, Flatten
from sklearn.metrics import r2_score, accuracy_score
import numpy as np
import pandas as pd
import time


datasets = fetch_california_housing()

# print(datasets.feature_names)       # ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']


X = datasets.data
y = datasets.target

# print(X.shape)   # 20640, 8
X = X.reshape(20640, 4,2,1)
# [실습]
# R2 0.55 ~ 0.6

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1228)

# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)



# model = Sequential()
# model.add(Dense(10, input_dim=8))
# model.add(Dense(30))
# model.add(Dense(15))
# model.add(Dense(1))
from keras.models import Model
from keras.layers import Input
input = Input(shape=(4,2,1))
dense1 = Conv2D(10, (2,1), padding='same')(input)
f = Flatten()(dense1)
dense2 = Dense(30)(f)
dense3 = Dense(15)(dense2)
output = Dense(1)(dense3)
model = Model(inputs=input, outputs=output)


model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])



start_time = time.time()

from keras.callbacks import EarlyStopping

from keras.callbacks import ModelCheckpoint, EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath='..//_data//_save//MCP//keras26_california.hdf5')

hist = model.fit(X_train, y_train, epochs=500, batch_size=142, validation_split=0.3, callbacks=[es])

model.save('..//_data//_save//MCP//keras26_california_1.h5')
end_time = time.time()

y_pred = model.predict(X_test)

loss = model.evaluate(X_test, y_test)

r2 = r2_score(y_test, y_pred)

print("loss : ", loss)
print("r2 : ", r2)
print("걸린 시간 : ", round(end_time - start_time,2), "초")

# cpu : 3.09
# gpu : 5.46
# dnn 0.61
# cnn 0.422