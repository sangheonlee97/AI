import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. data
X = np.array(range(1, 17))
y = np.array(range(1, 17))

# # 잘라라!

X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.375, random_state=312)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size= 0.2, shuffle=False)
print(X_train)  # [ 1  2  3  4  5  6  7  8  9 10, 11, 12, 13]
# print(X_val)    # [11 12 13]
print(X_test)   # [14 15 16]


model = Sequential()
model.add(Dense(1,input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train,y_train, epochs=100, batch_size=1, validation_split=0.3)

loss = model.evaluate(X_test,y_test)
pred = model.predict([1100000, 7])
print("pred : ", pred)