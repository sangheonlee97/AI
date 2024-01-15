from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd

X = np.array([1,2,3])
y = np.array([1,2,3])
print(X.shape)
model = Sequential()
model.add(Dense(5, input_shape=(1, )))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

model.summary()

y_pre = model.predict([1,2,3])
print(y_pre)