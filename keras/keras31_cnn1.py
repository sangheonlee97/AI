from tensorflow.python.keras.models import Sequential
# from tensorflow.keras.models import Sequential
# from keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D


model = Sequential()
# model.add(Dense(10, input_shape=(3, )))
model.add(Conv2D(10, (2,2), input_shape=(10, 10, 1)))
model.add(Dense(5))
model.add(Dense(1))