from keras.models import Sequential
from keras.layers import Dense

import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

# 이 데이터를 훈련해서 최소의 loss를 만들어라
model = Sequential()
model.add(Dense(1, input_dim=1))

model.compile(loss="mse", optimizer="adam")
model.fit(x,y, epochs=10000)

loss = model.evaluate(x,y)
result = model.predict([7])

print("result : ", result)