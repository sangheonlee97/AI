from keras.models import Sequential
from keras.layers import Dense

import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

# 이 데이터를 훈련해서 최소의 loss를 만들어라
model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

model.compile(loss="mse", optimizer="adam")
model.fit(x,y, epochs=100)

loss = model.evaluate(x,y)
result = model.predict([7])

print("result : ", result)


# 1/1 [==============================] - 0s 789us/step - loss: 0.3239
# Epoch 99/100
# 1/1 [==============================] - 0s 0s/step - loss: 0.3240
# Epoch 100/100
# 1/1 [==============================] - 0s 0s/step - loss: 0.3240
# 1/1 [==============================] - 0s 84ms/step - loss: 0.3240
# 1/1 [==============================] - 0s 75ms/step
# result :  [[6.825711]]