from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf
import keras
print("tf 버전 : ", tf.__version__)
print("keras 버전 : ", keras.__version__)


#실습 : 레이어 6개, 배치사이즈만 수정해서 loss를 0.32 이하로 만들 것

#1. DATA
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#2. modeling
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(9))
model.add(Dense(13))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

#3. compile, train
model.compile(loss="mse", optimizer='adam')
model.fit(x, y, epochs=100,  batch_size=1)

#4. evaluate, predict
loss = model.evaluate(x, y)
results = model.predict([7])
print("loss : ", loss)
print("result : ", results)

# batch_size=1
# loss :  0.32383087277412415
# result :  [[6.7984085]] 