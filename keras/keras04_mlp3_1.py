import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. data
x = np.array([range(10), range(21,31), range(201, 211)])
# print(x)
print(x.shape)      # (3, 10)

x = x.T
# print(x)
print(x.shape)      # (10, 3)

y = np.array([range(1,11),
              [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9]
             ])
y = y.T

# [실습]
# 예측 : [10, 31, 211]

#2. modeling
model = Sequential()
model.add(Dense(5,input_dim=3))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))

#3. compile, train
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=1000, batch_size=1)

#4 evaluate, predict
loss = model.evaluate(x,y)
pred = model.predict([[10, 31, 211]])

print("loss : ", loss)
print("pred : ", pred)

# loss :  2.7977930585620925e-06
# pred :  [[11.003771   1.9998683]]