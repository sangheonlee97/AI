# [실습]
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#1. data
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
             [9,8,7,6,5,4,3,2,1,0]
             ])
y = np.array([1,2,3,4,5,6,7,8,9,10])
print(x.shape, y.shape)         # (3,10) (10, )

x = x.T
print(x.shape)                  # (10, 3)

model = Sequential()
model.add(Dense(3,input_dim=3))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=100, batch_size=1)

loss = model.evaluate(x,y)
pred = model.predict([[10, 1.3, 0]])

print("re : ", pred)
# loss: 8.1430e-07
# re :  [[9.999004]] / eochs=100, batch_size=1

# loss: 8.4963e-08
# re :  [[10.000688]] / eochs=100, batch_size=1