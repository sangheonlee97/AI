import numpy as np
from keras.models import Sequential
from keras.layers import Dense

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,6,5,7,8,9,10])

# [실습] 넘파이 리스트의 슬라이싱!! 7:3으로 잘라!!
x_train = x[:-3]
y_train = y[:-3]
x_test = x[-3:]
y_test = y[-3:]

print(x_train)
print(y_train)
print(x_test)
print(y_test)


model = Sequential()
model.add(Dense(1,input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=1000, batch_size=1)

loss = model.evaluate(x_test,y_test)
pred = model.predict([1100000, 7])

print("loss : ", loss)
print("pred : ", pred)

# loss :  0.2474503517150879
# pred :  [[1.02295694e+06] [6.64593267e+00]]