# 06_1 복사
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,6,5,7,8,9,10])

x_train = np.array([1,2,3,4,5,])
y_train = np.array([1,2,3,4,6,])

x_val = np.array([6,7])
y_val = np.array([5,7])

x_test = np.array([8,9,10])
y_test = np.array([8,9,10])

model = Sequential()
model.add(Dense(1,input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=1000, batch_size=1, validation_data=(x_val,y_val))

loss = model.evaluate(x_test,y_test)
pred = model.predict([1100000, 7])

print("loss : ", loss)
print("pred : ", pred)

# loss :  0.2474503517150879
# pred :  [[1.02295694e+06] [6.64593267e+00]]