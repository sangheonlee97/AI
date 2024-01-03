import numpy as np
from keras.models import Sequential
from keras.layers import Dense

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

model = Sequential()
model.add(Dense(1,input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=1000, batch_size=1)

loss = model.evaluate(x,y)
pred = model.predict([11000, 7])

print("loss : ", loss)
print("pred : ", pred)

# loss :  1.435296380445511e-13
# pred :  [[1.1e+04] [7.0e+00]]