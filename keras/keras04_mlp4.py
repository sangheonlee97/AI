import numpy as np
from keras.models import Sequential
from keras.layers import Dense

x = np.array([range(10)])
y = np.array([range(1,11),
              [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],
              range(9,-1,-1)
             ])

x = x.T
y = y.T
model = Sequential()
model.add(Dense(5,input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(3))

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

loss = model.evaluate(x, y)
pred = model.predict([10])

print("loss : ", loss)
print("pred : ", pred)

# loss :  1.7234510259894753e-12
# pred :  [[10.999998   2.0000021 -1.0000029]]