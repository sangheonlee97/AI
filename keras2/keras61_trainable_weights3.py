import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import tensorflow as tf
tf.random.set_seed(777)
np.random.seed(777)
print(tf.__version__)   # 2.9.0

X = np.array([1,2])
y = np.array([1,2])

model = Sequential()
model.add(Dense(2, input_dim=1))
# model.add(Dense(2))
model.add(Dense(1))

model.trainable = False    # 중요******************
# model.trainable = True  # default

print(model.weights)

model.compile(loss='mse', optimizer='adam', )
model.fit(X, y, batch_size=1, epochs=1000, verbose=0)

y_pred = model.predict(X)
print(y_pred)
print(model.weights)
