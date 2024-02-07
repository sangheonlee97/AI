import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense

X_data = np.array([[0,0], [0,1], [1,0], [1,1]])
y_data = np.array([0, 1, 1, 0])
print(X_data.shape, y_data.shape)

# model = LinearSVC()
# model = Perceptron()
model = Sequential()
model.add(Dense(10,input_shape=(2, ), activation='swish'))
model.add(Dense(20, activation='swish'))
model.add(Dense(20, activation='swish'))
model.add(Dense(20, activation='swish'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(X_data, y_data, batch_size=1, epochs=100)

res = model.evaluate(X_data, y_data)
# acc = model.score(X_data, y_data)
y_pred = model.predict(X_data).round().reshape(-1).astype(int)
print(y_pred)

print("model.evalutate : ", res[1])


accc = accuracy_score(y_data, y_pred)
print("acc : ", accc)