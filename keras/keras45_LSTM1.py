import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, SimpleRNN, LSTM

# 1. data
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

X = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9]])
y = np.array([4, 5, 6, 7, 8, 9, 10])
X = X.reshape(-1, 3, 1)
print(X.shape) # 731

# 2. model
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', mode='min', patience=150, restore_best_weights=True)
model = Sequential()
# model.add(SimpleRNN(50, input_shape=(3,1)))
model.add(LSTM(50, input_shape=(3,1)))
model.add(Dense(570, activation='relu'))
model.add(Dense(370, activation='relu'))
model.add(Dense(170, activation='relu'))
model.add(Dense(270, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(1))

# 3. compile
model.compile(loss='mse', optimizer='adam')
model.fit(X, y, epochs=500, batch_size=1,)
# 4. evaluate
res = model.evaluate(X,y)
print("loss : ", res)
y_pred = model.predict([[[8],[9],[10]]])
print("predict : ", y_pred)