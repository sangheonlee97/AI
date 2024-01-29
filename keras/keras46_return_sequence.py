import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
X = np.array([
                [1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11], [10,11,12],
                [20,30,40], [30,40,50], [40,50,60]
            ])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
X = X.reshape(-1, 3, 1)
model = Sequential()
model.add(LSTM(27, return_sequences=True, input_shape=(3,1)))
model.add(LSTM(31))
model.add(Dense(97, activation='relu'))
model.add(Dense(91, activation='relu'))
model.add(Dense(21, activation='relu'))
model.add(Dense(161, activation='relu'))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='loss', mode='min', patience=100, restore_best_weights=True)
model.fit(X, y, epochs=5000,batch_size=32, callbacks=[es])

l = model.evaluate(X, y)
print("loss : ", l)
X_predict = np.array([50,60,70])
X_predict = X_predict.reshape(-1,3,1)
p = model.predict(X_predict)
print("predict : ", p)

# loss :  0.004179507028311491
# predict :  [[79.21568]]